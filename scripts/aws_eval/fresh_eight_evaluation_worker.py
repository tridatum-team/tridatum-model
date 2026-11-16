"""One loaded frozen CUDA model, one fixed smoke pair, then all400 fresh answers.

This module never constructs an optimizer, trains, selects, retries, resumes,
merges an adapter, or imports the historical training implementation.
"""
import contextlib
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if __package__:
    from . import fresh_eight_evaluation as E
else:
    import fresh_eight_evaluation as E


def phase_kind(phase):
    return phase if phase in ('runtime', 'evaluation') else None


def helpers(base):
    sys.path.insert(0, str(Path(base) / 'code/scripts'))
    import evaluate_local_eight
    return evaluate_local_eight


def save_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_row(stream, row):
    stream.write(E.canonical(row).decode() + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def cuda_memory(torch):
    torch.cuda.synchronize()
    return dict(device='cuda:0', allocated_bytes=torch.cuda.memory_allocated(0),
        reserved_bytes=torch.cuda.memory_reserved(0), peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0), peak_scope='single loaded model process lifetime')


def parameter_inventory(model, torch):
    inventory = dict(base={}, adapter={})
    for name, parameter in model.named_parameters():
        group = 'adapter' if '.lora_A.' in name or '.lora_B.' in name else 'base'
        expected = torch.float32 if group == 'adapter' else torch.bfloat16
        E.require(parameter.device.type == 'cuda' and parameter.device.index == 0
                  and parameter.dtype == expected and not parameter.requires_grad,
                  'Actual frozen parameter/device/dtype differs; no casting fallback')
        key = str(parameter.dtype) + '@' + str(parameter.device)
        entry = inventory[group].setdefault(key, dict(tensors=0, elements=0, bytes=0))
        entry['tensors'] += 1
        entry['elements'] += parameter.numel()
        entry['bytes'] += parameter.numel() * parameter.element_size()
    E.parameter_inventory(inventory)
    return inventory


def generation_binding(folder, policy, mode):
    return dict(run_manifest_sha256=E.pin(folder / 'run-manifest.json')['sha256'],
        loaded_runtime_sha256=E.pin(folder / 'loaded-runtime.json')['sha256'], model_id=E.MODEL_ID,
        model_revision=E.MODEL_REVISION, input_sha256=E.INPUT_SHA, nominated_adapter_weights_sha256=E.ADAPTER_SHA,
        mode=mode, decoder_profile='unconstrained',
        decoder_profile_sha256=E.pin(folder / 'decoder-profile.json')['sha256'],
        generation_policy_sha256=E.object_sha(policy), candidate_manifest_sha256=E.CANDIDATE_SHA)


def run_pairs(core, helper, torch, model, tokenizer, prepared, config, binding, folder):
    """Exclusive journals make retry/reuse impossible; every call has fresh input/KV."""
    mode = binding['mode']
    with contextlib.ExitStack() as stack:
        streams = {arm: stack.enter_context((folder / (arm + '-outputs.jsonl')).open('x', encoding='utf-8'))
                   for arm in E.ARMS}
        attempts = stack.enter_context((folder / 'attempts.jsonl').open('x', encoding='utf-8'))
        for case in prepared:
            for arm in E.ARMS:
                append_row(attempts, dict(id=case['id'], arm=arm, mode=mode, state='started',
                    timestamp=dt.datetime.now(dt.timezone.utc).isoformat()))
                started = time.perf_counter()
                tokens = []
                row_written = False
                try:
                    torch.manual_seed(E.GENERATION['seed'])
                    torch.cuda.manual_seed_all(E.GENERATION['seed'])
                    ids = torch.tensor([case['prompt_token_ids']], dtype=torch.long, device='cuda:0')
                    attention = torch.ones_like(ids)
                    torch.cuda.synchronize()
                    with (model.disable_adapter() if arm == 'base' else contextlib.nullcontext()):
                        with torch.inference_mode(), helper.observe_generation_policy(model, config) as observed:
                            result = model.generate(input_ids=ids, attention_mask=attention,
                                generation_config=config, **helper.GREEDY_DISPATCH)
                    torch.cuda.synchronize()
                    E.require(result.sequences.shape[0] == 1
                              and result.sequences[0, :len(case['prompt_token_ids'])].tolist() == case['prompt_token_ids'],
                              'Generation changed the prompt or batch dimension')
                    tokens = result.sequences[0, len(case['prompt_token_ids']):].tolist()
                    decoded = core.decode_generation(tokenizer, tokens, config.eos_token_id, 768)
                    row = dict(**case, arm=arm, **binding, generation=dict(E.GENERATION),
                        adapter_weights_sha256=None if arm == 'base' else E.ADAPTER_SHA,
                        generation_config=config.to_dict(), output_token_ids=tokens, **decoded,
                        exception=None, context_truncated=False, skipped_context=False,
                        truncated_output=decoded['termination_reason'] == 'length',
                        elapsed_seconds=time.perf_counter() - started, cuda_memory=cuda_memory(torch),
                        effective_generation_mode=observed['generation_mode'])
                    # Persist the real answer before smoke admission: an empty/length
                    # smoke failure remains evidence, and is never replaced or retried.
                    append_row(streams[arm], row)
                    row_written = True
                    E.validate_answer(row, case, arm, mode, binding, config.to_dict())
                    print(json.dumps(dict(event='answer_complete', mode=mode, id=case['id'], arm=arm,
                        generated_tokens=len(tokens), termination_reason=row['termination_reason'],
                        elapsed_seconds=row['elapsed_seconds'])), flush=True)
                    # Drop all per-answer state, including returned past_key_values.
                    del result, ids, attention
                except BaseException as error:
                    if not row_written:
                        append_row(streams[arm], dict(**case, arm=arm, **binding,
                            adapter_weights_sha256=None if arm == 'base' else E.ADAPTER_SHA,
                            generation=dict(E.GENERATION), generation_config=config.to_dict(),
                            output_token_ids=tokens, generated_tokens=len(tokens), raw_output=None,
                            response_text=None, termination_reason='error', terminal_stop_token_id=None,
                            context_truncated=False, skipped_context=False, truncated_output=False,
                            elapsed_seconds=time.perf_counter() - started,
                            exception=dict(type=type(error).__name__, message=str(error))))
                    append_row(attempts, dict(id=case['id'], arm=arm, mode=mode, state='failed',
                        elapsed_seconds=time.perf_counter() - started,
                        exception=dict(type=type(error).__name__, message=str(error)),
                        output_token_ids=tokens))
                    raise


def child_stage(core, plan, base, phase):
    E.require(phase == 'evaluation' and E.is_evaluation(plan), 'Only the explicit evaluation child is allowed')
    base = Path(base)
    root = base / 'output'
    folder = root / 'evaluation'
    folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    active = folder
    try:
        admitted = E.validate_data(base / 'code', plan)
        core.validate_model(base / 'model', plan)
        torch = core.native_runtime(plan, base)
        helper = helpers(base)
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from peft import PeftModel
        actual = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}
        E.require(actual == E.RUNTIME_PACKAGES, 'Evaluation installed runtime changed after admission')
        release = json.loads((base / 'release.json').read_bytes())
        E.require(os.environ['CUDA_VISIBLE_DEVICES'] == release['gpu']['uuid'], 'Evaluation GPU UUID differs')
        runtime_receipt = json.loads((root / 'runtime.json').read_bytes())
        E.require(runtime_receipt['packages'] == actual and runtime_receipt['plan_sha256'] == core.digest(base / 'plan.json')
                  and runtime_receipt['instance_id'] == release['instance_id'], 'Root runtime admission binding differs')
        tokenizer = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
        prepared = helper.preflight_prompts(tokenizer, admitted['evaluation_inputs'])
        E.require(len(prepared) == 200, 'All200 must preflight before loading weights')
        model_files = [E.pin(base / 'model' / p['path'], base / 'model') for p in plan['model']['files']]
        model_identity = dict(id=E.MODEL_ID, revision=E.MODEL_REVISION, files=model_files,
            acquisition_manifest=plan['workload']['acquisition_manifest'],
            representation='hf-safetensors-bfloat16', body_hashes_authenticated=True)
        source_files = [E.pin(base / 'code' / p['path'], base / 'code') for p in admitted['sources']]
        E.require(source_files == admitted['sources'], 'Current executable sources changed')
        code = {p['path']: p for p in source_files}
        runtime = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            machine=platform.machine(), device='cuda', dtype='torch.bfloat16', packages=actual,
            gpu_uuid=release['gpu']['uuid'], instance_id=release['instance_id'],
            torch_threads=torch.get_num_threads(), cpu_fallback=False, cuda_equivalence_claim=False)
        decoder = dict(profile='unconstrained', semantic_truth_guarantee=False)
        shared = dict(schema=E.RUN_SCHEMA, model=model_identity,
            adapter=dict(step=150, files={Path(p['path']).name: p for p in plan['workload']['adapter_files']},
                         config=admitted['adapter_config']),
            input=plan['workload']['evaluation_inputs'], generation=dict(E.GENERATION), decoder_profile=decoder,
            runtime=runtime, code=code, arm_order=list(E.ARMS),
            paired_order='each input: base with adapter disabled, then registered adapter enabled',
            grading_access=False, retries=False, resume=False, plan=E.pin(root / 'plan.json', root),
            lineage_binding=E.pin(root / 'lineage-binding.json', root),
            source_manifest=E.pin(root / 'source-manifest.json', root),
            evaluation_contract=admitted['lineage']['provenance'])
        selections = {'smoke': [p for p in prepared if p['id'] == E.SMOKE_ID], 'evaluation': prepared}
        # Both preflight receipts are durable before any weights are loaded.
        for mode, selected in selections.items():
            target = root / mode
            if mode == 'smoke':
                target.mkdir(exist_ok=False)
            save_new(target / 'decoder-profile.json', decoder)
            save_new(target / 'run-manifest.json', dict(shared, mode=mode, selected_ids=[p['id'] for p in selected]))
            save_new(target / 'prompt-preflight.json', dict(status='passed', prompts=selected,
                all_selected_preflighted_before_weights=True, all_200_preflighted_before_weights=True, truncation=False))
        load_started = time.perf_counter()
        # One official unquantized model load, never a cached prior answer/model instance.
        native = AutoModelForCausalLM.from_pretrained(base / 'model', local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa')
        model = PeftModel.from_pretrained(native, base / 'code/adapter', is_trainable=False,
            local_files_only=True, autocast_adapter_dtype=True)
        model.eval()
        inventory = parameter_inventory(model, torch)
        E.require(not any(getattr(m, 'merged', False) for m in model.modules()), 'Merged adapters forbidden')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=768, use_cache=True,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, return_dict_in_generate=True)
        requested = config.to_dict()
        config, policy = helper.resolve_generation_policy(model, config)
        loaded = dict(loader=dict(implementation='transformers.AutoModelForCausalLM.from_pretrained',
            local_files_only=True, trust_remote_code=False), elapsed_seconds=time.perf_counter() - load_started,
            attention_implementation=model.config._attn_implementation,
            generation_config=config.to_dict(), requested_generation_config=requested, generation_policy=policy,
            cuda_memory=cuda_memory(torch), parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            parameter_inventory=inventory, buffer_dtypes=sorted({str(b.dtype) for b in model.buffers()}),
            code=code, frozen=True, adapter_merged=False, model_loads=1, gpu_uuid=release['gpu']['uuid'])
        for mode in ('smoke', 'evaluation'):
            active = root / mode
            selected = selections[mode]
            save_new(active / 'loaded-runtime.json', loaded)
            binding = generation_binding(active, policy, mode)
            run_pairs(core, helper, torch, model, tokenizer, selected, config, binding, active)
            save_new(active / 'completion.json', dict(status='complete', mode=mode,
                elapsed_seconds=time.perf_counter() - started, rows_per_arm=len(selected),
                outputs={arm: E.pin(active / (arm + '-outputs.jsonl'), active) for arm in E.ARMS},
                cuda_memory=cuda_memory(torch), run_manifest_sha256=binding['run_manifest_sha256']))
            E.validate_mode(root, mode, plan)
        save_new(folder / 'evaluation-worker.json', dict(schema=E.WORKER_SCHEMA, status='complete',
            run_id=E.RUN_ID, plan_sha256=core.digest(base / 'plan.json'), evaluation_success=True,
            completion_scope=E.COMPLETION_SCOPE, quality_accepted=False, rows_per_arm=200,
            smoke_rows_per_arm=1, generation_calls=402, model_loads=1,
            smoke=E.pin(root / 'smoke/completion.json', root), evaluation=E.pin(folder / 'completion.json', root)))
        E.validate_completion(root, plan)
    except BaseException as error:
        save_new(active / 'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            elapsed_seconds=time.perf_counter() - started, completed_rows_retained=True,
            automatic_retry=False, quality_accepted=False))
        raise


def run_experiment(core, plan, base, report, gpu, account, work_deadline,
                   acquisition_deadline, budget, hosts, units):
    E.require(E.is_evaluation(plan), 'Explicit evaluation profile required')
    base = Path(base)
    root = base / 'output'
    for row in plan['model']['files']:
        core.download(row, base / 'model', hosts, acquisition_deadline, budget)
    core.validate_model(base / 'model', plan)
    save_new(root / 'model-acquisition.json', dict(status='authenticated', model_id=E.MODEL_ID,
        revision=E.MODEL_REVISION, plan_sha256=core.digest(base / 'plan.json'),
        files=[E.pin(base / 'model' / row['path'], base / 'model') for row in plan['model']['files']]))
    for path in (base / 'model').iterdir():
        path.chmod(0o444)
    # Root-owned portable metadata is evidence, not a remote pathname assertion.
    for source, destination in ((base / 'plan.json', 'plan.json'),
        (base / 'code' / plan['workload']['lineage_binding']['path'], 'lineage-binding.json'),
        (base / 'code' / plan['workload']['source_manifest']['path'], 'source-manifest.json')):
        target = root / destination
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        target.chmod(0o444)
    unit = core.supervise_phase(plan, base, 'evaluation', gpu, account.pw_uid, account.pw_gid,
        work_deadline, units, attempt=report.setdefault('phase_attempt', {}))
    E.require(unit['cleanup_verified'] is True and not units, 'All owned evaluation processes must be reaped')
    report['stages'].append(unit)
    E.validate_completion(root, plan)
    save_new(root / 'evaluation/supervisor-result.json', dict(schema=E.SUPERVISION_SCHEMA,
        phase='evaluation', status='accepted', cleanup_verified=True, run_id=E.RUN_ID,
        plan_sha256=core.digest(base / 'plan.json'),
        worker_result=E.pin(root / 'evaluation/evaluation-worker.json', root), unit=unit))
    for name in ('smoke', 'evaluation'):
        folder = root / name
        for path in [folder, *folder.rglob('*')]:
            E.require(not path.is_symlink(), 'Unsafe evaluation output')
            os.chown(path, 0, 0)
            path.chmod(0o555 if path.is_dir() else 0o444)
    E.validate_supervision(root, plan)
