"""Sealed public_work_v1 CUDA execution; never a CPU guard/authentication receipt.

Acquire pinned public weights, generate both full suites on the base, train exactly
two exposures of each admitted row, then generate the same suites with final LoRA.
The shared root supervisor owns resource enforcement, freezing, export and shutdown.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

if __package__:
    from . import public_work_experiment as P
    from . import public_evidence_worker as shared
else:
    import public_work_experiment as P
    import public_evidence_worker as shared

RUNTIME_SCHEMA = 'public_work_gpu_runtime_v1'
COMPLETION_SCHEMA = 'public_work_gpu_completion_v1'
RECEIPT_SCHEMA = 'public_work_gpu_hf_v1'
SUITES = {'holdout': ('development_holdout', 170), 'diagnostic': ('diagnostic', 24)}
TOKENIZER_FILES = frozenset(('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                             'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json'))
save_new = shared.save_new
append_row = shared.append_row


def helpers():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import evaluate_public_model
    return evaluate_public_model


def evaluator():
    helpers()
    import evaluate_public_work
    return evaluate_public_work


def read_json(path):
    return helpers().load_json(path)


def relative_pin(path, root):
    return dict(P.pin(Path(path)), path=os.path.relpath(path, root))


def copy_new(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def adapter_identity(root, plan):
    return shared.adapter_identity(root, plan)


def exposure_counts(rows):
    return Counter((row['id'], row['source_sha256'], row['source_line']) for row in rows)


def training_observer(plan, torch, eos_token_id):
    """Check the real trainer's items and consumed identities, not planned counts."""
    recipe = P.validate_recipe(plan['recipe'])
    measured, expected = {}, Counter()

    def observe(event, **values):
        if event == 'admission':
            items, args = values['train'], values['args']
            P.require(not values['valid'] and values['dropped'] == values['valid_dropped'] == 0
                      and len(items) == recipe['rows'] * recipe['epochs'],
                      'Trainer changed admitted exposure set')
            P.require(args.checkpoint_every is None and args.valid_every == 0
                      and args.init_adapter is None and not args.merge,
                      'Intermediate checkpoint, validation, warm start or merge forbidden')
            maximum_context = maximum_answer = 0
            for item in items:
                ids, labels = item['ids'], item['labels']
                P.require(len(ids) == len(labels) and -100 in labels, 'Missing answer-only mask')
                first = next((i for i, token in enumerate(labels) if token != -100), len(labels))
                P.require(0 < first < len(labels) and labels[:first] == [-100] * first
                          and labels[first:] == ids[first:] and ids[-1] == eos_token_id,
                          'Prompt masking, answer targets or supervised EOS differs')
                identity = item['identity']
                expected[(identity['id'], identity['source_sha256'], identity['source_line'])] += 1
                maximum_context = max(maximum_context, len(ids))
                maximum_answer = max(maximum_answer, len(ids) - first)
            P.require(len(expected) == recipe['rows'] and set(expected.values()) == {recipe['epochs']}
                      and {key[1] for key in expected} == {plan['data']['train']['sha256']}
                      and {key[2] for key in expected} == set(range(1, recipe['rows'] + 1)),
                      'Each admitted physical training row must appear exactly twice')
            P.require(maximum_context <= recipe['context_tokens']
                      and maximum_answer <= recipe['answer_tokens'], 'Tokenization exceeds envelope')
            measured['masking'] = dict(rows=len(items), all_prompt_tokens_masked=True,
                all_answer_tokens_supervised=True, eos_supervised=True,
                max_context_tokens=maximum_context, max_answer_tokens=maximum_answer)
        elif event == 'selection':
            return dict(status='unselected', reason='Fixed last step; no evaluation-based checkpoint selection')
        elif event == 'model':
            model = values['model']
            measured['parameter_inventory'] = shared.parameter_inventory(model, torch, training=True)
            matrices = [p for n, p in model.named_parameters() if p.requires_grad and '.lora_B.' in n]
            P.require(matrices and all(int(torch.count_nonzero(p)) == 0 for p in matrices)
                      and not values['optimizer'].state, 'Fresh zero-B adapter and optimizer required')
            measured['fresh_zero_b'] = True
        elif event == 'completion':
            P.require(expected and exposure_counts(values['coverage']) == expected
                      and values['steps'] == recipe['steps'], 'Training skipped or repeated exposures')
            measured.update(observed_steps=values['steps'], coverage_exactly_twice=True)
        elif event == 'report':
            return dict(checkpoint_every=0, valid_every=0)
        elif event in ('before_step', 'after_step'):
            return {}
        return None

    return observe, measured


def train_stage(core, plan, base, torch):
    from transformers import AutoTokenizer
    base = Path(base)
    root = base / 'output'
    # Baseline acceptance is a prerequisite, never inferred from the phase name.
    validate_outputs(root, plan, 'baseline')
    tokenizer = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    observer, measured = training_observer(plan, torch, tokenizer.eos_token_id)
    argv = core.train_argv(plan, base)
    core.trainer_module(base)['main'](argv[1:], observer=observer)
    save_new(root / 'training/provenance.json', dict(schema=P.ADAPTER_SCHEMA,
        run_id=plan['identity']['run_id'], plan_sha256=core.digest(base / 'plan.json'),
        training=P.pin(root / 'training/training.json', root), adapter=adapter_identity(root, plan),
        source_manifest=plan['workload']['source_manifest'], runtime=P.pin(root / 'runtime.json', root),
        fresh=True, merged=False, **measured))
    validate_training(root, plan)


def stream_responses(model, tokenizer, rows, prepared, path, *, binding):
    """Same official greedy controls as the CPU runner; distinct CUDA receipts."""
    import torch
    from evaluate_evidence_conditioning import observe_policy, decode_generation
    E = helpers()
    P.require([p['id'] for p in prepared] == [row['id'] for row in rows], 'Full ordered preflight required')
    effective, dispatch = E.generation_controls(model, tokenizer, 512)
    config = effective.to_dict()
    counts = dict(rows=0, exception_rows=0, length_rows=0)
    with Path(path).open('x', encoding='utf-8') as stream:
        for prompt in prepared:
            started = time.monotonic()
            record = dict(prompt, generation_receipt=RECEIPT_SCHEMA, response='', raw_output='',
                finish_reason='exception', error=None, output_token_ids=[], output_tokens=0,
                terminal_stop_token_id=None, generation_config=config, generation_policy=None,
                dispatch_kwargs=dispatch, template_kwargs=dict(E.TEMPLATE_KWARGS), seed=0,
                mode='evaluation', context_limit=4096, context_truncated=False, truncated_output=False,
                binding=binding)
            try:
                ids = torch.tensor([prompt['prompt_token_ids']], dtype=torch.long, device='cuda:0')
                torch.manual_seed(0)
                with torch.inference_mode(), observe_policy(model, effective, dispatch) as observed:
                    result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            generation_config=effective, **dispatch)
                record['generation_policy'] = dict(observed)
                output = result.sequences[0, prompt['input_tokens']:].tolist()
                record.update(output_token_ids=output, output_tokens=len(output))
                if output:
                    decoded = decode_generation(tokenizer, output, config['eos_token_id'], 512)
                    record.update(response=decoded['response_text'], raw_output=decoded['raw_output'],
                        finish_reason=decoded['termination_reason'],
                        terminal_stop_token_id=decoded['terminal_stop_token_id'],
                        truncated_output=decoded['termination_reason'] == 'length')
                else:
                    record['finish_reason'] = 'empty'
            except Exception as error:
                record['error'] = dict(type=type(error).__name__, message=str(error))
                record['finish_reason'] = 'exception'
            record['wall_seconds'] = time.monotonic() - started
            append_row(stream, record)
            counts['rows'] += 1
            counts['exception_rows'] += record['error'] is not None
            counts['length_rows'] += record['finish_reason'] == 'length'
    return counts


def evaluation_stage(core, plan, base, torch, arm):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = Path(base)
    root, E = base / 'output', helpers()
    P.require(arm in ('baseline', 'adapter'), 'Unknown public-work evaluation arm')
    cases = P.validate_data(base / 'code', plan)['evaluation_inputs']
    folder = root / arm
    folder.mkdir(mode=0o700)
    tokenizer = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    template = tokenizer.get_chat_template()
    P.require(isinstance(template, str) and template, 'Official native template missing')
    prepared = E.preflight_prompts(tokenizer, cases, context_limit=4096, max_new_tokens=512)
    save_new(folder / 'prompt-preflight.json', dict(status='passed', all_cases_preflighted_before_weights=True,
        truncation=False, prompts=prepared))
    adapter = None
    if arm == 'adapter':
        validate_training(root, plan)
        adapter = adapter_identity(root, plan)
    model = AutoModelForCausalLM.from_pretrained(base / 'model', torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    P.require(model.config.max_position_embeddings >= 4096, 'Requested context exceeds model limit')
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, root / 'training/adapter', is_trainable=False,
                                         local_files_only=True, autocast_adapter_dtype=True)
        shared.verify_loaded_adapter(model, root / 'training/adapter/adapter_model.safetensors', torch)
    model.requires_grad_(False)
    model.eval()
    inventory = shared.parameter_inventory(model, torch)
    P.require(bool(inventory['adapter']) == (arm == 'adapter'), 'Wrong adapter loaded for arm')
    release = read_json(base / 'release.json')
    loaded = dict(device='cuda', dtype='torch.bfloat16', frozen=True, adapter_merged=False,
        parameter_inventory=inventory, adapter=adapter, cpu_fallback=False,
        adapter_reload_equal=True if adapter else None, gpu_uuid=release['gpu']['uuid'],
        instance_id=release['instance_id'], plan_sha256=core.digest(base / 'plan.json'),
        source_manifest=plan['workload']['source_manifest'], runtime=P.pin(root / 'runtime.json', root),
        model_files=[P.pin(base / 'model' / p['path'], base / 'model') for p in plan['model']['files']],
        local_files_only=True, trust_remote_code=False, attention_implementation=model.config._attn_implementation)
    save_new(folder / 'loaded-runtime.json', loaded)
    indexed = {p['id']: p for p in prepared}
    binding = P.generation_binding(plan, arm, adapter)
    totals = dict(rows=0, exception_rows=0, length_rows=0)
    completions = {}
    for name, (scope, count) in SUITES.items():
        suite_folder = folder / name
        suite_folder.mkdir(mode=0o700)
        suite_path = root / 'suites' / name / 'manifest.json'
        suite = read_json(suite_path)
        inputs_path = root / 'suites' / name / 'inputs.json'
        rows = read_json(inputs_path)
        P.require(suite['scope'] == scope and len(rows) == count, 'Sealed suite identity differs')
        identity = dict(repo_id=plan['model']['id'], revision=plan['model']['revision'])
        runtime_adapter = None if adapter is None else dict(
            provenance=relative_pin(root / 'training/provenance.json', suite_folder),
            training=relative_pin(root / 'training/training.json', suite_folder),
            files=[relative_pin(root / adapter[k]['path'], suite_folder) for k in ('config', 'weights')],
            model_identity=identity, representation='unmerged_peft', steps=plan['recipe']['steps'])
        runtime = dict(schema=RUNTIME_SCHEMA, suite_scope=scope,
            suite_manifest=relative_pin(suite_path, suite_folder), source=suite['source'],
            inputs=relative_pin(inputs_path, suite_folder), model_identity=identity,
            model_manifest=relative_pin(root / 'model-acquisition.json', suite_folder),
            tokenizer_identity=dict(files=[relative_pin(root / 'tokenizer' / p['path'], suite_folder)
                for p in plan['model']['files'] if p['path'] in TOKENIZER_FILES],
                template_sha256=hashlib.sha256(template.encode()).hexdigest()),
            adapter=runtime_adapter, generation_policy=E.requested_policy(512, 4096),
            device='cuda', cpu_fallback=False, status='finished', research_only=True, quality_accepted=False,
            binding=binding, plan=relative_pin(root / 'plan.json', suite_folder),
            loaded_runtime=relative_pin(folder / 'loaded-runtime.json', suite_folder))
        save_new(suite_folder / 'runtime.json', runtime)
        counts = stream_responses(model, tokenizer, rows, [indexed[row['id']] for row in rows],
                                 suite_folder / 'responses.jsonl', binding=binding)
        for key in totals:
            totals[key] += counts[key]
        ids = [row['id'] for row in rows]
        save_new(suite_folder / 'completion.json', dict(schema=COMPLETION_SCHEMA,
            expected_ids=ids, observed_ids=ids, missing_ids=[], complete=counts['exception_rows'] == 0,
            worker_exit_code=0, guard_failure=None, runtime=relative_pin(suite_folder / 'runtime.json', suite_folder),
            responses=relative_pin(suite_folder / 'responses.jsonl', suite_folder), quality_accepted=False))
        completions[name] = P.pin(suite_folder / 'completion.json', root)
    save_new(folder / 'completion.json', dict(status='complete', arm=arm, **totals,
                                            suites=completions, quality_accepted=False))
    validate_outputs(root, plan, arm)


def validate_training(root, plan):
    root = Path(root)
    recipe = P.validate_recipe(plan['recipe'])
    steps, exposures = recipe['steps'], recipe['rows'] * recipe['epochs']
    report = read_json(root / 'training/training.json')
    provenance = read_json(root / 'training/provenance.json')
    P.output_provenance(root, plan, provenance)
    P.require(provenance['schema'] == P.ADAPTER_SCHEMA and provenance['run_id'] == plan['identity']['run_id']
        and provenance['fresh'] is True and provenance['merged'] is False
        and provenance['training'] == P.pin(root / 'training/training.json', root), 'Fresh adapter provenance differs')
    P.require(report['status'] == 'complete' and report['steps'] == report['cumulative_steps'] == steps
        and report['train_rows'] == exposures and report['valid_rows'] == 0
        and report['dropped_over_max_tokens'] == report['valid_dropped'] == 0
        and report['init_adapter'] is None and report['device'] == 'cuda' and report['dtype'] == 'torch.bfloat16'
        and report['base_weights'] == dict(representation='hf-safetensors', dtype='torch.bfloat16')
        and report['manifest_sha256'] == plan['data']['manifest']['sha256']
        and report['base_model_id'] == plan['model']['id'] and report['base_revision'] == plan['model']['revision'],
        'Final-step numerical/data/base acceptance differs')
    for key, recipe_key in dict(batch_size='batch_size', micro_batch_size='micro_batch_size',
            lr='lr', seed='seed', max_tokens='context_tokens', max_answer_tokens='answer_tokens',
            checkpoint_every='checkpoint_every', valid_every='valid_every').items():
        P.require(report[key] == recipe[recipe_key], 'Actual training recipe differs: ' + key)
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    P.require(all(report['lora'][k] == recipe[k] for k in ('rank', 'alpha', 'dropout'))
              and set(report['lora']['targets']) == targets, 'Actual LoRA recipe differs')
    admission = read_json(root / 'admission.json')
    expected = Counter({(identifier, plan['data']['train']['sha256'], line): recipe['epochs']
                        for line, identifier in enumerate(admission['training_ids'], 1)})
    P.require(len(expected) == recipe['rows'] and exposure_counts(report['coverage']) == expected,
              'Exactly-twice source line/id coverage failed')
    P.validate_inventory(provenance['parameter_inventory'])
    masking = provenance['masking']
    P.require(provenance['fresh_zero_b'] is True and provenance['observed_steps'] == steps
        and provenance['coverage_exactly_twice'] is True and masking['rows'] == exposures
        and all(masking[k] is True for k in ('all_prompt_tokens_masked', 'all_answer_tokens_supervised', 'eos_supervised'))
        and 0 < masking['max_context_tokens'] <= recipe['context_tokens']
        and 0 < masking['max_answer_tokens'] <= recipe['answer_tokens'],
        'Observed initialization/masking/EOS/coverage differs')
    P.require(len(report['history']) == steps, 'Incomplete optimizer history')
    # Match train_open_weight_lora.main's LambdaLR, including one-step runs.
    warmup = max(1, steps // 20)
    def multiplier(step):
        return ((step + 1) / warmup if step < warmup else
                0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
    for step, row in enumerate(report['history'], 1):
        P.require(row['step'] == step and math.isfinite(row['train_loss'])
            and 'valid_loss_by_task' not in row
            and math.isclose(row['applied_lr'], recipe['lr'] * multiplier(step - 1), rel_tol=1e-12, abs_tol=1e-15)
            and math.isclose(row['next_lr'], recipe['lr'] * multiplier(step), rel_tol=1e-12, abs_tol=1e-15),
            'Numerical schedule/history differs')
    adapter = adapter_identity(root, plan)
    P.require(provenance['adapter'] == adapter, 'Final adapter file identity differs')
    P.generation_binding(plan, 'adapter', adapter)
    config = read_json(root / adapter['config']['path'])
    P.require(config['r'] == recipe['rank'] and config['lora_alpha'] == recipe['alpha']
        and config['lora_dropout'] == recipe['dropout']
        and config['peft_type'] == 'LORA' and config['task_type'] == 'CAUSAL_LM'
        and config['bias'] == 'none' and not config.get('modules_to_save')
        and set(config['target_modules']) == targets
        and not any(config.get(key) for key in ('use_dora', 'use_rslora', 'use_qalora', 'rank_pattern',
            'alpha_pattern', 'layer_replication', 'layers_to_transform', 'target_parameters',
            'trainable_token_indices', 'lora_bias', 'fan_in_fan_out', 'loftq_config', 'megatron_config')),
        'Unmerged adapter structure differs')
    P.require(not (root / 'training/checkpoints').exists() and not (root / 'training/merged').exists(),
              'Intermediate checkpoints or merged base forbidden')
    return report


def validate_gpu_runtime(directory, runtime):
    """Offline source-bound CUDA admission, deliberately separate from CPU auth."""
    E = evaluator()
    directory = Path(directory).resolve()
    root = directory.parent.parent
    plan_path = E.pinned_path(directory, runtime['plan'])
    P.require(plan_path.resolve() == root / 'plan.json', 'GPU plan location differs')
    plan = read_json(plan_path)
    P.validate_recipe(plan['recipe'])
    P.require(plan['schema'] == P.SCHEMA
              and plan['model']['id'] == P.MODEL_ID and plan['model']['revision'] == P.MODEL_REVISION,
              'Unsupported GPU public-work contract')
    arm, suite_name = directory.parent.name, directory.name
    P.require(arm in ('baseline', 'adapter') and suite_name in SUITES, 'Unknown GPU suite/arm')
    P.require(runtime['schema'] == RUNTIME_SCHEMA and runtime['device'] == 'cuda'
        and runtime['cpu_fallback'] is False and runtime['research_only'] is True
        and runtime['quality_accepted'] is False
        and runtime['generation_policy'] == helpers().requested_policy(512, 4096),
        'GPU runtime/policy differs; CPU authentication cannot be substituted')
    P.require(runtime['model_identity'] == dict(repo_id=P.MODEL_ID, revision=P.MODEL_REVISION),
              'GPU base identity differs')
    loaded_path = E.pinned_path(directory, runtime['loaded_runtime'])
    P.require(loaded_path.resolve() == root / arm / 'loaded-runtime.json', 'GPU loaded runtime location differs')
    loaded = read_json(loaded_path)
    P.output_provenance(root, plan, loaded)
    P.require(loaded['device'] == 'cuda' and loaded['dtype'] == 'torch.bfloat16'
        and loaded['frozen'] is True and loaded['adapter_merged'] is False and loaded['cpu_fallback'] is False
        and loaded['local_files_only'] is True and loaded['trust_remote_code'] is False
        and loaded['attention_implementation'] == 'sdpa'
        and isinstance(loaded['gpu_uuid'], str) and loaded['gpu_uuid'].startswith('GPU-')
        and isinstance(loaded['instance_id'], str) and loaded['instance_id'].startswith('i-'),
        'Actual CUDA load receipt differs')
    expected_files = [{key: p[key] for key in ('path', 'bytes', 'sha256')} for p in plan['model']['files']]
    P.require(loaded['model_files'] == expected_files, 'Loaded model inventory differs')
    inventory = loaded['parameter_inventory']
    P.require(inventory['base'] and all(key == 'torch.bfloat16' and value['tensors'] > 0
        and value['elements'] > 0 and value['bytes'] == value['elements'] * 2
        for key, value in inventory['base'].items()), 'Actual CUDA base precision differs')
    adapter = None
    if arm == 'adapter':
        validate_training(root, plan)
        P.validate_inventory(inventory)
        adapter = adapter_identity(root, plan)
        P.require(loaded['adapter_reload_equal'] is True and runtime['adapter'] is not None,
                  'Missing actual adapter reload proof')
    else:
        P.require(not inventory['adapter'] and runtime['adapter'] is None
                  and loaded['adapter_reload_equal'] is None, 'Baseline contains adapter')
    P.require(loaded['adapter'] == adapter and runtime['binding'] == P.generation_binding(plan, arm, adapter),
              'GPU generation binding differs')
    acquisition_path = E.pinned_path(directory, runtime['model_manifest'])
    P.require(acquisition_path.resolve() == root / 'model-acquisition.json', 'GPU acquisition location differs')
    acquisition = read_json(acquisition_path)
    P.require(acquisition['files'] == expected_files and acquisition['status'] == 'authenticated'
              and acquisition['plan_sha256'] == P.pin(plan_path)['sha256'], 'GPU acquisition inventory differs')
    for key, field in (('suite_manifest', suite_name + '_manifest'), ('inputs', suite_name + '_inputs')):
        actual = P.pin(E.pinned_path(directory, runtime[key]))
        P.require(all(actual[k] == plan['workload'][field][k] for k in ('sha256', 'bytes')),
                  'GPU suite not bound to sealed plan')
    scope, count = SUITES[suite_name]
    suite = read_json(E.pinned_path(directory, runtime['suite_manifest']))
    rows = helpers().validate_public_rows(read_json(E.pinned_path(directory, runtime['inputs'])))
    P.require(runtime['suite_scope'] == suite['scope'] == scope and len(rows) == count,
              'GPU suite scope/count differs')
    return plan


def validate_outputs(root, plan, arm, *, smoke=False):
    P.require(not smoke and arm in ('baseline', 'adapter'), 'Only complete GPU paired evaluation admitted')
    root, E = Path(root), evaluator()
    completion = read_json(root / arm / 'completion.json')
    preflight = read_json(root / arm / 'prompt-preflight.json')
    cases = helpers().validate_public_rows(read_json(root / 'evaluation-inputs.json'))
    P.require(preflight['status'] == 'passed' and preflight['all_cases_preflighted_before_weights'] is True
              and preflight['truncation'] is False and len(cases) == 194
              and [p['id'] for p in preflight['prompts']] == [c['id'] for c in cases],
              'Complete preweight prompt admission missing')
    prepared = {p['id']: p for p in preflight['prompts']}
    observed, exceptions, lengths = [], 0, 0
    for name, (scope, count) in SUITES.items():
        folder = root / arm / name
        suite_path = root / 'suites' / name / 'manifest.json'
        suite = read_json(suite_path)
        rows = read_json(root / 'suites' / name / 'inputs.json')
        P.require(suite['scope'] == scope and len(rows) == count, 'Sealed GPU suite differs')
        records, runtime, result, _ = E.load_arm(folder, rows, suite_path, suite)
        P.require(result['complete'] is True and len(records) == count
                  and completion['suites'][name] == P.pin(folder / 'completion.json', root),
                  'Incomplete GPU suite')
        for row in rows:
            record = records[row['id']]
            P.require(all(record.get(key) == value for key, value in prepared[row['id']].items())
                      and record['binding'] == runtime['binding'], 'Preflight/binding changed during generation')
            observed.append(record)
            exceptions += record['error'] is not None
            lengths += record['finish_reason'] == 'length'
    P.require([r['id'] for r in observed] == [c['id'] for c in cases]
        and completion['status'] == 'complete' and completion['arm'] == arm
        and completion['rows'] == 194 and completion['exception_rows'] == exceptions == 0
        and completion['length_rows'] == lengths and completion['quality_accepted'] is False,
        'Full ordered paired generation acceptance differs')
    return observed


def child_stage(core, plan, base, phase):
    P.require(plan['schema'] == P.SCHEMA and phase in ('baseline', 'train', 'adapter'),
              'Unknown public-work worker stage')
    base = Path(base)
    core.validate_model(base / 'model', plan)
    torch = core.native_runtime(plan, base)
    if phase == 'train':
        train_stage(core, plan, base, torch)
    else:
        evaluation_stage(core, plan, base, torch, phase)


def run_experiment(core, plan, base, report, gpu, account, work_deadline,
                   acquisition_deadline, budget, hosts, units):
    base = Path(base)
    root = base / 'output'
    os.chown(root, 0, account.pw_gid)
    root.chmod(0o1770)
    shared.freeze(root / 'runtime.json')
    for row in plan['model']['files']:
        core.download(row, base / 'model', hosts, acquisition_deadline, budget)
    core.validate_model(base / 'model', plan)
    for path in (base / 'model').iterdir():
        path.chmod(0o444)
    save_new(root / 'model-acquisition.json', dict(status='authenticated', repo_id=plan['model']['id'],
        revision=plan['model']['revision'], plan_sha256=core.digest(base / 'plan.json'),
        files=[P.pin(base / 'model' / p['path'], base / 'model') for p in plan['model']['files']]))
    copies = [(base / 'plan.json', 'plan.json'),
        (base / 'code' / plan['workload']['source_manifest']['path'], 'source-manifest.json'),
        (base / 'code' / plan['workload']['evaluation_inputs']['path'], 'evaluation-inputs.json'),
        (base / 'code' / plan['data']['admission']['path'], 'admission.json')]
    for name in SUITES:
        copies.extend((base / 'code' / plan['workload'][name + '_' + key]['path'],
                       'suites/' + name + '/' + target)
                      for key, target in (('manifest', 'manifest.json'), ('inputs', 'inputs.json')))
    copies.extend((base / 'model' / p['path'], 'tokenizer/' + p['path'])
                  for p in plan['model']['files'] if p['path'] in TOKENIZER_FILES)
    for source, target in copies:
        copy_new(source, root / target)
        shared.freeze(root / target)
    for name in ('model-acquisition.json', 'suites', 'tokenizer'):
        shared.freeze(root / name)
    for phase in ('baseline', 'train', 'adapter'):
        unit = core.supervise_phase(plan, base, phase, gpu, account.pw_uid, account.pw_gid,
                                    work_deadline, units)
        P.require(unit['cleanup_verified'] is True and not units, 'Owned native processes remain')
        report['stages'].append(unit)
        if phase == 'train':
            validate_training(root, plan)
            shared.freeze(root / 'training')
        else:
            validate_outputs(root, plan, phase)
            shared.freeze(root / phase)
    save_new(root / 'worker-result.json', dict(status='complete', run_id=plan['identity']['run_id'],
        plan_sha256=core.digest(base / 'plan.json'), assessment_answers=388,
        quality_accepted=False, stages=report['stages']))
    shared.freeze(root / 'worker-result.json')
