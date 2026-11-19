"""Finite public-only HF worker; cloud supervision stays in grounded_hf_worker.

The separate injected-model CPU hook is local transport evidence, never GPU admission.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

if __package__:
    from . import public_evidence_experiment as P
else:
    import public_evidence_experiment as P


def helpers(base=None):
    scripts = Path(base) / 'code/scripts' if base is not None else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scripts))
    import evaluate_evidence_conditioning
    return evaluate_evidence_conditioning


def save_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        append_row(stream, value)


def append_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def preflight_prompts(tokenizer, cases, *, context_limit, max_new_tokens):
    """Reject the whole corpus before any model load or response-file creation."""
    E = helpers()
    P.require(type(context_limit) is int and type(max_new_tokens) is int
              and 0 < max_new_tokens < context_limit, 'Invalid native token envelope')
    P.require(isinstance(cases, list) and cases, 'Public cases must be nonempty')
    seen, prepared = set(), []
    for case in cases:
        P.require(isinstance(case, dict) and set(case) == {'id', 'messages'}
                  and isinstance(case['id'], str) and case['id'] and case['id'] not in seen,
                  'Invalid/duplicate public case; private fields forbidden')
        messages = case['messages']
        P.require(isinstance(messages, list) and messages and all(
            isinstance(m, dict) and set(m) == {'role', 'content'}
            and m['role'] in ('system', 'user', 'assistant') and isinstance(m['content'], str)
            for m in messages) and messages[-1]['role'] == 'user', 'Invalid public messages')
        seen.add(case['id'])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        P.require(isinstance(prompt, str) and prompt.endswith('<think>\n\n</think>\n\n'),
                  'Native Qwen3 nonthinking prefix differs')
        ids = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
        P.require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids),
                  'Invalid native token IDs')
        P.require(len(ids) + max_new_tokens <= context_limit,
                  'Full-corpus prompt preflight exceeds context; no truncation')
        prepared.append(dict(id=case['id'], prompt_sha256=E.object_sha(messages),
            rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_token_ids=ids, input_tokens=len(ids)))
    return prepared


def stream_responses(model, tokenizer, cases, prepared, output_path, *, max_new_tokens,
                     context_limit, mode='evaluation', binding=None):
    """One real helper call per case; retain exceptions and length stops without retry."""
    E = helpers()
    P.require(len(prepared) == len(cases) and [p['id'] for p in prepared] == [c['id'] for c in cases],
              'Complete ordered prompt preflight required')
    counts = dict(rows=0, exception_rows=0, length_rows=0)
    with Path(output_path).open('x', encoding='utf-8') as stream:
        for case, prompt in zip(cases, prepared):
            records = E.generate_responses(model, tokenizer, [case], max_new_tokens=max_new_tokens,
                context_limit=context_limit, profile=E.SAMPLING_PROFILE, mode=mode)
            P.require(len(records) == 1, 'One response required per generation attempt')
            record = records[0]
            if binding is not None:
                record['binding'] = binding
            # Durable raw evidence precedes any receipt acceptance decision.
            append_row(stream, record)
            E.validate_generation_receipt(record)
            P.require(record['id'] == case['id'] and record['prompt_sha256'] == prompt['prompt_sha256'],
                      'Generation changed the preflight case identity')
            if record['error'] is None:
                P.require(record['rendered_prompt_sha256'] == prompt['rendered_prompt_sha256']
                          and record['prompt_token_ids'] == prompt['prompt_token_ids'],
                          'Generation prompt differs from full preflight')
            counts['rows'] += 1
            counts['exception_rows'] += record['error'] is not None
            counts['length_rows'] += record['finish_reason'] == 'length'
    return counts


def tiny_cpu_generation(model, tokenizer, cases, output_path, *, max_new_tokens=8, context_limit=4096):
    """Actual caller-owned tiny CPU model only; no downloads, CUDA receipt or quality claim."""
    parameters = list(model.parameters())
    P.require(parameters and all(p.device.type == 'cpu' for p in parameters),
              'Local smoke requires an actual all-CPU model')
    model_context = getattr(model.config, 'max_position_embeddings', context_limit)
    if type(model_context) is int and model_context > 0:
        context_limit = min(context_limit, model_context)
    prepared = preflight_prompts(tokenizer, cases, context_limit=context_limit,
                                 max_new_tokens=max_new_tokens)
    counts = stream_responses(model, tokenizer, cases, prepared, output_path,
        max_new_tokens=max_new_tokens, context_limit=context_limit, mode='smoke')
    # Exercise the same native-receipt reader used by the local assessor.
    helpers().load_responses(output_path, cases)
    return dict(role='smoke_only', device='cpu', gpu_admission=False, quality_accepted=False,
                output=P.pin(Path(output_path)), **counts)


def parameter_inventory(model, torch, *, training=False):
    inventory = dict(base={}, adapter={})
    for name, parameter in model.named_parameters():
        arm = 'adapter' if '.lora_' in name else 'base'
        expected = torch.float32 if arm == 'adapter' else torch.bfloat16
        P.require(parameter.dtype == expected and parameter.device.type == 'cuda',
                  'Actual CUDA base/adapter dtype differs')
        P.require(parameter.requires_grad == (training and arm == 'adapter'),
                  'Unexpected trainable parameter')
        key = str(parameter.dtype)
        item = inventory[arm].setdefault(key, dict(tensors=0, elements=0, bytes=0))
        item['tensors'] += 1
        item['elements'] += parameter.numel()
        item['bytes'] += parameter.numel() * parameter.element_size()
    P.require(inventory['base'], 'Missing native base parameters')
    P.require(not any(getattr(module, 'merged', False) for module in model.modules()),
              'Merged adapter forbidden')
    return inventory


def capacity_stage(core, plan, base, torch):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc
    base = Path(base)
    recipe = plan['recipe']
    admitted = P.validate_data(base / 'code', plan)
    trainer = core.trainer_module(base)
    tokenizer = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    manifest_path = base / 'code' / plan['data']['manifest']['path']
    manifest = json.loads(manifest_path.read_text())
    items, dropped = trainer['build_items'](tokenizer, manifest['train'], manifest_path.parent,
        'qwen3', recipe['context_tokens'], recipe['answer_tokens'], True)
    P.require(dropped == 0 and len(items) == recipe['rows'] * recipe['epochs'],
              'Capacity tokenization dropped or repeated admitted rows')
    admitted_rows = len(items)
    del items
    preflight_prompts(tokenizer, admitted['evaluation_inputs'], context_limit=recipe['context_tokens'],
                      max_new_tokens=recipe['answer_tokens'])
    folder = base / 'output/capacity'
    folder.mkdir(mode=0o700)
    item = core.synthetic_capacity_item(tokenizer, trainer, folder, recipe['context_tokens'], recipe['answer_tokens'])
    model = AutoModelForCausalLM.from_pretrained(base / 'model', torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False).to('cuda')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'],
        lora_dropout=recipe['dropout'], target_modules=trainer['TARGETS'], task_type='CAUSAL_LM'))
    inventory = parameter_inventory(model, torch, training=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    adapter_b = [p for n, p in model.named_parameters() if p.requires_grad and '.lora_B.' in n]
    P.require(adapter_b and all(int(torch.count_nonzero(p)) == 0 for p in adapter_b),
              'Synthetic adapter must initialize fresh')
    optimizer = torch.optim.AdamW(parameters, lr=recipe['lr'], weight_decay=0.0)
    losses = []
    model.train()
    for update in range(2):
        if update:
            P.require(all(p in optimizer.state and optimizer.state[p]['exp_avg'].device.type == 'cuda'
                          and optimizer.state[p]['exp_avg_sq'].device.type == 'cuda' for p in parameters),
                      'Adam resident state missing before second full-envelope backward')
        optimizer.zero_grad(set_to_none=True)
        for _ in range(recipe['batch_size']):
            loss = trainer['batch_loss'](model, [item], 'cuda',
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                'qwen3') / recipe['batch_size']
            P.require(bool(torch.isfinite(loss)), 'Nonfinite capacity loss')
            loss.backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        P.require(bool(torch.isfinite(norm)) and float(norm) > 0, 'Invalid capacity gradients')
        optimizer.step()
        P.require(all(bool(torch.isfinite(p).all()) for p in parameters)
                  and any(int(torch.count_nonzero(p)) > 0 for p in adapter_b), 'Invalid synthetic update')
    model.save_pretrained(folder / 'adapter', safe_serialization=True)
    expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    del loss, optimizer, parameters, adapter_b, model
    gc.collect()
    torch.cuda.empty_cache()
    native = AutoModelForCausalLM.from_pretrained(base / 'model', torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False).to('cuda')
    loaded = PeftModel.from_pretrained(native, folder / 'adapter', is_trainable=False, local_files_only=True)
    actual = dict(loaded.named_parameters())
    P.require(all(n in actual and torch.equal(p, actual[n].detach().cpu()) for n, p in expected.items()),
              'Synthetic adapter reload differs')
    loaded.eval()
    with torch.no_grad():
        result = loaded(input_ids=torch.tensor([item['ids'][:32]], device='cuda'), use_cache=False)
        P.require(bool(torch.isfinite(result.logits).all()), 'Reloaded synthetic forward is nonfinite')
    save_new(base / 'output/capacity-receipt.json', dict(synthetic=True, discarded=False,
        optimizer_updates=2, microbatches=2 * recipe['batch_size'], microbatches_per_update=recipe['batch_size'],
        optimizer_state_resident_before_second_update=True, gradient_checkpointing=True,
        context_tokens=len(item['ids']), answer_tokens=recipe['answer_tokens'], losses=losses,
        finite_parameter_update=True, adapter_reload_equal=True, parameter_inventory=inventory,
        synthetic_row_sha256=core.digest(folder / 'synthetic.jsonl'), admitted_expanded_rows=admitted_rows,
        plan_sha256=core.digest(base / 'plan.json'), source_manifest=plan['workload']['source_manifest'],
        runtime=P.pin(base / 'output/runtime.json', base / 'output'),
        max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved()))


def adapter_identity(root, plan):
    root = Path(root)
    return dict(config=P.pin(root / 'training/adapter/adapter_config.json', root),
                weights=P.pin(root / 'training/adapter/adapter_model.safetensors', root),
                step=plan['recipe']['steps'])


def verify_loaded_adapter(model, weights, torch):
    from peft import get_peft_model_state_dict
    from safetensors import safe_open
    actual = get_peft_model_state_dict(model)
    with safe_open(weights, framework='pt', device='cpu') as saved:
        P.require(actual and set(actual) == set(saved.keys()), 'Loaded adapter tensor coverage differs')
        for name, parameter in actual.items():
            expected = saved.get_tensor(name)
            P.require(parameter.dtype == expected.dtype and torch.equal(parameter.detach().cpu(), expected),
                      'Loaded adapter tensor differs from authenticated final weights')


def evaluation_stage(core, plan, base, torch, arm):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = Path(base)
    root = base / 'output'
    P.require(arm in ('baseline', 'adapter'), 'Unknown public evaluation arm')
    admitted = P.validate_data(base / 'code', plan)
    cases = admitted['evaluation_inputs']
    folder = root / arm
    folder.mkdir(mode=0o700)
    tokenizer = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    recipe = plan['recipe']
    prepared = preflight_prompts(tokenizer, cases, context_limit=recipe['context_tokens'],
                                 max_new_tokens=recipe['answer_tokens'])
    save_new(folder / 'prompt-preflight.json', dict(status='passed', all_cases_preflighted_before_weights=True,
                                                   truncation=False, prompts=prepared))
    adapter = None
    if arm == 'adapter':
        P.validate_training(root, plan)
        adapter = adapter_identity(root, plan)
    model = AutoModelForCausalLM.from_pretrained(base / 'model', torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, root / 'training/adapter', is_trainable=False,
                                         local_files_only=True, autocast_adapter_dtype=True)
        verify_loaded_adapter(model, root / 'training/adapter/adapter_model.safetensors', torch)
    model.requires_grad_(False)
    model.eval()
    inventory = parameter_inventory(model, torch)
    P.require(bool(inventory['adapter']) == (arm == 'adapter'), 'Wrong adapter loaded for arm')
    release = json.loads((base / 'release.json').read_text())
    save_new(folder / 'loaded-runtime.json', dict(device='cuda', dtype='torch.bfloat16', frozen=True,
        adapter_merged=False, parameter_inventory=inventory, adapter=adapter, cpu_fallback=False,
        adapter_reload_equal=True if adapter is not None else None,
        gpu_uuid=release['gpu']['uuid'], instance_id=release['instance_id'],
        plan_sha256=core.digest(base / 'plan.json'), runtime=P.pin(root / 'runtime.json', root),
        model_files=[P.pin(base / 'model' / p['path'], base / 'model') for p in plan['model']['files']],
        local_files_only=True, trust_remote_code=False, attention_implementation=model.config._attn_implementation))
    counts = stream_responses(model, tokenizer, cases, prepared, folder / 'responses.jsonl',
        max_new_tokens=recipe['answer_tokens'], context_limit=recipe['context_tokens'],
        binding=P.generation_binding(plan, arm, adapter))
    save_new(folder / 'completion.json', dict(status='complete', arm=arm, **counts,
        outputs=P.pin(folder / 'responses.jsonl', root), quality_accepted=False))
    P.validate_outputs(root, plan, arm)


def child_stage(core, plan, base, phase):
    P.require(plan['schema'] == P.SCHEMA and phase in ('capacity', 'baseline', 'train', 'adapter'),
              'Unknown public worker stage')
    base = Path(base)
    core.validate_model(base / 'model', plan)
    torch = core.native_runtime(plan, base)
    if phase == 'capacity':
        capacity_stage(core, plan, base, torch)
    elif phase == 'train':
        train_stage(core, plan, base, torch)
    else:
        evaluation_stage(core, plan, base, torch, phase)


def training_observer(plan, torch, eos_token_id):
    """Observe the real trainer's already-built items, model, and consumed coverage."""
    P.require(plan['recipe'] == P.RECIPE, 'Sealed paired training recipe required')
    measured, expected = {}, set()

    def identity(row):
        return row['id'], row['source_sha256'], row['source_line']

    def observe(event, **values):
        if event == 'admission':
            items = values['train']
            recipe = P.RECIPE
            P.require(not values['valid'] and values['dropped'] == 0 and values['valid_dropped'] == 0
                      and len(items) == recipe['rows'] * recipe['epochs'],
                      'Trainer changed the finite admitted exposure set')
            max_context, max_answer = 0, 0
            for item in items:
                ids, labels = item['ids'], item['labels']
                P.require(len(ids) == len(labels) and -100 in labels, 'Missing answer-only mask')
                first = next((i for i, token in enumerate(labels) if token != -100), len(labels))
                P.require(0 < first < len(labels) and labels[:first] == [-100] * first
                          and labels[first:] == ids[first:] and ids[-1] == eos_token_id,
                          'Prompt masking, answer targets or supervised EOS differs')
                key = identity(item['identity'])
                P.require(key not in expected, 'Repeated physical training exposure')
                expected.add(key)
                max_context = max(max_context, len(ids))
                max_answer = max(max_answer, len(ids) - first)
            P.require(max_context <= recipe['context_tokens'] and max_answer <= recipe['answer_tokens'],
                      'Observed tokenization exceeds admitted envelope')
            measured['masking'] = dict(rows=len(items), all_prompt_tokens_masked=True,
                all_answer_tokens_supervised=True, eos_supervised=True,
                max_context_tokens=max_context, max_answer_tokens=max_answer)
        elif event == 'selection':
            return dict(status='unselected', reason='Fixed last step; no evaluation-based checkpoint selection')
        elif event == 'model':
            model = values['model']
            measured['parameter_inventory'] = parameter_inventory(model, torch, training=True)
            matrices = [p for n, p in model.named_parameters() if p.requires_grad and '.lora_B.' in n]
            P.require(matrices and all(int(torch.count_nonzero(p)) == 0 for p in matrices),
                      'Training adapter is not fresh zero-B initialization')
            P.require(not values['optimizer'].state, 'Optimizer is not fresh')
            measured['fresh_zero_b'] = True
        elif event == 'completion':
            coverage = [identity(row) for row in values['coverage']]
            P.require(len(coverage) == len(expected) and set(coverage) == expected
                      and values['steps'] == P.RECIPE['steps'], 'Training wrapped, skipped or repeated rows')
            measured.update(observed_steps=values['steps'], coverage_exactly_once=True)
        elif event in ('before_step', 'after_step', 'report'):
            return {}
        return None

    return observe, measured


def train_stage(core, plan, base, torch):
    from transformers import AutoTokenizer
    base = Path(base)
    root = base / 'output'
    tokenizer = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    observer, measured = training_observer(plan, torch, tokenizer.eos_token_id)
    argv = core.train_argv(plan, base)
    core.trainer_module(base)['main'](argv[1:], observer=observer)
    save_new(root / 'training/provenance.json', dict(schema=P.ADAPTER_SCHEMA,
        run_id=plan['identity']['run_id'], plan_sha256=core.digest(base / 'plan.json'),
        training=P.pin(root / 'training/training.json', root), adapter=adapter_identity(root, plan),
        source_manifest=plan['workload']['source_manifest'], runtime=P.pin(root / 'runtime.json', root),
        fresh=True, merged=False, **measured))
    P.validate_training(root, plan)


def freeze(path):
    path = Path(path)
    members = [path, *path.rglob('*')] if path.is_dir() else [path]
    for member in members:
        P.require(not member.is_symlink(), 'Unsafe accepted output path')
        os.chown(member, 0, 0)
        member.chmod(0o555 if member.is_dir() else 0o444)


def run_experiment(core, plan, base, report, gpu, account, work_deadline,
                   acquisition_deadline, budget, hosts, units):
    base = Path(base)
    root = base / 'output'
    # Dedicated-group sticky parent protects accepted root-owned sibling artifacts.
    os.chown(root, 0, account.pw_gid)
    root.chmod(0o1770)
    freeze(root / 'runtime.json')
    for row in plan['model']['files']:
        core.download(row, base / 'model', hosts, acquisition_deadline, budget)
    core.validate_model(base / 'model', plan)
    for path in (base / 'model').iterdir():
        path.chmod(0o444)
    save_new(root / 'model-acquisition.json', dict(status='authenticated', model_id=plan['model']['id'],
        revision=plan['model']['revision'], plan_sha256=core.digest(base / 'plan.json'),
        files=[P.pin(base / 'model' / p['path'], base / 'model') for p in plan['model']['files']]))
    for source, target in ((base / 'plan.json', 'plan.json'),
        (base / 'code' / plan['workload']['source_manifest']['path'], 'source-manifest.json'),
        (base / 'code' / plan['workload']['evaluation_inputs']['path'], 'evaluation-inputs.json'),
        (base / 'code' / plan['data']['admission']['path'], 'admission.json')):
        with source.open('rb') as src, (root / target).open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        freeze(root / target)
    freeze(root / 'model-acquisition.json')
    for phase in ('capacity', 'baseline', 'train', 'adapter'):
        unit = core.supervise_phase(plan, base, phase, gpu, account.pw_uid, account.pw_gid,
                                    work_deadline, units)
        P.require(unit['cleanup_verified'] is True and not units, 'Owned native processes remain')
        report['stages'].append(unit)
        if phase == 'capacity':
            scratch = root / 'capacity'
            P.require(scratch.is_dir() and not scratch.is_symlink(), 'Unsafe capacity scratch')
            shutil.rmtree(scratch)
            receipt = json.loads((root / 'capacity-receipt.json').read_text())
            receipt['discarded'] = True
            core.save(root / 'capacity-receipt.json', receipt)
            P.validate_capacity(root, plan)
            freeze(root / 'capacity-receipt.json')
        elif phase == 'train':
            P.validate_training(root, plan)
            freeze(root / 'training')
        else:
            P.validate_outputs(root, plan, phase)
            freeze(root / phase)
    save_new(root / 'worker-result.json', dict(status='complete', run_id=plan['identity']['run_id'],
        plan_sha256=core.digest(base / 'plan.json'), assessment_answers=sum(
            json.loads((root / arm / 'completion.json').read_text())['rows'] for arm in ('baseline', 'adapter')),
        quality_accepted=False, stages=report['stages']))
    freeze(root / 'worker-result.json')
