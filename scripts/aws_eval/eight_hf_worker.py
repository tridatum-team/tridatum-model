"""One fixed 8B SFT: offline native stages and root-owned delayed assessment channel."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

from grounded_hf_lifecycle import (MAX_EXPORT, digest, exclusive_intent,
    object_sha, remaining, require, save)
import eight_hf_experiment as E


CHECKPOINT_PHASES = {f'checkpoint-step-{step:06d}-dev': step for step in (12, 24, 36)}
DEV_EXPORT_MAX = 32 * 1024**2
FINAL_MEMBERS = {'final-input-receipt.json', 'final-base-outputs.jsonl',
                 'final-adapter-outputs.jsonl', 'final_base.log', 'final_adapter.log'}
SELECTION_MEMBERS = {'checkpoint-selection.json', 'parent_selection-receipt.json'}


def phase_kind(phase):
    if phase in ('runtime', 'capacity', 'train'):
        return phase
    if phase in ('baseline_dev', 'final_base', 'final_adapter') or phase in CHECKPOINT_PHASES:
        return 'generation'
    return None


def validate_prompt_budget(plan, prompt_tokens):
    generation = plan['workload']['generation']
    require(type(prompt_tokens) is int and prompt_tokens > 0
            and prompt_tokens + generation['max_new_tokens'] <= generation['context_tokens'],
            'Input exceeds matched context envelope; no truncation')


def input_rows(path, pin, ids):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'],
            'Input-only bundle authentication failed')
    rows = json.loads(path.read_text())
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
            and [row.get('id') for row in rows] == ids and len(set(ids)) == len(ids),
            'Input-only cohort membership/order differs')
    for row in rows:
        require(set(row) == {'id', 'messages'} and isinstance(row['messages'], list) and row['messages'],
                'Input-only bundle contains references or malformed messages')
        require(all(isinstance(m, dict) and set(m) == {'role', 'content'}
                    and m['role'] in ('system', 'user', 'assistant') and isinstance(m['content'], str)
                    for m in row['messages']) and row['messages'][-1]['role'] == 'user',
                'Input-only message contract differs')
    return rows


def adapter_hashes(plan, base):
    base = Path(base)
    root = base / 'output/training'
    hashes, ledger = E.checkpoint_evidence(plan, base / 'output')
    for row in ledger['checkpoints']:
        step = row['step']
        folder = root / 'checkpoints' / f'step-{step:06d}'
        adapter = folder / 'adapter'
        require(Path(row['path']) == adapter and adapter.is_dir() and not adapter.is_symlink(),
                'Checkpoint adapter path differs')
        require({p.name for p in adapter.iterdir()} == {'adapter_config.json', 'adapter_model.safetensors', 'README.md'},
                'Reloadable checkpoint membership differs')
        for path in adapter.iterdir():
            require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
                    'Unsafe checkpoint adapter member')
        config = json.loads((adapter / 'adapter_config.json').read_text())
        require(config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0
                and config['task_type'] == 'CAUSAL_LM', 'Checkpoint adapter recipe differs')
    require(Path(ledger['last_step']['path']) == root / 'adapter', 'Final training adapter path differs')
    return hashes


def exposure_tracker(trainer):
    """Observe unchanged trainer inputs and real loss calls, without altering sampling/loss."""
    scope = trainer['main'].__globals__
    build, loss = scope['build_items'], scope['batch_loss']
    expected, observed, validation = Counter(), [], []
    def fingerprint(item):
        return object_sha({key: item[key] for key in ('ids', 'labels', 'task', 'file')})
    def tracked_build(*args, **kwargs):
        items, dropped = build(*args, **kwargs)
        expand = args[6] if len(args) > 6 else kwargs['expand']
        require(dropped == 0, 'Training/development tokenization dropped rows')
        if expand:
            require(not expected and len(items) == 72, 'Actual expanded training exposure differs')
            expected.update(fingerprint(item) for item in items)
            require(len(expected) == 24 and set(expected.values()) == {3}, 'Each training row must occur exactly three times')
        else:
            require(not validation and len(items) == 8, 'Actual held-out development cohort differs')
            validation.extend(fingerprint(item) for item in items)
            require(len(set(validation)) == 8 and not set(validation) & set(expected),
                    'Development rows are duplicated or included in training')
        return items, dropped
    def tracked_loss(model, items, *args, **kwargs):
        if model.training:
            require(len(items) == 1, 'Training microbatch differs')
            observed.extend(fingerprint(item) for item in items)
            require(len(observed) <= 72 and all(item in expected for item in observed[-len(items):]),
                    'Unadmitted training exposure')
        else:
            require(all(fingerprint(item) in validation for item in items), 'Validation consumed non-development data')
        return loss(model, items, *args, **kwargs)
    scope['build_items'], scope['batch_loss'] = tracked_build, tracked_loss
    def finish():
        require(len(observed) == 72 and Counter(observed) == expected and len(validation) == 8,
                'Actual optimizer exposures differ from 24 rows times three')
        return dict(training_exposures=72, unique_training_rows=24, exposures_per_row=3,
                    development_rows=8, dropped_rows=0, training_item_hashes=observed,
                    development_item_hashes=validation)
    return finish


def training_stage(worker, plan, base, torch):
    trainer = worker.trainer_module(base)
    finish = exposure_tracker(trainer)
    old_argv = sys.argv
    try:
        sys.argv = worker.train_argv(plan, base)
        trainer['main']()
    finally:
        sys.argv = old_argv
    receipt = finish()
    report = json.loads((Path(base) / 'output/training/training.json').read_text())
    require(report['status'] == 'complete' and report['steps'] == report['cumulative_steps'] == 36
            and report['train_rows'] == 72 and report['valid_rows'] == 8
            and report['dropped_over_max_tokens'] == report['valid_dropped'] == 0
            and report['init_adapter'] is None
            and report['base_weights'] == {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'}
            and [row['step'] for row in report['history']] == list(range(1, 37))
            and [row['step'] for row in report['history'] if 'valid_loss_by_task' in row] == [12, 24, 36],
            'Finite native training receipt differs')
    receipt.update(run_id=plan['identity']['run_id'], plan_sha256=digest(Path(base) / 'plan.json'),
                   manifest_sha256=plan['data']['manifest']['sha256'], checkpoint_hashes=adapter_hashes(plan, base))
    save(Path(base) / 'output/training/exposure-receipt.json', receipt)


def validate_decision(plan, base, decision, checkpoint):
    step = E.validate_selection(plan, digest(Path(base) / 'plan.json'), decision,
                                checkpoint['archive']['sha256'], checkpoint['checkpoint_hashes'])
    hashes, ledger = E.checkpoint_evidence(plan, Path(base) / 'output')
    require(hashes == checkpoint['checkpoint_hashes'], 'Checkpoint weights changed after development export')
    for candidate, row in zip(decision['aggregate']['candidates'], ledger['checkpoints']):
        losses = row['valid_loss_by_task']
        require(candidate['teacher_forced_dev_loss'] == sum(losses.values()) / len(losses),
                'Selection development loss differs from authenticated checkpoint ledger')
    return step


def selected_step(plan, base):
    base = Path(base)
    decision = json.loads((base / 'checkpoint-selection.json').read_text())
    receipt = json.loads((base / 'dev-checkpoint.json').read_text())
    return validate_decision(plan, base, decision, receipt)


def generation_stage(worker, plan, base, torch, phase):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    base = Path(base)
    work = plan['workload']
    generation = work['generation']
    final = phase in ('final_base', 'final_adapter')
    step = CHECKPOINT_PHASES.get(phase)
    if final:
        selection = selected_step(plan, base)
        require(selection in (12, 24, 36), 'Final generation requires an eligible selected checkpoint')
        step = selection if phase == 'final_adapter' else None
        ids = work['final_case_ids']['general'] + work['final_case_ids']['mois']
        cases = input_rows(base / 'final-inputs.json', work['final_inputs'], ids)
        target = 'final-base-outputs.jsonl' if step is None else 'final-adapter-outputs.jsonl'
    else:
        require(phase == 'baseline_dev' or step is not None, 'Unknown development generation phase')
        cases = input_rows(base / 'code' / work['development_inputs']['path'],
                           work['development_inputs'], work['development_case_ids'])
        target = 'baseline-dev-outputs.jsonl' if step is None else f'checkpoint-step-{step:06d}-dev-outputs.jsonl'
    arm = 'base' if step is None else f'step-{step:06d}'
    weights_sha = None if step is None else adapter_hashes(plan, base)[arm]
    tok = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base / 'model', torch_dtype=torch.bfloat16,
                                                local_files_only=True, trust_remote_code=False).to('cuda')
    if step is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, base / 'output/training/checkpoints' / arm / 'adapter',
                                         is_trainable=False, local_files_only=True)
    model.eval()
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=generation['max_new_tokens'],
        use_cache=True, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        bos_token_id=tok.bos_token_id, return_dict_in_generate=True)
    with (base / 'output' / target).open('x') as stream:
        for case in cases:
            torch.manual_seed(14); torch.cuda.manual_seed_all(14)
            prompt = tok.apply_chat_template(case['messages'], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
            width = inputs['input_ids'].shape[1]
            validate_prompt_budget(plan, width)
            with torch.inference_mode():
                result = model.generate(**inputs, generation_config=config)
            tokens = result.sequences[0, width:].tolist()
            decoded = worker.decode_generation(tok, tokens, config.eos_token_id, generation['max_new_tokens'])
            cohort = ('general' if case['id'] in work['final_case_ids']['general'] else 'mois') if final else 'dev'
            row = dict(id=case['id'], cohort=cohort, arm=arm, adapter_weights_sha256=weights_sha,
                run_id=plan['identity']['run_id'], plan_sha256=digest(base / 'plan.json'),
                model_id=plan['model']['id'], model_revision=plan['model']['revision'],
                input_sha256=work['final_inputs' if final else 'development_inputs']['sha256'],
                messages=case['messages'], prompt_sha256=object_sha(case['messages']),
                rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_token_ids=inputs['input_ids'][0].tolist(), output_token_ids=tokens,
                generation=work['generation'], **decoded)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    rows = [json.loads(line) for line in (base / 'output' / target).read_text().splitlines()]
    E.validate_outputs(plan, digest(base / 'plan.json'), rows, 'final' if final else 'dev',
                       step, {} if step is None else {arm: weights_sha})


def child_stage(worker, plan, base, phase):
    require(E.is_eight(plan) and phase_kind(phase) not in (None, 'runtime'), 'Unknown experimental native phase')
    worker.validate_model(Path(base) / 'model', plan)
    torch = worker.native_runtime(plan, base)
    if phase == 'capacity':
        worker.capacity_stage(plan, base, torch)
    elif phase == 'train':
        training_stage(worker, plan, base, torch)
    else:
        generation_stage(worker, plan, base, torch, phase)


def await_object(client, bucket, key, target, deadline, maximum, *, max_polls=360):
    """Only a complete exact-key empty LIST proves absence; every service error is fatal."""
    target = Path(target)
    require(not target.exists(), 'No control-channel download reuse')
    require(type(max_polls) is int and 0 < max_polls <= 360, 'Invalid control-channel request cap')
    old_deadline = client.deadline
    client.deadline = min(old_deadline, deadline)
    try:
        for _ in range(max_polls):
            remaining(deadline)
            listing = client('s3api', 'list-objects-v2', '--bucket', bucket, '--prefix', key, '--max-keys', '2')
            contents = listing.get('Contents', [])
            require(listing.get('Name') == bucket and listing.get('Prefix') == key
                    and listing.get('IsTruncated') is False and not listing.get('NextContinuationToken')
                    and not listing.get('CommonPrefixes') and listing.get('MaxKeys') == 2
                    and isinstance(contents, list) and len(contents) <= 1
                    and type(listing.get('KeyCount')) is int and listing['KeyCount'] == len(contents),
                    'Control-channel listing is not exact and complete')
            if not contents:
                time.sleep(min(5, remaining(deadline)))
                continue
            row = contents[0]
            require(isinstance(row, dict) and row.get('Key') == key and type(row.get('Size')) is int
                    and 0 < row['Size'] <= maximum, 'Control-channel key or size differs')
            client('s3api', 'get-object', '--bucket', bucket, '--key', key,
                   '--range', f'bytes=0-{maximum}', str(target))
            remaining(deadline)
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == row['Size']
                    and target.stat().st_size <= maximum, 'Control-channel body size differs')
            return target
        raise RuntimeError('Control-channel LIST request cap exhausted')
    finally:
        client.deadline = old_deadline


def aggregate(base, report, phase, action):
    receipt = dict(phase=phase, state='incomplete', started_epoch=time.time(),
                   run_id=report['run_id'], plan_sha256=report['plan_sha256'], instance_id=report['instance_id'])
    path = Path(base) / 'output' / (phase + '-receipt.json')
    save(path, receipt)
    report['stages'].append(receipt)
    try:
        result = action()
        receipt['state'] = 'complete'
        return result
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['finished_epoch'] = time.time()
        save(path, receipt)


def make_dev_export(worker, plan, base, report, client, deadline):
    base = Path(base)
    inventory = worker.export_inventory(base / 'output', MAX_EXPORT)
    actual = {row['path'] for row in inventory['files']}
    required = set(E.output_members(plan)) - FINAL_MEMBERS - SELECTION_MEMBERS
    require(actual == required, 'Development output membership differs: ' + str(sorted(actual ^ required)))
    members = set(E.dev_output_members(plan))
    files = [row for row in inventory['files'] if row['path'] in members]
    require({row['path'] for row in files} == members, 'Development export membership differs')
    predicted = 10240 + sum(512 + ((r['bytes'] + 511) // 512) * 512 for r in files)
    require(predicted <= DEV_EXPORT_MAX, 'Development archive exceeds output envelope')
    target = base / 'dev-checkpoint.tar'
    with tarfile.open(target, 'x:', format=tarfile.USTAR_FORMAT) as archive:
        for row in files:
            remaining(deadline)
            path = base / 'output' / row['path']
            require(digest(path) == row['sha256'], 'Development output changed before export')
            info = tarfile.TarInfo(row['path']); info.size = row['bytes']; info.mode = 0o444
            with path.open('rb') as source:
                archive.addfile(info, source)
    require(target.stat().st_size <= DEV_EXPORT_MAX, 'Development archive exceeds output envelope')
    receipt = dict(schema='trid-eight-sft-dev-export-v1',
        run_id=plan['identity']['run_id'], plan_sha256=digest(base / 'plan.json'),
        instance_id=report['instance_id'], checkpoint_hashes=adapter_hashes(plan, base), files=files,
        archive=dict(bytes=target.stat().st_size, sha256=digest(target)))
    receipt_path = base / 'dev-checkpoint.json'
    save(receipt_path, receipt)
    receipt_path.chmod(0o444)
    require(receipt_path.stat().st_size <= 1024**2, 'Development receipt exceeds envelope')
    E.validate_dev_export(plan, receipt['plan_sha256'], receipt, base / 'output')
    for path, label in ((target, 'archive'), (receipt_path, 'receipt')):
        client('s3api', 'put-object', '--bucket', plan['identity']['bucket'],
            '--key', plan['workload']['channel_keys'][label], '--body', str(path), '--if-none-match', '*',
            '--server-side-encryption', 'AES256',
            '--metadata', json.dumps({'sha256': digest(path), 'run': plan['identity']['run_id']}))
    target.unlink()
    return receipt


def run_experiment(worker, plan, base, report, client, gpu, account, work_deadline,
                   acquisition_deadline, budget, hosts, units):
    base = Path(base)
    require(E.is_eight(plan) and not (base / 'final-inputs.json').exists(), 'Final inputs must not precede selection')
    for row in plan['model']['files']:
        worker.download(row, base / 'model', hosts, acquisition_deadline, budget)
    worker.validate_model(base / 'model', plan)
    def run(phase, end=work_deadline, seconds=None):
        started = time.time()
        allocation = plan['phase_seconds'][phase] if seconds is None else seconds
        phase_end = min(end, started + allocation)
        receipt = worker.supervise_phase(plan, base, phase, gpu, account.pw_uid, account.pw_gid,
                                          phase_end, units, seconds=allocation)
        finished = time.time()
        require(finished <= phase_end + 1, 'Native phase exceeded its frozen ceiling')
        receipt.update(started_epoch=started, finished_epoch=finished)
        return receipt
    for phase in ('capacity', 'baseline_dev', 'train'):
        report['stages'].append(run(phase))
        if phase == 'capacity':
            shutil.rmtree(base / 'output/capacity')
            receipt_path = base / 'output/capacity-receipt.json'
            receipt = json.loads(receipt_path.read_text())
            require(not (base / 'output/capacity').exists(), 'Synthetic capacity artifacts remain')
            receipt['discarded'] = True
            save(receipt_path, receipt)
    checkpoint_end = min(work_deadline, time.time() + plan['phase_seconds']['checkpoint_dev'])
    def checkpoint_development():
        receipts = []
        for phase in CHECKPOINT_PHASES:
            receipts.append(run(phase, checkpoint_end, max(1, int(remaining(checkpoint_end)) - 15)))
        return receipts
    checkpoint_receipts = aggregate(base, report, 'checkpoint_dev', checkpoint_development)
    report['stages'][-1]['native_stages'] = checkpoint_receipts
    save(base / 'output/checkpoint_dev-receipt.json', report['stages'][-1])
    checkpoint = make_dev_export(worker, plan, base, report, client, work_deadline)
    report['dev_export_sha256'] = checkpoint['archive']['sha256']
    selection_end = min(work_deadline, time.time() + plan['phase_seconds']['parent_selection'])
    def selection():
        exclusive_intent(base / 'selection-dispatch.json', dict(dev_export_sha256=checkpoint['archive']['sha256']))
        path = await_object(client, plan['identity']['bucket'], plan['workload']['channel_keys']['decision'],
                            base / 'checkpoint-selection.json', selection_end, 1024**2)
        decision = json.loads(path.read_text())
        step = validate_decision(plan, base, decision, checkpoint)
        # Native stages cannot replace the root-owned decision in the root-owned base directory.
        path.chmod(0o444)
        save(base / 'output/checkpoint-selection.json', decision)
        return step
    step = aggregate(base, report, 'parent_selection', selection)
    report.update(selection_sha256=digest(base / 'output/checkpoint-selection.json'), selected_step=step,
                  experiment_outcome='no_eligible_checkpoint' if step is None else 'selected')
    if step is None:
        require(not (base / 'final-inputs.json').exists(), 'No-selection run received final inputs')
        actual = {row['path'] for row in worker.export_inventory(base / 'output', MAX_EXPORT)['files']}
        require(actual == set(E.output_members(plan)) - FINAL_MEMBERS,
                'No-selection output membership differs')
        return
    pin = plan['workload']['final_inputs']
    path = await_object(client, plan['identity']['bucket'], plan['workload']['channel_keys']['final_inputs'],
                        base / 'final-inputs.json', selection_end, pin['bytes'])
    ids = plan['workload']['final_case_ids']['general'] + plan['workload']['final_case_ids']['mois']
    input_rows(path, pin, ids)
    path.chmod(0o444)
    save(base / 'output/final-input-receipt.json', dict(run_id=report['run_id'], plan_sha256=report['plan_sha256'],
        sha256=digest(path), bytes=path.stat().st_size, selection_sha256=report['selection_sha256']))
    for phase in ('final_base', 'final_adapter'):
        report['stages'].append(run(phase))
    # Fail closed on missing members; the shared exporter separately rejects every undeclared file.
    inventory = worker.export_inventory(base / 'output', MAX_EXPORT)
    require({row['path'] for row in inventory['files']} == set(E.output_members(plan)),
            'Successful experimental output membership differs')
