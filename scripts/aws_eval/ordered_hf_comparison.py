"""Frozen, inference-only HF comparison inside one independently bounded lease."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

from grounded_hf_lifecycle import (LIMITS, canonical, digest, exclusive_intent,
                                  object_sha, relative, remaining, require, save,
                                  inference_mode, followup_mode)

SCHEMA = 'trid-ordered-hf-inference-v1'
CHECKPOINT_MAX = 32 * 1024 ** 2
GROUPS = ('nonthinking', 'thinking')
SEEDS = (14, 29)


def is_ordered(plan):
    return plan.get('schema') == SCHEMA


def phase_seconds(plan):
    return plan['workload']['phase_seconds'] if inference_mode(plan) else plan['phase_seconds']


def profiles_for(plan, model_key):
    require(model_key in ('qwen8', 'qwen14') and model_key in plan['workload']['models'], 'Unknown admitted model')
    return [dict(profile, name=profile['name'].replace('qwen8-', model_key + '-', 1))
            for profile in plan['workload']['profiles']]


def phase_kind(plan, phase):
    if not inference_mode(plan):
        return None
    for key in plan['workload']['models']:
        if phase == key + '-select':
            return 'select'
        for cohort in ('dev', 'val'):
            if phase in [key + '-' + cohort + '-' + p['name'].removeprefix(key + '-')
                         for p in profiles_for(plan, key)]:
                return 'inference'
    return None


def profile_phase(model_key, cohort, profile):
    return model_key + '-' + cohort + '-' + profile['name'].removeprefix(model_key + '-')


def shared_inference():
    # Import only pure prompt/decoder helpers, never configure_offline (USE_TORCH=0).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mlx_inference
    return mlx_inference


def pinned_json(base, pin):
    path = Path(base) / 'code' / relative(pin['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes']
            and digest(path) == pin['sha256'], 'Input pin differs')
    return json.loads(path.read_text())


def input_cases(plan, base, cohort):
    workload = plan['workload']
    prefix = 'development' if cohort == 'dev' else 'validation'
    cases = pinned_json(base, workload[prefix + '_inputs'])
    require(isinstance(cases, list) and [c['id'] for c in cases] == workload[prefix + '_case_ids'],
            'Input cohort/order differs')
    require(all(set(c) == {'id', 'messages'} for c in cases), 'Inference inputs must contain no labels')
    return cases


def sampler_kwargs(profile):
    shared_inference().validate_profile(profile)
    require(profile['temperature'] > 0 and profile['min_p'] == 0
            and profile['presence_penalty'] == 0 and profile['repetition_penalty'] == 1,
            'Unsupported frozen sampler')
    return dict(do_sample=True, num_beams=1, temperature=profile['temperature'],
                top_p=profile['top_p'], top_k=profile['top_k'], min_p=None,
                repetition_penalty=1.0, max_new_tokens=profile['max_new_tokens'],
                use_cache=True)


def reset_case_rng(torch, profile):
    torch.manual_seed(profile['seed'])
    torch.cuda.manual_seed_all(profile['seed'])


def check_question_deadline(path, now=None):
    if not Path(path).exists():
        return
    receipt = json.loads(Path(path).read_text())
    end = receipt.get('question_deadline_epoch')
    require(end is None or (time.time() if now is None else now) < end,
            'External 180-second question deadline exceeded')


def atomic_receipt(path, value):
    # The external root guard must never read a half-written deadline receipt.
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        stream.write(canonical(value).decode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def memory(torch):
    return dict(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def generation_stage(worker, plan, base, phase, model_key, cohort, profile):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    base = Path(base)
    inference = shared_inference()
    inference.validate_profile(profile)
    torch = worker.native_runtime(plan, base)
    model_dir = base / 'model' / model_key
    manifest = plan['workload']['models'][model_key]
    worker.validate_model(model_dir, dict(plan, model=manifest))
    cases = input_cases(plan, base, cohort)
    receipt_path = base / 'output' / (phase + '-receipt.json')
    target = base / 'output' / (phase + '-outputs.jsonl')
    receipt = dict(state='incomplete', phase=phase, model_key=model_key, run_id=plan['identity']['run_id'],
                   plan_sha256=digest(base / 'plan.json'), model_sha256=object_sha(manifest),
                   profile=profile, input_sha256=plan['workload'][
                       ('development' if cohort == 'dev' else 'validation') + '_inputs']['sha256'],
                   completed_case_ids=[], question_deadline_epoch=None)
    atomic_receipt(receipt_path, receipt)
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                    local_files_only=True, trust_remote_code=False).to('cuda')
        model.eval()
        require(model.config.max_position_embeddings >= profile['context_tokens'], 'Native context too small')
        stops = model.generation_config.eos_token_id
        require(type(stops) is int or (isinstance(stops, list) and stops
                and all(type(x) is int for x in stops)), 'Native EOS declaration missing')
        with target.open('x') as stream:
            for case in cases:
                started = time.monotonic()
                receipt['question_deadline_epoch'] = time.time() + 180
                generated = []
                receipt['partial'] = dict(id=case['id'], profile=profile['name'], output_token_ids=generated)
                atomic_receipt(receipt_path, receipt)
                def timeout(signum, frame):
                    raise TimeoutError('180-second question deadline exceeded')
                old = signal.signal(signal.SIGALRM, timeout)
                signal.setitimer(signal.ITIMER_REAL, 180)
                try:
                    text, ids = inference.render_prompt(tok, case['messages'], profile)
                    receipt['partial'].update(rendered_prompt=text, prompt_token_ids=ids)
                    atomic_receipt(receipt_path, receipt)
                    reset_case_rng(torch, profile)
                    torch.cuda.reset_peak_memory_stats()
                    inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
                    class Capture:
                        first = True
                        def put(self, tokens):
                            if self.first:
                                self.first = False
                                return
                            generated.extend(tokens.reshape(-1).tolist())
                            check_question_deadline(receipt_path)
                            atomic_receipt(receipt_path, receipt)
                        def end(self):
                            pass
                    # Disable model-default merging explicitly: a fresh config alone
                    # does not isolate suppression, forced EOS, or sampler penalties.
                    config = GenerationConfig(**sampler_kwargs(profile), eos_token_id=stops,
                        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                        bos_token_id=tok.bos_token_id)
                    with torch.inference_mode():
                        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                                generation_config=config, use_model_defaults=False,
                                                streamer=Capture())
                    torch.cuda.synchronize()
                    require(generated == result[0, len(ids):].tolist(), 'Streamed native tokens differ')
                    decoded = inference.decode_generation(tok, generated, stops, profile['max_new_tokens'], profile['thinking'])
                    usage = memory(torch)
                    require(usage['peak_cuda_allocated_bytes'] <= LIMITS['allocator_bytes'], 'CUDA allocator cap exceeded')
                    row = dict(id=case['id'], profile=profile['name'], profile_parameters=profile,
                        native_reference=False, penalty_scope='generated_only',
                        prompt_sha256=object_sha(case['messages']), rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        rendered_prompt=text, prompt_token_ids=ids, output_token_ids=generated, **decoded,
                        elapsed_seconds=time.monotonic() - started, memory=usage,
                        backend='transformers', representation='native_bfloat16', prefill='full_prompt',
                        effective_prefill_tokens=len(ids), model_key=model_key, model_sha256=object_sha(manifest),
                        model_id=manifest['id'], revision=manifest['revision'],
                        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     packages=plan['runtime']['packages']), fresh_cache=True, rng_reset_per_case=True)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    receipt['completed_case_ids'].append(case['id'])
                    receipt.pop('partial')
                    receipt['question_deadline_epoch'] = None
                    atomic_receipt(receipt_path, receipt)
                    del result, inputs
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
        receipt.update(state='complete', output_sha256=digest(target), output_bytes=target.stat().st_size)
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['question_deadline_epoch'] = None
        atomic_receipt(receipt_path, receipt)


def exact_typed(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_typed(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_typed(a, b) for a, b in zip(left, right))
    return left == right


def joint_correct(row, reference):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Non-JSON constant')
    try:
        answer = json.loads(row['response_text'], object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError):
        return False
    return (isinstance(answer, dict) and exact_typed(answer, reference)
            and row.get('termination_reason') == 'eos' and type(row.get('terminal_stop_token_id')) is int
            and row.get('framing_status') == 'complete' and row.get('reasoning_complete') is True)


def selection_stage(plan, base, model_key):
    require(os.geteuid() != 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not any(k.startswith('AWS_') and k != 'AWS_EC2_METADATA_DISABLED' for k in os.environ),
            'Selector must be nonroot, model-free, CPU-only, credential-free')
    base = Path(base)
    work = plan['workload']
    cases = input_cases(plan, base, 'dev')
    references = pinned_json(base, work['development_references'])['cases']
    require([c['id'] for c in references] == work['development_case_ids']
            and all(c['messages'] == r['messages'] for c, r in zip(cases, references)), 'Development labels differ')
    groups, outputs = [], []
    for group in GROUPS:
        counts, elapsed, peak, names = [], [], 0, []
        for profile in profiles_for(plan, model_key):
            if profile['thinking'] != (group == 'thinking'):
                continue
            phase = profile_phase(model_key, 'dev', profile)
            path = base / 'output' / (phase + '-outputs.jsonl')
            receipt = json.loads((base / 'output' / (phase + '-receipt.json')).read_text())
            require(receipt['state'] == 'complete' and receipt['completed_case_ids'] == work['development_case_ids']
                    and receipt['profile'] == profile and receipt['plan_sha256'] == digest(base / 'plan.json')
                    and receipt['model_sha256'] == object_sha(work['models'][model_key])
                    and receipt['input_sha256'] == work['development_inputs']['sha256']
                    and receipt['output_sha256'] == digest(path) and receipt['output_bytes'] == path.stat().st_size,
                    'Failed/incomplete development phase cannot be scored')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            require([r['id'] for r in rows] == work['development_case_ids']
                    and all(r['profile_parameters'] == profile
                            and r['prompt_sha256'] == object_sha(c['messages'])
                            and r['model_sha256'] == object_sha(work['models'][model_key])
                            for r, c in zip(rows, cases)), 'Development output identity differs')
            counts.append(sum(joint_correct(row, ref['reference']) for row, ref in zip(rows, references)))
            for row in rows:
                seconds = row['elapsed_seconds']
                allocated = row['memory']['peak_cuda_allocated_bytes']
                require(type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0
                        and type(allocated) is int and allocated >= 0, 'Invalid development measurements')
                elapsed.append(seconds)
                peak = max(peak, allocated)
            names.append(profile['name'])
            outputs.append(dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path)))
        require(len(counts) == 2 and {p['seed'] for p in profiles_for(plan, model_key)
                if p['thinking'] == (group == 'thinking')} == set(SEEDS), 'Both frozen seeds required')
        groups.append(dict(group=group, minimum_correct=min(counts), combined_correct=sum(counts),
                           mean_elapsed_seconds=sum(elapsed) / len(elapsed), peak_cuda_allocated_bytes=peak,
                           profiles=names))
    groups.sort(key=selection_rank)
    selected = groups[0]
    output_by_path = {row['path']: row for row in outputs}
    outputs = [output_by_path[profile_phase(model_key, 'dev', profile) + '-outputs.jsonl']
               for profile in profiles_for(plan, model_key)]
    receipt = dict(run_id=plan['identity']['run_id'], plan_sha256=digest(base / 'plan.json'),
        model_key=model_key, model_sha256=object_sha(work['models'][model_key]),
        profiles_sha256=object_sha(profiles_for(plan, model_key)),
        development_inputs_sha256=work['development_inputs']['sha256'],
        development_references_sha256=work['development_references']['sha256'],
        selection_policy=work['selection_policy'], state='complete', selected_group=selected['group'],
        selected_profiles=selected['profiles'], groups=groups, dev_outputs=outputs)
    save(base / 'output' / (model_key + '-selection.json'), receipt)


def selection_rank(group):
    return (-group['minimum_correct'], -group['combined_correct'], group['mean_elapsed_seconds'],
            group['peak_cuda_allocated_bytes'], group['group'])


def child_stage(worker, plan, base, phase):
    kind = phase_kind(plan, phase)
    require(kind is not None, 'Inference schema forbids training/capacity/unknown dispatch')
    model_key = phase.split('-')[0]
    if kind == 'select':
        selection_stage(plan, base, model_key)
        return
    cohort = phase.split('-')[1]
    profile = next(p for p in profiles_for(plan, model_key) if profile_phase(model_key, cohort, p) == phase)
    generation_stage(worker, plan, base, phase, model_key, cohort, profile)


def aggregate(base, report, phase, action):
    receipt = dict(phase=phase, state='incomplete', started_epoch=time.time(),
                   run_id=report['run_id'], plan_sha256=report['plan_sha256'])
    path = Path(base) / 'output' / (phase + '-receipt.json')
    save(path, receipt)
    report['stages'].append(receipt)
    try:
        value = action()
        receipt['state'] = 'complete'
        return value
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['finished_epoch'] = time.time()
        save(path, receipt)


def make_checkpoint(worker, plan, base, report, client, deadline):
    from grounded_hf_lifecycle import ordered_output_members, validate_checkpoint_receipt
    require(is_ordered(plan), 'Followup has no checkpoint')
    base = Path(base)
    selection = json.loads((base / 'output/qwen8-selection.json').read_text())
    members = ordered_output_members(plan, model_key='qwen8', selected_group=selection['selected_group'])
    inventory = worker.export_inventory(base / 'output', CHECKPOINT_MAX)
    require({r['path'] for r in inventory['files']} == set(members), 'Checkpoint membership differs')
    predicted = 10240 + sum(512 + ((r['bytes'] + 511) // 512) * 512 for r in inventory['files'])
    require(predicted <= CHECKPOINT_MAX, 'Checkpoint tar overhead exceeds bound')
    target = base / 'qwen8-checkpoint.tar'
    with tarfile.open(target, 'x:', format=tarfile.USTAR_FORMAT) as archive:
        for row in inventory['files']:
            remaining(deadline)
            info = tarfile.TarInfo(row['path']); info.size = row['bytes']; info.mode = 0o444
            with (base / 'output' / row['path']).open('rb') as stream:
                archive.addfile(info, stream)
    require(target.stat().st_size <= CHECKPOINT_MAX, 'Checkpoint archive exceeds 32MiB')
    work, ident = plan['workload'], plan['identity']
    receipt = dict(run_id=ident['run_id'], plan_sha256=digest(base / 'plan.json'),
        instance_id=report['instance_id'], model_sha256=object_sha(work['models']['qwen8']),
        profiles_sha256=object_sha(profiles_for(plan, 'qwen8')),
        development_inputs_sha256=work['development_inputs']['sha256'],
        validation_inputs_sha256=work['validation_inputs']['sha256'], phase='val8',
        selected_group=selection['selected_group'], export=dict(bytes=target.stat().st_size, sha256=digest(target)),
        files=inventory['files'])
    validate_checkpoint_receipt(receipt, plan, receipt['plan_sha256'], report['instance_id'])
    receipt_path = base / 'qwen8-checkpoint.json'
    save(receipt_path, receipt)
    require(receipt_path.stat().st_size <= 1024 ** 2, 'Checkpoint receipt overflow')
    for path, key in ((target, work['checkpoint_keys']['archive']), (receipt_path, work['checkpoint_keys']['receipt'])):
        client('s3api', 'put-object', '--bucket', ident['bucket'], '--key', key, '--body', str(path),
               '--if-none-match', '*', '--server-side-encryption', 'AES256',
               '--metadata', json.dumps({'sha256': digest(path), 'run': ident['run_id']}))
    return receipt


def await_decision(plan, base, client, checkpoint, deadline):
    from grounded_hf_lifecycle import validate_parent_decision
    require(is_ordered(plan), 'Followup has no parent decision')
    base = Path(base)
    target = base / 'parent-decision-download.json'
    bucket, key = plan['identity']['bucket'], plan['workload']['checkpoint_keys']['decision']
    exclusive_intent(base / 'parent-decision-dispatch.json', dict(checkpoint_sha256=checkpoint['export']['sha256']))
    old_deadline = client.deadline
    client.deadline = min(old_deadline, deadline)
    try:
        while True:
            remaining(deadline)
            # A missing GET may be AccessDenied with prefix-scoped ListBucket.
            # Only an exact, complete empty LIST proves absence; no AWS error
            # (including a failed GET after a successful LIST) is retried here.
            # The frozen 300s/5s wait now costs <=60 LISTs plus <=1 GET.
            listing = client('s3api', 'list-objects-v2', '--bucket', bucket,
                             '--prefix', key, '--max-keys', '2')
            contents = listing.get('Contents', [])
            require(listing.get('Name') == bucket and listing.get('Prefix') == key
                    and listing.get('IsTruncated') is False and not listing.get('NextContinuationToken')
                    and not listing.get('CommonPrefixes') and listing.get('MaxKeys') == 2
                    and isinstance(contents, list) and len(contents) <= 1
                    and type(listing.get('KeyCount')) is int and listing['KeyCount'] == len(contents),
                    'Parent decision listing is not exact and complete')
            if not contents:
                time.sleep(min(5, remaining(deadline)))
                continue
            row = contents[0]
            require(isinstance(row, dict) and row.get('Key') == key
                    and type(row.get('Size')) is int and 0 < row['Size'] <= 1024 ** 2,
                    'Parent decision listing key or size differs')
            remaining(deadline)
            client('s3api', 'get-object', '--bucket', bucket, '--key', key,
                   '--range', 'bytes=0-1048576', str(target))
            remaining(deadline)
            require(target.stat().st_size <= 1024 ** 2, 'Parent decision overflow')
            decision = json.loads(target.read_text())
            validate_parent_decision(decision, plan, digest(base / 'plan.json'), checkpoint['export']['sha256'])
            save(base / 'output/parent-decision.json', decision)
            return decision
    finally:
        client.deadline = old_deadline


def run_ordered(worker, plan, base, report, client, gpu, account, work_deadline,
                acquisition_deadline, budget, hosts, units):
    base = Path(base)
    require(inference_mode(plan), 'Native comparison requires an inference plan')
    def run_model(model_key):
        suffix = '8' if model_key == 'qwen8' else '14'
        phase = 'acquire' + suffix
        initial = followup_mode(plan) or model_key == 'qwen8'
        end = acquisition_deadline if initial else min(work_deadline, time.time() + phase_seconds(plan)[phase])
        def acquire():
            exclusive_intent(base / (phase + '-dispatch.json'), dict(model_sha256=object_sha(plan['workload']['models'][model_key])))
            directory = base / 'model' / model_key
            directory.mkdir(mode=0o755)
            for row in plan['workload']['models'][model_key]['files']:
                worker.download(row, directory, hosts, end, budget)
            worker.validate_model(directory, dict(plan, model=plan['workload']['models'][model_key]))
            # Bootstrap's restrictive umask also applies to this later-created
            # root-owned directory. Expose traversal only after authentication.
            directory.chmod(0o755)
        aggregate(base, report, phase, acquire)
        for cohort in ('dev', 'val'):
            phase = cohort + suffix
            end = min(work_deadline, time.time() + phase_seconds(plan)[phase])
            def cohort_work():
                candidates = profiles_for(plan, model_key)
                if cohort == 'val':
                    selection = json.loads((base / 'output' / (model_key + '-selection.json')).read_text())
                    require(selection['state'] == 'complete', 'Incomplete development selection')
                    candidates = [p for p in candidates if p['name'] in selection['selected_profiles']]
                    require(len(candidates) == 2, 'Validation requires selected seed pair')
                for profile in candidates:
                    worker.supervise_phase(plan, base, profile_phase(model_key, cohort, profile), gpu,
                        account.pw_uid, account.pw_gid, end, units, seconds=int(remaining(end)))
                if cohort == 'dev':
                    worker.supervise_phase(plan, base, model_key + '-select', gpu,
                        account.pw_uid, account.pw_gid, end, units, seconds=min(60, int(remaining(end))))
            aggregate(base, report, phase, cohort_work)
    if followup_mode(plan):
        run_model('qwen14')
        return
    run_model('qwen8')
    checkpoint = make_checkpoint(worker, plan, base, report, client, work_deadline)
    report['checkpoint'] = checkpoint
    end = min(work_deadline, time.time() + phase_seconds(plan)['parent_decision'])
    decision = aggregate(base, report, 'parent_decision', lambda: await_decision(plan, base, client, checkpoint, end))
    report['decision'] = decision
    if decision['decision'] == 'compare14':
        run_model('qwen14')
