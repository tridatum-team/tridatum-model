"""One fresh, explicitly sealed HF experiment. No implicit pins or launch retries.

`validate` is offline. `preflight` is read-only AWS admission. `launch` consumes
approval before preflight and dispatches once; `reconcile` can only terminate.
The old CPU/CUDA identities and approval paths are deliberately untouched.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time

if __package__:
    from .qwen_cuda_smoke import (require, digest, save, exclusive_intent, read_bound,
                                 validate_ami_mappings, gpu_minor_from_xml)
else:
    from qwen_cuda_smoke import (require, digest, save, exclusive_intent, read_bound,
                                validate_ami_mappings, gpu_minor_from_xml)

GIB = 1024 ** 3
LIMITS = dict(instance_count=1, lifetime_seconds=14400, reserve_seconds=1800,
              root_gib=120, rss_bytes=48 * GIB, allocator_bytes=40 * GIB,
              gpu_bytes=42 * GIB, output_bytes=2 * GIB, cpu_cores=6, swap_bytes=0)
# Completion/readiness receipts share, rather than enlarge, the output envelope.
MAX_EXPORT = LIMITS['output_bytes'] - 8 * 1024**2
RECIPE = dict(rank=16, alpha=32, dropout=0.0, lr=2e-5, batch_size=2,
              micro_batch_size=1, context_tokens=8192, answer_tokens=768, seed=14)
SOURCES = {'scripts/aws_eval/grounded_hf_lifecycle.py',
           'scripts/aws_eval/grounded_hf_worker.py',
           'scripts/aws_eval/grounded_hf_bootstrap.sh',
           'scripts/aws_eval/qwen_cpu_smoke.py', 'scripts/aws_eval/qwen_cuda_smoke.py',
           'scripts/train_open_weight_lora.py', 'trid/__init__.py', 'trid/prompts.py',
           'trid/text.py', 'trid/case_prompts.py'}
SHA = re.compile('[0-9a-f]{64}')
INFERENCE_SCHEMA = 'trid-ordered-hf-inference-v1'
CHECKPOINT_MAX = 32 * 1024**2
CHECKPOINT_CHANNEL_BYTES = CHECKPOINT_MAX + 3 * 1024**2
ORDERED_PHASES = dict(acquire8=900, dev8=2400, val8=2400, parent_decision=300,
                      acquire14=900, dev14=2400, val14=2400)
ORDERED_MODELS = {
    'qwen8': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
    'qwen14': ('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18')}


def inference_mode(plan):
    return plan.get('schema') == INFERENCE_SCHEMA


def success_field(plan):
    return 'inference_success' if inference_mode(plan) else 'training_success'


def ordered_sources():
    return (SOURCES - {'scripts/train_open_weight_lora.py'}) | {
        'scripts/mlx_inference.py', 'scripts/aws_eval/ordered_hf_comparison.py'}


def ordered_output_members(plan, model_key=None, selected_group=None):
    """All candidates for admission; exact selected membership for checkpoints."""
    require(selected_group in (None, 'nonthinking', 'thinking'), 'Unknown selected group')
    members = {'runtime.json', 'runtime.log'}
    for key in ([model_key] if model_key else ORDERED_MODELS):
        require(key in ORDERED_MODELS, 'Unknown ordered model')
        suffix = '8' if key == 'qwen8' else '14'
        members.update({f'acquire{suffix}-receipt.json', f'dev{suffix}-receipt.json',
                        f'val{suffix}-receipt.json', f'{key}-select.log', f'{key}-selection.json'})
        for cohort in ('dev', 'val'):
            groups = [selected_group] if cohort == 'val' and selected_group else ('nonthinking', 'thinking')
            for group in groups:
                for seed in (14, 29):
                    stem = f'{key}-{cohort}-{group}-seed{seed}'
                    members.update({stem + '.log', stem + '-outputs.jsonl', stem + '-receipt.json'})
                    if model_key is None and selected_group is None:
                        members.add(stem + '-receipt.json.pending')
    if model_key is None:
        members.update({'parent-decision.json', 'parent_decision-receipt.json'})
    return members


def validate_ordered_workload(plan, bundled):
    workload = plan['workload']
    require(set(workload) == {'models', 'profiles', 'development_inputs', 'validation_inputs',
                'development_references', 'development_case_ids', 'validation_case_ids',
                'selection_policy', 'checkpoint_keys', 'phase_seconds', 'source_closure',
                'backend', 'dtype', 'prefill', 'question_timeout_seconds'},
            'Ordered workload requires exact inference-only fields')
    require(workload['backend'] == 'transformers' and workload['dtype'] == 'bfloat16'
            and workload['prefill'] == 'full_prompt' and workload['question_timeout_seconds'] == 180
            and all(m['dtype'] == 'bfloat16' for m in workload['models'].values()),
            'Native BF16 full-prompt inference contract differs')
    expected_profiles = []
    for thinking in (False, True):
        for seed in (14, 29):
            expected_profiles.append(dict(name=f'qwen8-{"thinking" if thinking else "nonthinking"}-seed{seed}',
                thinking=thinking, temperature=0.6 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8, top_k=20, min_p=0.0, presence_penalty=0.0,
                repetition_penalty=1.0, penalty_context_size=0, seed=seed,
                max_new_tokens=4096 if thinking else 1024, context_tokens=8192, prefill_step_size=128))
    require(workload['profiles'] == expected_profiles, 'Four frozen original paired profiles required')
    require(workload['selection_policy'] == dict(metric='joint_correct', require_both_seeds=True,
                tie_breaks=['minimum_correct', 'combined_correct', 'mean_elapsed_seconds',
                            'peak_cuda_allocated_bytes', 'lexical_group'],
                validation_references_in_bundle=False), 'Frozen development-only selection policy differs')
    phases = workload['phase_seconds']
    require(phases == ORDERED_PHASES and all(type(v) is int for v in phases.values())
            and sum(phases.values()) <= LIMITS['lifetime_seconds'] - LIMITS['reserve_seconds'] - 600,
            'Ordered shared lease allocations differ')
    ident = plan['identity']
    require(workload['checkpoint_keys'] == dict(archive=ident['output_prefix'] + 'qwen8-checkpoint.tar',
                receipt=ident['output_prefix'] + 'qwen8-checkpoint.json',
                decision=ident['code_prefix'] + 'parent-decision.json'),
            'Checkpoint/decision keys must bind exact run prefixes')
    data_paths = []
    for key in ('development_inputs', 'validation_inputs', 'development_references', 'source_closure'):
        pin = workload[key]
        pins([pin], 32 * 1024**2)
        require(set(pin) == {'path', 'bytes', 'sha256'} and pin['path'] in bundled
                and pin == {k: bundled[pin['path']][k] for k in pin},
                'Missing sealed ordered input/source pin')
        data_paths.append(pin['path'])
    require(len(set(data_paths)) == len(data_paths), 'Input/reference/closure paths must be separate')
    dev, valid = workload['development_case_ids'], workload['validation_case_ids']
    require(isinstance(dev, list) and isinstance(valid, list) and len(dev) == 6 and len(valid) == 16
            and all(isinstance(v, str) and v for v in dev + valid)
            and len(set(dev + valid)) == 22, 'Disjoint six-development/sixteen-validation IDs required')
    require(not any('reference' in n.lower() and n != workload['development_references']['path']
                    for n in bundled), 'Only development references may enter the bundle')


def validate_ordered_data(root, plan):
    root = Path(root)
    workload = plan['workload']
    validate_ordered_workload(plan, {r['path']: r for r in plan['bundle']['files']})
    loaded = {}
    for key in ('development_inputs', 'validation_inputs', 'development_references', 'source_closure'):
        pin = workload[key]
        path = root / pin['path']
        require(not path.is_symlink() and path.stat().st_size == pin['bytes'], 'Ordered data envelope differs')
        loaded[key] = read_bound(path, pin['sha256'])
    for cohort in ('development', 'validation'):
        rows = loaded[cohort + '_inputs']
        require(isinstance(rows, list) and [r['id'] for r in rows] == workload[cohort + '_case_ids'],
                'Ordered cohort input membership differs')
        for row in rows:
            blind_messages(row)
    references = loaded['development_references']
    require(isinstance(references, dict) and isinstance(references.get('cases'), list)
            and all(isinstance(row, dict) and {'id', 'messages', 'reference'} <= set(row)
                    for row in references['cases'])
            and [row['id'] for row in references['cases']] == workload['development_case_ids'],
            'Development references require registered IDs, messages, and reference fields')
    require(all(reference['messages'] == case['messages']
                for reference, case in zip(references['cases'], loaded['development_inputs'])),
            'Development reference messages differ from sealed inference inputs')
    closure = dict(schema='trid-ordered-source-closure-v1', run_id=plan['identity']['run_id'],
        development_inputs_sha256=workload['development_inputs']['sha256'],
        validation_inputs_sha256=workload['validation_inputs']['sha256'],
        development_references_sha256=workload['development_references']['sha256'],
        development_case_ids=workload['development_case_ids'], validation_case_ids=workload['validation_case_ids'],
        validation_references_in_bundle=False, source_verified=True, protected_sets_excluded=True)
    observed = loaded['source_closure']
    require(set(observed) == set(closure) | {'provenance', 'scorer_sha256'}
            and all(observed[key] == value for key, value in closure.items())
            and observed['scorer_sha256'] == '0c25aaeee6075d49f90b06c06f2f4389c698c7a543ab6500e72c5a262c8c3e8a',
            'Frozen ordered source closure differs')
    provenance = observed['provenance']
    require(isinstance(provenance, dict) and set(provenance) == {
                'comparison_protocol', 'case_review', 'reference_validation', 'scope', 'scorer'},
            'Independent ordered provenance pins are missing')
    pins(list(provenance.values()), 32 * 1024**2)
    bundled = {row['path'] for row in plan['bundle']['files']}
    require(all(set(pin) == {'path', 'bytes', 'sha256'} and pin['path'] not in bundled
                for pin in provenance.values())
            and provenance['scorer']['sha256'] == observed['scorer_sha256'],
            'Private review/reference provenance must remain local, hash-only in cloud')
    return {key: loaded[key] for key in ('development_inputs', 'validation_inputs')}


def validate_parent_decision(decision, plan, plan_sha, checkpoint_sha):
    require(isinstance(decision, dict) and set(decision) == {
                'run_id', 'plan_sha256', 'checkpoint_sha256', 'decision', 'reason'}
            and decision['run_id'] == plan['identity']['run_id'] and decision['plan_sha256'] == plan_sha
            and decision['checkpoint_sha256'] == checkpoint_sha and SHA.fullmatch(checkpoint_sha)
            and decision['decision'] in ('complete', 'compare14')
            and isinstance(decision['reason'], str) and 0 < len(decision['reason'].strip()) <= 4096,
            'Parent decision is missing, malformed, or not bound to this checkpoint')
    return decision


def validate_checkpoint_receipt(receipt, plan, plan_sha, instance_id):
    workload = plan['workload']
    require(inference_mode(plan) and set(receipt) == {'run_id', 'plan_sha256', 'instance_id',
                'model_sha256', 'profiles_sha256', 'development_inputs_sha256',
                'validation_inputs_sha256', 'phase', 'export', 'files', 'selected_group'}
            and receipt['run_id'] == plan['identity']['run_id'] and receipt['plan_sha256'] == plan_sha
            and receipt['instance_id'] == instance_id and receipt['phase'] == 'val8'
            and receipt['model_sha256'] == object_sha(workload['models']['qwen8'])
            and receipt['profiles_sha256'] == object_sha(workload['profiles'])
            and receipt['development_inputs_sha256'] == workload['development_inputs']['sha256']
            and receipt['validation_inputs_sha256'] == workload['validation_inputs']['sha256']
            and receipt['selected_group'] in ('nonthinking', 'thinking'),
            'Checkpoint ownership/model/profile/input binding differs')
    pin = receipt['export']
    require(set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int
            and 0 < pin['bytes'] <= CHECKPOINT_MAX and SHA.fullmatch(pin['sha256']),
            'Checkpoint archive exceeds finite envelope')
    members = pins(receipt['files'], CHECKPOINT_MAX)
    require(set(members) == ordered_output_members(plan, 'qwen8', receipt['selected_group']),
            'Checkpoint requires exact complete selected 8B members')
    return pin

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Invalid artifact path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == name
            and name != '.', 'Noncanonical artifact path')
    return name


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timestamp requires timezone')
    return parsed.timestamp()


def remaining(deadline, now=None):
    seconds = deadline - (time.time() if now is None else now)
    require(seconds > 0, 'Absolute deadline exhausted')
    return seconds


def pins(rows, cap):
    require(isinstance(rows, list) and 0 < len(rows) <= 4096, 'Empty/excess artifact inventory')
    result = {}
    for row in rows:
        name = relative(row['path'])
        require(name not in result and SHA.fullmatch(row['sha256'])
                and type(row['bytes']) is int and 0 <= row['bytes'] <= cap,
                'Duplicate/invalid artifact pin')
        result[name] = row
    require(sum(row['bytes'] for row in rows) <= cap, 'Artifact envelope overflow')
    return result


def authenticate_archive(path, bundle, destination=None):
    """Authenticate every byte before extracting anything; never use extractall."""
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_size == bundle['bytes'] and digest(path) == bundle['sha256'],
            'Bundle authentication failed')
    expected = pins(bundle['files'], bundle['unpacked_bytes'])
    with tarfile.open(path, 'r:') as archive:
        members = []
        seen = set()
        for member in archive:
            name = relative(member.name)
            require(member.isfile() and not member.pax_headers and not member.sparse
                    and name not in seen and name in expected and member.size == expected[name]['bytes'],
                    'Unsafe/unapproved archive member')
            seen.add(name)
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == expected[name]['sha256'], 'Bundle member hash mismatch')
            members.append(member)
        require(seen == set(expected), 'Missing bundle member')
        if destination is not None:
            root = Path(destination)
            root.mkdir(mode=0o755)  # fresh destination, including on a repeated bootstrap
            for member in members:
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as sink:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        sink.write(block)
                target.chmod(0o444)
    return expected


def cost_bound(plan):
    """Conservative bill model, not a claim that AWS exposes a spending hard cap."""
    cost = plan['cost']
    prices = cost['rates']
    require(set(prices) == {'ec2_hour', 'ipv4_hour', 'ebs_gb_month', 's3_gb_month',
                            'egress_gb', 's3_get', 's3_put', 'scheduler_invocation'},
            'Missing GET/PUT/storage/egress/scheduler price closure')
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in prices.values()),
            'Prices require finite observed paid rates; no free-tier assumption')
    cap = 20 if inference_mode(plan) else 50
    require(cost['cap_usd'] == cap and 0 < cost['retention_days'] <= 30
            and type(cost['retention_days']) is int
            and cost['ancillary_bound_usd'] >= 0
            and math.isfinite(cost['ancillary_bound_usd']), 'Invalid all-in cost registration')
    for key in ('get_requests', 'put_requests', 'control_requests', 'download_requests',
                'network_egress_bytes', 'retained_bytes', 'termination_grace_seconds'):
        require(type(cost[key]) is int and cost[key] > 0, 'Missing finite cost envelope: ' + key)
    require(cost['get_requests'] >= 512 and cost['put_requests'] >= 16
            and cost['control_requests'] >= 1000 and cost['termination_grace_seconds'] >= 300,
            'Request/termination reserve does not cover bounded controller')
    extra = CHECKPOINT_CHANNEL_BYTES if inference_mode(plan) else 0
    if inference_mode(plan):
        require(cost['get_requests'] >= 579 and cost['put_requests'] >= 20
                and cost['control_requests'] >= 1067 and cost['ancillary_bound_usd'] >= 5,
                'Ordered checkpoint channel and ancillary reserve are not priced')
    required_storage = plan['bundle']['bytes'] + LIMITS['output_bytes'] + 8 * 1024**2 + extra
    require(type(plan['network']['egress_byte_cap']) is int
            and plan['network']['egress_byte_cap'] >= LIMITS['output_bytes'] + extra
            and cost['retained_bytes'] >= required_storage
            and cost['network_egress_bytes'] >= plan['network']['egress_byte_cap'] + LIMITS['output_bytes'] + extra,
            'Storage/host-egress plus independent S3 export-download allowance insufficient')
    hours = (LIMITS['lifetime_seconds'] + cost['termination_grace_seconds']) / 3600
    total = hours * (prices['ec2_hour'] + prices['ipv4_hour'])
    total += 120 * prices['ebs_gb_month'] * hours / (28 * 24)
    # Expiration eligibility rounds up to the next UTC midnight. S3 stops billing
    # expired objects at eligibility, not eventual asynchronous physical removal.
    total += cost['retained_bytes'] / 10**9 * prices['s3_gb_month'] * (cost['retention_days'] + 1) / 28
    total += cost['network_egress_bytes'] / 10**9 * prices['egress_gb']
    total += cost['get_requests'] * prices['s3_get'] + cost['put_requests'] * prices['s3_put']
    total += 2 * prices['scheduler_invocation'] + cost['ancillary_bound_usd']
    require(math.isfinite(total) and total <= cap, f'All-in conservative bound exceeds USD{cap}')
    return dict(additional_pretax_bound_usd=total, invoice=False, spending_hard_cap=False)


def ebs_key_arn(plan):
    identity = plan['identity']
    prefix = f'arn:aws:kms:{identity["region"]}:{identity["account"]}:key/'
    arn = plan.get('ebs_kms_key_arn')
    require(isinstance(arn, str) and arn.startswith(prefix)
            and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', arn[len(prefix):]),
            'Explicit same-account/region EBS key ARN required; no alias or default')
    return arn


def validate_gpu_memory_contract(runtime):
    fields = ('gpu_catalog_mib', 'gpu_nvml_min_mib', 'gpu_nvml_max_mib')
    require(all(type(runtime[field]) is int for field in fields)
            and tuple(runtime[field] for field in fields) == (45776, 45776, 49152),
            'Approved catalog memory and physical NVML interval differ')


def normalized_gpu_devices(devices):
    require(isinstance(devices, list) and all(isinstance(device, dict) for device in devices),
            'EC2 GPU catalog devices must be objects')
    normalized = []
    for device in devices:
        copied = dict(device)
        if 'Workloads' in device:
            workloads = device['Workloads']
            require(isinstance(workloads, list) and all(isinstance(value, str) for value in workloads),
                    'EC2 GPU Workloads must be an enum list')
            copied['Workloads'] = sorted(workloads)
        normalized.append(copied)
    return normalized


def verify_gpu_catalog(plan, observation):
    validate_gpu_memory_contract(plan['runtime'])
    devices = observation['Gpus']
    require(normalized_gpu_devices(devices) == normalized_gpu_devices(plan['gpu_type_observation'])
            and len(devices) == 1
            and devices[0]['Count'] == 1 and devices[0]['Name'] == 'L40S'
            and devices[0]['Manufacturer'] == 'NVIDIA'
            and devices[0]['MemoryInfo']['SizeInMiB'] == plan['runtime']['gpu_catalog_mib']
            and observation['TotalGpuMemoryInMiB'] == plan['runtime']['gpu_catalog_mib'],
            'Exact EC2 GPU catalog observation changed')


def verify_gpu_observation(observation, plan, expected=None):
    runtime = plan['runtime']
    validate_gpu_memory_contract(runtime)
    require(set(observation) == {'uuid', 'minor', 'name', 'total_mib', 'driver'}
            and observation['name'] == runtime['gpu_name'] and observation['driver'] == runtime['driver_version']
            and type(observation['total_mib']) is int
            and runtime['gpu_nvml_min_mib'] <= observation['total_mib'] <= runtime['gpu_nvml_max_mib']
            and re.fullmatch(r'GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}', observation['uuid'])
            and type(observation['minor']) is int and 0 <= observation['minor'] < 2**20,
            'Physical NVML GPU identity/capacity is outside the approved contract')
    if expected is not None:
        require(observation == expected, 'Released NVML GPU observation changed')


def validate_worker_policy(plan):
    identity = plan['identity']
    expected = [dict(Effect='Allow', Action=['s3:GetObject'], Resource=[
        f"arn:aws:s3:::{identity['bucket']}/{identity['code_prefix']}*"]),
        dict(Effect='Allow', Action=['s3:PutObject'], Resource=[
        f"arn:aws:s3:::{identity['bucket']}/{identity['output_prefix']}*"])]
    if inference_mode(plan):
        expected.append(dict(Effect='Allow', Action=['s3:ListBucket'],
            Resource=[f"arn:aws:s3:::{identity['bucket']}"],
            Condition={'StringEquals': {'s3:prefix': identity['code_prefix'] + 'parent-decision.json'}}))
    statements = [s for p in plan['iam']['instance_role']['inline'].values() for s in p['Statement']]
    require(statements == expected,
            'Worker role must have only exact-run read/write and, for ordered inference, exact-decision listing scope')


def validate_plan(plan, approval, plan_sha, *, now=None):
    require(plan['schema'] in ('trid-grounded-hf-v1', INFERENCE_SCHEMA), 'Unknown plan schema')
    ordered = inference_mode(plan)
    ident = plan['identity']
    run = ident['run_id']
    require(re.fullmatch('[a-z][a-z0-9-]{8,49}', run) and not run.startswith('qwen013')
            and ident['experiment'] != 'qwen-specialization-013'
            and ident['account'] == '148761639313' and ident['region'] == 'ap-northeast-2',
            'Fresh explicit experiment identity required')
    require(ident['tags'] == {'trid:eval': ident['eval_tag'],
                            'trid:experiment': ident['experiment'], 'trid:run': run},
            'Exact explicit ownership tags required')
    for key in ('code_prefix', 'output_prefix'):
        relative(ident[key].rstrip('/'))
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*/', ident[key])
                and ident[key].endswith('/' + run + '/'), 'Dedicated literal exact-run S3 prefix required')
    require(ident['code_prefix'] != ident['output_prefix'], 'Separate code/output prefixes required')
    require(plan['limits'] == LIMITS and plan['instance_type'] == 'g6e.2xlarge'
            and plan['vcpus'] == 8 and plan['host_mib'] == 65536
            and plan['architecture'] == 'x86_64', 'Fixed resource envelope differs')
    if ordered:
        key_arn = ebs_key_arn(plan)
        require(not ({'recipe', 'data', 'evaluation', 'phase_seconds'} & set(plan)),
                'Inference plan cannot admit training or legacy workload fields')
        require(approval == dict(schema='trid-ordered-hf-inference-approval-v1', run_id=run,
                    plan_sha256=plan_sha, instance_limit=1, additional_pretax_usd=20,
                    workload_sha256=object_sha(plan['workload']), limits=LIMITS,
                    ebs_kms_key_arn=key_arn, automatic_retry=False, approved=True),
                'Exact USD20 inference-only approval is missing or mismatched')
    else:
        recipe = plan['recipe']
        require(all(recipe[k] == v for k, v in RECIPE.items())
                and type(recipe['epochs']) is int and recipe['epochs'] == 3,
                'Recipe requires explicit final approval; no adaptive epochs')
        require(type(recipe['rows']) is int and recipe['rows'] > 0
                and recipe['rows'] * recipe['epochs'] % recipe['batch_size'] == 0
                and recipe['steps'] == recipe['rows'] * recipe['epochs'] // recipe['batch_size'],
                'Finite steps must cover exactly three epochs without wrapped extra examples')
        key_arn = ebs_key_arn(plan)
        require(approval == dict(schema='trid-grounded-hf-approval-v1', run_id=run,
                    plan_sha256=plan_sha, instance_limit=1, additional_pretax_usd=50,
                    recipe=recipe, limits=LIMITS, ebs_kms_key_arn=key_arn, automatic_retry=False, approved=True),
                'Exact new final approval is missing or mismatched')
    current = time.time() if now is None else now
    require(0 <= current - timestamp(plan['checked_at']) <= 3600, 'Admission observations are stale')
    require(plan['ami']['ImageId'] == 'ami-040ebe879ae70cc30'
            and plan['ami']['OwnerId'] == '898082745236' and plan['ami']['root_volume_gib'] == 75,
            'Only the parent-selected standard Ubuntu24 DLAMI is admitted; no newer/Ubuntu Pro fallback')
    if ordered:
        models = plan['workload']['models']
        require(set(models) == set(ORDERED_MODELS), 'Exactly two registered model manifests required')
        require('model' not in plan or plan['model'] == models['qwen8'], 'Initial model must be immutable 8B')
    else:
        models = {'qwen14': plan['model']}
    file_count = 0
    for model_key, model in models.items():
        model_id, revision = ORDERED_MODELS[model_key]
        require(model['id'] == model_id and model['revision'] == revision
                and model['trust_remote_code'] is False, 'Unapproved model identity')
        files = pins(model['files'], model['max_bytes'])
        file_count += len(files)
        require({'config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'model.safetensors.index.json', 'LICENSE'} <= set(files)
                and any(n.endswith('.safetensors') for n in files)
                and all('/' not in n and (n.endswith(('.json', '.safetensors', '.txt')) or n == 'LICENSE')
                        for n in files), 'Incomplete/unsafe native HF acquisition manifest')
        for row in files.values():
            require(row['url'] == f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{row['path']}",
                    'Model acquisition URL must bind immutable official revision')
    runtime = plan['runtime']
    require(runtime['python_version'] and runtime['packages'] and runtime['driver_version']
            and runtime['torch_cuda'] and runtime['gpu_name'] == 'NVIDIA L40S', 'Complete pinned runtime required')
    validate_gpu_memory_contract(runtime)
    wheels = pins(runtime['wheels'], runtime['max_bytes'])
    packages = {'torch', 'transformers', 'tokenizers', 'safetensors', 'accelerate', 'psutil'}
    if not ordered:
        packages.add('peft')
    require(all(n.endswith('.whl') and '/' not in n for n in wheels)
            and packages <= set(runtime['packages']), 'Complete offline wheelhouse required')
    for row in wheels.values():
        require(row['url'].startswith('https://') and row['url'].split('/')[2] in runtime['download_hosts'],
                'Unapproved wheel origin')
    require(plan['bundle']['bytes'] <= 128 * 1024**2
            and plan['bundle']['unpacked_bytes'] <= 256 * 1024**2, 'Code/data bundle too large')
    bundled = pins(plan['bundle']['files'], plan['bundle']['unpacked_bytes'])
    bootstrap = runtime['bootstrap']
    require(set(bootstrap) == {'admission', 'nvidia_modprobe'},
            'Only verified same-AMI venv plus the pinned historical modprobe package is supported')
    require(bootstrap['admission']['path'] in bundled
            and bootstrap['admission']['sha256'] == bundled[bootstrap['admission']['path']]['sha256'],
            'Missing sealed bootstrap dependency admission')
    deb = bootstrap['nvidia_modprobe']
    pins([deb], 32 * 1024**2)
    require(deb['path'] == 'nvidia-modprobe.deb' and deb['package'] == 'nvidia-modprobe'
            and deb['architecture'] == 'amd64' and deb['version'] == '620.70.06-1ubuntu1'
            and deb['member']['path'] == 'usr/bin/nvidia-modprobe'
            and 0 < deb['member']['bytes'] <= 8 * 1024**2 and SHA.fullmatch(deb['member']['sha256'])
            and 0 < deb['max_unpacked_bytes'] <= 32 * 1024**2
            and deb['url'].startswith('https://') and deb['url'].split('/')[2] in runtime['download_hosts'],
            'Pinned historical NVIDIA package/archive/member contract differs')
    required_sources = ordered_sources() if ordered else SOURCES
    require(required_sources <= set(bundled) and set(plan['sources']) >= required_sources
            and {n for n in bundled if n.endswith(('.py', '.sh'))} <= set(plan['sources'])
            and all(bundled[k]['sha256'] == v for k, v in plan['sources'].items()),
            'All executable source pins required')
    require(all(not any(x in n.lower() for x in ('final24', 'dev36', '.env', 'credentials', '.pem'))
                for n in bundled), 'Protected input or credential-like bundle member')
    if ordered:
        validate_ordered_workload(plan, bundled)
        validate_worker_policy(plan)
    else:
        data = plan['data']
        for key in ('manifest', 'blind_inputs', 'admission', 'allocation'):
            row = data[key]
            require(row['path'] in bundled and bundled[row['path']]['sha256'] == row['sha256'],
                    'Missing sealed data/blind/admission pin')
        require(data['blind_cases'] and len({c['id'] for c in data['blind_cases']}) == len(data['blind_cases'])
                and all(SHA.fullmatch(c['prompt_sha256']) for c in data['blind_cases']),
                'Explicit blind case IDs and prompt hashes required')
        validate_evaluation(plan)
        require(set(plan['phase_seconds']) == {'acquire', 'capacity', 'baseline', 'train', 'adapter'}
                and all(type(v) is int and v > 0 for v in plan['phase_seconds'].values())
                and sum(plan['phase_seconds'].values()) <= 14400 - 1800 - 600,
                'Phase budgets must leave boot and final export/stop reserve')
    require(sum(model['max_bytes'] for model in models.values()) + 2 * runtime['max_bytes'] + plan['disk_os_reserve_bytes']
            + 4 * GIB + 128 * 1024**2 + plan['bundle']['unpacked_bytes'] < 120 * GIB
            and plan['disk_os_reserve_bytes'] >= 20 * GIB, 'Root disk envelope insufficient')
    require(plan['cost']['download_requests'] >= 6 * (file_count + len(wheels) + 1),
            'Redirect/download request envelope insufficient')
    require(plan['network']['download_hosts'] and plan['network']['egress_byte_cap']
            <= plan['cost']['network_egress_bytes'], 'Missing enforced network envelope')
    require(set(plan['price_queries']) == set(plan['cost']['rates']), 'Missing live price queries')
    price_filters = {
        'ec2_hour': ('AmazonEC2', 'Hrs', dict(regionCode=ident['region'], instanceType='g6e.2xlarge',
            operatingSystem='Linux', tenancy='Shared', preInstalledSw='NA', capacitystatus='Used')),
        'ebs_gb_month': ('AmazonEC2', 'GB-Mo', dict(regionCode=ident['region'], volumeApiName='gp3', productFamily='Storage')),
        'ipv4_hour': ('AmazonVPC', 'Hrs', dict(regionCode=ident['region'], usagetype='APN2-PublicIPv4:InUseAddress')),
        'egress_gb': ('AWSDataTransfer', 'GB', dict(fromLocation='Asia Pacific (Seoul)',
            toLocation='External', transferType='AWS Outbound')),
        's3_gb_month': ('AmazonS3', 'GB-Mo', dict(regionCode=ident['region'], productFamily='Storage', volumeType='Standard')),
        's3_get': ('AmazonS3', 'Requests', dict(regionCode=ident['region'], group='S3-API-Tier2')),
        's3_put': ('AmazonS3', 'Requests', dict(regionCode=ident['region'], group='S3-API-Tier1'))}
    for category, (service, unit, filters_) in price_filters.items():
        query = plan['price_queries'][category]
        require(query['service'] == service and query['unit'] == unit
                and all(query['filters'].get(k) == v for k, v in filters_.items()),
                'Price query category/region/unit differs: ' + category)
    evidence = plan['ancillary_evidence']
    require(evidence['path'] in bundled and evidence['sha256'] == bundled[evidence['path']]['sha256'],
            'Ancillary billing/security inventory must be sealed before launch')
    required_outputs = {'runtime.json', 'runtime.log', 'capacity.log', 'baseline.log', 'train.log', 'adapter.log',
        'capacity-receipt.json', 'baseline-outputs.jsonl', 'adapter-outputs.jsonl',
        'training/training.json', 'training/adapter/adapter_config.json',
        'training/adapter/adapter_model.safetensors', 'training/adapter/README.md'}
    if ordered:
        required_outputs = ordered_output_members(plan)
    require(len(plan['output_allowlist']) == len(required_outputs)
            and set(plan['output_allowlist']) == required_outputs,
            'Exact registered export allowlist required')
    return cost_bound(plan)


def validate_bootstrap_admission(root, plan):
    bootstrap = plan['runtime']['bootstrap']
    receipt = read_bound(Path(root) / bootstrap['admission']['path'], bootstrap['admission']['sha256'])
    require(receipt['schema'] == 'trid-grounded-hf-bootstrap-admission-v1'
            and receipt['status'] == 'admitted'
            and receipt['image_id'] == plan['ami']['ImageId'] and receipt['owner_id'] == plan['ami']['OwnerId']
            and receipt['python_version'] == plan['runtime']['python_version']
            and receipt['prerequisite_expectations_bound'] is True
            and receipt['same_ami_preinstalled_venv_verified'] is True
            and receipt['venv_package_version'] == '3.12.3-1ubuntu0.17'
            and receipt['aws_cli_version'] == '2.39.5'
            and receipt['no_unpinned_package_installation'] is True
            and receipt['artifact_inventory_sha256'] == object_sha(bootstrap['nvidia_modprobe'])
            and receipt['ubuntu_24_04_amd64_abi_verified'] is True,
            'Known-unknown AMI/bootstrap dependencies or changed historical artifact cannot consume a GPU launch')
    required = {'python3', 'aws', 'systemd-run', 'systemctl', 'iptables', 'ip6tables',
                'mount', 'useradd', 'shutdown', 'nvidia-smi', 'sha256sum', 'timeout', 'modprobe', 'dpkg-deb', 'dpkg-query'}
    levels = {'direct', 'executed_path_unobserved', 'mandatory_path_inference', 'service_observed', 'unobserved'}
    require(set(receipt['expected_commands']) == required
            and isinstance(receipt['postboot_required_names'], list)
            and len(receipt['postboot_required_names']) == len(required)
            and set(receipt['postboot_required_names']) == required
            and set(receipt['historical_evidence_levels']) == required
            and all(level in levels for level in receipt['historical_evidence_levels'].values())
            and all(set(row) in ({'path', 'probe_args'}, {'path', 'probe_args', 'sha256'})
                    and Path(row['path']).is_absolute()
                    and row['probe_args'] == (['--help'] if name in ('useradd', 'shutdown') else ['--version'])
                    and ('sha256' not in row or SHA.fullmatch(row['sha256']))
                    for name, row in receipt['expected_commands'].items()),
            'Complete bounded postboot command expectations and historical evidence levels required')
    bundle = {row['path']: row for row in plan['bundle']['files']}
    require(receipt['evidence'], 'Bootstrap prerequisites need actual source evidence')
    for pin in receipt['evidence']:
        require(pin['path'] in bundle and pin['sha256'] == bundle[pin['path']]['sha256']
                and digest(Path(root) / pin['path']) == pin['sha256'], 'Bootstrap source evidence differs')
    return receipt


def verify_prerequisites(actual, expected):
    """Actual postboot observations, never historical command-path assumptions."""
    require(isinstance(actual, dict) and actual.get('prerequisites_verified') is True
            and isinstance(actual.get('commands'), dict)
            and set(actual['commands']) == set(expected['postboot_required_names'])
            and all(actual.get(key) == expected[key]
                    for key in ('python_version', 'venv_package_version', 'aws_cli_version')),
            'Actual postboot prerequisite/version proof missing or changed')
    for name, pin in expected['expected_commands'].items():
        observed = actual['commands'][name]
        require(isinstance(observed, dict) and observed.get('found_path') == pin['path']
                and isinstance(observed.get('resolved_path'), str)
                and Path(observed['resolved_path']).is_absolute()
                and isinstance(observed.get('sha256'), str) and SHA.fullmatch(observed['sha256'])
                and ('sha256' not in pin or observed['sha256'] == pin['sha256'])
                and observed.get('probe_args') == pin['probe_args']
                and isinstance(observed.get('stdout'), str),
                'Actual postboot command capability proof differs: ' + name)
    return actual


def validate_cost_admission(root, plan):
    pin = plan['ancillary_evidence']
    receipt = read_bound(Path(root) / pin['path'], pin['sha256'])
    require(receipt['schema'] == 'trid-grounded-hf-cost-admission-v1'
            and receipt['status'] == 'admitted' and receipt['run_id'] == plan['identity']['run_id']
            and receipt['account'] == plan['identity']['account'] and receipt['region'] == plan['identity']['region']
            and receipt['ancillary_bound_usd'] == plan['cost']['ancillary_bound_usd']
            and receipt['ancillary_checks_sha256'] == object_sha(plan['ancillary_checks'])
            and receipt['pricing_products_sha256'] == {k: q['products_sha256'] for k, q in plan['price_queries'].items()}
            and receipt['bucket_lifecycle_configuration_sha256']
                == object_sha(plan['cost']['bucket_lifecycle_configuration'])
            and receipt['bucket_policy_status_sha256'] == object_sha(plan['bucket_policy_status'])
            and receipt['ebs_kms_key_arn'] == ebs_key_arn(plan)
            and receipt['no_unpriced_services'] is True and receipt['retention_closure_reviewed'] is True
            and receipt['network_request_closure_reviewed'] is True
            and 0 <= timestamp(plan['checked_at']) - timestamp(receipt['checked_at']) <= 3600,
            'Complete independent ancillary/retention/network/request cost admission is missing')
    validate_lifecycle_configuration(plan, plan['cost']['bucket_lifecycle_configuration'])
    validate_policy_status_evidence(root, plan, receipt['bucket_policy_status_evidence'])
    validate_ebs_key_evidence(root, plan, receipt['ebs_kms_key_evidence'])
    return receipt


def validate_ebs_key_evidence(root, plan, evidence):
    arn = ebs_key_arn(plan)
    bundled = {row['path']: row for row in plan['bundle']['files']}
    require(evidence['path'] in bundled and evidence['sha256'] == bundled[evidence['path']]['sha256'],
            'EBS managed-key source evidence is not bundled')
    raw = read_bound(Path(root) / evidence['path'], evidence['sha256'])
    query = raw['query']
    require(raw['exit_code'] == 0 and raw.get('error_code') is None
            and query[:3] == ['aws', 'kms', 'describe-key']
            and query.count('--key-id') == 1 and query[query.index('--key-id') + 1] == 'alias/aws/ebs'
            and query.count('--region') == 1 and query[query.index('--region') + 1] == plan['identity']['region'],
            'Successful actual AWS-managed EBS key observation required')
    metadata = raw['response']['KeyMetadata']
    require(metadata['Arn'] == arn and metadata['KeyId'] == arn.rsplit('/', 1)[1]
            and metadata['AWSAccountId'] == plan['identity']['account']
            and metadata['KeyManager'] == 'AWS' and metadata['Origin'] == 'AWS_KMS'
            and metadata['Enabled'] is True and metadata['KeyState'] == 'Enabled'
            and metadata['KeyUsage'] == 'ENCRYPT_DECRYPT' and metadata['KeySpec'] == 'SYMMETRIC_DEFAULT'
            and metadata['EncryptionAlgorithms'] == ['SYMMETRIC_DEFAULT'] and metadata['MultiRegion'] is False,
            'Observed EBS key differs from the enabled, same-account AWS-managed key approved for cost')
    return metadata


def validate_evaluation(plan):
    evaluation = plan['evaluation']
    laws = {'law:009916', 'law:005046', 'law:005048'}
    require(evaluation['cluster_id'] == 'local-government-service-compensation-connected'
            and evaluation['checkpoint_selection'] == 'final_only' and evaluation['adaptive_selection'] is False
            and len(evaluation['grounded_families']) == 3 and set(evaluation['grounded_families']) == laws
            and evaluation['synthetic_general_retention_count'] == 6,
            'Sealed data cluster, three grounded laws and six separate retention controls required')
    grounded, retention = [], []
    for case in plan['data']['blind_cases']:
        if case['kind'] == 'grounded':
            require(case['family'] in laws, 'Grounded case has unapproved law family')
            grounded.append(case)
        else:
            require(case['kind'] == case['family'] == 'synthetic-general-retention',
                    'Retention control must not masquerade as a grounded law case')
            retention.append(case)
    require({c['family'] for c in grounded} == laws and len(retention) == 6,
            'Every grounded family and exactly six separate general-retention cases must be represented')


def blind_messages(case):
    """Plain chat and grounded JSON-content chat; never import target labels."""
    require(set(case) == {'id', 'messages'}, 'Blind labels/extra fields rejected')
    messages = case['messages']
    require(isinstance(messages, list) and len(messages) in (1, 2)
            and all(isinstance(m, dict) and set(m) == {'role', 'content'}
                    and isinstance(m['content'], str) and m['content'].strip() for m in messages)
            and [m['role'] for m in messages] in (['user'], ['system', 'user']),
            'Blind input must be plain user chat or exact system/user messages')
    return messages


def validate_data(root, plan):
    if inference_mode(plan):
        return validate_ordered_data(root, plan)
    root = Path(root)
    data = plan['data']
    validate_evaluation(plan)
    manifest = read_bound(root / data['manifest']['path'], data['manifest']['sha256'])
    admission = read_bound(root / data['admission']['path'], data['admission']['sha256'])
    allocation = read_bound(root / data['allocation']['path'], data['allocation']['sha256'])
    cluster = allocation['prospective_evaluation_cluster']
    require(allocation['schema_version'] == 1 and allocation['owner'] == 'Main'
            and allocation['status'] == 'EVALUATION_ADMITTED'
            and allocation['train_allocation_changed'] is False
            and cluster['id'] == plan['evaluation']['cluster_id']
            == 'local-government-service-compensation-connected'
            and set(cluster['families']) == {'law:009916', 'law:005046', 'law:005048'}
            and cluster['separate_validation_split'] is False
            and allocation['synthetic_general_retention_count'] == 6
            and allocation['questions_authored_after_source_closure'] is True
            and allocation['blind_inputs_sha256'] == data['blind_inputs']['sha256']
            and allocation['blind_cases_sha256'] == object_sha(data['blind_cases']),
            'Prospective/source-only disposition is not sealed evaluation admission')
    require(admission['status'] == 'admitted' and admission['run_id'] == plan['identity']['run_id']
            and admission['manifest_sha256'] == data['manifest']['sha256']
            and admission['blind_inputs_sha256'] == data['blind_inputs']['sha256']
            and admission['evaluation_allocation_sha256'] == data['allocation']['sha256']
            and all(admission[k] is True for k in ('rights_reviewed', 'source_verified',
                   'overlap_reviewed', 'targets_reviewed', 'protected_sets_excluded')),
            'Independent data admission is not sealed')
    require(not manifest.get('valid') and manifest['train'], 'No undeclared validation split')
    bundle = {p['path']: p for p in plan['bundle']['files']}
    count, ids = 0, set()
    parent = PurePosixPath(data['manifest']['path']).parent
    for entry in manifest['train']:
        name = relative((parent / relative(entry['path'])).as_posix())
        require(name in bundle and entry['sha256'] == bundle[name]['sha256']
                and entry['adapter'] == 'grounded_messages' and entry['epochs'] == plan['recipe']['epochs']
                and entry.get('split') in (None, 'train'), 'Training manifest contract differs')
        path = root / name
        require(digest(path) == entry['sha256'], 'Training input changed')
        for line in path.read_text().splitlines():
            row = json.loads(line)
            require(row['split'] == 'train' and row['id'] not in ids and row['answer'].strip()
                    and len(row['messages']) == 2
                    and [m['role'] for m in row['messages']] == ['system', 'user']
                    and all(set(m) == {'role', 'content'} and m['content'].strip() for m in row['messages']),
                    'Training row is not a fresh grounded target')
            ids.add(row['id']); count += 1
    require(count == plan['recipe']['rows'], 'Approved row count differs')
    blind = root / data['blind_inputs']['path']
    require(digest(blind) == data['blind_inputs']['sha256'], 'Blind inputs changed')
    cases = [json.loads(line) for line in blind.read_text().splitlines()]
    require(len(cases) == len(data['blind_cases']), 'Blind case count differs')
    normalized = []
    for case, pin in zip(cases, data['blind_cases']):
        messages = blind_messages(case)
        require(case['id'] == pin['id'] and case['id'] not in ids
                and object_sha(messages) == pin['prompt_sha256'],
                'Blind labels, altered prompts, or training overlap rejected')
        require(pin['kind'] != 'grounded' or [m['role'] for m in messages] == ['system', 'user'],
                'Grounded input must retain the actual system/user builder messages')
        normalized.append(dict(id=case['id'], messages=messages, kind=pin['kind'], family=pin['family']))
    return normalized


class AWSCallError(RuntimeError):
    def __init__(self, service, operation, stderr):
        match = re.search(r'An error occurred \(([^)]+)\)', stderr)
        self.code = match.group(1) if match else None
        super().__init__(f'{service} {operation} failed: {stderr[:500]}')


class Client:
    def __init__(self, identity, deadline=None, request_limit=4096):
        self.identity, self.deadline = identity, deadline
        self.request_limit, self.requests = request_limit, 0

    def __call__(self, service, operation, *args, region=None):
        require(self.requests < self.request_limit, 'AWS request envelope exhausted')
        self.requests += 1
        seconds = min(45, remaining(self.deadline)) if self.deadline else 45
        region = region or self.identity['region']
        endpoint = ['--endpoint-url', f'https://s3.{region}.amazonaws.com'] if service == 's3api' else []
        result = subprocess.run(['aws', service, operation, *args, '--region', region, *endpoint,
            '--output', 'json', '--no-cli-pager', '--no-paginate', '--cli-connect-timeout', '5', '--cli-read-timeout', '30'],
            capture_output=True, text=True, timeout=seconds,
            env={**os.environ, 'AWS_PAGER': '', 'AWS_MAX_ATTEMPTS': '1', 'AWS_RETRY_MODE': 'standard',
                 'AWS_IGNORE_CONFIGURED_ENDPOINT_URLS': 'true'})
        if result.returncode:
            raise AWSCallError(service, operation, result.stderr)
        value = json.loads(result.stdout) if result.stdout.strip() else {}
        require(not value.get('NextToken') and not value.get('IsTruncated') and not value.get('NextMarker'),
                'Incomplete AWS observation')
        return value


def filters(identity):
    return [f'Name=tag:{key},Values={value}' for key, value in identity['tags'].items()]


def discover(client, identity):
    value = client('ec2', 'describe-instances', '--filters', *filters(identity))
    return [r for group in value['Reservations'] for r in group['Instances']]


def owned(row, identity, expected=None):
    tags = row.get('Tags', [])
    actual = {v['Key']: v['Value'] for v in tags}
    require(len(actual) == len(tags) and all(actual.get(k) == v for k, v in identity['tags'].items())
            and row.get('ClientToken') == identity['run_id']
            and (expected is None or row.get('InstanceId') == expected), 'Instance ownership conflict')
    return row['InstanceId']


def volume_identity(identity, plan_sha, root_device_name):
    return dict(schema='trid-grounded-hf-volume-identity-v1', run_id=identity['run_id'],
                plan_sha256=plan_sha, root_device_name=root_device_name,
                instance_id=None, root_volume_ids=[], volumes={})


def load_volume_identity(state, plan, plan_sha):
    path = Path(state) / 'volume-identity.json'
    blank = volume_identity(plan['identity'], plan_sha, plan['ami']['RootDeviceName'])
    if not path.exists():
        return blank
    evidence = json.loads(path.read_text())
    require(all(evidence[k] == blank[k] for k in
                ('schema', 'run_id', 'plan_sha256', 'root_device_name'))
            and isinstance(evidence['volumes'], dict)
            and isinstance(evidence['root_volume_ids'], list)
            and set(evidence['root_volume_ids']) <= set(evidence['volumes']),
            'Persisted volume identity differs from consumed plan')
    return evidence


def observe_volume_identity(client, identity, row, evidence, evidence_path=None):
    """Persist owned instance mappings before any subsequent AWS call can fail."""
    instance_id = owned(row, identity, evidence['instance_id'])
    evidence['instance_id'] = instance_id
    for mapping in row.get('BlockDeviceMappings', []):
        volume_id = mapping.get('Ebs', {}).get('VolumeId')
        if not volume_id:
            continue
        record = evidence['volumes'].setdefault(volume_id, dict(instance_mappings=[], owned_attachments=[]))
        if mapping not in record['instance_mappings']:
            record['instance_mappings'].append(mapping)
        if mapping['DeviceName'] == evidence['root_device_name'] and volume_id not in evidence['root_volume_ids']:
            evidence['root_volume_ids'].append(volume_id)
    if evidence_path is not None:
        save(evidence_path, evidence)
    errors = []
    for volume_id, record in evidence['volumes'].items():
        try:
            volumes = client('ec2', 'describe-volumes', '--volume-ids', volume_id)['Volumes']
        except AWSCallError as exc:
            if exc.code != 'InvalidVolume.NotFound':
                errors.append(str(exc))
            continue
        except Exception as exc:
            errors.append(str(exc))
            continue
        require(len(volumes) == 1 and volumes[0]['VolumeId'] == volume_id,
                'Exact volume identity observation incomplete')
        volume = volumes[0]
        tags = volume.get('Tags', [])
        actual = {tag['Key']: tag['Value'] for tag in tags}
        require(len(actual) == len(tags) and all(actual.get(k) == v for k, v in identity['tags'].items())
                and all(a['InstanceId'] == instance_id for a in volume.get('Attachments', [])),
                'Volume ownership conflict; refuse destructive action')
        for attachment in volume.get('Attachments', []):
            observation = dict(attachment=attachment, tags=tags)
            if observation not in record['owned_attachments']:
                record['owned_attachments'].append(observation)
                if evidence_path is not None:
                    save(evidence_path, evidence)
    return errors


def cleanup_cutoff(plan, plan_sha, state):
    """One absolute priced cutoff, including an ambiguous launch's saved dispatch."""
    require(inference_mode(plan), 'Absolute ordered cleanup cutoff requires inference plan')
    state = Path(state)
    identity = plan['identity']
    consumed = json.loads((state / 'approval-consumed.json').read_text())
    require(consumed == dict(run_id=identity['run_id'], plan_sha256=plan_sha),
            'Cleanup cutoff requires exact consumed approval')
    dispatch = json.loads((state / 'launch-dispatched.json').read_text())
    require(set(dispatch) == {'run_id', 'plan_sha256', 'request_sha256', 'at'}
            and dispatch['run_id'] == identity['run_id'] and dispatch['plan_sha256'] == plan_sha
            and SHA.fullmatch(dispatch['request_sha256'])
            and type(dispatch['at']) in (int, float) and math.isfinite(dispatch['at']) and dispatch['at'] > 0,
            'Cleanup cutoff requires a finite authenticated dispatch time')
    # Dispatch precedes resource creation, so this fallback is conservative even
    # when RunInstances succeeded but its response or later discovery was lost.
    launch_times = [dispatch['at']]
    for name in ('termination.json', 'release.json'):
        path = state / name
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        require(receipt['run_id'] == identity['run_id'] and receipt['plan_sha256'] == plan_sha
                and type(receipt['launch_time']) in (int, float)
                and math.isfinite(receipt['launch_time']) and receipt['launch_time'] > 0
                and receipt['deadline'] == receipt['launch_time'] + LIMITS['lifetime_seconds'],
                'Cleanup cutoff launch evidence differs')
        verify_schedule(receipt['schedule'], schedule_spec(plan, receipt['instance_id'], receipt['launch_time']))
        launch_times.append(receipt['launch_time'])
    return min(launch_times) + LIMITS['lifetime_seconds'] + plan['cost']['termination_grace_seconds']


def reconcile(client, identity, expected, deadline, *, evidence, evidence_path=None,
              clock=time.time, pause=time.sleep):
    if not hasattr(client, 'deadline'):
        return _reconcile(client, identity, expected, deadline, evidence=evidence,
                          evidence_path=evidence_path, clock=clock, pause=pause)
    previous_deadline = client.deadline
    client.deadline = min(previous_deadline, deadline) if previous_deadline is not None else deadline
    try:
        return _reconcile(client, identity, expected, client.deadline, evidence=evidence,
                          evidence_path=evidence_path, clock=clock, pause=pause)
    finally:
        client.deadline = previous_deadline


def _reconcile(client, identity, expected, deadline, *, evidence, evidence_path=None,
               clock=time.time, pause=time.sleep):
    require(expected is None or evidence['instance_id'] in (None, expected), 'Persisted instance identity conflict')
    expected = expected or evidence['instance_id']
    result = dict(cleanup_state='unresolved', instance_id=expected, errors=[], observations=0,
                  absent_volume_ids=[], root_volume_ids=evidence['root_volume_ids'])
    while clock() < deadline:
        try:
            rows = discover(client, identity)
            result['observations'] += 1
            if rows:
                require(len(rows) == 1, 'Multiple exact-run instances; refuse destructive action')
                row = rows[0]
                expected = owned(row, identity, expected)
                result['instance_id'] = expected
                result['errors'].extend(observe_volume_identity(client, identity, row, evidence, evidence_path))
                if row['State']['Name'] == 'terminated':
                    result['instance_terminated'] = True
                    volumes = client('ec2', 'describe-volumes', '--filters', *filters(identity))['Volumes']
                    remaining_ids = {v['VolumeId'] for v in volumes}
                    absent = []
                    for volume_id in evidence['volumes']:
                        try:
                            observed = client('ec2', 'describe-volumes', '--volume-ids', volume_id)['Volumes']
                            require(len(observed) == 1 and observed[0]['VolumeId'] == volume_id,
                                    'Exact volume absence is not proven by an empty response')
                            remaining_ids.add(volume_id)
                        except AWSCallError as exc:
                            if exc.code != 'InvalidVolume.NotFound':
                                raise
                            absent.append(volume_id)
                    result.update(remaining_volumes=sorted(remaining_ids), absent_volume_ids=sorted(absent),
                                  recorded_volume_ids=sorted(evidence['volumes']))
                    if (evidence['root_volume_ids'] and not remaining_ids
                            and set(absent) == set(evidence['volumes'])):
                        result['cleanup_state'] = 'complete'
                        return result
                elif row['State']['Name'] != 'shutting-down':
                    client('ec2', 'terminate-instances', '--instance-ids', expected)
        except Exception as exc:
            result['errors'].append(str(exc))
        if clock() < deadline:
            pause(min(5, deadline - clock()))
    return result


def role_inventory(client, name):
    role = client('iam', 'get-role', '--role-name', name)['Role']
    require(not role.get('PermissionsBoundary'), 'Unregistered IAM permissions boundary')
    names = client('iam', 'list-role-policies', '--role-name', name)['PolicyNames']
    inline = {n: client('iam', 'get-role-policy', '--role-name', name,
                       '--policy-name', n)['PolicyDocument'] for n in names}
    attached = client('iam', 'list-attached-role-policies', '--role-name', name)['AttachedPolicies']
    # Instance bootstrap needs only prefix-scoped S3, never SSM/control credentials.
    require(not attached, 'Managed/SSM policies are outside the worker permission boundary')
    return dict(arn=role['Arn'], trust=role['AssumeRolePolicyDocument'], inline=inline)


def verify_scheduler_trust(inventory, identity):
    group_arn = f"arn:aws:scheduler:{identity['region']}:{identity['account']}:schedule-group/{identity['schedule_group']}"
    expected = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow',
        'Principal': {'Service': 'scheduler.amazonaws.com'}, 'Action': 'sts:AssumeRole',
        'Condition': {'StringEquals': {'aws:SourceAccount': identity['account']},
                      'ArnEquals': {'aws:SourceArn': group_arn}}}]}
    require(inventory['trust'] == expected, 'Scheduler trust must bind schedule-GROUP SourceArn, not schedule ARN')


def validate_lifecycle_configuration(plan, observed):
    """Price every possible overlapping rule, without mutating bucket rules."""
    require(observed == plan['cost']['bucket_lifecycle_configuration'],
            'Full bucket lifecycle configuration changed since cost admission')
    require(isinstance(observed.get('Rules'), list), 'Missing bucket lifecycle rules')
    for prefix in (plan['identity']['code_prefix'], plan['identity']['output_prefix']):
        exact = []
        for rule in observed['Rules']:
            require(rule.get('Status') in ('Enabled', 'Disabled'), 'Ambiguous lifecycle rule status')
            if rule['Status'] == 'Disabled':
                continue
            # Only literal prefix selectors have a fully known object scope.
            # Tags, size bounds and AND filters cannot prove cost closure.
            if 'Filter' in rule:
                selector = rule['Filter']
                require('Prefix' not in rule and isinstance(selector, dict)
                        and set(selector) <= {'Prefix'}, 'Ambiguous lifecycle filter')
                selected = selector.get('Prefix', '')
            else:
                require('Prefix' in rule, 'Ambiguous lifecycle filter')
                selected = rule['Prefix']
            require(isinstance(selected, str), 'Ambiguous lifecycle prefix')
            if not (prefix.startswith(selected) or selected.startswith(prefix)):
                continue
            require(not any(k in rule for k in ('Transitions', 'NoncurrentVersionTransitions')),
                    'Unpriced overlapping lifecycle transitions')
            require(set(rule) <= {'ID', 'Status', 'Filter', 'Prefix', 'Expiration',
                                 'AbortIncompleteMultipartUpload', 'NoncurrentVersionExpiration'},
                    'Unpriced overlapping lifecycle action')
            if selected == prefix:
                exact.append(rule)
        require(len(exact) == 1 and exact[0].get('Expiration') == {'Days': plan['cost']['retention_days']},
                'Existing exact-prefix expiration closure missing; do not overwrite bucket rules')
    return observed


def private_policy_status(response):
    require(isinstance(response, dict) and response.get('PolicyStatus') == {'IsPublic': False}
            and response['PolicyStatus']['IsPublic'] is False, 'Public or unknown bucket policy rejected')
    return dict(state='present', is_public=False)


def validate_policy_status_evidence(root, plan, evidence):
    bundled = {row['path']: row for row in plan['bundle']['files']}
    require(evidence['path'] in bundled and evidence['sha256'] == bundled[evidence['path']]['sha256'],
            'Bucket policy status source evidence is not bundled')
    raw = read_bound(Path(root) / evidence['path'], evidence['sha256'])
    query = raw['query']
    require(query[:3] == ['aws', 's3api', 'get-bucket-policy-status']
            and query.count('--bucket') == 1 and query[query.index('--bucket') + 1] == plan['identity']['bucket']
            and query.count('--region') == 1 and query[query.index('--region') + 1] == plan['identity']['region'],
            'Bucket policy status evidence targets another operation or resource')
    if raw['exit_code'] == 0:
        require(raw.get('error_code') is None, 'Successful policy status evidence contains an error')
        policy_status = private_policy_status(raw['response'])
    else:
        require(type(raw['exit_code']) is int and raw['exit_code'] > 0
                and raw.get('error_code') == 'NoSuchBucketPolicy' and raw.get('response') is None
                and AWSCallError('s3api', 'get-bucket-policy-status', raw.get('stderr', '')).code == 'NoSuchBucketPolicy',
                'Unverified bucket policy status failure is not absence')
        policy_status = dict(state='absent', error_code='NoSuchBucketPolicy')
    require(policy_status == plan['bucket_policy_status'], 'Sealed bucket policy status evidence differs')
    return policy_status


def validate_bucket_encryption(encryption):
    require(set(encryption) == {'Rules'} and len(encryption['Rules']) == 1,
            'Only one cost-closed SSE-S3 encryption rule admitted')
    rule = encryption['Rules'][0]
    require(set(rule) <= {'ApplyServerSideEncryptionByDefault', 'BucketKeyEnabled', 'BlockedEncryptionTypes'}
            and rule.get('ApplyServerSideEncryptionByDefault') == {'SSEAlgorithm': 'AES256'}
            and ('BucketKeyEnabled' not in rule or rule['BucketKeyEnabled'] is False)
            and ('BlockedEncryptionTypes' not in rule
                 or rule['BlockedEncryptionTypes'] == {'EncryptionType': ['SSE-C']}),
            'Only known cost-closed SSE-S3 bucket encryption admitted')


def observe_private_bucket(plan, client):
    identity = plan['identity']
    args = ['--bucket', identity['bucket'], '--expected-bucket-owner', identity['account']]
    block = client('s3api', 'get-public-access-block', *args)['PublicAccessBlockConfiguration']
    require(set(block) == {'BlockPublicAcls', 'IgnorePublicAcls', 'BlockPublicPolicy', 'RestrictPublicBuckets'}
            and all(value is True for value in block.values()), 'Private bucket public-access blocks required')
    ownership = client('s3api', 'get-bucket-ownership-controls', *args)['OwnershipControls']
    require(ownership == {'Rules': [{'ObjectOwnership': 'BucketOwnerEnforced'}]}, 'Bucket-owner-enforced ownership required')
    acl = client('s3api', 'get-bucket-acl', *args)
    owner, grants = acl.get('Owner', {}), acl.get('Grants', [])
    require(isinstance(owner.get('ID'), str) and owner['ID'] and len(grants) == 1,
            'Private bucket ACL owner and sole owner grant required')
    grant = grants[0]
    grantee = grant.get('Grantee', {})
    require(set(grant) == {'Grantee', 'Permission'} and grant['Permission'] == 'FULL_CONTROL'
            and set(grantee) <= {'Type', 'ID', 'DisplayName'} and grantee.get('Type') == 'CanonicalUser'
            and grantee.get('ID') == owner['ID'], 'Non-owner or public ACL grant rejected')
    try:
        response = client('s3api', 'get-bucket-policy-status', *args)
    except AWSCallError as exc:
        if exc.code != 'NoSuchBucketPolicy':
            raise
        policy_status = dict(state='absent', error_code=exc.code)
    else:
        policy_status = private_policy_status(response)
    require(policy_status == plan['bucket_policy_status'], 'Observed bucket policy status changed')
    return policy_status


def observe_ancillary_check(name, check, identity, policy_status, client):
    require(check['operation'].startswith(('get-', 'describe-', 'list-')), 'Ancillary checks are read-only')
    try:
        observed = client(check['service'], check['operation'], *check['args'])
    except AWSCallError as exc:
        absent = {'bucket_replication': ('get-bucket-replication', 'ReplicationConfigurationNotFoundError'),
                  'bucket_policy': ('get-bucket-policy', 'NoSuchBucketPolicy')}
        require(name in absent and check['service'] == 's3api'
                and (check['operation'], exc.code) == absent[name]
                and check['args'] == ['--bucket', identity['bucket']]
                and check.get('expected_absence') == exc.code,
                'Unregistered ancillary failure is not evidence of absence')
        if name == 'bucket_policy':
            require(policy_status == dict(state='absent', error_code='NoSuchBucketPolicy'),
                    'Policy content absence disagrees with its status')
    else:
        require(not check.get('expected_absence') and object_sha(observed) == check['response_sha256'],
                'Ancillary billing observation changed')
        if name == 'bucket_policy':
            require(policy_status == dict(state='present', is_public=False),
                    'Policy content presence disagrees with its status')


def live_preflight(plan, client):
    identity = plan['identity']
    caller = client('sts', 'get-caller-identity')
    require(caller['Account'] == identity['account'] and caller['Arn'] == plan['controller_arn'],
            'Controller principal differs from explicit approval')
    ami = client('ec2', 'describe-images', '--image-ids', plan['ami']['ImageId'])['Images']
    require(len(ami) == 1 and all(ami[0].get(k) == plan['ami'][k] for k in
            ('ImageId', 'OwnerId', 'Name', 'Architecture', 'RootDeviceName'))
            and ami[0]['State'] == 'available' and ami[0]['RootDeviceType'] == 'ebs'
            and ami[0]['Architecture'] == 'x86_64'
            and not ami[0].get('BillingProducts') and not ami[0].get('ProductCodes'),
            'AMI identity or unpriced license billing differs')
    validate_ami_mappings(ami[0], plan['ami'])
    require(plan['ami']['root_volume_gib'] <= 120, 'AMI exceeds root volume')
    types = client('ec2', 'describe-instance-types', '--instance-types', plan['instance_type'])['InstanceTypes']
    require(len(types) == 1 and types[0]['VCpuInfo']['DefaultVCpus'] == 8
            and types[0]['MemoryInfo']['SizeInMiB'] == 65536, 'Instance host shape changed')
    verify_gpu_catalog(plan, types[0]['GpuInfo'])
    quota = client('service-quotas', 'get-service-quota', '--service-code', 'ec2',
                   '--quota-code', 'L-DB2E81BA')['Quota']['Value']
    all_instances = client('ec2', 'describe-instances', '--filters',
                          'Name=instance-state-name,Values=pending,running,stopping,stopped')
    gpu_types = [r['InstanceType'] for g in all_instances['Reservations'] for r in g['Instances']
                 if r['InstanceType'].split('.')[0].startswith(('g', 'vt'))]
    sizes = {kind: client('ec2', 'describe-instance-types', '--instance-types', kind)['InstanceTypes'][0]
             ['VCpuInfo']['DefaultVCpus'] for kind in set(gpu_types)}
    require(sum(sizes[k] for k in gpu_types) + 8 <= quota, 'Conservative GPU quota insufficient')
    network = plan['network']
    subnet = client('ec2', 'describe-subnets', '--subnet-ids', network['subnet_id'])['Subnets']
    require(len(subnet) == 1 and subnet[0]['VpcId'] == network['vpc_id']
            and subnet[0]['AvailabilityZone'] == network['availability_zone']
            and subnet[0]['State'] == 'available', 'Subnet observation differs')
    routes = client('ec2', 'describe-route-tables', '--route-table-ids', network['route_table_id'])['RouteTables']
    require(len(routes) == 1 and routes[0] == network['route_table_observation'], 'Route table pin changed')
    require(any(r.get('DestinationCidrBlock') == '0.0.0.0/0'
                and r.get('GatewayId', '').startswith('igw-') and r['State'] == 'active'
                for r in routes[0]['Routes']) and not any(r.get('NatGatewayId') for r in routes[0]['Routes']),
            'Direct IGW required; paid NAT route rejected')
    effective = client('ec2', 'describe-route-tables', '--filters',
                       f"Name=association.subnet-id,Values={network['subnet_id']}")['RouteTables']
    if not effective:
        effective = client('ec2', 'describe-route-tables', '--filters',
                           f"Name=vpc-id,Values={network['vpc_id']}", 'Name=association.main,Values=true')['RouteTables']
    require(len(effective) == 1 and effective[0]['RouteTableId'] == network['route_table_id'],
            'Pinned route table is not effective for subnet')
    groups = client('ec2', 'describe-security-groups', '--group-ids', network['security_group_id'])['SecurityGroups']
    require(len(groups) == 1 and not groups[0]['IpPermissions']
            and groups[0] == network['security_group_observation'], 'No-ingress security group pin differs')
    group = client('scheduler', 'get-schedule-group', '--name', identity['schedule_group'])
    require(group['State'] == 'ACTIVE' and group['Arn']
            == f"arn:aws:scheduler:{identity['region']}:{identity['account']}:schedule-group/{identity['schedule_group']}",
            'Independent scheduler group is not admitted')
    for key in ('instance_role', 'scheduler_role'):
        observed = role_inventory(client, identity[key])
        require(observed == plan['iam'][key], 'IAM trust/policies changed: ' + key)
        if key == 'scheduler_role':
            verify_scheduler_trust(observed, identity)
    require(plan['iam']['instance_role']['trust'] == {'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]},
        'Instance trust is not exclusively EC2')
    validate_worker_policy(plan)
    scheduler_statements = [s for p in plan['iam']['scheduler_role']['inline'].values() for s in p['Statement']]
    require(scheduler_statements == [dict(Effect='Allow', Action='ec2:TerminateInstances',
        Resource=f"arn:aws:ec2:{identity['region']}:{identity['account']}:instance/*",
        Condition={'StringEquals': {f'ec2:ResourceTag/{k}': v for k, v in identity['tags'].items()}})],
        'Scheduler termination must require all fresh ownership tags')
    profile = client('iam', 'get-instance-profile', '--instance-profile-name', identity['instance_profile'])['InstanceProfile']
    require([r['Arn'] for r in profile['Roles']] == [plan['iam']['instance_role']['arn']],
            'Instance profile role differs')
    for role, action, resource in (
        ('instance_role', 's3:GetObject', f"arn:aws:s3:::{identity['bucket']}/{identity['code_prefix']}release.json"),
        ('instance_role', 's3:PutObject', f"arn:aws:s3:::{identity['bucket']}/{identity['output_prefix']}export.tar"),
        ('scheduler_role', 'ec2:TerminateInstances',
         f"arn:aws:ec2:{identity['region']}:{identity['account']}:instance/i-00000000000000000")):
        context = [dict(ContextKeyName=f'ec2:ResourceTag/{k}', ContextKeyType='string', ContextKeyValues=[v])
                   for k, v in identity['tags'].items()] if role == 'scheduler_role' else []
        args = ['--policy-source-arn', plan['iam'][role]['arn'], '--action-names', action,
                '--resource-arns', resource]
        if context:
            args += ['--context-entries', json.dumps(context)]
        decisions = client('iam', 'simulate-principal-policy', *args)['EvaluationResults']
        require(len(decisions) == 1 and decisions[0]['EvalActionName'] == action
                and decisions[0]['EvalResourceName'] == resource and decisions[0]['EvalDecision'] == 'allowed'
                and not decisions[0].get('MissingContextValues'), 'Required exact-role action not allowed')
    bucket = identity['bucket']
    require(client('s3api', 'get-bucket-location', '--bucket', bucket)['LocationConstraint'] == identity['region'],
            'Bucket is not same region')
    require(not client('s3api', 'get-bucket-versioning', '--bucket', bucket).get('Status'),
            'Versioned/suspended bucket retention is not admitted')
    policy_status = observe_private_bucket(plan, client)
    encryption = client('s3api', 'get-bucket-encryption', '--bucket', bucket)['ServerSideEncryptionConfiguration']
    validate_bucket_encryption(encryption)
    lifecycle = client('s3api', 'get-bucket-lifecycle-configuration', '--bucket', bucket)
    validate_lifecycle_configuration(plan, lifecycle)
    # Exact read-only observations close ancillary costs (replication, logging,
    # CloudTrail data events, flow logs, customer-managed keys, scheduled backups).
    # Parent supplies API-specific expected success/error shapes; no assumed zero.
    required = {'bucket_replication', 'bucket_logging', 'cloudtrail', 'flow_logs', 'backup', 'bucket_policy'}
    require(set(plan['ancillary_checks']) == required, 'Ancillary API inventory incomplete')
    for name, check in plan['ancillary_checks'].items():
        observe_ancillary_check(name, check, identity, policy_status, client)
    for category, query in plan['price_queries'].items():
        result = client('pricing', 'get-products', '--service-code', query['service'], '--filters',
            json.dumps([dict(Type='TERM_MATCH', Field=k, Value=v) for k, v in query['filters'].items()]),
            region='us-east-1')
        products = [json.loads(s) for s in result['PriceList']]
        require(products and object_sha(products) == query['products_sha256'], 'Price product evidence changed')
        values = [float(d['pricePerUnit']['USD']) for p in products for term in p.get('terms', {}).get('OnDemand', {}).values()
                  for d in term['priceDimensions'].values() if d['unit'] == query['unit']]
        require(values and all(math.isfinite(v) and v >= 0 for v in values)
                and max(values) == plan['cost']['rates'][category], 'Live price rate differs')
    return dict(caller=caller, cost=cost_bound(plan), bucket_policy_status=policy_status, checked_at=time.time())


def verify_launched_resources(client, plan, row):
    identity = plan['identity']
    instance_id = owned(row, identity)
    require(row['ImageId'] == plan['ami']['ImageId'] and row['InstanceType'] == 'g6e.2xlarge'
            and row['SubnetId'] == plan['network']['subnet_id']
            and row['Placement']['AvailabilityZone'] == plan['network']['availability_zone']
            and row['MetadataOptions']['HttpTokens'] == 'required'
            and row['MetadataOptions']['HttpPutResponseHopLimit'] == 1
            and len(row['NetworkInterfaces']) == 1
            and not row['NetworkInterfaces'][0].get('Ipv6Addresses')
            and not row['NetworkInterfaces'][0].get('Association', {}).get('AllocationId'),
            'Launched instance/network/metadata envelope differs')
    mappings = row['BlockDeviceMappings']
    require(len(mappings) == 1 and mappings[0]['DeviceName'] == plan['ami']['RootDeviceName']
            and mappings[0]['Ebs']['DeleteOnTermination'] is True, 'Launched root mapping differs')
    volumes = client('ec2', 'describe-volumes', '--volume-ids', mappings[0]['Ebs']['VolumeId'])['Volumes']
    require(len(volumes) == 1, 'Owned root volume not observed')
    volume = volumes[0]
    require(all({r['Key']: r['Value'] for r in volume.get('Tags', [])}.get(k) == v
                for k, v in identity['tags'].items())
            and volume['Size'] == 120 and volume['VolumeType'] == 'gp3'
            and volume['Iops'] == 3000 and volume['Throughput'] == 125 and volume['Encrypted'] is True
            and len(volume['Attachments']) == 1 and volume['Attachments'][0]['InstanceId'] == instance_id,
            'Owned root volume shape/attachment differs')
    require(volume.get('KmsKeyId') == ebs_key_arn(plan), 'Owned root volume encryption key differs from approval')
    shutdown = client('ec2', 'describe-instance-attribute', '--instance-id', instance_id,
                      '--attribute', 'instanceInitiatedShutdownBehavior')
    require(shutdown['InstanceInitiatedShutdownBehavior']['Value'] == 'terminate',
            'Instance shutdown must terminate, not stop')


def schedule_spec(plan, instance_id, launch_time):
    identity = plan['identity']
    # Schedule one minute early: scheduler's 60-second precision cannot push the
    # requested invocation past the four-hour deadline.
    when = dt.datetime.fromtimestamp(launch_time + 14400 - 60, dt.timezone.utc)
    expression = 'at(' + when.strftime('%Y-%m-%dT%H:%M:%S') + ')'
    target = dict(Arn='arn:aws:scheduler:::aws-sdk:ec2:terminateInstances',
                  RoleArn=plan['iam']['scheduler_role']['arn'],
                  Input=json.dumps({'InstanceIds': [instance_id]}),
                  RetryPolicy={'MaximumEventAgeInSeconds': 60, 'MaximumRetryAttempts': 0})
    return dict(Name=identity['run_id'], GroupName=identity['schedule_group'],
                ScheduleExpression=expression, ScheduleExpressionTimezone='UTC',
                FlexibleTimeWindow={'Mode': 'OFF'}, State='ENABLED',
                ActionAfterCompletion='NONE', Target=target)


def verify_schedule(observed, spec):
    require(all(observed.get(k) == v for k, v in spec.items() if k != 'Target')
            and all(observed.get('Target', {}).get(k) == v for k, v in spec['Target'].items()),
            'Independent termination schedule readback mismatch')


def ordered_schedule_readback(observed, spec, identity):
    """Authenticate the complete configuration and the particular creation."""
    snapshot = {key: value for key, value in observed.items() if key != 'ResponseMetadata'}
    arn = (f"arn:aws:scheduler:{identity['region']}:{identity['account']}:schedule/"
           f"{identity['schedule_group']}/{identity['run_id']}")
    require(set(snapshot) == set(spec) | {'Arn', 'CreationDate', 'LastModificationDate'}
            and snapshot['Arn'] == arn
            and all(snapshot[key] == value for key, value in spec.items()),
            'Ordered schedule complete configuration/readback differs')
    require(isinstance(snapshot['CreationDate'], str) and isinstance(snapshot['LastModificationDate'], str)
            and timestamp(snapshot['CreationDate']) <= timestamp(snapshot['LastModificationDate']),
            'Ordered schedule creation/modification identity is invalid')
    return snapshot


def verify_termination_ticket(ticket, plan, plan_sha, instance_id, now):
    require(ticket['run_id'] == plan['identity']['run_id'] and ticket['plan_sha256'] == plan_sha
            and ticket['instance_id'] == instance_id and ticket['schedule_verified'] is True
            and ticket['deadline'] == ticket['launch_time'] + 14400
            and ticket['launch_time'] <= now < ticket['deadline'] - 1800,
            'Independent termination ticket rejected')
    verify_schedule(ticket['schedule'], schedule_spec(plan, instance_id, ticket['launch_time']))
    return ticket['deadline']


def verify_release(release, plan, plan_sha, instance_id, now):
    require(release['run_id'] == plan['identity']['run_id'] and release['plan_sha256'] == plan_sha
            and release['instance_id'] == instance_id and release['schedule_verified'] is True
            and release['deadline'] == release['launch_time'] + 14400
            and release['launch_time'] <= now < release['deadline'] - 1800,
            'Worker release barrier rejected')
    verify_gpu_observation(release['gpu'], plan)
    verify_schedule(release['schedule'], schedule_spec(plan, instance_id, release['launch_time']))
    return release['deadline'] - 1800


def put_json(client, identity, key, value, path):
    save(path, value)
    require(Path(path).stat().st_size <= 1024**2, 'Serialized control document exceeds 1MiB')
    client('s3api', 'put-object', '--bucket', identity['bucket'], '--key', key,
           '--body', str(path), '--server-side-encryption', 'AES256', '--if-none-match', '*')


def bootstrap_text(plan, plan_sha, root):
    path = Path(root) / 'scripts/aws_eval/grounded_hf_bootstrap.sh'
    require(digest(path) == plan['sources']['scripts/aws_eval/grounded_hf_bootstrap.sh'], 'Staged bootstrap changed')
    values = dict(RUN=plan['identity']['run_id'], BUCKET=plan['identity']['bucket'],
                  REGION=plan['identity']['region'], CODE_PREFIX=plan['identity']['code_prefix'],
                  PLAN_SHA=plan_sha, BUNDLE_SHA=plan['bundle']['sha256'],
                  BUNDLE_BYTES=str(plan['bundle']['bytes']))
    text = path.read_text().replace('# INJECT_BOUND_VALUES', '\n'.join(k + '=' + shlex.quote(v) for k, v in values.items()))
    require(len(text.encode()) <= 16384, 'Raw user-data exceeds EC2 limit')
    return text

def collect_checkpoint(client, plan, plan_sha, state, *, clock=time.time, pause=time.sleep):
    """Finite read-only watcher; never releases or removes termination protection."""
    require(inference_mode(plan), 'Checkpoint collection is inference-only')
    state = Path(state)
    release = json.loads((state / 'release.json').read_text())
    instance_id = release['instance_id']
    work_deadline = verify_release(release, plan, plan_sha, instance_id, clock())
    deadline = min(work_deadline, clock() + 64 * 120)
    keys = plan['workload']['checkpoint_keys']
    receipt_path = state / 'qwen8-checkpoint.json'
    archive_path = state / 'qwen8-checkpoint.tar'
    for attempt in range(64):
        require(clock() < deadline, 'Checkpoint polling deadline exhausted')
        try:
            client('s3api', 'get-object', '--bucket', plan['identity']['bucket'],
                   '--key', keys['receipt'], '--range', 'bytes=0-1048575', str(receipt_path))
        except AWSCallError as exc:
            require(exc.code in ('NoSuchKey', '404'),
                    'Checkpoint retrieval failed; only missing objects may be polled')
            if attempt < 63:
                pause(min(120, remaining(deadline, clock())))
            continue
        require(receipt_path.stat().st_size <= 1024**2, 'Checkpoint receipt too large')
        receipt = json.loads(receipt_path.read_text())
        pin = validate_checkpoint_receipt(receipt, plan, plan_sha, instance_id)
        client('s3api', 'get-object', '--bucket', plan['identity']['bucket'],
               '--key', keys['archive'], '--range', f'bytes=0-{pin["bytes"] - 1}', str(archive_path))
        authenticate_archive(archive_path, dict(bytes=pin['bytes'], sha256=pin['sha256'],
            unpacked_bytes=CHECKPOINT_MAX, files=receipt['files']))
        with tarfile.open(archive_path, 'r:') as archive:
            def document(name):
                member = archive.getmember(name)
                require(member.isfile() and member.size <= 1024**2, 'Checkpoint receipt member too large')
                return json.loads(archive.extractfile(member).read())
            validate_ordered_model_evidence(plan, plan_sha, 'qwen8', receipt['selected_group'],
                {r['path']: r for r in receipt['files']}, document)
            for phase in ('acquire8', 'dev8', 'val8'):
                stage = document(phase + '-receipt.json')
                require(stage['phase'] == phase and stage['state'] == 'complete'
                        and stage['run_id'] == plan['identity']['run_id'] and stage['plan_sha256'] == plan_sha,
                        'Checkpoint phase is incomplete or unbound')
        return dict(state='checkpoint_collected', checkpoint=receipt,
                    checkpoint_path=str(archive_path), semantic_acceptance='not_evaluated')
    raise RuntimeError('No authenticated 8B checkpoint within finite polling allowance')


def validate_ordered_model_evidence(plan, plan_sha, model_key, selected_group, files, document):
    """Bind complete per-profile execution evidence, not semantic correctness."""
    workload = plan['workload']
    profiles = [{**p, 'name': p['name'].replace('qwen8-', model_key + '-', 1)}
                for p in workload['profiles']]
    selection = document(model_key + '-selection.json')
    require(selection['run_id'] == plan['identity']['run_id'] and selection['plan_sha256'] == plan_sha
            and selection['model_key'] == model_key
            and selection['model_sha256'] == object_sha(workload['models'][model_key])
            and selection['profiles_sha256'] == object_sha(profiles)
            and selection['development_inputs_sha256'] == workload['development_inputs']['sha256']
            and selection['development_references_sha256'] == workload['development_references']['sha256']
            and selection['selection_policy'] == workload['selection_policy']
            and selection['state'] == 'complete'
            and selection['selected_group'] in ('nonthinking', 'thinking'),
            'Completed selection is not bound to registered model/development evidence')
    group = selection['selected_group']
    require(selected_group in (None, group), 'Checkpoint selection changed')
    require(selection['selected_profiles'] == [f'{model_key}-{group}-seed{seed}' for seed in (14, 29)],
            'Selected paired profile membership differs')
    development_outputs = []
    for cohort, input_key in (('dev', 'development'), ('val', 'validation')):
        for profile in profiles:
            profile_group = 'thinking' if profile['thinking'] else 'nonthinking'
            if cohort == 'val' and profile_group != group:
                continue
            stem = f'{model_key}-{cohort}-{profile_group}-seed{profile["seed"]}'
            receipt = document(stem + '-receipt.json')
            output = files[stem + '-outputs.jsonl']
            require(receipt['run_id'] == plan['identity']['run_id'] and receipt['plan_sha256'] == plan_sha
                    and receipt['state'] == 'complete' and receipt['model_key'] == model_key
                    and receipt['phase'] == stem
                    and receipt['model_sha256'] == object_sha(workload['models'][model_key])
                    and receipt['profile'] == profile
                    and receipt['input_sha256'] == workload[input_key + '_inputs']['sha256']
                    and receipt['completed_case_ids'] == workload[input_key + '_case_ids']
                    and receipt['question_deadline_epoch'] is None
                    and receipt['output_sha256'] == output['sha256'] and receipt['output_bytes'] == output['bytes'],
                    'Profile completion/model/profile/input/output binding differs')
            if cohort == 'dev':
                development_outputs.append(output)
    require(selection['dev_outputs'] == development_outputs, 'Selection development output hashes differ')
    return group


def validate_ordered_completion(completion, plan, plan_sha, instance_id, archive, manifest):
    require('training_success' not in completion, 'Training completion cannot authorize inference')
    receipt = completion['checkpoint']
    checkpoint = validate_checkpoint_receipt(receipt, plan, plan_sha, instance_id)
    decision = validate_parent_decision(completion['decision'], plan, plan_sha, checkpoint['sha256'])
    selected = {'qwen8': receipt['selected_group']}
    if decision['decision'] == 'compare14':
        selected['qwen14'] = None
    expected = {'parent-decision.json', 'parent_decision-receipt.json'}
    with tarfile.open(archive, 'r:') as stream:
        def document(name):
            member = stream.getmember(name)
            require(member.isfile() and member.size <= 1024**2, 'Ordered receipt exceeds bounded envelope')
            return json.loads(stream.extractfile(member).read())
        require(document('parent-decision.json') == decision, 'Exported parent decision differs')
        for key in selected:
            group = validate_ordered_model_evidence(plan, plan_sha, key, selected[key],
                        {r['path']: r for r in manifest['files']}, document)
            expected.update(ordered_output_members(plan, key, group))
    require({r['path'] for r in manifest['files']} == expected,
            'Native inference export requires exact conditional output membership')
    final_pins = {r['path']: r for r in manifest['files']}
    require(all(final_pins[r['path']] == r for r in receipt['files']),
            'Final export changed authenticated 8B checkpoint outputs')
    phases = ['acquire8', 'dev8', 'val8', 'parent_decision']
    if decision['decision'] == 'compare14':
        phases += ['acquire14', 'dev14', 'val14']
    require([stage['phase'] for stage in completion['stages']] == phases
            and all(stage['state'] == 'complete' and stage['run_id'] == plan['identity']['run_id']
                    and stage['plan_sha256'] == plan_sha for stage in completion['stages']),
            'Incomplete ordered native stage receipt')



def collect_export(client, plan, plan_sha, state, instance_id):
    """Retrieve bounded evidence only; native completion is not quality acceptance."""
    state = Path(state)
    identity = plan['identity']
    completion_path = state / 'completion.json'
    client('s3api', 'get-object', '--bucket', identity['bucket'],
           '--key', identity['output_prefix'] + 'completion.json', '--range', 'bytes=0-1048575',
           str(completion_path))
    completion = json.loads(completion_path.read_text())
    require(completion['run_id'] == identity['run_id'] and completion['instance_id'] == instance_id
            and completion['plan_sha256'] == plan_sha, 'Completion ownership differs')
    pin = completion['export']
    require(type(pin['bytes']) is int and 0 < pin['bytes'] <= MAX_EXPORT
            and SHA.fullmatch(pin['sha256']), 'Export envelope differs')
    path = state / 'export.tar'
    client('s3api', 'get-object', '--bucket', identity['bucket'],
           '--key', identity['output_prefix'] + 'export.tar', '--range', f'bytes=0-{pin["bytes"] - 1}', str(path))
    require(path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'], 'Export bytes differ')
    with tarfile.open(path, 'r:') as archive:
        member = archive.getmember('export-manifest.json')
        require(member.isfile() and member.size <= 8 * 1024**2, 'Export manifest is unsafe')
        with archive.extractfile(member) as stream:
            raw = stream.read()
        manifest = json.loads(raw)
    require(manifest['run_id'] == identity['run_id'] and manifest['plan_sha256'] == plan_sha
            and {r['path'] for r in manifest['files']} <= set(plan['output_allowlist']),
            'Export membership/identity differs')
    rows = manifest['files'] + [dict(path='export-manifest.json', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())]
    authenticate_archive(path, dict(bytes=pin['bytes'], sha256=pin['sha256'],
        unpacked_bytes=LIMITS['output_bytes'], files=rows))
    native = completion[success_field(plan)] is True and not completion['stop_errors']
    if inference_mode(plan):
        require('training_success' not in completion, 'Training completion cannot authorize inference')
        if native:
            for key, field in (('receipt', 'checkpoint'), ('decision', 'decision')):
                proof_path = state / ('final-' + field + '.json')
                client('s3api', 'get-object', '--bucket', identity['bucket'],
                    '--key', plan['workload']['checkpoint_keys'][key],
                    '--range', 'bytes=0-1048575', str(proof_path))
                require(proof_path.stat().st_size <= 1024**2
                        and json.loads(proof_path.read_text()) == completion[field],
                        'Final completion differs from independently retrieved checkpoint/parent decision')
            validate_ordered_completion(completion, plan, plan_sha, instance_id, path, manifest)
    else:
        require(not native or [s['phase'] for s in completion['stages']]
                == ['capacity', 'baseline', 'train', 'adapter'], 'Incomplete native stage receipt')
    if native:
        bound = json.loads((state / 'release.json').read_text())
        require(bound['run_id'] == identity['run_id'] and bound['instance_id'] == instance_id
                and bound['plan_sha256'] == plan_sha, 'Persisted GPU release ownership differs')
        for stage in [completion['runtime_unit'], *completion['stages']]:
            if not inference_mode(plan) or 'gpu' in stage:
                verify_gpu_observation(stage['gpu'], plan, bound['gpu'])
    return dict(native_pipeline_success=native, semantic_acceptance='not_evaluated',
                completion=completion, export_sha256=pin['sha256'])


def cleanup_schedule(client, identity, state, cleanup, *, plan=None, plan_sha=None):
    require(cleanup['cleanup_state'] == 'complete' and cleanup.get('instance_terminated') is True
            and cleanup.get('root_volume_ids') and not cleanup.get('remaining_volumes')
            and set(cleanup['root_volume_ids']) <= set(cleanup.get('recorded_volume_ids', []))
            and set(cleanup.get('absent_volume_ids', [])) == set(cleanup.get('recorded_volume_ids', [])),
            'No termination proof; retain independent protection')
    path = Path(state) / 'schedule-intent.json'
    if not path.exists():
        require(plan is None or not inference_mode(plan),
                'Missing ordered schedule intent; retain independent protection')
        return
    spec = json.loads(path.read_text())
    require(spec['Name'] == identity['run_id'] and spec['GroupName'] == identity['schedule_group']
            and json.loads(spec['Target']['Input']) == {'InstanceIds': [cleanup['instance_id']]},
            'Schedule cleanup ownership conflict')
    snapshot = None
    if plan is not None and inference_mode(plan):
        require(identity == plan['identity'] and SHA.fullmatch(plan_sha or ''),
                'Ordered schedule cleanup requires exact plan identity')
        consumed = json.loads((Path(state) / 'approval-consumed.json').read_text())
        require(consumed == dict(run_id=identity['run_id'], plan_sha256=plan_sha),
                'Ordered schedule cleanup approval identity differs')
        created = json.loads((Path(state) / 'schedule-created.json').read_text())
        persisted = json.loads((Path(state) / 'schedule-readback.json').read_text())
        snapshot = ordered_schedule_readback(persisted, spec, identity)
        require(set(created) <= {'ScheduleArn', 'ResponseMetadata'}
                and created.get('ScheduleArn') == snapshot['Arn'],
                'Ordered schedule successful creation provenance differs')
    try:
        observed = client('scheduler', 'get-schedule', '--name', identity['run_id'], '--group-name', identity['schedule_group'])
    except AWSCallError as exc:
        if exc.code == 'ResourceNotFoundException':
            return
        raise
    if snapshot is not None:
        require(ordered_schedule_readback(observed, spec, identity) == snapshot,
                'Ordered schedule was replaced or modified; retain independent protection')
    else:
        verify_schedule(observed, spec)
    client('scheduler', 'delete-schedule', '--name', identity['run_id'], '--group-name', identity['schedule_group'])


def require_no_dispatch_evidence(state):
    """Absence check only; parent separately authenticates any historical failure."""
    root = Path(state)
    names = ('dispatch-authorization.json', 'remote-approval-intent.json', 'launch-dispatched.json',
             'volume-identity.json', 'schedule-intent.json', 'schedule-created.json',
             'schedule-readback.json', 'termination.json', 'release-intent.json', 'release.json')
    for name in names:
        if os.path.lexists(root / name):
            raise FileExistsError('Possible dispatch evidence; continuation forbidden: ' + str(root / name))


def launch(plan, approval, plan_sha, root, state, client, *, dispatch_state=None,
           clock=time.time, pause=time.sleep):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    identity = plan['identity']
    ordered = inference_mode(plan)
    dispatch_root = Path(dispatch_state) if dispatch_state is not None else state
    if ordered:
        require(dispatch_root.is_dir() and state.resolve().is_relative_to(dispatch_root.resolve()),
                'Lifecycle generation must remain inside its shared dispatch ledger')
        require_no_dispatch_evidence(dispatch_root)
        if state != dispatch_root:
            require_no_dispatch_evidence(state)
        if os.path.lexists(state / 'controller-result.json'):
            raise FileExistsError('Preserve prior controller result; use an explicitly admitted generation')
        exclusive_intent(state / 'controller-invocation.json',
                         dict(run_id=identity['run_id'], plan_sha256=plan_sha, at=clock()))
    else:
        exclusive_intent(state / 'approval-consumed.json', dict(run_id=identity['run_id'], plan_sha256=plan_sha))
    expected = None
    dispatched = False
    schedule_created = False
    result = {success_field(plan): False, 'state': 'admission', 'phase': 'readonly_admission',
              'dispatch_attempted': False, 'remote_intent_attempted': False}
    evidence_path = state / 'volume-identity.json'
    evidence = volume_identity(identity, plan_sha, plan['ami']['RootDeviceName'])
    try:
        validate_plan(plan, approval, plan_sha, now=clock())
        authenticate_archive(Path(root) / plan['bundle']['local_path'], plan['bundle'])
        validate_data(root, plan)
        validate_cost_admission(root, plan)
        bootstrap_admission = validate_bootstrap_admission(root, plan)
        for path, sha in plan['sources'].items():
            require(digest(Path(root) / path) == sha, 'Staged executable differs: ' + path)
        result['phase'] = 'readonly_preflight'
        result['preflight'] = live_preflight(plan, client)
        require(not discover(client, identity), 'Fresh run identity already has instances')
        user_data = bootstrap_text(plan, plan_sha, root)
        result['phase'] = 'readonly_staged_objects'
        # Parent has staged immutable plan and archive. Read head checks prevent
        # accidental launch against missing/different object contents.
        for suffix, sha, size in (('plan.json', plan_sha, None),
                                  ('bundle.tar', plan['bundle']['sha256'], plan['bundle']['bytes'])):
            head = client('s3api', 'head-object', '--bucket', identity['bucket'],
                          '--key', identity['code_prefix'] + suffix)
            require((head['ContentLength'] == size if size is not None else 0 < head['ContentLength'] <= 2 * 1024**2)
                    and head['Metadata'].get('sha256') == sha, 'Staged S3 object metadata differs')
        request = dict(ImageId=plan['ami']['ImageId'], InstanceType='g6e.2xlarge', MinCount=1, MaxCount=1,
            ClientToken=identity['run_id'], IamInstanceProfile={'Name': identity['instance_profile']},
            InstanceInitiatedShutdownBehavior='terminate', DisableApiTermination=False,
            MetadataOptions={'HttpTokens': 'required', 'HttpEndpoint': 'enabled', 'HttpPutResponseHopLimit': 1},
            NetworkInterfaces=[dict(DeviceIndex=0, SubnetId=plan['network']['subnet_id'],
                Groups=[plan['network']['security_group_id']], AssociatePublicIpAddress=True, DeleteOnTermination=True)],
            BlockDeviceMappings=[dict(DeviceName=plan['ami']['RootDeviceName'], Ebs=dict(
                VolumeSize=120, VolumeType='gp3', Iops=3000, Throughput=125, Encrypted=True,
                KmsKeyId=ebs_key_arn(plan), DeleteOnTermination=True))],
            TagSpecifications=[dict(ResourceType=kind, Tags=[{'Key': k, 'Value': v} for k, v in identity['tags'].items()])
                               for kind in ('instance', 'volume')], UserData=user_data)
        if ordered:
            dispatch_record = dict(run_id=identity['run_id'], plan_sha256=plan_sha,
                                   request_sha256=object_sha(request), at=clock())
            exclusive_intent(dispatch_root / 'dispatch-authorization.json', dispatch_record)
            exclusive_intent(state / 'approval-consumed.json',
                             dict(run_id=identity['run_id'], plan_sha256=plan_sha))
        result.update(phase='remote_launch_intent', remote_intent_attempted=True)
        save(state / 'controller-result.json', result)
        put_json(client, identity, identity['code_prefix'] + 'approval-consumed.json',
                 dict(run_id=identity['run_id'], plan_sha256=plan_sha), state / 'remote-approval-intent.json')
        if not ordered:
            dispatch_record = dict(request_sha256=object_sha(request), at=clock())
        exclusive_intent(state / 'launch-dispatched.json', dispatch_record)
        dispatched = True
        result.update(phase='run_instances', dispatch_attempted=True)
        save(state / 'controller-result.json', result)
        response = client('ec2', 'run-instances', '--cli-input-json', json.dumps(request))
        require(len(response['Instances']) == 1, 'Ambiguous launch response')
        expected = owned(response['Instances'][0], identity)
        observe_volume_identity(client, identity, response['Instances'][0], evidence, evidence_path)
        rows = discover(client, identity)
        require(len(rows) == 1 and owned(rows[0], identity, expected) == expected, 'Launch not exactly observable')
        observe_volume_identity(client, identity, rows[0], evidence, evidence_path)
        launch_time = timestamp(rows[0]['LaunchTime'])
        deadline = launch_time + 14400
        require(clock() < deadline - 1800, 'Launch visibility too late')
        result.update(instance_id=expected, launch_time=launch_time, deadline=deadline,
                      state='scheduled', phase='scheduler_setup')
        spec = schedule_spec(plan, expected, launch_time)
        exclusive_intent(state / 'schedule-intent.json', spec)
        # Treat a lost create-schedule response as potentially created. Cleanup
        # keeps protection until observed termination, never retries creation.
        schedule_created = True
        created = client('scheduler', 'create-schedule', '--cli-input-json', json.dumps(spec), '--client-token', identity['run_id'])
        if inference_mode(plan):
            exclusive_intent(state / 'schedule-created.json', created)
        observed = client('scheduler', 'get-schedule', '--name', identity['run_id'], '--group-name', identity['schedule_group'])
        if inference_mode(plan):
            exclusive_intent(state / 'schedule-readback.json', observed)
            snapshot = ordered_schedule_readback(observed, spec, identity)
            require(set(created) <= {'ScheduleArn', 'ResponseMetadata'}
                    and created.get('ScheduleArn') == snapshot['Arn'],
                    'Ordered schedule successful creation provenance differs')
        else:
            verify_schedule(observed, spec)
        ticket = dict(run_id=identity['run_id'], plan_sha256=plan_sha, instance_id=expected,
                      launch_time=launch_time, deadline=deadline, schedule_verified=True, schedule=spec)
        put_json(client, identity, identity['code_prefix'] + 'termination.json', ticket, state / 'termination.json')
        readiness = None
        # Finite polls, not model/launch retries; GET requests are pre-budgeted.
        for attempt in range(120):
            require(clock() < min(launch_time + 600, deadline - 1800), 'Bootstrap readiness deadline exhausted')
            try:
                path = state / 'ready-download.json'
                client('s3api', 'get-object', '--bucket', identity['bucket'],
                       '--key', identity['output_prefix'] + 'ready.json', '--range', 'bytes=0-1048575', str(path))
                readiness = json.loads(path.read_text())
                break
            except Exception:
                pause(5)
        require(readiness is not None and readiness['instance_id'] == expected
                and readiness['plan_sha256'] == plan_sha and readiness['run_id'] == identity['run_id'],
                'No authenticated exact-instance bootstrap readiness')
        verify_prerequisites(readiness.get('prerequisites'), bootstrap_admission)
        rows = discover(client, identity)
        require(len(rows) == 1 and owned(rows[0], identity, expected) == expected,
                'Exact instance visibility lost before release')
        observe_volume_identity(client, identity, rows[0], evidence, evidence_path)
        verify_launched_resources(client, plan, rows[0])
        release = dict(run_id=identity['run_id'], plan_sha256=plan_sha, instance_id=expected,
                       launch_time=launch_time, deadline=deadline, schedule_verified=True,
                       schedule=spec, gpu=readiness['gpu'])
        verify_release(release, plan, plan_sha, expected, clock())
        exclusive_intent(state / 'release-intent.json', release)
        put_json(client, identity, identity['code_prefix'] + 'release.json', release, state / 'release.json')
        result['state'] = 'released'
        # Worker's completion report is evidence, not semantic acceptance.
        while clock() < deadline - 120:
            rows = discover(client, identity)
            require(len(rows) == 1, 'Instance visibility lost after release')
            owned(rows[0], identity, expected)
            if any(mapping not in evidence['volumes'].get(mapping.get('Ebs', {}).get('VolumeId'), {})
                   .get('instance_mappings', []) for mapping in rows[0].get('BlockDeviceMappings', [])):
                observe_volume_identity(client, identity, rows[0], evidence, evidence_path)
            if rows[0]['State']['Name'] in ('shutting-down', 'terminated'):
                break
            pause(min(30, deadline - 120 - clock()))
    except Exception as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        if dispatched:
            cleanup_deadline = clock() + plan['cost']['termination_grace_seconds']
            if inference_mode(plan):
                cleanup_deadline = min(cleanup_deadline, cleanup_cutoff(plan, plan_sha, state))
            result['cleanup'] = reconcile(client, identity, expected, cleanup_deadline,
                evidence=evidence, evidence_path=evidence_path, clock=clock, pause=pause)
            if result['cleanup']['cleanup_state'] == 'complete' and schedule_created:
                try:
                    cleanup_schedule(client, identity, state, result['cleanup'], plan=plan, plan_sha=plan_sha)
                except Exception as exc:
                    result['schedule_cleanup_error'] = str(exc)
            if result['cleanup']['cleanup_state'] == 'complete' and result['state'] == 'released':
                try:
                    result['evidence'] = collect_export(client, plan, plan_sha, state, result['cleanup']['instance_id'])
                    result[success_field(plan)] = result['evidence']['native_pipeline_success']
                    result['state'] = 'complete' if result[success_field(plan)] else 'native_failed'
                except Exception as exc:
                    result['evidence_error'] = str(exc)
        save(state / 'controller-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'preflight', 'launch', 'reconcile', 'checkpoint'))
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    plan = read_bound(args.plan, args.plan_sha256)
    client = Client(plan['identity'], request_limit=plan['cost']['control_requests'])
    if args.action == 'checkpoint':
        client.deadline = time.time() + 64 * 120
        client.request_limit = 65
        result = collect_checkpoint(client, plan, args.plan_sha256, args.state)
    elif args.action == 'reconcile':
        intent = json.loads((args.state / 'approval-consumed.json').read_text())
        require(intent == dict(run_id=plan['identity']['run_id'], plan_sha256=args.plan_sha256),
                'Reconciliation requires exact local consumed approval')
        evidence = load_volume_identity(args.state, plan, args.plan_sha256)
        cleanup_deadline = time.time() + plan['cost']['termination_grace_seconds']
        if inference_mode(plan):
            cleanup_deadline = min(cleanup_deadline, cleanup_cutoff(plan, args.plan_sha256, args.state))
        result = reconcile(client, plan['identity'], evidence['instance_id'], cleanup_deadline,
                           evidence=evidence, evidence_path=args.state / 'volume-identity.json')
        save(args.state / 'reconcile-result.json', result)
        if result['cleanup_state'] == 'complete':
            cleanup_schedule(client, plan['identity'], args.state, result, plan=plan, plan_sha=args.plan_sha256)
    else:
        require(args.approval and args.approval_sha256, 'Explicit SHA-bound new approval required')
        approval = read_bound(args.approval, args.approval_sha256)
        if args.action == 'launch':
            result = launch(plan, approval, args.plan_sha256, args.root, args.state, client)
        else:
            result = validate_plan(plan, approval, args.plan_sha256)
            authenticate_archive(args.root / plan['bundle']['local_path'], plan['bundle'])
            validate_data(args.root, plan)
            validate_cost_admission(args.root, plan)
            validate_bootstrap_admission(args.root, plan)
            if args.action == 'preflight':
                result['preflight'] = live_preflight(plan, client)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if (result.get('state') in ('failed', 'native_failed') or result.get('cleanup', {}).get('cleanup_state') == 'unresolved'
            or result.get('cleanup_state') == 'unresolved' or result.get('evidence_error')
            or args.action == 'launch' and not result.get(success_field(plan))):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
