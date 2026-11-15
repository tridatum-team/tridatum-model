"""Fresh600 CUDA contract: one final endpoint, one paired development evaluation.

Completion means numerical acceptance and reaped owned processes, not quality
qualification. The separate controller must still prove EC2/EBS cleanup.
"""
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SCHEMA = 'trid-fresh-eight-gpu-experiment-v1'
APPROVAL_SCHEMA = 'trid-fresh-eight-gpu-approval-v1'
RUN_ID = 'mois-fresh-eight-gpu-20261115-094150'
EXPERIMENT_ID = 'mois-fresh-eight-gpu-20261115T094150Z'
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
AUTHORIZATION_SHA = '7a1f8ee7a943ded4fd231250ef79365d7065c856e8033bd3d7ded66a870d55e5'
RECIPE = dict(rows=600, epochs=1, steps=150, batch_size=4, micro_batch_size=1,
              rank=8, alpha=16, dropout=0.0, lr=1e-5, seed=14,
              context_tokens=3072, answer_tokens=768, checkpoint_every=50, valid_every=0)
PHASE_SECONDS = dict(acquire=900, capacity=900, train=5400, evaluate=3600)
GENERATION = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
                  max_new_tokens=768, context_tokens=4096, dtype='bfloat16', backend='transformers-cuda')
CHECKPOINT_STEPS = (50, 100, 150)
SOURCE_NAMES = {'scripts/aws_eval/fresh_eight_experiment.py', 'scripts/aws_eval/fresh_eight_worker.py',
    'scripts/local_mps_training.py', 'scripts/local_eight_candidate.py', 'scripts/evaluate_local_eight.py'}
OPTIONAL_OUTPUT_MEMBERS = {'evaluation/attempts.jsonl', 'evaluation/failure.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_fresh(plan):
    return plan.get('schema') == SCHEMA


def pin(path, root=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or unsafe pinned file: ' + str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(root).as_posix() if root is not None else str(path),
                bytes=path.stat().st_size, sha256=digest)


def bound(root, record):
    name = PurePosixPath(record['path'])
    require(not name.is_absolute() and '..' not in name.parts, 'Unsafe relative artifact pin')
    path = Path(root) / name
    require(pin(path, root) == record, 'Pinned bytes differ: ' + str(name))
    return path


def load(root, record):
    return json.loads(bound(root, record).read_bytes())


def output_members(plan=None):
    names = {'runtime.json', 'runtime.log', 'capacity.log', 'train.log', 'evaluate.log',
        'plan.json', 'candidate.json', 'model-acquisition.json',
        'capacity/capacity-worker.json', 'capacity/supervisor-result.json',
        'train/training-worker.json', 'train/supervisor-result.json', 'train/training.json',
        'train/checkpoints.json', 'evaluation/supervisor-result.json'}
    names.update('evaluation/' + n for n in ('run-manifest.json', 'decoder-profile.json',
        'prompt-preflight.json', 'loaded-runtime.json', 'completion.json',
        'base-outputs.jsonl', 'candidate-outputs.jsonl'))
    for prefix in ['train', *[f'train/checkpoints/step-{s:06d}' for s in CHECKPOINT_STEPS]]:
        names.update(prefix + '/adapter/' + n for n in ('adapter_config.json', 'adapter_model.safetensors', 'README.md'))
        names.add(prefix + '/training.json')
    return names | OPTIONAL_OUTPUT_MEMBERS


def validate_workload(plan, bundled):
    require(is_fresh(plan) and plan['identity']['run_id'] == RUN_ID, 'Fresh GPU identity differs')
    require(plan['recipe'] == RECIPE and plan['phase_seconds'] == PHASE_SECONDS, 'Frozen fresh600 recipe differs')
    require(plan['model']['id'] == MODEL_ID and plan['model']['revision'] == MODEL_REVISION, 'Fresh base differs')
    require(set(plan['data']) == {'policy', 'implementation_contract', 'manifest', 'frozen_dataset', 'train'},
            'Fresh data closure differs')
    w = plan['workload']
    require(set(w) == {'evaluation_inputs', 'review_protocol', 'source_manifest', 'generation'},
            'No final-input or secondary evaluation channel is allowed')
    require(w['generation'] == GENERATION, 'Fresh effective generation request differs')
    records = [*plan['data'].values(), *(w[k] for k in ('evaluation_inputs', 'review_protocol', 'source_manifest')),
               plan['authorization']]
    require(all(set(p) == {'path', 'bytes', 'sha256'} and bundled.get(p['path']) == p for p in records),
            'Fresh authenticated input closure differs')
    require(plan['authorization']['sha256'] == AUTHORIZATION_SHA, 'New USD10 authorization differs')
    require(plan['data']['policy']['sha256'] == '79f30890da65f9a12605dae79dfae20c8e7887833c49bfde4d1e621228dd92a4',
            'Registered CUDA execution policy bytes differ')
    require(SOURCE_NAMES <= set(plan['sources']), 'Fresh executable source closure incomplete')
    require(set(plan['output_allowlist']) == output_members() and len(plan['output_allowlist']) == len(output_members()),
            'Fresh output membership differs')
    require(not any('reference' in n.lower() or 'final-input' in n.lower() for n in bundled),
            'Private references/final inputs must not be delivered')


def validate_data(root, plan):
    validate_workload(plan, {p['path']: p for p in plan['bundle']['files']})
    data, w = plan['data'], plan['workload']
    require(data['implementation_contract']['sha256'] == '6ef6b5d1b381daa416c761b0b7f6d66b8bbc5826415747462d8bcf79bea7ed1c',
            'Immutable implementation contract bytes differ')
    contract = load(root, data['implementation_contract'])
    require(contract['schema'] == 'trid-fresh-eight-gpu-implementation-contract-v1'
            and contract['run_id'] == EXPERIMENT_ID and contract['authorization'] == plan['authorization'], 'Immutable intent differs')
    policy = load(root, data['policy'])
    expected = copy.deepcopy(contract['intent'])
    expected['model']['device'] = 'cuda'
    expected['evaluation']['generation']['backend'] = 'transformers-cuda'
    require(all(policy[k] == expected[k] for k in ('model', 'data', 'training', 'capacity', 'evaluation')),
            'CUDA policy changed frozen training/evaluation intent')
    require(data['manifest'] == expected['data']['training_manifest']
            and data['frozen_dataset'] == expected['data']['frozen_dataset']
            and w['evaluation_inputs'] == expected['data']['evaluation_inputs']
            and w['review_protocol'] == expected['evaluation']['review_protocol'], 'Frozen input lineage differs')
    bound(root, policy['model']['acquisition_manifest'])
    auth = load(root, plan['authorization'])
    require(auth['additional_budget_usd'] == '10.00' and auth['new_paid_execution_limit'] == 1
            and auth['launch_started'] is False, 'New authorization is not unused USD10 grant')
    manifest = load(root, data['manifest'])
    require(manifest['rows'] == 600 and manifest['valid'] == [] and len(manifest['train']) == 1, 'Fresh600 manifest differs')
    entry = manifest['train'][0]
    require(entry['adapter'] == 'grounded_messages' and entry['epochs'] == 1
            and entry['sha256'] == data['train']['sha256']
            and (PurePosixPath(data['manifest']['path']).parent / entry['path']).as_posix() == data['train']['path'],
            'Exactly-once source file differs')
    rows = [json.loads(line) for line in bound(root, data['train']).read_text().splitlines()]
    require(len(rows) == 600 and len({r['id'] for r in rows}) == 600, 'Training identities duplicate or missing')
    evaluation = load(root, w['evaluation_inputs'])
    require(isinstance(evaluation, list) and len(evaluation) == 200
            and len({r['id'] for r in evaluation}) == 200, 'Fixed200 evaluation differs')
    bound(root, data['frozen_dataset']); bound(root, w['review_protocol'])
    sources = load(root, w['source_manifest'])
    records = sources['files']
    require(isinstance(records, list) and {p['path'] for p in records} == set(plan['sources']), 'Source manifest closure differs')
    for record in records:
        bound(root, record)
        require(record['sha256'] == plan['sources'][record['path']], 'Plan/source manifest seal differs')
    return dict(policy=policy, manifest=manifest, rows=rows, evaluation_inputs=evaluation, sources=records)


def dtype_inventory(inventory):
    for group, dtype in (('base', 'torch.bfloat16'), ('adapter', 'torch.float32')):
        rows = inventory[group]
        require(rows and all(key.startswith(dtype + '@cuda') and value['tensors'] > 0
                and value['elements'] > 0 and value['bytes'] == value['elements'] * (2 if group == 'base' else 4)
                for key, value in rows.items()), 'Actual base/adapter dtype inventory differs')


def numerical_history(history, count, loss_key):
    require(len(history) == count, 'Incomplete numerical update history')
    for step, row in enumerate(history, 1):
        health = row['numerical_health']
        require(row['step'] == step and math.isfinite(row[loss_key])
                and math.isfinite(health['gradient_norm']) and health['gradient_norm'] > 0
                and health['changed'] is True and health['changed_tensor']
                and health['resident_moments'] is (step > 1), 'Failed optimizer/gradient invariant')
        require(health['gradients'] and health['optimizer'], 'Missing actual dtype/byte inventories')
        require(all(key.startswith('torch.float32@cuda') and value['bytes'] > 0
                    for key, value in health['gradients'].items())
                and any(key.startswith('torch.float32@cuda') and value['bytes'] > 0
                        for key, value in health['optimizer'].items()), 'Gradient/Adam CUDA FP32 inventory differs')
        applied = 1e-5 * ((step / 7) if step <= 7 else 0.5 * (1 + math.cos(math.pi * (step - 1 - 7) / 143)))
        following = 1e-5 * (((step + 1) / 7) if step < 7 else 0.5 * (1 + math.cos(math.pi * (step - 7) / 143)))
        require(math.isclose(row['applied_lr'], applied, rel_tol=1e-12, abs_tol=1e-15)
                and math.isclose(row['next_lr'], following, rel_tol=1e-12, abs_tol=1e-15), 'Scheduler application order differs')


def observer_provenance(root, plan, record):
    files = record['actual_source_files']
    bundled = {p['path']: p for p in plan['bundle']['files']}
    require(files and len(files) == len(plan['sources'])
            and {p['path'] for p in files} == set(plan['sources'])
            and all(p == bundled[p['path']] and p['sha256'] == plan['sources'][p['path']] for p in files),
            'Observed executable closure differs')
    runtime = record['runtime']
    require(runtime['device'] == 'cuda' and runtime['backend'] == 'transformers-cuda'
            and runtime['run_id'] == plan['identity']['run_id']
            and runtime['plan_sha256'] == pin(Path(root) / 'plan.json', root)['sha256'],
            'Observed runtime plan/run provenance differs')


def validate_capacity(root, plan):
    result = json.loads((Path(root) / 'capacity/capacity-worker.json').read_bytes())
    require(result['schema'] == 'trid-fresh-eight-cuda-capacity-v1' and result['status'] == 'complete'
            and result['policy'] == plan['data']['policy'] and result['source_manifest'] == plan['workload']['source_manifest']
            and result['runtime']['device'] == 'cuda', 'Capacity provenance differs')
    observer_provenance(root, plan, result)
    require(all(result[k] == v for k, v in dict(train_rows=600, dropped=0, sequence_tokens=3072,
        supervised_tokens=768, masked_prompt_tokens=2304, optimizer_updates=2, microbatches_per_update=4).items()),
        'Capacity envelope differs')
    require(all(result[k] is True for k in ('includes_eos', 'reload_exact', 'released_before_reload',
        'scratch_adapter_never_nominated')) and math.isfinite(result['reload_loss']), 'Capacity reload proof differs')
    dtype_inventory(result['dtype_inventory'])
    dtype_inventory(result['reload_dtype_inventory'])
    numerical_history(result['history'], 2, 'loss')
    require(all(len(row['microbatch_losses']) == 4 and all(math.isfinite(v) for v in row['microbatch_losses'])
                for row in result['history']), 'Capacity accumulation evidence incomplete')
    checks = result['history'][1]['resident_backward_checks']
    require(len(checks) == 4 and all(check['resident_moments'] is True and
            math.isfinite(check['gradient_norm']) and check['gradient_norm'] > 0 for check in checks),
            'Second full-envelope backward lacks resident Adam state')
    return result


def validate_supervision(root, phase, plan):
    folder = 'evaluation' if phase == 'evaluate' else phase
    receipt = json.loads((Path(root) / folder / 'supervisor-result.json').read_bytes())
    require(receipt['schema'] == 'trid-fresh-eight-cuda-supervision-v1' and receipt['phase'] == phase
            and receipt['status'] == 'accepted' and receipt['cleanup_verified'] is True
            and receipt['policy'] == plan['data']['policy']
            and receipt['source_manifest'] == plan['workload']['source_manifest']
            and receipt['run_id'] == plan['identity']['run_id'], 'Owned-process acceptance/provenance missing')
    expected_path = {'capacity': 'capacity/capacity-worker.json', 'train': 'train/training-worker.json',
                     'evaluate': 'evaluation/completion.json'}[phase]
    require(receipt['worker_result']['path'] == expected_path, 'Supervisor accepted a different worker artifact')
    require(receipt['unit']['state'] == 'complete' and receipt['unit']['cleanup_verified'] is True,
            'Supervisor did not prove cgroup cleanup')
    unit = receipt['unit']
    stopped = unit['stopped_state']
    plan_pin = pin(Path(root) / 'plan.json', root)
    require(load(root, plan_pin) == plan and receipt['plan_sha256'] == plan_pin['sha256']
            and unit['plan_sha256'] == plan_pin['sha256'] and unit['run_id'] == receipt['run_id']
            and unit['phase'] == phase and unit['invocation'] and unit['instance_id']
            and (stopped.get('LoadState') == 'not-found' or stopped.get('ActiveState') in ('inactive', 'failed')),
            'Supervisor plan/instance/stopped-state proof differs')
    load(root, receipt['worker_result'])
    return receipt


def validate_training(root, plan):
    root = Path(root)
    result = json.loads((root / 'train/training-worker.json').read_bytes())
    require(result['schema'] == 'trid-fresh-eight-cuda-training-v1' and result['status'] == 'complete'
            and result['coverage_complete'] is True and result['policy'] == plan['data']['policy']
            and result['source_manifest'] == plan['workload']['source_manifest'], 'Training worker provenance differs')
    require(result['report']['path'] == 'train/training.json', 'Worker accepted a different training report')
    report = load(root, result['report'])
    require(report['status'] == 'complete' and report['steps'] == 150 and report['train_rows'] == 600
            and report['valid_rows'] == 0 and report['dropped_over_max_tokens'] == 0 and report['valid_dropped'] == 0
            and report['coverage_complete'] is True and report['device'] == 'cuda'
            and report['base_weights'] == dict(representation='hf-safetensors', dtype='torch.bfloat16')
            and report['init_adapter'] is None and report['manifest_sha256'] == plan['data']['manifest']['sha256'],
            'Training final150/precision/data acceptance differs')
    dtype_inventory(report['dtype_inventory'])
    require(report['policy'] == plan['data']['policy'] and report['runtime']['device'] == 'cuda',
            'Training observer provenance differs')
    observer_provenance(root, plan, report)
    coverage = report['coverage']
    require(len(coverage) == 600 and len({(r['file'], r['source_sha256'], r['source_line'], r['id']) for r in coverage}) == 600
            and {r['source_sha256'] for r in coverage} == {plan['data']['train']['sha256']}
            and {r['source_line'] for r in coverage} == set(range(1, 601)), 'All600 exact-once coverage missing')
    numerical_history(report['history'], 150, 'train_loss')
    ledger = json.loads((root / 'train/checkpoints.json').read_bytes())
    require(ledger['run_status'] == 'complete' and [r['step'] for r in ledger['checkpoints']] == list(CHECKPOINT_STEPS)
            and ledger['last_step']['step'] == 150 and ledger['last_step']['equals_last_checkpoint'] is True
            and ledger['selection']['candidate_step'] == 150, 'Final-only checkpoint ledger differs')
    require(report['last_step'] == ledger['last_step'] and report['checkpoints'] == ledger['checkpoints'],
            'Final report/checkpoint ledger diverged')
    for row in [*ledger['checkpoints'], ledger['last_step']]:
        folder = root / ('train/adapter' if row is ledger['last_step'] else f'train/checkpoints/step-{row["step"]:06d}/adapter')
        require(row['path'] == folder.relative_to(root).as_posix(), 'Portable checkpoint endpoint differs')
        for name, field in [('adapter_config.json', 'adapter_config_sha256'), ('adapter_model.safetensors', 'adapter_weights_sha256')]:
            require(pin(folder / name)['sha256'] == row[field], 'Checkpoint bytes differ from ledger')
    return report


def validate_acquisition(root, plan):
    acquisition = json.loads((Path(root) / 'model-acquisition.json').read_bytes())
    require(acquisition['status'] == 'authenticated' and acquisition['model_id'] == MODEL_ID
            and acquisition['revision'] == MODEL_REVISION
            and acquisition['files'] == [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in plan['model']['files']],
            'Actual base acquisition differs')
    return acquisition


def validate_completion(root, plan):
    root = Path(root)
    required = output_members() - OPTIONAL_OUTPUT_MEMBERS - {n for n in output_members() if n.endswith(('.log', '/README.md'))}
    require(all((root / n).is_file() and not (root / n).is_symlink() for n in required), 'Successful export lacks mandatory artifacts')
    validate_acquisition(root, plan)
    validate_capacity(root, plan); validate_training(root, plan)
    for phase in ('capacity', 'train', 'evaluate'):
        validate_supervision(root, phase, plan)
    return validate_evaluation(root, plan)


def validate_evaluation(root, plan):
    root = Path(root)
    completed = json.loads((root / 'evaluation/completion.json').read_bytes())
    require(completed['status'] == 'complete' and completed['rows_per_arm'] == 200
            and not (root / 'evaluation/failure.json').exists(), 'Paired evaluation incomplete')
    manifest = json.loads((root / 'evaluation/run-manifest.json').read_bytes())
    require(manifest['generation'] == GENERATION and manifest['grading_access'] is False
            and manifest['retries'] is False and manifest['resume'] is False
            and manifest['arm_order'] == ['base', 'candidate']
            and len(manifest['selected_ids']) == 200 and len(set(manifest['selected_ids'])) == 200,
            'Fresh paired evaluation contract differs')
    loaded = json.loads((root / 'evaluation/loaded-runtime.json').read_bytes())
    config, effective = loaded['generation_config'], loaded['generation_policy']
    config_sha = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(config['do_sample'] is False and config['num_beams'] == 1 and config['max_new_tokens'] == 768
            and effective['generation_mode'] == 'greedy_search'
            and effective['dispatch_kwargs'] == dict(use_model_defaults=False, do_sample=False, num_beams=1)
            and effective['effective_config_sha256'] == config_sha, 'Effective greedy decoder proof differs')
    for arm in ('base', 'candidate'):
        rows = [json.loads(line) for line in (root / f'evaluation/{arm}-outputs.jsonl').read_text().splitlines()]
        require(len(rows) == 200 and len({r['id'] for r in rows}) == 200, 'Paired200 inference incomplete')
        require([r['id'] for r in rows] == manifest['selected_ids']
                and completed['outputs'][arm] == pin(root / f'evaluation/{arm}-outputs.jsonl', root),
                'Paired answer membership/hash differs')
        require(all(row['arm'] == arm and row['generation'] == GENERATION
                    and row['effective_generation_mode'] == 'greedy_search'
                    and row['generation_config'] == config and row['exception'] is None
                    and row['context_truncated'] is False and row['skipped_context'] is False
                    for row in rows), 'Actual paired decoder dispatch differs')
    return completed
