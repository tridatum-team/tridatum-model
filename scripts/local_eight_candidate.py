"""Authenticated final-endpoint contract; stdlib only, no private grading reads."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

POLICY_SHA = '1641b555673165307c053bef701d4a2399eb8a774916ba5cb3e2a0a4ec9f9679'
PROTOCOL_SHA = '222e0f233f19a1f994a80b65073bfc6563be42658a9c276829d091fb10d275ee'
MODEL_ID = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
# This is the supported PEFT serialization, not an open-ended bag of overrides.
# Unknown future fields require a deliberate review rather than silent defaults.
FIXED_LORA_CONFIG = dict(
    alpha_pattern={}, auto_mapping=None, bias='none', corda_config=None, eva_config=None,
    exclude_modules=None, fan_in_fan_out=False, inference_mode=True, init_lora_weights=True,
    layer_replication=None, layers_pattern=None, layers_to_transform=None, loftq_config={},
    lora_alpha=16, lora_bias=False, lora_dropout=0.0, megatron_config=None,
    megatron_core='megatron.core', modules_to_save=None, peft_type='LORA',
    qalora_group_size=16, r=8, rank_pattern={}, revision=None, target_parameters=None,
    task_type='CAUSAL_LM', trainable_token_indices=None,
    use_dora=False, use_qalora=False, use_rslora=False)


def validate_adapter_config(config, targets):
    require(isinstance(config, dict)
            and set(config) == set(FIXED_LORA_CONFIG) | {'base_model_name_or_path', 'target_modules'}
            and isinstance(config['base_model_name_or_path'], str)
            and isinstance(config['target_modules'], list)
            and len(config['target_modules']) == len(targets)
            and set(config['target_modules']) == set(targets),
            'adapter configuration/base isolation differs')
    # Preserve exact JSON types, including bool versus int. The base name is
    # informational: inference always supplies the separately authenticated base.
    effective = {key: config[key] for key in FIXED_LORA_CONFIG}
    require(json.dumps(effective, sort_keys=True) == json.dumps(FIXED_LORA_CONFIG, sort_keys=True),
            'adapter configuration/base isolation differs')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate identity key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite identity: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=invalid)


def file_pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
            'candidate evidence must be ordinary unshared file')
    path = path.resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha)


def bound(root, entry):
    require(isinstance(entry, dict) and set(entry) == {'path', 'bytes', 'sha256'}, 'invalid candidate pin')
    path = Path(entry['path'])
    path = path if path.is_absolute() else Path(root) / path
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
            'candidate evidence must be ordinary unshared file')
    actual = file_pin(path)
    require(all(actual[k] == entry[k] for k in ('bytes', 'sha256')), 'candidate evidence hash/size differs: ' + str(path))
    return path.resolve()


def same_pin(root, first, second):
    return bound(root, first) == bound(root, second) and first['sha256'] == second['sha256']


def supervised(root, entry, policy_pin, phase):
    result = load(bound(root, entry))
    require(result.get('schema') == 'trid-local-mps-supervised-v1'
            and result.get('phase') == phase and result.get('status') == 'accepted',
            'parent supervision did not accept ' + phase)
    cleanup = result.get('cleanup', {})
    require(cleanup.get('accepted') is True and cleanup.get('leader_reaped') is True
            and cleanup.get('remaining_members') == [], 'owned process cleanup not accepted')
    require(same_pin(root, result['policy'], policy_pin), 'supervisor policy differs')
    sources = load(bound(root, result['sources']))
    require(sources.get('schema') == 'trid-local-mps-sources-v1'
            and sources.get('files') == result.get('source_files') and sources['files'],
            'actual supervisor source set differs')
    paths = [bound(root, pin) for pin in sources['files']]
    require(len(paths) == len(set(paths)), 'duplicate source pin')
    worker = load(bound(root, result['worker_result']))
    require(same_pin(root, worker['policy'], policy_pin), 'worker policy differs')
    actual = worker.get('actual_source_files')
    require(actual and actual == result.get('actual_source_files'), 'worker actual source closure differs')
    actual_paths = [bound(root, pin) for pin in actual]
    require(len(actual_paths) == len(set(actual_paths)) and set(actual_paths) <= set(paths),
            'worker executed unregistered source')
    if phase == 'train':
        bound(root, result['ledger'])
        capacity, _ = supervised(root, result['capacity_result'], policy_pin, 'capacity')
        require(same_pin(root, result['sources'], capacity['sources']),
                'capacity and training source registration differ')
    else:
        require(worker.get('status') == 'complete' and worker.get('phase') == 'capacity'
                and worker.get('optimizer_updates') == 2 and worker.get('microbatches_per_update') == 4
                and worker.get('reload_exact') is True
                and type(worker.get('reload_loss')) in (int, float) and math.isfinite(worker['reload_loss']),
                'capacity proof did not complete the registered envelope/reload')
    return result, worker


def resolve_candidate(root, manifest_path, expected_sha):
    """Authenticate nomination without opening references, rubrics, or protocol text."""
    require(bool(manifest_path) == bool(expected_sha), 'candidate manifest and SHA must be paired')
    require(manifest_path is not None, 'candidate registration required')
    registration_pin = file_pin(manifest_path)
    require(registration_pin['sha256'] == expected_sha, 'candidate registration hash differs')
    registration = load(bound(root, registration_pin))
    if registration.get('schema') == 'trid-fresh-eight-cuda-training-candidate-v2':
        return resolve_cuda_candidate(root, manifest_path, expected_sha)
    require(set(registration) == {'schema', 'policy', 'candidate_id', 'arm', 'step', 'supervision',
                                 'training', 'checkpoint_ledger', 'adapter'}
            and registration['schema'] == 'trid-local-eight-candidate-v1', 'candidate registration schema differs')
    require(registration['policy']['sha256'] == POLICY_SHA, 'unregistered training policy')
    policy = load(bound(root, registration['policy']))
    require(policy.get('schema') == 'trid-local-eight-training-policy-v1'
            and registration['candidate_id'] == policy['experiment_id']
            and registration['arm'] == 'candidate' and registration['step'] == policy['training']['steps'] == 150,
            'only registered final step150 may be nominated')
    require(policy['evaluation']['review_protocol']['sha256'] == PROTOCOL_SHA, 'prospective protocol identity differs')
    # Only authenticate opaque protocol bytes here; never parse grading instructions.
    bound(root, policy['evaluation']['review_protocol'])
    training_manifest = load(bound(root, policy['data']['training_manifest']))
    require(training_manifest.get('schema') == 'trid-expanded-training-manifest-v1'
            and training_manifest.get('state') == 'ADMITTED' and training_manifest.get('rows') == 600
            and training_manifest.get('valid') == [], 'admitted training manifest differs')
    manifest_dir = bound(root, policy['data']['training_manifest']).parent
    admitted = set()
    for entry in training_manifest['train']:
        require(entry['epochs'] == 1, 'training rows repeated')
        path = bound(manifest_dir, {k: entry[k] for k in ('path', 'bytes', 'sha256')})
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    admitted.add((entry['sha256'], line_number))
    require(len(admitted) == 600, 'admitted source row denominator differs')
    acquisition = load(bound(root, policy['model']['acquisition_manifest']))
    require(acquisition['repo_id'] == MODEL_ID and acquisition['revision'] == REVISION,
            'official acquisition identity differs')
    supervisor, training = supervised(root, registration['supervision'], registration['policy'], 'train')
    require(same_pin(root, supervisor['worker_result'], registration['training'])
            and same_pin(root, supervisor['ledger'], registration['checkpoint_ledger']),
            'nomination is not accepted parent result')
    ledger = load(bound(root, registration['checkpoint_ledger']))
    for record in (training, ledger):
        require(record['base_model_id'] == MODEL_ID and record['base_revision'] == REVISION
                and record['base_weights'] == {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'}
                and record['init_adapter'] is None
                and record['manifest_sha256'] == policy['data']['training_manifest']['sha256'],
                'fresh official base/training lineage differs')
    require(training['status'] == 'complete' and training['steps'] == training['cumulative_steps'] == 150
            and ledger['run_status'] == 'complete' and ledger['planned_steps'] == 150
            and training['train_rows'] == 600 and training['valid_rows'] == 0
            and training['dropped_over_max_tokens'] == training['valid_dropped'] == 0,
            'complete fixed endpoint and denominator required')
    coverage = training.get('coverage', [])
    require(training.get('coverage_complete') is True and len(coverage) == 600
            and {(row['source_sha256'], row['source_line']) for row in coverage} == admitted,
            'exactly-once admitted source coverage differs')
    inventory = training.get('dtype_inventory', {})
    for group, dtype in (('base', 'torch.bfloat16'), ('adapter', 'torch.float32')):
        values = inventory.get(group, {})
        require(values and all(key in (dtype + '@mps', dtype + '@mps:0') and entry['tensors'] > 0
                               and entry['elements'] > 0 and entry['bytes'] > 0 for key, entry in values.items()),
                'actual training dtype inventory differs: ' + group)
    for key in ('batch_size', 'micro_batch_size', 'lr', 'seed', 'max_tokens', 'max_answer_tokens', 'family'):
        require(training[key] == policy['training'][key], 'training configuration differs: ' + key)
    require(training['device'] == 'mps' and training['dtype'] == 'torch.bfloat16', 'training runtime differs')
    require(training['lora'] == {k: policy['training'][k] for k in ('rank', 'alpha', 'dropout', 'targets')},
            'training LoRA configuration differs')
    require([row['step'] for row in training['history']] == list(range(1, 151)), 'training history coverage differs')
    require(ledger['checkpoint_every'] == 50 and [c['step'] for c in ledger['checkpoints']] == [50, 100, 150]
            and training['checkpoints'] == ledger['checkpoints'], 'diagnostic checkpoint ledger differs')
    adapter = registration['adapter']
    require(set(adapter) == {'directory', 'files'}, 'candidate adapter fields differ')
    directory = Path(adapter['directory'])
    directory = (directory if directory.is_absolute() else Path(root) / directory).resolve()
    require(directory == bound(root, registration['training']).parent / 'adapter', 'only final adapter directory allowed')
    require(set(adapter['files']) == {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}
            and {p.name for p in directory.iterdir()} == set(adapter['files']), 'adapter membership differs')
    files = {}
    for name, entry in adapter['files'].items():
        require(bound(root, entry) == directory / name, 'adapter file path differs')
        files[name] = file_pin(directory / name)
    weights_sha = files['adapter_model.safetensors']['sha256']
    config_sha = files['adapter_config.json']['sha256']
    require(set(supervisor.get('adapter_files', {})) == {'adapter_config.json', 'adapter_model.safetensors'}
            and all(same_pin(root, supervisor['adapter_files'][name], files[name])
                    for name in supervisor['adapter_files']), 'accepted producer adapter files differ')
    last = ledger['last_step']
    last_path = Path(last['path'])
    last_path = last_path if last_path.is_absolute() else Path(root) / last_path
    require(training['last_step'] == last and last['step'] == 150 and last_path.resolve() == directory
            and last['adapter_weights_sha256'] == weights_sha
            and last.get('adapter_config_sha256') == config_sha
            and last['equals_last_checkpoint'] is True
            and ledger['checkpoints'][-1]['adapter_weights_sha256'] == weights_sha
            and ledger['checkpoints'][-1].get('adapter_config_sha256') == config_sha,
            'final training/ledger/config/weights binding differs')
    config = load(directory / 'adapter_config.json')
    validate_adapter_config(config, policy['training']['targets'])
    identity = dict(status='registered_final_development_candidate_not_selected', step=150,
                    files=files, checkpoint_ledger=file_pin(bound(root, registration['checkpoint_ledger'])), config=config)
    return dict(arms=['base', 'candidate'], adapter_directory=str(directory), adapter=identity,
                adapter_sha256=weights_sha, registration=registration_pin,
                policy=registration['policy'], policy_document=policy,
                candidate_id=registration['candidate_id'], review_protocol=policy['evaluation']['review_protocol'])


def authenticated_training_data(root, plan):
    """Validate CUDA source/data against the sealed upload, never today's checkout."""
    import tempfile
    from scripts.aws_eval import fresh_eight_experiment as profile
    from scripts.aws_eval.grounded_hf_lifecycle import authenticate_archive, relative
    archive = Path(root) / relative(plan['bundle']['local_path'])
    with tempfile.TemporaryDirectory(prefix='trid-training-evidence-') as temporary:
        extracted = Path(temporary) / 'bundle'
        authenticate_archive(archive, plan['bundle'], extracted)
        data = profile.validate_data(extracted, plan)
    return data, file_pin(archive)


def resolve_cuda_candidate(root, manifest_path, expected_sha):
    """CUDA has its own producer lineage; MPS receipts never satisfy it.

    Every consumer requires durable export and independent cloud cleanup.
    No grading references or review-protocol contents are read here.
    """
    from scripts.aws_eval import fresh_eight_experiment as profile
    manifest_path = Path(manifest_path).resolve()
    output = manifest_path.parent
    registration_pin = file_pin(manifest_path)
    require(registration_pin['sha256'] == expected_sha, 'candidate registration hash differs')
    registration = load(manifest_path)
    require(set(registration) == {'schema', 'plan', 'candidate_id', 'arm', 'step',
                                 'completion_scope', 'quality_accepted', 'cloud_evaluation'}
            and registration['schema'] == 'trid-fresh-eight-cuda-training-candidate-v2'
            and registration['completion_scope'] == 'cuda_capacity_train_export_only'
            and registration['quality_accepted'] is False and registration['cloud_evaluation'] is False
            and registration['candidate_id'] == profile.RUN_ID
            and registration['arm'] == 'candidate' and registration['step'] == 150,
            'CUDA final candidate registration differs')
    plan = load(profile.bound(output, registration['plan']))
    data, training_bundle = authenticated_training_data(root, plan)
    policy = data['policy']
    require(policy['evaluation']['review_protocol']['sha256'] == PROTOCOL_SHA,
            'prospective protocol identity differs')
    profile.validate_capacity(output, plan)
    for phase in ('capacity', 'train'):
        supervision = profile.validate_supervision(output, phase, plan)
        expected = output / phase / ('capacity-worker.json' if phase == 'capacity' else 'training-worker.json')
        require(profile.bound(output, supervision['worker_result']).resolve() == expected,
                'CUDA supervisor accepted a different worker')
    training = profile.validate_training(output, plan)
    ledger = load(output / 'train/checkpoints.json')
    for record in (training, ledger):
        require(record['base_model_id'] == MODEL_ID and record['base_revision'] == REVISION
                and record['base_weights'] == {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'}
                and record['init_adapter'] is None
                and record['manifest_sha256'] == plan['data']['manifest']['sha256'],
                'fresh official CUDA base/training lineage differs')
    require(training['cumulative_steps'] == 150 and ledger['planned_steps'] == 150
            and training['dtype'] == 'torch.bfloat16', 'CUDA final endpoint differs')
    for key in ('batch_size', 'micro_batch_size', 'lr', 'seed', 'max_tokens', 'max_answer_tokens', 'family'):
        require(training[key] == policy['training'][key], 'training configuration differs: ' + key)
    require(training['lora'] == {k: policy['training'][k] for k in ('rank', 'alpha', 'dropout', 'targets')},
            'training LoRA configuration differs')
    for group, dtype in (('base', 'torch.bfloat16'), ('adapter', 'torch.float32')):
        values = training.get('dtype_inventory', {}).get(group, {})
        require(values and all(key in (dtype + '@cuda', dtype + '@cuda:0')
                and entry['tensors'] > 0 and entry['elements'] > 0 and entry['bytes'] > 0
                for key, entry in values.items()), 'actual CUDA training dtype inventory differs: ' + group)
    require(training['checkpoints'] == ledger['checkpoints'] and training['last_step'] == ledger['last_step']
            and ledger['checkpoint_every'] == 50, 'CUDA report/ledger differ')
    directory = output / 'train/adapter'
    names = {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}
    require({p.name for p in directory.iterdir()} == names, 'CUDA adapter membership differs')
    files = {name: dict(file_pin(directory / name), path='train/adapter/' + name) for name in sorted(names)}
    last = ledger['last_step']
    for name, field in (('adapter_config.json', 'adapter_config_sha256'),
                        ('adapter_model.safetensors', 'adapter_weights_sha256')):
        require(last[field] == ledger['checkpoints'][-1][field] == files[name]['sha256'],
                'CUDA final150 config/weights binding differs')
    config = load(directory / 'adapter_config.json')
    validate_adapter_config(config, policy['training']['targets'])
    profile.validate_completion(output, plan)
    validate_cuda_export(output, plan, registration['plan']['sha256'])
    identity = dict(status='registered_final_development_candidate_not_selected', step=150, files=files,
        checkpoint_ledger=dict(file_pin(output / 'train/checkpoints.json'), path='train/checkpoints.json'), config=config)
    return dict(arms=['base', 'candidate'], adapter_directory=str(directory), adapter=identity,
        adapter_sha256=files['adapter_model.safetensors']['sha256'], registration=registration_pin,
        policy=plan['data']['policy'], policy_document=policy, candidate_id=registration['candidate_id'],
        review_protocol=policy['evaluation']['review_protocol'], training_runtime='cuda',
        inference_runtime='mps', producer='trid-fresh-eight-cuda-training-candidate-v2',
        completion_scope='cuda_capacity_train_export_only', quality_accepted=False,
        cloud_evaluation=False, plan=registration['plan'], training_bundle=training_bundle)


def validate_cuda_export(output, plan, plan_sha):
    """Require exported success and independent controller compute/storage cleanup."""
    from scripts.aws_eval import fresh_eight_experiment as profile
    output = Path(output)
    export = load(output / 'export-manifest.json')
    status = export.get('status', {})
    require(export.get('run_id') == plan['identity']['run_id'] and export.get('plan_sha256') == plan_sha
            and status.get('run_id') == plan['identity']['run_id'] and status.get('plan_sha256') == plan_sha
            and status.get('training_success') is True and status.get('stop_errors') == []
            and status.get('completion_scope') == 'cuda_capacity_train_export_only'
            and status.get('quality_accepted') is False and status.get('cloud_evaluation') is False
            and not any(key in status for key in ('error', 'export_error')),
            'CUDA cloud completion/export was not successful')
    members = export.get('files', [])
    names = [entry['path'] for entry in members]
    require(len(names) == len(set(names)) and set(names) <= set(plan['output_allowlist'])
            and {'candidate.json', 'plan.json'} <= set(names), 'CUDA export membership differs')
    actual = {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
    require(actual == set(names) | {'export-manifest.json'}, 'CUDA retrieved export membership differs')
    for entry in members:
        profile.bound(output, entry)
    require(export.get('total_bytes') == sum(entry['bytes'] for entry in members),
            'CUDA export byte denominator differs')
    completion = load(output.parent / 'completion.json')
    controller = load(output.parent / 'controller-result.json')
    evidence = controller.get('evidence', {})
    cleanup = controller.get('cleanup', {})
    archive = completion.get('export', {})
    require(type(archive.get('bytes')) is int and archive['bytes'] > 0
            and isinstance(archive.get('sha256'), str) and len(archive['sha256']) == 64
            and all(c in '0123456789abcdef' for c in archive['sha256']),
            'CUDA exported archive identity missing')
    require(all(completion.get(key) == value for key, value in status.items())
            and completion.get('training_success') is True
            and completion.get('stop_errors') == []
            and not any(key in completion for key in ('error', 'export_error'))
            and controller.get('state') == 'complete' and controller.get('training_success') is True
            and not any(key in controller for key in ('error', 'evidence_error', 'schedule_cleanup_error'))
            and evidence.get('native_pipeline_success') is True and evidence.get('completion') == completion
            and evidence.get('export_sha256') == completion.get('export', {}).get('sha256'),
            'CUDA independent cloud completion differs')
    require(cleanup.get('cleanup_state') == 'complete' and cleanup.get('instance_terminated') is True
            and isinstance(completion.get('instance_id'), str) and completion['instance_id'].startswith('i-')
            and cleanup.get('instance_id') == completion.get('instance_id')
            and cleanup.get('root_volume_ids') and not cleanup.get('remaining_volumes')
            and all(isinstance(value, str) and value.startswith('vol-') for value in cleanup['root_volume_ids'])
            and set(cleanup['root_volume_ids']) <= set(cleanup.get('recorded_volume_ids', []))
            and set(cleanup.get('absent_volume_ids', [])) == set(cleanup.get('recorded_volume_ids', [])),
            'CUDA instance/root-volume cleanup is not complete')


def public_contract(contract):
    return {key: value for key, value in contract.items() if key not in ('policy_document', 'adapter_directory', 'adapter')}


def review_protocol(root, contract):
    protocol = load(bound(root, contract['review_protocol']))
    require(protocol.get('schema') == 'trid-local-eight-review-protocol-v2'
            and protocol.get('paired_consistency') is True and protocol.get('critical_errors_are_codes') is True,
            'prospective review protocol differs')
    return {key: protocol[key] for key in ('instructions', 'paired_consistency', 'critical_error_codes', 'critical_errors_are_codes')}
