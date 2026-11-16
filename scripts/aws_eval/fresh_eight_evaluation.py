"""Recipe-free, single-instance CUDA evaluation of the immutable final150 adapter."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SCHEMA = PROFILE = 'trid-fresh-eight-cuda-evaluation-v1'
APPROVAL_SCHEMA = 'trid-cuda-evaluation-approval-v1'
RUN_SCHEMA = 'trid-cuda-eight-paired-v1'
LINEAGE_SCHEMA = 'trid-cuda-evaluation-lineage-v1'
WORKER_SCHEMA = 'trid-cuda-evaluation-worker-v1'
SUPERVISION_SCHEMA = 'trid-cuda-evaluation-supervision-v1'
RUN_ID = 'mois-eight-cuda-eval-20261116-134815'
COMPLETION_SCOPE = 'cuda_evaluation_only_existing_final150'
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
ADAPTER_SHA = 'b03712d2907c8e28b19c8a60d8a48001becac92c901116f9d027dd6b2a90db7b'
CANDIDATE_SHA = 'd62dbc553a494d48013fdaf0202a758c744c8b1147ddbb0fd5ef4ddb42072986'
INPUT_SHA = 'aaeea137535e824e7c496f59699f5d892321d307290fd51fae9514a5854a1570'
ACQUISITION_SHA = 'e0dea2b77e7d9108d58786af3d9210c6ba0600ee29733c144a8e7efd1ff55300'
SMOKE_ID = 'expanded-eval-mois-001'
INPUT_PATH = 'runs/mois-expanded-development-20261114T121509Z/frozen-dataset/evaluation-inputs.json'
ACQUISITION_SECONDS = 1800
ARMS = ('base', 'candidate')
GENERATION = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
                  max_new_tokens=768, context_tokens=4096, dtype='bfloat16', backend='transformers-cuda')
LIMITS = dict(instance_count=1, lifetime_seconds=18900, reserve_seconds=300,
              root_gib=120, rss_bytes=48 * 1024**3, allocator_bytes=40 * 1024**3,
              gpu_bytes=42 * 1024**3, output_bytes=512 * 1024**2, cpu_cores=6, swap_bytes=0)
REQUIRED_SOURCES = {
    'scripts/aws_eval/fresh_eight_evaluation.py', 'scripts/aws_eval/fresh_eight_evaluation_worker.py',
    'scripts/aws_eval/grounded_hf_lifecycle.py', 'scripts/aws_eval/grounded_hf_worker.py',
    'scripts/aws_eval/grounded_hf_bootstrap.sh', 'scripts/aws_eval/qwen_cuda_smoke.py',
    'scripts/aws_eval/qwen_cpu_smoke.py', 'scripts/aws_eval/fresh_eight_launch.py',
    'scripts/evaluate_local_eight.py'}
# Inherit the complete proven training environment, not just three headline packages.
RUNTIME_PACKAGES = {
    'accelerate': '1.15.0', 'certifi': '2026.10.23', 'charset-normalizer': '3.5.2',
    'filelock': '4.2.10', 'fsspec': '2026.9.0', 'hf-xet': '1.8.0', 'huggingface-hub': '0.36.2',
    'idna': '3.20', 'jinja2': '3.1.6', 'markupsafe': '3.0.4', 'mpmath': '1.3.0',
    'networkx': '3.7', 'numpy': '2.5.4', 'nvidia-cublas-cu12': '12.8.4.1',
    'nvidia-cuda-cupti-cu12': '12.8.90', 'nvidia-cuda-nvrtc-cu12': '12.8.93',
    'nvidia-cuda-runtime-cu12': '12.8.90', 'nvidia-cudnn-cu12': '9.10.2.21',
    'nvidia-cufft-cu12': '11.3.3.83', 'nvidia-cufile-cu12': '1.13.1.3',
    'nvidia-curand-cu12': '10.3.9.90', 'nvidia-cusolver-cu12': '11.7.3.90',
    'nvidia-cusparse-cu12': '12.5.8.93', 'nvidia-cusparselt-cu12': '0.7.1',
    'nvidia-nccl-cu12': '2.27.5', 'nvidia-nvjitlink-cu12': '12.8.93',
    'nvidia-nvshmem-cu12': '3.3.20', 'nvidia-nvtx-cu12': '12.8.90', 'packaging': '26.3',
    'peft': '0.17.1', 'pip': '25.3', 'psutil': '7.2.2', 'pyyaml': '6.0.3',
    'regex': '2026.11.9', 'requests': '2.34.2', 'safetensors': '0.8.0', 'setuptools': '84.2.1',
    'sympy': '1.14.0', 'tokenizers': '0.22.2', 'torch': '2.9.1+cu128', 'tqdm': '4.70.1',
    'transformers': '4.57.6', 'triton': '3.5.1', 'typing-extensions': '4.16.0', 'urllib3': '2.8.0'}
ADAPTER_FILES = {
    'adapter_model.safetensors': (87360584, ADAPTER_SHA),
    'adapter_config.json': (961, '0eb939231cee21e5d42a7303e2510df0fdb53304b58d7afb9c8707376abb6263'),
    'README.md': (5256, 'bf3df74db7fd460a9849d6f97045979ec59c4f48ee13879807b7d235b312cb99')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def is_evaluation(plan):
    return plan.get('schema') == SCHEMA


def pin(path, root=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or unsafe pinned file: ' + str(path))
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(root).as_posix() if root is not None else str(path),
                bytes=path.stat().st_size, sha256=sha)


def bound(root, record):
    require(set(record) == {'path', 'bytes', 'sha256'}, 'Invalid portable pin')
    name = PurePosixPath(record['path'])
    require(not name.is_absolute() and '..' not in name.parts and name.as_posix() == record['path'], 'Unsafe portable path')
    path = Path(root) / name
    require(not any(Path(root).joinpath(*name.parts[:i]).is_symlink() for i in range(1, len(name.parts) + 1))
            and pin(path, root) == record, 'Pinned bytes differ')
    return path


def load(root, record):
    return json.loads(bound(root, record).read_bytes())


def validate_limits(plan):
    require(plan['limits'] == LIMITS and all(type(v) is int for v in plan['limits'].values()), 'Fixed evaluation limits differ')
    require(plan['lifetime'] == dict(total_seconds=19800, termination_grace_seconds=900, automatic_extension=False),
            'Evaluation requires total19800/work18900/grace900')
    require(plan['cost']['termination_grace_seconds'] == 900, 'Priced shutdown grace differs')


def validate_approval(plan, approval, plan_sha, key_arn):
    expected = dict(schema=APPROVAL_SCHEMA, run_id=RUN_ID, plan_sha256=plan_sha,
        instance_limit=1, cap_usd='20', workload_sha256=object_sha(plan['workload']),
        authorization_sha256=plan['authorization']['sha256'], limits=LIMITS,
        ebs_kms_key_arn=key_arn, automatic_retry=False, automatic_extension=False,
        retraining=False, deployment=False, approved=True)
    require(approval == expected, 'Exact new USD20 evaluation approval required')


def output_members(plan=None):
    names = {'runtime.json', 'runtime.log', 'evaluation.log', 'plan.json', 'model-acquisition.json',
             'lineage-binding.json', 'source-manifest.json', 'evaluation/evaluation-worker.json',
             'evaluation/supervisor-result.json'}
    for mode in ('smoke', 'evaluation'):
        names.update(mode + '/' + name for name in ('run-manifest.json', 'decoder-profile.json',
            'prompt-preflight.json', 'loaded-runtime.json', 'completion.json', 'base-outputs.jsonl',
            'candidate-outputs.jsonl', 'attempts.jsonl', 'failure.json'))
    return names


def validate_workload(plan, bundled):
    require(is_evaluation(plan) and plan['identity']['run_id'] == RUN_ID, 'Evaluation profile/run identity differs')
    require(not ({'recipe', 'data', 'evaluation', 'phase_seconds'} & set(plan)), 'Recipe/training/alternate evaluation channels forbidden')
    validate_limits(plan)
    w = plan['workload']
    require(set(w) == {'evaluation_inputs', 'acquisition_manifest', 'lineage_binding', 'adapter_files',
                       'source_manifest', 'generation', 'smoke_id'}, 'Exact public evaluation workload required')
    require(canonical(w['generation']) == canonical(GENERATION) and w['smoke_id'] == SMOKE_ID, 'Frozen greedy/smoke contract differs')
    require(w['evaluation_inputs'] == dict(path=INPUT_PATH, bytes=568597, sha256=INPUT_SHA)
            and w['acquisition_manifest']['sha256'] == ACQUISITION_SHA, 'Frozen public/model metadata differs')
    require(plan['model']['id'] == MODEL_ID and plan['model']['revision'] == MODEL_REVISION, 'Official base identity differs')
    require(plan['runtime']['packages'] == RUNTIME_PACKAGES and plan['runtime']['torch_cuda'] == '12.8', 'Proven exact runtime differs')
    require(len(w['adapter_files']) == 3 and {p['path'] for p in w['adapter_files']} == {'adapter/' + n for n in ADAPTER_FILES},
            'Only the final adapter trio may be staged')
    for p in w['adapter_files']:
        require((p['bytes'], p['sha256']) == ADAPTER_FILES[PurePosixPath(p['path']).name], 'Final150 adapter bytes differ')
    records = [w[k] for k in ('evaluation_inputs', 'acquisition_manifest', 'lineage_binding', 'source_manifest')]
    records += [*w['adapter_files'], plan['authorization'], plan['resource_envelope'], plan['ancillary_evidence']]
    require(all(set(p) == {'path', 'bytes', 'sha256'} and bundled.get(p['path']) == p for p in records), 'Unbundled workload evidence')
    require(set(plan['sources']) == REQUIRED_SOURCES, 'Executable closure must be exact and training-free')
    require(set(plan['output_allowlist']) == output_members() and len(plan['output_allowlist']) == len(output_members()), 'Output allowlist differs')
    allowed = {p['path'] for p in records} | REQUIRED_SOURCES | {plan['runtime']['bootstrap']['admission']['path']}
    require(allowed <= set(bundled), 'Bundle missing approved execution inputs')


def validate_lineage(lineage, adapter_files):
    require(lineage['schema'] == LINEAGE_SCHEMA and lineage['candidate']['sha256'] == CANDIDATE_SHA
            and lineage['candidate']['bytes'] == 414 and lineage['adapter_sha256'] == ADAPTER_SHA,
            'Original final150 lineage differs')
    original = lineage['adapter_files']
    if isinstance(original, dict):
        original = list(original.values())
    require(len(original) == 3 and {PurePosixPath(p['path']).name: (p['bytes'], p['sha256']) for p in original} == ADAPTER_FILES,
            'Portable original adapter lineage differs')
    require({PurePosixPath(p['path']).name: (p['bytes'], p['sha256']) for p in adapter_files} == ADAPTER_FILES,
            'Staged adapter differs from original lineage')
    require(isinstance(lineage['provenance'], dict) and lineage['provenance'], 'Original authentication provenance missing')


def validate_data(root, plan):
    validate_workload(plan, {p['path']: p for p in plan['bundle']['files']})
    w = plan['workload']
    # Only these explicit nested admission pins extend the minimal public bundle.
    ancillary = load(root, plan['ancillary_evidence'])
    resources = load(root, plan['resource_envelope'])
    bootstrap_pin = plan['runtime']['bootstrap']['admission']
    bootstrap = load(root, bootstrap_pin)
    evidence = [resources['cost_evidence'], ancillary['bucket_policy_status_evidence'],
                ancillary['ebs_kms_key_evidence'], *bootstrap['evidence']]
    for record in evidence:
        bound(root, record)
    direct = [w[k] for k in ('evaluation_inputs', 'acquisition_manifest', 'lineage_binding', 'source_manifest')]
    direct += [*w['adapter_files'], plan['authorization'], plan['resource_envelope'],
               plan['ancillary_evidence'], bootstrap_pin, *evidence]
    allowed = REQUIRED_SOURCES | {p['path'] for p in direct}
    require({p['path'] for p in plan['bundle']['files']} == allowed,
            'Bundle contains unapproved history, references or executable inputs')
    lineage = load(root, w['lineage_binding'])
    validate_lineage(lineage, w['adapter_files'])
    for record in w['adapter_files']:
        bound(root, record)
    config = json.loads(bound(root, next(p for p in w['adapter_files'] if p['path'].endswith('adapter_config.json'))).read_bytes())
    require(config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0
            and config['peft_type'] == 'LORA' and config['task_type'] == 'CAUSAL_LM'
            and config['bias'] == 'none' and config['modules_to_save'] is None, 'Adapter disable semantics differ')
    acquisition = load(root, w['acquisition_manifest'])
    require(acquisition['repo_id'] == MODEL_ID and acquisition['revision'] == MODEL_REVISION
            and [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in acquisition['files']]
            == [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in plan['model']['files']], 'Official model manifest differs')
    rows = load(root, w['evaluation_inputs'])
    require(isinstance(rows, list) and len(rows) == 200 and len({r['id'] for r in rows}) == 200, 'Fixed200 denominator differs')
    for row in rows:
        require(set(row) == {'id', 'messages'} and isinstance(row['id'], str) and row['id'], 'Only public identity/messages permitted')
        messages = row['messages']
        require(isinstance(messages, list) and messages and messages[-1]['role'] == 'user'
                and all(set(m) == {'role', 'content'} and m['role'] in ('system', 'user', 'assistant')
                        and isinstance(m['content'], str) for m in messages), 'Invalid public messages')
    require(SMOKE_ID in {r['id'] for r in rows}, 'Registered smoke ID missing')
    sources = load(root, w['source_manifest'])['files']
    require(len(sources) == len(REQUIRED_SOURCES) and {p['path'] for p in sources} == REQUIRED_SOURCES, 'Source manifest membership differs')
    for record in sources:
        bound(root, record)
        require(record['sha256'] == plan['sources'][record['path']], 'Executed source seal differs')
    return dict(evaluation_inputs=rows, lineage=lineage, adapter_config=config, sources=sources)


def parameter_inventory(inventory):
    require(set(inventory) == {'base', 'adapter'}, 'Parameter groups differ')
    for group, dtype, size in (('base', 'torch.bfloat16', 2), ('adapter', 'torch.float32', 4)):
        require(inventory[group] and all(k == dtype + '@cuda:0' and v['tensors'] > 0
            and v['elements'] > 0 and v['bytes'] == v['elements'] * size for k, v in inventory[group].items()),
            'Actual frozen CUDA dtype inventory differs')


def validate_answer(row, case, arm, mode, binding, config):
    require(mode in ('smoke', 'evaluation') and arm in ARMS, 'Unknown paired arm/mode')
    require(canonical(row['generation']) == canonical(GENERATION)
            and canonical(row['generation_config']) == canonical(config), 'Generation types/values differ')
    require(all(row.get(k) == v for k, v in {**case, **binding, 'arm': arm, 'mode': mode,
        'generation': GENERATION, 'generation_config': config, 'exception': None,
        'adapter_weights_sha256': None if arm == 'base' else ADAPTER_SHA,
        'context_truncated': False, 'skipped_context': False, 'effective_generation_mode': 'greedy_search'}.items()),
        'Answer identity/generation binding differs')
    tokens = row['output_token_ids']
    require(isinstance(tokens, list) and 0 < len(tokens) <= 768 and all(type(t) is int and t >= 0 for t in tokens)
            and row['generated_tokens'] == len(tokens), 'Invalid generated token envelope')
    stops = config['eos_token_id']
    stops = [stops] if type(stops) is int else stops
    require(isinstance(stops, list) and stops and all(type(t) is int and t >= 0 for t in stops), 'Explicit stop tokens missing')
    terminal = tokens[-1] if tokens[-1] in stops else None
    require(row['terminal_stop_token_id'] == terminal and row['termination_reason'] == ('eos' if terminal is not None else 'length')
            and (terminal is not None or len(tokens) == 768) and row['truncated_output'] is (terminal is None), 'Unexplained termination')
    require(isinstance(row['response_text'], str) and isinstance(row['raw_output'], str)
            and isinstance(row['elapsed_seconds'], (int, float)) and math.isfinite(row['elapsed_seconds'])
            and row['elapsed_seconds'] >= 0, 'Invalid output/timing')
    require(isinstance(case['prompt_token_ids'], list) and case['prompt_token_ids']
            and all(type(t) is int and t >= 0 for t in case['prompt_token_ids'])
            and len(case['prompt_token_ids']) + 768 <= 4096, 'Prompt context truncated')
    require(case['prompt_sha256'] == object_sha(case['messages']), 'Prompt message identity differs')
    memory = row['cuda_memory']
    require(memory['device'] == 'cuda:0' and all(type(memory[k]) is int and memory[k] >= 0 for k in
        ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'))
        and memory['peak_reserved_bytes'] <= LIMITS['allocator_bytes'], 'Invalid CUDA memory evidence')
    if mode == 'smoke':
        require(row['termination_reason'] == 'eos' and row['response_text'].strip() and not row['truncated_output'], 'Smoke requires nonblank normal EOS')


def validate_mode(root, mode, plan):
    root = Path(root)
    require(mode in ('smoke', 'evaluation'), 'Unknown evaluation mode')
    folder = root / mode
    read = lambda name: json.loads((folder / name).read_bytes())
    manifest, loaded, completion = (read(name) for name in ('run-manifest.json', 'loaded-runtime.json', 'completion.json'))
    require(manifest['schema'] == RUN_SCHEMA and manifest['mode'] == mode and canonical(manifest['generation']) == canonical(GENERATION)
            and manifest['arm_order'] == list(ARMS) and manifest['grading_access'] is False
            and manifest['retries'] is False and manifest['resume'] is False, 'Paired mode contract differs')
    require(manifest['plan'] == pin(root / 'plan.json', root)
            and manifest['lineage_binding'] == pin(root / 'lineage-binding.json', root)
            and manifest['source_manifest'] == pin(root / 'source-manifest.json', root)
            and manifest['input'] == plan['workload']['evaluation_inputs'], 'Portable run evidence differs')
    lineage = load(root, manifest['lineage_binding'])
    validate_lineage(lineage, plan['workload']['adapter_files'])
    require(manifest['evaluation_contract'] == lineage['provenance'], 'Original candidate provenance differs')
    model = manifest['model']
    require(model['id'] == MODEL_ID and model['revision'] == MODEL_REVISION and model['body_hashes_authenticated'] is True
            and model['representation'] == 'hf-safetensors-bfloat16'
            and model['files'] == [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in plan['model']['files']], 'Worker-observed model body identities differ')
    require(manifest['adapter']['step'] == 150 and manifest['adapter']['files'] == {PurePosixPath(p['path']).name: p for p in plan['workload']['adapter_files']}, 'Applied adapter identity differs')
    runtime = manifest['runtime']
    require(runtime['device'] == 'cuda' and runtime['dtype'] == 'torch.bfloat16' and runtime['cpu_fallback'] is False
            and runtime['cuda_equivalence_claim'] is False and runtime['packages'] == RUNTIME_PACKAGES
            and runtime['gpu_uuid'] == loaded['gpu_uuid'] and runtime['gpu_uuid'].startswith('GPU-'), 'CUDA runtime identity differs')
    source_files = load(root, manifest['source_manifest'])['files']
    code = {p['path']: p for p in source_files}
    require(manifest['code'] == loaded['code'] == code and set(code) == REQUIRED_SOURCES
            and all(p['sha256'] == plan['sources'][n] for n, p in code.items()), 'Executed source closure differs')
    require(loaded['parameter_dtypes'] == ['torch.bfloat16', 'torch.float32'] and loaded['frozen'] is True
            and loaded['model_loads'] == 1 and loaded['adapter_merged'] is False, 'Loaded inference state differs')
    parameter_inventory(loaded['parameter_inventory'])
    config, policy = loaded['generation_config'], loaded['generation_policy']
    require(config['do_sample'] is False and type(config['num_beams']) is int and config['num_beams'] == 1
            and type(config['max_new_tokens']) is int and config['max_new_tokens'] == 768
            and config['return_dict_in_generate'] is True and config['use_cache'] is True
            and policy['generation_mode'] == 'greedy_search' and policy['effective_config_sha256'] == object_sha(config)
            and policy['dispatch_kwargs'] == dict(use_model_defaults=False, do_sample=False, num_beams=1), 'Effective greedy policy differs')
    decoder = read('decoder-profile.json')
    require(decoder == manifest['decoder_profile'] == dict(profile='unconstrained', semantic_truth_guarantee=False), 'Secondary decoder forbidden')
    preflight = read('prompt-preflight.json')
    require(preflight['status'] == 'passed' and preflight['all_selected_preflighted_before_weights'] is True
            and preflight['all_200_preflighted_before_weights'] is True and preflight['truncation'] is False, 'Prompt preflight missing')
    prepared = preflight['prompts']
    ids = [p['id'] for p in prepared]
    require(ids == manifest['selected_ids'] and len(set(ids)) == len(ids)
            and (ids == [SMOKE_ID] if mode == 'smoke' else len(ids) == 200), 'Fixed evaluation denominator differs')
    binding = dict(run_manifest_sha256=pin(folder / 'run-manifest.json')['sha256'],
        loaded_runtime_sha256=pin(folder / 'loaded-runtime.json')['sha256'], model_id=MODEL_ID,
        model_revision=MODEL_REVISION, input_sha256=INPUT_SHA, nominated_adapter_weights_sha256=ADAPTER_SHA,
        decoder_profile='unconstrained', decoder_profile_sha256=pin(folder / 'decoder-profile.json')['sha256'],
        generation_policy_sha256=object_sha(policy), candidate_manifest_sha256=CANDIDATE_SHA)
    require(completion['status'] == 'complete' and completion['mode'] == mode and completion['rows_per_arm'] == len(ids)
            and completion['run_manifest_sha256'] == binding['run_manifest_sha256'] and set(completion['outputs']) == set(ARMS), 'Mode completion differs')
    for arm in ARMS:
        require(completion['outputs'][arm]['path'] == arm + '-outputs.jsonl', 'Output endpoint differs')
        rows = [json.loads(line) for line in bound(folder, completion['outputs'][arm]).read_text().splitlines()]
        require([r['id'] for r in rows] == ids, 'Missing, duplicate or reordered answers')
        for row, case in zip(rows, prepared):
            validate_answer(row, case, arm, mode, binding, config)
    attempts = [json.loads(line) for line in (folder / 'attempts.jsonl').read_text().splitlines()]
    require([(a['id'], a['arm'], a['state']) for a in attempts] == [(i, arm, 'started') for i in ids for arm in ARMS], 'Attempt coverage differs')
    require(not (folder / 'failure.json').exists(), 'Failure evidence forbids completion')
    return manifest


def validate_completion(root, plan):
    root = Path(root)
    require(json.loads((root / 'plan.json').read_bytes()) == plan and is_evaluation(plan), 'Export plan differs')
    validate_limits(plan)
    manifests = {mode: validate_mode(root, mode, plan) for mode in ('smoke', 'evaluation')}
    for field in ('model', 'adapter', 'input', 'runtime', 'code', 'evaluation_contract'):
        require(manifests['smoke'][field] == manifests['evaluation'][field], 'Smoke/full loaded identity differs')
    for key, filename in (('lineage_binding', 'lineage-binding.json'), ('source_manifest', 'source-manifest.json')):
        actual = pin(root / filename)
        require(all(actual[k] == plan['workload'][key][k] for k in ('bytes', 'sha256')),
                'Export metadata differs from sealed input')
    runtime = json.loads((root / 'runtime.json').read_bytes())
    require(runtime['packages'] == RUNTIME_PACKAGES and runtime['torch'] == RUNTIME_PACKAGES['torch']
            and runtime['cuda'] == '12.8' and runtime['count'] == 1 and runtime['bf16'] is True
            and runtime['name'] == plan['runtime']['gpu_name'] and runtime['run_id'] == RUN_ID
            and runtime['plan_sha256'] == pin(root / 'plan.json')['sha256']
            and runtime['instance_id'] == manifests['evaluation']['runtime']['instance_id'],
            'Exported native runtime proof differs')
    acquisition = json.loads((root / 'model-acquisition.json').read_bytes())
    require(acquisition == dict(status='authenticated', model_id=MODEL_ID, revision=MODEL_REVISION,
        plan_sha256=pin(root / 'plan.json')['sha256'], files=manifests['evaluation']['model']['files']),
        'Observed model acquisition receipt differs')
    full_prompts = json.loads((root / 'evaluation/prompt-preflight.json').read_bytes())['prompts']
    smoke_prompts = json.loads((root / 'smoke/prompt-preflight.json').read_bytes())['prompts']
    require(smoke_prompts == [p for p in full_prompts if p['id'] == SMOKE_ID], 'Smoke prompt differs from fixed200')
    result = json.loads((root / 'evaluation/evaluation-worker.json').read_bytes())
    require(result['schema'] == WORKER_SCHEMA and result['status'] == 'complete' and result['evaluation_success'] is True
            and result['completion_scope'] == COMPLETION_SCOPE and result['quality_accepted'] is False
            and result['run_id'] == RUN_ID and result['plan_sha256'] == pin(root / 'plan.json')['sha256']
            and result['rows_per_arm'] == 200 and result['smoke_rows_per_arm'] == 1
            and result['generation_calls'] == 402 and result['model_loads'] == 1, 'Child completion scope/count differs')
    for mode in ('smoke', 'evaluation'):
        require(result[mode] == pin(root / mode / 'completion.json', root), 'Worker accepted different completion')
    return result


def validate_supervision(root, plan):
    root = Path(root)
    receipt = json.loads((root / 'evaluation/supervisor-result.json').read_bytes())
    require(receipt['schema'] == SUPERVISION_SCHEMA and receipt['phase'] == 'evaluation'
            and receipt['status'] == 'accepted' and receipt['cleanup_verified'] is True
            and receipt['run_id'] == RUN_ID and receipt['plan_sha256'] == pin(root / 'plan.json')['sha256']
            and receipt['worker_result'] == pin(root / 'evaluation/evaluation-worker.json', root), 'Owned supervision acceptance differs')
    unit = receipt['unit']
    stopped = unit['stopped_state']
    require(unit['state'] == 'complete' and unit['cleanup_verified'] is True and unit['run_id'] == RUN_ID
            and unit['phase'] == 'evaluation' and unit['plan_sha256'] == receipt['plan_sha256']
            and unit['invocation'] and unit['instance_id']
            and (stopped.get('LoadState') == 'not-found' or stopped.get('ActiveState') in ('inactive', 'failed')), 'Owned process reap proof missing')
    return receipt
