"""Offline, CPU BF16 public_work research runner under an enforced Linux envelope.

Run in a root-owned transient systemd service with User=nonroot, PrivateNetwork=yes,
CPUQuota=400%, MemoryMax=32G, MemorySwapMax=0 and an independent RuntimeMaxSec.
The default 1800-second owned-process guard allows an explicit cap up to 3600.
No downloads, private reference reads, token clipping, retries or dtype fallback.
The internal --worker transport is authenticated and requires its live guard.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import qwen_linux_guard as guard
from prepare_public_work import SUITE_KINDS, TASKS

PROFILES = guard.EXACT_CPU_PROFILES
PINS = {'torch': '2.9.1+cpu', 'transformers': '4.57.6', 'tokenizers': '0.22.2',
        'safetensors': '0.8.0', 'peft': '0.17.1', 'accelerate': '1.15.0',
        'numpy': '2.5.4', 'psutil': '7.2.2'}
TEMPLATE_KWARGS = {'add_generation_prompt': True, 'enable_thinking': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def load_json(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular nonsymlink input required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def validate_pin(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str) and value['path']
            and type(value.get('bytes')) is int and value['bytes'] >= 0
            and isinstance(value.get('sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Invalid artifact pin')
    return value


def relative_file(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts, 'Expected relative confined artifact path')
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root), 'Artifact escapes its root')
    require(not any(part.is_symlink() for part in [path, *path.parents] if part != root.parent),
            'Symlink artifact path forbidden')
    return path


def check_pin(root, expected):
    validate_pin(expected)
    path = Path(expected['path'])
    if not path.is_absolute():
        path = relative_file(root, expected['path'])
    actual = pin(path)
    require(all(actual[key] == expected[key] for key in ('bytes', 'sha256')), 'Artifact pin mismatch')
    return path


def anchored_json(path, sha256):
    require(isinstance(sha256, str) and re.fullmatch('[0-9a-f]{64}', sha256), 'External SHA256 required')
    actual = pin(path)
    require(actual['sha256'] == sha256, 'External manifest SHA256 mismatch')
    return load_json(path), actual


def append_row(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                            allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        append_row(stream, value)


def replace_receipt(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    save_new(temporary, value)
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_public_rows(rows):
    require(isinstance(rows, list) and rows, 'Nonempty public input list required')
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'id', 'messages'}
                and isinstance(row['id'], str) and row['id'] and row['id'] not in seen,
                'Invalid/duplicate public input ID; private fields forbidden')
        seen.add(row['id'])
        messages = row['messages']
        require(isinstance(messages, list) and messages, 'Nonempty messages required')
        roles = []
        for message in messages:
            require(isinstance(message, dict) and set(message) == {'role', 'content'}
                    and message['role'] in ('system', 'user', 'assistant')
                    and isinstance(message['content'], str) and message['content'].strip(),
                    'Invalid public message; private fields forbidden')
            roles.append(message['role'])
        turns = roles[1:] if roles[0] == 'system' else roles
        require(turns and turns[-1] == 'user'
                and all(role == ('user' if index % 2 == 0 else 'assistant')
                        for index, role in enumerate(turns)), 'Invalid native dialogue ordering')
    return rows


def public_suite(path, sha256):
    manifest, identity = anchored_json(path, sha256)
    require(manifest.get('schema') == 'trid-public-work-suite-v1'
            and manifest.get('profile') == 'public_work_v1' and manifest.get('scope') in SUITE_KINDS
            and all(manifest.get(key) is False for key in
                    ('training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen')),
            'Only declared synthetic public-work suites are admitted')
    require(manifest.get('expected_tasks') == list(TASKS), 'Required task inventory mismatch')
    # Source/reference pins are metadata only. Never open either private artifact.
    root = Path(path).parent
    source = dict(validate_pin(manifest['source']))
    source_path = relative_file(root, source['path'])
    source['path'] = str(source_path)
    reference = validate_pin(manifest['references'])
    reference_path = relative_file(root, reference['path'])
    input_pin = validate_pin(manifest['inputs'])
    require(input_pin['path'] == 'inputs.json', 'Canonical public inputs.json path required')
    inputs = relative_file(root, input_pin['path'])
    for private_path in (source_path, reference_path):
        require(inputs.resolve() != private_path.resolve()
                and not (inputs.exists() and private_path.exists() and inputs.samefile(private_path)),
                'Public input path aliases private source/references')
    # Only after path and inode admission may public input contents be opened.
    check_pin(root, input_pin)
    rows = validate_public_rows(load_json(inputs))
    require(type(manifest.get('case_count')) is int and manifest['case_count'] == len(rows)
            and (len(rows) == 24 if manifest['scope'] == 'diagnostic' else len(rows) >= 1),
            'Suite case count mismatch')
    return rows, {'suite_manifest': identity, 'suite_scope': manifest['scope'],
                  'source': source, 'inputs': pin(inputs)}


def requested_policy(max_new_tokens, context_limit):
    require(type(max_new_tokens) is int and type(context_limit) is int
            and 0 < max_new_tokens < context_limit, 'Invalid token envelope')
    return {'profile': 'public_work_greedy_v1', 'do_sample': False, 'num_beams': 1,
            'max_new_tokens': max_new_tokens, 'context_limit': context_limit,
            'seed': 0, 'seed_policy': 'constant', 'mode': 'evaluation',
            'template_kwargs': {'enable_thinking': False}}


def preflight_prompts(tokenizer, rows, *, context_limit, max_new_tokens):
    requested_policy(max_new_tokens, context_limit)
    validate_public_rows(rows)  # Validate ALL rows before invoking any tokenizer.
    prepared = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False, **TEMPLATE_KWARGS)
        require(isinstance(prompt, str) and prompt, 'Native template must return nonempty text')
        # Official Instruct templates need not end in the older Qwen3 thinking suffix.
        tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
        require(isinstance(tokens, list) and tokens
                and all(type(token) is int and token >= 0 for token in tokens), 'Invalid prompt tokens')
        require(len(tokens) + max_new_tokens <= context_limit, 'Full preflight exceeds context; no truncation')
        prepared.append({'id': row['id'], 'prompt_sha256': object_sha(row['messages']),
                         'rendered_prompt': prompt,
                         'rendered_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'prompt_token_ids': tokens, 'input_tokens': len(tokens)})
    return prepared


def validate_architecture(config):
    require(config.get('model_type') == 'qwen3'
            and config.get('architectures') == ['Qwen3ForCausalLM']
            and config.get('torch_dtype', config.get('dtype')) == 'bfloat16'
            and not any(key in config for key in ('quantization_config', 'auto_map', 'compression_config')),
            'Only plain official Qwen3 BF16 is supported; no quantization/custom code/fallback')
    require(type(config.get('max_position_embeddings')) is int
            and config['max_position_embeddings'] > 0, 'Invalid model context limit')


def bf16_header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, 'Missing safetensors header')
        size = struct.unpack('<Q', prefix)[0]
        require(0 < size <= 16 * 1024 * 1024, 'Unbounded safetensors header')
        header = strict_json(stream.read(size).decode('utf-8'))
    tensors = {key: item for key, item in header.items() if key != '__metadata__'}
    require(tensors and all(isinstance(item, dict) and item.get('dtype') == 'BF16'
                           for item in tensors.values()), 'All base tensors must be stored BF16')
    return set(tensors)


def model_identity(root, manifest_path, manifest_sha256):
    manifest, manifest_pin = anchored_json(manifest_path, manifest_sha256)
    identity = {key: manifest.get(key) for key in ('repo_id', 'revision')}
    require(isinstance(identity['repo_id'], str) and identity['repo_id'].startswith('Qwen/Qwen3-')
            and isinstance(identity['revision'], str)
            and re.fullmatch('[0-9a-f]{40}', identity['revision']), 'Pinned official Qwen3 identity required')
    files = manifest.get('files')
    require(isinstance(files, list) and files, 'Acquisition file inventory required')
    inventory = {}
    for item in files:
        validate_pin(item)
        name = item['path']
        require(name not in inventory, 'Duplicate acquisition path')
        inventory[name] = item
        relative_file(root, name)
    require({'config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'} <= inventory.keys(),
            'Missing native configuration/tokenizer files')
    config_path = check_pin(root, inventory['config.json'])
    config = load_json(config_path)
    validate_architecture(config)  # Reject unsupported architecture before opening weight payloads.
    actual_names = {str(path.relative_to(root)) for path in Path(root).rglob('*') if path.is_file()}
    require(actual_names == set(inventory), 'Model directory differs from pinned acquisition inventory')
    actual = []
    for item in files:
        verified = check_pin(root, item)
        actual.append({'path': str(verified.resolve()), 'bytes': item['bytes'], 'sha256': item['sha256']})
    weights = {name for name in inventory if name.endswith('.safetensors')}
    require(weights, 'Missing safetensors weights')
    keys = set()
    for name in sorted(weights):
        current = bf16_header(relative_file(root, name))
        require(not keys.intersection(current), 'Duplicate base tensor across shards')
        keys.update(current)
    if 'model.safetensors.index.json' in inventory:
        index = load_json(relative_file(root, 'model.safetensors.index.json'))['weight_map']
        require(set(index) == keys and set(index.values()) == weights, 'Weight index coverage mismatch')
    else:
        require(weights == {'model.safetensors'}, 'Unindexed weight shards forbidden')
    tokenizer_files = [item for item in actual if Path(item['path']).name in
                       ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                        'special_tokens_map.json', 'vocab.json', 'merges.txt')]
    return {'model_manifest': manifest_pin, 'model_identity': identity, 'model_files': actual,
            'tokenizer_identity': {'files': tokenizer_files, 'template_sha256': None}}, config


def discard_verified_weight_cache(identity):
    """Request eviction after authentication, before resident model allocations.

    Advice is not proof of reclaimed pages. Do not hash again after this request:
    doing so would refill the same cache that the bounded native loader releases.
    """
    require(sys.platform == 'linux' and callable(getattr(os, 'posix_fadvise', None))
            and hasattr(os, 'POSIX_FADV_DONTNEED'),
            'Linux CPU runner requires POSIX_FADV_DONTNEED support')
    weights = [item for item in identity['model_files'] if item['path'].endswith('.safetensors')]
    require(weights, 'Verified weight file inventory required for cache advice')
    report = {'policy': 'drop_after_identity', 'advice': 'POSIX_FADV_DONTNEED',
              'requests': 0, 'bytes_requested': 0, 'reclaim_guaranteed': False, 'files': []}
    for item in weights:
        with Path(item['path']).open('rb', buffering=0) as source:
            require(os.fstat(source.fileno()).st_size == item['bytes'], 'Verified weight file size changed')
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        report['requests'] += 1
        report['bytes_requested'] += item['bytes']
        report['files'].append({'path': item['path'], 'bytes_requested': item['bytes']})
    return report


def adapter_identity(adapter, provenance_path, provenance_sha256, identity):
    provenance, provenance_pin = anchored_json(provenance_path, provenance_sha256)
    require(provenance.get('schema') == 'trid-public-evidence-adapter-v2'
            and provenance.get('fresh') is True and provenance.get('merged') is False,
            'Unsupported adapter provenance schema or representation')
    provenance_path, adapter = Path(provenance_path), Path(adapter)
    require(provenance_path.name == 'provenance.json' and provenance_path.parent.name == 'training'
            and adapter.resolve() == (provenance_path.parent / 'adapter').resolve()
            and provenance['training']['path'] == 'training/training.json'
            and provenance['adapter']['config']['path'] == 'training/adapter/adapter_config.json'
            and provenance['adapter']['weights']['path'] == 'training/adapter/adapter_model.safetensors',
            'Sealed export provenance paths differ')
    export_root = provenance_path.parent.parent
    training_path = check_pin(export_root, provenance['training'])
    training = load_json(training_path)
    require(training.get('base_model_id') == identity['repo_id']
            and training.get('base_revision') == identity['revision'] and training.get('family') == 'qwen3'
            and training.get('status') == 'complete'
            and type(training.get('steps')) is int and training['steps'] > 0
            and training['steps'] == provenance.get('observed_steps') == provenance['adapter'].get('step')
            and training.get('base_weights') == {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'},
            'Training provenance does not bind this BF16 base')
    files = [provenance['adapter']['config'], provenance['adapter']['weights']]
    actual = [pin(check_pin(export_root, item)) for item in files]
    config = load_json(Path(adapter) / 'adapter_config.json')
    lora = training['lora']
    require(config.get('peft_type') == 'LORA' and config.get('task_type') == 'CAUSAL_LM'
            and config.get('bias') == 'none' and not config.get('modules_to_save')
            and not any(config.get(key) for key in ('use_dora', 'use_rslora', 'use_qalora', 'rank_pattern',
                                                   'alpha_pattern', 'layer_replication', 'layers_to_transform',
                                                   'target_parameters', 'trainable_token_indices', 'lora_bias',
                                                   'fan_in_fan_out', 'loftq_config', 'megatron_config'))
            and config.get('r') == lora['rank'] and config.get('lora_alpha') == lora['alpha']
            and config.get('lora_dropout') == lora['dropout']
            and set(config.get('target_modules', [])) == set(lora['targets']), 'Unsupported/mismatched LoRA config')
    # base_model_name_or_path may be the original sealed training path. Do not rewrite it.
    return {'provenance': provenance_pin, 'training': pin(training_path), 'files': actual,
            'model_identity': identity, 'representation': 'unmerged_peft', 'steps': training['steps']}


def runtime_packages():
    actual = {name: importlib.metadata.version(name) for name in PINS}
    require(actual == PINS, 'CPU runtime package pins differ; no fallback')
    return actual


def enforced_envelope(profile, *, admission=False):
    require(sys.platform == 'linux' and os.getuid() != 0, 'Unprivileged Linux service required')
    root = guard.owned_cgroup()
    require(root.name.endswith('.service') and root.stat().st_uid == 0, 'Root-owned dedicated service required')
    boundary = guard.cgroup_metrics(root, profile=profile)
    quota = (root / 'cpu.max').read_text().split()
    require(len(quota) == 2 and quota[0] != 'max' and int(quota[0]) > 0
            and int(quota[1]) > 0 and int(quota[0]) == 4 * int(quota[1]), 'Exact CPUQuota400% required')
    interfaces = [line.split(':', 1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]]
    require(interfaces == ['lo'], 'PrivateNetwork loopback-only namespace required')
    host = guard.host_metrics()
    reason = guard.resource_reason(host, None, 0, admission=admission, profile=profile)
    require(reason is None, f'Host reserve admission failed: {reason}')
    return {'profile': profile, 'cgroup_path': str(root), 'cgroup': boundary,
            'cpu_max': quota, 'network_interfaces': interfaces, 'uid': os.getuid(), 'host': host}


def verify_worker_owner(output, seconds, profile):
    record = load_json(Path(output) / 'guard.json')
    require(profile in PROFILES and record.get('resource_profile') == profile and record.get('seconds') == seconds
            and record.get('child_pid') == os.getpid() and record.get('owned_pgid') == os.getpgrp()
            and record.get('supervisor_pid') == os.getppid()
            and record.get('first_stop_reason') is None
            and record.get('cgroup_path') == str(guard.owned_cgroup()), 'Live owned guard transport required')
    require(guard.process_stat(os.getppid())['state'] not in ('Z', 'X', 'x'), 'Guard is not alive')
    require(time.monotonic() < record['started_monotonic'] + seconds, 'Guard deadline expired')


def parameter_inventory(model, torch, adapter):
    inventory = {'base': {}, 'adapter': {}}
    for name, parameter in model.named_parameters():
        kind = 'adapter' if '.lora_' in name else 'base'
        require(parameter.device.type == 'cpu' and parameter.dtype ==
                (torch.float32 if kind == 'adapter' else torch.bfloat16)
                and not parameter.requires_grad, 'Frozen CPU BF16 base/FP32 adapter required')
        item = inventory[kind].setdefault(str(parameter.dtype), {'tensors': 0, 'elements': 0, 'bytes': 0})
        item['tensors'] += 1
        item['elements'] += parameter.numel()
        item['bytes'] += parameter.numel() * parameter.element_size()
    require(inventory['base'] and bool(inventory['adapter']) == adapter, 'Parameter arm inventory mismatch')
    require(not any(getattr(module, 'merged', False) for module in model.modules()), 'Merged adapter forbidden')
    return inventory


def generation_controls(model, tokenizer, max_new_tokens):
    from transformers import GenerationConfig
    from evaluate_evidence_conditioning import generation_target
    dispatch = {'use_model_defaults': False, 'do_sample': False, 'num_beams': 1}
    requested = GenerationConfig(bos_token_id=model.generation_config.bos_token_id,
                                 eos_token_id=model.generation_config.eos_token_id,
                                 pad_token_id=model.generation_config.pad_token_id)
    requested.update(do_sample=False, num_beams=1, num_beam_groups=1, max_new_tokens=max_new_tokens,
                     min_new_tokens=None, min_length=0, return_dict_in_generate=True,
                     output_scores=False, output_logits=False, num_return_sequences=1, use_cache=True,
                     temperature=1.0, top_p=1.0, top_k=50, typical_p=1.0, repetition_penalty=1.0,
                     no_repeat_ngram_size=0, forced_bos_token_id=None, forced_eos_token_id=None,
                     stop_strings=None, max_time=None)
    if requested.pad_token_id is None:
        requested.pad_token_id = tokenizer.pad_token_id
    effective, unused = generation_target(model)._prepare_generation_config(requested, **dispatch)
    require(not unused and effective.get_generation_mode().value == 'greedy_search', 'Greedy policy differs')
    return effective, dispatch


def stream_responses(model, tokenizer, rows, prepared, path, *, max_new_tokens, context_limit):
    import torch
    from evaluate_evidence_conditioning import observe_policy, decode_generation
    require([item['id'] for item in prepared] == [row['id'] for row in rows], 'Full ordered preflight required')
    effective, dispatch = generation_controls(model, tokenizer, max_new_tokens)
    config = effective.to_dict()
    with Path(path).open('x', encoding='utf-8') as stream:
        for row, prompt in zip(rows, prepared):
            started = time.monotonic()
            record = dict(prompt, generation_receipt='public_work_hf_v1', response='', raw_output='',
                          finish_reason='exception', error=None, output_token_ids=[], output_tokens=0,
                          terminal_stop_token_id=None, generation_config=config,
                          generation_policy=None, dispatch_kwargs=dispatch,
                          template_kwargs=dict(TEMPLATE_KWARGS), seed=0, mode='evaluation',
                          context_limit=context_limit, context_truncated=False, truncated_output=False)
            try:
                ids = torch.tensor([prompt['prompt_token_ids']], dtype=torch.long, device='cpu')
                torch.manual_seed(0)
                with torch.inference_mode(), observe_policy(model, effective, dispatch) as observed:
                    result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            generation_config=effective, **dispatch)
                record['generation_policy'] = dict(observed)
                output = result.sequences[0, prompt['input_tokens']:].tolist()
                record.update(output_token_ids=output, output_tokens=len(output))
                if output:
                    decoded = decode_generation(tokenizer, output, config['eos_token_id'], max_new_tokens)
                    record.update(response=decoded['response_text'], raw_output=decoded['raw_output'],
                                  finish_reason=decoded['termination_reason'],
                                  terminal_stop_token_id=decoded['terminal_stop_token_id'],
                                  truncated_output=decoded['termination_reason'] == 'length')
                else:
                    record['finish_reason'] = 'empty'
            except Exception as error:
                record['error'] = {'type': type(error).__name__, 'message': str(error)}
                record['finish_reason'] = 'exception'
            record['wall_seconds'] = time.monotonic() - started
            append_row(stream, record)


def worker(job_path, sha256):
    job, _ = anchored_json(job_path, sha256)
    output = Path(job['output'])
    verify_worker_owner(output, job['seconds'], job['profile'])
    resources = enforced_envelope(job['profile'])
    packages = runtime_packages()
    rows, suite = public_suite(job['suite_manifest'], job['manifest_sha256'])
    identity, config = model_identity(Path(job['model_root']), job['model_manifest'], job['model_manifest_sha256'])
    require(job['context_limit'] <= config['max_position_embeddings'], 'Requested context exceeds model limit')
    weight_cache_advice = discard_verified_weight_cache(identity)
    adapter = None
    if job['adapter'] is not None:
        adapter = adapter_identity(job['adapter'], job['adapter_provenance'],
                                   job['adapter_provenance_sha256'], identity['model_identity'])
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='',
                      OMP_NUM_THREADS=str(job['threads']), MKL_NUM_THREADS=str(job['threads']))
    import torch
    require(torch.__version__ == PINS['torch'] and torch.version.cuda is None
            and not torch.cuda.is_available(), 'CPU-only Torch build required')
    torch.set_num_threads(job['threads'])
    torch.set_num_interop_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(job['model_root'], local_files_only=True, trust_remote_code=False)
    template = tokenizer.get_chat_template()
    require(isinstance(template, str) and template, 'Official native template missing')
    identity['tokenizer_identity']['template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
    prepared = preflight_prompts(tokenizer, rows, context_limit=job['context_limit'],
                                 max_new_tokens=job['max_new_tokens'])
    runtime = {'schema': 'public_work_hf_runtime_v1', **suite, **identity, 'adapter': adapter,
               'packages': packages, 'resources': resources, 'device': 'cpu',
               'weight_cache_advice': weight_cache_advice,
               'generation_policy': requested_policy(job['max_new_tokens'], job['context_limit']),
               'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
               'torch_build': torch.__config__.show(), 'python': sys.version,
               'machine': os.uname().machine, 'status': 'preflight_complete',
               'research_only': True, 'quality_accepted': False}
    replace_receipt(output / 'runtime.json', runtime)
    # Every public row and provenance/config boundary is admitted before weight allocation.
    from train_open_weight_lora import load_qwen3_pread
    model, loader = load_qwen3_pread(job['model_root'], torch.bfloat16, 'cpu',
                                    cache_policy='drop_after_shard')
    if adapter is not None:
        from peft import PeftModel
        from aws_eval.public_evidence_worker import verify_loaded_adapter
        model = PeftModel.from_pretrained(model, job['adapter'], is_trainable=False,
                                         local_files_only=True, autocast_adapter_dtype=True)
        verify_loaded_adapter(model, Path(job['adapter']) / 'adapter_model.safetensors', torch)
    model.requires_grad_(False)
    model.eval()
    runtime.update(parameter_inventory=parameter_inventory(model, torch, adapter is not None),
                   loader=loader, status='loaded', attention_implementation=model.config._attn_implementation)
    replace_receipt(output / 'runtime.json', runtime)
    stream_responses(model, tokenizer, rows, prepared, output / 'responses.jsonl',
                     max_new_tokens=job['max_new_tokens'], context_limit=job['context_limit'])
    runtime['status'] = 'finished'
    replace_receipt(output / 'runtime.json', runtime)
    return 0


def finalize(output, expected_ids, record, error=None):
    output = Path(output)
    journal = output / 'responses.jsonl'
    if not journal.exists():
        with journal.open('x', encoding='utf-8') as stream:
            stream.flush()
            os.fsync(stream.fileno())
    observed, journal_error, exception_rows = [], None, 0
    with journal.open('rb') as stream:
        for line in stream:
            try:
                row = strict_json(line.decode('utf-8'))
                require(isinstance(row, dict) and row.get('id') in expected_ids
                        and row['id'] not in observed, 'Unknown/duplicate journal ID')
                observed.append(row['id'])
                exception_rows += row.get('error') is not None or row.get('finish_reason') == 'exception'
            except (ValueError, KeyError, TypeError) as exc:
                journal_error = str(exc)
                break  # Preserve the raw partial journal, never repair or invent rows.
    failure = error or record.get('first_stop_reason')
    if not failure and not guard.clean_exit(record):
        failure = record.get('stop_reason') or 'guard_cleanup_or_worker_failure'
    failure = failure or journal_error
    missing = [identifier for identifier in expected_ids if identifier not in observed]
    completion = {'schema': 'public_work_hf_completion_v1', 'expected_ids': expected_ids,
                  'observed_ids': observed, 'missing_ids': missing,
                  'complete': (not missing and failure is None and exception_rows == 0
                               and record.get('child_exit_code') == 0),
                  'exception_rows': exception_rows,
                  'worker_exit_code': record.get('child_exit_code'), 'guard_failure': failure,
                  'journal_error': journal_error, 'responses': pin(journal),
                  'runtime': pin(output / 'runtime.json'), 'research_only': True, 'quality_accepted': False}
    if (output / 'guard.json').is_file():
        completion['guard'] = pin(output / 'guard.json')
    save_new(output / 'completion.json', completion)
    return completion


def supervise(args):
    guard.validate_seconds(args.seconds, profile=args.profile)
    require(args.threads == 4, 'Explicit four CPU threads required')
    policy = requested_policy(args.max_new_tokens, args.context_limit)
    require(bool(args.adapter) == bool(args.adapter_provenance) == bool(args.adapter_provenance_sha256),
            'Adapter, provenance and external provenance hash must be supplied together')
    rows, suite = public_suite(args.suite_manifest, args.manifest_sha256)
    model_manifest, model_pin = anchored_json(args.model_manifest, args.model_manifest_sha256)
    output = args.output.resolve()
    protected = [args.suite_manifest.parent.resolve(), args.model_root.resolve()]
    if args.adapter:
        protected.append(args.adapter.parent.resolve())
    require(not any(output == path or output.is_relative_to(path) or path.is_relative_to(output)
                    for path in protected), 'Output must be separate from all input trees')
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    runtime = {'schema': 'public_work_hf_runtime_v1', **suite, 'model_manifest': model_pin,
               'model_identity': {key: model_manifest.get(key) for key in ('repo_id', 'revision')},
               'tokenizer_identity': {'files': [], 'template_sha256': None}, 'adapter': None,
               'packages': {}, 'resources': {}, 'generation_policy': policy, 'status': 'not_started',
               'research_only': True, 'quality_accepted': False}
    save_new(output / 'runtime.json', runtime)
    record, error = {}, None
    try:
        runtime['packages'] = runtime_packages()
        runtime['resources'] = enforced_envelope(args.profile, admission=True)
        replace_receipt(output / 'runtime.json', runtime)
        job = {key: (str(value.resolve()) if isinstance(value, Path) else value)
               for key, value in vars(args).items() if key not in ('worker', 'job_sha256')}
        job['output'] = str(output)
        save_new(output / 'worker-job.json', job)
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--worker',
                   str(output / 'worker-job.json'), '--job-sha256', pin(output / 'worker-job.json')['sha256']]
        record = guard.Guard(command, output / 'guard.json', args.seconds, profile=args.profile).run()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    completion = finalize(output, [row['id'] for row in rows], record, error)
    return 0 if completion['complete'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--job-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--suite-manifest', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--model-manifest', type=Path)
    parser.add_argument('--model-manifest-sha256')
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--adapter-provenance', type=Path)
    parser.add_argument('--adapter-provenance-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--context-limit', type=int, default=4096)
    parser.add_argument('--seconds', type=float, default=1800)
    parser.add_argument('--profile', choices=PROFILES, default=PROFILES[0])
    args = parser.parse_args(argv)
    if args.worker:
        try:
            return worker(args.worker, args.job_sha256)
        except Exception as error:
            print(f'{type(error).__name__}: {error}', file=sys.stderr)
            return 1
    required = ('suite_manifest', 'manifest_sha256', 'model_manifest', 'model_manifest_sha256', 'model_root', 'output')
    if any(getattr(args, name) is None for name in required):
        parser.error('Missing required ' + ', '.join('--' + name.replace('_', '-') for name in required))
    try:
        return supervise(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
