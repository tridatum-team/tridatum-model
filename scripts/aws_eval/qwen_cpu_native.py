"""Bound Linux native jobs, using the recovered evaluator's unchanged generation algorithms.

Only convert/probe/case are supported. No grading, selection, or final inputs are
read. Source archives stay immutable; the separately pinned static harness port
changes only Linux measurements and the already-consumed Linux startup gate.
A successful child receipt is provisional until validate_receipt observes the
outer guard's completed cleanup receipt.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import signal
import socket
import stat
import struct
import subprocess
import sys
import time

BASE_SHA = '29508ae00d82764d261287099ea3077dd08dcbf5cb86fe14427e9407b913c3c7'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
COMMIT = '8ca4060b7e30e76baaa5670ee1ec1b760b7f8552'
CONVERTER_SHA = 'ddb143ac34022b0646e51d9c35e9cfa3ec3d0a36091975ebc88f553dd18ca350'
PINS = {
    'harness_source': '9b07f62effa358ff059601849e01ab6be0dfb21541d878cc3b1a58ee5c1154a8',
    'development_source': '05286dacf7c6f4a7f810cf055b709f74382fd86818511952e6a66576a816365d',
    'dialogue_runner': 'e1a35e447565290fe49368a9b4ab0341ee3c0d6afcb2b5d1d6081772b33b7fda',
    'scenarios': 'a501be86d09bf01e614d8454775c8568e30a1c3b83dfcf99a25866f99f68ea52',
    'owner_scenarios_1': '13fa5a06f264678ee95611467b289f18bab81a5a15c07972ddd85af32824d4c8',
    'owner_scenarios_2': 'bc613942b59dd6ec6ace8bfe865012da2f591161bd52ae9bec1da1d3b61f47d7',
    'official_render': '7d3fa3cb7e04e10087cbfb94d1066213e01ac766f8def8dfa449761a945480e1',
    'probe_suite': '39ca251353fba46da2a72f097a127ba26a8c2b2121c8045322331abed6c9ccb0',
    'probe_plan': '4ff2a50cd94eed043076266e7d1a58ab60d826e60a3081be2fecf90ac2f27321',
    'source_training': '06ec73142360baec2f7c60ab81c2c15ce47b361dadb01c8b03a44619d073d870',
    'source_weights': 'b92d1e73100bb313c1b86f37686b1a50c0cbecd20849647ff6d026a0fd9dfaee',
    'source_config': 'a7f9204434bc0a509e3697856e0cdbeff7d10047f283275d1bf6d78fe4d572dd',
}
GIB = 1024 ** 3
BUILD_RECEIPT = Path('/var/lib/qwen013/native-build.json')
SERVER = Path('/Users/eomgyuhyeon/사업/trid/runs/tools/llama.cpp-8ca4060/llama-server')
CONVERTER = Path('/opt/qwen013-native/source/convert_lora_to_gguf.py')
CONVERTER_PYTHON = Path('/opt/qwen013-native/converter-venv/bin/python')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def bound(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), 'Not a regular input: ' + str(path))
    return {'path': str(path), 'sha256': digest(path)}


def check_bound(item):
    require(isinstance(item, dict) and set(('path', 'sha256')) <= item.keys(), 'Missing file binding')
    require(digest(item['path']) == item['sha256'], 'Input hash changed: ' + item['path'])
    return Path(item['path'])


def resolve_input(root, item):
    path = Path(item['path'])
    path = path if path.is_absolute() else Path(root) / path
    result = bound(path)
    require(result['sha256'] == item['sha256'], 'Predeclared input changed: ' + str(path))
    return result


def source_inputs(root, native):
    required = set(PINS) | {'manifest', 'hf_config', 'linux_harness'}
    require(required <= native['inputs'].keys(), 'Native input declarations incomplete')
    result = {key: resolve_input(root, native['inputs'][key]) for key in sorted(required)}
    for key, expected in PINS.items():
        require(result[key]['sha256'] == expected, 'Frozen source pin mismatch: ' + key)
    return result


def original_development(inputs):
    """Compile only the original read-only scenario/parity algorithms, not its Mac controller."""
    source = check_bound(inputs['development_source']).read_text()
    require(inputs['development_source']['sha256'] == PINS['development_source'], 'Unreviewed development source')
    tree = ast.parse(source)
    functions = {'load_scenarios', 'compare_with_baseline'}
    constants = {'EXPECTED_BINDING', 'SHARED_IDENTITY'}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in functions) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    require(len(nodes) == 4, 'Original algorithm declaration count changed')
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib, 'sha256': digest,
                 'THREADS': 4, 'CONTEXT': 2048, 'MAX_NEW': 384,
                 'GROUPS': ('domain', 'uncertainty', 'retention'), 'GROUP_SIZE': 12,
                 'SCENARIOS': check_bound(inputs['scenarios']),
                 'OWNER_SCENARIOS': {check_bound(inputs[k]): inputs[k]['sha256']
                                     for k in ('owner_scenarios_1', 'owner_scenarios_2')}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(inputs['development_source']['path']), 'exec'), namespace)
    return namespace


def native_cases(plan):
    native = plan['native']
    root = Path(native.get('root', plan.get('root', Path(__file__).resolve().parents[2]))).resolve()
    inputs = source_inputs(root, native)
    scenarios = original_development(inputs)['load_scenarios']()
    cases = [{'id': sid, **row} for sid, row in scenarios.items()]
    require(len(cases) == 36 and sum(len(c['users']) for c in cases) == 46, 'Frozen DEV36/46 shape changed')
    return cases


def checkpoint_binding(root, checkpoint, inputs, *, runtime=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    if checkpoint.is_file():
        checkpoint = checkpoint.parent.parent
    weights = bound(checkpoint / 'adapter/adapter_model.safetensors')
    config = bound(checkpoint / 'adapter/adapter_config.json')
    training = bound(checkpoint / 'training.json')
    trained = load(training['path'])
    require(trained.get('status') == 'checkpoint' and trained.get('steps') in (2, 8, 16, 23), 'Not an admitted packed checkpoint')
    if runtime is None:
        require(trained.get('device') != 'cuda' and 'cuda_runtime' not in trained,
                'CUDA checkpoint requires an explicit authenticated runtime')
    if runtime is not None:
        require(runtime.get('schema') == 'trid-cuda-runtime-v1'
                and runtime.get('architecture') == 'x86_64' and runtime.get('device') == 'cuda'
                and runtime.get('dtype') == 'torch.float32'
                and runtime.get('trainer_entry') == 'scripts/train_qwen_preferences_cuda.py'
                and trained.get('cuda_runtime') == runtime and trained.get('engine') == runtime.get('engine')
                and isinstance(runtime.get('engine'), dict) and bool(runtime['engine'])
                and trained.get('versions') == runtime.get('versions')
                and trained.get('device') == 'cuda' and trained.get('dtype') == 'torch.float32',
                'Checkpoint CUDA runtime/engine mismatch')
        require(isinstance(trained.get('cuda_runtime_sha256'), str)
                and len(trained['cuda_runtime_sha256']) == 64, 'Checkpoint CUDA manifest binding missing')
    record = trained.get('checkpoint', {})
    require(record.get('adapter_weights_sha256') == weights['sha256'] and record.get('step') == trained['steps'], 'Checkpoint/training hash or step mismatch')
    require(Path(record['path']).resolve() == checkpoint / 'adapter', 'Checkpoint adapter path mismatch')
    require(trained.get('base_revision') == REVISION and trained.get('manifest_sha256') == inputs['manifest']['sha256'], 'Training base/manifest mismatch')
    require(trained.get('base_weights', {}).get('gguf_sha256') == BASE_SHA and trained['base_weights'].get('representation') == 'q8_0', 'Wrong serving source GGUF')
    ancestry_pins = {'adapter_weights_sha256': PINS['source_weights'],
                     'adapter_config_sha256': PINS['source_config'], 'training_json_sha256': PINS['source_training']}
    for name in ('init_adapter', 'reference'):
        ancestry = trained.get(name, {})
        require(all(ancestry.get(k) == v for k, v in ancestry_pins.items()) and
                ancestry.get('source_steps') == 24 and ancestry.get('source_cumulative_steps') == 96,
                'Source011 ancestry mismatch: ' + name)
    init = trained['init_adapter']
    require(Path(init['path']).resolve() == Path(inputs['source_weights']['path']).parent.resolve() and
            init.get('optimizer', '').startswith('reset: fresh AdamW') and
            init.get('scheduler', '').startswith('reset: fresh warmup+cosine'), 'Nonfresh source011 initialization')
    cfg = load(config['path'])
    require(cfg.get('peft_type') == 'LORA' and cfg.get('r') == 16 and cfg.get('lora_alpha') == 32 and
            set(cfg.get('target_modules', [])) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'} and
            not cfg.get('use_dora') and not cfg.get('use_rslora') and not cfg.get('modules_to_save'), 'Unexpected LoRA configuration')
    with Path(weights['path']).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        require(0 < size < 1024 * 1024, 'Invalid safetensors header size')
        metadata = json.loads(stream.read(size))
    tensors = {k: v for k, v in metadata.items() if k != '__metadata__'}
    require(len(tensors) == 392 and all(v.get('dtype') == 'F32' and len(v.get('shape', [])) == 2 and
            16 in v['shape'] and ('.lora_A.' in k or '.lora_B.' in k) for k, v in tensors.items()), 'Expected 392 rank16 F32 LoRA tensors')
    return {'directory': str(checkpoint), 'weights': weights, 'config': config, 'training': training,
            'step': trained['steps'], 'manifest_sha256': trained['manifest_sha256'], 'tensor_metadata': tensors}


def root_controlled(path):
    path = Path(path)
    require(path.is_absolute(), 'Root artifact must be absolute')
    for part in (path, *path.parents):
        info = part.stat()
        require(info.st_uid == 0 and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH), 'Artifact not root-controlled: ' + str(part))
    require(not path.is_symlink(), 'Root artifact cannot be symlink: ' + str(path))


def build_binding(root, native, *, runtime=None):
    path = Path(native['build_receipt'])
    require(path == BUILD_RECEIPT, 'Unexpected native build receipt location')
    root_controlled(path)
    receipt = load(path)
    require(receipt.get('completed') is True and receipt.get('source_commit') == COMMIT, 'Native build not complete/pinned')
    archive = resolve_input(root, {'path': native['build']['source_archive'],
                                   'sha256': native['build']['source_archive_sha256']})
    require(receipt.get('source_archive') == archive, 'Native source archive binding mismatch')
    server, converter = receipt['server'], receipt['converter']
    require(Path(server['path']) == SERVER and Path(converter['path']) == CONVERTER and
            Path(converter['python']) == CONVERTER_PYTHON and converter['sha256'] == CONVERTER_SHA,
            'Unapproved native runtime/converter')
    require(server.get('build') == 'b1-8ca4060' and '8ca4060' in server.get('version', ''), 'Native build metadata mismatch')
    require(receipt.get('training_versions_before') == receipt.get('training_versions_after'), 'Native setup changed training environment')
    if runtime is None:
        require(receipt.get('architecture', 'aarch64') == 'aarch64',
                'Non-CPU-default build requires an explicit authenticated runtime')
    if runtime is not None:
        validate_build_runtime(receipt, runtime)
    for item in [server, converter, *server['shared_libraries']]:
        root_controlled(item['path'])
        check_bound(item)
    # Python is normally a root-owned venv symlink; validate its resolved target and parents.
    root_controlled(CONVERTER_PYTHON.resolve(strict=True))
    return {'receipt': bound(path), 'record': receipt}


def validate_build_runtime(receipt, runtime):
    require(receipt.get('architecture') == runtime['architecture'] == 'x86_64'
            and 'x86_64' in receipt['server'].get('version', '')
            and receipt['converter'].get('torch_cuda') is None
            and receipt['converter'].get('torch_version') == '2.11.0+cpu'
            and receipt['converter']['versions'].get('torch') == '2.11.0+cpu'
            and not any(name.startswith(('nvidia-', 'cuda-'))
                        for name in receipt['converter']['versions']),
            'Native x86_64 build/CPU converter identity mismatch')
    flags = receipt.get('cmake_flags', [])
    require('-DGGML_CUDA=OFF' in flags and '-DGGML_NATIVE=OFF' in flags
            and not any(flag.startswith('-DGGML_CPU_ARM_ARCH=') for flag in flags),
            'Native CPU-only architecture flags mismatch')


def build_job(mode, root, output, checkpoint, plan, guard_receipt, case=None, adapter_dir=None, *, runtime=None):
    require(mode in ('convert', 'probe', 'case'), 'Only convert/probe/case are admitted')
    root = Path(root).resolve(strict=True)
    native = plan['native']
    inputs = source_inputs(root, native)
    base = resolve_input(root, native['base_gguf'])
    require(base['sha256'] == BASE_SHA, 'Wrong native base (site base cannot substitute)')
    job = {'schema_version': 1, 'mode': mode, 'root': str(root), 'output': str(Path(output).resolve()),
           'guard_receipt': str(Path(guard_receipt).resolve()), 'inputs': inputs, 'base_gguf': base,
           'checkpoint': checkpoint_binding(root, checkpoint, inputs, runtime=runtime),
           'build': build_binding(root, native, runtime=runtime),
           'executor': bound(Path(__file__)), 'guard_source': bound(root / 'scripts/qwen_linux_guard.py')}
    if runtime is not None:
        job['cuda_runtime'] = runtime
    if mode != 'convert':
        directory = Path(adapter_dir) if adapter_dir else Path(job['checkpoint']['directory']) / 'native-conversion'
        conversion_path = directory / 'conversion.json'
        conversion = load(conversion_path)
        validate_conversion(job, conversion)
        job['conversion'] = {'receipt': bound(conversion_path), 'record': conversion,
                             'adapter_gguf': bound(directory / 'adapter.gguf')}
        require(job['conversion']['adapter_gguf'] == conversion['adapter_gguf'],
                'Converted adapter path/hash does not match its receipt')
    if mode == 'probe':
        require(job['checkpoint']['step'] == 2, 'TRAIN feature proof must use two-update smoke checkpoint')
    if mode == 'case':
        require(job['checkpoint']['step'] in (8, 16, 23), 'DEV checkpoint is not 8/16/23')
        scenarios = original_development(inputs)['load_scenarios']()
        require(isinstance(case, dict) and case.get('id') in scenarios, 'Unknown DEV case')
        expected = {'id': case['id'], **scenarios[case['id']]}
        require(case == expected, 'Case input does not match original order/users')
        job['case'] = expected
        group = case['group']
        execution = resolve_input(root, native['baseline_executions'][group])
        baseline = resolve_input(root, native['baseline_receipts'][case['id']])
        frozen = load(execution['path'])
        rows = frozen.get('cases', [])
        matches = [r for r in rows if r.get('id') == case['id']]
        require(frozen.get('state') == 'COMPLETE' and frozen.get('arm') == 'model' and
                frozen.get('requested') == frozen.get('completed') == 12 and frozen.get('remaining_ids') == [] and
                len(matches) == 1 and matches[0].get('verified') is True and matches[0].get('returncode') == 0,
                'Frozen baseline execution incomplete')
        require(matches[0].get('harness_receipt_sha256') == baseline['sha256'], 'Baseline execution/receipt mismatch')
        job['baseline'] = {'execution': execution, 'receipt': baseline}
    job['job_sha256'] = canonical(job)
    return job


def validate_job(job):
    unsigned = {k: v for k, v in job.items() if k != 'job_sha256'}
    require(canonical(unsigned) == job.get('job_sha256'), 'Native job binding changed')
    require(job.get('mode') in ('convert', 'probe', 'case'), 'Unsupported native job')
    require(all(job['inputs'].get(key, {}).get('sha256') == expected for key, expected in PINS.items()),
            'Native job changed a frozen source pin')
    require(job['base_gguf']['sha256'] == BASE_SHA, 'Native job changed the source GGUF')
    for item in [*job['inputs'].values(), job['base_gguf'], job['executor'], job['guard_source'],
                 job['checkpoint']['weights'], job['checkpoint']['config'], job['checkpoint']['training'], job['build']['receipt']]:
        check_bound(item)
    require(job['executor']['sha256'] == digest(__file__), 'Executor source differs from job')
    runtime = job.get('cuda_runtime')
    require(checkpoint_binding(Path(job['root']), job['checkpoint']['directory'], job['inputs'],
                               runtime=runtime) == job['checkpoint'],
            'Checkpoint facts differ from bound adapter/config/training inputs')
    require(load(job['build']['receipt']['path']) == job['build']['record'], 'Build receipt record changed')
    if runtime is not None:
        validate_build_runtime(job['build']['record'], runtime)
    root_controlled(job['build']['receipt']['path'])
    for item in [job['build']['record']['server'], job['build']['record']['converter'],
                 *job['build']['record']['server']['shared_libraries']]:
        root_controlled(item['path'])
        check_bound(item)
    if 'conversion' in job:
        check_bound(job['conversion']['receipt'])
        check_bound(job['conversion']['adapter_gguf'])
        require(load(job['conversion']['receipt']['path']) == job['conversion']['record'], 'Conversion receipt changed')
        require(job['conversion']['adapter_gguf'] == job['conversion']['record']['adapter_gguf'],
                'Converted adapter identity changed')
        validate_conversion(job, job['conversion']['record'])
    if 'baseline' in job:
        for item in job['baseline'].values():
            check_bound(item)


def guard_module(job):
    path = check_bound(job['guard_source'])
    spec = importlib.util.spec_from_file_location('_qwen_native_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_guard(job):
    require(sys.platform == 'linux' and os.geteuid() != 0 and pwd.getpwuid(os.geteuid()).pw_name == 'qwen-smoke',
            'Native jobs require unprivileged Linux qwen-smoke')
    guard = guard_module(job)
    receipt_path = Path(job['guard_receipt'])
    info = receipt_path.stat()
    require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600, 'Unowned guard receipt')
    record = load(receipt_path)
    parent = os.getppid()
    me = guard.process_stat(os.getpid())
    require(record.get('supervisor_pid') == parent and record.get('child_pid') == os.getpid() and
            record.get('owned_pgid') == me['pgid'] == os.getpid() and record.get('seconds') == 180 and
            record.get('first_stop_reason') is None and record.get('reaped') is False, 'Guard does not own native PID/PGID')
    command = Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    require(os.fsencode(job['guard_source']['path']) in command and os.fsencode(job['guard_receipt']) in command,
            'Actual parent is not the declared Linux guard')
    actual_argv = [os.fsdecode(part) for part in Path('/proc/self/cmdline').read_bytes().split(b'\0') if part]
    require(actual_argv == record.get('argv'), 'Actual native process argv differs from guard ownership')
    require(Path(f'/proc/{parent}/cgroup').read_text() == Path('/proc/self/cgroup').read_text(), 'Guard cgroup differs')
    cgroup = guard.owned_cgroup()
    require(str(cgroup) == record.get('cgroup_path'), 'Guard cgroup path mismatch')
    metrics = guard.cgroup_metrics(cgroup)
    require(metrics['memory_max_bytes'] == 6 * GIB and metrics['swap_max_bytes'] == 0, 'Native cgroup boundary differs')
    owner = record['owner']
    require(guard.process_stat(owner['pid'])['start_ticks'] == owner['start_ticks'], 'Guard owner process changed')
    runtime = job.get('cuda_runtime')
    if runtime is not None:
        uuid = runtime['gpu_uuid']
        require(record.get('gpu', {}).get('expected_uuid') == uuid
                and b'--gpu-uuid' in command
                and command[command.index(b'--gpu-uuid') + 1] == os.fsencode(uuid),
                'Active native GPU guard binding missing')
        baseline = record['samples'][0].get('gpu')
        require(isinstance(baseline, dict), 'Active native GPU admission evidence missing')
        before = guard.group_members(me['pgid'])
        actual_gpu = guard.gpu_metrics(uuid)
        observations = []
        for process in actual_gpu['compute_processes']:
            pid = process['pid']
            try:
                after = guard.process_stat(pid)
            except (FileNotFoundError, ProcessLookupError):
                after = None
            observations.append({'pid': pid, 'after': after})
        members = guard.resolve_gpu_ownership(actual_gpu, before, observations, me['pgid'])
        require(guard.gpu_resource_reason(actual_gpu, baseline, members) is None,
                'Active native GPU resource/ownership failure')
        require(all(actual_gpu.get(key) == value for key, value in runtime['gpu'].items()),
                'Active native GPU differs from authenticated training runtime')
    # BOOTSTRAP consumes G before exec; never read a second byte or fabricate Mac pressure.
    return {'platform': sys.platform, 'pid': os.getpid(), 'parent_pid': parent, 'pgid': me['pgid'],
            'cgroup_path': str(cgroup), 'cgroup': metrics, 'host': guard.host_metrics(),
            'guard_receipt': str(receipt_path), 'guard_snapshot': record}


def gguf_metadata(path):
    """Read real GGUF descriptors without importing torch/numpy or loading tensor payloads."""
    with Path(path).open('rb') as stream:
        def unpack(fmt):
            size = struct.calcsize('<' + fmt)
            data = stream.read(size)
            require(len(data) == size, 'Truncated GGUF')
            return struct.unpack('<' + fmt, data)[0]
        def string():
            size = unpack('Q')
            require(size <= 16 * 1024 * 1024, 'Oversized GGUF string')
            data = stream.read(size)
            require(len(data) == size, 'Truncated GGUF string')
            return data.decode('utf-8')
        def value(kind):
            scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
            if kind in scalar:
                return unpack(scalar[kind])
            if kind == 8:
                return string()
            require(kind == 9, 'Unknown GGUF metadata type')
            subtype, count = unpack('I'), unpack('Q')
            require(count < 1000000, 'Oversized GGUF array')
            return [value(subtype) for _ in range(count)]
        require(stream.read(4) == b'GGUF' and unpack('I') == 3, 'Expected GGUF v3')
        count, entries = unpack('Q'), unpack('Q')
        require(count == 392 and entries < 10000, 'Expected exactly 392 converted tensors')
        metadata = {}
        for _ in range(entries):
            name = string()
            require(name not in metadata, 'Duplicate GGUF metadata')
            metadata[name] = value(unpack('I'))
        tensors = []
        for _ in range(count):
            name, dimensions = string(), unpack('I')
            require(dimensions == 2, 'Adapter tensor must be rank two')
            shape = [unpack('Q') for _ in range(dimensions)]
            kind, offset = unpack('I'), unpack('Q')
            require(kind == 0 and 16 in shape and name.endswith(('.lora_a', '.lora_b')), 'Non-F32/rank16 adapter tensor')
            tensors.append({'name': name, 'shape': shape, 'ggml_type': kind, 'offset': offset})
        require(len({t['name'] for t in tensors}) == 392, 'Duplicate GGUF tensors')
        require(metadata.get('general.architecture') == 'qwen3' and metadata.get('general.type') == 'adapter' and
                metadata.get('adapter.type') == 'lora' and metadata.get('adapter.lora.alpha') == 32, 'GGUF adapter metadata mismatch')
        alignment = metadata.get('general.alignment', 32)
        require(type(alignment) is int and alignment > 0, 'Invalid GGUF alignment')
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        size = Path(path).stat().st_size
        ends = sorted((t['offset'], t['offset'] + math.prod(t['shape']) * 4) for t in tensors)
        require(all(data_start + end <= size for _, end in ends) and
                all(ends[i][1] <= ends[i + 1][0] for i in range(len(ends) - 1)), 'GGUF tensor data bounds overlap/truncate')
        return {'metadata': metadata, 'tensors': tensors, 'tensor_count': count, 'data_start': data_start}


def validate_conversion(job, conversion):
    checkpoint = job['checkpoint']
    require(conversion.get('status') == 'complete' and conversion.get('checkpoint') == checkpoint and
            conversion.get('build') == job['build'] and conversion.get('source_gguf') == job['base_gguf'] and
            conversion.get('hf_config') == job['inputs']['hf_config'], 'Conversion/checkpoint/source binding mismatch')
    check_bound(conversion['adapter_gguf'])
    require(conversion.get('gguf') == gguf_metadata(conversion['adapter_gguf']['path']), 'Converted tensor metadata changed')
    require(conversion.get('converter_sha256') == CONVERTER_SHA and conversion.get('returncode') == 0,
            'Conversion did not use pinned successful converter')
    config = check_bound(conversion['config_only'])
    require(config.name == 'config.json' and conversion['config_only']['sha256'] == job['inputs']['hf_config']['sha256']
            and {p.name for p in config.parent.iterdir()} == {'config.json'}, 'Converter base was not config-only')
    expected_argv = [str(CONVERTER_PYTHON), '-B', str(CONVERTER), '--base', str(config.parent),
                     '--outfile', conversion['adapter_gguf']['path'], '--outtype', 'f32',
                     str(Path(checkpoint['weights']['path']).parent)]
    require(conversion.get('argv') == expected_argv, 'Converter argv differs from pinned config-only F32 command')
    check_bound(conversion['log'])
    if job['mode'] != 'convert':
        validate_final_guard({**job, 'guard_receipt': conversion['guard_receipt']},
                             {'ownership': conversion['ownership'], 'argv': conversion['guard_argv']})


def convert(job, ownership):
    out = Path(job['output'])
    config_dir = out / 'hf-config-only'
    config_dir.mkdir()
    config = check_bound(job['inputs']['hf_config'])
    (config_dir / 'config.json').write_bytes(config.read_bytes())
    require(set(p.name for p in config_dir.iterdir()) == {'config.json'}, 'Converter base must be config-only')
    target = out / 'adapter.gguf'
    argv = [str(CONVERTER_PYTHON), '-B', str(CONVERTER), '--base', str(config_dir), '--outfile', str(target),
            '--outtype', 'f32', str(Path(job['checkpoint']['weights']['path']).parent)]
    started = time.monotonic()
    with (out / 'converter.log').open('xb') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=165, check=False,
                                env={'PATH': os.defpath, 'HOME': str(out), 'PYTHONNOUSERSITE': '1',
                                     'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    require(result.returncode == 0, 'Pinned adapter converter failed')
    record = {'status': 'complete', 'checkpoint': job['checkpoint'], 'build': job['build'],
              'source_gguf': job['base_gguf'], 'hf_config': job['inputs']['hf_config'],
              'config_only': bound(config_dir / 'config.json'),
              'guard_receipt': job['guard_receipt'], 'ownership': ownership,
              'guard_argv': ownership['guard_snapshot']['argv'], 'conversion_job_sha256': job['job_sha256'],
              'converter_sha256': CONVERTER_SHA, 'argv': argv, 'returncode': result.returncode,
              'elapsed_seconds': time.monotonic() - started, 'adapter_gguf': bound(target),
              'gguf': gguf_metadata(target), 'log': bound(out / 'converter.log')}
    save(out / 'conversion.json', record)
    return {'adapter_gguf': record['adapter_gguf'], 'conversion_receipt': bound(out / 'conversion.json')}


def linux_harness(job, ownership):
    """Load the separately SHA-bound static Linux port and the frozen request helper."""
    runner = check_bound(job['inputs']['dialogue_runner'])
    runner_spec = importlib.util.spec_from_file_location('base_dialogue_runner', runner)
    runner_module = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_module)
    sys.modules['base_dialogue_runner'] = runner_module
    path = check_bound(job['inputs']['linux_harness'])
    spec = importlib.util.spec_from_file_location('_qwen_native_linux_harness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.NATIVE_JOB = job
    module.T_START = ownership['started_monotonic']
    return module, {'original_source': job['inputs']['harness_source'],
                    'linux_source': job['inputs']['linux_harness'], 'ownership': ownership}


def validate_harness(job, record):
    expected = original_development(job['inputs'])['EXPECTED_BINDING']
    require(record.get('status') == 'complete' and record.get('arms') == ['adapter'] and
            record.get('backend') == 'base_gguf+lora' and record.get('model_sha256') == BASE_SHA and
            record.get('model_file') == job['base_gguf']['path'], 'Native identity/status mismatch')
    adapter = job['conversion']['adapter_gguf']
    require(record.get('lora_file') == adapter['path'] and record.get('lora_sha256') == adapter['sha256'], 'Wrong LoRA receipt identity')
    require(all(record.get('binding', {}).get(k) == v for k, v in expected.items()), 'Native generation binding changed')
    require(record.get('env', {}).get('LD_LIBRARY_PATH') == str(SERVER.parent), 'Wrong native library environment')
    require(record.get('timeouts', {}).get('duration') == 175 and record['timeouts'].get('ready') == 30, 'Native timeout contract changed')
    parity = record.get('render_parity', {})
    require(all(parity.get(k) is True for k in ('text_equal', 'token_ids_equal', 'no_think_suffix')) and
            parity.get('official_render') == job['inputs']['official_render']['path'], 'Official render parity failed')
    server = record.get('server', {})
    require(server.get('build') == job['build']['record']['server']['build'] and server.get('chat_template_sha256'), 'Actual server build/template mismatch')
    adapters = server.get('lora_adapters', [])
    require(len(adapters) == 1 and adapters[0].get('path') == adapter['path'] and adapters[0].get('scale') == 1.0,
            'Actual active LoRA path/scale mismatch')
    require([s.get('arm') for s in record.get('arm_switches', [])] == ['adapter'] and
            all(s.get('scale') == 1.0 for s in record['arm_switches']), 'Adapter activation missing')
    require(record.get('live_members_after_shutdown') == [] and record.get('memory', {}).get('under_ceiling') is True and
            not record.get('server_killed') and record.get('stop_signal') is None, 'Native resource/shutdown failure')
    suite_key = 'probe_suite' if job['mode'] == 'probe' else 'scenarios'
    require(record.get('suite_sha256') == job['inputs'][suite_key]['sha256'] and
            record.get('suite_file') == job['inputs'][suite_key]['path'], 'Native suite identity mismatch')
    if job['mode'] == 'probe':
        planned = load(job['inputs']['probe_plan']['path'])['rows']
        require(len(planned) == 4, 'Probe must have exactly four TRAIN rows')
        specs = [(r['id'], 0, r['user']) for r in planned]
        require(record.get('ids') is None, 'Probe selection was narrowed')
    else:
        case = job['case']
        specs = [(case['id'], i, user) for i, user in enumerate(case['users'])]
        require(record.get('ids') == [case['id']], 'DEV case selection mismatch')
    rows = record.get('requests', [])
    require(len(rows) == len(specs), 'Missing or repeated native turns')
    settings = None
    for index, (row, (sid, turn, user)) in enumerate(zip(rows, specs)):
        require(row.get('kind') == 'native' and row.get('arm') == 'adapter' and row.get('session') == sid and
                row.get('turn') == turn and row.get('user') == user and row.get('history_turns') == turn and
                row.get('reset') is False and row.get('http_status') == 200 and row.get('fits_context') is True and
                isinstance(row.get('output'), str), 'Native turn/context/history failure')
        require(row.get('rendered_prompt_sha256') == hashlib.sha256(row['prompt'].encode()).hexdigest() and
                row.get('prompt_token_ids_sha256') == hashlib.sha256(json.dumps(row['prompt_token_ids']).encode()).hexdigest() and
                len(row['prompt_token_ids']) + 384 <= 2048, 'Native render/token digest mismatch')
        require(isinstance(row.get('pressure'), dict) and 'full' in row['pressure'], 'Missing real Linux PSI')
        require(row.get('generation_settings'), 'Missing full generation settings')
        settings = settings or row['generation_settings']
        require(row['generation_settings'] == settings, 'Generation settings changed between turns')
        if job['mode'] == 'probe':
            require(row.get('stopped_at_eos') is True and row.get('stop_type') == 'eos' and
                    not row.get('truncated') and row['output'].strip(), 'TRAIN EOS/nonempty feature proof failed')
            original = planned[index]
            require(original['messages'] == [{'role': 'user', 'content': user}] and
                    row['prompt'] == original['rendered_prompt'] and row['prompt_token_ids'] == original['prompt_token_ids'],
                    'TRAIN exact render/token parity failed')
    first = rows[0]
    return {'status': 'complete', 'turns_total': len(rows), 'turns_completed': len(rows),
            'seconds': round(sum(r['wall_seconds'] for r in rows), 3), 'elapsed_seconds': record['elapsed_seconds'],
            'startup_seconds': record['timeouts']['startup'], 'request_timeout': record['timeouts']['request'],
            'server_build': server['build'], 'chat_template_sha256': server['chat_template_sha256'],
            'generation_settings_sha256': hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
            'turn0_rendered_prompt_sha256': first['rendered_prompt_sha256'],
            'turn0_prompt_token_ids_sha256': first['prompt_token_ids_sha256'],
            'prompt_tokens': [r['prompt_tokens'] for r in rows], 'generated_tokens': [r['generated_tokens'] for r in rows],
            'stop_type': [r['stop_type'] for r in rows], 'stopped_at_eos': all(r.get('stopped_at_eos') for r in rows),
            'truncated': any(bool(r.get('truncated')) for r in rows),
            'fits_context': True, 'empty_output': any(not r['output'].strip() for r in rows),
            'peak_group_rss_bytes': record['memory']['group_rss_bytes'],
            'pressure': [r['pressure'] for r in rows]}


def baseline_parity(job, harness_path):
    original = check_bound(job['baseline']['receipt'])
    require(load(original).get('model_sha256') == BASE_SHA, 'Wrong baseline source base')
    result = original_development(job['inputs'])['compare_with_baseline'](
        Path(harness_path), original, job['case']['users'])
    require(result['equal'] is True and all(result['checks'].values()),
            'Original baseline parity failed: ' + str(result['checks']))
    return result


def run_harness(job, ownership):
    module, port_record = linux_harness(job, ownership)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    require(port not in module.LIVE_DEMO_PORTS, 'Allocated port conflicts with frozen exclusions')
    out = Path(job['output'])
    harness_path = out / 'harness.json'
    suite = job['inputs']['probe_suite' if job['mode'] == 'probe' else 'scenarios']['path']
    adapter = job['conversion']['adapter_gguf']
    argv = ['qwen-native-linux-port', '--model-file', job['base_gguf']['path'], '--expected-sha256', BASE_SHA,
            '--lora', adapter['path'], '--lora-sha256', adapter['sha256'], '--suite', suite,
            '--receipt', str(harness_path), '--server-log', str(out / 'server.log'), '--port', str(port),
            '--context', '2048', '--max-new', '384', '--threads', '4', '--arms', 'adapter',
            '--official-render', job['inputs']['official_render']['path'], '--duration', '175', '--ready-timeout', '30']
    if job['mode'] == 'case':
        argv.extend(['--ids', job['case']['id']])
    old_argv = sys.argv
    old_library = os.environ.get('LD_LIBRARY_PATH')
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        sys.argv = argv
        os.environ['LD_LIBRARY_PATH'] = str(SERVER.parent)
        try:
            module.main()
        except SystemExit as exit_status:
            require(exit_status.code == 0, 'Recovered native harness failed: ' + str(exit_status.code))
    finally:
        sys.argv = old_argv
        if old_library is None:
            os.environ.pop('LD_LIBRARY_PATH', None)
        else:
            os.environ['LD_LIBRARY_PATH'] = old_library
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    record = load(harness_path)
    result = validate_harness(job, record)
    result.update(receipt=str(harness_path), harness_receipt_sha256=digest(harness_path), linux_port=port_record)
    if job['mode'] == 'case':
        result.update(id=job['case']['id'], group=job['case']['group'], baseline_parity=baseline_parity(job, harness_path))
    else:
        result['accepted'] = True
    return result


def validate_final_guard(job, receipt):
    guard = load(job['guard_receipt'])
    runtime = job.get('cuda_runtime')
    if runtime is not None:
        guard_module(job).validate_gpu_receipt(guard, runtime['gpu_uuid'])
        require(all(guard['gpu']['identity'].get(key) == value for key, value in runtime['gpu'].items()),
                'Native guard GPU identity differs from training runtime')
    ownership = receipt['ownership']
    require(guard.get('supervisor_pid') == ownership['parent_pid'] and guard.get('child_pid') == ownership['pid'] and
            guard.get('owned_pgid') == ownership['pgid'] and guard.get('cgroup_path') == ownership['cgroup_path'] and
            guard.get('seconds') == 180 and guard.get('startup_released') is True and
            guard.get('stop_reason') == 'child_exit' and guard.get('first_stop_reason') is None and
            guard.get('child_exit_code') == 0 and guard.get('reaped') is True and
            guard.get('cleanup_state') == 'completed' and guard.get('cleanup_errors') == [] and
            guard.get('live_members_before_reap') == [] and guard.get('guard_exit_code') == 0 and
            guard.get('rss_peak_bytes', 6 * GIB) < 6 * GIB and
            0 < guard.get('elapsed_seconds', 181) <= 180, 'Native outer guard cleanup/resource failure')
    require(guard.get('argv') == receipt.get('argv'), 'Guard argv does not bind native execution')
    require(guard.get('samples'), 'Guard has no resource samples')
    initial = guard['samples'][0]
    require(initial['host']['memory_bytes']['MemAvailable'] >= 8 * GIB, 'Native admission memory gate failed')
    for sample in guard['samples']:
        require(sample['cgroup']['memory_max_bytes'] == 6 * GIB and sample['cgroup']['swap_max_bytes'] == 0 and
                sample['cgroup']['swap_current_bytes'] == 0 and sample['cgroup']['events'] == initial['cgroup']['events'] and
                sample['host']['memory_psi']['full']['total'] == initial['host']['memory_psi']['full']['total'],
                'Native cgroup/PSI changed during execution')
        require(sample['host']['memory_bytes']['MemAvailable'] >= 6 * GIB and
                sample['rss_bytes'] < 6 * GIB, 'Native runtime memory gate failed')
    return guard


def validate_receipt(job, receipt):
    validate_job(job)
    receipt = load(receipt) if isinstance(receipt, (str, Path)) else receipt
    require(receipt.get('status') == 'complete' and receipt.get('job_sha256') == job['job_sha256'] and
            receipt.get('mode') == job['mode'], 'Native receipt/job mismatch')
    guard = validate_final_guard(job, receipt)
    result = receipt['result']
    if job['mode'] == 'convert':
        conversion = load(check_bound(result['conversion_receipt']))
        validate_conversion(job, conversion)
        require(result['adapter_gguf'] == conversion['adapter_gguf'] and
                conversion['conversion_job_sha256'] == job['job_sha256'] and
                conversion['ownership'] == receipt['ownership'] and conversion['guard_receipt'] == job['guard_receipt'] and
                Path(result['conversion_receipt']['path']) == Path(job['output']) / 'conversion.json',
                'Conversion result/guard/job binding changed')
    else:
        harness = Path(result['receipt'])
        require(harness == Path(job['output']) / 'harness.json' and digest(harness) == result['harness_receipt_sha256'],
                'Native harness receipt path/hash changed')
        record = load(harness)
        ownership = receipt['ownership']
        require(record.get('harness_pid') == ownership['pid'] and record.get('pgid') == ownership['pgid'] and
                all(record.get('linux_guard_ownership', {}).get(k) == ownership[k]
                    for k in ('pid', 'parent_pid', 'pgid', 'cgroup_path', 'guard_receipt')),
                'Native harness not owned by this completed guard')
        normalized = validate_harness(job, record)
        require(all(result.get(k) == v for k, v in normalized.items()), 'Native normalized facts changed')
        if job['mode'] == 'case':
            parity = result['baseline_parity']
            original = check_bound(job['baseline']['receipt'])
            require(parity['baseline_receipt'] == str(original) and
                    parity['baseline_receipt_sha256'] == job['baseline']['receipt']['sha256'],
                    'Baseline receipt identity changed')
            checked = original_development(job['inputs'])['compare_with_baseline'](harness, original, job['case']['users'])
            require(checked['equal'] is True and checked['checks'] == parity['checks'], 'Original baseline checks failed')
    return {**result, 'guard_receipt': bound(job['guard_receipt']), 'guard': guard,
            'native_job_sha256': job['job_sha256'], 'native_build': job['build']}


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('convert', 'probe', 'case'), required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    job = load(args.job)
    require(job.get('mode') == args.mode, 'CLI/job mode mismatch')
    require(not args.receipt.exists(), 'Refusing to overwrite native receipt')
    validate_job(job)
    ownership = authenticate_guard(job)
    ownership['started_monotonic'] = started
    out = Path(job['output'])
    require(not out.exists() or not any(out.iterdir()), 'Native output must be fresh')
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'schema_version': 1, 'mode': args.mode, 'job_sha256': job['job_sha256'],
               'ownership': ownership, 'argv': ownership['guard_snapshot']['argv'], 'status': 'running'}
    try:
        receipt['result'] = convert(job, ownership) if args.mode == 'convert' else run_harness(job, ownership)
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        save(args.receipt, receipt)
        raise
    save(args.receipt, receipt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
