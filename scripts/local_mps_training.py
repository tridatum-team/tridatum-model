"""Offline fresh MPS LoRA execution; only the parent can accept a worker artifact.

capacity/train require immutable policy and source manifests. Limits are sampled
stops, not additive unified-memory counters or instantaneous host hard caps.
"""
import argparse
import ctypes
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
POLICY_SHA256 = '1641b555673165307c053bef701d4a2399eb8a774916ba5cb3e2a0a4ec9f9679'
REQUIRED_SOURCES = ('scripts/local_mps_training.py', 'scripts/train_open_weight_lora.py',
                    'scripts/aws_eval/grounded_hf_worker.py', 'scripts/aws_eval/grounded_hf_lifecycle.py')
WORKER_ENV = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_HIGH_WATERMARK_RATIO='1.0',
                  PYTORCH_MPS_LOW_WATERMARK_RATIO='0.5', TOKENIZERS_PARALLELISM='false')


def worker_environment(base=None):
    """Environment for a new child, with inherited runtime controls overridden."""
    return dict(os.environ if base is None else base, **WORKER_ENV)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def pin(path):
    path = resolve(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def authenticate(record):
    actual = pin(record['path'])
    require(actual['sha256'] == record['sha256'] and actual['bytes'] == record['bytes'],
            'Artifact pin mismatch: ' + record['path'])
    return resolve(record['path'])


def load_bound(path, digest):
    record = pin(path)
    require(record['sha256'] == digest, 'SHA256 mismatch: ' + str(path))
    return json.loads(resolve(path).read_bytes()), record


def save(path, value):
    # Atomic replacement inside an exclusively owned output directory.
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_contract(path, digest):
    manifest, record = load_bound(path, digest)
    require(manifest.get('schema') == 'trid-local-mps-sources-v1', 'Source manifest schema')
    files = manifest['files']
    paths = [authenticate(item) for item in files]
    require(len(paths) == len(set(paths)), 'Duplicate source paths')
    require(all(resolve(name) in paths for name in REQUIRED_SOURCES), 'Missing execution source pins')
    return record, files


def actual_sources(files):
    allowed = {authenticate(item) for item in files}
    used = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and any(path.is_relative_to(ROOT / part) for part in ('scripts', 'trid')):
                used.add(path)
    require(used <= allowed, 'Unpinned imported local sources: ' + str(sorted(map(str, used - allowed))))
    return [pin(path) for path in sorted(used)]


def policy_contract(path, digest):
    require(digest == POLICY_SHA256, 'This runner accepts only the frozen fresh-eight policy')
    policy, record = load_bound(path, digest)
    require(policy['schema'] == 'trid-local-eight-training-policy-v1', 'Policy schema')
    return policy, record


def authenticate_inputs(policy):
    model = policy['model']
    acquisition = json.loads(authenticate(model['acquisition_manifest']).read_bytes())
    require(acquisition['repo_id'] == model['id'] and acquisition['revision'] == model['revision'],
            'Official model identity mismatch')
    directory = resolve(model['directory'])
    require({str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()} ==
            {entry['path'] for entry in acquisition['files']}, 'Unpinned or missing model/tokenizer files')
    for entry in acquisition['files']:
        path = (directory / entry['path']).resolve()
        require(path.is_relative_to(directory), 'Model path escape')
        authenticate(dict(entry, path=str(path)))
    manifest_path = authenticate(policy['data']['training_manifest'])
    authenticate(policy['data']['frozen_dataset'])
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['state'] == 'ADMITTED' and manifest['valid'] == [], 'Not admitted or nonempty validation')
    require(manifest['max_tokens'] == 3072 and manifest['max_answer_tokens'] == 768, 'Manifest envelope')
    count = 0
    for entry in manifest['train']:
        require(entry['adapter'] == 'grounded_messages' and entry.get('epochs', 1) == 1,
                'Only one-pass grounded rows allowed')
        path = Path(entry['path'])
        path = path if path.is_absolute() else manifest_path.parent / path
        source = pin(path)
        require(source['sha256'] == entry['sha256'] and source['bytes'] == entry['bytes'],
                'Training source hash/size mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(all(row['split'] == 'train' for row in rows), 'Training split mismatch')
        count += len(rows)
    require(count == manifest['rows'] == policy['data']['train_rows'] == 600,
            'Physical training rows must equal declared manifest and policy600')
    return manifest_path, manifest


def capacity_contract(path, digest, policy_pin, sources_pin):
    result, record = load_bound(path, digest)
    require(result['schema'] == 'trid-local-mps-supervised-v1' and result['phase'] == 'capacity'
            and result['status'] == 'accepted', 'Capacity was not accepted by parent')
    cleanup = result['cleanup']
    require(cleanup['accepted'] and cleanup['leader_reaped'] and cleanup['remaining_members'] == [],
            'Capacity cleanup not accepted')
    require(result['policy']['sha256'] == policy_pin['sha256'] and
            result['sources']['sha256'] == sources_pin['sha256'], 'Capacity lineage differs')
    authenticate(result['policy']); authenticate(result['sources'])
    for item in result['source_files']:
        authenticate(item)
    worker = json.loads(authenticate(result['worker_result']).read_bytes())
    require(worker['status'] == 'complete' and worker['phase'] == 'capacity'
            and worker['optimizer_updates'] == 2 and worker['reload_exact'] is True
            and math.isfinite(worker['reload_loss']), 'Capacity numerical proof incomplete')
    return record


class RusageInfoV0(ctypes.Structure):
    # Darwin SDK sys/resource.h RUSAGE_INFO_V0: UUID followed by ten uint64_t.
    _fields_ = [('ri_uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups', 'ri_interrupt_wkups',
        'ri_pageins', 'ri_wired_size', 'ri_resident_size', 'ri_phys_footprint',
        'ri_proc_start_abstime', 'ri_proc_exit_abstime')]


def physical_footprint(pid):
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    function.restype = ctypes.c_int
    usage = RusageInfoV0()
    if function(pid, 0, ctypes.byref(usage)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, 'proc_pid_rusage failed', str(pid))
    return int(usage.ri_phys_footprint)


def query(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=1).stdout


def host_probe():
    pressure = int(query(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level']).strip())
    swap = query(['/usr/sbin/sysctl', '-n', 'vm.swapusage'])
    match = re.search(r'\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)\b', swap)
    require(match is not None, 'Cannot parse swap usage')
    swap_bytes = int(float(match[1]) * 1024 ** ('', 'K', 'M', 'G', 'T').index(match[2]))
    # -Q is query-only; never use pressure allocation/test modes.
    text = query(['/usr/bin/memory_pressure', '-Q'])
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    require(match is not None and 0 <= int(match[1]) <= 100, 'Cannot parse pressure percentage')
    power = query(['/usr/bin/pmset', '-g', 'batt'])
    return dict(pressure=pressure, swap_bytes=swap_bytes, pressure_free_percent=int(match[1]),
                ac_power="Now drawing from 'AC Power'" in power)


def group_members(pgid, footprint=True):
    rows = []
    for line in query(['/bin/ps', '-axo', 'pid=,pgid=,stat=,rss=']).splitlines():
        pid, group, state, rss = line.split()
        if int(group) == pgid:
            row = dict(pid=int(pid), pgid=int(group), status=state, rss_bytes=int(rss) * 1024)
            row['footprint_bytes'] = physical_footprint(int(pid)) if footprint and not state.startswith('Z') else 0
            rows.append(row)
    return rows


def stop_reason(limits, host, rows, baseline):
    if host['pressure'] != limits['pressure_required']:
        return 'host_pressure'
    if host['pressure_free_percent'] < limits['minimum_pressure_free_percent']:
        return 'pressure_free_percent_floor'
    if limits['require_ac_power'] and not host['ac_power']:
        return 'ac_power_required'
    if host['swap_bytes'] - baseline > limits['swap_growth_limit_bytes']:
        return 'swap_growth'
    for field, bound in (('rss_bytes', 'process_group_rss_limit_bytes'),
                         ('footprint_bytes', 'process_group_footprint_limit_bytes')):
        if sum(row[field] for row in rows) > limits[bound]:
            return bound
    return None


def telemetry_check(path, limits):
    if not path.exists():
        return None
    sample = json.loads(path.read_bytes())
    for field, bound in (('allocator_bytes', 'mps_allocator_limit_bytes'), ('driver_bytes', 'mps_driver_limit_bytes')):
        require(type(sample[field]) is int and sample[field] >= 0, 'Invalid MPS telemetry')
        require(sample[field] <= limits[bound], bound)
    require(sample.get('failure') is None, 'Worker telemetry failure')
    return sample


def supervise(args):
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='trid-local-mps-supervised-v1', phase=args.phase, status='failed',
                  started_unix=time.time(), startup_released=False, signals=[], samples=[],
                  cleanup=dict(accepted=False, leader_reaped=False, remaining_members=None))
    save(out / 'result.json', result)
    child = None
    gate_read = gate_write = None
    lock = log = None
    handlers = {}
    started = time.monotonic()
    limits = None

    def interrupt(signum, frame):
        raise TimeoutError('Supervisor deadline' if signum == signal.SIGALRM else 'Supervisor interrupted')

    def send(sig):
        if child is None:
            return
        attempt = dict(signal=signal.Signals(sig).name, pgid=child.pid)
        result['signals'].append(attempt)
        try:
            os.killpg(child.pid, sig)
            attempt['result'] = 'sent'
        except ProcessLookupError:
            attempt['result'] = 'absent'
        except PermissionError:
            rows = group_members(child.pid, footprint=False)
            require(len(rows) == 1 and rows[0]['pid'] == child.pid and rows[0]['status'].startswith('Z'),
                    'Cannot signal living owned process group')
            attempt['result'] = 'sole_unreaped_zombie'

    try:
        require(sys.platform == 'darwin', 'MPS execution requires Darwin')
        policy, result['policy'] = policy_contract(args.policy, args.policy_sha256)
        limits = policy['supervision']
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, interrupt)
        deadline = started + limits[args.phase + '_deadline_seconds' if args.phase == 'capacity' else 'training_deadline_seconds']
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        result['sources'], result['source_files'] = source_contract(args.sources, args.sources_sha256)
        if args.phase == 'train':
            require(args.capacity_result and args.capacity_result_sha256, 'Training requires authenticated capacity result')
            result['capacity_result'] = capacity_contract(args.capacity_result, args.capacity_result_sha256,
                                                          result['policy'], result['sources'])
        # A fixed per-user lock excludes all runs of this MPS training runner.
        lock_path = Path.home() / '.trid-local-mps-training.lock'
        lock = lock_path.open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result['device_lock'] = str(lock_path)
        host = host_probe()
        baseline = host['swap_bytes']
        result['swap_baseline_bytes'] = baseline
        require(stop_reason(limits, host, [], baseline) is None, 'Host startup guard refused')
        if args.phase == 'train':
            attempt_path = ROOT / 'runs' / policy['experiment_id'] / 'training-attempt.json'
            with attempt_path.open('x') as attempt:
                json.dump(dict(policy=result['policy'], out=str(out), started_unix=time.time()), attempt)
                attempt.flush()
                os.fsync(attempt.fileno())
            result['training_attempt'] = pin(attempt_path)
        gate_read, gate_write = os.pipe()
        env = worker_environment()
        env.update(TRID_MPS_GATE_FD=str(gate_read), TRID_MPS_PARENT=str(os.getpid()))
        argv = [sys.executable, '-I', str(Path(__file__).resolve()), args.phase, '--worker',
                '--policy', str(resolve(args.policy)), '--policy-sha256', args.policy_sha256,
                '--sources', str(resolve(args.sources)), '--sources-sha256', args.sources_sha256,
                '--out', str(out)]
        log = (out / 'worker.log').open('xb')
        child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 pass_fds=(gate_read,), start_new_session=True, cwd=ROOT)
        os.close(gate_read); gate_read = None
        require(os.getpgid(child.pid) == child.pid, 'Worker process group ownership')
        result['worker_pid'] = child.pid
        rows = group_members(child.pid)
        reason = stop_reason(limits, host_probe(), rows, baseline)
        require(reason is None, 'Startup guard: ' + str(reason))
        os.write(gate_write, b'G'); os.close(gate_write); gate_write = None
        result['startup_released'] = True
        save(out / 'result.json', result)
        while True:
            host = host_probe()
            rows = group_members(child.pid)
            sample = dict(elapsed_seconds=time.monotonic() - started, host=host, members=rows,
                          mps=telemetry_check(out / 'telemetry.json', limits))
            result['samples'].append(sample)
            reason = stop_reason(limits, host, rows, baseline)
            require(reason is None, 'External guard: ' + str(reason))
            require(time.monotonic() < deadline, 'Phase deadline exceeded')
            leader = next((row for row in rows if row['pid'] == child.pid), None)
            if leader is None or leader['status'].startswith('Z'):
                result['stop_reason'] = 'worker_exit'
                break
            time.sleep(limits['sample_seconds'])
    except BaseException as error:
        result['stop_reason'] = str(error)
        result['error_type'] = type(error).__name__
    finally:
        # Disable phase alarm, then allow only bounded cleanup. Never reap before
        # final group signaling: the unreaped leader prevents PGID reuse.
        signal.setitimer(signal.ITIMER_REAL, 0)
        errors = []
        if child is not None:
            grace = limits['term_grace_seconds']
            cleanup_seconds = limits['kill_cleanup_seconds']
            try:
                send(signal.SIGTERM)
                until = time.monotonic() + grace
                while time.monotonic() < until:
                    rows = group_members(child.pid, footprint=False)
                    if not any(not row['status'].startswith('Z') for row in rows):
                        break
                    time.sleep(0.05)
            except BaseException as error:
                errors.append(repr(error))
            try:
                send(signal.SIGKILL)
            except BaseException as error:
                errors.append(repr(error))
            try:
                until = time.monotonic() + cleanup_seconds
                while True:
                    rows = group_members(child.pid, footprint=False)
                    if all(row['pid'] == child.pid and row['status'].startswith('Z') for row in rows):
                        break
                    require(time.monotonic() < until, 'Owned descendants did not exit')
                    time.sleep(0.05)
                result['cleanup']['members_before_reap'] = rows
                result['exit_code'] = child.wait(timeout=max(0.01, until - time.monotonic()))
                result['cleanup']['leader_reaped'] = True
                result['cleanup']['remaining_members'] = group_members(child.pid, footprint=False)
            except BaseException as error:
                errors.append(repr(error))
        else:
            result['cleanup']['remaining_members'] = []
        for fd in (gate_read, gate_write):
            if fd is not None:
                os.close(fd)
        if log:
            log.close()
        result['cleanup']['errors'] = errors
        result['cleanup']['accepted'] = bool(child is not None and not errors and
            result['cleanup']['leader_reaped'] and result['cleanup']['remaining_members'] == [])
        try:
            require(result.get('stop_reason') == 'worker_exit' and result.get('exit_code') == 0,
                    'Worker did not exit successfully within guards')
            require(result['cleanup']['accepted'], 'Owned process cleanup failed')
            require(telemetry_check(out / 'telemetry.json', limits) is not None, 'Missing MPS telemetry')
            worker_path = out / 'worker' / ('capacity.json' if args.phase == 'capacity' else 'training.json')
            worker = json.loads(worker_path.read_bytes())
            require(worker['status'] == 'complete' and worker['policy']['sha256'] == result['policy']['sha256'],
                    'Worker result mismatch')
            result['worker_result'] = pin(worker_path)
            result['actual_source_files'] = worker['actual_source_files']
            for item in result['actual_source_files']:
                authenticate(item)
            if args.phase == 'train':
                require(worker['steps'] == 150 and worker['coverage_complete'] is True, 'Incomplete training')
                result['ledger'] = pin(out / 'worker' / 'checkpoints.json')
                ledger = json.loads(authenticate(result['ledger']).read_bytes())
                require(ledger['run_status'] == 'complete' and ledger['last_step']['step'] == 150
                        and [item['step'] for item in ledger['checkpoints']] == [50, 100, 150],
                        'Final checkpoint ledger differs')
                result['adapter_files'] = {}
                for name, field in (('adapter_model.safetensors', 'adapter_weights_sha256'),
                                    ('adapter_config.json', 'adapter_config_sha256')):
                    artifact = pin(out / 'worker' / 'adapter' / name)
                    require(artifact['sha256'] == ledger['last_step'][field] == worker['last_step'][field] ==
                            ledger['checkpoints'][-1][field], 'Final adapter differs from producer ledger: ' + name)
                    result['adapter_files'][name] = artifact
            else:
                require(worker['phase'] == 'capacity' and worker['optimizer_updates'] == 2
                        and worker['reload_exact'] is True and math.isfinite(worker['reload_loss']),
                        'Capacity numerical proof incomplete')
            # All supplied pins must still be intact after execution.
            authenticate(result['policy']); authenticate(result['sources'])
            for item in result['source_files']:
                authenticate(item)
            result['status'] = 'accepted'
        except BaseException as error:
            result['acceptance_error'] = repr(error)
        result['elapsed_seconds'] = time.monotonic() - started
        save(out / 'result.json', result)
        if lock:
            lock.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def tensor_inventory(named):
    summary = {}
    for name, tensor in named:
        key = str(tensor.dtype) + '@' + str(tensor.device)
        bucket = summary.setdefault(key, dict(tensors=0, elements=0, bytes=0))
        bucket['tensors'] += 1
        bucket['elements'] += tensor.numel()
        bucket['bytes'] += tensor.numel() * tensor.element_size()
    return summary


def parameter_contract(model, device='mps'):
    import torch
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    frozen = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    require(trainable and frozen, 'Missing base or adapter tensors')
    require(all(value.device.type == device and value.dtype == torch.float32 and 'lora_' in name
                for name, value in trainable), 'Adapter dtype/device/identity differs')
    require(all(value.device.type == device and value.dtype == torch.bfloat16 for _, value in frozen),
            'Frozen base dtype/device differs')
    return dict(base=tensor_inventory(frozen), adapter=tensor_inventory(trainable))


def gradient_health(model, optimizer, resident=False):
    import torch
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    require(all(value.grad is not None for _, value in parameters), 'Missing adapter gradient')
    require(all(bool(torch.isfinite(value.grad).all()) for _, value in parameters), 'Nonfinite adapter gradient')
    norm = math.sqrt(sum(float(value.grad.detach().float().square().sum()) for _, value in parameters))
    require(math.isfinite(norm) and norm > 0, 'Nonpositive or nonfinite gradient norm')
    if resident:
        require(all('exp_avg' in optimizer.state[value] and 'exp_avg_sq' in optimizer.state[value]
                    for _, value in parameters), 'Missing resident Adam moments')
    return dict(gradient_norm=norm, gradients=tensor_inventory((name, value.grad) for name, value in parameters),
                resident_moments=resident)


def optimizer_health(model, optimizer):
    import torch
    states = []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        require(bool(torch.isfinite(value).all()), 'Nonfinite updated adapter')
        state = optimizer.state[value]
        require(set(('step', 'exp_avg', 'exp_avg_sq')) <= set(state), 'Missing Adam state')
        for key, tensor in state.items():
            if torch.is_tensor(tensor):
                require(bool(torch.isfinite(tensor).all()), 'Nonfinite Adam state')
                if key != 'step':
                    require(tensor.dtype == value.dtype and tensor.device == value.device, 'Adam moment dtype/device differs')
                states.append((name + '.' + key, tensor))
        require(float(state['step']) > 0, 'Adam step did not advance')
    require(any(bool(value.abs().sum() > 0) for name, value in states if name.endswith('.exp_avg_sq')),
            'All optimizer second moments are zero')
    return dict(optimizer=tensor_inventory(states))


class Observer:
    def __init__(self, policy, policy_pin, files, out, runtime=None):
        self.policy, self.policy_pin, self.files, self.out = policy, policy_pin, files, out
        self.runtime = runtime
        self.expected = set()
        self.complete = False
        self.inventory = None
        self.snapshot = None
        self.changed_name = None

    def telemetry(self):
        import torch
        torch.mps.synchronize()
        sample = dict(monotonic=time.monotonic(), allocator_bytes=int(torch.mps.current_allocated_memory()),
                      driver_bytes=int(torch.mps.driver_allocated_memory()))
        save(self.out / 'telemetry.json', sample)
        telemetry_check(self.out / 'telemetry.json', self.policy['supervision'])
        return sample

    def __call__(self, event, **data):
        import torch
        if event == 'selection':
            return dict(status='unselected', default_artifact='adapter (final step150)',
                        rule='final150 compared on fixed200 development; no validation-loss selection',
                        diagnostic_steps=[50, 100], candidate_step=150)
        if event == 'admission':
            train = data['train']
            require(len(train) == 600 and not data['valid'] and not data['dropped'] and not data['valid_dropped'],
                    '600-row zero-drop empty-validation admission failed')
            self.expected = {identity_key(dict(file=item['file'], **item['identity'])) for item in train}
            require(len(self.expected) == 600, 'Duplicate training row identities')
        elif event == 'model':
            self.inventory = parameter_contract(data['model'])
            require(all(not bool(value.detach().count_nonzero()) for name, value in data['model'].named_parameters()
                        if 'lora_B' in name), 'Fresh adapter B matrices must start at zero')
            self.telemetry()
        elif event == 'backward':
            self.telemetry()
        elif event == 'before_step':
            model, optimizer = data['model'], data['optimizer']
            health = gradient_health(model, optimizer, resident=data['step'] > 1)
            self.snapshot = {name: value.detach().clone() for name, value in model.named_parameters()
                             if value.requires_grad}
            return health
        elif event == 'after_step':
            self.changed_name = next((name for name, value in data['model'].named_parameters()
                                      if value.requires_grad and
                                      not torch.equal(value.detach(), self.snapshot[name])), None)
            require(self.changed_name is not None, 'Optimizer did not change any adapter tensor')
            health = optimizer_health(data['model'], data['optimizer'])
            self.telemetry()
            self.snapshot = None
            return dict(health, changed_tensor=self.changed_name, changed=True)
        elif event == 'completion':
            coverage = data['coverage']
            require(data['steps'] == 150 and len(coverage) == 600 and
                    {identity_key(item) for item in coverage} == self.expected, 'Actual row coverage differs')
            self.complete = True
        elif event == 'report':
            return dict(policy=self.policy_pin, actual_source_files=actual_sources(self.files),
                        coverage_complete=self.complete, dtype_inventory=self.inventory, runtime=self.runtime,
                        endpoint_selection='final150 only; checkpoints50/100 diagnostics; no validation-loss selection')
        return None


def identity_key(item):
    require(item.get('source_sha256') and type(item.get('source_line')) is int and item.get('id'),
            'Missing source-row identity')
    return item['file'], item['source_sha256'], item['source_line'], item['id']


def fresh_model(trainer, policy):
    import torch
    from peft import LoraConfig, get_peft_model
    recipe = policy['training']
    model, loading = trainer.load_qwen3_pread(resolve(policy['model']['directory']), torch.bfloat16, 'mps')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'],
        lora_dropout=recipe['dropout'], target_modules=recipe['targets'], task_type='CAUSAL_LM'),
        autocast_adapter_dtype=True)
    return model


def verify_adapter_reload(model, expected):
    import torch
    actual = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require(actual.keys() == expected.keys(), 'Reload tensor names differ')
    require(all(actual[name].dtype == value.dtype and torch.equal(actual[name].detach().cpu(), value)
                for name, value in expected.items()), 'Reload tensor dtype/bytes differ')


def capacity_worker(trainer, policy, policy_pin, files, out, runtime):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / 'scripts' / 'aws_eval'))
    from grounded_hf_worker import synthetic_capacity_item
    folder = out / 'worker'
    folder.mkdir()
    tok = AutoTokenizer.from_pretrained(resolve(policy['model']['directory']), local_files_only=True)
    manifest_path, manifest = authenticate_inputs(policy)
    train, dropped = trainer.build_items(tok, manifest['train'], manifest_path.parent, 'qwen3', 3072, 768, True)
    observer = Observer(policy, policy_pin, files, out, runtime)
    observer('admission', train=train, valid=[], dropped=dropped, valid_dropped=0)
    item = synthetic_capacity_item(tok, vars(trainer), folder, 3072, 768)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = fresh_model(trainer, policy)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                 lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / 7 if s < 7 else
        0.5 * (1 + math.cos(math.pi * (s - 7) / 143)))
    observer('model', model=model)
    model.train()
    history = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(4):
            loss = trainer.batch_loss(model, [item], 'mps', pad, 'qwen3') / 4
            require(bool(torch.isfinite(loss)), 'Nonfinite full-envelope capacity loss')
            losses.append(float(loss))
            loss.backward()
            observer('backward', model=model, optimizer=optimizer, step=step)
            # The second update's every backward must coexist with real moments.
            if step == 2:
                gradient_health(model, optimizer, resident=True)
        health = observer('before_step', model=model, optimizer=optimizer, step=step)
        torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad],
                                      1.0, error_if_nonfinite=True)
        applied = optimizer.param_groups[0]['lr']
        optimizer.step(); schedule.step()
        health.update(observer('after_step', model=model, optimizer=optimizer, step=step))
        history.append(dict(step=step, loss=sum(losses), applied_lr=applied,
                            next_lr=schedule.get_last_lr()[0], numerical_health=health))
    adapter = folder / 'scratch-adapter'
    model.save_pretrained(adapter)
    expected = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
    # Clear every alias holding model, gradients, optimizer moments or graph.
    del loss, optimizer, schedule, model
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    release_memory = observer.telemetry()
    model = fresh_model(trainer, policy)
    trainer.load_init_adapter(model, adapter, 'mps')
    verify_adapter_reload(model, expected)
    parameter_contract(model)
    model.eval()
    with torch.no_grad():
        reload_loss = float(trainer.batch_loss(model, [item], 'mps', pad, 'qwen3'))
    require(math.isfinite(reload_loss), 'Nonfinite full-envelope reloaded loss')
    observer.telemetry()
    save(folder / 'capacity.json', dict(schema='trid-local-mps-capacity-v1', phase='capacity', status='complete',
        policy=policy_pin, actual_source_files=actual_sources(files), train_rows=600, dropped=0, runtime=runtime,
        sequence_tokens=3072, supervised_tokens=768, masked_prompt_tokens=2304, includes_eos=True,
        optimizer_updates=2, microbatches_per_update=4, dtype_inventory=observer.inventory, history=history,
        scheduler=dict(planned_training_updates=150, executed_probe_updates=2, warmup_updates=7,
                       cosine_denominator=143, order='optimizer_then_scheduler'),
        reload_exact=True, reload_loss=reload_loss, scratch_adapter_never_nominated=True,
        released_before_reload=True, release_memory=release_memory,
        scratch_weights=pin(adapter / 'adapter_model.safetensors')))


def configure_allocator(torch, limits):
    """Set the hard allocation ceiling before loading model code or tensors.

    Environment watermarks must already be installed before importing Torch:
    recommended_max_memory itself can initialize the allocator on Darwin.
    """
    require(all(os.environ.get(key) == value for key, value in WORKER_ENV.items()),
            'Worker environment was not installed before runtime initialization')
    initial_high = float(WORKER_ENV['PYTORCH_MPS_HIGH_WATERMARK_RATIO'])
    low = float(WORKER_ENV['PYTORCH_MPS_LOW_WATERMARK_RATIO'])
    require(0 < low <= initial_high <= 2, 'Invalid initial allocator watermarks')
    recommended = int(torch.mps.recommended_max_memory())
    require(recommended > 0, 'Invalid Metal recommended working set')
    fraction = limits['mps_allocator_limit_bytes'] / recommended
    require(math.isfinite(fraction) and 0 < fraction <= 2, 'Invalid allocator cap fraction')
    require(low <= fraction, 'Low watermark exceeds the final policy allocation ceiling')
    torch.mps.set_per_process_memory_fraction(fraction)
    return dict(initial_high_watermark_ratio=initial_high, low_watermark_ratio=low,
                final_high_watermark_ratio=fraction, recommended_max_memory_bytes=recommended,
                initial_high_watermark_bytes=int(initial_high * recommended),
                low_watermark_bytes=int(low * recommended),
                allocator_limit_bytes=limits['mps_allocator_limit_bytes'], environment=dict(WORKER_ENV))


def worker(args):
    # No Torch/Transformers/PEFT import may precede the parent-owned gate.
    fd = int(os.environ['TRID_MPS_GATE_FD'])
    require(os.getppid() == int(os.environ['TRID_MPS_PARENT']), 'Worker parent differs')
    try:
        require(os.read(fd, 1) == b'G', 'Parent did not release startup gate')
    finally:
        os.close(fd)
    os.environ.update(worker_environment())
    policy, policy_pin = policy_contract(args.policy, args.policy_sha256)
    _, files = source_contract(args.sources, args.sources_sha256)
    out = resolve(args.out)
    manifest_path, _ = authenticate_inputs(policy)
    import torch
    require(torch.backends.mps.is_available(), 'MPS unavailable; CPU fallback prohibited')
    runtime = configure_allocator(torch, policy['supervision'])
    print(json.dumps(dict(runtime=runtime)), flush=True)
    torch.manual_seed(policy['training']['seed'])
    sys.path.insert(0, str(ROOT / 'scripts'))
    import train_open_weight_lora as trainer
    actual_sources(files)
    if args.phase == 'capacity':
        capacity_worker(trainer, policy, policy_pin, files, out, runtime)
    else:
        observer = Observer(policy, policy_pin, files, out, runtime)
        recipe = policy['training']
        argv = ['--model', str(resolve(policy['model']['directory'])), '--model-id', policy['model']['id'],
                '--model-revision', policy['model']['revision'], '--family', 'qwen3', '--manifest', str(manifest_path),
                '--device', 'mps', '--out', str(out / 'worker')]
        for option in ('steps', 'batch_size', 'micro_batch_size', 'lr', 'rank', 'alpha', 'dropout',
                       'max_tokens', 'max_answer_tokens', 'seed', 'checkpoint_every'):
            argv.extend(['--' + option.replace('_', '-'), str(recipe[option])])
        trainer.main(argv, observer=observer)
        observer.telemetry()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capacity', 'train'))
    for name in ('policy', 'policy-sha256', 'sources', 'sources-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--capacity-result')
    parser.add_argument('--capacity-result-sha256')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return 0
    result = supervise(args)
    print(json.dumps(dict(status=result['status'], result=str(resolve(args.out) / 'result.json'))), flush=True)
    return 0 if result['status'] == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
