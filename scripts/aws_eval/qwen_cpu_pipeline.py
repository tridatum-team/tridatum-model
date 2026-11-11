"""One-shot packed run continuation: fresh full epoch, native gates, retained evidence.

Only the accepted two-update argv is extended, never the smoke adapter. CPU
training is traced; CUDA uses its source-bound entry's monotonic load evidence.
Native jobs retain the same independently bounded guard. This controller does
not grade, select, regenerate the baseline, or open final24.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

GIB = 1024 ** 3
MAX_OUTPUT = 512 * 1024 ** 2
STEPS = (8, 16, 23)
GROUPS = {'domain': 12, 'uncertainty': 14, 'retention': 20}
SOURCE = 'runs/qwen-specialization-011/train/checkpoints/step-000024'
SOURCE_PINS = {
    'adapter_weights_sha256': 'b92d1e73100bb313c1b86f37686b1a50c0cbecd20849647ff6d026a0fd9dfaee',
    'adapter_config_sha256': 'a7f9204434bc0a509e3697856e0cdbeff7d10047f283275d1bf6d78fe4d572dd',
    'training_json_sha256': '06ec73142360baec2f7c60ab81c2c15ce47b361dadb01c8b03a44619d073d870'}
SOURCE_FILES = {'adapter_weights_sha256': 'adapter/adapter_model.safetensors',
                'adapter_config_sha256': 'adapter/adapter_config.json',
                'training_json_sha256': 'training.json'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def forecast(history, supervised_seconds):
    require(len(history) == 2 and [r['step'] for r in history] == [1, 2],
            'Forecast requires exactly two fresh updates')
    t1, t2 = [r['elapsed_seconds'] for r in history]
    require(all(finite(x) and x > 0 for x in (t1, t2, supervised_seconds)),
            'Forecast observations must be finite and positive')
    update = t2 - t1
    startup = t1 - update
    overhead = supervised_seconds - t2
    require(update > 0 and startup >= 0 and overhead >= 0,
            'Invalid forecast components')
    estimate = startup + 23 * update + 3 * overhead + 30
    require(finite(estimate) and 0 < estimate <= 900, 'Full-training forecast exceeds 900 seconds')
    return {'t1': t1, 't2': t2, 'supervised_seconds': supervised_seconds,
            'update_seconds': update, 'startup_seconds': startup,
            'overhead_seconds': overhead, 'estimated_full_seconds': estimate,
            'cap_seconds': 900, 'accepted': True}


def remaining(deadline, seconds, now=None):
    now = time.time() if now is None else now
    # Finish unit return before wrapper TERM at W-305, not its KILL at W-300.
    require(finite(deadline) and finite(now) and now + seconds + 15 + 305 < deadline,
            'Insufficient worker lifetime for child, 15-second unit return, 5-second wrapper grace, and 300-second export reserve')


def publish_job(path, job):
    """Root-authored immutable job readable by the unprivileged native child."""
    save(path, job)
    path.chmod(0o444)
    return digest(path)


def traced_training_argv(python, train, trace_path):
    """Wrap canonical trainer arguments without changing training semantics."""
    require(train and train[0] == 'scripts/train_qwen_preferences.py',
            'Unexpected canonical trainer argv')
    return [python, '-B', 'scripts/trace_qwen_preferences.py',
            '--trace', str(trace_path), '--', *train[1:]]


def validate_runtime(runtime):
    require(isinstance(runtime, dict) and runtime.get('schema') == 'trid-cuda-runtime-v1'
            and runtime.get('trainer_entry') == 'scripts/train_qwen_preferences_cuda.py'
            and runtime.get('device') == 'cuda' and runtime.get('dtype') == 'torch.float32'
            and runtime.get('architecture') == 'x86_64'
            and isinstance(runtime.get('versions'), dict) and runtime['versions']
            and isinstance(runtime.get('engine'), dict) and runtime['engine']
            and isinstance(runtime.get('gpu_uuid'), str) and runtime['gpu_uuid'].startswith('GPU-')
            and isinstance(runtime.get('gpu'), dict)
            and runtime['gpu'].get('uuid') == runtime['gpu_uuid'],
            'Invalid explicit CUDA runtime contract')


def training_argv(python, train, trace_path, *, runtime=None):
    if runtime is None:
        return traced_training_argv(python, train, trace_path)
    validate_runtime(runtime)
    require(train and train[0] == runtime['trainer_entry'] and len(train) % 2 == 1,
            'Unexpected CUDA trainer argv')
    for option in ('--cuda-runtime', '--cuda-runtime-sha256'):
        require(train.count(option) == 1, 'Missing/duplicate CUDA runtime binding: ' + option)
    path = train[train.index('--cuda-runtime') + 1]
    sha = train[train.index('--cuda-runtime-sha256') + 1]
    require(digest(path) == sha and load(path) == runtime, 'CUDA runtime manifest changed')
    return [python, '-B', *train]


def cuda_load_timing(training):
    timing = training.get('load_timing', {})
    origin, start, end = [timing.get(key) for key in
                          ('elapsed_origin_monotonic', 'load_start_monotonic', 'load_end_monotonic')]
    require(all(finite(value) and value >= 0 for value in (origin, start, end))
            and origin <= start < end <= origin + training['history'][0]['elapsed_seconds'],
            'Missing/invalid CUDA load timing')
    return timing, end - start


def cuda_forecast(training, supervised_seconds):
    timing, duration = cuda_load_timing(training)
    result = forecast(training['history'], supervised_seconds)
    result.update(load_timing=timing, load_seconds=duration,
                  timing_basis='CUDA derivative monotonic history includes load once; no CPU tracer')
    return result


def full_argv(train, cache_shards, output, *, runtime=None):
    if runtime is not None:
        training_argv('', train, None, runtime=runtime)
    entry = 'scripts/train_qwen_preferences.py' if runtime is None else runtime['trainer_entry']
    require(train and train[0] == entry and len(train) % 2 == 1,
            'Unexpected accepted smoke argv')
    require('--init-adapter' not in train and '--ids' not in train, 'Fresh source initialization required')
    result = list(train)
    for option, old, new in (('--mode', 'train', 'train'), ('--max-steps', '2', '23'),
                             ('--checkpoint-every', '2', '8'), ('--valid-every', '2', '8')):
        require(train.count(option) == 1 and train[train.index(option) + 1] == old,
                'Accepted smoke argv changed: ' + option)
        result[result.index(option) + 1] = new
    require(train.count('--out') == 1 and len(cache_shards) == 6, 'Fresh output and six accepted shards required')
    for option, key in (('--cache', 'path'), ('--cache-sha256', 'sha256')):
        require([train[i + 1] for i in range(1, len(train), 2) if train[i] == option]
                == [r[key] for r in cache_shards], 'Accepted cache argv mismatch')
    require(not Path(output).exists(), 'Full training output must not exist')
    result[result.index('--out') + 1] = str(output)
    return result


def validate_ancestry(training, root):
    for label in ('init_adapter', 'reference'):
        ancestry = training[label]
        require(all(ancestry.get(k) == v for k, v in SOURCE_PINS.items())
                and ancestry.get('source_steps') == 24 and ancestry.get('source_cumulative_steps') == 96,
                'Source011 local24/cumulative96 ancestry mismatch')
    require(Path(training['init_adapter']['path']).resolve() == (root / SOURCE / 'adapter').resolve(),
            'Training did not freshly initialize from source011')
    require(training['init_adapter']['optimizer'].startswith('reset: fresh AdamW')
            and training['init_adapter']['scheduler'].startswith('reset: fresh warmup+cosine'),
            'Optimizer/scheduler must reset, not continue smoke')
    for key, relative in SOURCE_FILES.items():
        require(digest(root / SOURCE / relative) == SOURCE_PINS[key], 'Original source changed: ' + relative)


def validate_consumption(training, steps):
    require(steps in (2, *STEPS), 'Unexpected update count')
    pairs, replay = min(steps * 4, 91), steps * 288 // 23
    c = training['consumption']
    expected = {'pairs_admitted': 91, 'pairs_planned': 8 if steps == 2 else 91,
                'pairs_consumed_through_step': pairs, 'full_epoch_steps': 23,
                'planned_steps': 2 if steps == 2 else 23, 'bounded_by_max_steps': steps == 2,
                'epoch_status': 'complete' if steps == 23 else 'partial',
                'replay_rows_admitted': 288, 'replay_rows_planned': 25 if steps == 2 else 288,
                'replay_rows_consumed_through_step': replay,
                'replay_epoch_status': 'complete' if steps == 23 else 'partial',
                'epoch_fraction_consumed': round(pairs / 91, 4),
                'replay_epoch_fraction_consumed': round(replay / 288, 4)}
    require(all(c.get(k) == v for k, v in expected.items()), 'Incomplete pair/replay consumption')
    history = training['history']
    require(len(history) == steps and [r['step'] for r in history] == list(range(1, steps + 1)),
            'History is incomplete or reordered')
    previous = 0
    for i, row in enumerate(history, 1):
        require(row['pairs'] == min(4, 91 - 4 * (i - 1))
                and row['replay_rows'] == i * 288 // 23 - (i - 1) * 288 // 23,
                'History consumption differs from full-epoch partition')
        for key in ('train_loss', 'preference_loss', 'replay_ce', 'reward_margin',
                    'preferred_fraction', 'lr', 'elapsed_seconds'):
            require(finite(row[key]), 'Nonfinite history: ' + key)
        require(row['elapsed_seconds'] > previous, 'History elapsed time is not increasing')
        previous = row['elapsed_seconds']
        if i in ((2,) if steps == 2 else STEPS):
            losses = row.get('valid_loss_by_task')
            require(isinstance(losses, dict) and losses
                    and all(finite(v) and v >= 0 for v in losses.values()), 'Nonfinite/missing DEV CE')
    require(sum(r['pairs'] for r in history) == pairs
            and sum(r['replay_rows'] for r in history) == replay, 'History totals differ')


def validate_training(training, steps, cache_shards, root, checkpoint=False, *, runtime=None):
    if runtime is not None:
        validate_runtime(runtime)
        require(training.get('engine') == runtime['engine'] and training.get('cuda_runtime') == runtime
                and training.get('versions') == runtime['versions'],
                'Training CUDA runtime/engine identity mismatch')
        sha = training.get('cuda_runtime_sha256')
        require(isinstance(sha, str) and len(sha) == 64
                and all(c in '0123456789abcdef' for c in sha), 'Missing CUDA runtime SHA256')
        cuda_load_timing(training)
    require(training['status'] == ('checkpoint' if checkpoint else 'complete')
            and training['steps'] == steps and training['cumulative_steps'] == 96 + steps
            and training['planned_steps'] == (2 if steps == 2 else 23)
            and training['device'] == ('cpu' if runtime is None else runtime['device'])
            and training['dtype'] == 'torch.float32'
            and training['train_rows'] == training['train_pairs_admitted'] == 91
            and training['replay_rows'] == 288 and training['valid_rows'] == 36,
            'Training completion/population mismatch')
    check = training['initial_check']
    require(check['tolerance'] == 1e-3 and finite(check['initial_cache_max_abs_error'])
            and 0 <= check['initial_cache_max_abs_error'] <= 1e-3, 'Initial cache tolerance failed')
    validate_ancestry(training, root)
    validate_consumption(training, steps)
    objective = training['objective']
    require(objective['planned_epochs'] == 1 and objective['pairs_admitted'] == 91
            and objective['batch_pairs'] == objective['replay_micro_batch'] == 4,
            'Full-epoch objective changed')
    replay = objective['replay']
    planned = 2 if steps == 2 else 23
    row_counts = [i * 288 // 23 - (i - 1) * 288 // 23 for i in range(1, planned + 1)]
    require(replay['full_epoch_steps'] == 23 and replay['planned_steps'] == planned
            and replay['rows_admitted'] == 288 and replay['rows_planned'] == sum(row_counts)
            and replay['rows_per_step'] == row_counts and replay['seed'] == 20261031
            and replay['micro_batch'] == 4 and replay['weight'] == 1,
            'Replay plan changed, truncated, or repartitioned')
    cache = objective['cache']
    require([{k: row[k] for k in ('path', 'sha256')} for row in cache] == cache_shards
            and [row['ids'] for row in cache] == [16] * 5 + [11], 'Training cache bindings differ')
    for row in cache_shards:
        require(digest(row['path']) == row['sha256'], 'Accepted cache changed')
        if runtime is not None:
            shard = load(row['path'])
            require(shard.get('engine') == runtime['engine'] and shard.get('cuda_runtime') == runtime
                    and shard.get('versions') == runtime['versions']
                    and shard.get('cuda_runtime_sha256') == training['cuda_runtime_sha256'],
                    'Accepted cache is not bound to this CUDA runtime')


def validate_full(full, smoke, cache_shards, root, *, runtime=None):
    training = load(full / 'training.json')
    validate_training(training, 23, cache_shards, root, runtime=runtime)
    if runtime is not None:
        require(training.get('cuda_runtime_sha256') == smoke.get('cuda_runtime_sha256')
                and isinstance(training.get('cuda_runtime_sha256'), str)
                and len(training['cuda_runtime_sha256']) == 64, 'Full CUDA manifest binding differs')
    for key in ('base_model_id', 'base_revision', 'family', 'manifest', 'manifest_sha256',
                'manifest_id', 'files', 'replay_files', 'lora', 'max_tokens', 'seed',
                'base_weights', 'init_adapter', 'reference', 'tokenizer', 'versions',
                'lr', 'beta', 'batch_size', 'replay_micro_batch_size'):
        require(training[key] == smoke[key], 'Full run changed smoke source/objective binding: ' + key)
    require([c['step'] for c in training['checkpoints']] == list(STEPS),
            'Full run did not retain checkpoints8/16/23')
    ledger = load(full / 'checkpoints.json')
    require(ledger['run_status'] == 'complete' and ledger['checkpoint_every'] == 8
            and ledger['planned_steps'] == 23 and ledger['checkpoints'] == training['checkpoints'],
            'Full checkpoint ledger differs')
    for row in training['checkpoints']:
        step = row['step']
        checkpoint = full / 'checkpoints' / f'step-{step:06d}'
        require(Path(row['path']).resolve() == (checkpoint / 'adapter').resolve(),
                'Checkpoint output escaped fresh full-training directory')
        actual = digest(checkpoint / 'adapter/adapter_model.safetensors')
        require(actual == row['adapter_weights_sha256'], 'Checkpoint weight binding mismatch')
        metadata = load(checkpoint / 'training.json')
        validate_training(metadata, step, cache_shards, root, checkpoint=True, runtime=runtime)
        if runtime is not None:
            require(metadata.get('cuda_runtime_sha256') == training['cuda_runtime_sha256'],
                    'Checkpoint CUDA manifest binding differs')
        require(metadata['checkpoint'] == row and metadata['history'] == training['history'][:step],
                'Checkpoint history/binding mismatch')
        require(row['pairs_consumed'] == min(step * 4, 91)
                and row['replay_rows_consumed'] == step * 288 // 23,
                'Checkpoint consumption mismatch')
    require(training['last_step']['equals_last_checkpoint'] is True
            and training['last_step']['step'] == 23
            and training['last_step']['adapter_weights_sha256']
            == training['checkpoints'][-1]['adapter_weights_sha256']
            == digest(full / 'adapter/adapter_model.safetensors'),
            'Final root adapter does not match retained checkpoint23')
    return training


def validate_guard(result, seconds, argv, *, gpu_uuid=None):
    if gpu_uuid is not None:
        import importlib.util
        path = Path(__file__).resolve().parents[1] / 'qwen_linux_guard.py'
        spec = importlib.util.spec_from_file_location('_qwen_pipeline_guard', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_gpu_receipt(result, gpu_uuid)
    require(result.get('argv') == argv and result.get('seconds') == seconds
            and result.get('startup_released') is True and result.get('stop_reason') == 'child_exit'
            and result.get('child_exit_code') == 0 and result.get('first_stop_reason') is None
            and result.get('reaped') is True and result.get('cleanup_state') == 'completed'
            and result.get('cleanup_errors') == [] and result.get('live_members_before_reap') == []
            and finite(result.get('elapsed_seconds')) and 0 < result['elapsed_seconds'] <= seconds
            and 0 < result.get('rss_peak_bytes', 0) <= 6 * GIB,
            'Guard failed, incomplete, or not bound to dispatched child')
    samples = result.get('samples', [])
    require(samples, 'No real resource observations')
    totals = []
    for sample in samples:
        cg = sample['cgroup']
        require(cg['memory_max_bytes'] == 6 * GIB and cg['swap_max_bytes'] == cg['swap_current_bytes'] == 0
                and 0 <= sample['rss_bytes'] <= 6 * GIB
                and all(cg['events'].get(k, 0) == 0 for k in ('max', 'oom', 'oom_kill', 'oom_group_kill')),
                'Invalid resource boundary or memory event')
        totals.append(sample['host']['memory_psi']['full']['total'])
    require(all(type(t) is int and t >= 0 for t in totals) and len(set(totals)) == 1,
            'Host memory PSI full increased')


class Dispatcher:
    """Persist intent before every actual child; failed calls poison this pipeline."""
    def __init__(self, config, output, unit, root, *, runtime=None):
        self.config, self.output, self.unit, self.root = config, Path(output), unit, Path(root)
        if runtime is not None:
            validate_runtime(runtime)
        self.runtime = runtime
        self.python = '/opt/qwen013-venv/bin/python'
        self.state = {'state': 'RUNNING', 'automatic_retries': 0, 'invocations': []}
        self.failed = False
        self.persist()

    def persist(self):
        save(self.output / 'pipeline-outcome.json', self.state)

    def run(self, name, argv, seconds=180, allow_loopback=False):
        require(not self.failed, 'Earlier stage failed; no subsequent child permitted')
        entry = {'name': name, 'argv': argv, 'seconds': seconds, 'state': 'admission_pending'}
        self.state['invocations'].append(entry)
        self.persist()
        receipt = self.output / f'{name}-supervisor.json'
        try:
            remaining(self.config['worker_deadline_unix'], seconds)
            require(not receipt.exists(), 'Guard receipt must be fresh')
            entry['state'] = 'dispatch_requested'
            if name == 'full-training':
                self.state['full_training_dispatch_requested'] = True
            if name.startswith('native-case-'):
                self.state['native_generation_dispatch_requested'] = True
            self.persist()
            command = [self.python, '-B', str(self.root / 'scripts/qwen_linux_guard.py'),
                       '--receipt', str(receipt), '--seconds', str(seconds)]
            if self.runtime is not None:
                command += ['--gpu-uuid', self.runtime['gpu_uuid']]
            command += ['--', *argv]
            rc = (self.unit(name, command, seconds, allow_loopback=True) if allow_loopback
                  else self.unit(name, command, seconds))
            entry['returncode'] = rc
            require(rc == 0, name + ' failed; no retry')
            result = load(receipt)
            validate_guard(result, seconds, argv,
                           gpu_uuid=None if self.runtime is None else self.runtime['gpu_uuid'])
            if self.runtime is not None:
                require(result['gpu']['identity'] == self.runtime['gpu'],
                        'Guard GPU identity differs from authenticated runtime')
            entry.update(state='guard_accepted', supervisor_sha256=digest(receipt))
            return result
        except BaseException as error:
            self.failed = True
            entry.update(state='FAILED_NO_RETRY', error=f'{type(error).__name__}: {error}',
                         supervisor_receipt_present=receipt.exists())
            self.state['state'] = 'FAILED_NO_RETRY'
            raise
        finally:
            self.persist()


def run_full_training(dispatch, argv, cache_shards, *, runtime=None):
    """Publish immutable training intent; the unit owns *-command.json receipts."""
    output = dispatch.output
    trace_path = output / 'full-training-phases.jsonl' if runtime is None else None
    dispatched_argv = training_argv(dispatch.python, argv, trace_path, runtime=runtime)
    plan = {
        'child_argv': argv, 'dispatched_argv': dispatched_argv,
        'trace_path': str(trace_path) if trace_path is not None else None,
        'cuda_runtime': runtime,
        'fresh_source_checkpoint': SOURCE, 'never_continue_smoke': True,
        'guard_seconds': 900, 'optimizer_updates': 23, 'checkpoints': list(STEPS),
        'cache_shards': cache_shards, 'forecast': 'runtime-admission.json'}
    path = output / 'full-training-plan.json'
    with path.open('x') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return dispatch.run('full-training', dispatched_argv, 900)


def export_manifest(output, exclusions):
    """Inventory, not delete: only explicitly proven duplicates/probe weights omitted."""
    output = Path(output)
    retained, omitted = [], []
    for path in sorted(output.rglob('*')):
        require(not path.is_symlink(), 'Export symlink refused')
        if not path.is_file() or path.name == 'export-manifest.json':
            continue
        relative = path.relative_to(output).as_posix()
        row = {'path': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        if relative in exclusions:
            omitted.append({**row, 'reason': exclusions[relative]})
        else:
            retained.append(row)
    payload = {'schema_version': 1, 'complete': True, 'files': retained, 'excluded': omitted, 'inputs_deleted': False,
               'retained_bytes': sum(r['bytes'] for r in retained), 'max_bytes': MAX_OUTPUT,
               'manifest_included_separately': True}
    # Allow the manifest itself and the bootstrap's final small receipts.
    require(payload['retained_bytes'] + 4 * 1024 ** 2 <= min(MAX_OUTPUT, 500 * 1024 ** 2),
            'Retained output exceeds 500 MiB payload bound/reserve')
    save(output / 'export-manifest.json', payload)
    return payload


def duplicate_exclusions(output, full, smoke_succeeded=False):
    exclusions = {}
    if smoke_succeeded:
        # Probe artifacts are intentionally excluded, not mislabelled duplicates.
        for path in (output / 'train').rglob('adapter_model.safetensors'):
            exclusions[path.relative_to(output).as_posix()] = 'accepted two-update smoke weights; full run is fresh source011'
        for path in (output / 'native/smoke').rglob('*.gguf'):
            exclusions[path.relative_to(output).as_posix()] = 'accepted TRAIN-only native feature-probe adapter; not a DEV candidate'
    root_weight = full / 'adapter/adapter_model.safetensors'
    last_weight = full / 'checkpoints/step-000023/adapter/adapter_model.safetensors'
    if root_weight.is_file() and last_weight.is_file():
        require(digest(root_weight) == digest(last_weight), 'Root adapter is not the last-checkpoint duplicate')
        exclusions[root_weight.relative_to(output).as_posix()] = 'byte-identical step-000023 checkpoint weight retained'
    return exclusions


def writable_directory(path, output):
    path.mkdir(parents=True, exist_ok=False)
    owner = output.stat()
    os.chown(path, owner.st_uid, owner.st_gid)


def continue_after_smoke(config, output, plan, train, cache_shards, unit, *, runtime=None):
    """Called only after the existing controller accepts the real smoke receipts."""
    import importlib.util
    import sys

    output = Path(output)
    root = Path.cwd()
    full = output / 'full-training'
    dispatch = Dispatcher(config, output, unit, root, runtime=runtime)
    probe_accepted = False
    executions = []
    try:
        require(config['run_id'] == output.parent.name, 'Continuation run identity differs')
        acceptance = load(output / 'smoke-acceptance.json')
        smoke_path = output / 'train/training.json'
        require(acceptance['accepted'] is True and acceptance['updates'] == 2
                and acceptance['training_sha256'] == digest(smoke_path), 'Accepted smoke binding changed')
        smoke = load(smoke_path)
        validate_training(smoke, 2, cache_shards, root, runtime=runtime)
        smoke_guard = load(output / 'model-smoke-supervisor.json')
        validate_guard(smoke_guard, 180, training_argv(
            dispatch.python, train, output / 'model-smoke-phases.jsonl', runtime=runtime),
            gpu_uuid=None if runtime is None else runtime['gpu_uuid'])
        if runtime is not None:
            require(smoke_guard['gpu']['identity'] == runtime['gpu'],
                    'Smoke guard GPU identity differs from authenticated runtime')
            require(smoke.get('cuda_runtime_sha256') == train[train.index('--cuda-runtime-sha256') + 1],
                    'Smoke CUDA manifest binding differs')
        argv = full_argv(train, cache_shards, full, runtime=runtime)
        try:
            admission = (forecast(smoke['history'], smoke_guard['elapsed_seconds']) if runtime is None
                         else cuda_forecast(smoke, smoke_guard['elapsed_seconds']))
        except (RuntimeError, KeyError, TypeError) as error:
            save(output / 'runtime-admission.json', {
                'accepted': False, 'error': str(error), 'cap_seconds': 900,
                'observed_history': repr(smoke.get('history')),
                'observed_supervised_seconds': repr(smoke_guard.get('elapsed_seconds'))})
            raise
        save(output / 'runtime-admission.json', admission)
        # Authenticate code before importing the stdlib-only native job builder.
        for path, sha in plan['source_pins'].items():
            require(digest(root / path) == sha, 'Source changed before continuation: ' + path)
        native_path = root / 'scripts/aws_eval/qwen_cpu_native.py'
        require(plan['source_pins'].get('scripts/aws_eval/qwen_cpu_native.py') == digest(native_path),
                'Native executor is not authenticated by this run')
        spec = importlib.util.spec_from_file_location('qwen_cpu_native_pipeline', native_path)
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        cases = native.native_cases(plan)
        require(len(cases) == len({c['id'] for c in cases}) == 36, 'Native DEV population differs')
        for group, turns in GROUPS.items():
            selected = [c for c in cases if c['group'] == group]
            require(len(selected) == 12 and sum(len(c['users']) for c in selected) == turns,
                    'Frozen DEV group/turn count differs')
        native_root = output / 'native'
        for path in (native_root, native_root / 'jobs', native_root / 'receipts', native_root / 'smoke'):
            writable_directory(path, output)

        def native_job(name, mode, directory, checkpoint, case=None, adapter_dir=None):
            remaining(config['worker_deadline_unix'], 180)
            require(not directory.exists(), 'Native output must be fresh')
            receipt_path = native_root / 'receipts' / f'{name}.json'
            job_path = native_root / 'jobs' / f'{name}.json'
            require(not receipt_path.exists() and not job_path.exists(), 'Native job/receipt reused')
            job = native.build_job(mode, root, directory, checkpoint, plan,
                                   output / f'{name}-supervisor.json', case=case, adapter_dir=adapter_dir,
                                   runtime=runtime)
            job_sha256 = publish_job(job_path, job)
            command = [dispatch.python, '-B', str(native_path), '--mode', mode,
                       '--job', str(job_path), '--receipt', str(receipt_path)]
            dispatch.run(name, command, allow_loopback=mode in ('probe', 'case'))
            try:
                require(digest(job_path) == job_sha256, 'Authenticated native job changed during dispatch')
                result = native.validate_receipt(job, load(receipt_path))
                dispatch.state['invocations'][-1].update(
                    state='accepted', native_job_sha256=job_sha256,
                    native_receipt_sha256=digest(receipt_path))
                return result
            except BaseException as error:
                dispatch.failed = True
                dispatch.state['invocations'][-1].update(
                    state='INVALID_NO_RETRY', error=f'{type(error).__name__}: {error}',
                    native_receipt_present=receipt_path.exists())
                raise
            finally:
                dispatch.persist()

        smoke_checkpoint = output / 'train/checkpoints/step-000002'
        smoke_conversion = native_root / 'smoke/conversion'
        native_job('native-convert-smoke', 'convert', smoke_conversion, smoke_checkpoint)
        probe = native_job('native-probe-train', 'probe', native_root / 'smoke/probe',
                           smoke_checkpoint, adapter_dir=smoke_conversion)
        require(probe['accepted'] is True and probe['turns_total'] == probe['turns_completed'] == 4,
                'Four actual TRAIN native requests did not pass')
        probe_accepted = True
        save(output / 'native-feature-acceptance.json', probe)
        run_full_training(dispatch, argv, cache_shards, runtime=runtime)
        trained = validate_full(full, smoke, cache_shards, root, runtime=runtime)
        save(output / 'full-training-acceptance.json', {
            'accepted': True, 'training_sha256': digest(full / 'training.json'),
            'consumption': trained['consumption'], 'checkpoints': trained['checkpoints']})
        for step in STEPS:
            checkpoint = full / 'checkpoints' / f'step-{step:06d}'
            step_dir = native_root / f'step-{step:06d}'
            writable_directory(step_dir, output)
            conversion = step_dir / 'conversion'
            converted = native_job(f'native-convert-{step:06d}', 'convert', conversion, checkpoint)
            for group, turn_count in GROUPS.items():
                selected = [c for c in cases if c['group'] == group]
                group_dir = step_dir / group
                writable_directory(group_dir, output)
                group_plan = {
                    'arm': 'adapter', 'group': group, 'ids': [c['id'] for c in selected],
                    'turns': {c['id']: len(c['users']) for c in selected}, 'checkpoint': str(checkpoint),
                    'child_seconds': 180,
                    'platform': 'linux-' + ('aarch64' if runtime is None else runtime['architecture']),
                    'source_pins': plan['source_pins'], 'native': plan['native'],
                    'baseline_policy': 'frozen run005 only; actual exact parity required; never regenerate/regrade',
                    'grading_performed': False, 'final24_opened': False}
                save(group_dir / 'plan.json', group_plan)
                execution = {
                    'state': 'RUNNING', 'arm': 'adapter', 'group': group, 'checkpoint_step': step,
                    'checkpoint': str(checkpoint),
                    'adapter_weights_sha256': digest(checkpoint / 'adapter/adapter_model.safetensors'),
                    'adapter_gguf_sha256': converted['adapter_gguf']['sha256'],
                    'base_gguf_sha256': plan['native']['base_gguf']['sha256'],
                    'baseline': str(Path(plan['native']['baseline_executions'][group]['path']).parent),
                    'requested': 12, 'completed': 0, 'remaining_ids': [c['id'] for c in selected],
                    'automatic_retries': 0, 'cases': [], 'plan': str(group_dir / 'plan.json')}
                executions.append((group_dir / 'execution.json', execution))
                save(group_dir / 'execution.json', execution)
                for index, case in enumerate(selected, 1):
                    name = f'native-case-{step:06d}-{group}-{index:03d}'
                    entry = {'index': index, 'id': case['id'], 'group': group, 'arm': 'adapter',
                             'verified': False, 'state': 'dispatch_requested',
                             'supervisor_receipt': str(output / f'{name}-supervisor.json'),
                             'log': str(output / f'{name}.log')}
                    execution['cases'].append(entry)
                    save(group_dir / 'execution.json', execution)
                    record = native_job(name, 'case', group_dir / f'{index:03d}', checkpoint,
                                        case=case, adapter_dir=conversion)
                    require(record['id'] == case['id'] and record['group'] == group
                            and record['status'] == 'complete'
                            and record['turns_total'] == record['turns_completed'] == len(case['users'])
                            and record['baseline_parity']['equal'] is True,
                            'Native case incomplete or frozen baseline parity failed')
                    entry.update(record)
                    entry.update(verified=True, state='accepted', parity=record['baseline_parity'])
                    execution['completed'] += 1
                    execution['remaining_ids'].remove(case['id'])
                    save(group_dir / 'execution.json', execution)
                require(sum(c['turns_completed'] for c in execution['cases']) == turn_count,
                        'Native group turn population differs')
                execution['state'] = 'COMPLETE'
                save(group_dir / 'execution.json', execution)
        for path, sha in plan['source_pins'].items():
            require(digest(root / path) == sha, 'Source changed during full pipeline: ' + path)
        validate_ancestry(trained, root)
        dispatch.state.update(state='COMPLETE_AWAITING_BLIND_SELECTION', full_training_started=True,
                              native_generation_started=True, candidate_cases=108, candidate_turns=138,
                              baseline_regenerated=False, semantic_grading_performed=False,
                              selection_performed=False, final24_opened=False)
    except BaseException as error:
        dispatch.failed = True
        dispatch.state.update(state='FAILED_NO_RETRY', error=f'{type(error).__name__}: {error}')
        for path, execution in executions:
            if execution['state'] == 'RUNNING':
                execution.update(state='FAILED_NO_RETRY', error=dispatch.state['error'])
                save(path, execution)
        raise
    finally:
        dispatch.persist()
        try:
            export_manifest(output, duplicate_exclusions(output, full, probe_accepted))
        except BaseException as error:
            dispatch.state.update(export_state='FAILED', export_error=f'{type(error).__name__}: {error}')
            dispatch.persist()
            raise
    return dispatch.state
