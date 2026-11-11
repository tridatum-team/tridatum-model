"""Pure receipt/dispatch boundary tests; never run models, systemd, or AWS."""
import copy
import importlib.util
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aws_eval import qwen_cpu_smoke as cpu
from scripts.aws_eval import qwen_cuda_smoke as cuda

SPEC = importlib.util.spec_from_file_location(
    'qwen_cpu_pipeline', Path(__file__).resolve().parents[1] / 'scripts/aws_eval/qwen_cpu_pipeline.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def history(steps):
    return [dict(step=i, pairs=min(4, 91 - (i - 1) * 4),
                 replay_rows=i * 288 // 23 - (i - 1) * 288 // 23,
                 train_loss=1.0, preference_loss=0.5, replay_ce=0.5, reward_margin=0.1,
                 preferred_fraction=0.5, lr=0.00001, elapsed_seconds=float(i * 10),
                 **({'valid_loss_by_task': {'domain': 1.0, 'retention': 2.0}}
                    if i in (2, 8, 16, 23) else {})) for i in range(1, steps + 1)]


def training(steps=23):
    pairs, replay = min(steps * 4, 91), steps * 288 // 23
    return {'history': history(steps), 'consumption': {
        'pairs_admitted': 91, 'pairs_planned': 8 if steps == 2 else 91,
        'pairs_consumed_through_step': pairs, 'full_epoch_steps': 23,
        'planned_steps': 2 if steps == 2 else 23, 'bounded_by_max_steps': steps == 2,
        'epoch_status': 'complete' if steps == 23 else 'partial',
        'replay_rows_admitted': 288, 'replay_rows_planned': 25 if steps == 2 else 288,
        'replay_rows_consumed_through_step': replay,
        'replay_epoch_status': 'complete' if steps == 23 else 'partial',
        'epoch_fraction_consumed': round(pairs / 91, 4),
        'replay_epoch_fraction_consumed': round(replay / 288, 4)}}


def guard(argv, seconds=180):
    sample = {'rss_bytes': 100, 'cgroup': {'memory_max_bytes': 6 * pipeline.GIB,
              'swap_max_bytes': 0, 'swap_current_bytes': 0,
              'events': {'max': 0, 'oom': 0, 'oom_kill': 0}},
              'host': {'memory_psi': {'full': {'total': 100}}}}
    return {'argv': argv, 'seconds': seconds, 'startup_released': True,
            'stop_reason': 'child_exit', 'child_exit_code': 0, 'first_stop_reason': None,
            'reaped': True, 'cleanup_state': 'completed', 'cleanup_errors': [],
            'live_members_before_reap': [], 'elapsed_seconds': 120,
            'rss_peak_bytes': 100, 'samples': [sample, copy.deepcopy(sample)]}


def cuda_runtime():
    uuid = 'GPU-11111111-2222-3333-4444-555555555555'
    gpu = {'uuid': uuid, 'name': 'NVIDIA L4', 'pci_bus_id': '00000000:00:1E.0',
           'total_bytes': 24 * pipeline.GIB, 'driver_version': '580.95.05'}
    return {'schema': 'trid-cuda-runtime-v1',
            'trainer_entry': 'scripts/train_qwen_preferences_cuda.py',
            'device': 'cuda', 'dtype': 'torch.float32', 'architecture': 'x86_64',
            'versions': {'torch': '2.9.1+cu128', 'transformers': '4.57.6', 'peft': '0.17.1'},
            'engine': {'fixture_source_sha256': 'a' * 64, 'device': 'cuda'},
            'gpu_uuid': uuid, 'gpu': gpu}


def gpu_guard(argv, seconds=180):
    result = guard(argv, seconds)
    runtime = cuda_runtime()
    result.update(guard_exit_code=0, owned_pgid=123,
                  gpu={'schema': 'trid-gpu-guard-v1', 'expected_uuid': runtime['gpu_uuid'],
                       'memory_limit_bytes': 6 * pipeline.GIB, 'identity': runtime['gpu'],
                       'used_peak_bytes': 100})
    sample = result['samples'][0]
    result['samples'] = []
    for index, phase in enumerate(('admission', 'pre_exec', 'runtime')):
        row = copy.deepcopy(sample)
        row.update(gpu_phase=phase, elapsed_seconds=index, members=[], gpu_process_observations=[],
                   gpu={**runtime['gpu'], 'used_bytes': 100, 'compute_processes': []})
        result['samples'].append(row)
    return result


class ForecastTests(unittest.TestCase):
    def test_exact_900_admitted_but_next_instant_rejected(self):
        rows = [{'step': 1, 'elapsed_seconds': 50}, {'step': 2, 'elapsed_seconds': 80}]
        self.assertEqual(pipeline.forecast(rows, 80 + 160 / 3)['estimated_full_seconds'], 900)
        with self.assertRaises(RuntimeError):
            pipeline.forecast(rows, 80 + 160 / 3 + 0.001)

    def test_nonfinite_nonpositive_and_negative_components(self):
        for t1, t2, supervised in ((float('nan'), 20, 30), (10, float('inf'), 30),
                                   (10, 20, float('nan')), (0, 10, 20),
                                   (10, 10, 20), (10, 30, 40), (10, 20, 19)):
            with self.subTest(values=(t1, t2, supervised)), self.assertRaises(RuntimeError):
                pipeline.forecast([{'step': 1, 'elapsed_seconds': t1},
                                   {'step': 2, 'elapsed_seconds': t2}], supervised)

    def test_uses_prevalidation_history_not_training_elapsed(self):
        result = pipeline.forecast([{'step': 1, 'elapsed_seconds': 30},
                                    {'step': 2, 'elapsed_seconds': 50}], 60)
        self.assertEqual(result['estimated_full_seconds'], 530)


class ConsumptionTests(unittest.TestCase):
    def test_every_checkpoint_and_smoke_prefix(self):
        for step in (2, 8, 16, 23):
            pipeline.validate_consumption(training(step), step)
        self.assertEqual(training()['history'][-1]['pairs'], 3)
        self.assertEqual(training(2)['consumption']['replay_rows_consumed_through_step'], 25)

    def test_repartition_or_missing_rows_refused(self):
        for field in ('pairs_consumed_through_step', 'replay_rows_consumed_through_step',
                      'pairs_admitted', 'replay_rows_planned', 'full_epoch_steps'):
            row = training()
            row['consumption'][field] -= 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                pipeline.validate_consumption(row, 23)
        row = training()
        row['history'][0]['replay_rows'] += 1
        row['history'][1]['replay_rows'] -= 1
        with self.assertRaises(RuntimeError):
            pipeline.validate_consumption(row, 23)

    def test_nonfinite_history_or_dev_ce_refused(self):
        for key in ('train_loss', 'replay_ce', 'lr', 'elapsed_seconds'):
            row = training()
            row['history'][4][key] = float('nan')
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                pipeline.validate_consumption(row, 23)
        row = training()
        row['history'][15]['valid_loss_by_task']['domain'] = float('inf')
        with self.assertRaises(RuntimeError):
            pipeline.validate_consumption(row, 23)

    def test_wrong_ancestry_fails_before_source_reads(self):
        ancestry = {**pipeline.SOURCE_PINS, 'source_steps': 24, 'source_cumulative_steps': 96,
                    'path': '/repo/' + pipeline.SOURCE + '/adapter',
                    'optimizer': 'reset: fresh AdamW', 'scheduler': 'reset: fresh warmup+cosine'}
        row = {'init_adapter': ancestry, 'reference': copy.deepcopy(ancestry)}
        for field, value in (('source_steps', 2), ('source_cumulative_steps', 98),
                             ('adapter_config_sha256', 'smoke'), ('training_json_sha256', 'wrong')):
            changed = copy.deepcopy(row)
            changed['init_adapter'][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                pipeline.validate_ancestry(changed, Path('/repo'))
        with patch.object(pipeline, 'digest', side_effect=list(pipeline.SOURCE_PINS.values())):
            pipeline.validate_ancestry(row, Path('/repo'))
        with patch.object(pipeline, 'digest', return_value='mutated'), self.assertRaises(RuntimeError):
            pipeline.validate_ancestry(row, Path('/repo'))


class DispatchTests(unittest.TestCase):
    def test_reserve_boundary_strict(self):
        with self.assertRaises(RuntimeError):
            pipeline.remaining(1500, 180, now=1000)
        pipeline.remaining(1500.001, 180, now=1000)
        with self.assertRaises(RuntimeError):
            pipeline.remaining(float('inf'), 180, now=1000)
        with self.assertRaises(RuntimeError):
            pipeline.remaining(2220, 900, now=1000)
        pipeline.remaining(2220.001, 900, now=1000)
        # The old cutoff admitted work whose return could overlap wrapper TERM.
        with self.assertRaises(RuntimeError):
            pipeline.remaining(1495.001, 180, now=1000)
        with self.assertRaises(RuntimeError):
            pipeline.remaining(2215.001, 900, now=1000)

    def test_failed_child_keeps_intent_and_stops_following_model(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def unit(name, argv, seconds):
                recorded = json.loads((Path(directory) / 'pipeline-outcome.json').read_text())
                self.assertEqual(recorded['invocations'][-1]['state'], 'dispatch_requested')
                calls.append(name)
                return 1
            dispatcher = pipeline.Dispatcher({'worker_deadline_unix': 10000}, directory, unit, '/repo')
            with patch.object(pipeline.time, 'time', return_value=0):
                with self.assertRaises(RuntimeError):
                    dispatcher.run('native-probe-train', ['native'])
                with self.assertRaises(RuntimeError):
                    dispatcher.run('full-training', ['train'], 900)
            self.assertEqual(calls, ['native-probe-train'])
            self.assertEqual(dispatcher.state['state'], 'FAILED_NO_RETRY')

    def test_late_stage_does_not_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            def unit(*args):
                self.fail('Late child dispatched')
            dispatcher = pipeline.Dispatcher({'deadline_unix': 1800, 'worker_deadline_unix': 1500},
                                             directory, unit, '/repo')
            with patch.object(pipeline.time, 'time', return_value=1000), self.assertRaises(RuntimeError):
                dispatcher.run('case', ['native'])
            self.assertEqual(dispatcher.state['invocations'][0]['state'], 'FAILED_NO_RETRY')

    def test_actual_guard_argv_and_full_bound_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def unit(name, command, seconds, **kwargs):
                calls.append((name, seconds, kwargs))
                receipt = Path(command[command.index('--receipt') + 1])
                child = command[command.index('--') + 1:]
                self.assertEqual(int(command[command.index('--seconds') + 1]), seconds)
                receipt.write_text(json.dumps(guard(child, seconds)))
                return 0
            dispatcher = pipeline.Dispatcher({'worker_deadline_unix': 10000}, directory, unit, '/repo')
            with patch.object(pipeline.time, 'time', return_value=0):
                dispatcher.run('full-training', ['python', 'train'], 900)
                dispatcher.run('native-case-000008-domain-001', ['python', 'native'],
                               allow_loopback=True)
            self.assertEqual(calls, [('full-training', 900, {}),
                                    ('native-case-000008-domain-001', 180, {'allow_loopback': True})])
            self.assertTrue(dispatcher.state['full_training_dispatch_requested'])
            self.assertTrue(dispatcher.state['native_generation_dispatch_requested'])

    def test_traced_full_dispatch_binds_wrapper_sidecar_and_trainer_arguments(self):
        for mutation in ('exact', 'unwrapped', 'other-sidecar', 'changed-trainer'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                canonical = ['scripts/train_qwen_preferences.py', '--mode', 'train',
                             '--out', str(output / 'full-training'), '--max-steps', '23',
                             '--lr', '0.00005']
                original = canonical[:]
                trace = output / 'full-training-phases.jsonl'
                calls = []

                def unit(name, command, seconds):
                    calls.append(name)
                    self.assertEqual(seconds, 900)
                    child = command[command.index('--') + 1:]
                    self.assertEqual(child[:6], [
                        '/opt/qwen013-venv/bin/python', '-B', 'scripts/trace_qwen_preferences.py',
                        '--trace', str(trace), '--'])
                    self.assertEqual(child[6:], original[1:])
                    reported = child[:]
                    if mutation == 'unwrapped':
                        reported = [child[0], '-B', *original]
                    elif mutation == 'other-sidecar':
                        reported[4] = str(output / 'model-smoke-phases.jsonl')
                    elif mutation == 'changed-trainer':
                        reported[reported.index('--lr') + 1] = '0.1'
                    receipt = Path(command[command.index('--receipt') + 1])
                    receipt.write_text(json.dumps(guard(reported, seconds)))
                    return 0

                dispatcher = pipeline.Dispatcher({'worker_deadline_unix': 10000}, output, unit, '/repo')
                argv = pipeline.traced_training_argv(dispatcher.python, canonical, trace)
                with patch.object(pipeline.time, 'time', return_value=0):
                    if mutation == 'exact':
                        pipeline_result = dispatcher.run('full-training', argv, 900)
                        self.assertEqual(pipeline_result['argv'], argv)
                        self.assertEqual(dispatcher.state['invocations'][0]['state'], 'guard_accepted')
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'not bound to dispatched child'):
                            dispatcher.run('full-training', argv, 900)
                        with self.assertRaisesRegex(RuntimeError, 'no subsequent child'):
                            dispatcher.run('native-case-000008-domain-001', ['native'])
                        self.assertEqual(dispatcher.state['state'], 'FAILED_NO_RETRY')
                self.assertEqual(calls, ['full-training'])
                self.assertEqual(canonical, original)
                self.assertEqual(dispatcher.state['invocations'][0]['argv'], argv)

    def test_zero_exit_without_actual_guard_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            dispatcher = pipeline.Dispatcher({'worker_deadline_unix': 10000}, directory,
                                             lambda *args: 0, '/repo')
            with patch.object(pipeline.time, 'time', return_value=0), self.assertRaises(FileNotFoundError):
                dispatcher.run('native-probe-train', ['python', 'native'])
            self.assertTrue(dispatcher.failed)

    def test_job_readable_under_root_umask_without_mutable_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'job.json'
            previous = os.umask(0o077)
            try:
                sha256 = pipeline.publish_job(path, {'mode': 'probe', 'run_id': 'approved-run'})
            finally:
                os.umask(previous)
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            self.assertEqual(sha256, pipeline.digest(path))
            self.assertEqual(json.loads(path.read_text())['run_id'], 'approved-run')

    def test_guard_binding_pressure_cleanup_and_duration(self):
        argv = ['python', 'native.py']
        pipeline.validate_guard(guard(argv), 180, argv)
        for field, value in (('argv', ['other']), ('seconds', 900), ('reaped', False),
                             ('elapsed_seconds', 181), ('first_stop_reason', 'deadline'),
                             ('live_members_before_reap', [123])):
            receipt = guard(argv)
            receipt[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                pipeline.validate_guard(receipt, 180, argv)
        receipt = guard(argv)
        receipt['samples'][1]['host']['memory_psi']['full']['total'] += 1
        with self.assertRaises(RuntimeError):
            pipeline.validate_guard(receipt, 180, argv)


class FullCommandTests(unittest.TestCase):
    def test_changes_only_output_schedule_and_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_shards = [{'path': f'accepted-linux-{i}', 'sha256': str(i)} for i in range(6)]
            argv = ['scripts/train_qwen_preferences.py', '--mode', 'train', '--max-steps', '2',
                    '--checkpoint-every', '2', '--valid-every', '2', '--out', 'smoke',
                    '--manifest', 'source011-manifest', '--lr', '0.00005']
            for row in cache_shards:
                argv += ['--cache', row['path'], '--cache-sha256', row['sha256']]
            full = pipeline.full_argv(argv, cache_shards, Path(directory) / 'full')
            changed = {argv[i - 1] for i in range(1, len(argv)) if argv[i] != full[i]}
            self.assertEqual(changed, {'--out', '--max-steps', '--checkpoint-every', '--valid-every'})
            self.assertEqual(argv[argv.index('--max-steps') + 1], '2')
            with self.assertRaises(RuntimeError):
                pipeline.full_argv(argv + ['--init-adapter', 'smoke/adapter'],
                                   cache_shards, Path(directory) / 'full')
            with self.assertRaises(RuntimeError):
                pipeline.full_argv(argv, cache_shards[:-1], Path(directory) / 'full')
            with self.assertRaises(RuntimeError):
                pipeline.full_argv(argv, cache_shards, Path(directory))


class TracedSmokeAdmissionTests(unittest.TestCase):
    """Exercise continuation admission, not a model or synthetic success fallback."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'authorized-run/remote-output'
        self.output.mkdir(parents=True)
        self.cache_shards = []
        for index in range(6):
            path = self.root / f'accepted-linux-shard-{index}.json'
            path.write_text(json.dumps({'fixture_shard': index}))
            self.cache_shards.append({'path': str(path), 'sha256': pipeline.digest(path)})
        self.train = ['scripts/train_qwen_preferences.py', '--mode', 'train',
                      '--max-steps', '2', '--checkpoint-every', '2', '--valid-every', '2',
                      '--out', str(self.output / 'train'), '--lr', '0.00005']
        for row in self.cache_shards:
            self.train += ['--cache', row['path'], '--cache-sha256', row['sha256']]
        self.trace = self.output / 'model-smoke-phases.jsonl'
        self.command = pipeline.traced_training_argv(
            '/opt/qwen013-venv/bin/python', self.train, self.trace)
        self.smoke = {
            **training(2), 'status': 'complete', 'steps': 2, 'cumulative_steps': 98,
            'planned_steps': 2, 'device': 'cpu', 'dtype': 'torch.float32',
            'train_rows': 91, 'train_pairs_admitted': 91, 'replay_rows': 288, 'valid_rows': 36,
            'initial_check': {'tolerance': 1e-3, 'initial_cache_max_abs_error': 0.0},
            'objective': {
                'planned_epochs': 1, 'pairs_admitted': 91, 'batch_pairs': 4, 'replay_micro_batch': 4,
                'replay': {'full_epoch_steps': 23, 'planned_steps': 2, 'rows_admitted': 288,
                           'rows_planned': 25, 'rows_per_step': [12, 13], 'seed': 20261031,
                           'micro_batch': 4, 'weight': 1},
                'cache': [{**row, 'ids': 16 if index < 5 else 11}
                          for index, row in enumerate(self.cache_shards)]}}
        # Source ancestry has separate byte/metadata tests; all other admission
        # validators here operate on their real receipt and training fixtures.
        ancestry = patch.object(pipeline, 'validate_ancestry')
        ancestry.start()
        self.addCleanup(ancestry.stop)
        cwd = patch.object(pipeline.Path, 'cwd', return_value=self.root)
        cwd.start()
        self.addCleanup(cwd.stop)

    def write_receipts(self, command=None):
        train_dir = self.output / 'train'
        train_dir.mkdir(exist_ok=True)
        path = train_dir / 'training.json'
        path.write_text(json.dumps(self.smoke))
        pipeline.save(self.output / 'smoke-acceptance.json', {
            'accepted': True, 'updates': 2, 'training_sha256': pipeline.digest(path)})
        pipeline.save(self.output / 'model-smoke-supervisor.json', guard(
            self.command if command is None else command))

    def continue_to_source_authentication(self):
        def no_child(*args, **kwargs):
            self.fail('Admission test must not dispatch any child')
        # Deliberately stop at the next real boundary, after runtime admission.
        # This is not a successful native/full-training receipt.
        return pipeline.continue_after_smoke(
            {'run_id': 'authorized-run', 'worker_deadline_unix': 10000},
            self.output, {'source_pins': {'unavailable-source.py': 'unavailable'}},
            self.train, self.cache_shards, no_child)

    def test_exact_traced_smoke_admitted_without_terminal_trace_event(self):
        self.trace.write_text('{"event":"optimizer_update_returned","optimizer_updates":1}\n')
        self.write_receipts()
        with self.assertRaises(FileNotFoundError):
            self.continue_to_source_authentication()
        admission = pipeline.load(self.output / 'runtime-admission.json')
        self.assertTrue(admission['accepted'])
        self.assertEqual(admission['estimated_full_seconds'], 560)
        outcome = pipeline.load(self.output / 'pipeline-outcome.json')
        self.assertEqual(outcome['state'], 'FAILED_NO_RETRY')
        self.assertEqual(outcome['invocations'], [])
        manifest = pipeline.load(self.output / 'export-manifest.json')
        retained = {row['path']: row for row in manifest['files']}
        self.assertEqual(retained[self.trace.name]['sha256'], pipeline.digest(self.trace))

    def test_unwrapped_other_sidecar_or_changed_trainer_receipt_refused(self):
        variants = {
            'unwrapped': [self.command[0], '-B', *self.train],
            'other-sidecar': [*self.command[:4], str(self.output / 'other-phases.jsonl'),
                              *self.command[5:]],
            'changed-trainer': self.command[:]}
        variants['changed-trainer'][self.command.index('--lr') + 1] = '0.1'
        for label, command in variants.items():
            with self.subTest(mutation=label):
                self.write_receipts(command)
                with self.assertRaisesRegex(RuntimeError, 'not bound to dispatched child'):
                    self.continue_to_source_authentication()
                self.assertFalse((self.output / 'runtime-admission.json').exists())
                self.assertEqual(pipeline.load(self.output / 'pipeline-outcome.json')['invocations'], [])

    def test_diagnostic_completion_cannot_replace_trainer_completion(self):
        self.trace.write_text('{"event":"training_main_returned","optimizer_updates":2}\n')
        for status in ('checkpoint', 'running', 'failed'):
            with self.subTest(status=status):
                self.smoke['status'] = status
                self.write_receipts()
                with self.assertRaisesRegex(RuntimeError, 'Training completion/population mismatch'):
                    self.continue_to_source_authentication()
                self.assertFalse((self.output / 'runtime-admission.json').exists())
        (self.output / 'train/training.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.continue_to_source_authentication()
        self.assertFalse((self.output / 'runtime-admission.json').exists())

    def test_diagnostics_cannot_override_forecast_or_resource_rejection(self):
        self.trace.write_text('{"event":"training_main_returned","optimizer_updates":2}\n')
        self.smoke['history'][0]['elapsed_seconds'] = 50
        self.smoke['history'][1]['elapsed_seconds'] = 100
        self.write_receipts()
        with self.assertRaisesRegex(RuntimeError, 'forecast exceeds 900'):
            self.continue_to_source_authentication()
        self.assertFalse(pipeline.load(self.output / 'runtime-admission.json')['accepted'])
        self.smoke['history'] = history(2)
        self.write_receipts()
        receipt = pipeline.load(self.output / 'model-smoke-supervisor.json')
        receipt['rss_peak_bytes'] = 6 * pipeline.GIB + 1
        pipeline.save(self.output / 'model-smoke-supervisor.json', receipt)
        with self.assertRaisesRegex(RuntimeError, 'Guard failed'):
            self.continue_to_source_authentication()
        self.assertEqual(pipeline.load(self.output / 'pipeline-outcome.json')['invocations'], [])

    def test_helper_refuses_noncanonical_script_before_dispatch(self):
        for train in ([], ['scripts/trace_qwen_preferences.py'], ['other/train_qwen_preferences.py']):
            with self.subTest(train=train), self.assertRaisesRegex(RuntimeError, 'canonical trainer argv'):
                pipeline.traced_training_argv('/python', train, self.trace)


class CudaDownstreamTests(unittest.TestCase):
    def setUp(self):
        TracedSmokeAdmissionTests.setUp(self)
        self.runtime = cuda_runtime()
        self.runtime_path = self.root / 'cuda-runtime.json'
        self.runtime_path.write_text(json.dumps(self.runtime))
        self.runtime_sha = pipeline.digest(self.runtime_path)
        self.train[0] = self.runtime['trainer_entry']
        self.train += ['--cuda-runtime', str(self.runtime_path),
                       '--cuda-runtime-sha256', self.runtime_sha]
        self.smoke.update(device='cuda', versions=self.runtime['versions'],
                          engine=self.runtime['engine'], cuda_runtime=self.runtime,
                          cuda_runtime_sha256=self.runtime_sha,
                          load_timing={'elapsed_origin_monotonic': 100,
                                       'load_start_monotonic': 101, 'load_end_monotonic': 105})
        for index, row in enumerate(self.cache_shards):
            Path(row['path']).write_text(json.dumps({
                'engine': self.runtime['engine'], 'cuda_runtime': self.runtime,
                'versions': self.runtime['versions'], 'cuda_runtime_sha256': self.runtime_sha}))
            row['sha256'] = pipeline.digest(row['path'])
            self.train[self.train.index(row['path']) + 2] = row['sha256']
            self.smoke['objective']['cache'][index]['sha256'] = row['sha256']

    def test_rehashed_cpu_cache_cannot_be_relabelled_cuda(self):
        row = self.cache_shards[0]
        Path(row['path']).write_text(json.dumps({'device': 'cpu', 'engine': {}}))
        row['sha256'] = pipeline.digest(row['path'])
        self.smoke['objective']['cache'][0]['sha256'] = row['sha256']
        with self.assertRaisesRegex(RuntimeError, 'cache is not bound'):
            pipeline.validate_training(self.smoke, 2, self.cache_shards, self.root, runtime=self.runtime)

    def test_full_checkpoints_bind_exact_cuda_engine_and_manifest(self):
        full = self.output / 'full-training'
        common = copy.deepcopy(self.smoke)
        for key in ('base_model_id', 'base_revision', 'family', 'manifest', 'manifest_sha256',
                    'manifest_id', 'files', 'replay_files', 'lora', 'max_tokens', 'seed',
                    'base_weights', 'init_adapter', 'reference', 'tokenizer',
                    'lr', 'beta', 'batch_size', 'replay_micro_batch_size'):
            common[key] = self.smoke[key] = 'unchanged-fixture'
        common.update(training(23), steps=23, cumulative_steps=119, planned_steps=23)
        common['objective']['replay'].update(
            planned_steps=23, rows_planned=288,
            rows_per_step=[i * 288 // 23 - (i - 1) * 288 // 23 for i in range(1, 24)])
        rows = []
        for step in pipeline.STEPS:
            checkpoint = full / 'checkpoints' / f'step-{step:06d}'
            adapter = checkpoint / 'adapter'
            adapter.mkdir(parents=True)
            weight = adapter / 'adapter_model.safetensors'
            weight.write_bytes(f'checkpoint-fixture-{step}'.encode())
            record = {'step': step, 'path': str(adapter),
                      'adapter_weights_sha256': pipeline.digest(weight),
                      'pairs_consumed': min(step * 4, 91), 'replay_rows_consumed': step * 288 // 23}
            rows.append(record)
            metadata = copy.deepcopy(common)
            metadata.update(training(step), status='checkpoint', steps=step,
                            cumulative_steps=96 + step, checkpoint=record)
            pipeline.save(checkpoint / 'training.json', metadata)
        common['checkpoints'] = rows
        common['last_step'] = {'equals_last_checkpoint': True, 'step': 23,
                               'adapter_weights_sha256': rows[-1]['adapter_weights_sha256']}
        (full / 'adapter').mkdir()
        (full / 'adapter/adapter_model.safetensors').write_bytes(b'checkpoint-fixture-23')
        pipeline.save(full / 'training.json', common)
        pipeline.save(full / 'checkpoints.json', {
            'run_status': 'complete', 'checkpoint_every': 8, 'planned_steps': 23, 'checkpoints': rows})
        pipeline.validate_full(full, self.smoke, self.cache_shards, self.root, runtime=self.runtime)
        path = full / 'checkpoints/step-000016/training.json'
        original = pipeline.load(path)
        for key, value in (('engine', {}), ('cuda_runtime_sha256', 'b' * 64)):
            pipeline.save(path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                pipeline.validate_full(full, self.smoke, self.cache_shards, self.root, runtime=self.runtime)
        pipeline.save(path, original)

    def test_exact_cuda_identity_and_cpu_default(self):
        pipeline.validate_training(self.smoke, 2, self.cache_shards, self.root, runtime=self.runtime)
        with self.assertRaisesRegex(RuntimeError, 'completion/population'):
            pipeline.validate_training(self.smoke, 2, self.cache_shards, self.root)
        for field, value in (('engine', {}), ('cuda_runtime', {}), ('versions', {}),
                             ('device', 'cpu'), ('dtype', 'torch.float16')):
            changed = copy.deepcopy(self.smoke)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                pipeline.validate_training(changed, 2, self.cache_shards, self.root, runtime=self.runtime)

    def test_full_direct_command_retains_runtime_and_fresh_source(self):
        original = self.train[:]
        full = pipeline.full_argv(self.train, self.cache_shards, self.output / 'full', runtime=self.runtime)
        self.assertEqual(pipeline.training_argv('/python', full, None, runtime=self.runtime),
                         ['/python', '-B', *full])
        self.assertEqual(full[full.index('--cuda-runtime-sha256') + 1], self.runtime_sha)
        self.assertEqual(self.train, original)
        self.assertNotIn('--init-adapter', full)
        self.assertEqual(full[full.index('--max-steps') + 1], '23')
        self.assertEqual(full[full.index('--checkpoint-every') + 1], '8')
        with self.assertRaisesRegex(RuntimeError, 'Fresh source'):
            pipeline.full_argv(self.train + ['--init-adapter', 'smoke/adapter'],
                               self.cache_shards, self.output / 'full', runtime=self.runtime)

    def test_changed_missing_or_rebound_runtime_manifest_rejected(self):
        for argv in (self.train[:-4], self.train + self.train[-4:],
                     ['scripts/train_qwen_preferences.py', *self.train[1:]]):
            with self.assertRaises(RuntimeError):
                pipeline.training_argv('/python', argv, None, runtime=self.runtime)
        self.runtime_path.write_text(json.dumps({**self.runtime, 'device': 'cpu'}))
        with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
            pipeline.training_argv('/python', self.train, None, runtime=self.runtime)
        self.train[-1] = pipeline.digest(self.runtime_path)
        with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
            pipeline.training_argv('/python', self.train, None, runtime=self.runtime)

    def test_cuda_load_evidence_is_required_and_counted_once(self):
        observed = pipeline.cuda_forecast(self.smoke, 120)
        self.assertEqual(observed['estimated_full_seconds'], 560)
        self.assertEqual(observed['load_seconds'], 4)
        for timing in ({}, {'elapsed_origin_monotonic': 100, 'load_start_monotonic': 99,
                           'load_end_monotonic': 105},
                       {'elapsed_origin_monotonic': 100, 'load_start_monotonic': 101,
                        'load_end_monotonic': 111},
                       {'elapsed_origin_monotonic': 100, 'load_start_monotonic': float('nan'),
                        'load_end_monotonic': 105}):
            with self.subTest(timing=timing), self.assertRaises(RuntimeError):
                pipeline.cuda_forecast({**self.smoke, 'load_timing': timing}, 120)

    def test_gpu_receipt_missing_stale_foreign_or_peak_evidence_rejected(self):
        command = pipeline.training_argv('/python', self.train, None, runtime=self.runtime)
        pipeline.validate_guard(gpu_guard(command), 180, command, gpu_uuid=self.runtime['gpu_uuid'])
        with self.assertRaises(ValueError):
            pipeline.validate_guard(guard(command), 180, command, gpu_uuid=self.runtime['gpu_uuid'])
        for mutation in ('uuid', 'identity', 'missing-sample', 'foreign-process', 'peak'):
            receipt = gpu_guard(command)
            if mutation == 'uuid':
                receipt['gpu']['expected_uuid'] = 'GPU-00000000-2222-3333-4444-555555555555'
            elif mutation == 'identity':
                receipt['samples'][2]['gpu']['driver_version'] = 'stale-driver'
            elif mutation == 'missing-sample':
                receipt['samples'].pop()
            elif mutation == 'foreign-process':
                receipt['samples'][2]['gpu']['compute_processes'] = [
                    {'uuid': self.runtime['gpu_uuid'], 'pid': 999}]
            else:
                receipt['gpu']['used_peak_bytes'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pipeline.validate_guard(receipt, 180, command, gpu_uuid=self.runtime['gpu_uuid'])


class ArtifactOwnershipTests(unittest.TestCase):
    """Real plan, dispatcher, sandbox argv and receipt writers; only OS launch is mocked."""

    def boundary(self, output, controller):
        config = {'run_id': controller.RUN_ID, 'worker_deadline_unix': 10000}
        runtime = cuda_runtime() if controller is cuda else None
        train = ['scripts/train_qwen_preferences.py', '--mode', 'train',
                 '--max-steps', '2', '--checkpoint-every', '2', '--valid-every', '2',
                 '--out', str(output / 'train')]
        shards = [{'path': f'/accepted/shard-{i}.json', 'sha256': f'{i:064x}'} for i in range(6)]
        for row in shards:
            train += ['--cache', row['path'], '--cache-sha256', row['sha256']]
        if runtime is not None:
            runtime_path = output / 'cuda-runtime.json'
            pipeline.save(runtime_path, runtime)
            train[0] = runtime['trainer_entry']
            train += ['--cuda-runtime', str(runtime_path),
                      '--cuda-runtime-sha256', pipeline.digest(runtime_path)]
            unit = partial(cuda.run_unit, config, output, runtime['gpu_uuid'],
                           ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'])
        else:
            unit = partial(cpu.run_unit, config, output)
        argv = pipeline.full_argv(train, shards, output / 'full-training', runtime=runtime)
        dispatch = pipeline.Dispatcher(config, output, unit, output, runtime=runtime)
        return dispatch, argv, shards, runtime, unit

    def test_plan_and_receipt_have_distinct_owners_on_both_controllers(self):
        for controller in (cpu, cuda):
            with self.subTest(controller=controller.__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                dispatch, argv, shards, runtime, unit = self.boundary(output, controller)
                plan_path = output / 'full-training-plan.json'
                receipt_path = output / 'full-training-command.json'
                supervisor_path = output / 'full-training-supervisor.json'
                observed_plan_bytes = []

                def launch_boundary(command, **kwargs):
                    # Inspect actual disk state at the last boundary before systemd.
                    # Deliberate launch failure: never fabricate successful guard/model evidence.
                    observed_plan_bytes.append(plan_path.read_bytes())
                    self.assertEqual(plan_path.stat().st_mode & 0o777, 0o444)
                    plan = pipeline.load(plan_path)
                    self.assertEqual(plan['child_argv'], argv)
                    self.assertEqual(plan['cuda_runtime'], runtime)
                    self.assertEqual(plan['guard_seconds'], 900)
                    self.assertEqual(plan['optimizer_updates'], 23)
                    self.assertEqual(plan['checkpoints'], [8, 16, 23])
                    self.assertEqual(plan['cache_shards'], shards)
                    receipt = pipeline.load(receipt_path)
                    self.assertEqual(receipt, {'argv': command, 'state': 'dispatch_requested'})
                    self.assertEqual(command[0], 'systemd-run')
                    self.assertEqual(command[-len(plan['dispatched_argv']):], plan['dispatched_argv'])
                    self.assertEqual(kwargs['timeout'], 915)
                    self.assertFalse(supervisor_path.exists())
                    return SimpleNamespace(returncode=7)

                with patch.object(pipeline.time, 'time', return_value=1000), \
                        patch.object(controller.subprocess, 'run', side_effect=launch_boundary) as launch:
                    with self.assertRaisesRegex(RuntimeError, 'full-training failed; no retry'):
                        pipeline.run_full_training(dispatch, argv, shards, runtime=runtime)
                    launch.assert_called_once()
                    self.assertEqual(plan_path.read_bytes(), observed_plan_bytes[0])
                    receipt_bytes = receipt_path.read_bytes()
                    receipt = pipeline.load(receipt_path)
                    self.assertEqual(receipt['state'], 'finished')
                    self.assertEqual(receipt['returncode'], 7)
                    self.assertNotIn('child_argv', receipt)
                    self.assertFalse(supervisor_path.exists())
                    self.assertEqual(dispatch.state['state'], 'FAILED_NO_RETRY')
                    self.assertFalse(dispatch.state['invocations'][-1]['supervisor_receipt_present'])
                    # A real duplicate unit still fails before log creation or OS launch.
                    with self.assertRaisesRegex(RuntimeError, 'Unit already dispatched; no retries'):
                        unit('full-training', ['must-not-run'], 900)
                    # Immutable intent cannot be rewritten, even after a failed dispatch.
                    with self.assertRaises(FileExistsError):
                        pipeline.run_full_training(dispatch, argv, shards, runtime=runtime)
                    with self.assertRaisesRegex(RuntimeError, 'Earlier stage failed'):
                        dispatch.run('later-child', ['must-not-run'])
                    launch.assert_called_once()
                    self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
                    self.assertEqual(plan_path.read_bytes(), observed_plan_bytes[0])
                manifest = pipeline.export_manifest(output, {})
                retained = {row['path']: row['sha256'] for row in manifest['files']}
                for path in (plan_path, receipt_path):
                    self.assertEqual(retained[path.name], pipeline.digest(path))

    def test_existing_dispatch_receipt_is_not_overwritten_or_treated_as_a_plan(self):
        for controller in (cpu, cuda):
            with self.subTest(controller=controller.__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                dispatch, argv, shards, runtime, unit = self.boundary(output, controller)
                receipt_path = output / 'full-training-command.json'
                prior = b'{"state":"dispatch_requested","argv":["prior-owned-command"]}\n'
                receipt_path.write_bytes(prior)
                with patch.object(pipeline.time, 'time', return_value=1000), \
                        patch.object(controller.subprocess, 'run') as launch:
                    with self.assertRaisesRegex(RuntimeError, 'Unit already dispatched; no retries'):
                        pipeline.run_full_training(dispatch, argv, shards, runtime=runtime)
                    launch.assert_not_called()
                self.assertEqual(receipt_path.read_bytes(), prior)
                self.assertTrue((output / 'full-training-plan.json').is_file())
                self.assertFalse((output / 'full-training.log').exists())
                self.assertFalse((output / 'full-training-supervisor.json').exists())
                self.assertEqual(dispatch.state['state'], 'FAILED_NO_RETRY')


class ExportTests(unittest.TestCase):
    def test_retains_checkpoints_excludes_only_proven_duplicate_without_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            full = output / 'full-training'
            checkpoint = full / 'checkpoints/step-000023/adapter/adapter_model.safetensors'
            root = full / 'adapter/adapter_model.safetensors'
            for path in (checkpoint, root):
                path.parent.mkdir(parents=True)
                path.write_bytes(b'actual-identical-weight')
            exclusions = pipeline.duplicate_exclusions(output, full)
            result = pipeline.export_manifest(output, exclusions)
            self.assertEqual([r['path'] for r in result['files']], [checkpoint.relative_to(output).as_posix()])
            self.assertEqual(result['excluded'][0]['sha256'], pipeline.digest(checkpoint))
            self.assertTrue(root.exists())
            root.write_bytes(b'not-a-duplicate')
            with self.assertRaises(RuntimeError):
                pipeline.duplicate_exclusions(output, full)

    def test_both_trace_sidecars_survive_probe_weight_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            probe = output / 'train/checkpoints/step-000002/adapter/adapter_model.safetensors'
            probe.parent.mkdir(parents=True)
            probe.write_bytes(b'smoke-only')
            names = ('model-smoke-phases.jsonl', 'full-training-phases.jsonl')
            for name in names:
                (output / name).write_text('{"event":"optimizer_update_returned"}\n')
            exclusions = pipeline.duplicate_exclusions(
                output, output / 'full-training', smoke_succeeded=True)
            manifest = pipeline.export_manifest(output, exclusions)
            self.assertEqual({row['path'] for row in manifest['files']}, set(names))
            self.assertEqual([row['path'] for row in manifest['excluded']],
                             [probe.relative_to(output).as_posix()])
            for row in manifest['files']:
                self.assertEqual(row['sha256'], pipeline.digest(output / row['path']))
                self.assertTrue((output / row['path']).is_file())

    def test_export_bound_is_not_silent_artifact_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weight'
            path.write_bytes(b'large-weight')
            with patch.object(pipeline, 'MAX_OUTPUT', 4 * 1024 ** 2), self.assertRaises(RuntimeError):
                pipeline.export_manifest(directory, {})
            self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
