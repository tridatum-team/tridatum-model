"""Behavioural boundary tests; no downloaded models or MPS device required."""
import contextlib
import ctypes
import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import local_mps_training as local


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.limits = dict(pressure_required=1, minimum_pressure_free_percent=10, require_ac_power=True,
            swap_growth_limit_bytes=100, process_group_rss_limit_bytes=200,
            process_group_footprint_limit_bytes=300, mps_allocator_limit_bytes=400, mps_driver_limit_bytes=500)
        self.host = dict(pressure=1, pressure_free_percent=10, ac_power=True, swap_bytes=1100)
        self.rows = [dict(rss_bytes=200, footprint_bytes=300)]

    def test_each_counter_has_separate_inclusive_bound(self):
        self.assertIsNone(local.stop_reason(self.limits, self.host, self.rows, 1000))
        for name, expected in [('rss_bytes', 'process_group_rss_limit_bytes'),
                               ('footprint_bytes', 'process_group_footprint_limit_bytes')]:
            row = dict(self.rows[0]); row[name] += 1
            self.assertEqual(local.stop_reason(self.limits, self.host, [row], 1000), expected)
        for fields, expected in [({'pressure': 2}, 'host_pressure'),
                                ({'pressure_free_percent': 9}, 'pressure_free_percent_floor'),
                                ({'ac_power': False}, 'ac_power_required'),
                                ({'swap_bytes': 1101}, 'swap_growth')]:
            self.assertEqual(local.stop_reason(self.limits, dict(self.host, **fields), self.rows, 1000), expected)

    def test_driver_is_not_allocator_or_rss(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.json'
            local.save(path, dict(allocator_bytes=400, driver_bytes=500))
            self.assertEqual(local.telemetry_check(path, self.limits)['driver_bytes'], 500)
            local.save(path, dict(allocator_bytes=1, driver_bytes=501))
            with self.assertRaisesRegex(ValueError, 'mps_driver_limit_bytes'):
                local.telemetry_check(path, self.limits)

    def test_sdk_v0_layout_and_real_buffer_address(self):
        self.assertEqual(ctypes.sizeof(local.RusageInfoV0), 96)
        self.assertEqual(local.RusageInfoV0.ri_resident_size.offset, 64)
        self.assertEqual(local.RusageInfoV0.ri_phys_footprint.offset, 72)

        def rusage(pid, flavor, pointer):
            self.assertEqual((pid, flavor), (123, 0))
            value = ctypes.cast(pointer, ctypes.POINTER(local.RusageInfoV0)).contents
            value.ri_resident_size = 20
            value.ri_phys_footprint = 37
            return 0
        library = SimpleNamespace(proc_pid_rusage=rusage)
        with patch.object(local.ctypes, 'CDLL', return_value=library):
            self.assertEqual(local.physical_footprint(123), 37)
        library.proc_pid_rusage = lambda *args: -1
        with patch.object(local.ctypes, 'CDLL', return_value=library):
            with self.assertRaises(OSError):
                local.physical_footprint(123)

    def test_host_probe_parse_and_query_only_pressure(self):
        outputs = ['1\n', 'total = 2G used = 1.5G free = 0.5G',
                   'System-wide memory free percentage: 19%', "Now drawing from 'AC Power'\n"]
        with patch.object(local, 'query', side_effect=outputs) as query:
            observation = local.host_probe()
        self.assertEqual(observation['swap_bytes'], 1610612736)
        self.assertEqual(observation['pressure_free_percent'], 19)
        self.assertEqual(query.call_args_list[2].args[0], ['/usr/bin/memory_pressure', '-Q'])
        with patch.object(local, 'query', side_effect=['1', 'unparseable']):
            with self.assertRaises(ValueError):
                local.host_probe()


class AllocatorTests(unittest.TestCase):
    def test_worker_environment_overrides_inherited_runtime_controls(self):
        inherited = dict(PYTORCH_MPS_LOW_WATERMARK_RATIO='1.4', PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.0',
                         PYTORCH_ENABLE_MPS_FALLBACK='1', HF_HUB_OFFLINE='0', UNRELATED='preserved')
        environment = local.worker_environment(inherited)
        self.assertEqual(environment['PYTORCH_MPS_LOW_WATERMARK_RATIO'], '0.5')
        self.assertEqual(environment['PYTORCH_MPS_HIGH_WATERMARK_RATIO'], '1.0')
        self.assertEqual(environment['PYTORCH_ENABLE_MPS_FALLBACK'], '0')
        self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
        self.assertEqual(environment['UNRELATED'], 'preserved')
        self.assertEqual(inherited['PYTORCH_ENABLE_MPS_FALLBACK'], '1')

    def test_explicit_watermarks_and_final_ceiling_are_recorded(self):
        applied = []
        runtime = SimpleNamespace(mps=SimpleNamespace(recommended_max_memory=lambda: 100,
                                                     set_per_process_memory_fraction=applied.append))
        with patch.dict(local.os.environ, local.WORKER_ENV):
            record = local.configure_allocator(runtime, dict(mps_allocator_limit_bytes=80))
        self.assertEqual(applied, [0.8])
        self.assertEqual(record['initial_high_watermark_bytes'], 100)
        self.assertEqual(record['low_watermark_bytes'], 50)
        self.assertEqual(record['allocator_limit_bytes'], 80)
        self.assertEqual(record['environment']['PYTORCH_ENABLE_MPS_FALLBACK'], '0')

    def test_inherited_or_incompatible_watermarks_fail_before_cap_application(self):
        applied = []
        runtime = SimpleNamespace(mps=SimpleNamespace(recommended_max_memory=lambda: 100,
                                                     set_per_process_memory_fraction=applied.append))
        with patch.dict(local.os.environ, dict(local.WORKER_ENV, PYTORCH_MPS_LOW_WATERMARK_RATIO='1.4')):
            with self.assertRaisesRegex(ValueError, 'environment'):
                local.configure_allocator(runtime, dict(mps_allocator_limit_bytes=80))
        with patch.dict(local.os.environ, local.WORKER_ENV):
            with self.assertRaisesRegex(ValueError, 'Low watermark exceeds'):
                local.configure_allocator(runtime, dict(mps_allocator_limit_bytes=49))
            with self.assertRaisesRegex(ValueError, 'Invalid allocator cap'):
                local.configure_allocator(runtime, dict(mps_allocator_limit_bytes=201))
        self.assertEqual(applied, [])


class NumericalTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Module()
        model.register_parameter('base', torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False))
        model.register_parameter('lora_A', torch.nn.Parameter(torch.ones(3)))
        model.register_parameter('lora_B', torch.nn.Parameter(torch.zeros(3)))
        return model

    def test_dtype_bytes_and_no_cpu_model_fallback(self):
        model = self.model()
        inventory = local.parameter_contract(model, device='cpu')
        self.assertEqual(inventory['base']['torch.bfloat16@cpu']['bytes'], 6)
        self.assertEqual(inventory['adapter']['torch.float32@cpu']['bytes'], 24)
        with self.assertRaisesRegex(ValueError, 'dtype/device'):
            local.parameter_contract(model)
        model.lora_A.data = model.lora_A.data.to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'Adapter dtype'):
            local.parameter_contract(model, device='cpu')

    def test_resident_second_backward_and_real_update(self):
        model = self.model()
        optimizer = torch.optim.AdamW([model.lora_A, model.lora_B], lr=1e-5, weight_decay=0)
        observer = local.Observer({}, {}, [], Path('.'))
        with patch.object(observer, 'telemetry'):
            for step in (1, 2):
                optimizer.zero_grad(set_to_none=True)
                for _ in range(4):
                    ((model.lora_A * model.lora_B - 1).square().mean() / 4).backward()
                health = observer('before_step', model=model, optimizer=optimizer, step=step)
                self.assertGreater(health['gradient_norm'], 0)
                self.assertEqual(health['resident_moments'], step == 2)
                torch.nn.utils.clip_grad_norm_([model.lora_A, model.lora_B], 1, error_if_nonfinite=True)
                optimizer.step()
                receipt = observer('after_step', model=model, optimizer=optimizer, step=step)
                self.assertTrue(receipt['changed'])
                self.assertIn('torch.float32@cpu', receipt['optimizer'])

    def test_missing_nan_zero_gradients_and_unadvanced_optimizer_fail(self):
        model = self.model()
        optimizer = torch.optim.AdamW([model.lora_A, model.lora_B])
        with self.assertRaisesRegex(ValueError, 'Missing adapter gradient'):
            local.gradient_health(model, optimizer)
        for parameter in (model.lora_A, model.lora_B):
            parameter.grad = torch.zeros_like(parameter)
        with self.assertRaisesRegex(ValueError, 'Nonpositive'):
            local.gradient_health(model, optimizer)
        model.lora_B.grad[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            local.gradient_health(model, optimizer)
        model.lora_B.grad[0] = 1
        with self.assertRaisesRegex(ValueError, 'resident Adam'):
            local.gradient_health(model, optimizer, resident=True)
        with self.assertRaisesRegex(ValueError, 'Missing Adam state'):
            local.optimizer_health(model, optimizer)

    def test_reload_requires_exact_names_values_and_dtype(self):
        model = self.model()
        expected = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
        local.verify_adapter_reload(model, expected)
        with self.assertRaisesRegex(ValueError, 'names'):
            local.verify_adapter_reload(model, {'lora_B': expected['lora_B']})
        changed_dtype = dict(expected, lora_A=expected['lora_A'].to(torch.bfloat16))
        with self.assertRaisesRegex(ValueError, 'dtype/bytes'):
            local.verify_adapter_reload(model, changed_dtype)
        with torch.no_grad():
            model.lora_B[0] = 1e-8
        with self.assertRaisesRegex(ValueError, 'dtype/bytes'):
            local.verify_adapter_reload(model, expected)

    def test_coverage_requires_each_physical_source_row_once(self):
        observer = local.Observer({}, {}, [], Path('.'))
        rows = [dict(file='/data/train.jsonl', identity=dict(id=str(index), source_sha256='abc', source_line=index + 2))
                for index in range(600)]
        observer('admission', train=rows, valid=[], dropped=0, valid_dropped=0)
        coverage = [dict(file=row['file'], **row['identity']) for row in reversed(rows)]
        observer('completion', coverage=coverage, steps=150)
        self.assertTrue(observer.complete)
        with self.assertRaisesRegex(ValueError, 'Actual row coverage'):
            observer('completion', coverage=coverage[:-1] + [coverage[0]], steps=150)


class LineageTests(unittest.TestCase):
    def test_capacity_requires_parent_acceptance_cleanup_and_unchanged_worker_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            policy = root / 'policy.json'; policy.write_text('{}')
            sources = root / 'sources.json'; sources.write_text('{}')
            worker = root / 'capacity.json'
            local.save(worker, dict(status='complete', phase='capacity', optimizer_updates=2,
                                    reload_exact=True, reload_loss=1.2))
            result = dict(schema='trid-local-mps-supervised-v1', phase='capacity', status='accepted',
                cleanup=dict(accepted=True, leader_reaped=True, remaining_members=[]),
                policy=local.pin(policy), sources=local.pin(sources), source_files=[], worker_result=local.pin(worker))
            path = root / 'result.json'

            def check(value):
                local.save(path, value)
                return local.capacity_contract(path, local.pin(path)['sha256'], local.pin(policy), local.pin(sources))

            self.assertTrue(Path(check(result)['path']).samefile(path))
            with self.assertRaisesRegex(ValueError, 'not accepted'):
                check(dict(result, status='failed'))
            with self.assertRaisesRegex(ValueError, 'cleanup'):
                check(dict(result, cleanup=dict(accepted=True, leader_reaped=True, remaining_members=[123])))
            worker.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                check(result)

    def test_admission_uses_frozen_manifest_top_level_rows_not_entry_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / 'model'; model.mkdir()
            config = model / 'config.json'; config.write_text('{}')
            acquisition = root / 'acquisition.json'
            local.save(acquisition, dict(repo_id='Qwen/Qwen3-8B', revision='frozen-revision',
                files=[dict(local.pin(config), path='config.json')]))
            rows = root / 'train-all.jsonl'
            rows.write_text('\n' + ''.join(json.dumps(dict(split='train', id=str(index))) + '\n'
                                           for index in range(600)))
            # Match the actual frozen schema: row count belongs to the manifest,
            # while train entries carry only file identity and adapter/epochs.
            entry = dict(local.pin(rows), path=rows.name, adapter='grounded_messages', epochs=1)
            manifest = dict(state='ADMITTED', rows=600, train=[entry], valid=[],
                            max_tokens=3072, max_answer_tokens=768)
            manifest_path = root / 'training-manifest.json'
            local.save(manifest_path, manifest)
            frozen = root / 'frozen-dataset.json'; frozen.write_text('{}')
            policy = dict(model=dict(id='Qwen/Qwen3-8B', revision='frozen-revision',
                                    directory=str(model), acquisition_manifest=local.pin(acquisition)),
                          data=dict(train_rows=600, training_manifest=local.pin(manifest_path),
                                    frozen_dataset=local.pin(frozen)))
            admitted_path, admitted = local.authenticate_inputs(policy)
            self.assertTrue(admitted_path.samefile(manifest_path))
            self.assertEqual(admitted['rows'], 600)
            self.assertNotIn('rows', admitted['train'][0])
            manifest['rows'] = 599
            local.save(manifest_path, manifest)
            policy['data']['training_manifest'] = local.pin(manifest_path)
            with self.assertRaisesRegex(ValueError, 'Physical training rows'):
                local.authenticate_inputs(policy)

    def test_frozen_policy_hash_cannot_be_replaced_by_caller(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'frozen'):
                local.policy_contract(path, local.pin(path)['sha256'])


class SupervisorTests(unittest.TestCase):
    def exercise(self, failure=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        out = root / 'run'
        identity = root / 'identity.json'; identity.write_text('{}')
        bound = local.pin(identity)
        limits = dict(capacity_deadline_seconds=100, training_deadline_seconds=100, sample_seconds=0.001,
            term_grace_seconds=0.01, kill_cleanup_seconds=0.1, pressure_required=1,
            minimum_pressure_free_percent=10, require_ac_power=True, swap_growth_limit_bytes=100,
            process_group_rss_limit_bytes=1000, process_group_footprint_limit_bytes=1000,
            mps_allocator_limit_bytes=1000, mps_driver_limit_bytes=1000)
        host = dict(pressure=1, pressure_free_percent=50, ac_power=True, swap_bytes=0)
        calls = []
        child = SimpleNamespace(pid=12345)
        state = dict(reaped=False, groups=0, probes=0)

        def wait(timeout):
            calls.append('reap'); state['reaped'] = True
            local.os.close(state['gate_reader'])
            return 0
        child.wait = wait

        def spawn(*args, **kwargs):
            state['gate_reader'] = local.os.dup(kwargs['pass_fds'][0])
            folder = out / 'worker'; folder.mkdir()
            local.save(folder / 'capacity.json', dict(status='complete', phase='capacity', policy=bound,
                actual_source_files=[], optimizer_updates=2, reload_exact=True, reload_loss=1.0))
            local.save(out / 'telemetry.json', dict(allocator_bytes=50, driver_bytes=75))
            return child

        def groups(pgid, footprint=True):
            state['groups'] += 1
            if state['reaped']:
                return []
            return [dict(pid=12345, pgid=12345, status='R' if state['groups'] == 1 else 'Z',
                         rss_bytes=50, footprint_bytes=75)]

        def probe():
            state['probes'] += 1
            if failure and state['probes'] >= 3:
                if failure == 'probe':
                    raise OSError('pressure probe failed')
                return dict(host, pressure=2)
            return host

        args = SimpleNamespace(out=str(out), phase='capacity', policy=str(identity), policy_sha256='x',
                               sources=str(identity), sources_sha256='y')
        with contextlib.ExitStack() as stack:
            for target, value in [((local.sys, 'platform'), 'darwin'),
                                  ((local, 'policy_contract'), lambda *a: ({'supervision': limits}, bound)),
                                  ((local, 'source_contract'), lambda *a: (bound, [])),
                                  ((local, 'host_probe'), probe), ((local, 'group_members'), groups),
                                  ((local.subprocess, 'Popen'), spawn), ((local.os, 'getpgid'), lambda pid: pid),
                                  ((local.os, 'killpg'), lambda pid, sig: calls.append(sig)),
                                  ((local.signal, 'setitimer'), lambda *a: None),
                                  ((local.signal, 'signal'), lambda *a: signal.SIG_DFL)]:
                stack.enter_context(patch.object(*target, value))
            stack.enter_context(patch.object(Path, 'home', return_value=root))
            result = local.supervise(args)
        self.assertEqual(json.loads((out / 'result.json').read_bytes()), result)
        self.assertTrue(result['cleanup']['accepted'])
        self.assertLess(calls.index(signal.SIGKILL), calls.index('reap'))
        return result

    def test_success_requires_exit_and_owned_cleanup(self):
        result = self.exercise()
        self.assertEqual(result['status'], 'accepted')
        self.assertTrue(result['startup_released'])
        self.assertEqual(result['cleanup']['remaining_members'], [])

    def test_worker_complete_cannot_override_pressure_failure(self):
        result = self.exercise('pressure')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('host_pressure', result['stop_reason'])

    def test_probe_error_is_durable_failure_even_with_complete_worker(self):
        result = self.exercise('probe')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('probe failed', result['stop_reason'])


if __name__ == '__main__':
    unittest.main()
