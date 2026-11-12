"""Synthetic offline GPU-contract regressions; no device or model execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import grounded_hf_lifecycle as L
import grounded_hf_worker as W


def fixture():
    device = dict(Name='L40S', Manufacturer='NVIDIA', Count=1, MemoryInfo={'SizeInMiB': 45776})
    catalog = dict(Gpus=[device], TotalGpuMemoryInMiB=45776)
    plan = dict(identity=dict(run_id='synthetic-gpu-observation', schedule_group='synthetic-group',
                             account='148761639313', region='ap-northeast-2'),
                runtime=dict(gpu_name='NVIDIA L40S', driver_version='synthetic-driver',
                             gpu_catalog_mib=45776, gpu_nvml_min_mib=45776, gpu_nvml_max_mib=49152,
                             python_version='3.12.3', packages={'torch': 'synthetic-torch'}, torch_cuda='12.8'),
                gpu_type_observation=copy.deepcopy(catalog['Gpus']), recipe={'seed': 14},
                iam={'scheduler_role': {'arn': 'arn:aws:iam::148761639313:role/synthetic'}})
    observed = dict(uuid='GPU-11111111-2222-3333-4444-555555555555', minor=3, name='NVIDIA L40S',
                    total_mib=47000, driver='synthetic-driver')
    return plan, catalog, observed


class GPUQuantityTests(unittest.TestCase):
    def test_catalog_and_different_physical_observation_are_independently_admitted(self):
        plan, catalog, observed = fixture()
        L.verify_gpu_catalog(plan, catalog)
        L.verify_gpu_observation(observed, plan)
        self.assertNotEqual(observed['total_mib'], catalog['TotalGpuMemoryInMiB'])

    def test_physical_range_is_inclusive_and_does_not_admit_outside_values(self):
        plan, _, observed = fixture()
        for total in (45776, 47000, 49152):
            with self.subTest(total=total):
                L.verify_gpu_observation({**observed, 'total_mib': total}, plan)
        for total in (45775, 49153, '47000'):
            with self.subTest(total=total), self.assertRaisesRegex(RuntimeError, 'Physical NVML'):
                L.verify_gpu_observation({**observed, 'total_mib': total}, plan)

    def test_catalog_must_remain_exact_even_when_wrong_value_is_physically_permitted(self):
        plan, catalog, _ = fixture()
        catalog['Gpus'][0]['MemoryInfo']['SizeInMiB'] = 47000
        catalog['TotalGpuMemoryInMiB'] = 47000
        plan['gpu_type_observation'] = copy.deepcopy(catalog['Gpus'])
        with self.assertRaisesRegex(RuntimeError, 'Exact EC2 GPU catalog'):
            L.verify_gpu_catalog(plan, catalog)

    def test_catalog_total_and_single_device_contract_cannot_drift(self):
        for mutate in (lambda catalog: catalog.update(TotalGpuMemoryInMiB=49152),
                       lambda catalog: catalog['Gpus'][0].update(Count=2)):
            with self.subTest(mutate=mutate):
                plan, catalog, _ = fixture()
                mutate(catalog)
                plan['gpu_type_observation'] = copy.deepcopy(catalog['Gpus'])
                with self.assertRaisesRegex(RuntimeError, 'Exact EC2 GPU catalog'):
                    L.verify_gpu_catalog(plan, catalog)

    def test_approved_interval_cannot_be_relaxed(self):
        plan, _, observed = fixture()
        plan['runtime']['gpu_nvml_max_mib'] = 49153
        with self.assertRaisesRegex(RuntimeError, 'Approved catalog memory'):
            L.verify_gpu_observation(observed, plan)

    def test_all_bound_observation_fields_must_remain_consistent(self):
        plan, _, observed = fixture()
        changes = dict(total_mib=47001, minor=4, uuid='GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
                       name='another-model', driver='another-driver')
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                L.verify_gpu_observation({**observed, field: value}, plan, observed)

    def test_release_accepts_measured_noncatalog_total_but_not_out_of_range_total(self):
        plan, _, observed = fixture()
        instance_id = 'i-0123456789abcdef0'
        release = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64, instance_id=instance_id,
                       launch_time=1000, deadline=15400, schedule_verified=True,
                       schedule=L.schedule_spec(plan, instance_id, 1000), gpu=observed)
        self.assertEqual(L.verify_release(release, plan, 'a' * 64, instance_id, 1001), 13600)
        release['gpu'] = {**observed, 'total_mib': 49153}
        with self.assertRaisesRegex(RuntimeError, 'Physical NVML'):
            L.verify_release(release, plan, 'a' * 64, instance_id, 1001)


class GPUOwnedMonitorTests(unittest.TestCase):
    def exercise_guard(self, changed, used):
        plan, _, approved = fixture()
        state = dict(started=False, stopped=False)
        owned = []

        def command(argv, *args, **kwargs):
            if argv[0] == 'systemd-run':
                state['started'] = True
            elif argv[:2] == ['systemctl', 'stop']:
                state['stopped'] = True
            else:
                raise AssertionError(argv)
            return ''

        def unit_state(*args):
            if not state['started']:
                return {'LoadState': 'not-found'}
            return dict(LoadState='loaded', ActiveState='inactive' if state['stopped'] else 'active',
                        SubState='exited', Result='success', ExecMainStatus='0', ControlGroup='',
                        Environment='TRID_OWNED_RUN=' + plan['identity']['run_id'], InvocationID='synthetic-invocation',
                        MemoryMax=str(L.LIMITS['rss_bytes']), MemorySwapMax='0', DevicePolicy='closed', User='1001', Group='1001')

        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'plan.json').write_text(json.dumps(plan))
            with patch.object(W, 'command', side_effect=command), patch.object(W, 'unit_state', side_effect=unit_state), \
                 patch.object(W, 'gpu_observation', side_effect=[approved, (changed, used)]), \
                 patch.object(W.time, 'time', return_value=1000):
                with self.assertRaises(RuntimeError) as caught:
                    W.supervise_phase(plan, temp, 'runtime', approved, 1001, 1001, 1100, owned)
        self.assertTrue(state['stopped'])
        self.assertEqual(owned, [])
        return str(caught.exception)

    def test_changed_nvml_total_inside_range_stops_owned_stage(self):
        _, _, approved = fixture()
        error = self.exercise_guard({**approved, 'total_mib': 47001}, 0)
        self.assertIn('Released NVML GPU observation changed', error)

    def test_physical_range_does_not_relax_external_usage_cap(self):
        _, _, approved = fixture()
        error = self.exercise_guard(approved, L.LIMITS['gpu_bytes'] + 1024**2)
        self.assertIn('External 42GiB GPU cap exceeded', error)


class NativeAllocatorQuantityTests(unittest.TestCase):
    def test_torch_capacity_may_differ_while_absolute_allocator_limit_is_preserved(self):
        plan, _, observed = fixture()
        torch_total = 45 * 1024**3
        fractions = []
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True,
            get_device_properties=lambda device: SimpleNamespace(total_memory=torch_total, name='NVIDIA L40S'),
            set_per_process_memory_fraction=lambda fraction, device: fractions.append((fraction, device)))
        torch = SimpleNamespace(__version__='synthetic-torch', version=SimpleNamespace(cuda='12.8'), cuda=cuda,
            set_num_threads=lambda count: None, manual_seed=lambda seed: None,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)))
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'release.json').write_text(json.dumps(dict(deadline=15400, gpu=observed)))
            with patch.object(W.os, 'geteuid', return_value=1001), \
                 patch.dict(W.os.environ, {'CUDA_VISIBLE_DEVICES': observed['uuid']}, clear=True), \
                 patch.object(W.platform, 'machine', return_value='x86_64'), \
                 patch.object(W.platform, 'python_version', return_value='3.12.3'), \
                 patch.object(W, 'gpu_observation', return_value=observed), \
                 patch.object(W.time, 'time', return_value=1000), patch.object(W.resource, 'setrlimit'), \
                 patch.dict(sys.modules, {'torch': torch}):
                self.assertIs(W.native_runtime(plan, temp), torch)
        self.assertNotEqual(torch_total, observed['total_mib'] * 1024**2)
        self.assertEqual(len(fractions), 1)
        self.assertEqual(fractions[0][0] * torch_total, L.LIMITS['allocator_bytes'])


if __name__ == '__main__':
    unittest.main()
