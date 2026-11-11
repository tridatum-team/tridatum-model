"""Offline controller boundaries only; no cloud, GPU, model, or subprocess execution."""
import base64
import copy
import datetime as dt
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.aws_eval import qwen_cuda_smoke as cuda
from scripts.aws_eval import qwen_cpu_smoke as cpu


UUID = 'GPU-11111111-2222-3333-4444-555555555555'
DEVICES = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']


def admission():
    return dict(schema='trid-cuda-admission-v1', run_id=cuda.RUN_ID,
        checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        ami=dict(ImageId='ami-040ebe879ae70cc30', OwnerId='898082745236',
            Name='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261106',
            RootDeviceName='/dev/sda1', SnapshotId='snap-03b75e669328006a7', root_volume_gib=75),
        root_volume_gib=75,
        prices=dict(ec2=0.9896, ebs=0.0912, ip=0.005, transfer=0.126, storage=0.025, put=0.0000045),
        runtime=dict(packages=copy.deepcopy(cuda.PACKAGES),
            torch_index_url='https://download.pytorch.org/whl/cu128',
            torch_runtime_version='2.9.1+cu128', cuda_version='12.8', minimum_driver_version='570.26',
            driver_compatibility_source='https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-driver',
            gpu_model='L4', gpu_count=1, gpu_min_total_bytes=6 * 1024**3, gpu_max_total_bytes=24 * 1024**3,
            torch_wheel_url='https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl',
            torch_wheel_sha256='7cb4018f4ce68b61fd3ef87dc1c4ca520731c7b5b200e360ad47b612d7844063'))


def approval(sha='a' * 64):
    return dict(run_id=cuda.RUN_ID, instance_limit=1, region=cuda.REGION, instance_type=cuda.TYPE,
        subnet_id=cuda.SUBNET, availability_zone=cuda.AZ, max_seconds=7200,
        additional_cap_usd=4.0, cumulative_cap_usd=26.0,
        prior_conservative_bound_usd=21.772065718278096, automatic_model_or_launch_retry=False,
        instance_or_az_fallback=False, fresh_pricing_and_owned_cleanup_required=True, admission_sha256=sha)


def frozen_plan():
    pins = [dict(path=f'/audit/shard-{n:03d}.json', sha256=f'{n:064x}') for n in range(1, 7)]
    child = ['scripts/train_qwen_preferences.py', '--model', '/immutable/model',
             '--model-id', 'Qwen/Qwen3-1.7B', '--model-revision', 'frozen-revision',
             '--manifest', '/immutable/manifest.json', '--base-gguf', '/immutable/base.gguf',
             '--base-gguf-sha256', 'b' * 64, '--mode', 'train', '--out', '/old/train',
             '--max-steps', '2', '--batch-pairs', '4', '--checkpoint-every', '2', '--valid-every', '2',
             '--replay-micro-batch', '4', '--seed', '20261030']
    for row in pins:
        child += ['--cache', row['path'], '--cache-sha256', row['sha256']]
    return dict(argv=['guard', '--', *child], verified_cache_shards=pins)


class AdmissionTests(unittest.TestCase):
    def test_live_inputs_required_and_capability_not_fabricated_identity(self):
        record = admission()
        self.assertIs(cuda.validate_admission(record), record)
        self.assertNotIn('driver_version', record['runtime'])
        self.assertNotIn('gpu_total_bytes', record['runtime'])
        for key in ('prices', 'ami', 'runtime', 'checked_at'):
            broken = copy.deepcopy(record)
            del broken[key]
            with self.subTest(key=key), self.assertRaises((RuntimeError, KeyError)):
                cuda.validate_admission(broken)

    def test_rejects_stale_future_and_naive_admission(self):
        record = admission()
        now = dt.datetime.now(dt.timezone.utc)
        for timestamp in (now - dt.timedelta(hours=2), now + dt.timedelta(seconds=10), now.replace(tzinfo=None)):
            record['checked_at'] = timestamp.isoformat()
            with self.subTest(timestamp=timestamp), self.assertRaises(RuntimeError):
                cuda.validate_admission(record, now=now.timestamp())

    def test_exact_pins_disk_and_floor(self):
        for where, key, value in (
            ('runtime', 'minimum_driver_version', '525.60.13'),
            ('runtime', 'cuda_version', '12.6'), ('runtime', 'gpu_count', 2),
            ('runtime', 'torch_wheel_sha256', '0' * 64),
            ('runtime', 'gpu_max_total_bytes', 48 * 1024**3), ('ami', 'root_volume_gib', 100)):
            record = admission()
            record[where][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                cuda.validate_admission(record)

    def test_budget_counts_75gib_cleanup_input_output_and_prior(self):
        record = admission()
        result = cuda.budget_bound(record['prices'], 75)
        self.assertEqual(result['ebs_gb'], 75)
        self.assertEqual(result['compute_hours'], 7500 / 3600)
        self.assertEqual(result['prior_bound_usd'], 21.772065718278096)
        self.assertLessEqual(result['additional_bound_usd'], 4)
        self.assertLessEqual(result['bound_usd'], 26)
        self.assertFalse(result['invoice'])
        self.assertGreater(result['bound_usd'], cuda.budget_bound(record['prices'], 64)['bound_usd'])
        for value in (float('nan'), float('inf'), -1, 0, True, 50):
            broken = dict(record['prices'], ec2=value)
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                cuda.budget_bound(broken, 75)
        with self.assertRaises(RuntimeError):
            cuda.budget_bound(dict(record['prices'], put=0.001), 75)

    def test_approval_binds_every_exact_boundary(self):
        cuda.verify_approval(approval(), 'a' * 64)
        for key, value in dict(run_id=cpu.RUN_ID, instance_limit=2, max_seconds=7201,
                additional_cap_usd=5, cumulative_cap_usd=14, subnet_id='subnet-other',
                availability_zone='ap-northeast-2b', automatic_model_or_launch_retry=True,
                instance_or_az_fallback=True, admission_sha256='b' * 64).items():
            record = approval()
            record[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                cuda.verify_approval(record, 'a' * 64)

    def test_consumed_gpu_approval_cannot_authorize_new_run(self):
        self.assertEqual(cuda.RUN_ID, 'qwen013-cuda-20261111015752')
        previous = approval()
        previous.update(run_id='qwen013-cuda-20261111002536', cumulative_cap_usd=22.0,
                        prior_conservative_bound_usd=18.544022428889907)
        with self.assertRaisesRegex(RuntimeError, 'Exact CUDA approval mismatch'):
            cuda.verify_approval(previous, 'a' * 64)

    def test_bound_file_requires_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text('{"ok": true}')
            sha = cuda.digest(path)
            self.assertEqual(cuda.read_bound(path, sha), {'ok': True})
            path.write_text('{"ok": false}')
            with self.assertRaises(RuntimeError):
                cuda.read_bound(path, sha)
            link = Path(tmp) / 'link.json'
            link.symlink_to(path)
            with self.assertRaises(RuntimeError):
                cuda.read_bound(link, cuda.digest(path))


class AmiMappingTests(unittest.TestCase):
    def fixture(self):
        pin = admission()['ami']
        root = dict(DeviceName='/dev/sda1', Ebs=dict(SnapshotId='snap-03b75e669328006a7',
                    VolumeSize=75, VolumeType='gp3', Iops=3000, Throughput=125, DeleteOnTermination=True))
        ami = dict(RootDeviceName='/dev/sda1', BlockDeviceMappings=[
            root, dict(DeviceName='/dev/sdb', VirtualName='ephemeral0'),
            dict(DeviceName='/dev/sdc', VirtualName='ephemeral1')])
        return ami, pin

    def test_observed_root_and_two_instance_store_descriptors_are_accepted(self):
        ami, pin = self.fixture()
        original = copy.deepcopy(ami)
        root = cuda.validate_ami_mappings(ami, pin)
        self.assertEqual(root, original['BlockDeviceMappings'][0])
        self.assertEqual(ami, original)
        # DescribeImages ordering is not a root identity guarantee.
        ami['BlockDeviceMappings'].reverse()
        self.assertEqual(cuda.validate_ami_mappings(ami, pin), root)
        ami['BlockDeviceMappings'] = [root]
        self.assertEqual(cuda.validate_ami_mappings(ami, pin), root)

    def test_extra_ebs_and_changed_root_snapshot_size_or_device_are_rejected(self):
        ami, pin = self.fixture()
        extra = copy.deepcopy(ami['BlockDeviceMappings'][0])
        extra['DeviceName'] = '/dev/sdd'
        ami['BlockDeviceMappings'].append(extra)
        with self.assertRaisesRegex(RuntimeError, 'Additional EBS'):
            cuda.validate_ami_mappings(ami, pin)
        for key, value in (('SnapshotId', 'snap-other'), ('VolumeSize', 64), ('VolumeSize', True)):
            ami, pin = self.fixture()
            ami['BlockDeviceMappings'][0]['Ebs'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                cuda.validate_ami_mappings(ami, pin)
        ami, pin = self.fixture()
        ami['BlockDeviceMappings'][0]['DeviceName'] = '/dev/sdd'
        with self.assertRaises(RuntimeError):
            cuda.validate_ami_mappings(ami, pin)
        ami, pin = self.fixture()
        ami['BlockDeviceMappings'] = ami['BlockDeviceMappings'][1:]
        with self.assertRaisesRegex(RuntimeError, 'Exactly one'):
            cuda.validate_ami_mappings(ami, pin)

    def test_ambiguous_duplicate_malformed_and_nodevice_mappings_are_rejected(self):
        replacements = [
            dict(DeviceName='/dev/sdb', VirtualName='ephemeral0', Ebs={'VolumeSize': 75}),
            dict(DeviceName='/dev/sdb', VirtualName='ephemeral0', NoDevice=''),
            dict(DeviceName='/dev/sdb', NoDevice=''),
            dict(DeviceName='/dev/sdb'),
            dict(DeviceName='/dev/sdb', VirtualName='root'),
            dict(DeviceName='/dev/sdb', VirtualName='ephemeral-1'),
            dict(DeviceName='/dev/sdb', VirtualName='ephemeral00'),
            dict(DeviceName='/dev/sdb', VirtualName=None),
            dict(DeviceName='/dev/../sdb', VirtualName='ephemeral0'),
            dict(DeviceName='/dev/sda1', VirtualName='ephemeral0'),
            dict(DeviceName='/dev/xvda', VirtualName='ephemeral0'),
            dict(DeviceName='/dev/sdc', VirtualName='ephemeral0'),
            dict(DeviceName='/dev/sdb', VirtualName='ephemeral1'),
        ]
        for replacement in replacements:
            ami, pin = self.fixture()
            ami['BlockDeviceMappings'][1] = replacement
            with self.subTest(mapping=replacement), self.assertRaises(RuntimeError):
                cuda.validate_ami_mappings(ami, pin)
        ami, pin = self.fixture()
        ami['BlockDeviceMappings'][0]['NoDevice'] = ''
        with self.assertRaises(RuntimeError):
            cuda.validate_ami_mappings(ami, pin)


class ConservativeQuotaTests(unittest.TestCase):
    def test_stopped_stopping_pending_running_gpu_vcpus_count_without_mutation(self):
        rows = [dict(InstanceId=f'i-stopped-{n}', InstanceType='g4dn.xlarge',
                     State={'Name': 'stopped'}) for n in range(3)]
        rows += [
            dict(InstanceId='i-running', InstanceType='g6.xlarge', State={'Name': 'running'}),
            dict(InstanceId='i-stopping', InstanceType='g5.xlarge', State={'Name': 'stopping'}),
            dict(InstanceId='i-pending', InstanceType='vt1.3xlarge', State={'Name': 'pending'}),
            dict(InstanceId='i-spot', InstanceType='g5.xlarge', State={'Name': 'stopped'},
                 InstanceLifecycle='spot'),
            dict(InstanceId='i-cpu', InstanceType='m7g.2xlarge', State={'Name': 'running'}),
        ]
        original = copy.deepcopy(rows)
        calls = []
        quota = 40

        def client(service, operation, *args):
            calls.append((service, operation, args))
            if (service, operation) == ('service-quotas', 'get-service-quota'):
                self.assertEqual(args, ('--service-code', 'ec2', '--quota-code', 'L-DB2E81BA'))
                return {'Quota': {'Value': quota}}
            if (service, operation) == ('ec2', 'describe-instances'):
                self.assertEqual(args, ('--filters',
                    'Name=instance-state-name,Values=pending,running,stopping,stopped'))
                return {'Reservations': [{'Instances': rows}]}
            if (service, operation) == ('ec2', 'describe-instance-types'):
                kind = args[1]
                self.assertNotEqual(kind, 'm7g.2xlarge')
                return {'InstanceTypes': [dict(InstanceType=kind,
                    VCpuInfo={'DefaultVCpus': 12 if kind == 'vt1.3xlarge' else 4})]}
            self.fail(f'Quota inventory must be read-only: {service}/{operation}')

        result = cuda.conservative_gpu_quota(client)
        self.assertEqual(result['used_vcpu_conservative'], 36)
        self.assertEqual({r['instance_id'] for r in result['instances']},
                         {r['InstanceId'] for r in rows if r['InstanceId'] != 'i-cpu'})
        self.assertEqual(rows, original)
        quota = 39
        with self.assertRaisesRegex(RuntimeError, 'including stopped GPUs'):
            cuda.conservative_gpu_quota(client)
        self.assertEqual(rows, original)
        self.assertTrue(all(operation in ('get-service-quota', 'describe-instances', 'describe-instance-types')
                            for _, operation, _ in calls))


class GpuMinorXmlTests(unittest.TestCase):
    @staticmethod
    def xml(minor='7', uuid=UUID, attached='1'):
        # Minimal vendor-shaped protocol input, not a captured hardware receipt.
        return ('<?xml version="1.0"?>\n'
                '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">\n'
                '<nvidia_smi_log><driver_version>580.95.05</driver_version>'
                f'<attached_gpus>{attached}</attached_gpus>'
                '<gpu id="00000000:00:1E.0"><product_name>NVIDIA L4</product_name>'
                f'<uuid>{uuid}</uuid><minor_number>{minor}</minor_number>'
                '</gpu></nvidia_smi_log>')

    def test_exact_uuid_resolves_actual_minor_not_enumeration_index(self):
        for minor in (0, 7, 255, 2**20 - 1):
            with self.subTest(minor=minor):
                self.assertEqual(cuda.gpu_minor_from_xml(self.xml(str(minor)), UUID), minor)
        self.assertEqual(cuda.gpu_minor_from_xml(self.xml(' 7\n'), UUID), 7)

    def test_uuid_mismatch_invalid_expected_uuid_and_unavailable_minor_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'UUID mismatch'):
            cuda.gpu_minor_from_xml(self.xml(uuid='GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'), UUID)
        for expected in ('0', '', 'GPU-' + '-' * 36, None):
            with self.subTest(expected=expected), self.assertRaisesRegex(RuntimeError, 'expected GPU UUID'):
                cuda.gpu_minor_from_xml(self.xml(), expected)
        for minor in ('N/A', 'Not Supported', '', '-1', '+1', '01', '1.5', '0x7', '٧', str(2**20)):
            with self.subTest(minor=minor), self.assertRaises(RuntimeError):
                cuda.gpu_minor_from_xml(self.xml(minor), UUID)

    def test_missing_duplicate_nested_and_multiple_gpu_identities_rejected(self):
        valid = self.xml()
        cases = [
            self.xml(attached='0'), self.xml(attached='2'), self.xml(attached='N/A'),
            valid.replace('<attached_gpus>1</attached_gpus>', ''),
            valid.replace('<attached_gpus>1</attached_gpus>', '<attached_gpus>1</attached_gpus>' * 2),
            valid.replace(f'<uuid>{UUID}</uuid>', ''),
            valid.replace(f'<uuid>{UUID}</uuid>', f'<uuid>{UUID}</uuid>' * 2),
            valid.replace('<minor_number>7</minor_number>', ''),
            valid.replace('<minor_number>7</minor_number>', '<minor_number>7</minor_number>' * 2),
            valid.replace('<minor_number>7</minor_number>', '<minor_number><value>7</value></minor_number>'),
            valid.replace('</nvidia_smi_log>', '<gpu><uuid>' + UUID + '</uuid><minor_number>8</minor_number></gpu></nvidia_smi_log>'),
            valid.replace('</gpu>', '<nested><gpu/></nested></gpu>'),
        ]
        for index, text in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                cuda.gpu_minor_from_xml(text, UUID)

    def test_malformed_wrong_root_entities_and_oversize_output_rejected(self):
        cases = ('', '<nvidia_smi_log>', '<unrelated/>', 'x' * (1024**2 + 1),
                 '<!DOCTYPE nvidia_smi_log [<!ENTITY x "7">]><nvidia_smi_log/>')
        for index, text in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                cuda.gpu_minor_from_xml(text, UUID)

    def test_cli_failure_retains_bounded_diagnostics_without_another_query(self):
        result = SimpleNamespace(returncode=2, stdout='query diagnostic ' + 'o' * 1000,
                                 stderr='driver diagnostic ' + 'e' * 2000)
        with mock.patch.object(cuda.subprocess, 'run', return_value=result) as run:
            with self.assertRaises(RuntimeError) as caught:
                cuda.gpu_device_paths(UUID)
        message = str(caught.exception)
        self.assertIn('rc=2', message)
        self.assertIn('driver diagnostic', message)
        self.assertIn('query diagnostic', message)
        self.assertLess(len(message), 1800)
        run.assert_called_once_with(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, timeout=5)

    def test_parse_failure_and_timeout_preserve_diagnostic_context(self):
        result = SimpleNamespace(returncode=0, stdout='<broken', stderr='driver warning')
        with mock.patch.object(cuda.subprocess, 'run', return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, 'Malformed NVIDIA XML.*driver warning'):
                cuda.gpu_device_paths(UUID)
            self.assertEqual(run.call_count, 1)
        timeout = cuda.subprocess.TimeoutExpired(['nvidia-smi', '-q', '-x'], 5,
                                                stderr=b'NVML initialization blocked')
        with mock.patch.object(cuda.subprocess, 'run', side_effect=timeout) as run:
            with self.assertRaisesRegex(RuntimeError, 'exceeded5 seconds.*NVML initialization blocked'):
                cuda.gpu_device_paths(UUID)
            self.assertEqual(run.call_count, 1)


class CommandTests(unittest.TestCase):
    def test_gpu_devices_explicit_with_other_restrictions_intact(self):
        argv = cuda.sandbox_argv('model-smoke', ['python', 'train.py'], 180, Path('/output'), UUID, DEVICES)
        properties = [argv[i + 1] for i, value in enumerate(argv) if value == '--property']
        for value in ('RuntimeMaxSec=175', 'TimeoutStopSec=5', 'NoNewPrivileges=yes',
                      'ProtectSystem=strict', 'MemoryMax=6G', 'MemorySwapMax=0',
                      'IPAddressDeny=any', 'DevicePolicy=closed', 'PrivateDevices=no',
                      'KillMode=control-group'):
            self.assertIn(value, properties)
        self.assertEqual([p for p in properties if p.startswith('DeviceAllow=')],
                         [f'DeviceAllow={p} rw' for p in DEVICES])
        self.assertIn('CUDA_VISIBLE_DEVICES=' + UUID, argv)
        self.assertEqual(argv[argv.index('--uid') + 1], 'qwen-smoke')
        self.assertNotIn('IPAddressAllow=localhost', properties)
        for name in ('cuda-cache-001', 'model-smoke', 'cuda-inspect'):
            with self.assertRaises(RuntimeError):
                cuda.sandbox_argv(name, [], 180, Path('/o'), UUID, DEVICES, allow_loopback=True)
        for devices in ([], ['/dev/*'], DEVICES + ['/dev/nvidia1']):
            with self.assertRaises(RuntimeError):
                cuda.sandbox_argv('model-smoke', [], 180, Path('/o'), UUID, devices)

    def test_full_outer_and_native_loopback(self):
        argv = cuda.sandbox_argv('full-training', [], 900, Path('/o'), UUID, DEVICES)
        self.assertIn('RuntimeMaxSec=895', argv)
        argv = cuda.sandbox_argv('native-case-000008-domain-001', [], 180,
                                 Path('/o'), UUID, DEVICES, allow_loopback=True)
        self.assertIn('IPAddressAllow=localhost', argv)
        self.assertIn('DevicePolicy=closed', argv)

    def test_fresh_cuda_child_preserves_frozen_data_and_two_updates(self):
        plan = frozen_plan()
        before = copy.deepcopy(plan)
        shards = [dict(path=f'/fresh/shard-{i}.json', sha256='f' * 64) for i in range(6)]
        child = cuda.cuda_child(plan, Path('/new'), Path('/runtime.json'), 'a' * 64, shards)
        self.assertEqual(plan, before)
        self.assertEqual(child[0], cuda.TRAINER)
        self.assertEqual(child[child.index('--max-steps') + 1], '2')
        self.assertEqual(child[child.index('--out') + 1], '/new/train')
        self.assertEqual(child[child.index('--model') + 1], '/immutable/model')
        self.assertEqual(child.count('--cache'), 6)
        self.assertEqual(child.count('--cuda-runtime'), 1)
        self.assertFalse(any('/audit/' in value for value in child))
        self.assertNotIn('qwen_frozen_head_vjp.py', child)
        ids = [f'pair-{i}' for i in range(16)]
        cache = cuda.cache_child(child, Path('/cache/shard.json'), ids)
        self.assertEqual(cache[cache.index('--mode') + 1], 'cache')
        self.assertEqual(cache[cache.index('--ids') + 1:], ids)
        for flag in ('--cache', '--cache-sha256', '--max-steps', '--checkpoint-every'):
            self.assertNotIn(flag, cache)
        self.assertIn('--cuda-runtime-sha256', cache)

    def test_client_does_not_mutate_cpu_deadline_or_identity(self):
        before = (cpu.END, cpu.RUN_ID, cpu.TYPE, cpu.PRIOR_BOUND_USD, cpu.ROOT)
        with mock.patch.object(cuda.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='{}', stderr='')) as run:
            cuda.Client()('ec2', 'describe-instances')
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs['env']['AWS_MAX_ATTEMPTS'], '1')
        self.assertEqual(kwargs['timeout'], 90)
        self.assertEqual(before, (cpu.END, cpu.RUN_ID, cpu.TYPE, cpu.PRIOR_BOUND_USD, cpu.ROOT))
        with mock.patch.object(cuda.time, 'time', return_value=100), mock.patch.object(cuda.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                cuda.Client(99)('ec2', 'describe-instances')
            run.assert_not_called()


class RuntimeTests(unittest.TestCase):
    def runtime(self):
        gpu = dict(uuid=UUID, name='NVIDIA L4', pci_bus_id='00000000:00:1E.0',
                   total_bytes=22888 * 1024**2, driver_version='580.95.05')
        versions = dict(torch='2.9.1+cu128', transformers='4.57.6', peft='0.17.1')
        engine = dict(cuda_runtime='12.8', gpu=gpu,
            math=dict(cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, float32_matmul_precision='highest'),
            code={**{Path(p).name: h for p, h in cpu.ENGINE_PINS.items()},
                  'train_qwen_preferences_cuda.py': 'f' * 64})
        return dict(schema='trid-cuda-runtime-v1', trainer_entry=cuda.TRAINER, device='cuda',
                    dtype='torch.float32', architecture='x86_64', versions=versions, gpu_uuid=UUID,
                    gpu=gpu, engine=engine)

    def test_postboot_actual_driver_identity_admitted_then_exactly_bound(self):
        runtime = self.runtime()
        with mock.patch.object(cuda, 'digest', return_value='f' * 64):
            cuda.validate_runtime(runtime, admission(), UUID)
            for key, value in (('driver_version', '569.99'), ('name', 'NVIDIA A10G'),
                               ('total_bytes', 5 * 1024**3), ('uuid', 'GPU-other')):
                broken = copy.deepcopy(runtime)
                broken['gpu'][key] = value
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    cuda.validate_runtime(broken, admission(), UUID)
            broken = copy.deepcopy(runtime)
            broken['engine']['math']['cuda_matmul_allow_tf32'] = True
            with self.assertRaises(RuntimeError):
                cuda.validate_runtime(broken, admission(), UUID)
            broken = copy.deepcopy(runtime)
            broken['engine']['code']['train_qwen_preferences.py'] = '0' * 64
            with self.assertRaises(RuntimeError):
                cuda.validate_runtime(broken, admission(), UUID)


class FakeClient:
    def __init__(self, fail=None):
        self.calls, self.fail, self.target, self.expression = [], fail, None, None
        self.deadline = None
        self.discovery = 0

    def instances(self, run):
        self.discovery += 1
        if self.discovery == 1:
            return []
        return [dict(InstanceId='i-offline', ClientToken=run, Tags=cpu.tags(run), State={'Name': 'terminated'})]

    def simulate(self, *args):
        return {'offline': True}

    def __call__(self, service, operation, *args, **kwargs):
        self.calls.append((service, operation, args))
        if operation == 'head-object':
            return dict(ContentLength=1, ServerSideEncryption='AES256')
        if operation == 'get-object':
            Path(args[-1]).write_bytes(b'offline archive')
            return {}
        if operation == 'run-instances':
            if self.fail == 'ambiguous':
                raise RuntimeError('ambiguous one-shot API outcome')
            return {'Instances': [dict(InstanceId='i-offline', ClientToken=cuda.RUN_ID, Tags=cpu.tags(cuda.RUN_ID))]}
        if operation == 'create-schedule':
            self.target = json.loads(args[args.index('--target') + 1])
            self.expression = args[args.index('--schedule-expression') + 1]
            return {}
        if operation == 'get-schedule':
            result = dict(State='ENABLED', ActionAfterCompletion='DELETE', ScheduleExpression=self.expression,
                          ScheduleExpressionTimezone='UTC', FlexibleTimeWindow={'Mode': 'OFF'}, Target=self.target)
            if self.fail == 'schedule':
                result['Target'] = dict(self.target, Input='{"InstanceIds":["i-unrelated"]}')
            return result
        if operation in ('put-object', 'delete-schedule'):
            return {}
        raise AssertionError((service, operation, args))


class LifecycleTests(unittest.TestCase):
    def run_launch(self, failure=None):
        client = FakeClient(failure)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(run_id=cuda.RUN_ID, bundle_sha256='b' * 64,
                bundle_s3_uri=f's3://{cuda.BUCKET}/{cuda.PREFIX}/code/{cuda.RUN_ID}/inputs.tar',
                state_dir=tmp, admission_sha256='a' * 64, approval_sha256='c' * 64,
                max_seconds=7200, approved_usd=16, allow_console_root=False)
            preflight = dict(admission=admission(), ami=dict(ImageId='ami-040ebe879ae70cc30', RootDeviceName='/dev/sda1'),
                             security_group={'GroupId': 'sg-offline'}, cost={'fixture': True})
            source_pins = {name: cuda.digest(path) for name, path in cuda.staged_source_paths().items()}
            with mock.patch.object(cuda, 'preflight', return_value=preflight), \
                 mock.patch.object(cuda, 'authenticate_archive', return_value={
                     'all_members_authenticated': True, 'source_pins': source_pins}) as authenticate, \
                 mock.patch.object(cpu, 'reconcile_cleanup', return_value={'cleanup_state': 'complete', 'launch_outcome': 'instance_found'}) as cleanup:
                if failure:
                    with self.assertRaises(RuntimeError):
                        cuda.launch(args, client)
                else:
                    cuda.launch(args, client)
                authenticate.assert_called_once()
                cleanup.assert_called_once()
                self.assertTrue((Path(tmp) / f'{cuda.RUN_ID}-approval.intent.json').is_file())
                with self.assertRaises(FileExistsError):
                    cuda.launch(args, client)
        return client

    def test_one_launch_raw_userdata_disk_tags_and_scheduler_before_release(self):
        client = self.run_launch()
        calls = [operation for _, operation, _ in client.calls]
        self.assertEqual(calls.count('run-instances'), 1)
        self.assertLess(calls.index('get-object'), calls.index('run-instances'))
        self.assertLess(calls.index('get-schedule'), calls.index('put-object'))
        args = next(args for _, op, args in client.calls if op == 'run-instances')
        request = json.loads(args[1])
        self.assertEqual(request['MinCount'], 1)
        self.assertEqual(request['MaxCount'], 1)
        self.assertEqual(request['ClientToken'], cuda.RUN_ID)
        self.assertEqual(request['InstanceType'], 'g6.xlarge')
        self.assertEqual(request['NetworkInterfaces'][0]['SubnetId'], cuda.SUBNET)
        self.assertEqual(request['BlockDeviceMappings'][0]['Ebs'], dict(VolumeSize=75, VolumeType='gp3', Encrypted=True, DeleteOnTermination=True))
        self.assertTrue(request['UserData'].startswith('#!/bin/bash\n'))
        self.assertLessEqual(len(request['UserData'].encode()), 16384)
        self.assertIn('shutdown -h +115', request['UserData'])
        self.assertEqual(client.target['RetryPolicy']['MaximumRetryAttempts'], 0)
        encoded = next(line.split('=', 1)[1] for line in request['UserData'].splitlines()
                       if line.startswith('QWEN_CONFIG='))
        config = json.loads(gzip.decompress(base64.b64decode(encoded)))
        self.assertEqual(config['admission']['root_volume_gib'], 75)
        self.assertEqual(config['deadline_unix'] - config['worker_deadline_unix'], 300)
        scratch = next(args[-1] for _, op, args in client.calls if op == 'get-object')
        self.assertTrue(scratch.startswith('/tmp/' + cuda.RUN_ID + '-inputs-'))
        self.assertFalse(Path(scratch).exists())

    def test_schedule_mismatch_never_releases(self):
        client = self.run_launch('schedule')
        self.assertNotIn('put-object', [op for _, op, _ in client.calls])
        self.assertEqual(sum(op == 'run-instances' for _, op, _ in client.calls), 1)

    def test_ambiguous_launch_never_retries_or_schedules(self):
        client = self.run_launch('ambiguous')
        self.assertEqual(sum(op == 'run-instances' for _, op, _ in client.calls), 1)
        self.assertNotIn('create-schedule', [op for _, op, _ in client.calls])
        self.assertNotIn('put-object', [op for _, op, _ in client.calls])

    def test_failed_archive_prevents_runinstances(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(run_id=cuda.RUN_ID, bundle_sha256='b' * 64,
                bundle_s3_uri=f's3://{cuda.BUCKET}/{cuda.PREFIX}/code/{cuda.RUN_ID}/inputs.tar',
                state_dir=tmp, admission_sha256='a' * 64, approval_sha256='c' * 64)
            with mock.patch.object(cuda, 'preflight', return_value={}), \
                 mock.patch.object(cuda, 'authenticate_archive', side_effect=RuntimeError('bad archive')):
                with self.assertRaisesRegex(RuntimeError, 'bad archive'):
                    cuda.launch(args, client)
        self.assertNotIn('run-instances', [op for _, op, _ in client.calls])

    def test_staged_bootstrap_controller_or_helper_mismatch_prevents_launch(self):
        paths = cuda.staged_source_paths()
        self.assertEqual(set(paths), {cuda.BOOTSTRAP, 'scripts/aws_eval/qwen_cuda_smoke.py',
                                      'scripts/aws_eval/qwen_cpu_smoke.py'})
        for changed in paths:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                client = FakeClient()
                args = SimpleNamespace(run_id=cuda.RUN_ID, bundle_sha256='b' * 64,
                    bundle_s3_uri=f's3://{cuda.BUCKET}/{cuda.PREFIX}/code/{cuda.RUN_ID}/inputs.tar',
                    state_dir=tmp, admission_sha256='a' * 64, approval_sha256='c' * 64)
                pins = {name: cuda.digest(path) for name, path in paths.items()}
                pins[changed] = '0' * 64
                with mock.patch.object(cuda, 'preflight', return_value={'admission': admission()}), \
                     mock.patch.object(cuda, 'authenticate_archive', return_value={'source_pins': pins}):
                    with self.assertRaisesRegex(RuntimeError, 'Staged launcher source differs'):
                        cuda.launch(args, client)
                self.assertNotIn('run-instances', [op for _, op, _ in client.calls])
                self.assertNotIn('create-schedule', [op for _, op, _ in client.calls])
                self.assertNotIn('put-object', [op for _, op, _ in client.calls])


class ArchiveAndBootstrapTests(unittest.TestCase):
    def test_original_129_member_archive_cap_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'inputs.tar'
            with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as tar:
                for number in range(130):
                    tar.addfile(tarfile.TarInfo(f'file-{number}'))
            with self.assertRaisesRegex(RuntimeError, 'excess archive members'):
                cuda.authenticate_archive(path, cuda.digest(path), 'a' * 64, 'b' * 64)

    def test_archive_rejects_path_traversal_symlinks_duplicates_and_final24(self):
        for name, kind in (('../escape', 'file'), ('safe', 'symlink'), ('final24/private.json', 'file'), ('duplicate', 'duplicate')):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'inputs.tar'
                with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as tar:
                    item = tarfile.TarInfo(name)
                    if kind == 'symlink':
                        item.type = tarfile.SYMTYPE
                        item.linkname = '/etc/passwd'
                        tar.addfile(item)
                    else:
                        item.size = 1
                        tar.addfile(item, io.BytesIO(b'x'))
                        if kind == 'duplicate':
                            tar.addfile(item, io.BytesIO(b'x'))
                with self.assertRaises(RuntimeError):
                    cuda.authenticate_archive(path, cuda.digest(path), 'a' * 64, 'b' * 64)



if __name__ == '__main__':
    unittest.main()
