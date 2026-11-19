"""Behavioral worker guards; no downloads, cloud requests, or GPU execution."""
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import tarfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import grounded_hf_worker as W
import grounded_hf_lifecycle as L


class PublicEvidenceSandboxTests(unittest.TestCase):
    def plan(self):
        return dict(schema=L.PUBLIC_EVIDENCE_SCHEMA, identity={'run_id': 'public-evidence-fixture'},
            limits=dict(L.LIMITS, lifetime_seconds=13500, reserve_seconds=300, output_bytes=512 * 1024**2),
            network={'egress_byte_cap': 2 * L.GIB})

    def test_all_four_stages_use_bounded_nonroot_offline_sandbox(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            for phase in ('runtime', 'capacity', 'baseline', 'train', 'adapter'):
                argv = W.stage_argv(plan, base, phase, 60, {'uuid': 'GPU-owned', 'minor': 0}, 1001, 1001)
                self.assertIn('PrivateNetwork=yes', argv)
                self.assertIn('NoNewPrivileges=yes', argv)
                self.assertIn('LimitFSIZE=536870912', argv)
                self.assertIn('ProtectSystem=strict', argv)
                self.assertEqual(argv[argv.index('--uid') + 1], '1001')
                self.assertIn('ReadWritePaths=' + str(base / 'output'), argv)
            with self.assertRaises(RuntimeError):
                W.stage_argv(plan, base, 'evaluation', 60, {}, 1001, 1001)
        self.assertEqual(W.replacement_egress_allowance(plan), 2 * L.GIB - 16 * 1024**2)

    def test_unregistered_partial_output_is_not_exported(self):
        plan = self.plan()
        plan.update(cost={'export_bytes': 400 * 1024**2}, output_allowlist=['runtime.json'])
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'output').mkdir()
            (base / 'output/private-references.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Unregistered output member'):
                W.export_outputs(base, plan, {'public_evidence_success': False}, 1000)
            self.assertFalse((base / 'export.tar').exists())


class PublicEvidenceTrainingCommandTests(unittest.TestCase):
    def test_old_public_plan_cannot_fall_back_to_generic_training(self):
        plan = {'schema': 'trid-public-evidence-gpu-plan-v1'}
        self.assertFalse(L.public_evidence_mode(plan))
        with self.assertRaisesRegex(RuntimeError, 'training fallback forbidden'):
            W.stage_argv(plan, '.', 'train', 60, {}, 1001, 1001)
        with self.assertRaisesRegex(RuntimeError, 'training fallback forbidden'):
            W.bootstrap(plan, 'a' * 64, '.')

    def test_exact_1230_exposures_reach_the_real_trainer_as_615_updates(self):
        plan = dict(schema=L.PUBLIC_EVIDENCE_SCHEMA,
            model=dict(id='Qwen/Qwen3-8B', revision=L.ORDERED_MODELS['qwen8'][1]),
            data={'manifest': {'path': 'prepared/manifest.json'}},
            recipe=dict(rows=1230, epochs=1, steps=615, batch_size=2, micro_batch_size=1,
                lr=1e-5, rank=8, alpha=16, dropout=0.0, context_tokens=4096, answer_tokens=768, seed=14))
        argv = W.train_argv(plan, '/owned')
        for option, value in (('--steps', '615'), ('--batch-size', '2'), ('--micro-batch-size', '1'),
                              ('--max-tokens', '4096'), ('--max-answer-tokens', '768'),
                              ('--model-id', 'Qwen/Qwen3-8B'), ('--device', 'cuda')):
            self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertEqual(argv[0], '/owned/code/scripts/train_open_weight_lora.py')
        self.assertNotIn('--init-adapter', argv)
        self.assertNotIn('--checkpoint-every', argv)


class CudaEvaluationDispatchTests(unittest.TestCase):
    def plan(self):
        return dict(schema=L.EVALUATION_SCHEMA, identity={'run_id': 'cuda-evaluation-fixture'},
            limits=dict(L.LIMITS, lifetime_seconds=18900, reserve_seconds=300, output_bytes=512 * 1024**2))

    def test_evaluation_network_handoff_reserves_full_bootstrap_consumption(self):
        plan = self.plan()
        plan['network'] = {'egress_byte_cap': 2 * L.GIB}
        allowance = W.replacement_egress_allowance(plan)
        self.assertEqual(allowance, 2 * L.GIB - 16 * 1024**2)
        self.assertEqual(allowance + 16 * 1024**2, plan['network']['egress_byte_cap'])
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            owner = base / 'bootstrap-network-owner'
            owner.write_text(plan['identity']['run_id'] + '\n' + W.digest(base / 'plan.json') + '\n')
            owner.chmod(0o600)
            with patch.object(W, 'command') as command:
                W.install_network_cap(plan, 2000, base)
            calls = [call.args[0] for call in command.call_args_list]
            quota = next(argv for argv in calls if '--quota' in argv)
            self.assertEqual(quota[quota.index('--quota') + 1], str(allowance))
            self.assertEqual(calls[-1], ['nft', 'delete', 'table', 'inet', 'trid_bootstrap'])
            self.assertLess(calls.index(quota), len(calls) - 1)
        for cap in (16 * 1024**2, 0, float(2 * L.GIB)):
            plan['network']['egress_byte_cap'] = cap
            with self.assertRaisesRegex(RuntimeError, 'post-bootstrap allowance'):
                W.replacement_egress_allowance(plan)

    def test_only_runtime_and_evaluation_use_the_offline_sandbox(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            for phase in ('runtime', 'evaluation'):
                argv = W.stage_argv(plan, base, phase, 60, {'uuid': 'GPU-owned', 'minor': 0}, 1001, 1001)
                self.assertIn('PrivateNetwork=yes', argv)
                self.assertIn('RestrictAddressFamilies=AF_UNIX', argv)
                self.assertIn('LimitFSIZE=536870912', argv)
                self.assertIn('ReadWritePaths=' + str(base / 'scratch'), argv)
                self.assertIn('CUDA_VISIBLE_DEVICES=GPU-owned', argv)
            for phase in ('capacity', 'train', 'adapter', 'baseline', 'unknown'):
                with self.assertRaises((RuntimeError, ValueError)):
                    W.stage_argv(plan, base, phase, 60, {}, 1001, 1001)

    def test_unknown_profile_never_defaults_to_training(self):
        plan = self.plan()
        plan['schema'] = 'unregistered-cuda-profile'
        with self.assertRaisesRegex(RuntimeError, 'training fallback forbidden'):
            W.stage_argv(plan, '.', 'train', 60, {}, 1001, 1001)
        with self.assertRaisesRegex(RuntimeError, 'training fallback forbidden'):
            W.bootstrap(plan, 'a' * 64, '.')

    def test_evaluation_supervision_binds_cleanup_and_never_replays(self):
        plan = self.plan()
        gpu = {'uuid': 'GPU-owned', 'minor': 0}
        exited = dict(LoadState='loaded', ActiveState='inactive', SubState='exited', Result='success',
            ExecMainStatus='0', Environment='TRID_OWNED_RUN=cuda-evaluation-fixture',
            InvocationID='evaluation-invocation', MemoryMax=str(L.LIMITS['rss_bytes']),
            MemorySwapMax='0', DevicePolicy='closed', User='1001', Group='1001')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            (base / 'release.json').write_text(json.dumps({'instance_id': 'i-evaluation'}))
            units = []
            with patch.object(W.time, 'time', return_value=1000), \
                    patch.object(W, 'unit_state', side_effect=[{'LoadState': 'not-found'}, exited, {'LoadState': 'not-found'}]), \
                    patch.object(W, 'gpu_observation', side_effect=lambda deadline, **kw: (gpu, 0) if kw else gpu), \
                    patch.object(W, 'verify_gpu_observation'), patch.object(W, 'cgroup_empty', return_value=True), \
                    patch.object(W, 'command') as command, \
                    patch.object(W, 'stop_unit', return_value={'ActiveState': 'inactive'}) as stop:
                receipt = W.supervise_phase(plan, base, 'evaluation', gpu, 1001, 1001, 2000, units)
                self.assertTrue(receipt['cleanup_verified'])
                self.assertEqual(receipt['instance_id'], 'i-evaluation')
                self.assertEqual(receipt['plan_sha256'], W.digest(base / 'plan.json'))
                self.assertEqual(receipt['stopped_state'], {'ActiveState': 'inactive'})
                self.assertEqual(units, [])
                with self.assertRaises(FileExistsError):
                    W.supervise_phase(plan, base, 'evaluation', gpu, 1001, 1001, 2000, units, seconds=600)
                command.assert_called_once()
                self.assertIn('RuntimeMaxSec=985', command.call_args.args[0])
                stop.assert_called_once()


class RuntimeBoundaryTests(unittest.TestCase):
    def test_unapproved_transitive_distribution_cannot_produce_runtime_receipt(self):
        distribution = lambda name, version: SimpleNamespace(metadata={'Name': name}, version=version)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'output').mkdir()
            plan = {'runtime': {'packages': {'approved': '1'}}}
            installed = [distribution('approved', '1'), distribution('unapproved', '2')]
            with patch.object(importlib.metadata, 'distributions', return_value=installed), \
                 patch.object(W, 'native_runtime', side_effect=AssertionError('Must not initialize CUDA')):
                with self.assertRaisesRegex(RuntimeError, 'package inventory differs'):
                    W.runtime_stage(plan, temp)
            self.assertFalse((Path(temp) / 'output/runtime.json').exists())

    def test_root_cannot_initialize_native_runtime(self):
        with patch.object(W.os, 'geteuid', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'nonroot without cloud credentials'):
                W.native_runtime({}, '.')

    def test_inherited_cloud_credentials_cannot_initialize_native_runtime(self):
        with patch.object(W.os, 'geteuid', return_value=1001), \
             patch.dict(W.os.environ, {'AWS_SECRET_ACCESS_KEY': 'synthetic-unit-test'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'nonroot without cloud credentials'):
                W.native_runtime({}, '.')

    def test_followup_runtime_receipt_binds_current_run_plan_and_instance(self):
        distribution = SimpleNamespace(metadata={'Name': 'torch'}, version='fixture')
        torch = SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda='fixture-cuda'),
                                cuda=Mock())
        torch.cuda.device_count.return_value = 1
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.get_device_name.return_value = 'fixture-L40S'
        plan = dict(schema=L.FOLLOWUP_SCHEMA, identity={'run_id': 'followup-runtime'},
                    runtime={'packages': {'torch': 'fixture'}})
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'output').mkdir()
            (base / 'plan.json').write_text(json.dumps(plan))
            (base / 'release.json').write_text(json.dumps({'instance_id': 'i-new-followup'}))
            with patch.object(importlib.metadata, 'distributions', return_value=[distribution]), \
                    patch.object(W, 'native_runtime', return_value=torch) as native:
                W.runtime_stage(plan, base)
            native.assert_called_once_with(plan, base)
            receipt = json.loads((base / 'output/runtime.json').read_text())
            self.assertEqual(receipt, dict(
                run_id='followup-runtime', plan_sha256=W.digest(base / 'plan.json'),
                instance_id='i-new-followup', packages={'torch': 'fixture'}, torch='fixture',
                cuda='fixture-cuda', count=1, bf16=True, name='fixture-L40S'))

    def test_followup_missing_current_release_never_writes_runtime_receipt(self):
        torch = SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda='fixture-cuda'),
                                cuda=Mock())
        plan = dict(schema=L.FOLLOWUP_SCHEMA, identity={'run_id': 'followup-runtime'},
                    runtime={'packages': {}})
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'output').mkdir()
            (base / 'plan.json').write_text(json.dumps(plan))
            with patch.object(importlib.metadata, 'distributions', return_value=[]), \
                    patch.object(W, 'native_runtime', return_value=torch):
                with self.assertRaises(FileNotFoundError):
                    W.runtime_stage(plan, base)
            self.assertFalse((base / 'output/runtime.json').exists())


class FollowupSupervisionTests(unittest.TestCase):
    def exercise_runtime(self, failure=None):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'output').mkdir()
            plan = dict(schema=L.FOLLOWUP_SCHEMA, identity={'run_id': 'followup-supervisor'},
                        workload=dict(models={'qwen14': {}}, profiles=[],
                                      development_references={'path': 'dev-references.json'}))
            (base / 'plan.json').write_text(json.dumps(plan))
            (base / 'release.json').write_text(json.dumps({'instance_id': 'i-new-followup'}))
            unit = 'followup-supervisor-runtime'
            gpu = dict(uuid='GPU-followup', minor=0)
            exited = dict(LoadState='loaded', ActiveState='inactive', SubState='exited',
                          Result='exit-code' if failure == 'native' else 'success',
                          ExecMainStatus='1' if failure == 'native' else '0',
                          Environment='TRID_OWNED_RUN=followup-supervisor', InvocationID='new-invocation',
                          MemoryMax=str(W.LIMITS['rss_bytes']), MemorySwapMax='0',
                          DevicePolicy='closed', User='1001', Group='1001')
            states = [{'LoadState': 'not-found'}]
            if failure != 'dispatch':
                states.append(exited)
            # Even if systemd has forgotten the unit, the durable intent forbids replay.
            states.append({'LoadState': 'not-found'})
            units = []
            with patch.object(W.time, 'time', return_value=1000), \
                    patch.object(W, 'unit_state', side_effect=states), \
                    patch.object(W, 'gpu_observation', side_effect=lambda deadline, **kw:
                                 (gpu, 0) if kw.get('with_usage') else gpu), \
                    patch.object(W, 'verify_gpu_observation'), \
                    patch.object(W, 'command') as command, \
                    patch.object(W, 'stop_unit') as stop:
                if failure == 'dispatch':
                    command.side_effect = RuntimeError('lost systemd response')
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'lost systemd response|Native stage failed'):
                        W.supervise_phase(plan, base, 'runtime', gpu, 1001, 1001, 2000, units)
                else:
                    receipt = W.supervise_phase(plan, base, 'runtime', gpu, 1001, 1001, 2000, units)
                    self.assertEqual(receipt, dict(phase='runtime', state='complete', unit=unit,
                        invocation='new-invocation', gpu=gpu, run_id=plan['identity']['run_id'],
                        plan_sha256=W.digest(base / 'plan.json'), instance_id='i-new-followup'))
                command.assert_called_once()
                argv, deadline = command.call_args.args
                self.assertEqual(argv[argv.index('--unit') + 1], unit)
                self.assertIn('RuntimeMaxSec=60', argv)
                self.assertEqual(deadline, 2000)
                marker = base / 'runtime-dispatch.json'
                original = marker.read_bytes()
                self.assertEqual(json.loads(original), dict(unit=unit,
                    plan_sha256=W.digest(base / 'plan.json')))
                with self.assertRaises(FileExistsError):
                    W.supervise_phase(plan, base, 'runtime', gpu, 1001, 1001, 2000, units)
                command.assert_called_once()
                self.assertEqual(marker.read_bytes(), original)
                if failure == 'dispatch':
                    self.assertEqual(units, [unit], 'Ambiguous dispatch remains owned for outer cleanup')
                    stop.assert_not_called()
                else:
                    self.assertEqual(units, [])
                    stop.assert_called_once_with(unit, plan['identity']['run_id'], 1030)

    def test_followup_runtime_supervisor_records_current_identity_and_cannot_replay(self):
        self.exercise_runtime()

    def test_followup_lost_dispatch_response_never_retries(self):
        self.exercise_runtime('dispatch')

    def test_followup_failed_native_stage_stops_without_retry(self):
        self.exercise_runtime('native')


class BootstrapObservationTests(unittest.TestCase):
    def test_observed_command_hash_must_match_explicit_expectation(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / 'aws'
            executable.write_bytes(b'changed synthetic program; never executed')
            admission = {'expected_commands': {'aws': {'path': str(executable), 'probe_args': ['--version'],
                                                       'sha256': '0' * 64}}}
            with patch.object(W.shutil, 'which', return_value=str(executable)):
                with self.assertRaisesRegex(RuntimeError, 'Expected base-command hash differs'):
                    W.observe_prerequisites(admission, 1000)

    def test_missing_expected_command_is_not_postboot_proof(self):
        admission = {'expected_commands': {'dpkg-deb': {'path': '/expected/dpkg-deb', 'probe_args': ['--version']}}}
        with patch.object(W.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Expected base-command path differs'):
                W.observe_prerequisites(admission, 1000)

    def test_upstream_setuid_mode_is_not_preserved_when_copying_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b'synthetic helper bytes; never executed'
            archive_path = root / 'data.tar'
            with tarfile.open(archive_path, 'w') as archive:
                member = tarfile.TarInfo('./usr/bin/nvidia-modprobe')
                member.mode = 0o4755
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            pin = dict(max_unpacked_bytes=1024**2, member=dict(path='usr/bin/nvidia-modprobe',
                       bytes=len(payload), sha256=W.hashlib.sha256(payload).hexdigest()))
            target = W.extract_bootstrap_member(archive_path, pin, root / 'helper')
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(target.stat().st_mode & 0o7777, 0o555)



class FixtureTokenizer:
    """Deterministic tokenizer input double; the trainer's schema/encode code is real."""
    eos_token = '\x00'
    eos_token_id = 0

    def __init__(self, suffix):
        self.suffix = suffix

    def __call__(self, text, add_special_tokens=False):
        pieces = re.findall(r' x|[\s\S]', text)
        return {'input_ids': [0 if piece == self.eos_token else 257 if piece == ' x' else ord(piece) + 1
                              for piece in pieces]}

    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<' + message['role'] + '>' + message['content'] for message in messages) + '<assistant>' + self.suffix


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'tokenizers')),
                     'Real native trainer imports require installed torch/transformers/tokenizers; no installation performed')
class GroundedSyntheticCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = runpy.run_path(str(ROOT / 'scripts/train_open_weight_lora.py'), run_name='capacity_contract_test')

    def test_real_grounded_loader_encoder_and_builder_preserve_exact_eos_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FixtureTokenizer(self.trainer['QWEN3_NO_THINK'])
            item = W.synthetic_capacity_item(tokenizer, self.trainer, temp, 8192, 768)
            source = Path(temp) / 'synthetic.jsonl'
            row = json.loads(source.read_text())
            self.assertEqual(set(row), {'id', 'task', 'split', 'messages', 'answer'})
            self.assertEqual([message['role'] for message in row['messages']], ['system', 'user'])
            self.assertIn('fictional', row['messages'][0]['content'].lower())
            self.assertEqual(len(item['ids']), 8192)
            self.assertEqual(item['labels'][:7424], [-100] * 7424)
            self.assertEqual(item['labels'][7424:], item['ids'][7424:])
            self.assertEqual(sum(value != -100 for value in item['labels']), 768)
            self.assertEqual(item['labels'][-1], tokenizer.eos_token_id)
            self.assertEqual(item['file'], str(source))
            self.assertEqual(item['task'], 'synthetic-capacity')

    def test_optional_system_and_native_multiturn_remain_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FixtureTokenizer(self.trainer['QWEN3_NO_THINK'])
            W.synthetic_capacity_item(tokenizer, self.trainer, temp, 3072, 768)
            source = Path(temp) / 'synthetic.jsonl'
            row = json.loads(source.read_text())
            current_user = row['messages'][-1]
            for messages in ([current_user],
                             [dict(role='user', content='Earlier question.'),
                              dict(role='assistant', content='Earlier answer.'), current_user]):
                with self.subTest(turns=len(messages)):
                    row['messages'] = messages
                    source.write_text(json.dumps(row) + '\n')
                    entry = dict(path=source.name, sha256=W.digest(source), adapter='grounded_messages', split='train')
                    items, dropped = self.trainer['build_items'](
                        tokenizer, [entry], Path(temp), 'qwen3', 8192, 768, False)
                    self.assertEqual(dropped, 0)
                    self.assertEqual(len(items), 1)
                    self.assertEqual(sum(value != -100 for value in items[0]['labels']), 768)
                    self.assertEqual(items[0]['labels'][-1], tokenizer.eos_token_id)

    def test_unrepresentable_envelope_fails_without_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FixtureTokenizer(self.trainer['QWEN3_NO_THINK'])
            with self.assertRaisesRegex(RuntimeError, 'cannot fit token envelope'):
                W.synthetic_capacity_item(tokenizer, self.trainer, temp, 8, 2)
            self.assertFalse((Path(temp) / 'synthetic.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
