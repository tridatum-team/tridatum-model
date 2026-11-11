"""CUDA entry contract tests without GPU access, weights, downloads or cloud calls.

Runtime fakes isolate device inspection only. Mathematical checks execute the real
shared P/T implementation on tiny CPU tensors; they do not claim GPU agreement.
"""
import copy
import hashlib
import json
import math
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from scripts import train_qwen_preferences_cuda as C


UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
GPU = dict(uuid=UUID, name='NVIDIA L4', pci_bus_id='00000000:00:1E.0', total_bytes=23028 * 1024**2,
           driver_version='580.00')


def fake_torch(*, available=True, count=1, name='NVIDIA L4'):
    precision = ['medium']
    return types.SimpleNamespace(
        __version__='2.11.0+cu130', float32=torch.float32,
        version=types.SimpleNamespace(cuda='13.0'),
        cuda=types.SimpleNamespace(is_available=Mock(return_value=available), device_count=Mock(return_value=count),
                                   get_device_name=Mock(return_value=name), set_device=Mock()),
        backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
                                       cudnn=types.SimpleNamespace(allow_tf32=True, version=lambda: 91000)),
        set_float32_matmul_precision=lambda value: precision.__setitem__(0, value),
        get_float32_matmul_precision=lambda: precision[0])


class RuntimeTests(unittest.TestCase):
    def inspected(self, fake=None, gpu=None):
        fake = fake or fake_torch()
        metrics = Mock(return_value=dict(GPU if gpu is None else gpu, used_bytes=123, compute_processes=[]))
        guard = types.ModuleType('qwen_linux_guard')
        guard.gpu_metrics = metrics
        with patch.object(C, 'torch', fake), patch.object(C.platform, 'machine', return_value='x86_64'), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=UUID), patch.dict('sys.modules', qwen_linux_guard=guard):
            runtime = C.inspect_runtime()
        metrics.assert_called_once_with(UUID)
        return runtime

    def test_actual_identity_source_binding_and_explicit_fp32(self):
        fake = fake_torch()
        runtime = self.inspected(fake)
        self.assertEqual(runtime['schema'], C.RUNTIME_SCHEMA)
        self.assertEqual(runtime['trainer_entry'], C.TRAINER_ENTRY)
        self.assertEqual((runtime['device'], runtime['dtype'], runtime['architecture']),
                         ('cuda', 'torch.float32', 'x86_64'))
        self.assertEqual(runtime['gpu_uuid'], UUID)
        self.assertEqual(runtime['gpu'], GPU)
        self.assertNotIn('used_bytes', runtime['gpu'])
        self.assertNotIn('compute_processes', runtime['gpu'])
        engine = runtime['engine']
        original = C.P.engine_identity(runtime['versions'], 'cuda', torch.float32)
        for name, digest in original['code'].items():
            self.assertEqual(engine['code'][name], digest)
        self.assertEqual(set(engine['code']), set(original['code']) | {'train_qwen_preferences_cuda.py'})
        self.assertEqual(engine['code']['train_qwen_preferences_cuda.py'],
                         hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest())
        self.assertEqual(engine['math'], dict(cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
                                               float32_matmul_precision='highest'))
        self.assertEqual(engine['cuda_runtime'], '13.0')
        self.assertEqual(engine['cudnn_version'], 91000)
        self.assertEqual(engine['gpu'], GPU)
        self.assertEqual(engine['versions'], runtime['versions'])
        fake.cuda.set_device.assert_called_once_with(0)

    def test_no_cuda_no_multiple_devices_no_cpu_fallback(self):
        for fake in (fake_torch(available=False), fake_torch(count=0), fake_torch(count=2)):
            with self.subTest(fake=fake), patch.object(C, 'torch', fake), \
                    patch.object(C.platform, 'machine', return_value='x86_64'), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES=UUID), self.assertRaisesRegex(ValueError, 'no CPU fallback'):
                C.inspect_runtime()
            fake.cuda.set_device.assert_not_called()

    def test_explicit_single_uuid_and_architecture_are_mandatory(self):
        for visible in ('', '0', '0,1', UUID + ',' + UUID, 'GPU-short', ' ' + UUID, 'MIG-' + UUID):
            fake = fake_torch()
            with self.subTest(visible=visible), patch.object(C, 'torch', fake), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible), self.assertRaisesRegex(ValueError, 'UUID'):
                C.inspect_runtime()
            fake.cuda.is_available.assert_not_called()
        with patch.object(C.platform, 'machine', return_value='aarch64'), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=UUID), self.assertRaisesRegex(ValueError, 'x86_64'):
            C.inspect_runtime()

    def test_wrong_observed_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.inspected(fake_torch(name='Other GPU'))
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.inspected(gpu=dict(GPU, uuid='GPU-00000000-0000-0000-0000-000000000000'))

    def test_volatile_memory_does_not_enter_runtime(self):
        runtime = self.inspected()
        self.assertEqual(runtime['gpu'], runtime['engine']['gpu'])
        self.assertEqual(set(runtime['gpu']), {'uuid', 'name', 'pci_bus_id', 'total_bytes', 'driver_version'})

    def test_manifest_authentication_exact_bytes_types_and_current_identity(self):
        runtime = self.inspected()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runtime.json'
            path.write_text(json.dumps(runtime))
            digest = C.T.sha256(path)
            with patch.object(C, 'inspect_runtime', return_value=runtime) as inspect:
                self.assertEqual(C.authenticate_runtime(path, digest), runtime)
                inspect.assert_called_once()
            with patch.object(C, 'inspect_runtime') as inspect, self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                C.authenticate_runtime(path, '0' * 64)
            inspect.assert_not_called()
            with patch.object(C, 'inspect_runtime') as inspect, self.assertRaisesRegex(ValueError, 'lowercase SHA256'):
                C.authenticate_runtime(path, 'A' * 64)
            inspect.assert_not_called()
            for mutate in (
                lambda r: r['engine']['code'].__setitem__('train_qwen_preferences_cuda.py', '0' * 64),
                lambda r: r['engine']['math'].__setitem__('cuda_matmul_allow_tf32', True),
                lambda r: r['engine']['math'].__setitem__('cuda_matmul_allow_tf32', 0),
                lambda r: r['gpu'].__setitem__('driver_version', 'different'),
                lambda r: r['versions'].__setitem__('torch', 'different'),
                lambda r: r.__setitem__('device', 'cpu'),
                lambda r: r.__setitem__('unapproved', True),
            ):
                changed = copy.deepcopy(runtime)
                mutate(changed)
                path.write_text(json.dumps(changed))
                with patch.object(C, 'inspect_runtime', return_value=runtime), \
                        self.assertRaisesRegex(ValueError, 'differs from the actual'):
                    C.authenticate_runtime(path, C.T.sha256(path))

    def test_inspection_cli_is_exclusive_and_does_not_need_model_arguments(self):
        runtime = self.inspected()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'runtime.json'
            with patch.object(C, 'inspect_runtime', return_value=runtime):
                C.main(['--inspect-runtime', str(output)])
            self.assertEqual(json.loads(output.read_text()), runtime)
            with patch.object(C, 'inspect_runtime') as inspect, self.assertRaises(SystemExit):
                C.main(['--inspect-runtime', str(output)])
            inspect.assert_not_called()
            with self.assertRaises(FileExistsError):
                C.write_json(output, {})
            self.assertEqual(json.loads(output.read_text()), runtime)


class AdmissionTests(unittest.TestCase):
    def argv(self, root, mode='cache'):
        return ['--mode', mode, '--model', str(root / 'model'), '--model-id', 'Qwen/Qwen3-1.7B',
                '--model-revision', 'revision', '--base-gguf', str(root / 'base.gguf'),
                '--base-gguf-sha256', 'b' * 64, '--manifest', str(root / 'pairs.json'), '--out', str(root / 'out'),
                '--cuda-runtime', str(root / 'runtime.json'), '--cuda-runtime-sha256', 'a' * 64]

    def test_runtime_rejection_precedes_any_tokenizer_or_weight_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # No model, GGUF or data files exist: admission must stop before reading any of them.
            with patch.object(C, 'authenticate_runtime', side_effect=ValueError('runtime rejected')) as auth, \
                    patch.object(C, 'P', types.SimpleNamespace()) as unused, \
                    self.assertRaisesRegex(ValueError, 'runtime rejected'):
                C.main(self.argv(root))
            auth.assert_called_once_with(root / 'runtime.json', 'a' * 64)
            self.assertEqual(vars(unused), {})
            self.assertFalse((root / 'out').exists())

    def test_runtime_flags_are_required(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(C, 'authenticate_runtime') as auth:
            argv = self.argv(Path(tmp))
            with self.assertRaises(SystemExit):
                C.main(argv[:-4])
            auth.assert_not_called()

    def test_cpu_cache_rejected_before_weight_import_without_patching_frozen_globals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'
            source.write_text(json.dumps(dict(id='source', train=[], valid=[])))
            ref = dict(source_manifest=dict(path='source.json', sha256=C.T.sha256(source)), checkpoint='reference',
                       adapter_weights_sha256='w', adapter_config_sha256='c', training_json_sha256='t')
            (root / 'pairs.json').write_text(json.dumps(dict(reference=ref, train=[], valid=[])))
            shard = root / 'cpu-cache.json'
            shard.write_text(json.dumps(dict(status='complete', ids=['p1'], engine=dict(device='cpu'),
                                             scores=dict(p1=dict(chosen=-1.0, rejected=-2.0, pair_sha256='pair')))))
            p = types.SimpleNamespace(**vars(C.P))
            t = types.SimpleNamespace(**vars(C.T))
            p.replay_contract = Mock(return_value=dict(weight=0.7))
            p.AutoTokenizer = types.SimpleNamespace(from_pretrained=Mock(return_value=types.SimpleNamespace(pad_token_id=0)))
            p.tokenizer_identity = Mock(return_value={})
            p.source_rows = Mock(return_value=({}, set()))
            p.load_pairs = Mock(return_value=[dict(id='p1')])
            p.encode_pair = Mock(return_value={})
            p.pair_digest = Mock(return_value='pair')
            p.replay_rows = Mock(return_value=[])
            p.replay_digest = Mock(return_value='replay')
            t.init_adapter_ancestry = Mock(return_value=dict(adapter_weights_sha256='w', adapter_config_sha256='c',
                                                            training_json_sha256='t', source_steps=1,
                                                            source_cumulative_steps=1))
            t.load_init_adapter = Mock(side_effect=AssertionError('must not load weights'))
            fake = types.SimpleNamespace(float32=torch.float32, manual_seed=Mock())
            runtime = dict(versions={}, engine=dict(device='cuda', code={'train_qwen_preferences_cuda.py': 'digest'}))
            argv = self.argv(root, 'train') + ['--checkpoint-every', '1', '--cache', str(shard),
                                              '--cache-sha256', C.T.sha256(shard)]
            with patch.object(C, 'authenticate_runtime', return_value=runtime), patch.object(C, 'P', p), \
                    patch.object(C, 'T', t), patch.object(C, 'torch', fake), \
                    self.assertRaisesRegex(ValueError, 'reference cache rejected:.*engine differs'):
                C.main(argv)
            t.load_init_adapter.assert_not_called()
            self.assertFalse((root / 'out').exists())


class SharedMathTests(unittest.TestCase):
    def test_full_populations_and_smoke_prefix_use_unchanged_partition_math(self):
        batches = C.P.pair_batches(list(range(91)), 4)
        self.assertEqual([len(batch) for batch in batches], [4] * 22 + [3])
        self.assertEqual([p for batch in batches for p in batch], list(range(91)))
        bounds = C.P.replay_partition(288, len(batches))
        self.assertEqual([i for start, stop in bounds for i in range(start, stop)], list(range(288)))
        self.assertEqual(sum(stop - start for start, stop in bounds[:2]), 25)
        self.assertEqual([len(part) for part in C.P.pair_batches(list(range(91)), 16)], [16] * 5 + [11])

    def test_initial_agreement_is_still_one_millith_and_rejects_nonfinite(self):
        self.assertEqual(C.P.INITIAL_CACHE_MAX_ABS_ERROR, 1e-3)
        def row(delta):
            return [dict(id='p', policy=dict(chosen=-1.0 + delta, rejected=-2.0),
                         reference=dict(chosen=-1.0, rejected=-2.0))]
        self.assertLessEqual(C.P.check_initial_agreement(row(0.0009), 0.1)['initial_cache_max_abs_error'], 1e-3)
        for delta in (0.0011, float('nan'), float('inf')):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                C.P.check_initial_agreement(row(delta), 0.1)

    def test_real_shared_dpo_replay_gradient_equals_explicit_mean_objective(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.arange(49, dtype=torch.float32).reshape(7, 7) / 50)

            def forward(self, input_ids, attention_mask):
                return types.SimpleNamespace(logits=self.table[input_ids])

        model = TinyModel()
        def item(ids):
            return dict(ids=ids, labels=[-100] + ids[1:])
        pairs = [dict(chosen=item([1, 2, 3, 4]), rejected=item([1, 5])),
                 dict(chosen=item([2, 3]), rejected=item([2, 6, 1]))]
        replay = [item([1, 2]), item([2, 3, 4]), item([3, 5, 6, 1])]
        references = [(-2.0, -1.0), (-3.0, -4.0)]
        beta, weight = 0.1, 0.7
        preference = torch.stack([C.P.dpo_loss(C.P.pair_logprobs(model, pair, 'cpu', 0, 'generic'), ref, beta)[0]
                                  for pair, ref in zip(pairs, references)]).mean()
        summed, counts = C.P.sequence_losses(model, replay, 'cpu', 0, 'generic')
        replay_ce = (summed / counts).mean()
        expected = preference + weight * replay_ce
        expected.backward()
        gradient = model.table.grad.clone()
        for micro in (1, 2, 4):
            model.zero_grad(set_to_none=True)
            numbers = C.P.accumulate_step(model, [(str(i), pair, ref) for i, (pair, ref) in enumerate(zip(pairs, references))],
                                          [(str(i), row) for i, row in enumerate(replay)], beta, weight, 'cpu', 0,
                                          family='generic', replay_micro_batch=micro)
            self.assertTrue(torch.allclose(model.table.grad, gradient, atol=1e-6), micro)
            self.assertAlmostEqual(numbers['train_loss'], expected.item(), places=6)
            self.assertAlmostEqual(numbers['preference_loss'], preference.item(), places=6)
            self.assertAlmostEqual(numbers['replay_ce'], replay_ce.item(), places=6)
            self.assertEqual(numbers['replay_micro_batches'], math.ceil(len(replay) / micro))


if __name__ == '__main__':
    unittest.main()
