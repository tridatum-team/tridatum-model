"""Source-only unit coverage for native binding and fail-closed receipt validation.

Frozen algorithms consume synthetic temporary scenarios and receipts, never real DEV data.
No server, converter, model, cloud or guarded workload is launched by these tests.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('qwen_cpu_native', ROOT / 'scripts/aws_eval/qwen_cpu_native.py')
N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(N)
DEVELOPMENT_SOURCE = ROOT / 'tests/fixtures/qwen_native/run_native_development.py'


def json_file(path, value):
    path.write_text(json.dumps(value))
    return N.bound(path)


def development_inputs(directory):
    """Preserve the frozen DEV36/46 shape with entirely invented user text."""
    rows = []
    for index in range(12):
        for group in ('domain', 'uncertainty', 'retention'):
            sid = f'unit-{group}-{index:02d}'
            row = {'id': sid, 'group': group}
            if len(rows) < 10:
                row['turns'] = [f'Synthetic first turn for {sid}.', f'Synthetic second turn for {sid}.']
            else:
                row['prompt'] = f'Synthetic single turn for {sid}.'
            rows.append(row)
    return {
        'development_source': N.bound(DEVELOPMENT_SOURCE),
        'scenarios': json_file(directory / 'scenarios.json', {'results': rows}),
        'owner_scenarios_1': json_file(directory / 'domain-scenarios.json',
                                       {'results': [r for r in rows if r['group'] == 'domain']}),
        'owner_scenarios_2': json_file(directory / 'behavior-scenarios.json',
                                       {'results': [r for r in rows if r['group'] != 'domain']}),
    }


def cuda_runtime():
    uuid = 'GPU-11111111-2222-3333-4444-555555555555'
    return {'schema': 'trid-cuda-runtime-v1', 'device': 'cuda', 'dtype': 'torch.float32',
            'trainer_entry': 'scripts/train_qwen_preferences_cuda.py', 'architecture': 'x86_64',
            'versions': {'torch': '2.9.1+cu128'}, 'engine': {'source_sha256': 'a' * 64},
            'gpu_uuid': uuid, 'gpu': {'uuid': uuid, 'name': 'NVIDIA L4',
                'pci_bus_id': '00000000:00:1E.0', 'total_bytes': 24 * N.GIB, 'driver_version': '580.95.05'}}


class NativeBindingTests(unittest.TestCase):
    def test_job_digest_rejects_mutation_before_any_input_read(self):
        job = {'mode': 'convert', 'checkpoint': {'step': 2}}
        job['job_sha256'] = N.canonical(job)
        job['checkpoint']['step'] = 23
        with self.assertRaisesRegex(RuntimeError, 'job binding'):
            N.validate_job(job)

    def test_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            binding = json_file(path, {'source': 11})
            path.write_text('{"source":13}')
            with self.assertRaisesRegex(RuntimeError, 'hash changed'):
                N.check_bound(binding)

    def test_predeclared_source_hash_not_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.write_bytes(b'new bytes')
            with self.assertRaisesRegex(RuntimeError, 'Predeclared input changed'):
                N.resolve_input(directory, {'path': 'source', 'sha256': '0' * 64})

    def test_unknown_mode_rejected_before_filesystem(self):
        with self.assertRaisesRegex(RuntimeError, 'Only convert/probe/case'):
            N.build_job('final', '/', '/out', '/checkpoint', {}, '/guard')

    def test_source_pin_cannot_be_replaced_by_rehashed_job(self):
        job = {'mode': 'convert', 'inputs': {k: {'sha256': v} for k, v in N.PINS.items()}}
        job['inputs']['harness_source']['sha256'] = '0' * 64
        job['job_sha256'] = N.canonical(job)
        with self.assertRaisesRegex(RuntimeError, 'frozen source pin'):
            N.validate_job(job)

    def test_unowned_platform_rejected(self):
        with patch.object(N.sys, 'platform', 'darwin'):
            with self.assertRaisesRegex(RuntimeError, 'unprivileged Linux'):
                N.authenticate_guard({})

    def test_root_execution_rejected(self):
        with patch.object(N.sys, 'platform', 'linux'), patch.object(N.os, 'geteuid', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'unprivileged Linux'):
                N.authenticate_guard({})

    def test_conversion_checkpoint_binding_cannot_cross_steps(self):
        job = {'checkpoint': {'step': 8}, 'build': {}, 'base_gguf': {}, 'inputs': {'hf_config': {}}}
        conversion = {'status': 'complete', 'checkpoint': {'step': 16}, 'build': {}, 'source_gguf': {}, 'hf_config': {}}
        with self.assertRaisesRegex(RuntimeError, 'Conversion/checkpoint/source'):
            N.validate_conversion(job, conversion)

    def test_gguf_wrong_tensor_count_fails_without_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 391, 0))
            with self.assertRaisesRegex(RuntimeError, '392 converted tensors'):
                N.gguf_metadata(path)

    def test_gguf_truncation_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.gguf'
            path.write_bytes(b'GGUF')
            with self.assertRaisesRegex(RuntimeError, 'Truncated GGUF'):
                N.gguf_metadata(path)


class BuildBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        paths = {key: self.root / name for key, name in {
            'BUILD_RECEIPT': 'native-build.json', 'SERVER': 'llama-server',
            'CONVERTER': 'convert_lora_to_gguf.py', 'CONVERTER_PYTHON': 'python'}.items()}
        for name in ('source.tar', 'llama-server', 'convert_lora_to_gguf.py', 'python'):
            (self.root / name).write_bytes(name.encode())
        converter_hash = N.digest(paths['CONVERTER'])
        patches = [patch.multiple(N, **paths, CONVERTER_SHA=converter_hash),
                   patch.object(N, 'root_controlled')]
        for patched in patches:
            patched.start()
            self.addCleanup(patched.stop)
        self.archive = N.bound(self.root / 'source.tar')
        self.record = {
            'completed': True, 'source_commit': N.COMMIT, 'source_archive': self.archive,
            'server': {**N.bound(paths['SERVER']), 'build': 'b1-8ca4060',
                       'version': 'version: 1 (8ca4060)', 'shared_libraries': []},
            'converter': {**N.bound(paths['CONVERTER']), 'python': str(paths['CONVERTER_PYTHON'])},
            'training_versions_before': {'torch': 'unchanged'},
            'training_versions_after': {'torch': 'unchanged'},
        }
        json_file(paths['BUILD_RECEIPT'], self.record)
        self.native = {'build_receipt': str(paths['BUILD_RECEIPT']),
                       'build': {'source_archive': 'source.tar',
                                 'source_archive_sha256': self.archive['sha256']}}

    def test_setup_plan_archive_path_and_separate_hash_are_accepted(self):
        result = N.build_binding(self.root, self.native)
        self.assertEqual(result['record']['source_archive'], self.archive)
        self.assertEqual(result['receipt']['sha256'], N.digest(self.native['build_receipt']))

    def test_setup_plan_archive_bad_hash_is_refused(self):
        self.native['build']['source_archive_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'Predeclared input changed'):
            N.build_binding(self.root, self.native)

    def test_x86_build_requires_actual_cpu_converter_and_architecture(self):
        self.record.update(architecture='x86_64', cmake_flags=['-DGGML_CUDA=OFF', '-DGGML_NATIVE=OFF'])
        self.record['server']['version'] = 'build 1 (8ca4060) Linux x86_64'
        self.record['converter'].update(torch_cuda=None, torch_version='2.11.0+cpu',
                                        versions={'torch': '2.11.0+cpu'})
        json_file(N.BUILD_RECEIPT, self.record)
        N.build_binding(self.root, self.native, runtime=cuda_runtime())
        for field, value in (('torch_cuda', '12.8'), ('torch_version', '2.11.0')):
            altered = copy.deepcopy(self.record)
            altered['converter'][field] = value
            json_file(N.BUILD_RECEIPT, altered)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'CPU converter'):
                N.build_binding(self.root, self.native, runtime=cuda_runtime())
        for field, value in (('architecture', 'aarch64'),
                             ('cmake_flags', ['-DGGML_CUDA=ON', '-DGGML_NATIVE=OFF'])):
            json_file(N.BUILD_RECEIPT, {**self.record, field: value})
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                N.build_binding(self.root, self.native, runtime=cuda_runtime())


class CheckpointBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'step-000002'
        adapter = self.checkpoint / 'adapter'
        adapter.mkdir(parents=True)
        tensors = {f'base.layers.{i}.lora_A.weight': {'dtype': 'F32', 'shape': [16, 1],
                   'data_offsets': [i * 64, (i + 1) * 64]} for i in range(392)}
        header = json.dumps(tensors).encode()
        self.weights = adapter / 'adapter_model.safetensors'
        self.weights.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * (392 * 64))
        self.config = {'peft_type': 'LORA', 'r': 16, 'lora_alpha': 32,
                       'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
        json_file(adapter / 'adapter_config.json', self.config)
        source = self.root / 'source011' / 'adapter'
        source.mkdir(parents=True)
        self.inputs = {'manifest': {'sha256': 'manifest'}, 'source_weights': {'path': str(source / 'adapter_model.safetensors')}}
        ancestry = {'adapter_weights_sha256': N.PINS['source_weights'], 'adapter_config_sha256': N.PINS['source_config'],
                    'training_json_sha256': N.PINS['source_training'], 'source_steps': 24, 'source_cumulative_steps': 96,
                    'path': str(source), 'optimizer': 'reset: fresh AdamW', 'scheduler': 'reset: fresh warmup+cosine'}
        self.training = {'status': 'checkpoint', 'steps': 2, 'base_revision': N.REVISION, 'manifest_sha256': 'manifest',
                         'checkpoint': {'step': 2, 'adapter_weights_sha256': N.digest(self.weights), 'path': str(adapter)},
                         'base_weights': {'gguf_sha256': N.BASE_SHA, 'representation': 'q8_0'},
                         'init_adapter': ancestry, 'reference': dict(ancestry)}

    def bind(self, *, runtime=None):
        json_file(self.checkpoint / 'training.json', self.training)
        return N.checkpoint_binding(self.root, self.checkpoint, self.inputs, runtime=runtime)

    def test_nested_checkpoint_and_adjacent_training_bind(self):
        result = self.bind()
        self.assertEqual(result['step'], 2)
        self.assertEqual(len(result['tensor_metadata']), 392)
        self.assertEqual(result['training']['sha256'], N.digest(self.checkpoint / 'training.json'))

    def test_adapter_weights_must_match_training_hash(self):
        self.training['checkpoint']['adapter_weights_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint/training'):
            self.bind()

    def test_source011_ancestry_cannot_change(self):
        self.training['reference']['source_cumulative_steps'] = 98
        with self.assertRaisesRegex(RuntimeError, 'Source011 ancestry'):
            self.bind()

    def test_manifest_must_match_predeclared_input(self):
        self.training['manifest_sha256'] = 'other-manifest'
        with self.assertRaisesRegex(RuntimeError, 'base/manifest'):
            self.bind()

    def test_rank_cannot_change(self):
        self.config['r'] = 8
        json_file(self.checkpoint / 'adapter/adapter_config.json', self.config)
        with self.assertRaisesRegex(RuntimeError, 'LoRA configuration'):
            self.bind()

    def test_cuda_checkpoint_requires_exact_runtime_engine_and_fp32(self):
        runtime = cuda_runtime()
        self.training.update(device='cuda', dtype='torch.float32', cuda_runtime=runtime,
                             engine=runtime['engine'], versions=runtime['versions'],
                             cuda_runtime_sha256='b' * 64)
        self.assertEqual(self.bind(runtime=runtime)['step'], 2)
        with self.assertRaisesRegex(RuntimeError, 'explicit authenticated runtime'):
            self.bind()
        original = copy.deepcopy(self.training)
        for field, value in (('cuda_runtime', {}), ('engine', {}), ('versions', {}),
                             ('device', 'cpu'), ('dtype', 'torch.bfloat16'),
                             ('cuda_runtime_sha256', None)):
            self.training = {**original, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                self.bind(runtime=runtime)

    def test_cuda_runtime_is_part_of_hashed_native_job(self):
        job = {'mode': 'convert', 'cuda_runtime': cuda_runtime()}
        job['job_sha256'] = N.canonical(job)
        job['cuda_runtime']['gpu_uuid'] = 'GPU-other'
        with self.assertRaisesRegex(RuntimeError, 'job binding'):
            N.validate_job(job)


class RecoveredAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = development_inputs(self.root)
        self.algorithms = N.original_development(self.inputs)
        self.case_id, self.case = next(iter(self.algorithms['load_scenarios']().items()))
        self.users = self.case['users']
        self.baseline = {
            'model_file': str(self.root / 'synthetic-base.gguf'), 'model_sha256': N.BASE_SHA,
            'llama_cpp': N.COMMIT, 'suite_sha256': self.inputs['scenarios']['sha256'],
            'arms': ['model'], 'lora_file': None, 'lora_sha256': None,
            'binding': {**self.algorithms['EXPECTED_BINDING'], 'port': 18001},
            'server': {'build': 'unit-build-8ca4060', 'chat_template_sha256': 'b' * 64},
            'render_parity': {'served_prompt': 'Synthetic rendered prompt.', 'served_token_ids': [1, 2],
                              'text_equal': True, 'token_ids_equal': True},
            'requests': [
                {'user': user, 'generation_settings': {'temperature': 0, 'seed': 42, 'lora': []},
                 'rendered_prompt_sha256': hashlib.sha256(f'Synthetic prompt {index}'.encode()).hexdigest(),
                 'prompt_token_ids_sha256': hashlib.sha256(json.dumps([index + 1]).encode()).hexdigest()}
                for index, user in enumerate(self.users)],
        }
        baseline_path = self.root / 'baseline-receipt.json'
        json_file(baseline_path, self.baseline)
        execution_path = self.root / 'execution.json'
        json_file(execution_path, {'arm': 'model', 'state': 'complete', 'cases': [
            {'id': self.case_id, 'verified': True, 'receipt': str(baseline_path)}]})
        self.baseline_path = Path(N.load(execution_path)['cases'][0]['receipt'])
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate.update(arms=['adapter'], lora_file=str(self.root / 'synthetic-adapter.gguf'),
                              lora_sha256='a' * 64)
        self.candidate['binding']['port'] = 18002
        for request in self.candidate['requests']:
            request['generation_settings']['lora'] = [{'id': 0, 'scale': 1.0}]
        self.candidate_path = self.root / 'candidate-receipt.json'
        self.job = {'inputs': self.inputs, 'baseline': {'receipt': N.bound(self.baseline_path)},
                    'case': {'id': self.case_id, **self.case}}

    def compare(self):
        json_file(self.candidate_path, self.candidate)
        return self.algorithms['compare_with_baseline'](
            self.candidate_path, self.baseline_path, self.users)

    def test_original_scenario_order_and_turn_counts(self):
        cases = self.algorithms['load_scenarios']()
        original = N.load(self.inputs['scenarios']['path'])['results']
        self.assertEqual(list(cases), [r['id'] for r in original])
        self.assertEqual(len(cases), 36)
        self.assertEqual(sum(len(c['users']) for c in cases.values()), 46)
        self.assertEqual([c['order'] for c in cases.values()], list(range(36)))
        for row in original:
            self.assertEqual(cases[row['id']]['group'], row['group'])
            self.assertEqual(cases[row['id']]['users'],
                             row['turns'] if 'turns' in row else [row['prompt']])

    def test_original_scenarios_reject_changed_owner_inputs(self):
        for field, value in (('group', 'retention'), ('turns', ['Different synthetic user.'])):
            with self.subTest(field=field):
                owner = N.load(self.inputs['owner_scenarios_1']['path'])
                owner['results'][0][field] = value
                inputs = dict(self.inputs)
                inputs['owner_scenarios_1'] = json_file(self.root / 'changed-owner.json', owner)
                with self.assertRaisesRegex(ValueError, 'differs from the data-owner file'):
                    N.original_development(inputs)['load_scenarios']()

    def test_original_scenarios_reject_count_group_and_duplicate_id_changes(self):
        original = N.load(self.inputs['scenarios']['path'])
        for mutation in ('count', 'group', 'duplicate-id'):
            changed = copy.deepcopy(original)
            if mutation == 'count':
                changed['results'].pop()
            elif mutation == 'group':
                changed['results'][0]['group'] = 'uncertainty'
            else:
                changed['results'][0]['id'] = changed['results'][1]['id']
            inputs = dict(self.inputs)
            inputs['scenarios'] = json_file(self.root / 'changed-scenarios.json', changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                N.original_development(inputs)['load_scenarios']()

    def test_original_source_and_scenario_bindings_are_checked(self):
        self.assertEqual(self.inputs['development_source']['sha256'], N.PINS['development_source'])
        for key in self.inputs:
            inputs = copy.deepcopy(self.inputs)
            inputs[key]['sha256'] = '0' * 64
            with self.subTest(input=key), self.assertRaisesRegex(RuntimeError, 'hash changed'):
                N.original_development(inputs)

    def test_original_compare_rejects_server_build_difference(self):
        self.assertTrue(self.compare()['equal'])
        self.candidate['server']['build'] = 'b2-othercommit'
        different = self.compare()
        self.assertFalse(different['equal'])
        self.assertFalse(different['checks']['server_build_equal'])

    def test_original_compare_does_not_normalize_paths(self):
        self.candidate['model_file'] = str(self.root / 'different-base.gguf')
        result = self.compare()
        self.assertFalse(result['checks']['model_file'])
        self.assertFalse(result['equal'])

    def test_every_original_parity_check_is_retained(self):
        result = self.compare()
        self.assertTrue(result['equal'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(set(result['checks']), {
            'model_file', 'model_sha256', 'llama_cpp', 'suite_sha256', 'binding_equal', 'server_build_equal',
            'chat_template_sha256_equal', 'render_parity_equal', 'baseline_arm_model_without_lora',
            'candidate_arm_adapter', 'turn_count_equal', 'user_strings_equal', 'generation_settings_equal',
            'turn0_rendered_prompt_sha256_equal', 'turn0_prompt_token_ids_sha256_equal'})
        self.assertEqual(result['baseline_receipt_sha256'], N.digest(self.baseline_path))
        self.assertEqual(result['candidate_receipt_sha256'], N.digest(self.candidate_path))

    def test_original_parity_rejects_each_changed_binding(self):
        mutations = [
            ('model_file', ('model_file',), str(self.root / 'other-base.gguf')),
            ('model_sha256', ('model_sha256',), '0' * 64),
            ('llama_cpp', ('llama_cpp',), 'other-commit'),
            ('suite_sha256', ('suite_sha256',), '0' * 64),
            ('binding_equal', ('binding', 'threads'), 8),
            ('server_build_equal', ('server', 'build'), 'other-build'),
            ('chat_template_sha256_equal', ('server', 'chat_template_sha256'), '0' * 64),
            ('render_parity_equal', ('render_parity', 'served_token_ids'), [999]),
            ('candidate_arm_adapter', ('arms',), ['model']),
            ('turn_count_equal', ('requests',), self.candidate['requests'][:-1]),
            ('user_strings_equal', ('requests', 0, 'user'), 'Different synthetic user.'),
            ('generation_settings_equal', ('requests', 0, 'generation_settings', 'seed'), 43),
            ('turn0_rendered_prompt_sha256_equal', ('requests', 0, 'rendered_prompt_sha256'), '0' * 64),
            ('turn0_prompt_token_ids_sha256_equal', ('requests', 0, 'prompt_token_ids_sha256'), '0' * 64),
        ]
        original = copy.deepcopy(self.candidate)
        for check, path, value in mutations:
            self.candidate = copy.deepcopy(original)
            target = self.candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(check=check):
                result = self.compare()
                self.assertFalse(result['equal'])
                self.assertFalse(result['checks'][check])
                with self.assertRaisesRegex(RuntimeError, 'Original baseline parity failed'):
                    N.baseline_parity(self.job, self.candidate_path)
        self.candidate = original
        self.baseline['lora_file'] = str(self.root / 'unexpected-adapter.gguf')
        json_file(self.baseline_path, self.baseline)
        result = self.compare()
        self.assertFalse(result['equal'])
        self.assertFalse(result['checks']['baseline_arm_model_without_lora'])

    def test_original_parity_allows_own_history_after_identical_first_turn(self):
        self.candidate['requests'][1]['rendered_prompt_sha256'] = 'c' * 64
        self.candidate['requests'][1]['prompt_token_ids_sha256'] = 'd' * 64
        self.assertTrue(self.compare()['equal'])
        self.assertTrue(N.baseline_parity(self.job, self.candidate_path)['equal'])

    def test_baseline_receipt_cannot_be_changed_after_binding(self):
        self.compare()
        self.baseline['server']['build'] = 'changed-build'
        json_file(self.baseline_path, self.baseline)
        with self.assertRaisesRegex(RuntimeError, 'hash changed'):
            N.baseline_parity(self.job, self.candidate_path)


class ProbeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        inputs = development_inputs(directory)
        self.planned = [dict(id=f'train-{i}', user=f'user-{i}', messages=[{'role': 'user', 'content': f'user-{i}'}],
                             rendered_prompt=f'prompt-{i}', prompt_token_ids=[i + 1]) for i in range(4)]
        plan = json_file(directory / 'plan.json', {'rows': self.planned})
        self.binding = {'host': '127.0.0.1', 'threads': 4, 'context': 2048, 'max_new': 384, 'batch': 256,
                        'ubatch': 128, 'no_repack': True, 'no_context_shift': True, 'n_gpu_layers': 0,
                        'decoding': {'temperature': 0, 'seed': 42}, 'template_kwargs': {'enable_thinking': False},
                        'system_prompt': None, 'cache_prompt': False}
        self.job = {'mode': 'probe', 'base_gguf': {'path': '/actual/base.gguf'},
                    'conversion': {'adapter_gguf': {'path': '/actual/adapter.gguf', 'sha256': 'a' * 64}},
                    'build': {'record': {'server': {'build': 'b1-8ca4060'}}},
                    'inputs': {**inputs, 'probe_plan': plan, 'probe_suite': {'path': '/probe/suite', 'sha256': 'suite'},
                               'official_render': {'path': '/official/render'}}}
        rows = []
        for i, row in enumerate(self.planned):
            rows.append({'kind': 'native', 'arm': 'adapter', 'session': row['id'], 'turn': 0, 'user': row['user'],
                         'history_turns': 0, 'reset': False, 'http_status': 200, 'fits_context': True,
                         'stopped_at_eos': True, 'stop_type': 'eos', 'truncated': False, 'output': 'answer',
                         'prompt': row['rendered_prompt'], 'prompt_token_ids': row['prompt_token_ids'],
                         'rendered_prompt_sha256': hashlib.sha256(row['rendered_prompt'].encode()).hexdigest(),
                         'prompt_token_ids_sha256': hashlib.sha256(json.dumps(row['prompt_token_ids']).encode()).hexdigest(),
                         'pressure': {'full': {'total': 0}}, 'generation_settings': {'temperature': 0},
                         'wall_seconds': 1, 'prompt_tokens': 1, 'generated_tokens': 1})
        self.record = {'status': 'complete', 'arms': ['adapter'], 'backend': 'base_gguf+lora',
                       'model_sha256': N.BASE_SHA, 'model_file': '/actual/base.gguf',
                       'lora_file': '/actual/adapter.gguf', 'lora_sha256': 'a' * 64,
                       'binding': self.binding, 'env': {'LD_LIBRARY_PATH': str(N.SERVER.parent)},
                       'timeouts': {'duration': 175, 'ready': 30, 'startup': 1, 'request': 32},
                       'render_parity': {'text_equal': True, 'token_ids_equal': True, 'no_think_suffix': True,
                                         'official_render': '/official/render'},
                       'server': {'build': 'b1-8ca4060', 'chat_template_sha256': 'template',
                                  'lora_adapters': [{'path': '/actual/adapter.gguf', 'scale': 1.0}]},
                       'arm_switches': [{'arm': 'adapter', 'scale': 1.0}], 'live_members_after_shutdown': [],
                       'memory': {'under_ceiling': True, 'group_rss_bytes': 100}, 'stop_signal': None,
                       'suite_sha256': 'suite', 'suite_file': '/probe/suite', 'ids': None,
                       'requests': rows, 'elapsed_seconds': 8}

    def test_four_exact_train_turns_pass(self):
        result = N.validate_harness(self.job, self.record)
        self.assertEqual(result['turns_total'], 4)
        self.assertTrue(result['stopped_at_eos'])

    def test_wrong_active_adapter_fails(self):
        self.record['server']['lora_adapters'][0]['path'] = '/other.gguf'
        with self.assertRaisesRegex(RuntimeError, 'active LoRA'):
            N.validate_harness(self.job, self.record)

    def test_disabled_adapter_fails(self):
        self.record['server']['lora_adapters'][0]['scale'] = 0
        with self.assertRaisesRegex(RuntimeError, 'active LoRA'):
            N.validate_harness(self.job, self.record)

    def test_missing_turn_fails(self):
        self.record['requests'].pop()
        with self.assertRaisesRegex(RuntimeError, 'Missing or repeated'):
            N.validate_harness(self.job, self.record)

    def test_http_200_does_not_replace_eos(self):
        self.record['requests'][2].update(stop_type='limit', stopped_at_eos=False, truncated=True)
        with self.assertRaisesRegex(RuntimeError, 'EOS/nonempty'):
            N.validate_harness(self.job, self.record)

    def test_empty_answer_fails(self):
        self.record['requests'][0]['output'] = '  '
        with self.assertRaisesRegex(RuntimeError, 'EOS/nonempty'):
            N.validate_harness(self.job, self.record)

    def test_official_token_parity_failure(self):
        self.record['render_parity']['token_ids_equal'] = False
        with self.assertRaisesRegex(RuntimeError, 'Official render parity'):
            N.validate_harness(self.job, self.record)

    def test_train_tokens_cannot_be_rehashed_to_different_prompt(self):
        row = self.record['requests'][1]
        row['prompt_token_ids'] = [999]
        row['prompt_token_ids_sha256'] = hashlib.sha256(json.dumps([999]).encode()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, 'TRAIN exact render/token'):
            N.validate_harness(self.job, self.record)

    def test_mac_pressure_number_not_accepted(self):
        self.record['requests'][0]['pressure'] = 1
        with self.assertRaisesRegex(RuntimeError, 'real Linux PSI'):
            N.validate_harness(self.job, self.record)

    def test_live_server_cannot_be_called_complete(self):
        self.record['live_members_after_shutdown'] = [{'pid': 123}]
        with self.assertRaisesRegex(RuntimeError, 'resource/shutdown'):
            N.validate_harness(self.job, self.record)


class GuardReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'guard.json'
        self.job = {'guard_receipt': str(self.path)}
        self.receipt = {'ownership': {'parent_pid': 10, 'pid': 11, 'pgid': 11, 'cgroup_path': '/sys/fs/cgroup/owned'},
                        'argv': ['python', 'native.py']}
        sample = {'cgroup': {'memory_max_bytes': 6 * N.GIB, 'swap_max_bytes': 0, 'swap_current_bytes': 0,
                            'events': {'oom': 0, 'oom_kill': 0}}, 'rss_bytes': 100,
                  'host': {'memory_psi': {'full': {'total': 4}}, 'memory_bytes': {'MemAvailable': 9 * N.GIB}}}
        self.guard = {'supervisor_pid': 10, 'child_pid': 11, 'owned_pgid': 11,
                      'cgroup_path': '/sys/fs/cgroup/owned', 'seconds': 180, 'startup_released': True,
                      'stop_reason': 'child_exit', 'first_stop_reason': None, 'child_exit_code': 0,
                      'reaped': True, 'cleanup_state': 'completed', 'cleanup_errors': [],
                      'live_members_before_reap': [], 'guard_exit_code': 0, 'rss_peak_bytes': 100,
                      'elapsed_seconds': 12, 'argv': ['python', 'native.py'], 'samples': [sample, copy.deepcopy(sample)]}

    def validate(self):
        json_file(self.path, self.guard)
        return N.validate_final_guard(self.job, self.receipt)

    def test_complete_owned_guard_passes(self):
        self.assertTrue(self.validate()['reaped'])

    def test_unowned_guard_rejected(self):
        self.guard['supervisor_pid'] = 99
        with self.assertRaisesRegex(RuntimeError, 'outer guard'):
            self.validate()

    def test_provisional_guard_rejected(self):
        self.guard['cleanup_state'] = 'not_started'
        with self.assertRaisesRegex(RuntimeError, 'outer guard'):
            self.validate()

    def test_psi_full_increase_is_fatal(self):
        self.guard['samples'][1]['host']['memory_psi']['full']['total'] += 1
        with self.assertRaisesRegex(RuntimeError, 'cgroup/PSI'):
            self.validate()

    def test_wrong_command_rejected(self):
        self.guard['argv'] = ['python', 'other.py']
        with self.assertRaisesRegex(RuntimeError, 'argv'):
            self.validate()

    def test_swap_or_oom_is_fatal(self):
        self.guard['samples'][1]['cgroup']['swap_current_bytes'] = 1
        with self.assertRaisesRegex(RuntimeError, 'cgroup/PSI'):
            self.validate()

    def test_native_cuda_job_requires_complete_matching_gpu_evidence(self):
        runtime = cuda_runtime()
        self.job.update(cuda_runtime=runtime,
                        guard_source=N.bound(ROOT / 'scripts/qwen_linux_guard.py'))
        with self.assertRaises(ValueError):
            self.validate()
        self.guard['gpu'] = {
            'schema': 'trid-gpu-guard-v1', 'expected_uuid': runtime['gpu_uuid'],
            'memory_limit_bytes': 6 * N.GIB, 'identity': runtime['gpu'], 'used_peak_bytes': 100}
        sample = self.guard['samples'][0]
        self.guard['samples'] = []
        for index, phase in enumerate(('admission', 'pre_exec', 'runtime')):
            row = copy.deepcopy(sample)
            row.update(gpu_phase=phase, elapsed_seconds=index, members=[], gpu_process_observations=[],
                       gpu={**runtime['gpu'], 'used_bytes': 100, 'compute_processes': []})
            self.guard['samples'].append(row)
        self.assertTrue(self.validate()['reaped'])
        self.job['cuda_runtime'] = copy.deepcopy(runtime)
        self.job['cuda_runtime']['gpu']['driver_version'] = 'other-driver'
        with self.assertRaisesRegex(RuntimeError, 'GPU identity differs'):
            self.validate()

    def test_consuming_conversion_revalidates_gpu_receipt_and_guard_source(self):
        root = Path(self.temp.name).resolve(strict=True)
        runtime = cuda_runtime()
        config_dir = root / 'config-only'
        config_dir.mkdir()
        config = json_file(config_dir / 'config.json', {'model_type': 'qwen3'})
        adapter = root / 'adapter.gguf'
        def string(value):
            encoded = value.encode()
            return struct.pack('<Q', len(encoded)) + encoded
        data = bytearray(b'GGUF' + struct.pack('<IQQ', 3, 392, 4))
        for key, value in (('general.architecture', 'qwen3'), ('general.type', 'adapter'),
                           ('adapter.type', 'lora')):
            data.extend(string(key) + struct.pack('<I', 8) + string(value))
        data.extend(string('adapter.lora.alpha') + struct.pack('<II', 4, 32))
        for index in range(392):
            data.extend(string(f'layer.{index}.lora_a')
                        + struct.pack('<IQQIQ', 2, 16, 1, 0, index * 64))
        data.extend(b'\0' * ((-len(data)) % 32 + 392 * 64))
        adapter.write_bytes(data)
        log = root / 'converter.log'
        log.write_text('unit fixture: conversion contract validation only\n')
        checkpoint = {'weights': {'path': str(root / 'weights/adapter_model.safetensors')}}
        job = {'mode': 'probe', 'checkpoint': checkpoint, 'build': {}, 'base_gguf': {},
               'inputs': {'hf_config': config}, 'cuda_runtime': runtime,
               'guard_source': N.bound(ROOT / 'scripts/qwen_linux_guard.py'),
               'guard_receipt': str(root / 'consumer-guard.json')}
        conversion = {
            'status': 'complete', 'checkpoint': checkpoint, 'build': {}, 'source_gguf': {},
            'hf_config': config, 'config_only': config, 'adapter_gguf': N.bound(adapter),
            'gguf': N.gguf_metadata(adapter), 'converter_sha256': N.CONVERTER_SHA, 'returncode': 0,
            'argv': [str(N.CONVERTER_PYTHON), '-B', str(N.CONVERTER), '--base', str(config_dir),
                     '--outfile', str(adapter), '--outtype', 'f32', str(root / 'weights')],
            'log': N.bound(log), 'guard_receipt': str(self.path),
            'ownership': self.receipt['ownership'], 'guard_argv': self.receipt['argv']}
        self.guard['gpu'] = {
            'schema': 'trid-gpu-guard-v1', 'expected_uuid': runtime['gpu_uuid'],
            'memory_limit_bytes': 6 * N.GIB, 'identity': copy.deepcopy(runtime['gpu']),
            'used_peak_bytes': 100}
        sample = self.guard['samples'][0]
        self.guard['samples'] = []
        for index, phase in enumerate(('admission', 'pre_exec', 'runtime')):
            row = copy.deepcopy(sample)
            row.update(gpu_phase=phase, elapsed_seconds=index, members=[], gpu_process_observations=[],
                       gpu={**runtime['gpu'], 'used_bytes': 100, 'compute_processes': []})
            self.guard['samples'].append(row)
        for mode in ('probe', 'case'):
            job['mode'] = mode
            json_file(self.path, self.guard)
            N.validate_conversion(job, conversion)
            for mutation in ('missing-gpu', 'changed-runtime-identity', 'missing-ownership'):
                altered = copy.deepcopy(self.guard)
                if mutation == 'missing-gpu':
                    del altered['gpu']
                elif mutation == 'changed-runtime-identity':
                    altered['gpu']['identity']['driver_version'] = '999.1'
                    for row in altered['samples']:
                        row['gpu']['driver_version'] = '999.1'
                else:
                    del altered['samples'][2]['gpu_process_observations']
                json_file(self.path, altered)
                with self.subTest(mode=mode, mutation=mutation), self.assertRaises((RuntimeError, ValueError)):
                    N.validate_conversion(job, conversion)
            json_file(self.path, self.guard)
            stale_source = {**job, 'guard_source': {**job['guard_source'], 'sha256': '0' * 64}}
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'hash changed'):
                N.validate_conversion(stale_source, conversion)


if __name__ == '__main__':
    unittest.main()
