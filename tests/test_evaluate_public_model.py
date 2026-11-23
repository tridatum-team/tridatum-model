"""Deterministic admission/transport tests, not model quality or Linux execution proof."""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_public_model as runner


def row(identifier='one', text='질문'):
    return {'id': identifier, 'messages': [{'role': 'user', 'content': text}]}


class Tokenizer:
    pad_token_id = 0

    def get_chat_template(self):
        return 'official-instruct-template'

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return 'native:' + messages[-1]['content'] + '<|assistant|>'

    def __call__(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False}
        return {'input_ids': list(range(1, len(text) + 1))}


def relative_pin(root, path):
    return dict(runner.pin(path), path=str(Path(path).relative_to(root)))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return runner.pin(path)['sha256']


def make_suite(root, rows=None):
    root.mkdir()
    rows = rows if rows is not None else [row()]
    write_json(root / 'inputs.json', rows)
    # Private files intentionally absent: the worker must not consume them.
    manifest = {'schema': 'trid-public-work-suite-v1', 'profile': 'public_work_v1', 'scope': 'diagnostic',
                'training_ready': False, 'quality_accepted': False, 'expert_reviewed': False,
                'historically_unseen': False, 'case_count': len(rows),
                'inputs': relative_pin(root, root / 'inputs.json'),
                'source': {'path': 'source.private.json', 'bytes': 42, 'sha256': 'a' * 64},
                'references': {'path': 'references.private.json', 'bytes': 42, 'sha256': 'b' * 64}}
    path = root / 'manifest.json'
    return path, write_json(path, manifest)


def make_model(root, *, dtype='BF16', config_change=None):
    root.mkdir()
    config = {'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM'],
              'torch_dtype': 'bfloat16', 'max_position_embeddings': 4096}
    config.update(config_change or {})
    for name, value in [('config.json', config), ('tokenizer.json', {}),
                        ('tokenizer_config.json', {'chat_template': 'native'}), ('generation_config.json', {})]:
        write_json(root / name, value)
    header = json.dumps({'weight': {'dtype': dtype, 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0\0')
    manifest = {'repo_id': 'Qwen/Qwen3-8B', 'revision': 'a' * 40,
                'files': [relative_pin(root, path) for path in sorted(root.iterdir())]}
    path = root.parent / 'acquisition.json'
    return path, write_json(path, manifest)


def make_adapter(root, identity):
    training_root = root / 'training'
    adapter = training_root / 'adapter'
    adapter.mkdir(parents=True)
    lora = {'rank': 8, 'alpha': 16, 'dropout': 0.0, 'targets': ['q_proj']}
    training = {'base_model_id': identity['repo_id'], 'base_revision': identity['revision'],
                'family': 'qwen3', 'status': 'complete', 'steps': 615, 'lora': lora,
                'base_weights': {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'}}
    write_json(training_root / 'training.json', training)
    write_json(adapter / 'adapter_config.json', {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
        'base_model_name_or_path': '/original/sealed/model', 'bias': 'none', 'r': 8,
        'lora_alpha': 16, 'lora_dropout': 0.0, 'target_modules': ['q_proj']})
    (adapter / 'adapter_model.safetensors').write_bytes(b'opaque verified adapter bytes')
    provenance = {'schema': 'trid-public-evidence-adapter-v2', 'fresh': True, 'merged': False,
                  'observed_steps': 615, 'training': relative_pin(root, training_root / 'training.json'),
                  'adapter': {'step': 615,
                              'config': relative_pin(root, adapter / 'adapter_config.json'),
                              'weights': relative_pin(root, adapter / 'adapter_model.safetensors')}}
    path = training_root / 'provenance.json'
    return adapter, path, write_json(path, provenance)


class AdmissionTests(unittest.TestCase):
    def test_public_source_and_references_are_never_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = make_suite(Path(directory) / 'suite')
            rows, identity = runner.public_suite(path, digest)
            self.assertEqual(rows, [row()])
            self.assertEqual(identity['source']['sha256'], 'a' * 64)
            self.assertFalse((path.parent / 'source.private.json').exists())
            self.assertTrue(Path(identity['source']['path']).is_absolute())

    def test_private_aliases_reject_before_input_hash_or_content_open(self):
        for private_key in ('source', 'references'):
            for alias in ('input_path', 'private_path', 'hardlink'):
                with self.subTest(private_key=private_key, alias=alias), tempfile.TemporaryDirectory() as directory:
                    path, _ = make_suite(Path(directory) / 'suite')
                    manifest = runner.load_json(path)
                    if alias == 'input_path':
                        manifest['inputs'] = dict(manifest[private_key])
                    elif alias == 'private_path':
                        manifest[private_key]['path'] = './inputs.json'
                    else:
                        (path.parent / manifest[private_key]['path']).hardlink_to(path.parent / 'inputs.json')
                    digest = write_json(path, manifest)
                    with patch.object(runner, 'check_pin') as hash_input, \
                            patch.object(runner, 'load_json', wraps=runner.load_json) as load, \
                            self.assertRaises(ValueError):
                        runner.public_suite(path, digest)
                    hash_input.assert_not_called()
                    load.assert_called_once_with(path)

    def test_private_fields_duplicates_and_history_rejected_before_tokenizer(self):
        cases = [[dict(row(), target='private')], [row(), row()],
                 [dict(row(), messages=[{'role': 'assistant', 'content': 'bad'}])],
                 [dict(row(), messages=[{'role': 'user', 'content': 'x', 'rubric': []}])]]
        for rows in cases:
            tokenizer = Mock()
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                runner.preflight_prompts(tokenizer, rows, max_new_tokens=8, context_limit=128)
            tokenizer.apply_chat_template.assert_not_called()

    def test_instruct_native_template_has_no_forced_thinking_suffix(self):
        prepared = runner.preflight_prompts(Tokenizer(), [row()], context_limit=128, max_new_tokens=8)
        self.assertTrue(prepared[0]['rendered_prompt'].endswith('<|assistant|>'))
        self.assertEqual(prepared[0]['prompt_sha256'], runner.object_sha(row()['messages']))
        self.assertEqual(prepared[0]['input_tokens'], len(prepared[0]['prompt_token_ids']))

    def test_full_corpus_overflow_is_not_truncated(self):
        with self.assertRaisesRegex(ValueError, 'Full preflight'):
            runner.preflight_prompts(Tokenizer(), [row(), row('two', 'x' * 200)],
                                     context_limit=128, max_new_tokens=8)

    def test_input_hash_and_external_manifest_anchor_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = make_suite(Path(directory) / 'suite')
            with self.assertRaises(ValueError):
                runner.public_suite(path, '0' * 64)
            (path.parent / 'inputs.json').write_text('[]')
            with self.assertRaises(ValueError):
                runner.public_suite(path, digest)

    def test_architecture_quantization_and_dtype_reject_before_weight_header(self):
        changes = [{'model_type': 'qwen3_5'}, {'architectures': ['Other']}, {'torch_dtype': 'float32'},
                   {'quantization_config': {}}, {'auto_map': {}}, {'compression_config': {}}]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'model'
                manifest, digest = make_model(root, config_change=change)
                with patch.object(runner, 'bf16_header') as read_weights, self.assertRaises(ValueError):
                    runner.model_identity(root, manifest, digest)
                read_weights.assert_not_called()

    def test_authenticated_inventory_and_bf16_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'model'
            manifest, digest = make_model(root)
            identity, _ = runner.model_identity(root, manifest, digest)
            self.assertEqual(identity['model_identity']['repo_id'], 'Qwen/Qwen3-8B')
            (root / 'unlisted.py').write_text('not allowed')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                runner.model_identity(root, manifest, digest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'model'
            manifest, digest = make_model(root, dtype='F32')
            with self.assertRaisesRegex(ValueError, 'stored BF16'):
                runner.model_identity(root, manifest, digest)

    def test_verified_weight_cache_advice_does_not_rehash_or_modify_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'model'
            manifest, digest = make_model(root)
            identity, _ = runner.model_identity(root, manifest, digest)
            weight = next(item for item in identity['model_files'] if item['path'].endswith('.safetensors'))
            def advise(fd, offset, length, advice):
                self.assertEqual((offset, length, advice), (0, 0, 4))
                self.assertEqual(runner.os.fstat(fd).st_size, weight['bytes'])
                self.assertEqual(runner.os.lseek(fd, 0, runner.os.SEEK_CUR), 0)
            with patch.object(runner.sys, 'platform', 'linux'), \
                    patch.object(runner.os, 'POSIX_FADV_DONTNEED', 4, create=True), \
                    patch.object(runner.os, 'posix_fadvise', side_effect=advise, create=True) as advice, \
                    patch.object(runner, 'pin', side_effect=AssertionError('must not refill cache')):
                result = runner.discard_verified_weight_cache(identity)
            advice.assert_called_once()
            self.assertEqual(result['requests'], 1)
            self.assertEqual(result['bytes_requested'], weight['bytes'])
            self.assertFalse(result['reclaim_guaranteed'])
            self.assertEqual(result['files'], [{'path': weight['path'], 'bytes_requested': weight['bytes']}])
            self.assertEqual(runner.pin(weight['path']), weight)

    def test_required_identity_cache_advice_has_no_unsupported_or_error_fallback(self):
        with patch.object(runner.sys, 'platform', 'linux'), \
                patch.object(runner.os, 'posix_fadvise', None, create=True), \
                self.assertRaisesRegex(ValueError, 'requires POSIX_FADV_DONTNEED'):
            runner.discard_verified_weight_cache({'model_files': []})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'model'
            manifest, digest = make_model(root)
            identity, _ = runner.model_identity(root, manifest, digest)
            failure = OSError('cache advice denied')
            with patch.object(runner.sys, 'platform', 'linux'), \
                    patch.object(runner.os, 'POSIX_FADV_DONTNEED', 4, create=True), \
                    patch.object(runner.os, 'posix_fadvise', side_effect=failure, create=True), \
                    self.assertRaises(OSError) as raised:
                runner.discard_verified_weight_cache(identity)
            self.assertIs(raised.exception, failure)

    def test_model_adapter_revision_and_exact_byte_binding_without_rewriting(self):
        identity = {'repo_id': 'Qwen/Qwen3-8B', 'revision': 'a' * 40}
        with tempfile.TemporaryDirectory() as directory:
            adapter, provenance, digest = make_adapter(Path(directory), identity)
            before = (adapter / 'adapter_config.json').read_bytes()
            result = runner.adapter_identity(adapter, provenance, digest, identity)
            self.assertEqual(result['steps'], 615)
            self.assertEqual((adapter / 'adapter_config.json').read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'Training provenance'):
                runner.adapter_identity(adapter, provenance, digest, dict(identity, revision='b' * 40))
            (adapter / 'adapter_model.safetensors').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                runner.adapter_identity(adapter, provenance, digest, identity)

    def test_nonadjacent_and_merged_adapter_provenance_rejected(self):
        identity = {'repo_id': 'Qwen/Qwen3-8B', 'revision': 'a' * 40}
        with tempfile.TemporaryDirectory() as directory:
            adapter, provenance, _ = make_adapter(Path(directory), identity)
            value = runner.load_json(provenance)
            value['merged'] = True
            digest = write_json(provenance, value)
            with self.assertRaises(ValueError):
                runner.adapter_identity(adapter, provenance, digest, identity)

    def test_runtime_missing_or_cuda_dependency_never_falls_back(self):
        with patch.object(runner.importlib.metadata, 'version', side_effect=runner.PINS.__getitem__):
            self.assertEqual(runner.runtime_packages(), runner.PINS)
        versions = dict(runner.PINS, torch='2.9.1+cu128')
        with patch.object(runner.importlib.metadata, 'version', side_effect=versions.__getitem__), \
                self.assertRaises(ValueError):
            runner.runtime_packages()
        with patch.object(runner.importlib.metadata, 'version', side_effect=ImportError('missing')), \
                self.assertRaises(ImportError):
            runner.runtime_packages()

    def test_cpu_parameter_inventory_rejects_fp32_base_cuda_or_merged_adapter(self):
        torch = SimpleNamespace(bfloat16='bf16', float32='fp32')
        def parameter(dtype='bf16', device='cpu'):
            return SimpleNamespace(dtype=dtype, device=SimpleNamespace(type=device), requires_grad=False,
                                   numel=lambda: 2, element_size=lambda: 2)
        model = SimpleNamespace(named_parameters=lambda: [('layer.weight', parameter())], modules=lambda: [])
        self.assertEqual(runner.parameter_inventory(model, torch, False)['base']['bf16']['elements'], 2)
        for value in [parameter('fp32'), parameter(device='cuda')]:
            model.named_parameters = lambda: [('layer.weight', value)]
            with self.assertRaises(ValueError):
                runner.parameter_inventory(model, torch, False)
        model.named_parameters = lambda: [('layer.weight', parameter()),
                                          ('layer.lora_A.weight', parameter('fp32'))]
        self.assertTrue(runner.parameter_inventory(model, torch, True)['adapter'])
        model.modules = lambda: [SimpleNamespace(merged=True)]
        with self.assertRaisesRegex(ValueError, 'Merged'):
            runner.parameter_inventory(model, torch, True)

    def test_unenforced_cpu_quota_network_or_host_reserve_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'owned.service'
            quota, network = ['400000 100000'], ['header\nheader\nlo: counters\n']
            host = {'memory_bytes': {'MemAvailable': 45 * runner.guard.GIB},
                    'memory_psi': {'full': {'total': 0}}}
            def read_text(path, *args, **kwargs):
                return quota[0] if path.name == 'cpu.max' else network[0]
            with patch.object(runner.sys, 'platform', 'linux'), patch.object(runner.os, 'getuid', return_value=1000), \
                    patch.object(runner.guard, 'owned_cgroup', return_value=root), \
                    patch.object(Path, 'stat', return_value=SimpleNamespace(st_uid=0)), \
                    patch.object(Path, 'read_text', read_text), \
                    patch.object(runner.guard, 'cgroup_metrics', return_value={}), \
                    patch.object(runner.guard, 'host_metrics', return_value=host):
                self.assertEqual(runner.enforced_envelope('public8b_cpu', admission=True)['network_interfaces'], ['lo'])
                quota[0] = 'max 100000'
                with self.assertRaisesRegex(ValueError, 'CPUQuota'):
                    runner.enforced_envelope('public8b_cpu')
                quota[0] = '400000 100000'
                network[0] += 'eth0: counters\n'
                with self.assertRaisesRegex(ValueError, 'PrivateNetwork'):
                    runner.enforced_envelope('public8b_cpu')
                network[0] = 'header\nheader\nlo: counters\n'
                host['memory_bytes']['MemAvailable'] = 7 * runner.guard.GIB
                with self.assertRaisesRegex(ValueError, 'reserve'):
                    runner.enforced_envelope('public8b_cpu')

    def test_worker_full_preflight_precedes_loader_import_or_call(self):
        rows = [row(), row('two', 'x' * 200)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = {'output': str(output), 'seconds': 1800, 'profile': 'public8b_cpu',
                   'suite_manifest': 'suite', 'manifest_sha256': 'a' * 64,
                   'model_root': 'model', 'model_manifest': 'acquisition', 'model_manifest_sha256': 'b' * 64,
                   'context_limit': 128, 'max_new_tokens': 8, 'threads': 4, 'adapter': None}
            path = output / 'job.json'
            digest = write_json(path, job)
            torch = SimpleNamespace(__version__=runner.PINS['torch'], version=SimpleNamespace(cuda=None),
                                    cuda=SimpleNamespace(is_available=lambda: False),
                                    set_num_threads=Mock(), set_num_interop_threads=Mock())
            loader = Mock()
            modules = {'torch': torch, 'transformers': SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer()))),
                'train_open_weight_lora': SimpleNamespace(load_qwen3_pread=loader)}
            identity = {'model_identity': {'repo_id': 'Qwen/Qwen3-8B', 'revision': 'a' * 40},
                        'tokenizer_identity': {'files': [], 'template_sha256': None}}
            with patch.dict(sys.modules, modules), patch.dict('os.environ', {}, clear=False), \
                    patch.object(runner, 'verify_worker_owner'), patch.object(runner, 'enforced_envelope'), \
                    patch.object(runner, 'runtime_packages'), \
                    patch.object(runner, 'discard_verified_weight_cache', return_value={}), \
                    patch.object(runner, 'public_suite', return_value=(rows, {})), \
                    patch.object(runner, 'model_identity', return_value=(identity, {'max_position_embeddings': 4096})), \
                    self.assertRaisesRegex(ValueError, 'Full preflight'):
                runner.worker(path, digest)
            loader.assert_not_called()
            self.assertFalse((output / 'responses.jsonl').exists())


class JournalTests(unittest.TestCase):
    def test_requested_policy_is_explicit_and_shared(self):
        policy = runner.requested_policy(512, 4096)
        self.assertEqual(policy, runner.requested_policy(512, 4096))
        self.assertIs(policy['do_sample'], False)
        self.assertEqual(policy['seed'], 0)
        self.assertEqual(policy['mode'], 'evaluation')
        for new, limit in [(0, 4096), (4096, 4096), (True, 4096), (1, 0)]:
            with self.assertRaises(ValueError):
                runner.requested_policy(new, limit)

    def test_fsync_and_exclusive_journal_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'responses.jsonl'
            with patch.object(runner.os, 'fsync') as fsync:
                runner.save_new(path, {'id': 'one'})
                fsync.assert_called_once()
            with self.assertRaises(FileExistsError):
                runner.save_new(path, {'id': 'two'})
            self.assertEqual(runner.load_json(path), {'id': 'one'})

    def test_guard_failure_preserves_partial_journal_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save_new(root / 'runtime.json', {'status': 'loaded'})
            runner.save_new(root / 'responses.jsonl', {'id': 'one', 'response': 'native text'})
            before = (root / 'responses.jsonl').read_bytes()
            result = runner.finalize(root, ['one', 'two'], {'child_exit_code': -15, 'first_stop_reason': 'deadline'})
            self.assertEqual(result['observed_ids'], ['one'])
            self.assertEqual(result['missing_ids'], ['two'])
            self.assertEqual(result['guard_failure'], 'deadline')
            self.assertFalse(result['complete'])
            self.assertEqual((root / 'responses.jsonl').read_bytes(), before)

    def test_clean_worker_exit_with_exception_row_is_incomplete_without_losing_id(self):
        for error, finish_reason in [({'type': 'RuntimeError', 'message': 'native failure'}, 'exception'),
                                     (None, 'exception'), ({'type': 'RuntimeError'}, 'stop')]:
            with self.subTest(error=error, finish_reason=finish_reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner.save_new(root / 'runtime.json', {'status': 'finished'})
                runner.save_new(root / 'responses.jsonl',
                                {'id': 'one', 'response': '', 'error': error, 'finish_reason': finish_reason})
                before = (root / 'responses.jsonl').read_bytes()
                result = runner.finalize(root, ['one'], {
                    'startup_released': True, 'stop_reason': 'child_exit', 'first_stop_reason': None,
                    'child_exit_code': 0, 'reaped': True, 'cleanup_state': 'completed'})
                self.assertFalse(result['complete'])
                self.assertEqual(result['exception_rows'], 1)
                self.assertEqual(result['observed_ids'], ['one'])
                self.assertEqual(result['missing_ids'], [])
                self.assertEqual(result['worker_exit_code'], 0)
                self.assertIsNone(result['guard_failure'])
                self.assertEqual((root / 'responses.jsonl').read_bytes(), before)

    def test_unterminated_partial_row_not_repaired_or_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save_new(root / 'runtime.json', {'status': 'loaded'})
            (root / 'responses.jsonl').write_text('{"id":"one"}\n{"id":')
            result = runner.finalize(root, ['one', 'two'], {'child_exit_code': -9, 'first_stop_reason': 'deadline'})
            self.assertEqual(result['missing_ids'], ['two'])
            self.assertIsNotNone(result['journal_error'])
            self.assertTrue((root / 'responses.jsonl').read_text().endswith('{"id":'))

    def test_interrupted_korean_utf8_tail_keeps_completed_rows_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save_new(root / 'runtime.json', {'status': 'loaded'})
            completed = json.dumps({'id': 'one', 'response': '완료'}, ensure_ascii=False).encode() + b'\n'
            interrupted = b'{"id":"two","response":"' + '한'.encode()[:2]
            original = completed + interrupted
            (root / 'responses.jsonl').write_bytes(original)
            result = runner.finalize(root, ['one', 'two'],
                                     {'child_exit_code': -9, 'first_stop_reason': 'deadline'})
            self.assertEqual(result['observed_ids'], ['one'])
            self.assertEqual(result['missing_ids'], ['two'])
            self.assertIsNotNone(result['journal_error'])
            self.assertEqual(result['guard_failure'], 'deadline')
            self.assertFalse(result['complete'])
            self.assertEqual((root / 'responses.jsonl').read_bytes(), original)
            self.assertEqual(runner.load_json(root / 'completion.json'), result)

    def test_existing_output_rejected_without_admission_or_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, digest = make_suite(root / 'suite')
            model_manifest, model_digest = make_model(root / 'model')
            output = root / 'output'
            output.mkdir()
            args = SimpleNamespace(threads=4, seconds=1800, profile='public8b_cpu',
                max_new_tokens=512, context_limit=4096,
                adapter=None, adapter_provenance=None, adapter_provenance_sha256=None,
                suite_manifest=suite, manifest_sha256=digest, model_manifest=model_manifest,
                model_manifest_sha256=model_digest, model_root=root / 'model', output=output)
            with patch.object(runner, 'enforced_envelope') as admission, self.assertRaises(FileExistsError):
                runner.supervise(args)
            admission.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])
            args.output = suite.parent / 'nested-output'
            with self.assertRaisesRegex(ValueError, 'separate'):
                runner.supervise(args)
            self.assertFalse(args.output.exists())
            args.output = root / 'failed-output'
            with patch.object(runner, 'runtime_packages', side_effect=ValueError('runtime pins differ')), \
                    patch.object(runner.guard, 'Guard') as launch:
                self.assertEqual(runner.supervise(args), 1)
            launch.assert_not_called()
            completion = runner.load_json(args.output / 'completion.json')
            self.assertEqual(completion['missing_ids'], ['one'])
            self.assertEqual(completion['observed_ids'], [])
            self.assertFalse(completion['complete'])
            self.assertIn('runtime pins differ', completion['guard_failure'])
            self.assertEqual((args.output / 'responses.jsonl').read_bytes(), b'')

    def test_generation_exception_is_durable_once_without_retry(self):
        class Tensor:
            pass
        torch = SimpleNamespace(tensor=lambda *a, **k: Tensor(), long='long', manual_seed=Mock(),
                                inference_mode=nullcontext, ones_like=lambda value: value)
        @contextmanager
        def observer(*args):
            yield {'effective_config_sha256': 'a' * 64, 'generation_mode': 'greedy_search'}
        model = SimpleNamespace(generate=Mock(side_effect=RuntimeError('native failure')))
        helpers = SimpleNamespace(observe_policy=observer, decode_generation=Mock())
        prepared = runner.preflight_prompts(Tokenizer(), [row()], context_limit=128, max_new_tokens=8)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'torch': torch, 'evaluate_evidence_conditioning': helpers}), \
                patch.object(runner, 'generation_controls', return_value=(
                    SimpleNamespace(to_dict=lambda: {'eos_token_id': 2}), {'use_model_defaults': False})):
            path = Path(directory) / 'responses.jsonl'
            runner.stream_responses(model, Tokenizer(), [row()], prepared, path,
                                    max_new_tokens=8, context_limit=128)
            result = runner.load_json(path)
            self.assertEqual(result['finish_reason'], 'exception')
            self.assertEqual(result['error']['message'], 'native failure')
            self.assertEqual(result['output_token_ids'], [])
            self.assertGreaterEqual(result['wall_seconds'], 0)
            model.generate.assert_called_once()
            helpers.decode_generation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
