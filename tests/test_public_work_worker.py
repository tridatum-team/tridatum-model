"""Offline contract fixtures only: no cloud, native inference or quality claims."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import grounded_hf_worker as core
import public_work_experiment as P
import public_work_worker as W
import evaluate_public_work as E


class TokenizerFixture:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return 'official fixture: ' + messages[-1]['content']

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False)
        return {'input_ids': list(text.encode())}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return P.pin(path)


def item(index):
    return dict(ids=[10, 11, 12, 1], labels=[-100, -100, 12, 1],
        identity=dict(id=f'row-{index}', source_sha256='a' * 64, source_line=index + 1))


def plan_fixture(rows=530):
    return dict(schema=P.SCHEMA, recipe=P.recipe_for_rows(rows),
        model=dict(id=P.MODEL_ID, revision=P.MODEL_REVISION),
        data=dict(train=dict(sha256='a' * 64), manifest=dict(path='public/training-manifest.json')))


class PublicWorkWorkerTests(unittest.TestCase):
    def observer(self, items, *, rows=530, **changes):
        observer, measured = W.training_observer(plan_fixture(rows), None, 1)
        values = dict(args=SimpleNamespace(checkpoint_every=None, valid_every=0, init_adapter=None, merge=False),
                      train=items, valid=[], dropped=0, valid_dropped=0)
        values.update(changes)
        observer('admission', **values)
        return observer, measured

    def test_exact_recipe_and_full_training_argv(self):
        self.assertEqual(P.recipe_for_rows(530), dict(rows=530, epochs=2, steps=530, batch_size=2, micro_batch_size=1,
            lr=5e-5, rank=16, alpha=32, dropout=0.0, context_tokens=4096, answer_tokens=768,
            seed=14, checkpoint_every=0, valid_every=0))
        argv = core.train_argv(plan_fixture(), Path('/owned'))
        self.assertEqual(argv, ['/owned/code/scripts/train_open_weight_lora.py', '--model', '/owned/model',
            '--model-id', P.MODEL_ID, '--model-revision', P.MODEL_REVISION, '--family', 'qwen3',
            '--manifest', '/owned/code/public/training-manifest.json', '--out', '/owned/output/training',
            '--steps', '530', '--batch-size', '2', '--micro-batch-size', '1', '--lr', '5e-05',
            '--rank', '16', '--alpha', '32', '--dropout', str(plan_fixture()['recipe']['dropout']), '--max-tokens', '4096',
            '--max-answer-tokens', '768', '--seed', '14', '--device', 'cuda', '--valid-every', '0'])
        self.assertNotIn('--checkpoint-every', argv)
        self.assertNotIn('--init-adapter', argv)
        self.assertNotIn('--merge', argv)

    def test_observer_requires_exactly_two_of_each_physical_row(self):
        for rows in (1, 7, 39, 40, 41, 530, 531):
            with self.subTest(rows=rows):
                recipe = plan_fixture(rows)['recipe']
                items = [item(i) for _ in range(recipe['epochs']) for i in range(rows)]
                observer, measured = self.observer(items, rows=rows)
                coverage = [row['identity'] for row in reversed(items)]
                observer('completion', coverage=coverage, steps=recipe['steps'])
                self.assertTrue(measured['coverage_exactly_twice'])
                self.assertEqual(measured['masking']['rows'], rows * recipe['epochs'])
                self.assertTrue(measured['masking']['eos_supervised'])
                self.assertEqual(observer('report'), dict(checkpoint_every=0, valid_every=0))
                changed_line = copy.deepcopy(coverage)
                changed_line[0]['source_line'] = rows + 1
                for changed, steps in ((coverage[:-1], recipe['steps']),
                        (changed_line, recipe['steps']), (coverage, recipe['steps'] - 1)):
                    with self.subTest(steps=steps), self.assertRaises(ValueError):
                        observer('completion', coverage=changed, steps=steps)
                if rows > 1:
                    with self.assertRaises(ValueError):
                        observer('completion', coverage=coverage[:-1] + [coverage[0]], steps=recipe['steps'])

    def test_dynamic_recipe_training_argv_and_tampering(self):
        plan = plan_fixture(41)
        argv = core.train_argv(plan, Path('/owned'))
        self.assertEqual(argv[argv.index('--steps') + 1], '41')
        self.assertEqual(argv[argv.index('--batch-size') + 1], '2')
        self.assertEqual(argv[argv.index('--lr') + 1], '5e-05')
        for field, value in (('rows', 0), ('rows', -1), ('rows', True), ('rows', 41.0),
                ('steps', 40), ('steps', 530), ('epochs', 1), ('batch_size', 1),
                ('lr', 1e-4), ('rank', 8), ('alpha', 16), ('seed', 0), ('extra', 1)):
            changed = copy.deepcopy(plan)
            changed['recipe'][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    W.training_observer(changed, None, 1)
                with self.assertRaises(ValueError):
                    W.validate_training(Path('/unused'), changed)
        del plan['recipe']['seed']
        with self.assertRaises(ValueError):
            W.training_observer(plan, None, 1)
        with self.assertRaises(ValueError):
            W.validate_training(Path('/unused'), plan)

    def test_observer_rejects_mask_eos_drop_repeat_and_nonsealed_options(self):
        for failure in ('mask', 'eos', 'third_exposure', 'wrong_source', 'wrong_line', 'dropped', 'valid',
                        'checkpoint', 'validation', 'warm_start', 'merge'):
            items = [item(i) for _ in range(2) for i in range(7)]
            changes = {}
            if failure == 'mask':
                items[-1]['labels'][0] = 10
            elif failure == 'eos':
                items[-1]['ids'][-1] = items[-1]['labels'][-1] = 2
            elif failure == 'third_exposure':
                items[-1]['identity'] = copy.deepcopy(items[0]['identity'])
            elif failure == 'wrong_source':
                items[-1]['identity']['source_sha256'] = 'b' * 64
            elif failure == 'wrong_line':
                for row in items:
                    row['identity']['source_line'] += 1
            elif failure == 'dropped':
                changes['dropped'] = 1
            elif failure == 'valid':
                changes['valid'] = [item(0)]
            else:
                args = dict(checkpoint_every=None, valid_every=0, init_adapter=None, merge=False)
                args[{'checkpoint': 'checkpoint_every', 'validation': 'valid_every',
                      'warm_start': 'init_adapter', 'merge': 'merge'}[failure]] = 1
                changes['args'] = SimpleNamespace(**args)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.observer(items, rows=7, **changes)
        for field, value in (('steps', 265), ('rank', 8), ('epochs', 1), ('seed', 0)):
            plan = plan_fixture()
            plan['recipe'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                W.training_observer(plan, None, 1)

    def test_cpu_policy_reused_without_cpu_authentication(self):
        rows = [dict(id='first', messages=[dict(role='user', content='hello')]),
                dict(id='last', messages=[dict(role='user', content='world')])]
        prepared = W.helpers().preflight_prompts(TokenizerFixture(), rows, context_limit=4096, max_new_tokens=512)
        self.assertEqual([p['id'] for p in prepared], ['first', 'last'])
        self.assertEqual(W.helpers().requested_policy(512, 4096), P.GENERATION)
        rows[-1]['messages'][-1]['content'] = 'x' * 4096
        with self.assertRaisesRegex(ValueError, 'Full preflight'):
            W.helpers().preflight_prompts(TokenizerFixture(), rows, context_limit=4096, max_new_tokens=512)
        self.assertNotEqual(W.RUNTIME_SCHEMA, 'public_work_hf_runtime_v1')
        self.assertNotEqual(W.COMPLETION_SCHEMA, 'public_work_hf_completion_v1')
        self.assertNotEqual(W.RECEIPT_SCHEMA, 'public_work_hf_v1')

    def test_journal_is_durable_exclusive_and_export_excludes_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'responses.jsonl'
            with path.open('x') as stream:
                W.append_row(stream, dict(id='attempt', error=dict(type='RuntimeError'), finish_reason='exception'))
                self.assertEqual(json.loads(path.read_text())['id'], 'attempt')
            with self.assertRaises(FileExistsError):
                W.save_new(path, {})
        names = P.output_members()
        self.assertIn('training/adapter/adapter_model.safetensors', names)
        self.assertIn('baseline/holdout/responses.jsonl', names)
        self.assertIn('adapter/diagnostic/completion.json', names)
        self.assertFalse(any(name.startswith('model/') or 'model-00001' in name for name in names))
        self.assertFalse(any('capacity' in name for name in names))

    def test_explicit_dispatch_preserves_evidence_and_rejects_unknown(self):
        self.assertEqual(core.public_module(plan_fixture()).__name__, 'public_work_worker')
        self.assertEqual(core.public_module(dict(schema=P.evidence.SCHEMA)).__name__,
                         'public_evidence_worker')
        with self.assertRaises(RuntimeError):
            core.require_known_profile(dict(schema='trid-public-work-gpu-plan-v99'))
        with self.assertRaises(ValueError):
            W.child_stage(None, plan_fixture(), Path('/unused'), 'capacity')

    def test_supervised_order_is_acquire_baseline_train_adapter_without_capacity(self):
        for fail_baseline in (False, True):
            with self.subTest(fail_baseline=fail_baseline), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                (base / 'output').mkdir()
                (base / 'model').mkdir()
                (base / 'model/tokenizer.json').write_text('{}')
                plan = plan_fixture()
                plan['identity'] = dict(run_id='fixture-run')
                plan['model']['files'] = [P.pin(base / 'model/tokenizer.json', base / 'model')]
                source = base / 'code/public.json'
                put(source, {})
                data_pin = P.pin(source, base / 'code')
                plan['data']['admission'] = data_pin
                plan['workload'] = {name: data_pin for name in ('source_manifest', 'evaluation_inputs',
                    'holdout_inputs', 'holdout_manifest', 'diagnostic_inputs', 'diagnostic_manifest')}
                put(base / 'plan.json', plan)
                put(base / 'output/runtime.json', {})
                phases = []
                def supervise(plan, base, phase, *args):
                    phases.append(phase)
                    return dict(phase=phase, cleanup_verified=True)
                def accept(root, plan, arm):
                    if fail_baseline and arm == 'baseline':
                        raise ValueError('fixture baseline failure')
                transport = SimpleNamespace(download=lambda *args: phases.append('acquire'),
                    validate_model=lambda *args: None, digest=core.digest, supervise_phase=supervise)
                report = dict(stages=[])
                with patch.object(W.os, 'chown'), patch.object(W.shared, 'freeze'), \
                        patch.object(W, 'validate_outputs', side_effect=accept), \
                        patch.object(W, 'validate_training'):
                    if fail_baseline:
                        with self.assertRaisesRegex(ValueError, 'fixture baseline failure'):
                            W.run_experiment(transport, plan, base, report, {}, SimpleNamespace(pw_uid=1, pw_gid=1),
                                             100, 90, {}, set(), [])
                    else:
                        W.run_experiment(transport, plan, base, report, {}, SimpleNamespace(pw_uid=1, pw_gid=1),
                                         100, 90, {}, set(), [])
                self.assertEqual(phases, ['acquire', 'baseline'] if fail_baseline
                                 else ['acquire', 'baseline', 'train', 'adapter'])
                self.assertFalse((base / 'output/model').exists())
                self.assertFalse((base / 'output/capacity').exists())


class PublicWorkGpuEvaluatorTests(unittest.TestCase):
    """Synthetic source-bound GPU receipt fixtures, not an actual CUDA run."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'export'
        self.folder = self.root / 'baseline/diagnostic'
        self.folder.mkdir(parents=True)
        self.rows = [dict(id=f'case-{i}', messages=[dict(role='user', content='fixture question')]) for i in range(24)]
        token = self.root / 'tokenizer/tokenizer.json'
        config = self.root / 'tokenizer/tokenizer_config.json'
        put(token, {'fixture': True})
        put(config, {'chat_template': 'fixture official template'})
        files = [dict(P.pin(path), path=path.name) for path in (token, config)]
        self.suite_path = self.root / 'suites/diagnostic/manifest.json'
        inputs = self.root / 'suites/diagnostic/inputs.json'
        put(inputs, self.rows)
        self.manifest = dict(schema='trid-public-work-suite-v1', scope='diagnostic',
            source=dict(path='source.private.json', bytes=123, sha256='f' * 64),
            inputs=dict(P.pin(inputs), path='inputs.json'))
        put(self.suite_path, self.manifest)
        # The local original has identical bytes at a different location. Private
        # source is intentionally absent: GPU receipts authenticate metadata only.
        self.local_suite = Path(self.temp.name) / 'local/manifest.json'
        put(self.local_suite, self.manifest)
        source = dict(path='sources.json', bytes=12, sha256='d' * 64)
        self.plan = dict(schema=P.SCHEMA, recipe=P.recipe_for_rows(41), identity=dict(run_id='fixture-run'),
            runtime=dict(packages=P.RUNTIME_PACKAGES, torch_cuda='12.8'),
            model=dict(id=P.MODEL_ID, revision=P.MODEL_REVISION, files=files),
            workload=dict(source_manifest=source, evaluation_inputs=P.pin(inputs),
                diagnostic_manifest=P.pin(self.suite_path), diagnostic_inputs=P.pin(inputs)))
        put(self.root / 'plan.json', self.plan)
        plan_sha = P.pin(self.root / 'plan.json')['sha256']
        put(self.root / 'runtime.json', dict(name='NVIDIA L40S', count=1, bf16=True,
            packages=P.RUNTIME_PACKAGES, cuda='12.8', run_id='fixture-run', plan_sha256=plan_sha))
        put(self.root / 'model-acquisition.json', dict(status='authenticated', repo_id=P.MODEL_ID,
            revision=P.MODEL_REVISION, files=files, plan_sha256=plan_sha))
        self.loaded = dict(device='cuda', dtype='torch.bfloat16', frozen=True, adapter_merged=False,
            cpu_fallback=False, adapter=None, adapter_reload_equal=None, gpu_uuid='GPU-fixture',
            instance_id='i-fixture', local_files_only=True, trust_remote_code=False,
            attention_implementation='sdpa', plan_sha256=plan_sha, source_manifest=source,
            runtime=P.pin(self.root / 'runtime.json', self.root), model_files=files,
            parameter_inventory=dict(base={'torch.bfloat16': dict(tensors=1, elements=2, bytes=4)}, adapter={}))
        self.loaded_path = self.root / 'baseline/loaded-runtime.json'
        put(self.loaded_path, self.loaded)
        self.binding = P.generation_binding(self.plan, 'baseline')
        self.runtime = dict(schema=W.RUNTIME_SCHEMA, suite_scope='diagnostic',
            suite_manifest=W.relative_pin(self.suite_path, self.folder), source=self.manifest['source'],
            inputs=W.relative_pin(inputs, self.folder),
            model_identity=dict(repo_id=P.MODEL_ID, revision=P.MODEL_REVISION),
            model_manifest=W.relative_pin(self.root / 'model-acquisition.json', self.folder),
            tokenizer_identity=dict(files=[W.relative_pin(p, self.folder) for p in (token, config)],
                template_sha256=hashlib.sha256(b'fixture official template').hexdigest()),
            adapter=None, generation_policy=W.helpers().requested_policy(512, 4096), device='cuda',
            cpu_fallback=False, research_only=True, quality_accepted=False, binding=self.binding,
            plan=W.relative_pin(self.root / 'plan.json', self.folder),
            loaded_runtime=W.relative_pin(self.loaded_path, self.folder))
        generation = dict(max_new_tokens=512, do_sample=False, num_beams=1, eos_token_id=9)
        rendered = 'fixture rendered prompt'
        self.records = [dict(id=row['id'], generation_receipt=W.RECEIPT_SCHEMA,
            prompt_sha256=E.object_sha(row['messages']), response='fixture answer', raw_output='fixture answer',
            finish_reason='eos', error=None, prompt_token_ids=[1, 2], input_tokens=2,
            output_token_ids=[3, 9], output_tokens=2, rendered_prompt=rendered,
            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            template_kwargs=dict(enable_thinking=False, add_generation_prompt=True),
            generation_config=generation, generation_policy=dict(effective_config_sha256=E.object_sha(generation),
                generation_mode='greedy_search'), context_limit=4096, context_truncated=False,
            truncated_output=False, seed=0, mode='evaluation', wall_seconds=0.25, binding=self.binding)
            for row in self.rows]
        self.publish()

    def publish(self, **changes):
        put(self.folder / 'runtime.json', self.runtime)
        (self.folder / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.records))
        ids = [row['id'] for row in self.rows]
        completion = dict(schema=W.COMPLETION_SCHEMA, expected_ids=ids,
            observed_ids=[row['id'] for row in self.records], missing_ids=[], complete=True,
            worker_exit_code=0, guard_failure=None,
            runtime=W.relative_pin(self.folder / 'runtime.json', self.folder),
            responses=W.relative_pin(self.folder / 'responses.jsonl', self.folder))
        completion.update(changes)
        put(self.folder / 'completion.json', completion)

    def load(self):
        return E.load_arm(self.folder, self.rows, self.local_suite, self.manifest)

    def test_gpu_loads_portably_without_private_source_or_cpu_auth(self):
        records, runtime, completion, _ = self.load()
        self.assertEqual(len(records), 24)
        self.assertEqual(runtime['device'], 'cuda')
        self.assertTrue(completion['complete'])
        self.assertFalse((self.suite_path.parent / 'source.private.json').exists())
        self.assertNotIn('guard', runtime)

    def test_cpu_gpu_schema_cross_contamination_rejected(self):
        for target in ('completion', 'runtime', 'receipt'):
            with self.subTest(target=target):
                original_runtime, original_records = copy.deepcopy(self.runtime), copy.deepcopy(self.records)
                if target == 'runtime':
                    self.runtime['schema'] = 'public_work_hf_runtime_v1'
                elif target == 'receipt':
                    self.records[0]['generation_receipt'] = 'public_work_hf_v1'
                self.publish(**({'schema': 'public_work_hf_completion_v1'} if target == 'completion' else {}))
                with self.assertRaises(ValueError):
                    self.load()
                self.runtime, self.records = original_runtime, original_records
                self.publish()

    def test_policy_binding_precision_and_local_suite_tampering_rejected(self):
        for failure in ('seed', 'budget', 'truncation', 'binding', 'dtype', 'suite'):
            with self.subTest(failure=failure):
                runtime, records, loaded = copy.deepcopy(self.runtime), copy.deepcopy(self.records), copy.deepcopy(self.loaded)
                if failure == 'seed':
                    self.runtime['generation_policy']['seed'] = 14
                elif failure == 'budget':
                    self.runtime['generation_policy']['max_new_tokens'] = 768
                elif failure == 'truncation':
                    self.records[0]['context_truncated'] = True
                elif failure == 'binding':
                    self.records[0]['binding'] = dict(self.binding, arm='adapter')
                elif failure == 'dtype':
                    self.loaded['dtype'] = 'torch.float32'
                    put(self.loaded_path, self.loaded)
                    self.runtime['loaded_runtime'] = W.relative_pin(self.loaded_path, self.folder)
                else:
                    put(self.local_suite, dict(self.manifest, scope='development_holdout'))
                self.publish()
                with self.assertRaises(ValueError):
                    self.load()
                self.runtime, self.records, self.loaded = runtime, records, loaded
                put(self.loaded_path, self.loaded)
                put(self.local_suite, self.manifest)
                self.publish()

    def test_missing_exception_and_duplicate_rows_cannot_claim_complete(self):
        for failure in ('missing', 'exception', 'duplicate'):
            with self.subTest(failure=failure):
                original = copy.deepcopy(self.records)
                if failure == 'missing':
                    self.records.pop()
                elif failure == 'exception':
                    self.records[0].update(error=dict(type='RuntimeError'), finish_reason='exception')
                else:
                    self.records[-1] = copy.deepcopy(self.records[0])
                self.publish()
                with self.assertRaises(ValueError):
                    self.load()
                self.records = original
                self.publish()

    def test_gpu_runtime_rejects_tampered_dynamic_recipe(self):
        for field, value in (('rows', 0), ('steps', 530), ('epochs', 1), ('rank', 8), ('extra', 1)):
            changed = copy.deepcopy(self.plan)
            changed['recipe'][field] = value
            put(self.root / 'plan.json', changed)
            runtime = copy.deepcopy(self.runtime)
            runtime['plan'] = W.relative_pin(self.root / 'plan.json', self.folder)
            with self.subTest(field=field), self.assertRaises(ValueError):
                W.validate_gpu_runtime(self.folder, runtime)

    def training_fixture(self, rows=530):
        self.plan['recipe'] = recipe = P.recipe_for_rows(rows)
        steps, exposures = recipe['steps'], rows * recipe['epochs']
        self.plan['data'] = dict(train=dict(sha256='a' * 64), manifest=dict(sha256='b' * 64))
        put(self.root / 'plan.json', self.plan)
        plan_sha = P.pin(self.root / 'plan.json')['sha256']
        native = W.read_json(self.root / 'runtime.json')
        native['plan_sha256'] = plan_sha
        put(self.root / 'runtime.json', native)
        put(self.root / 'admission.json', dict(training_ids=[f'row-{i}' for i in range(rows)]))
        targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        config = dict(r=16, lora_alpha=32, lora_dropout=0, peft_type='LORA', task_type='CAUSAL_LM',
                      bias='none', target_modules=targets)
        put(self.root / 'training/adapter/adapter_config.json', config)
        weights = self.root / 'training/adapter/adapter_model.safetensors'
        weights.write_bytes(b'opaque fixture bytes; never loaded as native weights')
        warmup = max(1, steps // 20)
        def multiplier(step):
            return ((step + 1) / warmup if step < warmup else
                    0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        report = dict(status='complete', steps=steps, cumulative_steps=steps, train_rows=exposures, valid_rows=0,
            dropped_over_max_tokens=0, valid_dropped=0, init_adapter=None, device='cuda', dtype='torch.bfloat16',
            base_weights=dict(representation='hf-safetensors', dtype='torch.bfloat16'),
            manifest_sha256='b' * 64, base_model_id=P.MODEL_ID, base_revision=P.MODEL_REVISION,
            batch_size=2, micro_batch_size=1, lr=5e-5, seed=14, max_tokens=4096, max_answer_tokens=768,
            checkpoint_every=0, valid_every=0, lora=dict(rank=16, alpha=32, dropout=0, targets=targets),
            coverage=[item(i)['identity'] for _ in range(recipe['epochs']) for i in range(rows)],
            history=[dict(step=step, train_loss=1.0, applied_lr=recipe['lr'] * multiplier(step - 1),
                          next_lr=recipe['lr'] * multiplier(step)) for step in range(1, steps + 1)])
        put(self.root / 'training/training.json', report)
        provenance = dict(schema=P.ADAPTER_SCHEMA, run_id='fixture-run', plan_sha256=plan_sha,
            source_manifest=self.plan['workload']['source_manifest'],
            runtime=P.pin(self.root / 'runtime.json', self.root), fresh=True, merged=False,
            training=P.pin(self.root / 'training/training.json', self.root),
            adapter=W.adapter_identity(self.root, self.plan), fresh_zero_b=True, observed_steps=steps,
            coverage_exactly_twice=True, masking=dict(rows=exposures, all_prompt_tokens_masked=True,
                all_answer_tokens_supervised=True, eos_supervised=True, max_context_tokens=4, max_answer_tokens=2),
            parameter_inventory=dict(base={'torch.bfloat16': dict(tensors=1, elements=2, bytes=4)},
                                     adapter={'torch.float32': dict(tensors=1, elements=2, bytes=8)}))
        put(self.root / 'training/provenance.json', provenance)
        return report, provenance

    def test_offline_training_acceptance_checks_actual_coverage_schedule_and_final_adapter(self):
        for rows in (1, 7, 39, 40, 41, 530, 531):
            with self.subTest(rows=rows):
                report, provenance = self.training_fixture(rows)
                steps = self.plan['recipe']['steps']
                self.assertEqual(W.validate_training(self.root, self.plan)['steps'], steps)
                for failure in ('coverage', 'repeat', 'source_line', 'source_hash', 'schedule', 'next_lr',
                        'legacy_warmup', 'loss', 'nan_loss', 'history', 'steps', 'cumulative_steps',
                        'train_rows', 'observed_steps', 'masking_rows', 'checkpoint', 'rank', 'merge', 'freshness'):
                    with self.subTest(failure=failure):
                        changed, observed = copy.deepcopy(report), copy.deepcopy(provenance)
                        if failure == 'coverage':
                            changed['coverage'].pop()
                        elif failure == 'repeat':
                            changed['coverage'].append(copy.deepcopy(changed['coverage'][0]))
                        elif failure == 'source_line':
                            changed['coverage'][-1]['source_line'] = rows + 1
                        elif failure == 'source_hash':
                            changed['coverage'][-1]['source_sha256'] = 'c' * 64
                        elif failure == 'schedule':
                            changed['history'][0]['applied_lr'] = 0.0
                        elif failure == 'next_lr':
                            changed['history'][0]['next_lr'] *= 0.5
                        elif failure == 'legacy_warmup':
                            if max(1, steps // 20) == 26:
                                continue
                            changed['history'][0]['applied_lr'] = 5e-5 / 26
                        elif failure in ('loss', 'nan_loss'):
                            changed['history'][0]['train_loss'] = float('inf' if failure == 'loss' else 'nan')
                        elif failure == 'history':
                            changed['history'].pop()
                        elif failure in ('steps', 'cumulative_steps', 'train_rows'):
                            changed[failure] += 1
                        elif failure == 'observed_steps':
                            observed['observed_steps'] += 1
                        elif failure == 'masking_rows':
                            observed['masking']['rows'] += 1
                        elif failure == 'checkpoint':
                            changed['checkpoint_every'] = 10
                        elif failure == 'rank':
                            changed['lora']['rank'] = 8
                        elif failure == 'merge':
                            observed['merged'] = True
                        else:
                            observed['fresh_zero_b'] = False
                        put(self.root / 'training/training.json', changed)
                        observed['training'] = P.pin(self.root / 'training/training.json', self.root)
                        put(self.root / 'training/provenance.json', observed)
                        with self.assertRaises(ValueError):
                            W.validate_training(self.root, self.plan)


if __name__ == '__main__':
    unittest.main()
