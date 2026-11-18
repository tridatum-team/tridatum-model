"""CPU-only 8B worker boundaries, including tiny real HF generation; no cloud calls."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import eight_hf_worker as W
import eight_hf_experiment as E
import grounded_hf_worker as G


class InputBoundaryTests(unittest.TestCase):
    def write_inputs(self, root, rows):
        path = root / 'inputs.json'
        path.write_text(json.dumps(rows))
        return path, dict(bytes=path.stat().st_size, sha256=W.digest(path))

    def test_only_frozen_ordered_input_messages_are_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'id': 'a', 'messages': [{'role': 'user', 'content': 'Question'}]}]
            path, pin = self.write_inputs(root, rows)
            self.assertEqual(W.input_rows(path, pin, ['a']), rows)
            with self.assertRaisesRegex(RuntimeError, 'cohort'):
                W.input_rows(path, pin, ['b'])
            with self.assertRaisesRegex(RuntimeError, 'authentication'):
                W.input_rows(path, dict(pin, sha256='0' * 64), ['a'])
            rows[0]['answer'] = 'must never reach the native worker'
            path, pin = self.write_inputs(root, rows)
            with self.assertRaisesRegex(RuntimeError, 'references'):
                W.input_rows(path, pin, ['a'])

    def test_duplicate_case_and_non_user_terminal_message_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(id='a', messages=[dict(role='assistant', content='not a prompt')])
            path, pin = self.write_inputs(root, [row])
            with self.assertRaisesRegex(RuntimeError, 'message contract'):
                W.input_rows(path, pin, ['a'])
            path, pin = self.write_inputs(root, [row, row])
            with self.assertRaisesRegex(RuntimeError, 'cohort'):
                W.input_rows(path, pin, ['a', 'a'])


class ChannelBoundaryTests(unittest.TestCase):
    def listing(self, key, contents):
        return dict(Name='bucket', Prefix=key, IsTruncated=False, MaxKeys=2,
                    KeyCount=len(contents), Contents=contents)

    def test_absence_cannot_exceed_frozen_list_request_count(self):
        client = Mock(return_value=self.listing('key', []))
        client.deadline = time.time() + 100
        with tempfile.TemporaryDirectory() as temp, patch.object(W.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'LIST request cap exhausted'):
                W.await_object(client, 'bucket', 'key', Path(temp) / 'decision.json',
                               time.time() + 30, 100, max_polls=3)
        self.assertEqual(client.call_count, 3)
        self.assertTrue(all(call.args[1] == 'list-objects-v2' for call in client.call_args_list))

    def test_absence_lists_then_exact_present_get_without_retrying_access_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'decision.json'
            client = Mock()
            client.deadline = time.time() + 100
            client.side_effect = [self.listing('key', []),
                                  self.listing('key', [dict(Key='key', Size=2)]),
                                  RuntimeError('403 AccessDenied')]
            with patch.object(W.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, '403'):
                    W.await_object(client, 'bucket', 'key', target, time.time() + 30, 100)
            self.assertEqual([call.args[1] for call in client.call_args_list],
                             ['list-objects-v2', 'list-objects-v2', 'get-object'])

    def test_list_access_denied_is_fatal_without_get(self):
        client = Mock(side_effect=RuntimeError('403 AccessDenied'))
        client.deadline = time.time() + 100
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, '403'):
                W.await_object(client, 'bucket', 'key', Path(temp) / 'decision.json', time.time() + 30, 100)
        self.assertEqual(client.call_count, 1)

    def test_unexpected_or_truncated_listing_never_gets(self):
        cases = [dict(self.listing('key', []), IsTruncated=True),
                 self.listing('key', [dict(Key='key-other', Size=2)]),
                 self.listing('key', [dict(Key='key', Size=101)])]
        for listing in cases:
            with self.subTest(listing=listing), tempfile.TemporaryDirectory() as temp:
                client = Mock(return_value=listing)
                client.deadline = time.time() + 100
                with self.assertRaises(RuntimeError):
                    W.await_object(client, 'bucket', 'key', Path(temp) / 'decision.json', time.time() + 30, 100)
                self.assertEqual(client.call_count, 1)

    def test_exact_present_body_is_downloaded_once_and_client_deadline_restored(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'decision.json'
            def response(*args):
                if args[1] == 'list-objects-v2':
                    return self.listing('key', [dict(Key='key', Size=2)])
                target.write_text('{}')
                return {}
            client = Mock(side_effect=response)
            original = client.deadline = time.time() + 100
            self.assertEqual(W.await_object(client, 'bucket', 'key', target, time.time() + 30, 100), target)
            self.assertEqual(client.call_count, 2)
            self.assertEqual(client.deadline, original)


class ExposureBoundaryTests(unittest.TestCase):
    def fixture(self, train_count=72, dropped=0):
        train = [dict(ids=[i], labels=[i], task='train', file='train.jsonl') for i in range(24)] * 3
        valid = [dict(ids=[100 + i], labels=[100 + i], task='dev', file='dev.jsonl') for i in range(8)]
        scope = {'build_items': lambda *args: (train[:train_count] if args[6] else valid, dropped),
                 'batch_loss': lambda model, items, *args, **kwargs: 0.5}
        exec('def main(): pass', scope)
        finish = W.exposure_tracker(scope)
        return scope, train, valid, finish

    def test_authenticates_actual_exposures_not_just_declared_recipe(self):
        scope, train, valid, finish = self.fixture()
        scope['build_items'](None, None, None, None, None, None, True)
        scope['build_items'](None, None, None, None, None, None, False)
        for row in train:
            self.assertEqual(scope['batch_loss'](SimpleNamespace(training=True), [row]), 0.5)
        for row in valid:
            scope['batch_loss'](SimpleNamespace(training=False), [row])
        receipt = finish()
        self.assertEqual(receipt['training_exposures'], 72)
        self.assertEqual(len(receipt['training_item_hashes']), 72)
        self.assertEqual(len(set(receipt['training_item_hashes'])), 24)
        self.assertEqual(receipt['development_rows'], 8)

    def test_missing_actual_exposure_is_not_success(self):
        scope, train, valid, finish = self.fixture()
        scope['build_items'](None, None, None, None, None, None, True)
        scope['build_items'](None, None, None, None, None, None, False)
        for row in train[:-1]:
            scope['batch_loss'](SimpleNamespace(training=True), [row])
        with self.assertRaisesRegex(RuntimeError, 'Actual optimizer exposures'):
            finish()

    def test_dev_optimizer_exposure_and_dropped_data_fail(self):
        scope, train, valid, finish = self.fixture()
        scope['build_items'](None, None, None, None, None, None, True)
        scope['build_items'](None, None, None, None, None, None, False)
        with self.assertRaisesRegex(RuntimeError, 'Unadmitted training'):
            scope['batch_loss'](SimpleNamespace(training=True), [valid[0]])
        scope, _, _, _ = self.fixture(dropped=1)
        with self.assertRaisesRegex(RuntimeError, 'dropped rows'):
            scope['build_items'](None, None, None, None, None, None, True)


class EffectiveGenerationTests(unittest.TestCase):
    def test_real_hf_generation_is_greedy_for_base_and_adapter_phases(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers import Qwen3Config, Qwen3ForCausalLM

        class CPUInputs(dict):
            def to(self, device):
                return self

        phases = ('baseline_dev', 'checkpoint-step-000012-dev')
        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                base = Path(temp)
                (base / 'output').mkdir()
                (base / 'plan.json').write_text('{}')
                case = dict(id='synthetic', messages=[dict(role='user', content='Synthetic prompt')])
                generation = dict(do_sample=False, num_beams=1, context_tokens=32, max_new_tokens=4)
                plan = dict(identity=dict(run_id='tiny-fixture'),
                    model=dict(id='tiny-local', revision='synthetic'),
                    workload=dict(generation=generation, development_inputs=dict(path='unused', sha256='dev'),
                        development_case_ids=[case['id']], final_inputs=dict(sha256='final'),
                        final_case_ids=dict(general=[case['id']], mois=[])))
                model = Qwen3ForCausalLM(Qwen3Config(
                    vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=32, eos_token_id=0, pad_token_id=1,
                    bos_token_id=3, tie_word_embeddings=False)).eval()
                # Equal logits make real greedy generation choose token zero (EOS).
                with torch.no_grad():
                    model.lm_head.weight.zero_()
                model.generation_config = GenerationConfig(
                    do_sample=True, num_beams=4, temperature=.6, top_p=.95, top_k=4,
                    min_p=.8, repetition_penalty=2., no_repeat_ngram_size=2,
                    suppress_tokens=[0], begin_suppress_tokens=[2], forced_eos_token_id=7,
                    use_cache=False, eos_token_id=[0, 2], pad_token_id=5, bos_token_id=6,
                    transformers_version='4.57.6')
                defaults_before = model.generation_config.to_dict()
                if phase in W.CHECKPOINT_PHASES:
                    adapter = get_peft_model(model, LoraConfig(
                        r=2, lora_alpha=2, target_modules=['q_proj', 'v_proj'],
                        task_type='CAUSAL_LM', inference_mode=True)).eval()
                    stack.enter_context(patch.object(PeftModel, 'from_pretrained', return_value=adapter))
                ids = torch.tensor([[3, 4]], dtype=torch.long)
                tok = Mock(pad_token_id=1, eos_token_id=7, bos_token_id=3)
                tok.apply_chat_template.return_value = 'Synthetic prompt'
                tok.return_value = CPUInputs(input_ids=ids, attention_mask=torch.ones_like(ids))
                runtime = SimpleNamespace(manual_seed=torch.manual_seed, cuda=Mock(),
                                          bfloat16=torch.bfloat16, inference_mode=torch.inference_mode)
                worker = SimpleNamespace(decode_generation=Mock(return_value={}))
                observed = []
                prepare = model._prepare_generation_config
                def observe(*args, **kwargs):
                    resolved, model_kwargs = prepare(*args, **kwargs)
                    observed.append(deepcopy(resolved))
                    return resolved, model_kwargs
                stack.enter_context(patch.object(model, '_prepare_generation_config', side_effect=observe))
                stack.enter_context(patch.object(model, 'to', return_value=model))
                stack.enter_context(patch.object(AutoModelForCausalLM, 'from_pretrained', return_value=model))
                stack.enter_context(patch.object(AutoTokenizer, 'from_pretrained', return_value=tok))
                stack.enter_context(patch.object(W, 'input_rows', return_value=[case]))
                hashes = {'step-000012': 'synthetic-adapter'}
                stack.enter_context(patch.object(W, 'adapter_hashes', return_value=hashes))
                stack.enter_context(patch.object(E, 'validate_outputs'))
                W.generation_stage(worker, plan, base, runtime, phase)
                self.assertEqual(len(observed), 1)
                effective = observed[0]
                self.assertEqual(effective.get_generation_mode().value, 'greedy_search')
                self.assertFalse(effective.do_sample)
                self.assertEqual(effective.num_beams, 1)
                self.assertEqual(effective.max_new_tokens, 4)
                self.assertTrue(effective.use_cache)
                self.assertEqual(effective.temperature, 1.)
                self.assertEqual(effective.top_p, 1.)
                self.assertEqual(effective.repetition_penalty, 1.)
                self.assertEqual(effective.no_repeat_ngram_size, 0)
                for field in ('min_p', 'suppress_tokens', 'begin_suppress_tokens', 'forced_eos_token_id'):
                    self.assertIsNone(getattr(effective, field), field)
                self.assertEqual(effective.eos_token_id, [0, 2])
                self.assertEqual(effective.pad_token_id, 1)
                self.assertEqual(effective.bos_token_id, 3)
                row = json.loads(next((base / 'output').glob('*-outputs.jsonl')).read_text())
                self.assertEqual(row['output_token_ids'], [0])
                self.assertEqual(model.generation_config.to_dict(), defaults_before)


class DispatchBoundaryTests(unittest.TestCase):
    def test_historical_worker_import_does_not_require_experiment_only_sources(self):
        with patch.dict(sys.modules, {'eight_hf_experiment': None, 'eight_hf_worker': None}):
            worker = runpy.run_path(str(ROOT / 'scripts/aws_eval/grounded_hf_worker.py'),
                                   run_name='historical_worker_import')
            self.assertIsNone(worker['eight_module']({'schema': 'trid-grounded-hf-plan-v1'}))
            self.assertIsNone(worker['eight_module']({'schema': 'trid-hf14-followup-v1'}))

    def test_training_always_has_explicit_checkpoint_and_validation_schedule(self):
        plan = dict(schema=E.SCHEMA, recipe=E.RECIPE, model=dict(id=E.MODEL_ID, revision=E.MODEL_REVISION),
                    data={'manifest': {'path': 'data/manifest.json'}})
        argv = G.train_argv(plan, '/owned')
        self.assertEqual(argv[argv.index('--checkpoint-every') + 1], '12')
        self.assertEqual(argv[argv.index('--valid-every') + 1], '12')
        self.assertEqual(argv[argv.index('--max-tokens') + 1], '3072')
        self.assertEqual(argv[argv.index('--max-answer-tokens') + 1], '768')
        self.assertNotIn('--merge', argv)
        self.assertNotIn('--init-adapter', argv)

    def test_generation_uses_amended_context_without_trimming_answer_reserve(self):
        plan = {'workload': {'generation': {'context_tokens': 4096, 'max_new_tokens': 768}}}
        W.validate_prompt_budget(plan, 2339)
        W.validate_prompt_budget(plan, 4096 - 768)
        with self.assertRaisesRegex(RuntimeError, 'no truncation'):
            W.validate_prompt_budget(plan, 4096 - 768 + 1)

    def test_native_sandbox_does_not_admit_root_parent_selection_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text('{}')
            plan = dict(schema=E.SCHEMA, identity={'run_id': 'experiment'})
            with self.assertRaisesRegex(RuntimeError, 'Invalid phase'):
                G.stage_argv(plan, base, 'parent_selection', 30, dict(minor=0, uuid='GPU-test'), 1001, 1001)
            argv = G.stage_argv(plan, base, 'baseline_dev', 30, dict(minor=0, uuid='GPU-test'), 1001, 1001)
            self.assertIn('PrivateNetwork=yes', argv)
            self.assertIn('RestrictAddressFamilies=AF_UNIX', argv)
            self.assertIn(f'InaccessiblePaths=-{base / "final-inputs.json"}', argv)
            self.assertIn('AWS_EC2_METADATA_DISABLED=true', argv)

    def test_selection_loss_must_match_checkpoint_before_final_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text('{}')
            hashes = {'step-000012': 'a' * 64}
            checkpoint = dict(archive={'sha256': 'b' * 64}, checkpoint_hashes=hashes)
            decision = dict(aggregate={'candidates': [{'teacher_forced_dev_loss': 0.1}]})
            ledger = {'checkpoints': [{'valid_loss_by_task': {'task': 0.2}}]}
            with patch.object(E, 'validate_selection', return_value=12), \
                    patch.object(E, 'checkpoint_evidence', return_value=(hashes, ledger)):
                with self.assertRaisesRegex(RuntimeError, 'loss differs'):
                    W.validate_decision({}, base, decision, checkpoint)

    def test_no_eligible_checkpoint_never_requests_final_or_dispatches_final_process(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'output').mkdir()
            (base / 'plan.json').write_text('{}')
            plan = dict(schema=E.SCHEMA, model={'files': []}, identity=dict(run_id='experiment', bucket='bucket'),
                phase_seconds=dict(E.PHASE_SECONDS),
                workload={'channel_keys': {'decision': 'decision', 'final_inputs': 'forbidden-final'}})
            report = dict(run_id='experiment', plan_sha256=W.digest(base / 'plan.json'), instance_id='instance', stages=[])
            def stage(plan, base, phase, *args, **kwargs):
                if phase == 'capacity':
                    (base / 'output/capacity').mkdir()
                    W.save(base / 'output/capacity-receipt.json', {'discarded': False})
                return dict(phase=phase, state='complete')
            worker = Mock()
            worker.supervise_phase.side_effect = stage
            worker.export_inventory.return_value = {'files': [dict(path=name) for name in W.SELECTION_MEMBERS]}
            def decision(client, bucket, key, target, deadline, maximum):
                self.assertEqual(key, 'decision')
                target.write_text('{}')
                return target
            checkpoint = dict(archive={'sha256': 'a' * 64}, checkpoint_hashes={})
            with patch.object(W, 'make_dev_export', return_value=checkpoint), \
                    patch.object(W, 'await_object', side_effect=decision) as download, \
                    patch.object(W, 'validate_decision', return_value=None), \
                    patch.object(E, 'output_members', return_value=W.FINAL_MEMBERS | W.SELECTION_MEMBERS):
                W.run_experiment(worker, plan, base, report, Mock(), {},
                    SimpleNamespace(pw_uid=1001, pw_gid=1001), time.time() + 10000, time.time() + 100, [], set(), [])
            self.assertEqual(download.call_count, 1)
            self.assertEqual(report['experiment_outcome'], 'no_eligible_checkpoint')
            phases = [call.args[2] for call in worker.supervise_phase.call_args_list]
            self.assertEqual(phases, ['capacity', 'baseline_dev', 'train', *W.CHECKPOINT_PHASES])
            self.assertFalse((base / 'final-inputs.json').exists())


if __name__ == '__main__':
    unittest.main()
