"""CPU-only behavioral coverage for ordered and 14B followup inference; no cloud calls."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import grounded_hf_lifecycle as L
import grounded_hf_worker as W
import ordered_hf_comparison as O


def profile(thinking=False, seed=14):
    return dict(name=f'qwen8-{"thinking" if thinking else "nonthinking"}-seed{seed}',
        thinking=thinking, temperature=.6 if thinking else .7, top_p=.95 if thinking else .8,
        top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1,
        penalty_context_size=0, seed=seed, max_new_tokens=4096 if thinking else 1024,
        context_tokens=8192, prefill_step_size=128)


def plan_fixture():
    return dict(schema=O.SCHEMA, identity=dict(run_id='ordered-fixture', bucket='fixture'),
        workload=dict(profiles=[profile(t, s) for t in (False, True) for s in (14, 29)],
            models={key: dict(id=key, revision='frozen', files=[]) for key in ('qwen8', 'qwen14')},
            development_references=dict(path='dev-references.json'),
            phase_seconds=dict(acquire8=900, dev8=2400, val8=2400, parent_decision=300,
                               acquire14=900, dev14=2400, val14=2400),
            checkpoint_keys=dict(decision='code/parent-decision.json')))


def followup_plan_fixture():
    plan = plan_fixture()
    plan['schema'] = L.FOLLOWUP_SCHEMA
    plan['identity']['run_id'] = 'followup-fixture'
    work = plan['workload']
    del work['models']['qwen8']
    del work['checkpoint_keys']
    work['phase_seconds'] = dict(acquire14=900, dev14=2400, val14=2400)
    return plan


def decision_listing(*, present=True, **overrides):
    plan = plan_fixture()
    key = plan['workload']['checkpoint_keys']['decision']
    return dict(dict(Name=plan['identity']['bucket'], Prefix=key, MaxKeys=2,
        IsTruncated=False, KeyCount=int(present),
        Contents=[dict(Key=key, Size=512)] if present else []), **overrides)


def response(text='{"answer":1}', **overrides):
    return dict(dict(response_text=text, termination_reason='eos', terminal_stop_token_id=9,
                     framing_status='complete', reasoning_complete=True), **overrides)


class SamplerAndFramingTests(unittest.TestCase):
    def test_frozen_sample_parameters_are_explicit(self):
        for thinking in (False, True):
            p = profile(thinking)
            options = O.sampler_kwargs(p)
            self.assertTrue(options['do_sample'])
            self.assertEqual(options['num_beams'], 1)
            self.assertEqual(options['temperature'], p['temperature'])
            self.assertEqual(options['top_p'], p['top_p'])
            self.assertEqual(options['top_k'], 20)
            self.assertIsNone(options['min_p'])
            self.assertEqual(options['max_new_tokens'], p['max_new_tokens'])
        with self.assertRaisesRegex(RuntimeError, 'Unsupported frozen sampler'):
            O.sampler_kwargs(dict(profile(), presence_penalty=.1))

    def test_rng_is_reset_for_each_question(self):
        torch = Mock()
        O.reset_case_rng(torch, profile(seed=29))
        O.reset_case_rng(torch, profile(seed=29))
        self.assertEqual(torch.manual_seed.call_count, 2)
        self.assertEqual(torch.cuda.manual_seed_all.call_count, 2)
        torch.manual_seed.assert_called_with(29)
        torch.cuda.manual_seed_all.assert_called_with(29)

    def test_exact_typed_json_never_repairs(self):
        self.assertTrue(O.joint_correct(response(), {'answer': 1}))
        for text in ('{"answer":true}', '{"answer":1.0}', '{"answer":1e0}',
                     '{"answer":1,"answer":1}', '```json\n{"answer":1}\n```',
                     '{"answer":NaN}', '{"answer":1} trailing', '[1]',
                     '{"answer":1,"extra":0}'):
            with self.subTest(text=text):
                self.assertFalse(O.joint_correct(response(text), {'answer': 1}))
        self.assertFalse(O.exact_typed({'a': [1, 2]}, {'a': [2, 1]}))
        self.assertTrue(O.exact_typed({'b': 2, 'a': 1}, {'a': 1, 'b': 2}))

    def test_eos_and_complete_framing_are_required(self):
        for fields in (dict(termination_reason='length'), dict(terminal_stop_token_id=None),
                       dict(framing_status='incomplete_reasoning'), dict(reasoning_complete=False)):
            self.assertFalse(O.joint_correct(response(**fields), {'answer': 1}))

    def test_shared_decoder_preserves_framing(self):
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ''.join({1: '{"answer":1}', 9: '<eos>'}[i] for i in ids)
        inference = O.shared_inference()
        row = inference.decode_generation(Tokenizer(), [1, 9], 9, 8)
        self.assertEqual(row['raw_output'], '{"answer":1}<eos>')
        self.assertEqual(row['response_text'], '{"answer":1}')
        with self.assertRaisesRegex(RuntimeError, 'Unexplained generation termination'):
            inference.decode_generation(Tokenizer(), [1], 9, 8)
        with self.assertRaisesRegex(RuntimeError, 'Stop token before final'):
            inference.decode_generation(Tokenizer(), [9, 1], 9, 2)

    def test_fixed_rank_uses_all_tiebreaks(self):
        base = dict(group='thinking', minimum_correct=3, combined_correct=7,
                    mean_elapsed_seconds=5., peak_cuda_allocated_bytes=100)
        worse = [dict(base, minimum_correct=2, combined_correct=12),
                 dict(base, combined_correct=6), dict(base, mean_elapsed_seconds=6.),
                 dict(base, peak_cuda_allocated_bytes=101), dict(base, group='zzz')]
        for candidate in worse:
            self.assertLess(O.selection_rank(base), O.selection_rank(candidate))


class EffectiveSamplerTests(unittest.TestCase):
    def test_real_hf_generation_preserves_intentional_sampling_profiles(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model_key = 'qwen8'
        for thinking in (False, True):
            plan = plan_fixture()
            p = next(p for p in O.profiles_for(plan, model_key)
                     if p['thinking'] == thinking and p['seed'] == 14)
            phase = O.profile_phase(model_key, 'dev', p)
            with self.subTest(profile=p['name']), \
                    tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                base = Path(temp)
                (base / 'output').mkdir()
                (base / 'plan.json').write_text('{}')
                plan['workload']['development_inputs'] = dict(sha256='synthetic')
                case = dict(id='synthetic', messages=[dict(role='user', content='Synthetic prompt')])
                model = Qwen3ForCausalLM(Qwen3Config(
                    vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=8192, eos_token_id=0, pad_token_id=1,
                    bos_token_id=3, tie_word_embeddings=False)).eval()
                # Real multinomial sampling over equal logits; the model has no downloaded weights.
                with torch.no_grad():
                    model.lm_head.weight.zero_()
                model.generation_config = GenerationConfig(
                    do_sample=False, num_beams=4, min_p=.8, repetition_penalty=2.,
                    no_repeat_ngram_size=2, suppress_tokens=[0], begin_suppress_tokens=[2],
                    forced_eos_token_id=7, use_cache=False, eos_token_id=[0, 2],
                    pad_token_id=5, bos_token_id=6, transformers_version='4.57.6')
                defaults_before = model.generation_config.to_dict()
                tok = SimpleNamespace(pad_token_id=1, eos_token_id=7, bos_token_id=3)
                def cpu_tensor(data, *, device=None, **kwargs):
                    return torch.tensor(data, **kwargs)
                runtime = SimpleNamespace(
                    manual_seed=torch.manual_seed, cuda=Mock(), tensor=cpu_tensor,
                    ones_like=torch.ones_like, long=torch.long, bfloat16=torch.bfloat16,
                    inference_mode=torch.inference_mode, __version__=torch.__version__,
                    version=SimpleNamespace(cuda=None))
                worker = SimpleNamespace(native_runtime=Mock(return_value=runtime), validate_model=Mock())
                plan['runtime'] = dict(packages={})
                observed = []
                prepare = model._prepare_generation_config
                def observe(*args, **kwargs):
                    resolved, model_kwargs = prepare(*args, **kwargs)
                    observed.append(deepcopy(resolved))
                    return resolved, model_kwargs
                inference = O.shared_inference()
                stack.enter_context(patch.object(model, '_prepare_generation_config', side_effect=observe))
                stack.enter_context(patch.object(model, 'to', return_value=model))
                stack.enter_context(patch.object(AutoModelForCausalLM, 'from_pretrained', return_value=model))
                stack.enter_context(patch.object(AutoTokenizer, 'from_pretrained', return_value=tok))
                stack.enter_context(patch.object(O, 'input_cases', return_value=[case]))
                stack.enter_context(patch.object(inference, 'render_prompt',
                                                 return_value=('Synthetic prompt', [3, 4])))
                stack.enter_context(patch.object(inference, 'decode_generation', return_value={}))
                stack.enter_context(patch.object(O, 'memory', return_value=dict(peak_cuda_allocated_bytes=0)))
                stack.enter_context(patch.object(O.signal, 'signal'))
                stack.enter_context(patch.object(O.signal, 'setitimer'))
                O.generation_stage(worker, plan, base, phase, model_key, 'dev', p)
                self.assertEqual(len(observed), 1)
                effective = observed[0]
                self.assertEqual(effective.get_generation_mode().value, 'sample')
                for field, expected in O.sampler_kwargs(p).items():
                    self.assertEqual(getattr(effective, field), expected, field)
                self.assertEqual(effective.no_repeat_ngram_size, 0)
                for field in ('suppress_tokens', 'begin_suppress_tokens', 'forced_eos_token_id'):
                    self.assertIsNone(getattr(effective, field), field)
                self.assertEqual(effective.eos_token_id, [0, 2])
                self.assertEqual(effective.pad_token_id, 1)
                self.assertEqual(effective.bos_token_id, 3)
                row = json.loads((base / 'output' / (phase + '-outputs.jsonl')).read_text())
                self.assertEqual(row['profile_parameters'], p)
                self.assertTrue(row['output_token_ids'])
                self.assertIn(row['output_token_ids'][-1], [0, 2])
                receipt = json.loads((base / 'output' / (phase + '-receipt.json')).read_text())
                self.assertEqual(receipt['state'], 'complete')
                self.assertEqual(model.generation_config.to_dict(), defaults_before)


class GuardTests(unittest.TestCase):
    def test_selector_has_no_gpu_or_model_and_inference_cannot_read_labels(self):
        plan = plan_fixture()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text('{}')
            gpu = dict(uuid='GPU-fixture', minor=0)
            selector = W.stage_argv(plan, base, 'qwen8-select', 60, gpu, 1001, 1001)
            inference = W.stage_argv(plan, base, 'qwen8-dev-nonthinking-seed14', 300, gpu, 1001, 1001)
            self.assertIn('CUDA_VISIBLE_DEVICES=', selector)
            self.assertFalse(any(x.startswith('DeviceAllow=') for x in selector))
            self.assertIn('InaccessiblePaths=' + str(base / 'model'), selector)
            self.assertIn('InaccessiblePaths=' + str(base / 'code/dev-references.json'), inference)
            self.assertIn('PrivateNetwork=yes', selector)
            self.assertIn('MemorySwapMax=0', inference)
            self.assertIn('DevicePolicy=closed', inference)

    def test_unknown_or_training_dispatch_is_rejected(self):
        for phase in ('train', 'capacity', 'adapter', 'qwen14-dev-unregistered'):
            with self.assertRaisesRegex(RuntimeError, 'forbids training/capacity/unknown'):
                O.child_stage(Mock(), plan_fixture(), '.', phase)

    def test_external_question_guard_and_atomic_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'case-receipt.json'
            O.atomic_receipt(path, dict(question_deadline_epoch=100))
            O.check_question_deadline(path, now=99)
            with self.assertRaisesRegex(RuntimeError, '180-second'):
                O.check_question_deadline(path, now=100)
            self.assertFalse(path.with_name(path.name + '.pending').exists())

    def test_selector_refuses_root_and_credentials(self):
        with patch.object(O.os, 'geteuid', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'CPU-only'):
                O.selection_stage(plan_fixture(), '.', 'qwen8')
        with patch.object(O.os, 'geteuid', return_value=1001), patch.dict(O.os.environ,
                {'CUDA_VISIBLE_DEVICES': '', 'AWS_ACCESS_KEY_ID': 'fixture'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'credential-free'):
                O.selection_stage(plan_fixture(), '.', 'qwen8')


class SelectionTests(unittest.TestCase):
    def setup_development(self, base, plan=None, model_key='qwen8'):
        plan = plan_fixture() if plan is None else plan
        work = plan['workload']
        (base / 'code').mkdir()
        (base / 'output').mkdir()
        (base / 'plan.json').write_text('{}')
        inputs = [dict(id=f'dev-{i}', messages=[dict(role='user', content=f'Question {i}')]) for i in range(6)]
        refs = dict(cases=[dict(c, reference={'answer': 1}) for c in inputs])
        work['development_case_ids'] = [c['id'] for c in inputs]
        work['selection_policy'] = {'metric': 'joint_correct'}
        for key, data in (('development_inputs', inputs), ('development_references', refs)):
            path = base / 'code' / (key + '.json')
            path.write_text(json.dumps(data))
            work[key] = dict(path=path.name, bytes=path.stat().st_size, sha256=O.digest(path))
        for p in O.profiles_for(plan, model_key):
            stem = O.profile_phase(model_key, 'dev', p)
            path = base / 'output' / (stem + '-outputs.jsonl')
            rows = [dict(response(), id=c['id'], profile_parameters=p,
                         prompt_sha256=O.object_sha(c['messages']),
                         model_sha256=O.object_sha(work['models'][model_key]),
                         elapsed_seconds=2 if p['thinking'] else 1,
                         memory={'peak_cuda_allocated_bytes': 100}) for c in inputs]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            receipt = dict(state='complete', completed_case_ids=work['development_case_ids'], profile=p,
                plan_sha256=O.digest(base / 'plan.json'), model_sha256=O.object_sha(work['models'][model_key]),
                input_sha256=work['development_inputs']['sha256'], output_sha256=O.digest(path),
                output_bytes=path.stat().st_size)
            (base / 'output' / (stem + '-receipt.json')).write_text(json.dumps(receipt))
        return plan

    def test_selector_requires_both_completed_seeds_and_uses_dev_only(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(O.os, 'geteuid', return_value=1001), \
                patch.dict(O.os.environ, {'CUDA_VISIBLE_DEVICES': ''}, clear=True):
            base = Path(temp)
            plan = self.setup_development(base)
            O.selection_stage(plan, base, 'qwen8')
            selected = json.loads((base / 'output/qwen8-selection.json').read_text())
            self.assertEqual(selected['selected_group'], 'nonthinking')
            self.assertEqual(len(selected['selected_profiles']), 2)
            receipt = base / 'output/qwen8-dev-thinking-seed29-receipt.json'
            value = json.loads(receipt.read_text()); value['state'] = 'incomplete'
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'cannot be scored'):
                O.selection_stage(plan, base, 'qwen8')

    def test_followup_selects_independently_from_all_four_current_14b_development_outputs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(O.os, 'geteuid', return_value=1001), \
                patch.dict(O.os.environ, {'CUDA_VISIBLE_DEVICES': ''}, clear=True):
            base = Path(temp)
            plan = self.setup_development(base, followup_plan_fixture(), 'qwen14')
            # There is deliberately no 8B selection, history output, validation input,
            # or model directory available to the CPU selector.
            O.selection_stage(plan, base, 'qwen14')
            selected = json.loads((base / 'output/qwen14-selection.json').read_text())
            self.assertEqual(selected['run_id'], plan['identity']['run_id'])
            self.assertEqual(selected['plan_sha256'], O.digest(base / 'plan.json'))
            self.assertEqual(selected['model_key'], 'qwen14')
            self.assertEqual(selected['model_sha256'], O.object_sha(plan['workload']['models']['qwen14']))
            self.assertEqual(selected['selected_profiles'],
                             ['qwen14-nonthinking-seed14', 'qwen14-nonthinking-seed29'])
            self.assertEqual([row['path'] for row in selected['dev_outputs']], [
                O.profile_phase('qwen14', 'dev', p) + '-outputs.jsonl'
                for p in O.profiles_for(plan, 'qwen14')])
            self.assertEqual(len(selected['groups']), 2)
            self.assertTrue(all(group['minimum_correct'] == 6 and group['combined_correct'] == 12
                                for group in selected['groups']))

    def test_followup_selector_rejects_each_prior_plan_development_receipt(self):
        for p in O.profiles_for(followup_plan_fixture(), 'qwen14'):
            with self.subTest(profile=p['name']), tempfile.TemporaryDirectory() as temp, \
                    patch.object(O.os, 'geteuid', return_value=1001), \
                    patch.dict(O.os.environ, {'CUDA_VISIBLE_DEVICES': ''}, clear=True):
                base = Path(temp)
                plan = self.setup_development(base, followup_plan_fixture(), 'qwen14')
                path = base / 'output' / (O.profile_phase('qwen14', 'dev', p) + '-receipt.json')
                receipt = json.loads(path.read_text())
                receipt['plan_sha256'] = '0' * 64
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(RuntimeError, 'cannot be scored'):
                    O.selection_stage(plan, base, 'qwen14')
                self.assertFalse((base / 'output/qwen14-selection.json').exists())


class ConditionalTransitionTests(unittest.TestCase):
    def run_transition(self, decision, fail_phase=None):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'model').mkdir(); (base / 'output').mkdir()
            plan = plan_fixture()
            calls = []
            worker = Mock()
            def supervise(plan, base, phase, *args, **kwargs):
                calls.append(phase)
                directory = base / 'model' / phase.split('-')[0]
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755,
                                 'Nonroot profile must traverse authenticated model directory')
                if phase == fail_phase:
                    raise RuntimeError('registered phase failed')
                if phase.endswith('-select'):
                    key = phase.split('-')[0]
                    (base / 'output' / (key + '-selection.json')).write_text(json.dumps(dict(
                        state='complete', selected_profiles=[p['name'] for p in O.profiles_for(plan, key) if not p['thinking']])))
            worker.supervise_phase.side_effect = supervise
            report = dict(run_id='ordered-fixture', plan_sha256='fixture', instance_id='i-fixture', stages=[])
            with patch.object(O, 'make_checkpoint', return_value={'export': {'sha256': 'fixture'}}), \
                    patch.object(O, 'await_decision', return_value={'decision': decision}) as wait:
                if fail_phase:
                    with self.assertRaisesRegex(RuntimeError, 'registered phase failed'):
                        O.run_ordered(worker, plan, base, report, Mock(), {}, SimpleNamespace(pw_uid=1001, pw_gid=1001),
                                      time.time() + 10000, time.time() + 900, [100], set(), [])
                    self.assertFalse(wait.called)
                    self.assertEqual(report['stages'][-1]['state'], 'incomplete')
                else:
                    O.run_ordered(worker, plan, base, report, Mock(), {}, SimpleNamespace(pw_uid=1001, pw_gid=1001),
                                  time.time() + 10000, time.time() + 900, [100], set(), [])
                    self.assertEqual(wait.call_count, 1)
                    self.assertEqual((base / 'model/qwen14').exists(), decision == 'compare14')
            self.assertFalse(worker.capacity_stage.called)
            self.assertFalse(worker.train_stage.called)
            return calls

    def test_complete_skips_14b_acquisition_and_profiles(self):
        calls = self.run_transition('complete')
        self.assertEqual(len(calls), 7)
        self.assertFalse(any('qwen14' in phase for phase in calls))
        self.assertEqual(sum('-val-' in p for p in calls), 2)

    def test_compare14_runs_identical_frozen_comparison_in_same_lease(self):
        calls = self.run_transition('compare14')
        self.assertEqual(len(calls), 14)
        self.assertEqual(calls[:7], [p.replace('qwen14', 'qwen8') for p in calls[7:]])

    def test_model_acquisition_normalizes_actual_directories_under_umask077(self):
        previous = os.umask(0o077)
        try:
            # run_transition executes real acquisition mkdir/chmod for both
            # models and checks traversal bits before every guarded profile.
            calls = self.run_transition('compare14')
            self.assertTrue(any(phase.startswith('qwen8-') for phase in calls))
            self.assertTrue(any(phase.startswith('qwen14-') for phase in calls))
        finally:
            os.umask(previous)

    def test_failed_development_is_not_scored_retried_or_validated(self):
        calls = self.run_transition('compare14', fail_phase='qwen8-dev-nonthinking-seed14')
        self.assertEqual(calls, ['qwen8-dev-nonthinking-seed14'])

    def test_decision_timeout_preserves_failed_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'output').mkdir()
            report = dict(run_id='fixture', plan_sha256='fixture', stages=[])
            with self.assertRaises(TimeoutError):
                O.aggregate(base, report, 'parent_decision', Mock(side_effect=TimeoutError('expired')))
            receipt = json.loads((base / 'output/parent_decision-receipt.json').read_text())
            self.assertEqual(receipt['state'], 'incomplete')
            self.assertIn('expired', receipt['error'])

    def test_authenticated_decision_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'output').mkdir()
            (base / 'plan.json').write_text('{}')
            plan = plan_fixture()
            decision = dict(run_id=plan['identity']['run_id'], plan_sha256=O.digest(base / 'plan.json'),
                            checkpoint_sha256='a' * 64, decision='complete', reason='parent-reviewed')
            def download(*args):
                if args[1] == 'list-objects-v2':
                    return decision_listing()
                Path(args[-1]).write_text(json.dumps(decision))
            client = Mock(side_effect=download); client.deadline = time.time() + 1000
            received = O.await_decision(plan, base, client, {'export': {'sha256': 'a' * 64}}, time.time() + 300)
            self.assertEqual(received, decision)
            self.assertEqual(client.call_count, 2)
            with self.assertRaises((RuntimeError, FileExistsError)):
                O.await_decision(plan, base, client, {'export': {'sha256': 'a' * 64}}, time.time() + 300)
            self.assertEqual(client.call_count, 2)

    def test_mismatched_decision_is_not_polled_again(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'output').mkdir()
            (base / 'plan.json').write_text('{}')
            def download(*args):
                if args[1] == 'list-objects-v2':
                    return decision_listing()
                Path(args[-1]).write_text(json.dumps(dict(run_id='wrong', plan_sha256='x',
                    checkpoint_sha256='y', decision='compare14', reason='invalid')))
            client = Mock(side_effect=download); client.deadline = time.time() + 1000
            with self.assertRaises(RuntimeError):
                O.await_decision(plan_fixture(), base, client, {'export': {'sha256': 'a' * 64}}, time.time() + 300)
            self.assertEqual(client.call_count, 2)
            self.assertFalse((base / 'output/parent-decision.json').exists())

    def test_missing_decision_waits_until_publication_then_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'output').mkdir()
            (base / 'plan.json').write_text('{}')
            plan = plan_fixture()
            decision = dict(run_id=plan['identity']['run_id'], plan_sha256=O.digest(base / 'plan.json'),
                            checkpoint_sha256='a' * 64, decision='compare14', reason='parent-reviewed')
            listings = iter([decision_listing(present=False), decision_listing()])
            def transport(*args):
                self.assertEqual(client.deadline, 1300)
                if args[1] == 'list-objects-v2':
                    self.assertEqual(args, ('s3api', 'list-objects-v2', '--bucket', 'fixture',
                        '--prefix', 'code/parent-decision.json', '--max-keys', '2'))
                    return next(listings)
                self.assertEqual(args[:-1], ('s3api', 'get-object', '--bucket', 'fixture',
                    '--key', 'code/parent-decision.json', '--range', 'bytes=0-1048576'))
                Path(args[-1]).write_text(json.dumps(decision))
            client = Mock(side_effect=transport); client.deadline = 2000
            with patch.object(O.time, 'time', return_value=1000), patch.object(O.time, 'sleep') as sleep:
                received = O.await_decision(plan, base, client, {'export': {'sha256': 'a' * 64}}, 1300)
            sleep.assert_called_once_with(5)
            self.assertEqual(received, decision)
            self.assertEqual(json.loads((base / 'output/parent-decision.json').read_text()), decision)
            self.assertEqual([call.args[1] for call in client.call_args_list],
                             ['list-objects-v2', 'list-objects-v2', 'get-object'])
            self.assertEqual(client.deadline, 2000)

    def test_decision_listing_must_be_exact_complete_and_bounded(self):
        key = plan_fixture()['workload']['checkpoint_keys']['decision']
        invalid = [
            dict(Name='other'), dict(Prefix='code/'), dict(IsTruncated=True),
            dict(IsTruncated=None), dict(NextContinuationToken='next'),
            dict(CommonPrefixes=[{'Prefix': key + '/'}]), dict(MaxKeys=1000),
            dict(KeyCount=0), dict(KeyCount=True), dict(Contents=None),
            dict(Contents=[dict(Key=key + '.other', Size=10)]),
            dict(KeyCount=2, Contents=[dict(Key=key, Size=10), dict(Key=key + '.other', Size=10)]),
            *[dict(Contents=[dict(Key=key, Size=size)]) for size in (0, -1, True, '512', 1024 ** 2 + 1)],
        ]
        for mutation in invalid:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                base = Path(temp); (base / 'output').mkdir()
                client = Mock(return_value=decision_listing(**mutation)); client.deadline = 2000
                with patch.object(O.time, 'time', return_value=1000), patch.object(O.time, 'sleep') as sleep:
                    with self.assertRaisesRegex(RuntimeError, 'Parent decision listing'):
                        O.await_decision(plan_fixture(), base, client, {'export': {'sha256': 'a' * 64}}, 1300)
                self.assertEqual(client.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(client.deadline, 2000)
                self.assertFalse((base / 'parent-decision-download.json').exists())

    def test_decision_aws_errors_are_fatal_without_retry(self):
        for operation, code in [('list-objects-v2', 'AccessDenied'),
                                ('list-objects-v2', 'NoSuchBucket'),
                                ('get-object', 'AccessDenied'), ('get-object', 'NoSuchKey')]:
            with self.subTest(operation=operation, code=code), tempfile.TemporaryDirectory() as temp:
                base = Path(temp); (base / 'output').mkdir()
                error = L.AWSCallError('s3api', operation, f'An error occurred ({code})')
                replies = [error] if operation == 'list-objects-v2' else [decision_listing(), error]
                client = Mock(side_effect=replies); client.deadline = 2000
                with patch.object(O.time, 'time', return_value=1000), patch.object(O.time, 'sleep') as sleep:
                    with self.assertRaises(L.AWSCallError):
                        O.await_decision(plan_fixture(), base, client, {'export': {'sha256': 'a' * 64}}, 1300)
                self.assertEqual(client.call_count, len(replies))
                sleep.assert_not_called()
                self.assertEqual(client.deadline, 2000)

    def test_absence_stops_at_frozen_deadline_and_restores_client(self):
        with tempfile.TemporaryDirectory() as temp:
            now = [1000]
            def sleep(seconds):
                now[0] += seconds
            client = Mock(return_value=decision_listing(present=False)); client.deadline = 2000
            with patch.object(O.time, 'time', side_effect=lambda: now[0]), \
                    patch.object(O.time, 'sleep', side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, 'deadline'):
                    O.await_decision(plan_fixture(), temp, client, {'export': {'sha256': 'a' * 64}}, 1300)
            self.assertEqual(now[0], 1300)
            self.assertEqual(client.call_count, 60)
            self.assertTrue(all(call.args[1] == 'list-objects-v2' for call in client.call_args_list))
            self.assertEqual(client.deadline, 2000)

    def test_decision_wait_obeys_shared_request_envelope(self):
        with tempfile.TemporaryDirectory() as temp:
            client = L.Client({'region': 'ap-northeast-2'}, deadline=2000, request_limit=1)
            reply = SimpleNamespace(returncode=0, stdout=json.dumps(decision_listing()), stderr='')
            with patch.object(O.time, 'time', return_value=1000), \
                    patch.object(L.subprocess, 'run', return_value=reply) as run:
                with self.assertRaisesRegex(RuntimeError, 'request envelope'):
                    O.await_decision(plan_fixture(), temp, client, {'export': {'sha256': 'a' * 64}}, 1300)
            self.assertEqual(client.requests, 1)
            run.assert_called_once()
            self.assertEqual(client.deadline, 2000)


class FollowupTransitionTests(unittest.TestCase):
    def phases(self, thinking=False):
        group = 'thinking' if thinking else 'nonthinking'
        return [
            'qwen14-dev-nonthinking-seed14', 'qwen14-dev-nonthinking-seed29',
            'qwen14-dev-thinking-seed14', 'qwen14-dev-thinking-seed29', 'qwen14-select',
            f'qwen14-val-{group}-seed14', f'qwen14-val-{group}-seed29',
        ]

    def run_followup(self, *, thinking=False, fail_phase=None, replay=False,
                     now=1800, acquisition_deadline=1900):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'model').mkdir()
            (base / 'output').mkdir()
            plan = followup_plan_fixture()
            files = [dict(path='config.json'), dict(path='model.safetensors')]
            plan['workload']['models']['qwen14']['files'] = files
            calls, deadlines = [], []
            budget, hosts, units = [100], {'models.example'}, []
            worker, client = Mock(), Mock()
            report = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                          instance_id='i-followup', stages=[])

            def download(row, directory, allowed_hosts, deadline, request_budget):
                deadlines.append(deadline)
                self.assertEqual(directory, base / 'model/qwen14')
                self.assertIs(allowed_hosts, hosts)
                self.assertIs(request_budget, budget)
                L.remaining(deadline)
                if fail_phase == 'acquire14':
                    raise RuntimeError('registered phase failed')

            def supervise(plan, base, phase, gpu, uid, gid, deadline, owned_units, *, seconds):
                calls.append(phase)
                self.assertEqual((uid, gid), (1001, 1001))
                self.assertIs(owned_units, units)
                self.assertEqual(deadline, min(10000, now + 2400))
                self.assertEqual(seconds, 60 if phase.endswith('-select') else 2400)
                self.assertEqual(stat.S_IMODE((base / 'model/qwen14').stat().st_mode), 0o755)
                if phase == fail_phase:
                    raise RuntimeError('registered phase failed')
                if phase.endswith('-select'):
                    selected = [p['name'] for p in O.profiles_for(plan, 'qwen14')
                                if p['thinking'] == thinking]
                    (base / 'output/qwen14-selection.json').write_text(json.dumps(
                        dict(state='complete', selected_profiles=selected)))

            worker.download.side_effect = download
            worker.supervise_phase.side_effect = supervise
            args = (worker, plan, base, report, client, {}, SimpleNamespace(pw_uid=1001, pw_gid=1001),
                    10000, acquisition_deadline, budget, hosts, units)
            with patch.object(O.time, 'time', return_value=now), \
                    patch.object(O, 'make_checkpoint') as checkpoint, \
                    patch.object(O, 'await_decision') as decision:
                if fail_phase or acquisition_deadline <= now:
                    with self.assertRaisesRegex(RuntimeError, 'registered phase failed|deadline'):
                        O.run_ordered(*args)
                    self.assertEqual(report['stages'][-1]['state'], 'incomplete')
                else:
                    O.run_ordered(*args)
                first_stages = [dict(stage) for stage in report['stages']]
                if replay:
                    before = (worker.download.call_count, worker.validate_model.call_count,
                              worker.supervise_phase.call_count)
                    with self.assertRaises(FileExistsError):
                        O.run_ordered(*args)
                    self.assertEqual(before, (worker.download.call_count, worker.validate_model.call_count,
                                              worker.supervise_phase.call_count))
                checkpoint.assert_not_called()
                decision.assert_not_called()
            client.assert_not_called()
            self.assertEqual(client.mock_calls, [])
            worker.capacity_stage.assert_not_called()
            worker.train_stage.assert_not_called()
            self.assertFalse((base / 'model/qwen8').exists())
            self.assertNotIn('checkpoint', report)
            self.assertNotIn('decision', report)
            self.assertEqual(json.loads((base / 'acquire14-dispatch.json').read_text()),
                             dict(model_sha256=O.object_sha(plan['workload']['models']['qwen14'])))
            for stage in first_stages:
                if not replay:
                    self.assertEqual(json.loads((base / 'output' / (stage['phase'] + '-receipt.json')).read_text()),
                                     stage)
            return calls, deadlines, first_stages, worker

    def test_only_14b_acquisition_four_development_selector_and_selected_validation_run(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                calls, deadlines, stages, worker = self.run_followup(thinking=thinking)
                self.assertEqual(calls, self.phases(thinking))
                self.assertEqual([stage['phase'] for stage in stages], ['acquire14', 'dev14', 'val14'])
                self.assertTrue(all(stage['state'] == 'complete' for stage in stages))
                self.assertEqual(deadlines, [1900, 1900])
                worker.validate_model.assert_called_once()
                self.assertEqual(set(worker.validate_model.call_args.args[1]['workload']['models']), {'qwen14'})

    def test_acquisition_failure_never_retries_validates_or_starts_development(self):
        calls, deadlines, stages, worker = self.run_followup(fail_phase='acquire14')
        self.assertEqual(calls, [])
        self.assertEqual(deadlines, [1900])
        self.assertEqual([stage['phase'] for stage in stages], ['acquire14'])
        worker.validate_model.assert_not_called()

    def test_every_child_failure_stops_without_retry_or_next_cohort(self):
        phases = self.phases()
        for index, failed in enumerate(phases):
            with self.subTest(failed=failed):
                calls, _, stages, _ = self.run_followup(fail_phase=failed)
                self.assertEqual(calls, phases[:index + 1])
                expected = ['acquire14', 'dev14'] + (['val14'] if '-val-' in failed else [])
                self.assertEqual([stage['phase'] for stage in stages], expected)
                self.assertTrue(all(stage['state'] == 'complete' for stage in stages[:-1]))
                self.assertIn('registered phase failed', stages[-1]['error'])

    def test_initial_acquisition_deadline_is_not_renewed_after_bootstrap(self):
        calls, deadlines, stages, worker = self.run_followup(now=2000, acquisition_deadline=1900)
        self.assertEqual(deadlines, [1900])
        self.assertEqual(calls, [])
        self.assertEqual([stage['phase'] for stage in stages], ['acquire14'])
        worker.validate_model.assert_not_called()

    def test_acquisition_dispatch_cannot_be_reused_after_success_or_failure(self):
        for failure in (None, 'acquire14', 'qwen14-dev-nonthinking-seed14'):
            with self.subTest(failure=failure):
                self.run_followup(fail_phase=failure, replay=True)


class FollowupGuardTests(unittest.TestCase):
    def test_selector_and_each_inference_cohort_keep_distinct_isolation(self):
        plan = followup_plan_fixture()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            gpu = dict(uuid='GPU-followup', minor=3)
            selector = W.stage_argv(plan, base, 'qwen14-select', 60, gpu, 1001, 1001)
            self.assertIn('CUDA_VISIBLE_DEVICES=', selector)
            self.assertFalse(any(arg.startswith('DeviceAllow=') for arg in selector))
            self.assertIn('InaccessiblePaths=' + str(base / 'model'), selector)
            for cohort in ('dev', 'val'):
                for p in O.profiles_for(plan, 'qwen14'):
                    phase = O.profile_phase('qwen14', cohort, p)
                    argv = W.stage_argv(plan, base, phase, 300, gpu, 1001, 1001)
                    self.assertIn('InaccessiblePaths=' + str(base / 'code/dev-references.json'), argv)
                    self.assertIn('DeviceAllow=/dev/nvidia3 rw', argv)
                    self.assertIn('CUDA_VISIBLE_DEVICES=GPU-followup', argv)
                    for command in (selector, argv):
                        for boundary in ('PrivateNetwork=yes', 'MemorySwapMax=0', 'DevicePolicy=closed',
                                         'NoNewPrivileges=yes', 'ProtectSystem=strict',
                                         'AWS_EC2_METADATA_DISABLED=true', 'HF_HUB_OFFLINE=1'):
                            self.assertIn(boundary, command)
                        self.assertEqual(command[command.index('--uid') + 1], '1001')
                        self.assertEqual(command[command.index('--gid') + 1], '1001')

    def test_followup_rejects_8b_training_and_parent_channel_dispatch(self):
        plan = followup_plan_fixture()
        for phase in ('qwen8-select', 'qwen8-dev-nonthinking-seed14', 'qwen8-val-thinking-seed29',
                      'capacity', 'baseline', 'train', 'adapter', 'parent_decision', 'checkpoint'):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(RuntimeError, 'forbids training/capacity/unknown'):
                    O.child_stage(Mock(), plan, '.', phase)
                with self.assertRaisesRegex(RuntimeError, 'Invalid phase/deadline'):
                    W.stage_argv(plan, '.', phase, 60, {}, 1001, 1001)
        with self.assertRaisesRegex(RuntimeError, 'Followup has no checkpoint'):
            O.make_checkpoint(Mock(), plan, '.', {}, Mock(), 1000)
        with self.assertRaisesRegex(RuntimeError, 'Followup has no parent decision'):
            O.await_decision(plan, '.', Mock(), {}, 1000)

    def test_followup_selector_refuses_root_gpu_and_cloud_credentials(self):
        for uid, environment in ((0, {'CUDA_VISIBLE_DEVICES': ''}),
                                 (1001, {'CUDA_VISIBLE_DEVICES': 'GPU-followup'}),
                                 (1001, {'CUDA_VISIBLE_DEVICES': '', 'AWS_ACCESS_KEY_ID': 'fixture'})):
            with self.subTest(uid=uid, environment=environment), \
                    patch.object(O.os, 'geteuid', return_value=uid), \
                    patch.dict(O.os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'CPU-only, credential-free'):
                    O.selection_stage(followup_plan_fixture(), '.', 'qwen14')


class DecisionPolicyTests(unittest.TestCase):
    def policy_plan(self):
        plan = plan_fixture()
        plan['identity'].update(code_prefix='code/', output_prefix='output/')
        statements = [
            dict(Effect='Allow', Action=['s3:GetObject'], Resource=['arn:aws:s3:::fixture/code/*']),
            dict(Effect='Allow', Action=['s3:PutObject'], Resource=['arn:aws:s3:::fixture/output/*']),
            dict(Effect='Allow', Action=['s3:ListBucket'], Resource=['arn:aws:s3:::fixture'],
                 Condition={'StringEquals': {'s3:prefix': 'code/parent-decision.json'}}),
        ]
        plan['iam'] = {'instance_role': {'inline': {'ExactRunOnly': {'Statement': statements}}}}
        return plan

    def test_only_exact_decision_prefix_listing_is_admitted(self):
        L.validate_worker_policy(self.policy_plan())
        for mutation in ('missing', 'blanket', 'run_prefix', 'wildcard', 'wrong_bucket'):
            with self.subTest(mutation=mutation):
                plan = self.policy_plan()
                statements = plan['iam']['instance_role']['inline']['ExactRunOnly']['Statement']
                if mutation == 'missing':
                    statements.pop()
                elif mutation == 'blanket':
                    del statements[-1]['Condition']
                elif mutation == 'run_prefix':
                    statements[-1]['Condition']['StringEquals']['s3:prefix'] = 'code/'
                elif mutation == 'wildcard':
                    statements[-1]['Condition'] = {'StringLike': {'s3:prefix': 'code/*'}}
                else:
                    statements[-1]['Resource'] = ['arn:aws:s3:::other']
                with self.assertRaisesRegex(RuntimeError, 'exact-decision listing'):
                    L.validate_worker_policy(plan)

    def test_nonordered_worker_policy_remains_read_write_only(self):
        plan = self.policy_plan()
        plan['schema'] = 'trid-grounded-hf-v1'
        plan['iam']['instance_role']['inline']['ExactRunOnly']['Statement'].pop()
        L.validate_worker_policy(plan)


class NetworkQuotaHandoffTests(unittest.TestCase):
    def network_fixture(self, base):
        plan = dict(identity={'run_id': 'owned-fixture'}, network={'egress_byte_cap': 123456789})
        (base / 'plan.json').write_text(json.dumps(plan))
        owner = base / 'bootstrap-network-owner'
        owner.write_text(plan['identity']['run_id'] + '\n' + W.digest(base / 'plan.json') + '\n')
        owner.chmod(0o600)
        return plan

    def exercise_handoff(self, failure=None):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            plan = self.network_fixture(base)
            state = dict(early=True, ipv4=False, ipv6=False)
            calls = []
            def command(argv, deadline):
                calls.append(argv)
                if argv[0:2] == ['iptables', '-I'] and failure == 'ipv4':
                    raise RuntimeError('injected ipv4 failure')
                if argv[0] == 'ip6tables' and failure == 'ipv6':
                    raise RuntimeError('injected ipv6 failure')
                if argv[0:2] == ['iptables', '-I']:
                    state['ipv4'] = True
                if argv[0] == 'ip6tables':
                    state['ipv6'] = True
                if argv[0:2] == ['nft', 'delete']:
                    self.assertTrue(state['ipv4'] and state['ipv6'],
                                    'Bootstrap gate must remain until both full-family gates are active')
                    if failure == 'remove':
                        raise RuntimeError('injected removal failure')
                    state['early'] = False
                self.assertTrue(state['early'] or (state['ipv4'] and state['ipv6']),
                                'No unmetered transition is permitted')
                return ''
            with patch.object(W, 'command', side_effect=command):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        W.install_network_cap(plan, time.time() + 30, base)
                else:
                    W.install_network_cap(plan, time.time() + 30, base)
            return state, calls

    def test_full_gate_installs_before_bootstrap_gate_removal(self):
        state, calls = self.exercise_handoff()
        self.assertEqual(state, dict(early=False, ipv4=True, ipv6=True))
        self.assertEqual(calls[-1], ['nft', 'delete', 'table', 'inet', 'trid_bootstrap'])
        quota = next(argv for argv in calls if '--quota' in argv)
        self.assertEqual(quota[quota.index('--quota') + 1], '123456789')

    def test_full_gate_install_failure_keeps_early_quota(self):
        for failure in ('ipv4', 'ipv6'):
            with self.subTest(failure=failure):
                state, calls = self.exercise_handoff(failure)
                self.assertTrue(state['early'])
                self.assertFalse(any(argv[:2] == ['nft', 'delete'] for argv in calls))

    def test_removal_failure_remains_bounded_and_fails_closed(self):
        state, calls = self.exercise_handoff('remove')
        self.assertTrue(state['early'] and state['ipv4'] and state['ipv6'])

    def test_unowned_bootstrap_gate_is_never_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            plan = self.network_fixture(base)
            (base / 'bootstrap-network-owner').write_text('another-run\nanother-plan\n')
            with patch.object(W, 'command') as command:
                with self.assertRaisesRegex(RuntimeError, 'ownership differs'):
                    W.install_network_cap(plan, time.time() + 30, base)
                command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
