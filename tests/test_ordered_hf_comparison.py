"""CPU-only behavioral coverage for the inference-only ordered worker; no cloud calls."""
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
    def setup_development(self, base):
        plan = plan_fixture()
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
        for p in work['profiles']:
            stem = O.profile_phase('qwen8', 'dev', p)
            path = base / 'output' / (stem + '-outputs.jsonl')
            rows = [dict(response(), id=c['id'], profile_parameters=p,
                         prompt_sha256=O.object_sha(c['messages']),
                         model_sha256=O.object_sha(work['models']['qwen8']),
                         elapsed_seconds=2 if p['thinking'] else 1,
                         memory={'peak_cuda_allocated_bytes': 100}) for c in inputs]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            receipt = dict(state='complete', completed_case_ids=work['development_case_ids'], profile=p,
                plan_sha256=O.digest(base / 'plan.json'), model_sha256=O.object_sha(work['models']['qwen8']),
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
                Path(args[-1]).write_text(json.dumps(decision))
            client = Mock(side_effect=download); client.deadline = time.time() + 1000
            received = O.await_decision(plan, base, client, {'export': {'sha256': 'a' * 64}}, time.time() + 300)
            self.assertEqual(received, decision)
            self.assertEqual(client.call_count, 1)
            with self.assertRaises((RuntimeError, FileExistsError)):
                O.await_decision(plan, base, client, {'export': {'sha256': 'a' * 64}}, time.time() + 300)
            self.assertEqual(client.call_count, 1)

    def test_mismatched_decision_is_not_polled_again(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'output').mkdir()
            (base / 'plan.json').write_text('{}')
            def download(*args):
                Path(args[-1]).write_text(json.dumps(dict(run_id='wrong', plan_sha256='x',
                    checkpoint_sha256='y', decision='compare14', reason='invalid')))
            client = Mock(side_effect=download); client.deadline = time.time() + 1000
            with self.assertRaises(RuntimeError):
                O.await_decision(plan_fixture(), base, client, {'export': {'sha256': 'a' * 64}}, time.time() + 300)
            self.assertEqual(client.call_count, 1)
            self.assertFalse((base / 'output/parent-decision.json').exists())


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
