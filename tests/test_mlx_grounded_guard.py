"""Behavior regressions with mocked OS boundaries; never launch MLX or a child.

These tests do not establish real macOS process-group or memory-probe behavior.
The parent must separately authorize any actual local smoke run.
"""
from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import mlx_grounded_guard as guard


def plan_value(output_dir, *, phase='capacity'):
    plan = {'schema': 'trid-local-mlx-v2', 'output_dir': str(output_dir),
            'limits': {'rss_stop_bytes': 6 * guard.GIB, 'mlx_stop_bytes': 6 * guard.GIB,
                       'swap_growth_stop_bytes': 256 * 1024 ** 2, 'pressure_required': 1,
                       'capacity_seconds': 180, 'train_seconds': 900,
                       'evaluation_seconds': 1800, 'question_seconds': 180}}
    if phase == 'inference':
        plan['schema'] = 'trid-offline-mlx-inference-v1'
        plan['limits']['inference_seconds'] = 1800
    return plan


def event(**updates):
    return {'phase': 'capacity', 'event': 'complete', 'mlx_active_bytes': 1024,
            'mlx_cache_bytes': 0, 'mlx_peak_bytes': 2048, **updates}


def encoded(*events):
    return b''.join(json.dumps(item).encode() + b'\n' for item in events)


def case_start(*, phase='baseline', case_id='case-1', start=0, deadline=180):
    return {'event': 'case_started', 'phase': phase, 'case_id': case_id,
            'question_started_monotonic': start, 'question_deadline_monotonic': deadline}


def case_finish(*, phase='baseline', case_id='case-1', finished=1):
    return {'event': 'case_finished', 'phase': phase, 'case_id': case_id,
            'question_finished_monotonic': finished}


class LimitTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_value('/unused/results')
        self.limits = guard.read_limits(self.plan, 'capacity')

    def test_old_plan_schema_is_not_admitted(self):
        self.plan['schema'] = 'trid-local-mlx-v1'
        with self.assertRaisesRegex(ValueError, 'schema'):
            guard.read_limits(self.plan, 'capacity')

    def test_plan_schemas_are_exclusive_to_their_phases(self):
        for phase in guard.PHASE_SECONDS:
            with self.subTest(phase=phase):
                plan = plan_value('/unused/results', phase=phase)
                self.assertEqual(guard.read_limits(plan, phase)['duration_seconds'],
                                 guard.PHASE_SECONDS[phase])
                plan['schema'] = ('trid-local-mlx-v2' if phase == 'inference'
                                  else 'trid-offline-mlx-inference-v1')
                with self.assertRaisesRegex(ValueError, 'schema'):
                    guard.read_limits(plan, phase)

    def test_sample_thresholds_allow_exact_limit_and_stop_above(self):
        baseline = 777 * 1024 ** 2
        swap_cap = self.limits['swap_growth_stop_bytes']
        self.assertIsNone(guard.sample_stop_reason(self.limits, 6 * guard.GIB, 1, baseline + swap_cap, baseline))
        self.assertEqual(guard.sample_stop_reason(self.limits, 6 * guard.GIB + 1, 1, baseline, baseline), 'rss_limit')
        self.assertEqual(guard.sample_stop_reason(self.limits, 0, 2, baseline, baseline), 'pressure_not_1')
        self.assertEqual(guard.sample_stop_reason(self.limits, 0, 1, baseline + swap_cap + 1, baseline), 'swap_growth_limit')
        self.assertIsNone(guard.sample_stop_reason(self.limits, 0, 1, baseline - 1, baseline))

    def test_duration_cannot_exceed_phase_caps(self):
        for phase, ceiling in guard.PHASE_SECONDS.items():
            with self.subTest(phase=phase):
                plan = plan_value('/unused/results', phase=phase)
                key = guard.PHASE_LIMIT_KEYS.get(phase, 'audit_seconds')
                plan['limits'][key] = ceiling + 100
                self.assertEqual(guard.read_limits(plan, phase)['duration_seconds'], ceiling)
                plan['limits'][key] = 2
                self.assertEqual(guard.read_limits(plan, phase)['duration_seconds'], 2)
        for invalid in (True, -1, 0, float('nan'), float('inf')):
            self.plan['limits']['capacity_seconds'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                guard.read_limits(self.plan, 'capacity')

    def test_inference_uses_its_own_duration_and_bounded_question_deadline(self):
        plan = plan_value('/unused/results', phase='inference')
        plan['limits'].update(inference_seconds=90, evaluation_seconds=1, question_seconds=30)
        limits = guard.read_limits(plan, 'inference')
        self.assertEqual(limits['duration_seconds'], 90)
        self.assertEqual(limits['question_seconds'], 30)
        for key in ('inference_seconds', 'question_seconds'):
            for invalid in (True, -1, 0, float('nan'), float('inf')):
                candidate = plan_value('/unused/results', phase='inference')
                candidate['limits'][key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    guard.read_limits(candidate, 'inference')
        plan['limits']['question_seconds'] = 181
        with self.assertRaisesRegex(ValueError, 'question_seconds'):
            guard.read_limits(plan, 'inference')

    def test_resource_limit_plan_cannot_weaken_registration(self):
        for phase in ('capacity', 'inference'):
            for key, value in (('pressure_required', 2), ('pressure_required', True),
                               ('rss_stop_bytes', 6 * guard.GIB + 1),
                               ('mlx_stop_bytes', 6 * guard.GIB + 1),
                               ('swap_growth_stop_bytes', 256 * 1024 ** 2 + 1)):
                candidate = plan_value('/unused/results', phase=phase)
                candidate['limits'][key] = value
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    guard.read_limits(candidate, phase)

    def test_swap_uses_used_not_total_and_keeps_units(self):
        self.assertEqual(guard.parse_swap('total = 4096.00M used = 1280.50M free = 2815.50M (encrypted)'),
                         int(1280.5 * 1024 ** 2))
        self.assertEqual(guard.parse_swap('used = 1.25G'), int(1.25 * guard.GIB))
        for malformed in ('total = 100M free = 100M', 'used = -1M', 'used = unknown'):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                guard.parse_swap(malformed)

    def test_group_probe_never_includes_foreign_group(self):
        result = Mock(stdout='111 111 S 10\n112 111 Z 0\n999 999 R 999999\n')
        with patch.object(guard.subprocess, 'run', return_value=result):
            rows = guard.group_members(111)
        self.assertEqual([row['pid'] for row in rows], [111, 112])
        self.assertEqual(sum(row['rss_bytes'] for row in rows), 10240)


class TelemetryTests(unittest.TestCase):
    def test_split_lines_and_terminal_failure_without_newline(self):
        telemetry = guard.Telemetry('capacity', 4096)
        raw = encoded(event())
        telemetry.feed(b'native diagnostic\n' + raw[:10])
        telemetry.feed(raw[10:])
        telemetry.feed(json.dumps(event(event='failure', stop_reason='mlx_memory_limit')).encode(), final=True)
        self.assertTrue(telemetry.complete)
        self.assertEqual(telemetry.failure, 'worker_mlx_memory_limit')
        self.assertEqual(len(telemetry.samples), 2)

    def test_active_plus_cache_and_historical_peak_both_stop(self):
        for changes in ({'mlx_active_bytes': 3000, 'mlx_cache_bytes': 1097}, {'mlx_peak_bytes': 4097}):
            telemetry = guard.Telemetry('capacity', 4096)
            telemetry.feed(encoded(event(**changes)))
            self.assertEqual(telemetry.failure, 'mlx_limit')
        telemetry = guard.Telemetry('capacity', 4096)
        telemetry.feed(encoded(event(mlx_active_bytes=3000, mlx_cache_bytes=1096, mlx_peak_bytes=4096)))
        self.assertIsNone(telemetry.failure)

    def test_invalid_metrics_and_wrong_phase_fail_closed(self):
        for changes in ({'mlx_cache_bytes': -1}, {'mlx_active_bytes': True}, {'phase': 'train'}):
            telemetry = guard.Telemetry('capacity', 4096)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                telemetry.feed(encoded(event(**changes)))
        telemetry = guard.Telemetry('capacity', 4096)
        with self.assertRaises(ValueError):
            telemetry.feed(encoded({'phase': 'capacity', 'mlx_active_bytes': 1}))

    def test_active_case_deadline_is_independent_of_phase_and_clears_on_finish(self):
        telemetry = guard.Telemetry('baseline', 4096)
        telemetry.feed(encoded(case_start()))
        self.assertIsNone(telemetry.deadline_reason(179.999))
        self.assertEqual(telemetry.deadline_reason(180), 'question_deadline')
        telemetry.feed(encoded(case_finish(finished=179.9)))
        self.assertIsNone(telemetry.deadline_reason(1800))
        self.assertIsNone(telemetry.active_case)
        self.assertEqual(len(telemetry.case_events), 2)

    def test_completed_case_replay_still_rejects_overdue_finish(self):
        telemetry = guard.Telemetry('baseline', 4096)
        telemetry.feed(encoded(case_start(), case_finish(finished=180.001)))
        self.assertEqual(telemetry.failure, 'question_deadline')

    def test_deadline_schema_rejects_overlap_mismatch_and_weakened_cap(self):
        for start in (case_start(deadline=181), case_start(deadline=float('nan')),
                      case_start(start=10, deadline=9)):
            with self.subTest(start=start), self.assertRaises(ValueError):
                guard.Telemetry('baseline', 4096).feed(encoded(start))
        for second in (case_start(case_id='case-2'), case_finish(case_id='case-2')):
            telemetry = guard.Telemetry('baseline', 4096)
            telemetry.feed(encoded(case_start()))
            with self.assertRaises(ValueError):
                telemetry.feed(encoded(second))


class SupervisorTests(unittest.TestCase):
    def run_guard(self, *, samples=None, initial_events=None, late_events=None,
                  exit_code=0, remaining=False, kill_denied=False, missing_receipt=False,
                  phase='capacity', clock_step=0.01, probe_error=False, fail_spawn=False,
                  worker_receipt_updates=None, reap_error=False, fire_question_alarm=False,
                  permission_state='zombie'):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            output = root / 'results'
            output.mkdir()
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan_value(output, phase=phase)))
            receipt_path = root / 'guard.json'
            log_path = root / 'worker.log'
            actions = []
            state = {'waited': False, 'observations': 0, 'denied': False}
            child = Mock(pid=12345)
            initial_events = [event(phase=phase)] if initial_events is None else initial_events
            late_events = [] if late_events is None else late_events
            sample_values = samples or [{'pressure': 1, 'swap_bytes': 500 * 1024 ** 2}]
            sequence = iter(sample_values)
            current = sample_values[-1]

            def probe():
                nonlocal current
                actions.append('probe')
                if probe_error:
                    raise RuntimeError('sysctl unavailable')
                current = next(sequence, current)
                return current

            def members(pgid):
                self.assertEqual(pgid, 12345)
                state['observations'] += 1
                if state['waited']:
                    return ([{'pid': 12346, 'pgid': pgid, 'status': 'Z', 'rss_bytes': 0}]
                            if remaining else [])
                if state['denied']:
                    actions.append('permission_observation')
                    if permission_state == 'unknown':
                        raise RuntimeError('Cannot inspect group')
                    if permission_state == 'empty':
                        return []
                    leader = {'pid': pgid, 'pgid': pgid,
                              'status': 'S' if permission_state == 'live' else 'Z', 'rss_bytes': 0}
                    if permission_state == 'other':
                        return [leader, {'pid': pgid + 1, 'pgid': pgid, 'status': 'Z', 'rss_bytes': 0}]
                    if permission_state == 'wrong_leader':
                        return [{'pid': pgid + 1, 'pgid': pgid, 'status': 'Z', 'rss_bytes': 0}]
                    return [leader]
                return [{'pid': pgid, 'pgid': pgid, 'status': 'S' if state['observations'] == 1 else 'Z',
                         'rss_bytes': 1000}]

            def spawn(*args, **kwargs):
                actions.append('spawn')
                self.assertTrue(kwargs['start_new_session'])
                self.assertIn('TRID_UI_START_FD', kwargs['env'])
                self.assertEqual(kwargs['pass_fds'], (int(kwargs['env']['TRID_UI_START_FD']),))
                if fail_spawn:
                    raise OSError('spawn failed')
                kwargs['stdout'].write(encoded(*initial_events))
                if not missing_receipt:
                    (output / f'{phase}-receipt.json').write_text(json.dumps({
                        'schema': ('trid-offline-mlx-inference-receipt-v1' if phase == 'inference'
                                   else 'trid-local-mlx-receipt-v2'), 'phase': phase,
                        'status': 'complete', 'plan_sha256': guard.file_pin(plan_path)['sha256'],
                        **(worker_receipt_updates or {})}))
                return child

            def wait(timeout):
                actions.append('wait')
                if reap_error:
                    raise TimeoutError('Leader did not reap')
                state['waited'] = True
                with log_path.open('ab') as log:
                    log.write(encoded(*late_events))
                return exit_code

            def kill(pgid, sig):
                actions.append('kill')
                self.assertEqual(pgid, 12345)
                self.assertEqual(sig, signal.SIGKILL)
                self.assertFalse(state['waited'], 'Must not signal recyclable PGID after reap')
                if kill_denied:
                    state['denied'] = True
                    raise PermissionError(1, 'Operation not permitted')

            def release(fd, data):
                actions.append('release')
                self.assertEqual(data, b'G')
                self.assertIn('probe', actions)
                saved = json.loads(receipt_path.read_text())
                self.assertEqual(saved['owned_pgid'], 12345)
                self.assertEqual(saved['swap_baseline_bytes'], sample_values[0]['swap_bytes'])
                return 1

            child.wait.side_effect = wait
            for obj, name, value in ((guard, 'probe_system', probe), (guard, 'group_members', members),
                                     (guard.subprocess, 'Popen', spawn), (guard.os, 'killpg', kill),
                                     (guard.os, 'write', release)):
                stack.enter_context(patch.object(obj, name, side_effect=value))
            stack.enter_context(patch.object(guard.os, 'getpgid', return_value=12345))
            installed_handlers = {}
            stack.enter_context(patch.object(guard.signal, 'signal',
                                            side_effect=lambda sig, fn: installed_handlers.update({sig: fn})))

            def set_timer(which, seconds):
                if fire_question_alarm and which == signal.ITIMER_REAL and 0 < seconds <= 180:
                    installed_handlers[signal.SIGALRM](signal.SIGALRM, None)

            stack.enter_context(patch.object(guard.signal, 'setitimer', side_effect=set_timer))
            stack.enter_context(patch.object(guard.time, 'sleep'))
            ticks = itertools.count(0, clock_step)
            stack.enter_context(patch.object(guard.time, 'monotonic', side_effect=lambda: next(ticks)))
            record = guard.supervise(plan_path, phase, receipt_path, log_path, ['python', 'worker.py'])
            self.assertEqual(json.loads(receipt_path.read_text()), record)
            raw_log = log_path.read_bytes()
            return record, actions, raw_log

    def test_success_persists_actual_baseline_and_kills_before_reap(self):
        record, actions, log = self.run_guard()
        self.assertTrue(record['success'])
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['swap_baseline_bytes'], 500 * 1024 ** 2)
        self.assertEqual(record['remaining_owned_members'], [])
        self.assertLess(actions.index('probe'), actions.index('release'))
        self.assertLess(actions.index('kill'), actions.index('wait'))
        self.assertEqual(log, encoded(event()))
        self.assertEqual(record['peak_rss_bytes'], 1000)
        self.assertIn('sha256', record['worker_receipt'])

    def test_initial_pressure_denies_G_and_still_reaps(self):
        record, actions, _ = self.run_guard(samples=[{'pressure': 2, 'swap_bytes': 123}])
        self.assertFalse(record['startup_released'])
        self.assertNotIn('release', actions)
        self.assertTrue(record['reaped'])
        self.assertEqual(record['stop_reason'], 'pressure_not_1')

    def test_swap_growth_uses_actual_initial_baseline(self):
        baseline = 5 * guard.GIB
        record, _, _ = self.run_guard(samples=[{'pressure': 1, 'swap_bytes': baseline},
                                               {'pressure': 1, 'swap_bytes': baseline + 256 * 1024 ** 2 + 1}])
        self.assertEqual(record['stop_reason'], 'swap_growth_limit')
        self.assertEqual(record['swap_growth_peak_bytes'], 256 * 1024 ** 2 + 1)
        self.assertFalse(record['success'])

    def test_late_failure_even_after_complete_and_exit_zero_is_not_success(self):
        record, _, _ = self.run_guard(late_events=[event(event='failure', stop_reason='question_deadline')])
        self.assertEqual(record['stop_reason'], 'worker_question_deadline')
        self.assertFalse(record['success'])

    def test_zero_exit_without_metrics_completion_or_receipt_is_not_success(self):
        for kwargs, reason in (({'initial_events': []}, 'missing_mlx_telemetry'),
                               ({'initial_events': [event(event='progress')]}, 'missing_worker_completion'),
                               ({'missing_receipt': True}, 'missing_worker_receipt'),
                               ({'exit_code': 9}, 'worker_nonzero_exit')):
            with self.subTest(kwargs=kwargs):
                record, _, _ = self.run_guard(**kwargs)
                self.assertEqual(record['stop_reason'], reason)
                self.assertFalse(record['success'])

    def test_audit_can_complete_without_allocator_metrics(self):
        record, _, _ = self.run_guard(phase='audit', initial_events=[{'phase': 'audit', 'event': 'complete'}])
        self.assertTrue(record['success'])

    def test_remaining_owned_member_or_permission_failure_precludes_success(self):
        for kwargs in ({'remaining': True}, {'kill_denied': True, 'permission_state': 'live'}):
            with self.subTest(kwargs=kwargs):
                record, actions, _ = self.run_guard(**kwargs)
                self.assertFalse(record['success'])
                self.assertEqual(record['cleanup_state'], 'failed_or_unproven')
                self.assertLess(actions.index('kill'), actions.index('wait'))

    def test_permission_denied_for_sole_unreaped_zombie_is_proven_dead_not_signalled(self):
        record, actions, _ = self.run_guard(kill_denied=True)
        self.assertTrue(record['success'])
        self.assertEqual(record['cleanup_errors'], [])
        attempt = record['signals'][-1]
        self.assertIn('PermissionError', attempt['result'])
        self.assertEqual(attempt['errno'], 1)
        self.assertTrue(attempt['accepted_sole_unreaped_zombie'])
        self.assertEqual(attempt['permission_members_before_reap'],
                         [{'pid': 12345, 'pgid': 12345, 'status': 'Z', 'rss_bytes': 0}])
        self.assertLess(actions.index('kill'), actions.index('permission_observation'))
        self.assertLess(actions.index('permission_observation'), actions.index('wait'))
        self.assertEqual(record['remaining_owned_members'], [])

    def test_permission_denied_without_exact_sole_zombie_proof_remains_failure(self):
        for permission_state in ('live', 'other', 'wrong_leader', 'empty', 'unknown'):
            with self.subTest(permission_state=permission_state):
                record, _, _ = self.run_guard(kill_denied=True, permission_state=permission_state)
                self.assertFalse(record['success'])
                self.assertEqual(record['cleanup_state'], 'failed_or_unproven')
                self.assertFalse(record['signals'][-1]['accepted_sole_unreaped_zombie'])
                self.assertTrue(record['cleanup_errors'])

    def test_final_zombie_proof_does_not_retroactively_accept_alarm_permission_failure(self):
        record, _, _ = self.run_guard(
            phase='baseline', initial_events=[case_start()], fire_question_alarm=True, kill_denied=True)
        self.assertFalse(record['success'])
        self.assertEqual(record['stop_reason'], 'question_deadline')
        self.assertFalse(record['signals'][0]['accepted_sole_unreaped_zombie'])
        self.assertTrue(record['signals'][-1]['accepted_sole_unreaped_zombie'])
        self.assertEqual(record['cleanup_state'], 'failed_or_unproven')

    def test_failed_reap_cannot_report_success(self):
        record, _, _ = self.run_guard(reap_error=True)
        self.assertFalse(record['success'])
        self.assertFalse(record['reaped'])
        self.assertEqual(record['cleanup_state'], 'failed_or_unproven')

    def test_contradictory_or_wrong_plan_worker_receipt_fails_consumer(self):
        for updates in ({'status': 'failed'}, {'phase': 'train'},
                        {'plan_sha256': 'not this plan'}, {'schema': 'unrelated'},
                        {'schema': 'trid-local-mlx-receipt-v1'},
                        {'schema': 'trid-offline-mlx-inference-receipt-v1'}):
            with self.subTest(updates=updates):
                record, _, _ = self.run_guard(worker_receipt_updates=updates)
                self.assertFalse(record['success'])
                self.assertEqual(record['stop_reason'], 'invalid_worker_receipt')

    def test_inference_requires_its_actual_receipt_schema_and_plan_binding(self):
        events = [case_start(phase='inference'), case_finish(phase='inference'),
                  event(phase='inference')]
        for updates in ({'schema': 'trid-local-mlx-receipt-v2'}, {'schema': 'unrelated'},
                        {'phase': 'baseline'}, {'status': 'failed'},
                        {'plan_sha256': 'not this plan'}):
            with self.subTest(updates=updates):
                record, _, _ = self.run_guard(
                    phase='inference', initial_events=events, worker_receipt_updates=updates)
                self.assertFalse(record['success'])
                self.assertEqual(record['stop_reason'], 'invalid_worker_receipt')
        record, _, _ = self.run_guard(
            phase='inference', initial_events=events, missing_receipt=True)
        self.assertFalse(record['success'])
        self.assertEqual(record['stop_reason'], 'missing_worker_receipt')

    def test_malformed_late_metric_fails_consumer_after_zero_exit(self):
        record, _, _ = self.run_guard(late_events=[event(mlx_peak_bytes='unknown')])
        self.assertFalse(record['success'])
        self.assertEqual(record['stop_reason'], 'telemetry_or_receipt_error')

    def test_probe_and_spawn_errors_preserve_failure_receipt(self):
        record, actions, _ = self.run_guard(probe_error=True)
        self.assertEqual(record['stop_reason'], 'guard_error')
        self.assertNotIn('release', actions)
        self.assertTrue(record['reaped'])
        record, actions, _ = self.run_guard(fail_spawn=True)
        self.assertFalse(record['success'])
        self.assertNotIn('release', actions)
        self.assertNotIn('kill', actions)

    def test_parent_deadline_prevents_startup_even_if_child_cooperates(self):
        record, actions, _ = self.run_guard(clock_step=100)
        self.assertEqual(record['stop_reason'], 'phase_deadline')
        self.assertNotIn('release', actions)
        self.assertTrue(record['reaped'])

    def test_external_question_alarm_kills_without_worker_failure_or_completion(self):
        for phase in ('baseline', 'adapter', 'inference'):
            with self.subTest(phase=phase):
                record, actions, _ = self.run_guard(
                    phase=phase, initial_events=[case_start(phase=phase)], fire_question_alarm=True)
                self.assertFalse(record['success'])
                self.assertTrue(record['hard_alarm_fired'])
                self.assertEqual(record['stop_reason'], 'question_deadline')
                self.assertEqual(record['active_case_at_stop']['case_id'], 'case-1')
                self.assertLess(actions.index('kill'), actions.index('wait'))

    def test_already_expired_case_deadline_does_not_get_fresh_180_seconds(self):
        for phase in ('baseline', 'adapter', 'inference'):
            with self.subTest(phase=phase):
                record, _, _ = self.run_guard(
                    phase=phase, initial_events=[case_start(phase=phase, deadline=0.001)])
                self.assertEqual(record['stop_reason'], 'question_deadline')
                self.assertTrue(record['hard_alarm_fired'])

    def test_eval_and_inference_require_case_deadlines_and_accept_completed_pair(self):
        for phase in ('baseline', 'adapter', 'inference'):
            with self.subTest(phase=phase):
                record, _, _ = self.run_guard(phase=phase)
                self.assertFalse(record['success'])
                self.assertEqual(record['stop_reason'], 'missing_case_deadlines')
                events = [case_start(phase=phase), case_finish(phase=phase), event(phase=phase)]
                record, actions, log = self.run_guard(phase=phase, initial_events=events)
                self.assertTrue(record['success'])
                self.assertIsNone(record['active_case_at_stop'])
                self.assertEqual(len(record['case_events']), 2)
                self.assertEqual(log, encoded(*events))
                self.assertTrue(record['startup_released'])
                self.assertTrue(record['reaped'])
                self.assertEqual(record['remaining_owned_members'], [])
                self.assertLess(actions.index('probe'), actions.index('release'))
                self.assertLess(actions.index('kill'), actions.index('wait'))

    def test_inference_cannot_complete_with_unfinished_case(self):
        record, _, _ = self.run_guard(
            phase='inference', initial_events=[case_start(phase='inference'), event(phase='inference')])
        self.assertFalse(record['success'])
        self.assertEqual(record['stop_reason'], 'unfinished_case')

    def test_schema_phase_mismatch_never_starts_child_or_reserves_outputs(self):
        for phase in guard.PHASE_SECONDS:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = plan_value(root / 'results', phase=phase)
                plan['schema'] = ('trid-local-mlx-v2' if phase == 'inference'
                                  else 'trid-offline-mlx-inference-v1')
                plan_path = root / 'plan.json'
                plan_path.write_text(json.dumps(plan))
                receipt_path, log_path = root / 'guard.json', root / 'worker.log'
                with patch.object(guard.subprocess, 'Popen') as spawn, self.assertRaisesRegex(ValueError, 'schema'):
                    guard.supervise(plan_path, phase, receipt_path, log_path, ['python'])
                spawn.assert_not_called()
                self.assertFalse(receipt_path.exists())
                self.assertFalse(log_path.exists())

    def test_existing_output_is_not_overwritten_and_no_child_starts(self):
        for existing_name in ('receipt.json', 'worker.log'):
            with self.subTest(existing_name=existing_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = root / 'plan.json'
                plan.write_text(json.dumps(plan_value(root / 'results')))
                existing = root / existing_name
                existing.write_bytes(b'keep these original bytes')
                with patch.object(guard.subprocess, 'Popen') as spawn, self.assertRaises(FileExistsError):
                    guard.supervise(plan, 'capacity', root / 'receipt.json', root / 'worker.log', ['python'])
                spawn.assert_not_called()
                self.assertEqual(existing.read_bytes(), b'keep these original bytes')


if __name__ == '__main__':
    unittest.main()
