"""LOCAL MLX watchdog: sampled stop thresholds, never instantaneous hard caps.

The child must read G from TRID_UI_START_FD before importing/allocating MLX.
Only its dedicated process group is signalled. Its leader stays unreaped until
all group signals finish, preventing PGID reuse. macOS cannot reap arbitrary
grandchildren: remaining members are observed and fail closed if not gone.
RSS/system probes and worker MLX telemetry are sampled; mx.set_memory_limit is
an allocator guideline, NOT proof of a hardware-enforced allocation ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

GIB = 1024 ** 3
PHASE_SECONDS = {'audit': 120, 'capacity': 180, 'train': 900,
                 'baseline': 1800, 'adapter': 1800, 'inference': 1800}
PHASE_LIMIT_KEYS = {'capacity': 'capacity_seconds', 'train': 'train_seconds',
                    'baseline': 'evaluation_seconds', 'adapter': 'evaluation_seconds',
                    'inference': 'inference_seconds'}
MLX_KEYS = ('mlx_active_bytes', 'mlx_cache_bytes', 'mlx_peak_bytes')
SAMPLE_INTERVAL = 0.25
MAX_LOG_CHUNK = 1024 ** 2


def file_pin(path):
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 ** 2), b''):
            size += len(chunk)
            digest.update(chunk)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'bytes': size}


def read_limits(plan, phase):
    if phase not in PHASE_SECONDS:
        raise ValueError('Unknown phase')
    expected_schema = ('trid-offline-mlx-inference-v1' if phase == 'inference'
                       else 'trid-local-mlx-v2')
    if plan.get('schema') != expected_schema:
        raise ValueError(f'Unsupported plan schema for {phase}: expected {expected_schema}')
    limits = dict(plan['limits'])
    ceilings = {'rss_stop_bytes': 6 * GIB, 'mlx_stop_bytes': 6 * GIB,
                'swap_growth_stop_bytes': 256 * 1024 ** 2}
    for key, ceiling in ceilings.items():
        if type(limits.get(key)) is not int or not 0 < limits[key] <= ceiling:
            raise ValueError(f'{key} must be a positive integer <= {ceiling}')
    if type(limits.get('pressure_required')) is not int or limits['pressure_required'] != 1:
        raise ValueError('pressure_required must be 1')
    question_seconds = limits.get('question_seconds', 180)
    if (type(question_seconds) not in (int, float) or not math.isfinite(question_seconds)
            or not 0 < question_seconds <= 180):
        raise ValueError('question_seconds must be finite, positive and <= 180')
    limits['question_seconds'] = question_seconds
    duration = limits.get(PHASE_LIMIT_KEYS.get(phase, 'audit_seconds'), PHASE_SECONDS[phase])
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Phase duration must be finite and positive')
    limits['duration_seconds'] = min(duration, PHASE_SECONDS[phase])
    return limits


def parse_swap(text):
    match = re.search(r'\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)\b', text)
    if match is None:
        raise ValueError('Cannot parse actual vm.swapusage used bytes')
    return int(float(match.group(1)) * (1024 ** ('', 'K', 'M', 'G', 'T').index(match.group(2))))


def probe_system():
    def sysctl(name):
        return subprocess.run(['/usr/sbin/sysctl', '-n', name], check=True,
                              capture_output=True, text=True, timeout=1).stdout
    return {'pressure': int(sysctl('kern.memorystatus_vm_pressure_level').strip()),
            'swap_bytes': parse_swap(sysctl('vm.swapusage'))}


def group_members(pgid):
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,stat=,rss='],
                            check=True, capture_output=True, text=True, timeout=1)
    rows = []
    for line in result.stdout.splitlines():
        pid, group, state, rss = line.split()
        if int(group) == pgid:
            rows.append({'pid': int(pid), 'pgid': int(group), 'status': state,
                         'rss_bytes': int(rss) * 1024})
    return rows


def sample_stop_reason(limits, rss, pressure, swap, baseline):
    if pressure != limits['pressure_required']:
        return 'pressure_not_1'
    if rss > limits['rss_stop_bytes']:
        return 'rss_limit'
    if swap - baseline > limits['swap_growth_stop_bytes']:
        return 'swap_growth_limit'
    return None


class Telemetry:
    """Consume only actual worker JSON lines, preserving the original log bytes."""
    def __init__(self, phase, limit, question_seconds=180):
        self.phase = phase
        self.limit = limit
        self.pending = b''
        self.samples = []
        self.failure = None
        self.complete = False
        self.question_seconds = question_seconds
        self.case_events = []
        self.active_case = None

    def deadline_reason(self, now):
        if self.active_case is not None and now >= self.active_case['question_deadline_monotonic']:
            return 'question_deadline'
        return None

    def case_event(self, event):
        case_id = event.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ValueError('Case deadline event requires case_id')
        name = event['event']
        fields = (('question_started_monotonic', 'question_deadline_monotonic')
                  if name == 'case_started' else ('question_finished_monotonic',))
        for key in fields:
            value = event.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid case timestamp: {key}')
        if name == 'case_started':
            if self.active_case is not None:
                raise ValueError('Overlapping case deadlines')
            start = event['question_started_monotonic']
            deadline = event['question_deadline_monotonic']
            if not start < deadline <= start + self.question_seconds:
                raise ValueError('Case deadline exceeds registered question cap')
            self.active_case = {'case_id': case_id, 'question_started_monotonic': start,
                                'question_deadline_monotonic': deadline}
        else:
            if self.active_case is None or self.active_case['case_id'] != case_id:
                raise ValueError('Case finish does not match active case')
            finished = event['question_finished_monotonic']
            if finished < self.active_case['question_started_monotonic']:
                raise ValueError('Case finish precedes start')
            if finished > self.active_case['question_deadline_monotonic']:
                self.failure = 'question_deadline'
            self.active_case = None
        self.case_events.append({'event': name, 'case_id': case_id,
                                 **{key: event[key] for key in fields}})

    def feed(self, chunk, *, final=False):
        self.pending += chunk
        lines = self.pending.split(b'\n')
        self.pending = lines.pop()
        if final and self.pending:
            lines.append(self.pending)
            self.pending = b''
        if len(self.pending) > MAX_LOG_CHUNK:
            raise ValueError('Worker log line exceeds 1 MiB')
        for raw in lines:
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue  # Native library diagnostics remain in the byte-exact log.
            if not isinstance(event, dict):
                continue
            if not any(key in event for key in MLX_KEYS) and event.get('event') not in (
                    'failure', 'complete', 'case_started', 'case_finished'):
                continue
            if event.get('phase') != self.phase:
                raise ValueError('Worker telemetry phase mismatch')
            if event.get('event') in ('case_started', 'case_finished'):
                self.case_event(event)
            if event.get('event') == 'failure':
                self.failure = 'worker_' + str(event.get('stop_reason', 'error'))
            present = [key in event for key in MLX_KEYS]
            if any(present):
                if not all(present) or any(type(event[key]) is not int or event[key] < 0 for key in MLX_KEYS):
                    raise ValueError('Incomplete or invalid MLX telemetry')
                sample = {key: event[key] for key in MLX_KEYS}
                self.samples.append(sample)
                if max(sample['mlx_active_bytes'] + sample['mlx_cache_bytes'], sample['mlx_peak_bytes']) > self.limit:
                    self.failure = 'mlx_limit'
            if event.get('event') == 'complete':
                self.complete = True


def supervise(plan_path, phase, receipt_path, log_path, argv):
    paths = [Path(p) for p in (plan_path, receipt_path, log_path)]
    if not all(p.is_absolute() for p in paths):
        raise ValueError('Plan, receipt and log paths must be absolute')
    plan_path, receipt_path, log_path = paths
    if len({str(p.resolve()) for p in paths}) != 3:
        raise ValueError('Plan, receipt and log must be distinct')
    if not argv:
        raise ValueError('Explicit worker argv required after --')
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    limits = read_limits(plan, phase)
    output_dir = Path(plan['output_dir'])
    if not output_dir.is_absolute():
        raise ValueError('Plan output_dir must be absolute')
    worker_receipt = output_dir / f'{phase}-receipt.json'
    if worker_receipt.exists():
        raise FileExistsError('Worker phase receipt already exists')
    if worker_receipt.resolve() in {receipt_path.resolve(), log_path.resolve()}:
        raise ValueError('Guard outputs must not collide with worker receipt')
    # O_EXCL reserves both outputs before any child exists. Never replace a prior run.
    with receipt_path.open('x', encoding='utf-8') as receipt:
        os.chmod(receipt_path, 0o600)
        with log_path.open('xb', buffering=0) as log:
            os.chmod(log_path, 0o600)
            return _supervise(plan_path, plan_bytes, phase, limits, receipt, log,
                              log_path, worker_receipt, argv)


def _supervise(plan_path, plan_bytes, phase, limits, receipt, log, log_path, worker_receipt, argv):
    started = time.monotonic()
    record = {'schema': 'trid-local-mlx-guard-v1', 'phase': phase,
              'plan_path': str(plan_path), 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
              'argv': list(argv), 'supervisor_pid': os.getpid(), 'limits': limits,
              'started_unix': time.time(), 'duration_seconds': limits['duration_seconds'],
              'limit_semantics': 'Sampled stop thresholds, not instantaneous hardware-hard caps; MLX allocator limit is a guideline.',
              'startup_released': False, 'reaped': False, 'success': False,
              'exit_code': None, 'stop_reason': None, 'signals': [],
              'group_samples': [], 'pressure_samples': [], 'swap_samples': [],
              'peak_rss_bytes': 0, 'swap_baseline_bytes': None, 'swap_peak_bytes': None,
              'remaining_owned_members': None, 'cleanup_errors': [],
              'log_path': str(log_path), 'guard_source': file_pin(__file__)}
    deadline = started + limits['duration_seconds']
    record['planned_deadline_unix'] = record['started_unix'] + limits['duration_seconds']
    child = None
    pgid = None
    read_fd = write_fd = None
    hard_fired = False
    requested = []
    telemetry = Telemetry(phase, limits['mlx_stop_bytes'], limits['question_seconds'])
    armed_reason = 'phase_deadline'
    hard_reason = None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

    def save():
        receipt.seek(0)
        json.dump(record, receipt, ensure_ascii=False, indent=2)
        receipt.write('\n')
        receipt.truncate()
        receipt.flush()
        os.fsync(receipt.fileno())

    def kill_group(sig, *, accept_dead_leader=False):
        if pgid is None:
            return
        attempt = {'signal': signal.Signals(sig).name, 'pgid': pgid,
                   'elapsed_seconds': time.monotonic() - started}
        record['signals'].append(attempt)
        try:
            os.killpg(pgid, sig)
            attempt['result'] = 'sent'
        except ProcessLookupError:
            attempt['result'] = 'group_not_found'
        except PermissionError as error:
            attempt.update(result=repr(error), errno=error.errno)
            accepted = False
            # macOS can deny killpg for an already-dead, unreaped leader.
            # Only a fresh pre-reap observation of that sole zombie proves no
            # living member remains that could fork. Never pardon an earlier
            # alarm failure retrospectively using a later cleanup observation.
            if accept_dead_leader and child is not None and not record['reaped']:
                try:
                    rows = group_members(pgid)
                    attempt['permission_members_before_reap'] = rows
                    accepted = (len(rows) == 1 and rows[0]['pid'] == child.pid
                                and rows[0]['pgid'] == pgid
                                and rows[0]['status'].startswith('Z'))
                except Exception as observation_error:
                    attempt['permission_observation_error'] = repr(observation_error)
            attempt['accepted_sole_unreaped_zombie'] = accepted
            if not accepted:
                record['cleanup_errors'].append({'stage': 'signal', 'error': repr(error)})
        except OSError as error:
            attempt['result'] = repr(error)
            record['cleanup_errors'].append({'stage': 'signal', 'error': repr(error)})

    def alarm(_signum, _frame):
        nonlocal hard_fired, hard_reason
        hard_fired = True
        hard_reason = armed_reason
        kill_group(signal.SIGKILL)

    def arm_deadline():
        nonlocal armed_reason
        next_deadline = deadline
        armed_reason = 'phase_deadline'
        if telemetry.active_case is not None:
            case_deadline = telemetry.active_case['question_deadline_monotonic']
            if case_deadline < next_deadline:
                next_deadline = case_deadline
                armed_reason = 'question_deadline'
        if hard_fired:
            return
        remaining = next_deadline - time.monotonic()
        if remaining <= 0:
            alarm(signal.SIGALRM, None)
        else:
            signal.setitimer(signal.ITIMER_REAL, remaining)

    def request(signum, _frame):
        requested.append(signum)  # Never raise while Popen acquires ownership.

    def observe():
        rows = group_members(pgid)
        system = probe_system()
        elapsed = time.monotonic() - started
        rss = sum(row['rss_bytes'] for row in rows)
        record['group_samples'].append({'elapsed_seconds': elapsed, 'members': rows, 'rss_bytes': rss})
        record['peak_rss_bytes'] = max(record['peak_rss_bytes'], rss)
        record['pressure_samples'].append({'elapsed_seconds': elapsed, 'value': system['pressure']})
        record['swap_samples'].append({'elapsed_seconds': elapsed, 'bytes': system['swap_bytes']})
        if record['swap_baseline_bytes'] is None:
            record['swap_baseline_bytes'] = system['swap_bytes']
            record['swap_peak_bytes'] = system['swap_bytes']
        record['swap_peak_bytes'] = max(record['swap_peak_bytes'], system['swap_bytes'])
        return rows, sample_stop_reason(limits, rss, system['pressure'], system['swap_bytes'], record['swap_baseline_bytes'])

    try:
        signal.signal(signal.SIGTERM, request)
        signal.signal(signal.SIGINT, request)
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.monotonic()))
        read_fd, write_fd = os.pipe()
        env = dict(os.environ)
        env.update(TRID_UI_START_FD=str(read_fd), TRID_UI_SUPERVISOR_PID=str(os.getpid()))
        child = subprocess.Popen(argv, env=env, start_new_session=True, pass_fds=(read_fd,),
                                 stdout=log, stderr=subprocess.STDOUT)
        pgid = child.pid
        record.update(worker_pid=child.pid, owned_pgid=pgid)
        os.close(read_fd)
        read_fd = None
        if os.getpgid(child.pid) != pgid:
            raise RuntimeError('Dedicated worker process group not established')
        rows, reason = observe()
        leader = next((row for row in rows if row['pid'] == child.pid), None)
        if leader is None or leader['status'].startswith('Z'):
            reason = reason or 'worker_exit_before_startup'
        record['stop_reason'] = reason
        save()  # Ownership + actual swap baseline persisted before G.
        if hard_fired or time.monotonic() >= deadline:
            record['stop_reason'] = 'phase_deadline'
        elif requested:
            record['stop_reason'] = 'owner_stop_signal'
        if record['stop_reason'] is None:
            os.write(write_fd, b'G')
            record['startup_released'] = True
        os.close(write_fd)
        write_fd = None
        with log_path.open('rb') as reader:
            while record['stop_reason'] is None:
                telemetry.feed(reader.read(MAX_LOG_CHUNK))
                arm_deadline()  # Parent alarm also interrupts slow system probes.
                rows, reason = observe()
                leader = next((row for row in rows if row['pid'] == child.pid), None)
                if hard_fired:
                    reason = hard_reason
                elif time.monotonic() >= deadline:
                    reason = 'phase_deadline'
                elif telemetry.deadline_reason(time.monotonic()):
                    reason = 'question_deadline'
                elif requested:
                    reason = 'owner_stop_signal'
                elif telemetry.failure:
                    reason = telemetry.failure
                if reason:
                    record['stop_reason'] = reason
                elif leader is None or leader['status'].startswith('Z'):
                    record['stop_reason'] = 'worker_exit'
                else:
                    time.sleep(min(SAMPLE_INTERVAL, max(0, deadline - time.monotonic())))
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        record['stop_reason'] = record['stop_reason'] or 'guard_error'
    finally:
        record['phase_elapsed_seconds'] = time.monotonic() - started
        # No wait/poll before the last killpg: unreaped leader pins owned PGID.
        kill_group(signal.SIGKILL, accept_dead_leader=True)
        signal.setitimer(signal.ITIMER_REAL, 0)
        if pgid is not None:
            cleanup_deadline = time.monotonic() + 3
            try:
                while True:
                    rows = group_members(pgid)
                    record['members_before_reap'] = rows
                    others = [row for row in rows if row['pid'] != child.pid]
                    live = [row for row in rows if not row['status'].startswith('Z')]
                    if not others and not live:
                        break
                    if time.monotonic() >= cleanup_deadline:
                        break
                    time.sleep(0.05)
            except Exception as error:
                record['cleanup_errors'].append({'stage': 'before_reap', 'error': repr(error)})
        if child is not None:
            try:
                record['exit_code'] = child.wait(timeout=3)
                record['reaped'] = True
            except Exception as error:
                record['cleanup_errors'].append({'stage': 'leader_reap', 'error': repr(error)})
            # Never signal after wait: PID/PGID ownership could now be recycled.
            try:
                record['remaining_owned_members'] = group_members(pgid)
                if record['remaining_owned_members']:
                    record['cleanup_errors'].append({'stage': 'remaining_members', 'error': 'Owned group not proven empty'})
            except Exception as error:
                record['cleanup_errors'].append({'stage': 'after_reap', 'error': repr(error)})
        else:
            record['remaining_owned_members'] = []
        for fd in (read_fd, write_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    record['cleanup_errors'].append({'stage': 'pipe_close', 'error': repr(error)})
        try:
            # Reparse the complete immutable-after-cleanup log, including last-line
            # failure telemetry emitted immediately before a zero/nonzero exit.
            record['log_sha256'] = file_pin(log_path)['sha256']
            telemetry = Telemetry(phase, limits['mlx_stop_bytes'], limits['question_seconds'])
            with log_path.open('rb') as reader:
                for chunk in iter(lambda: reader.read(MAX_LOG_CHUNK), b''):
                    telemetry.feed(chunk)
                telemetry.feed(b'', final=True)
            if record['stop_reason'] == 'worker_exit':
                if telemetry.failure:
                    record['stop_reason'] = telemetry.failure
                elif record['exit_code'] != 0:
                    record['stop_reason'] = 'worker_nonzero_exit'
                elif phase != 'audit' and not telemetry.samples:
                    record['stop_reason'] = 'missing_mlx_telemetry'
                elif not telemetry.complete:
                    record['stop_reason'] = 'missing_worker_completion'
                elif telemetry.active_case is not None:
                    record['stop_reason'] = 'unfinished_case'
                elif phase in ('baseline', 'adapter', 'inference') and not telemetry.case_events:
                    record['stop_reason'] = 'missing_case_deadlines'
            if worker_receipt.exists():
                record['worker_receipt'] = file_pin(worker_receipt)
                worker_record = json.loads(worker_receipt.read_bytes())
                expected_schema = ('trid-offline-mlx-inference-receipt-v1' if phase == 'inference'
                                   else 'trid-local-mlx-receipt-v2')
                if record['stop_reason'] == 'worker_exit' and (
                        worker_record.get('schema') != expected_schema
                        or worker_record.get('phase') != phase
                        or worker_record.get('status') != 'complete'
                        or worker_record.get('plan_sha256') != record['plan_sha256']):
                    record['stop_reason'] = 'invalid_worker_receipt'
            elif record['stop_reason'] == 'worker_exit':
                record['stop_reason'] = 'missing_worker_receipt'
        except Exception as error:
            record['telemetry_error'] = repr(error)
            if record['stop_reason'] == 'worker_exit':
                record['stop_reason'] = 'telemetry_or_receipt_error'
        record['mlx_samples'] = telemetry.samples
        record['case_events'] = telemetry.case_events
        record['active_case_at_stop'] = telemetry.active_case
        for key in MLX_KEYS:
            record[key] = max((sample[key] for sample in telemetry.samples), default=None)
        record['hard_alarm_fired'] = hard_fired
        record['owner_stop_signals'] = requested
        record['elapsed_seconds'] = time.monotonic() - started
        record['swap_growth_peak_bytes'] = (None if record['swap_peak_bytes'] is None else
                                            max(0, record['swap_peak_bytes'] - record['swap_baseline_bytes']))
        record['cleanup_state'] = 'failed_or_unproven' if record['cleanup_errors'] else 'completed'
        record['success'] = (record['stop_reason'] == 'worker_exit' and record['exit_code'] == 0
                             and record['reaped'] and record['startup_released']
                             and record['cleanup_state'] == 'completed')
        try:
            save()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--phase', required=True, choices=tuple(PHASE_SECONDS))
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        record = supervise(args.plan, args.phase, args.receipt, args.log, command)
    except Exception as error:
        print(json.dumps({'success': False, 'error': str(error)}), flush=True)
        return 1
    print(json.dumps({'success': record['success'], 'phase': args.phase,
                      'stop_reason': record['stop_reason'], 'receipt': args.receipt}), flush=True)
    return 0 if record['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
