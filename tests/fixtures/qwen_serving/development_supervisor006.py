"""Absolute-deadline API revision of pinned supervisor14cad.
Only supervise() is exposed: no inherited UI CLI, app imports or model runtime.
RSS remains a sampled stop threshold; probe I/O adds to the0.25s sleep interval.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

def supervise(argv, receipt, *, absolute_deadline_unix, duration, pressure_required=1, env=None):
    """Parent may exercise this exact boundary with a benign child, never a model.

    Child inherits TRID_UI_START_FD and must await one G byte before work.
    Dedicated group is owned before that byte. Keep leader unreaped until final
    group kill, preventing its PID/PGID recycling while the watchdog owns it.
    """
    if type(absolute_deadline_unix) not in (int, float) or not math.isfinite(absolute_deadline_unix):
        raise ValueError('Explicit finite absolute deadline required')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Explicit positive duration cap required')
    receipt = Path(receipt)
    if receipt.exists():
        raise RuntimeError('Never overwrite a supervisor receipt')
    record = {'supervisor_pid': os.getpid(), 'argv': argv, 'group_samples': [], 'pressure': [],
              'duration_seconds': duration, 'absolute_deadline_unix': absolute_deadline_unix,
              'startup_released': False, 'signals': [], 'reaped': False,
              'first_soft_stop_reason': None, 'owner_stop_signals': []}
    child = None
    hard_at = None
    pgid = None
    requested = record['owner_stop_signals']  # Existing handler appends actual TERM/INT.
    hard_fired = False
    term_at = None
    started = None
    read_fd, write_fd = os.pipe()
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

    def save():
        temporary = receipt.with_suffix(receipt.suffix + '.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        temporary.chmod(0o600)
        temporary.replace(receipt)

    def kill_group(sig):
        if pgid is not None:
            attempt = {'signal': signal.Signals(sig).name,
                       'elapsed': None if started is None else time.monotonic() - started}
            record.setdefault('kill_attempts', []).append(attempt)
            try:
                os.killpg(pgid, sig)
                attempt['syscall_result'] = 'returned_success'
            except ProcessLookupError:
                attempt['syscall_result'] = 'group_not_found'
            except PermissionError as error:
                # Do not inspect ps inside the alarm handler. Final cleanup must
                # classify actual members, persist evidence, and fail if live/unknown.
                attempt.update(syscall_result='permission_denied', errno=error.errno)

    def request(signum, _frame):
        requested.append(signum)  # Never raise across Popen ownership acquisition.

    def hard_stop(_signum, _frame):
        nonlocal hard_fired, hard_at
        hard_fired = True
        hard_at = time.monotonic()
        kill_group(signal.SIGKILL)  # No I/O, locks or application cooperation.

    def members():
        result = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,stat=,rss='],
                                capture_output=True, text=True, check=True, timeout=1)
        rows = []
        for line in result.stdout.splitlines():
            pid, group, status, rss = line.split()
            if int(group) == pgid:
                rows.append({'pid': int(pid), 'pgid': int(group), 'status': status, 'rss_bytes': int(rss) * 1024})
        return rows

    try:
        signal.signal(signal.SIGTERM, request)
        signal.signal(signal.SIGINT, request)
        signal.signal(signal.SIGALRM, hard_stop)
        child_env = dict(os.environ if env is None else env)
        child_env['TRID_UI_SUPERVISOR_PID'] = str(os.getpid())
        child_env['TRID_UI_START_FD'] = str(read_fd)
        child = subprocess.Popen(argv, env=child_env, start_new_session=True, pass_fds=(read_fd,))
        pgid = child.pid
        os.close(read_fd)
        read_fd = None
        if os.getpgid(child.pid) != pgid:
            raise RuntimeError('Dedicated harness group not established')
        initial_pressure = subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                                          capture_output=True, text=True, check=True, timeout=1)
        if int(initial_pressure.stdout.strip()) != pressure_required:
            raise RuntimeError('Pressure gate before releasing startup')
        record.update(harness_pid=child.pid, owned_pgid=pgid)
        # Compute remaining time HERE, after spawn and actual initial pressure I/O.
        # Never reuse a caller estimate or add guessed startup/cleanup slack.
        deadline_unix = min(absolute_deadline_unix, time.time() + duration)
        started = time.monotonic()
        arm_unix = time.time()
        remaining = deadline_unix - arm_unix
        if remaining <= 0:
            record.update(timer_arm_unix=arm_unix, absolute_remaining_at_arm=absolute_deadline_unix-arm_unix,
                          stop_reason='absolute_deadline_expired_before_G')
            raise RuntimeError('Absolute deadline expired before startup release')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        duration = remaining
        record.update(timer_arm_unix=arm_unix, absolute_remaining_at_arm=absolute_deadline_unix-arm_unix,
                      started_unix=arm_unix, armed_duration_seconds=remaining,
                      planned_hard_deadline_unix=deadline_unix)
        save()  # PID receipt exists before the child can import the application.
        print(f'GENERATIVE_SUPERVISOR_OWNED supervisor={os.getpid()} harness={child.pid} pgid={pgid}', flush=True)
        if requested:
            raise RuntimeError('Supervisor stop received before startup release')
        if hard_fired or time.time() >= absolute_deadline_unix:
            record['stop_reason'] = 'absolute_deadline_expired_before_G'
            raise RuntimeError('Absolute deadline expired before startup release')
        os.write(write_fd, b'G')
        record['startup_released'] = True
        os.close(write_fd)
        write_fd = None
        while True:
            elapsed = time.monotonic() - started
            if time.time() >= absolute_deadline_unix and not hard_fired:
                hard_stop(signal.SIGALRM, None)
            rows = members()
            record['group_samples'].append({'elapsed': elapsed, 'members': rows})
            leader = next((r for r in rows if r['pid'] == child.pid), None)
            if hard_fired:
                record['stop_reason'] = 'external_hard_deadline_or_grace_expiry'
                break
            if leader is None or leader['status'].startswith('Z'):
                record['stop_reason'] = 'harness_exit'
                break
            pressure = subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                                      capture_output=True, text=True, check=True, timeout=1)
            level = int(pressure.stdout.strip())
            record['pressure'].append({'elapsed': time.monotonic() - started, 'value': level})
            reason = ('external_stop_signal' if requested else
                      'external_pressure_not_1' if level != pressure_required else
                      'external_6GiB_RSS_ceiling' if sum(r['rss_bytes'] for r in rows) > 6 * 1024**3 else
                      'external_348_second_cleanup' if time.monotonic() - started >= duration - 2 else None)
            if reason and term_at is None:
                record['first_soft_stop_reason'] = reason
                record['stop_reason'] = reason
                term_at = time.monotonic()
                kill_group(signal.SIGTERM)
                record['signals'].append({'signal': 'SIGTERM', 'elapsed': term_at - started})
                # Earlier pressure/owner stop receives <=2s grace; never extend350.
                remaining = min(2, duration - (time.monotonic() - started),
                                absolute_deadline_unix - time.time())
                if remaining <= 0:
                    hard_stop(signal.SIGALRM, None)
                else:
                    signal.setitimer(signal.ITIMER_REAL, remaining)
            save()
            time.sleep(0.25)
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        # Keep the leader unreaped through the last group signal. Record syscall
        # outcomes separately: a final EPERM says nothing about the earlier alarm.
        cleanup_errors = []
        try:
            kill_group(signal.SIGKILL)
        except Exception as error:
            cleanup_errors.append({'stage': 'final_kill', 'error': repr(error)})
        signal.setitimer(signal.ITIMER_REAL, 0)
        if pgid is not None:
            try:
                rows = members()
                record['members_before_reap'] = rows
                live = [row for row in rows if not row['status'].startswith('Z')]
                denied = [item for item in record.get('kill_attempts', []) if item['syscall_result'] == 'permission_denied']
                if denied:
                    record['permission_boundary'] = {
                        'post_cleanup_state': 'live_members' if live else 'zombie_only' if rows else 'empty_group',
                        'live_members': live,
                        'interpretation': 'Post-cleanup observation only; does not retroactively prove an earlier signal succeeded.'}
                    if live:
                        cleanup_errors.append({'stage': 'permission_boundary', 'error': 'EPERM with live group members; cleanup not proven'})
            except Exception as error:
                record['group_observation_error'] = repr(error)
                cleanup_errors.append({'stage': 'group_observation', 'error': repr(error)})
        if child is not None:
            try:
                record['harness_exit_code'] = child.wait(timeout=3)
                record['reaped'] = True
            except Exception as error:
                cleanup_errors.append({'stage': 'leader_reap', 'error': repr(error)})
        # Never signal this PGID again after reaping: it could be recycled.
        for fd in (read_fd, write_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    cleanup_errors.append({'stage': 'pipe_close', 'error': repr(error)})
        record['hard_alarm_fired'] = hard_fired
        record['elapsed_seconds'] = None if started is None else time.monotonic() - started
        record['hard_kill_elapsed_seconds'] = None if hard_at is None or started is None else hard_at - started
        record['cleanup_errors'] = cleanup_errors
        record['cleanup_state'] = 'failed_or_unproven' if cleanup_errors else 'completed'
        try:
            save()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if cleanup_errors:
            raise RuntimeError('Owned cleanup failed or unproven; final receipt preserved')
    return record

