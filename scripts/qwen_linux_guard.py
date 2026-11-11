"""Linux-only, fail-closed owned-process watchdog; no model imports or retries.

RSS is a sampled process-group sum, not a hard allocation limit. Run this guard
and its child in a dedicated systemd cgroup with MemoryMax=6G,
MemorySwapMax=0 and RuntimeMaxSec, plus an independent instance lifetime limit.
The cgroup limits are checked, not installed here. Optional --gpu-uuid monitoring
also bounds whole-device memory and rejects foreign compute users. All group
signals precede leader reap.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

GIB = 1024 ** 3
RSS_LIMIT = 6 * GIB
ADMISSION_AVAILABLE = 8 * GIB
RUNTIME_AVAILABLE = 6 * GIB
MAX_SECONDS = 900
GRACE_SECONDS = 2.0
SAMPLE_SECONDS = 0.25
GPU_MEMORY_LIMIT = 6 * GIB
GPU_MONITOR_SECONDS = 1.0
GPU_IDENTITY_FIELDS = ('uuid', 'name', 'pci_bus_id', 'total_bytes', 'driver_version')
DIAGNOSTIC_FILE_BYTES = 16 * 1024
DIAGNOSTIC_ANCESTORS = 4
DIAGNOSTIC_CGROUP_DEPTH = 64
MEMORY_STAT_FIELDS = (
    'anon', 'file', 'kernel', 'kernel_stack', 'pagetables', 'sock', 'shmem',
    'file_dirty', 'file_writeback', 'slab_reclaimable', 'slab_unreclaimable',
    'inactive_anon', 'active_anon', 'inactive_file', 'active_file', 'unevictable',
    'pgfault', 'pgmajfault', 'pgscan', 'pgsteal', 'pgscan_kswapd', 'pgscan_direct',
    'pgsteal_kswapd', 'pgsteal_direct', 'workingset_refault_anon', 'workingset_refault_file')
MEMORY_EVENT_FIELDS = ('low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill')
VMSTAT_FIELDS = (
    'pgscan_kswapd', 'pgscan_direct', 'pgscan_direct_throttle',
    'pgsteal_kswapd', 'pgsteal_direct',
    'allocstall_dma', 'allocstall_dma32', 'allocstall_normal',
    'allocstall_high', 'allocstall_movable',
    'compact_stall', 'compact_fail', 'compact_success',
    'compact_daemon_wake', 'compact_migrate_scanned', 'compact_free_scanned',
    'pswpin', 'pswpout', 'oom_kill')
BOOTSTRAP = """import os, sys
fd = int(sys.argv[1])
released = os.read(fd, 1)
os.close(fd)
if released != b'G':
    sys.exit(125)
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


def unsigned(value):
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError('Expected an unsigned decimal integer')
    return int(value)


def validate_seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError('seconds must be finite and in (0, 900]')
    return float(value)


def parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        key, separator, raw = line.partition(':')
        if key not in ('MemTotal', 'MemAvailable'):
            continue
        parts = raw.split()
        if not separator or key in values or len(parts) != 2 or parts[1] != 'kB':
            raise ValueError('Malformed or duplicate memory measurement')
        values[key] = unsigned(parts[0]) * 1024
    if set(values) != {'MemTotal', 'MemAvailable'}:
        raise ValueError('Missing MemTotal or MemAvailable')
    if not 0 < values['MemTotal'] or values['MemAvailable'] > values['MemTotal']:
        raise ValueError('Impossible memory measurement')
    return values


def parse_pressure(text):
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ('some', 'full') or parts[0] in result:
            raise ValueError('Malformed or duplicate memory PSI row')
        row = {}
        for part in parts[1:]:
            key, separator, value = part.partition('=')
            if not separator or key in row or key not in ('avg10', 'avg60', 'avg300', 'total'):
                raise ValueError('Malformed PSI field')
            if key == 'total':
                row[key] = unsigned(value)
            else:
                number = float(value)
                if not math.isfinite(number) or not 0 <= number <= 100:
                    raise ValueError('Invalid PSI average')
                row[key] = number
        if set(row) != {'avg10', 'avg60', 'avg300', 'total'}:
            raise ValueError('Missing PSI measurement')
        result[parts[0]] = row
    if set(result) != {'some', 'full'}:
        raise ValueError('Missing full or some PSI row')
    return result


def host_metrics():
    return {'memory_bytes': parse_meminfo(Path('/proc/meminfo').read_text()),
            'memory_psi': parse_pressure(Path('/proc/pressure/memory').read_text())}


def parse_stat(text, page_size):
    prefix, separator, suffix = text.rpartition(')')
    pid_text, opener, _name = prefix.partition(' (')
    fields = suffix.split()
    if not separator or not opener or len(fields) < 22 or page_size <= 0:
        raise ValueError('Malformed proc stat')
    if len(fields[0]) != 1 or fields[0] not in 'RSDZTtXxKWPI':
        raise ValueError('Unknown process state')
    return {'pid': unsigned(pid_text), 'state': fields[0],
            'ppid': unsigned(fields[1]), 'pgid': unsigned(fields[2]),
            'start_ticks': unsigned(fields[19]),
            'rss_bytes': unsigned(fields[21]) * page_size}


def process_stat(pid):
    row = parse_stat(Path(f'/proc/{pid}/stat').read_text(), os.sysconf('SC_PAGE_SIZE'))
    if row['pid'] != pid:
        raise ValueError('Process identity mismatch')
    return row


def group_members(pgid):
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isascii() or not entry.name.isdecimal():
            continue
        try:
            row = process_stat(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue  # A racing exit is not a fabricated zero-RSS measurement.
        if row['pgid'] == pgid:
            rows.append(row)
    return sorted(rows, key=lambda row: row['pid'])


def live_members(rows):
    return [row for row in rows if row['state'] not in ('Z', 'X', 'x')]


def owned_cgroup():
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    unified = [line[3:] for line in lines if line.startswith('0::')]
    if len(unified) != 1 or not unified[0].startswith('/'):
        raise ValueError('A dedicated cgroup v2 is required')
    relative = Path(unified[0].lstrip('/'))
    if '..' in relative.parts or str(relative) == '.':
        raise ValueError('Refusing the root or an ambiguous cgroup')
    root = Path('/sys/fs/cgroup') / relative
    return root


def cgroup_metrics(root):
    maximum = unsigned((root / 'memory.max').read_text().strip())
    swap_maximum = unsigned((root / 'memory.swap.max').read_text().strip())
    swap_current = unsigned((root / 'memory.swap.current').read_text().strip())
    events = {}
    for line in (root / 'memory.events').read_text().splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] in events:
            raise ValueError('Malformed cgroup memory events')
        events[parts[0]] = unsigned(parts[1])
    if not {'oom', 'oom_kill'} <= events.keys():
        raise ValueError('Missing cgroup OOM counters')
    if not 0 < maximum <= RSS_LIMIT or swap_maximum != 0 or swap_current != 0:
        raise ValueError('Required 6 GiB/no-swap cgroup boundary is absent')
    return {'memory_max_bytes': maximum, 'swap_max_bytes': swap_maximum,
            'swap_current_bytes': swap_current, 'events': events}


def parse_diagnostic_counters(text, fields):
    """Retain selected native counters; absent kernel fields are not zeros."""
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in fields:
            continue
        if len(parts) != 2 or parts[0] in values:
            raise ValueError('Malformed or duplicate diagnostic counter')
        values[parts[0]] = unsigned(parts[1])
    return {'values': values, 'missing': [field for field in fields if field not in values]}


def diagnostic_read(path, parser, *, dir_fd=None, cancel_requested=None):
    """Optional evidence only; never wrap a mandatory guard read with this."""
    try:
        if cancel_requested is not None and cancel_requested():
            return {'unavailable': {'reason': 'collection_cancelled'}}
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(path, flags, dir_fd=dir_fd), 'rb', buffering=0) as stream:
            if cancel_requested is not None and cancel_requested():
                return {'unavailable': {'reason': 'collection_cancelled'}}
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return {'unavailable': {'reason': 'not_regular_file'}}
            raw = stream.read(DIAGNOSTIC_FILE_BYTES + 1)
        if len(raw) > DIAGNOSTIC_FILE_BYTES:
            return {'unavailable': {'reason': 'file_size_limit'}}
        return {'value': parser(raw.decode('ascii'))}
    except (OSError, ValueError) as error:
        # No raw file contents, process argv or arbitrary exception messages.
        return {'unavailable': {'reason': 'read_or_parse_error',
                                'type': type(error).__name__,
                                'errno': getattr(error, 'errno', None)}}


def pressure_diagnostics(root, *, cgroup_mount=Path('/sys/fs/cgroup'),
                         vmstat_path=Path('/proc/vmstat'), cancel_requested=None):
    """Read-only, no-model pressure evidence; not an admission or health check.

    At most 47 file reads, each capped at 16 KiB plus one overflow byte:
    nine files in the own cgroup and each of four nearest visible ancestors,
    own memory.stat, and host vmstat. No enumeration, subprocesses or retries.
    At most 65 cgroup directory opens anchor traversal at the declared mount;
    symlinks at/below that mount and file leaves are never followed. Trusted
    parents above the declared mount may be canonical aliases (e.g. /tmp).
    Visible mount root is included when within the cap; hidden namespace
    ancestors cannot be observed. Missing files/fields and omitted ancestors
    are explicit. All windows use the host monotonic clock and are non-atomic.
    memory.stat values retain native units (byte gauges and event counters).
    An optional cancel_requested() predicate stops remaining optional I/O;
    it does not change mandatory guard reads or install a timer.
    """
    started = time.monotonic()
    root, cgroup_mount = Path(root), Path(cgroup_mount)
    result = {'started_monotonic': started, 'cgroup_mount': str(cgroup_mount),
              'ancestor_scope': 'visible_mount_only', 'ancestors': []}

    def cancelled():
        return cancel_requested is not None and cancel_requested()

    def limit(text):
        value = text.strip()
        return 'max' if value == 'max' else unsigned(value)

    def scope(path, directory_fd):
        observation = {'path': str(path), 'started_monotonic': time.monotonic()}
        for name, parser in (
                ('memory.current', lambda text: unsigned(text.strip())),
                ('memory.peak', lambda text: unsigned(text.strip())),
                ('memory.high', limit), ('memory.max', limit),
                ('memory.swap.max', limit),
                ('memory.swap.current', lambda text: unsigned(text.strip())),
                ('memory.events', lambda text: parse_diagnostic_counters(text, MEMORY_EVENT_FIELDS)),
                ('memory.events.local', lambda text: parse_diagnostic_counters(text, MEMORY_EVENT_FIELDS)),
                ('memory.pressure', parse_pressure)):
            if cancelled():
                observation['unavailable'] = {'reason': 'collection_cancelled'}
                break
            observation[name] = diagnostic_read(
                name, parser, dir_fd=directory_fd, cancel_requested=cancel_requested)
        if path == root and not cancelled():
            observation['memory.stat'] = diagnostic_read(
                'memory.stat', lambda text: parse_diagnostic_counters(text, MEMORY_STAT_FIELDS),
                dir_fd=directory_fd, cancel_requested=cancel_requested)
        if cancelled():
            observation['unavailable'] = {'reason': 'collection_cancelled'}
        observation['finished_monotonic'] = time.monotonic()
        return observation

    if (not root.is_absolute() or not cgroup_mount.is_absolute()
            or '..' in root.parts or '..' in cgroup_mount.parts
            or not root.is_relative_to(cgroup_mount)):
        result['unavailable'] = {'reason': 'cgroup_outside_visible_mount'}
    else:
        components = root.relative_to(cgroup_mount).parts
        depth = len(components)
        result['omitted_visible_ancestors'] = depth
        directory_fds = []
        try:
            if depth > DIAGNOSTIC_CGROUP_DEPTH:
                result['unavailable'] = {'reason': 'cgroup_depth_limit'}
            elif not cancelled():
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                directory_fds.append(os.open(cgroup_mount, flags))
                for component in components:
                    if cancelled():
                        break
                    directory_fds.append(os.open(component, flags, dir_fd=directory_fds[-1]))
                if not cancelled():
                    result['own_cgroup'] = scope(root, directory_fds[-1])
                    ancestor = root
                    for index in range(min(depth, DIAGNOSTIC_ANCESTORS)):
                        if cancelled():
                            break
                        ancestor = ancestor.parent
                        result['ancestors'].append(scope(ancestor, directory_fds[-2 - index]))
                    result['omitted_visible_ancestors'] = depth - len(result['ancestors'])
        except OSError as error:
            result['unavailable'] = {'reason': 'cgroup_directory_unavailable',
                                     'type': type(error).__name__, 'errno': error.errno}
        finally:
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)
    if cancelled():
        result['unavailable'] = {'reason': 'collection_cancelled'}
        result['host_vmstat'] = {'unavailable': {'reason': 'collection_cancelled'}}
    else:
        result['host_vmstat'] = {
            'started_monotonic': time.monotonic(),
            **diagnostic_read(Path(vmstat_path),
                              lambda text: parse_diagnostic_counters(text, VMSTAT_FIELDS),
                              cancel_requested=cancel_requested),
            'finished_monotonic': time.monotonic()}
    result['finished_monotonic'] = time.monotonic()
    return result


def resource_reason(metrics, baseline, rss, admission=False):
    available = metrics['memory_bytes']['MemAvailable']
    if available < (ADMISSION_AVAILABLE if admission else RUNTIME_AVAILABLE):
        return 'admission_mem_available_below_8GiB' if admission else 'mem_available_below_6GiB'
    full = metrics['memory_psi']['full']['total']
    if baseline is not None:
        previous = baseline['memory_psi']['full']['total']
        if full < previous:
            return 'memory_psi_full_counter_regressed'
        if full > previous:
            return 'memory_psi_full_increased'
    if rss >= RSS_LIMIT:
        return 'group_rss_at_6GiB'
    return None


def validate_gpu_uuid(value):
    if not isinstance(value, str) or re.fullmatch(
            r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value) is None:
        raise ValueError('A full explicit NVIDIA GPU UUID is required')
    return value


def gpu_identity(metrics):
    return {key: metrics[key] for key in GPU_IDENTITY_FIELDS}


def validate_gpu_metrics(metrics, expected_uuid):
    validate_gpu_uuid(expected_uuid)
    if not isinstance(metrics, dict) or set(metrics) != {
            *GPU_IDENTITY_FIELDS, 'used_bytes', 'compute_processes'}:
        raise ValueError('Missing or unknown GPU measurements')
    if metrics['uuid'] != expected_uuid:
        raise ValueError('GPU UUID mismatch')
    for key in ('name', 'driver_version'):
        value = metrics[key]
        if (not isinstance(value, str) or not value.strip() or value != value.strip()
                or any(ord(char) < 32 for char in value)
                or value.casefold() in ('n/a', '[n/a]', 'unknown', 'not supported', '[not supported]')):
            raise ValueError('Unavailable GPU identity')
    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', metrics['driver_version']) is None:
        raise ValueError('Malformed NVIDIA driver version')
    if re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',
                    str(metrics['pci_bus_id'])) is None:
        raise ValueError('Invalid GPU PCI identity')
    if (type(metrics['total_bytes']) is not int or type(metrics['used_bytes']) is not int
            or not 0 <= metrics['used_bytes'] <= metrics['total_bytes']
            or metrics['total_bytes'] <= 0):
        raise ValueError('Invalid whole-device GPU memory measurement')
    processes = metrics['compute_processes']
    if not isinstance(processes, list):
        raise ValueError('Missing GPU compute process measurement')
    seen = set()
    for row in processes:
        if (not isinstance(row, dict) or set(row) != {'uuid', 'pid'}
                or row['uuid'] != expected_uuid or type(row['pid']) is not int
                or row['pid'] <= 0 or row['pid'] in seen):
            raise ValueError('Invalid or duplicate GPU compute process')
        seen.add(row['pid'])


def gpu_metrics(expected_uuid, *, timeout_seconds=GPU_MONITOR_SECONDS):
    """Read whole-device MiB and compute users within one shared time budget."""
    validate_gpu_uuid(expected_uuid)
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= GPU_MONITOR_SECONDS):
        raise ValueError('GPU monitor timeout must be in (0, 1] seconds')
    deadline = time.monotonic() + timeout_seconds

    def query(option):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('GPU monitor deadline exceeded')
        result = subprocess.run(
            ['nvidia-smi', f'--id={expected_uuid}', option, '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True, timeout=remaining, shell=False)
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU monitor deadline exceeded')
        if result.stderr.strip():
            raise ValueError('GPU monitor reported an error: ' + result.stderr.strip())
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True, strict=True))

    rows = query('--query-gpu=uuid,name,pci.bus_id,memory.total,memory.used,driver_version')
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError('Exactly one complete GPU device row is required')
    uuid, name, pci, total, used, driver = (value.strip() for value in rows[0])
    metrics = {'uuid': uuid, 'name': name, 'pci_bus_id': pci,
               'total_bytes': unsigned(total) * 1024 ** 2,
               'used_bytes': unsigned(used) * 1024 ** 2,
               'driver_version': driver, 'compute_processes': []}
    for row in query('--query-compute-apps=gpu_uuid,pid'):
        if len(row) != 2:
            raise ValueError('Malformed GPU compute process row')
        metrics['compute_processes'].append({'uuid': row[0].strip(), 'pid': unsigned(row[1].strip())})
    validate_gpu_metrics(metrics, expected_uuid)
    return metrics


def gpu_resource_reason(metrics, baseline, members, *, admission=False):
    validate_gpu_metrics(metrics, baseline['uuid'])
    if gpu_identity(metrics) != gpu_identity(baseline):
        return 'gpu_identity_changed'
    if metrics['used_bytes'] >= GPU_MEMORY_LIMIT:
        return 'gpu_memory_at_6GiB'
    pids = {row['pid'] for row in members}
    if metrics['compute_processes'] and admission:
        return 'gpu_compute_users_before_exec'
    if any(row['pid'] not in pids for row in metrics['compute_processes']):
        return 'gpu_foreign_compute_user'
    return None


def resolve_gpu_ownership(metrics, members, observations, pgid):
    """Use fresh PID identity, retaining pre-probe ownership only after an exit."""
    if not isinstance(observations, list):
        raise ValueError('Missing post-probe GPU ownership evidence')
    expected = {row['pid'] for row in metrics['compute_processes']}
    before = {row['pid']: row for row in members}
    seen = set()
    owned = []
    for observation in observations:
        if (not isinstance(observation, dict) or set(observation) != {'pid', 'after'}
                or type(observation['pid']) is not int or observation['pid'] not in expected
                or observation['pid'] in seen):
            raise ValueError('Malformed post-probe GPU process observation')
        pid = observation['pid']
        seen.add(pid)
        previous = before.get(pid)
        after = observation['after']
        for row in (previous, after):
            if row is not None and (
                    not isinstance(row, dict) or row.get('pid') != pid
                    or type(row.get('pid')) is not int
                    or type(row.get('pgid')) is not int or row['pgid'] <= 0
                    or type(row.get('start_ticks')) is not int or row['start_ticks'] <= 0):
                raise ValueError('Invalid GPU process identity evidence')
        if after is not None:
            # A reused PID cannot inherit its predecessor's ownership, even if
            # the replacement happens to join the same owned group.
            if previous is not None and after['start_ticks'] != previous['start_ticks']:
                continue
            candidate = after
        else:
            candidate = previous
        if candidate is not None and candidate['pgid'] == pgid:
            owned.append(candidate)
    if seen != expected:
        raise ValueError('Incomplete post-probe GPU process observations')
    return owned


def validate_gpu_receipt(record, expected_uuid):
    """Validate successful GPU evidence; callers still validate host/argv bindings."""
    validate_gpu_uuid(expected_uuid)
    try:
        if (not clean_exit(record) or type(record['guard_exit_code']) is not int
                or record['guard_exit_code'] != 0 or record['cleanup_errors']
                or record['startup_released'] is not True or record['reaped'] is not True):
            raise ValueError('GPU guard did not complete successfully')
        gpu = record['gpu']
        if (gpu['schema'] != 'trid-gpu-guard-v1' or gpu['expected_uuid'] != expected_uuid
                or type(gpu['memory_limit_bytes']) is not int
                or gpu['memory_limit_bytes'] != GPU_MEMORY_LIMIT):
            raise ValueError('GPU receipt contract mismatch')
        samples = record['samples']
        if not isinstance(samples, list) or len(samples) < 3:
            raise ValueError('GPU admission, pre-exec and runtime evidence required')
        if type(record['owned_pgid']) is not int or record['owned_pgid'] <= 0:
            raise ValueError('GPU process group identity missing')
        seconds = validate_seconds(record['seconds'])
        previous = -1.0
        peak = 0
        baseline = samples[0]['gpu']
        for index, sample in enumerate(samples):
            phase = ('admission', 'pre_exec')[index] if index < 2 else 'runtime'
            if sample['gpu_phase'] != phase:
                raise ValueError('GPU observation phase mismatch')
            elapsed = sample['elapsed_seconds']
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                    or not 0 <= elapsed < seconds or elapsed < previous):
                raise ValueError('GPU observation time invalid')
            previous = elapsed
            measurement = sample['gpu']
            validate_gpu_metrics(measurement, expected_uuid)
            if gpu_identity(measurement) != gpu['identity']:
                raise ValueError('GPU receipt identity mismatch')
            members = sample['members']
            if not isinstance(members, list):
                raise ValueError('GPU ownership evidence missing')
            if index == 0 and members:
                raise ValueError('Unexpected process group before GPU admission')
            seen = set()
            for row in members:
                if (type(row['pid']) is not int or row['pid'] <= 0 or row['pid'] in seen
                        or type(row['pgid']) is not int or row['pgid'] != record['owned_pgid']):
                    raise ValueError('GPU process group ownership mismatch')
                seen.add(row['pid'])
            owned = resolve_gpu_ownership(
                measurement, members, sample['gpu_process_observations'], record['owned_pgid'])
            reason = gpu_resource_reason(measurement, baseline, owned, admission=index < 2)
            if reason:
                raise ValueError(reason)
            peak = max(peak, measurement['used_bytes'])
        if type(gpu['used_peak_bytes']) is not int or gpu['used_peak_bytes'] != peak:
            raise ValueError('GPU peak evidence mismatch')
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError('Incomplete GPU guard receipt') from error


class Guard:
    def __init__(self, argv, receipt, seconds, *, gpu_uuid=None):
        self.seconds = validate_seconds(seconds)
        self.gpu_uuid = validate_gpu_uuid(gpu_uuid) if gpu_uuid is not None else None
        if not argv or any(not isinstance(arg, str) or '\0' in arg for arg in argv) or not argv[0]:
            raise ValueError('Nonempty literal child argv required')
        self.argv = list(argv)
        self.receipt = Path(receipt)
        self.started = time.monotonic()
        self.deadline = self.started + self.seconds
        self.child = None
        self.pgid = None
        self.term_at = None
        self.hard_killed = False
        self.read_fd = None
        self.write_fd = None
        self.lock_fd = None
        self.group_signals_enabled = True
        self.record = {'argv': self.argv, 'supervisor_pid': os.getpid(),
                       'seconds': self.seconds, 'started_unix': time.time(),
                       'started_monotonic': self.started,
                       'startup_released': False, 'first_stop_reason': None,
                       'stop_reason': None, 'child_exit_code': None, 'reaped': False,
                       'rss_peak_bytes': 0, 'samples': [], 'signals': [],
                       'pressure_diagnostics': [],
                       'cleanup_errors': [], 'cleanup_state': 'not_started'}
        if self.gpu_uuid is not None:
            self.record['gpu'] = {'schema': 'trid-gpu-guard-v1', 'expected_uuid': self.gpu_uuid,
                                  'memory_limit_bytes': GPU_MEMORY_LIMIT,
                                  'identity': None, 'used_peak_bytes': 0}

    def save(self):
        temporary = self.receipt.with_name(self.receipt.name + f'.{os.getpid()}.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(self.record, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            os.replace(temporary, self.receipt)
        finally:
            temporary.unlink(missing_ok=True)

    def signal_group(self, signum):
        if self.pgid is None or self.record['reaped'] or not self.group_signals_enabled:
            return
        attempt = {'signal': signal.Signals(signum).name,
                   'elapsed_seconds': time.monotonic() - self.started}
        self.record['signals'].append(attempt)
        try:
            os.killpg(self.pgid, signum)
            attempt['result'] = 'sent'
        except ProcessLookupError:
            attempt['result'] = 'group_not_found'
        except OSError as error:
            attempt.update(result='error', error=repr(error))
            self.record['cleanup_errors'].append({'stage': 'signal', 'error': repr(error)})

    def stop(self, reason):
        if self.record['first_stop_reason'] is None:
            self.record['first_stop_reason'] = reason
            self.record['stop_reason'] = reason
        if self.term_at is None:
            self.term_at = time.monotonic()
            self.signal_group(signal.SIGTERM)

    def request_stop(self, signum, _frame):
        self.stop(f'owner_signal_{signal.Signals(signum).name}')

    def hard_deadline(self, _signum, _frame):
        self.stop('deadline')
        self.hard_killed = True
        self.signal_group(signal.SIGKILL)

    def owner_alive(self, owner):
        if os.getppid() != owner['pid']:
            return False
        try:
            current = process_stat(owner['pid'])
        except (FileNotFoundError, ProcessLookupError):
            return False
        return current['start_ticks'] == owner['start_ticks'] and bool(live_members([current]))

    def sample(self, cgroup, *, phase=None):
        # Diagnostics precede mandatory observations, never the rejection/stop
        # path. They may precede a pressure event; do not infer causation or
        # atomicity from a same-sample cgroup/host/GPU comparison.
        diagnostics = pressure_diagnostics(
            cgroup, cancel_requested=lambda: self.record['first_stop_reason'] is not None)
        host_started = time.monotonic()
        metrics = host_metrics()
        host_finished = time.monotonic()
        rows = group_members(self.pgid) if self.pgid is not None else []
        rss = sum(row['rss_bytes'] for row in rows)
        self.record['rss_peak_bytes'] = max(self.record['rss_peak_bytes'], rss)
        cgroup_started = time.monotonic()
        boundary = cgroup_metrics(cgroup)  # Still mandatory and fail-closed.
        cgroup_finished = time.monotonic()
        sample = {'elapsed_seconds': cgroup_started - self.started,
                  'host': metrics, 'members': rows, 'rss_bytes': rss,
                  'cgroup': boundary,
                  'observation_times': {
                      'host': {'started_monotonic': host_started, 'finished_monotonic': host_finished},
                      'cgroup': {'started_monotonic': cgroup_started,
                                 'finished_monotonic': cgroup_finished}}}
        self.record['samples'].append(sample)
        # Keep admission and only the latest two diagnostic observations.
        # Mandatory sample history is unchanged; indices link retained windows.
        history = self.record['pressure_diagnostics']
        history.append({'sample_index': len(self.record['samples']) - 1,
                        'observation': diagnostics})
        if len(history) > 3:
            del history[1]
        if self.gpu_uuid is not None:
            sample['gpu_phase'] = phase
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Guard deadline reached before GPU observation')
            gpu_started = time.monotonic()
            measurement = gpu_metrics(
                self.gpu_uuid, timeout_seconds=min(GPU_MONITOR_SECONDS, remaining))
            validate_gpu_metrics(measurement, self.gpu_uuid)
            sample['gpu'] = measurement
            observations = []
            sample['gpu_process_observations'] = observations
            for process in measurement['compute_processes']:
                try:
                    after = process_stat(process['pid'])
                except (FileNotFoundError, ProcessLookupError):
                    after = None
                observations.append({'pid': process['pid'], 'after': after})
            sample['elapsed_seconds'] = time.monotonic() - self.started
            sample['observation_times']['gpu'] = {
                'started_monotonic': gpu_started,
                'finished_monotonic': self.started + sample['elapsed_seconds']}
            gpu = self.record['gpu']
            if gpu['identity'] is None:
                gpu['identity'] = gpu_identity(measurement)
            gpu['used_peak_bytes'] = max(gpu['used_peak_bytes'], measurement['used_bytes'])
        return sample

    def cleanup(self):
        # Never poll()/wait() before the final group signal: the unreaped leader
        # reserves its PID/PGID even when it exits before its descendants.
        if self.child is not None:
            if self.term_at is None:
                self.term_at = time.monotonic()
                self.signal_group(signal.SIGTERM)
            grace_end = min(self.deadline, self.term_at + GRACE_SECONDS)
            try:
                while time.monotonic() < grace_end and not self.hard_killed:
                    rows = group_members(self.pgid)
                    if not live_members(rows):
                        break
                    time.sleep(min(SAMPLE_SECONDS, max(0, grace_end - time.monotonic())))
            except Exception as error:
                self.record['cleanup_errors'].append({'stage': 'grace_observation', 'error': repr(error)})
            self.signal_group(signal.SIGKILL)
            # Disable the callback as well as its timer: a pending SIGALRM must
            # not race wait() and signal a recycled PGID after the kernel reap.
            self.group_signals_enabled = False
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                observation_end = time.monotonic() + GRACE_SECONDS
                while True:
                    rows = group_members(self.pgid)
                    self.record['members_before_reap'] = rows
                    self.record['live_members_before_reap'] = live_members(rows)
                    if not live_members(rows) or time.monotonic() >= observation_end:
                        break
                    time.sleep(SAMPLE_SECONDS)
                if live_members(rows):
                    self.record['cleanup_errors'].append({'stage': 'group_observation', 'error': 'Live group members remain'})
            except Exception as error:
                self.record['cleanup_errors'].append({'stage': 'group_observation', 'error': repr(error)})
            try:
                self.record['child_exit_code'] = self.child.wait(timeout=GRACE_SECONDS)
                self.record['reaped'] = True
                self.pgid = None  # Never signal a potentially recycled group.
            except Exception as error:
                self.record['cleanup_errors'].append({'stage': 'leader_reap', 'error': repr(error)})
        else:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.record['members_before_reap'] = []
            self.record['live_members_before_reap'] = []
        for name in ('read_fd', 'write_fd', 'lock_fd'):
            fd = getattr(self, name)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    self.record['cleanup_errors'].append({'stage': name, 'error': repr(error)})
                setattr(self, name, None)
        self.record['cleanup_state'] = 'failed_or_unproven' if self.record['cleanup_errors'] else 'completed'
        self.record['elapsed_seconds'] = time.monotonic() - self.started

    def run(self):
        if sys.platform != 'linux':
            raise RuntimeError('Linux is required; no platform fallback')
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        handlers = {s: signal.getsignal(s) for s in (
            signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGCHLD)}
        try:
            # SIG_IGN survives exec on Linux and auto-reaps children; an inherited
            # custom handler may also reap them. Own SIGCHLD before spawning so
            # the leader reserves its PID/PGID until our explicit final wait().
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGALRM, self.hard_deadline)
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, self.deadline - time.monotonic()))
            # A nonblocking per-UID lock prohibits overlapping guarded workloads.
            # Inherit it through exec so supervisor death cannot unlock a live child.
            self.lock_fd = os.open(f'/tmp/qwen-linux-guard-{os.getuid()}.lock',
                                   os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = process_stat(os.getppid())
            self.record['owner'] = owner
            cgroup = owned_cgroup()
            self.record['cgroup_path'] = str(cgroup)
            initial = self.sample(cgroup, phase='admission')
            baseline = initial['host']
            reason = resource_reason(baseline, None, 0, admission=True)
            if self.gpu_uuid is not None:
                reason = reason or gpu_resource_reason(initial['gpu'], initial['gpu'], [], admission=True)
            if not self.owner_alive(owner):
                reason = 'owner_not_alive'
            if reason:
                self.stop(reason)
            if time.monotonic() >= self.deadline:
                self.stop('deadline_before_exec')
            if self.record['first_stop_reason'] is None:
                self.read_fd, self.write_fd = os.pipe()
                os.set_inheritable(self.lock_fd, True)
                self.child = subprocess.Popen(
                    [sys.executable, '-I', '-B', '-c', BOOTSTRAP, str(self.read_fd), *self.argv],
                    start_new_session=True, pass_fds=(self.read_fd, self.lock_fd), shell=False)
                self.pgid = self.child.pid
                self.record.update(child_pid=self.child.pid, owned_pgid=self.pgid)
                os.close(self.read_fd)
                self.read_fd = None
                if os.getpgid(self.child.pid) != self.pgid:
                    raise RuntimeError('Dedicated child process group not established')
                ready = self.sample(cgroup, phase='pre_exec')
                reason = resource_reason(ready['host'], baseline, ready['rss_bytes'], admission=True)
                if self.gpu_uuid is not None:
                    reason = reason or gpu_resource_reason(
                        ready['gpu'], initial['gpu'], ready['members'], admission=True)
                if not self.owner_alive(owner):
                    reason = 'owner_not_alive'
                if ready['cgroup']['events'] != initial['cgroup']['events']:
                    reason = 'cgroup_memory_events_changed_before_exec'
                if reason:
                    self.stop(reason)
                self.save()  # Ownership evidence exists before application exec.
                if time.monotonic() >= self.deadline:
                    self.stop('deadline_before_exec')
                if self.record['first_stop_reason'] is None:
                    os.write(self.write_fd, b'G')
                    self.record['startup_released'] = True
                os.close(self.write_fd)
                self.write_fd = None
                while self.record['first_stop_reason'] is None:
                    current = self.sample(cgroup, phase='runtime')
                    reason = resource_reason(current['host'], baseline, current['rss_bytes'])
                    if self.gpu_uuid is not None:
                        reason = reason or gpu_resource_reason(
                            current['gpu'], initial['gpu'], resolve_gpu_ownership(
                                current['gpu'], current['members'],
                                current['gpu_process_observations'], self.pgid))
                    events = current['cgroup']['events']
                    if any(events[key] != initial['cgroup']['events'][key] for key in ('oom', 'oom_kill')):
                        reason = reason or 'cgroup_oom_counter_changed'
                    if not self.owner_alive(owner):
                        reason = reason or 'owner_not_alive'
                    if time.monotonic() >= self.deadline - min(GRACE_SECONDS, self.seconds / 2):
                        reason = reason or 'deadline'
                    if reason:
                        self.stop(reason)
                        break
                    leader = next((row for row in current['members'] if row['pid'] == self.child.pid), None)
                    if leader is None:
                        self.stop('leader_identity_missing')
                        break
                    if not live_members([leader]):
                        if live_members(current['members']):
                            self.stop('leader_exited_with_live_descendants')
                        else:
                            self.record['stop_reason'] = 'child_exit'
                        break
                    time.sleep(SAMPLE_SECONDS)
        except BaseException as error:
            self.record['error'] = {'type': type(error).__name__, 'message': str(error)}
            self.stop('guard_error')
        finally:
            try:
                self.cleanup()
                self.save()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                # SIGCHLD is restored last, after cleanup disables all group
                # signals and attempts the explicit leader reap.
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
        self.record['guard_exit_code'] = 0 if clean_exit(self.record) else 1
        self.save()
        return self.record


def clean_exit(record):
    return (record['startup_released'] and record['stop_reason'] == 'child_exit'
            and record['first_stop_reason'] is None and record['child_exit_code'] == 0
            and record['reaped'] and record['cleanup_state'] == 'completed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--gpu-uuid', help='Require exclusive, monitored use of this NVIDIA GPU UUID')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        guard = Guard(command, args.receipt, args.seconds, gpu_uuid=args.gpu_uuid)
    except ValueError as error:
        parser.error(str(error))
    record = guard.run()
    print(json.dumps({key: record.get(key) for key in (
        'guard_exit_code', 'stop_reason', 'first_stop_reason', 'child_exit_code',
        'reaped', 'rss_peak_bytes', 'cleanup_state', 'cleanup_errors')}))
    return record['guard_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
