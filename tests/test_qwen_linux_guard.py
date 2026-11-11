"""Stdlib-only watchdog contract tests; all process/signalling boundaries are mocked.

These tests do not establish real Linux process-group, cgroup or PSI behavior.
That requires the separately authorized Linux smoke under the external limits.
"""
from collections import deque
from contextlib import ExitStack
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import qwen_linux_guard as guard


def metrics(available=8 * guard.GIB, full=100):
    return {'memory_bytes': {'MemTotal': 32 * guard.GIB, 'MemAvailable': available},
            'memory_psi': {kind: {'avg10': 0.0, 'avg60': 0.0, 'avg300': 0.0, 'total': full}
                           for kind in ('some', 'full')}}


def cgroup_metrics():
    return {'memory_max_bytes': guard.RSS_LIMIT, 'swap_max_bytes': 0,
            'swap_current_bytes': 0, 'events': {'low': 0, 'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}


def member(pid=123, state='S', rss=guard.GIB):
    return {'pid': pid, 'ppid': 1, 'pgid': 123, 'state': state,
            'start_ticks': 1000 + pid, 'rss_bytes': rss}


GPU_UUID = 'GPU-12345678-abcd-1234-abcd-123456789abc'
OTHER_GPU_UUID = 'GPU-87654321-abcd-1234-abcd-123456789abc'


def gpu_measurement(used=256 * 1024 ** 2, pids=(), **identity):
    return {'uuid': GPU_UUID, 'name': 'NVIDIA L4', 'pci_bus_id': '00000000:00:1E.0',
            'total_bytes': 23034 * 1024 ** 2, 'used_bytes': used,
            'driver_version': '570.195.03',
            'compute_processes': [{'uuid': GPU_UUID, 'pid': pid} for pid in pids], **identity}


def write_pressure_fixture(root, *, full=0, current=123, high='max'):
    root.mkdir(parents=True, exist_ok=True)
    files = {
        'memory.current': str(current), 'memory.peak': str(current + 10),
        'memory.high': high, 'memory.max': str(guard.RSS_LIMIT),
        'memory.swap.max': '0', 'memory.swap.current': '0',
        'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n',
        'memory.events.local': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n',
        'memory.pressure': (
            f'some avg10=0.00 avg60=0.00 avg300=0.00 total={full + 1}\n'
            f'full avg10=0.00 avg60=0.00 avg300=0.00 total={full}\n'),
        'memory.stat': 'anon 100\nfile 23\npgscan_direct 7\npgsteal_direct 5\n'}
    for name, value in files.items():
        (root / name).write_text(value)


class PressureDiagnosticTests(unittest.TestCase):
    def test_selected_counters_retain_units_and_explicitly_missing_fields(self):
        result = guard.parse_diagnostic_counters(
            'anon 100\npgscan_direct 7\nfuture_counter 999\n', guard.MEMORY_STAT_FIELDS)
        self.assertEqual(result['values'], {'anon': 100, 'pgscan_direct': 7})
        self.assertIn('file', result['missing'])
        self.assertNotIn('anon', result['missing'])
        self.assertNotIn('future_counter', result['values'])
        for text in ('anon -1', 'anon +1', 'anon １', 'anon 1.5',
                     'anon 1\nanon 2', 'anon', 'anon 1 extra'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                guard.parse_diagnostic_counters(text, guard.MEMORY_STAT_FIELDS)

    def test_collector_distinguishes_own_ancestor_and_host_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / 'cgroup'
            own = mount / 'slice' / 'job'
            write_pressure_fixture(own, full=10, current=123)
            write_pressure_fixture(own.parent, full=80, current=456, high='1024')
            (own.parent / 'memory.events').write_text('high 4\noom 0\noom_kill 0\n')
            vmstat = Path(directory) / 'vmstat'
            vmstat.write_text('pgscan_direct 91\ncompact_stall 3\npswpout 0\nnr_free_pages 999\n')
            result = guard.pressure_diagnostics(own, cgroup_mount=mount, vmstat_path=vmstat)
        local = result['own_cgroup']
        ancestor, visible_root = result['ancestors']
        self.assertEqual(local['memory.current']['value'], 123)
        self.assertEqual(local['memory.peak']['value'], 133)
        self.assertEqual(local['memory.high']['value'], 'max')
        self.assertEqual(local['memory.pressure']['value']['full']['total'], 10)
        self.assertEqual(local['memory.stat']['value']['values']['pgscan_direct'], 7)
        self.assertEqual(ancestor['memory.current']['value'], 456)
        self.assertEqual(ancestor['memory.high']['value'], 1024)
        self.assertEqual(ancestor['memory.pressure']['value']['full']['total'], 80)
        self.assertEqual(ancestor['memory.events']['value']['values']['high'], 4)
        self.assertEqual(ancestor['memory.events.local']['value']['values']['high'], 0)
        self.assertEqual(visible_root['path'], str(mount))
        self.assertIn('unavailable', visible_root['memory.current'])
        self.assertEqual(result['omitted_visible_ancestors'], 0)
        self.assertEqual(result['ancestor_scope'], 'visible_mount_only')
        self.assertEqual(result['host_vmstat']['value']['values'],
                         {'pgscan_direct': 91, 'compact_stall': 3, 'pswpout': 0})
        self.assertIn('compact_success', result['host_vmstat']['value']['missing'])
        previous = result['started_monotonic']
        for observation in (local, ancestor, visible_root, result['host_vmstat']):
            self.assertLessEqual(previous, observation['started_monotonic'])
            self.assertLessEqual(observation['started_monotonic'], observation['finished_monotonic'])
            previous = observation['finished_monotonic']
        self.assertLessEqual(previous, result['finished_monotonic'])

    def test_file_size_boundary_and_unavailable_data_are_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'counter'
            parser = lambda text: guard.unsigned(text.strip())
            path.write_bytes(b'7' + b' ' * (guard.DIAGNOSTIC_FILE_BYTES - 1))
            self.assertEqual(guard.diagnostic_read(path, parser), {'value': 7})
            path.write_bytes(b'7' + b' ' * guard.DIAGNOSTIC_FILE_BYTES)
            self.assertEqual(guard.diagnostic_read(path, parser)['unavailable']['reason'],
                             'file_size_limit')
            for raw in (b'', b'-1', b'\xff'):
                path.write_bytes(raw)
                result = guard.diagnostic_read(path, parser)
                self.assertNotIn('value', result)
                self.assertEqual(result['unavailable']['reason'], 'read_or_parse_error')
            path.unlink()
            self.assertEqual(guard.diagnostic_read(path, parser)['unavailable']['type'],
                             'FileNotFoundError')
            with patch.object(guard.os, 'open', side_effect=PermissionError(13, 'denied')):
                result = guard.diagnostic_read(path, parser)
            self.assertEqual(result['unavailable']['type'], 'PermissionError')
            self.assertEqual(result['unavailable']['errno'], 13)

    def test_oversized_file_consumes_only_limit_plus_overflow_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'counter'
            path.write_bytes(b'7' + b' ' * (guard.DIAGNOSTIC_FILE_BYTES * 2))
            original_open = os.open
            observers = []

            def open_with_observer(*args, **kwargs):
                descriptor = original_open(*args, **kwargs)
                observers.append(os.dup(descriptor))
                return descriptor

            try:
                with patch.object(guard.os, 'open', side_effect=open_with_observer):
                    result = guard.diagnostic_read(path, lambda text: guard.unsigned(text.strip()))
                self.assertEqual(result['unavailable']['reason'], 'file_size_limit')
                self.assertEqual(len(observers), 1)
                # A duplicate shares the kernel file offset, exposing any
                # buffered read-ahead even after the collector closes its FD.
                self.assertEqual(os.lseek(observers[0], 0, os.SEEK_CUR),
                                 guard.DIAGNOSTIC_FILE_BYTES + 1)
            finally:
                for observer in observers:
                    os.close(observer)

    def test_ancestor_cap_root_boundary_and_outside_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / 'cgroup'
            own = mount / 'a' / 'b' / 'c' / 'd' / 'e' / 'f'
            own.mkdir(parents=True)
            vmstat = Path(directory) / 'missing_vmstat'
            result = guard.pressure_diagnostics(own, cgroup_mount=mount, vmstat_path=vmstat)
            self.assertEqual([row['path'] for row in result['ancestors']],
                             [str(own.parents[index]) for index in range(4)])
            self.assertEqual(result['omitted_visible_ancestors'], 2)
            self.assertIn('unavailable', result['host_vmstat'])
            at_root = guard.pressure_diagnostics(mount, cgroup_mount=mount, vmstat_path=vmstat)
            self.assertEqual(at_root['ancestors'], [])
            self.assertEqual(at_root['omitted_visible_ancestors'], 0)
            for outside in (mount.parent, mount / '..' / 'elsewhere', Path('relative'),
                            Path(str(mount) + '-sibling')):
                with self.subTest(outside=outside):
                    result = guard.pressure_diagnostics(
                        outside, cgroup_mount=mount, vmstat_path=vmstat)
                    self.assertEqual(result['unavailable']['reason'], 'cgroup_outside_visible_mount')
                    self.assertNotIn('own_cgroup', result)
                    self.assertEqual(result['ancestors'], [])

    def test_directory_and_leaf_symlinks_cannot_escape_cgroup_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mount, outside = base / 'cgroup', base / 'outside'
            mount.mkdir()
            write_pressure_fixture(outside / 'job', current=987)
            (mount / 'linked_slice').symlink_to(outside, target_is_directory=True)
            vmstat = base / 'missing_vmstat'
            escaped = guard.pressure_diagnostics(
                mount / 'linked_slice' / 'job', cgroup_mount=mount, vmstat_path=vmstat)
            self.assertEqual(escaped['unavailable']['reason'], 'cgroup_directory_unavailable')
            self.assertNotIn('own_cgroup', escaped)
            self.assertEqual(escaped['ancestors'], [])

            own = mount / 'job'
            write_pressure_fixture(own)
            (own / 'memory.current').unlink()
            (own / 'memory.current').symlink_to(outside / 'job' / 'memory.current')
            result = guard.pressure_diagnostics(own, cgroup_mount=mount, vmstat_path=vmstat)
            self.assertNotIn('value', result['own_cgroup']['memory.current'])
            self.assertIn('unavailable', result['own_cgroup']['memory.current'])
            self.assertEqual(result['own_cgroup']['memory.peak']['value'], 133)

            linked_mount = base / 'linked_mount'
            linked_mount.symlink_to(mount, target_is_directory=True)
            result = guard.pressure_diagnostics(
                linked_mount / 'job', cgroup_mount=linked_mount, vmstat_path=vmstat)
            self.assertEqual(result['unavailable']['reason'], 'cgroup_directory_unavailable')
            self.assertNotIn('own_cgroup', result)

    def test_parent_alias_above_declared_mount_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mount = base / 'real_parent' / 'cgroup'
            write_pressure_fixture(mount / 'job')
            alias = base / 'parent_alias'
            alias.symlink_to(mount.parent, target_is_directory=True)
            declared_mount = alias / 'cgroup'
            result = guard.pressure_diagnostics(
                declared_mount / 'job', cgroup_mount=declared_mount,
                vmstat_path=base / 'missing_vmstat')
            self.assertEqual(result['own_cgroup']['memory.current']['value'], 123)
            self.assertEqual(result['ancestors'][0]['path'], str(declared_mount))

    def test_directory_traversal_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / 'cgroup'
            own = mount.joinpath(*(['level'] * guard.DIAGNOSTIC_CGROUP_DEPTH))
            write_pressure_fixture(own)
            vmstat = Path(directory) / 'missing_vmstat'
            accepted = guard.pressure_diagnostics(own, cgroup_mount=mount, vmstat_path=vmstat)
            self.assertEqual(accepted['own_cgroup']['memory.current']['value'], 123)
            rejected = guard.pressure_diagnostics(
                own / 'too_deep', cgroup_mount=mount, vmstat_path=vmstat)
            self.assertEqual(rejected['unavailable']['reason'], 'cgroup_depth_limit')
            self.assertNotIn('own_cgroup', rejected)

    def test_already_cancelled_collection_opens_nothing(self):
        with patch.object(guard.os, 'open') as opened:
            result = guard.pressure_diagnostics(
                Path('/sys/fs/cgroup/job'), cancel_requested=lambda: True)
        opened.assert_not_called()
        self.assertEqual(result['unavailable']['reason'], 'collection_cancelled')
        self.assertEqual(result['host_vmstat']['unavailable']['reason'], 'collection_cancelled')


class GpuParsingTests(unittest.TestCase):
    DEVICE = f'{GPU_UUID}, NVIDIA L4, 00000000:00:1E.0, 23034, 256, 570.195.03\n'

    def observe(self, device=None, processes=''):
        outputs = [subprocess.CompletedProcess([], 0, self.DEVICE if device is None else device, ''),
                   subprocess.CompletedProcess([], 0, processes, '')]
        with patch.object(guard.subprocess, 'run', side_effect=outputs):
            return guard.gpu_metrics(GPU_UUID)

    def test_whole_device_units_and_compute_pid_parsing(self):
        self.assertEqual(self.observe(), gpu_measurement())
        self.assertEqual(self.observe(processes=f'{GPU_UUID}, 123\n{GPU_UUID}, 124\n'),
                         gpu_measurement(pids=(123, 124)))

    def test_missing_extra_malformed_and_unknown_device_rows_fail_closed(self):
        for text in ('', '\n', self.DEVICE * 2, self.DEVICE + '\n',
                     self.DEVICE.replace('23034', '[N/A]'),
                     self.DEVICE.replace(', 256,', ', -1,'),
                     self.DEVICE.replace(', 256,', ', 23035,'),
                     self.DEVICE.replace(', 256,', ', 1.5,'),
                     self.DEVICE.replace('NVIDIA L4', '[Not Supported]'),
                     self.DEVICE.replace('570.195.03', 'N/A'),
                     self.DEVICE.replace('00000000:00:1E.0', 'unknown'),
                     self.DEVICE.replace(GPU_UUID, OTHER_GPU_UUID),
                     self.DEVICE.replace(', 570.195.03', ''),
                     self.DEVICE.replace(', 570.195.03', ', 570.195.03, extra')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.observe(device=text)

    def test_unknown_duplicate_foreign_and_invalid_compute_rows_fail_closed(self):
        for text in ('No running processes found\n', '[N/A]\n', '\n',
                     f'{OTHER_GPU_UUID}, 123\n', f'{GPU_UUID}, 0\n',
                     f'{GPU_UUID}, N/A\n', f'{GPU_UUID}, -1\n',
                     f'{GPU_UUID}, 123, extra\n', f'{GPU_UUID}, 123\n' * 2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.observe(processes=text)

    def test_monitor_failures_and_diagnostics_are_not_zero_measurements(self):
        for error in (FileNotFoundError('nvidia-smi'), subprocess.TimeoutExpired('nvidia-smi', 1),
                      subprocess.CalledProcessError(1, 'nvidia-smi')):
            with self.subTest(error=error), patch.object(guard.subprocess, 'run', side_effect=error):
                with self.assertRaises(type(error)):
                    guard.gpu_metrics(GPU_UUID)
        with patch.object(guard.subprocess, 'run', return_value=subprocess.CompletedProcess(
                [], 0, self.DEVICE, 'NVML unavailable')):
            with self.assertRaises(ValueError):
                guard.gpu_metrics(GPU_UUID)

    def test_queries_share_one_timeout_budget_and_literal_uuid_selector(self):
        clock = Clock()
        calls = []

        def query(argv, **kwargs):
            calls.append((argv, kwargs))
            clock.now += 0.4
            return subprocess.CompletedProcess(argv, 0, self.DEVICE if len(calls) == 1 else '', '')

        with patch.object(guard.time, 'monotonic', side_effect=clock.monotonic), \
                patch.object(guard.subprocess, 'run', side_effect=query):
            self.assertEqual(guard.gpu_metrics(GPU_UUID), gpu_measurement())
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[0][1]['timeout'], 1.0)
        self.assertAlmostEqual(calls[1][1]['timeout'], 0.6)
        for argv, kwargs in calls:
            self.assertEqual(argv[0], 'nvidia-smi')
            self.assertIn(f'--id={GPU_UUID}', argv)
            self.assertIn('--format=csv,noheader,nounits', argv)
            self.assertFalse(kwargs['shell'])
            self.assertTrue(kwargs['check'])

    def test_expired_shared_budget_does_not_issue_second_query(self):
        with patch.object(guard.time, 'monotonic', side_effect=[0, 0, 1.01]), \
                patch.object(guard.subprocess, 'run', return_value=subprocess.CompletedProcess(
                    [], 0, self.DEVICE, '')) as query:
            with self.assertRaises(TimeoutError):
                guard.gpu_metrics(GPU_UUID)
        query.assert_called_once()

    def test_invalid_uuid_or_unbounded_timeout_never_launches_monitor(self):
        with patch.object(guard.subprocess, 'run') as query:
            for uuid in ('', '0', 'GPU-short', GPU_UUID + ',0', None):
                with self.subTest(uuid=uuid), self.assertRaises(ValueError):
                    guard.gpu_metrics(uuid)
            for timeout in (0, -1, 1.001, math.inf, math.nan, True, '1'):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    guard.gpu_metrics(GPU_UUID, timeout_seconds=timeout)
        query.assert_not_called()

    def test_memory_boundary_and_owned_descendants_are_independent_of_rss(self):
        baseline = gpu_measurement()
        self.assertIsNone(guard.gpu_resource_reason(
            gpu_measurement(guard.GPU_MEMORY_LIMIT - 1, pids=(123, 124)),
            baseline, [member(), member(124)]))
        self.assertEqual(guard.gpu_resource_reason(
            gpu_measurement(guard.GPU_MEMORY_LIMIT), baseline, []), 'gpu_memory_at_6GiB')
        self.assertEqual(guard.gpu_resource_reason(
            gpu_measurement(pids=(124,)), baseline, [member()]), 'gpu_foreign_compute_user')
        self.assertEqual(guard.gpu_resource_reason(
            gpu_measurement(pids=(123,)), baseline, [member()], admission=True),
            'gpu_compute_users_before_exec')

    def test_each_nonvolatile_identity_change_is_rejected(self):
        for key, value in (('name', 'NVIDIA A10G'), ('driver_version', '580.1'),
                           ('pci_bus_id', '00000000:00:1F.0'), ('total_bytes', 24 * guard.GIB)):
            with self.subTest(key=key):
                self.assertEqual(guard.gpu_resource_reason(
                    gpu_measurement(**{key: value}), gpu_measurement(), []), 'gpu_identity_changed')

    def test_cli_gpu_mode_is_explicit_and_cpu_remains_default(self):
        for uuid in (None, GPU_UUID):
            with self.subTest(uuid=uuid), patch.object(guard, 'Guard') as constructor, \
                    patch('builtins.print'):
                constructor.return_value.run.return_value = {'guard_exit_code': 0}
                argv = ['--receipt', 'receipt.json', '--seconds', '180']
                if uuid is not None:
                    argv += ['--gpu-uuid', uuid]
                self.assertEqual(guard.main(argv + ['--', 'python', 'job.py']), 0)
                constructor.assert_called_once_with(
                    ['python', 'job.py'], Path('receipt.json'), 180.0, gpu_uuid=uuid)


class ParsingTests(unittest.TestCase):
    def test_duration_boundaries_and_nonfinite_values(self):
        for value in (0.001, 180, 900):
            self.assertEqual(guard.validate_seconds(value), float(value))
        for value in (0, -1, 900.001, math.inf, -math.inf, math.nan, True, '180', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                guard.validate_seconds(value)

    def test_memory_units_completeness_and_numeric_boundaries(self):
        self.assertEqual(guard.parse_meminfo('MemTotal: 9 kB\nMemAvailable: 8 kB\nSwapFree: 0 kB'),
                         {'MemTotal': 9216, 'MemAvailable': 8192})
        self.assertEqual(guard.parse_meminfo('MemTotal: 1 kB\nMemAvailable: 0 kB')['MemAvailable'], 0)
        invalid = ('MemTotal: 9 kB', 'MemTotal: 9 kB\nMemAvailable: 10 kB',
                   'MemTotal: 0 kB\nMemAvailable: 0 kB', 'MemTotal: 9 kB\nMemAvailable: -1 kB',
                   'MemTotal: 9 kB\nMemAvailable: 8 MB', 'MemTotal: 9 kB\nMemAvailable: 1.5 kB',
                   'MemTotal: 9 kB\nMemAvailable: +1 kB', 'MemTotal: 9 kB\nMemAvailable: １ kB',
                   'MemTotal: 9 kB\nMemAvailable: 1 kB\nMemAvailable: 1 kB')
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                guard.parse_meminfo(text)

    def test_pressure_requires_native_full_and_some_counters(self):
        text = ('some avg10=0.00 avg60=100.00 avg300=0.15 total=18446744073709551615\n'
                'full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n')
        parsed = guard.parse_pressure(text)
        self.assertEqual(parsed['some']['total'], 18446744073709551615)
        self.assertEqual(parsed['full']['total'], 0)
        for invalid in ('', text.splitlines()[0], text + text.splitlines()[1],
                        text.replace('total=0', 'total=-1'), text.replace('total=0', 'total=1.0'),
                        text.replace('avg10=0.00', 'avg10=nan'),
                        text.replace('avg10=0.00', 'avg10=inf'),
                        text.replace('avg10=0.00', 'avg10=100.01'),
                        text.replace('avg10=0.00', 'avg10=-0.1'),
                        text.replace('avg10=0.00', 'avg10=0 avg10=0'),
                        text.replace('avg10=0.00 ', '')):
            with self.subTest(text=invalid), self.assertRaises(ValueError):
                guard.parse_pressure(invalid)

    def test_proc_stat_handles_spaces_and_closing_parentheses_in_name(self):
        fields = ['S', '7', '123', '123'] + ['0'] * 15 + ['456', '9999', '10']
        parsed = guard.parse_stat('123 (worker ) with spaces) ' + ' '.join(fields), 4096)
        self.assertEqual(parsed, {'pid': 123, 'state': 'S', 'ppid': 7, 'pgid': 123,
                                  'start_ticks': 456, 'rss_bytes': 40960})
        for text in ('123 no-parentheses S 7', '123 (x) S 7',
                     '123 (x) ' + ' '.join(fields[:-1] + ['-1'])):
            with self.subTest(text=text), self.assertRaises(ValueError):
                guard.parse_stat(text, 4096)

    def test_thresholds_are_bytes_and_pressure_is_any_full_counter_increase(self):
        base = metrics()
        self.assertIsNone(guard.resource_reason(base, None, 0, admission=True))
        self.assertEqual(guard.resource_reason(metrics(8 * guard.GIB - 1), None, 0, True),
                         'admission_mem_available_below_8GiB')
        self.assertIsNone(guard.resource_reason(metrics(6 * guard.GIB), base, guard.RSS_LIMIT - 1))
        self.assertEqual(guard.resource_reason(metrics(6 * guard.GIB - 1), base, 0),
                         'mem_available_below_6GiB')
        self.assertEqual(guard.resource_reason(base, base, guard.RSS_LIMIT), 'group_rss_at_6GiB')
        self.assertEqual(guard.resource_reason(metrics(full=101), base, 0), 'memory_psi_full_increased')
        self.assertEqual(guard.resource_reason(metrics(full=99), base, 0), 'memory_psi_full_counter_regressed')

    def test_cgroup_hard_limit_and_zero_swap_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = {'memory.max': str(guard.RSS_LIMIT), 'memory.swap.max': '0',
                     'memory.swap.current': '0', 'memory.events': 'oom 0\noom_kill 0\n'}
            for name, value in valid.items():
                (root / name).write_text(value)
            self.assertEqual(guard.cgroup_metrics(root)['memory_max_bytes'], guard.RSS_LIMIT)
            for name, value in (('memory.max', 'max'), ('memory.max', '0'),
                                ('memory.max', str(guard.RSS_LIMIT + 1)),
                                ('memory.swap.max', '1'), ('memory.swap.current', '1'),
                                ('memory.events', 'oom 0\n'), ('memory.events', 'oom -1\noom_kill 0')):
                with self.subTest(name=name, value=value):
                    (root / name).write_text(value)
                    with self.assertRaises(ValueError):
                        guard.cgroup_metrics(root)
                    (root / name).write_text(valid[name])
            (root / 'memory.swap.current').unlink()
            with self.assertRaises(FileNotFoundError):
                guard.cgroup_metrics(root)

    def test_group_sample_includes_descendants_and_rejects_unreadable_metrics(self):
        entries = [Path('/proc/123'), Path('/proc/124'), Path('/proc/999'), Path('/proc/self')]
        rows = {123: member(), 124: member(124, rss=2 * guard.GIB),
                999: {**member(999), 'pgid': 999}}
        with patch.object(Path, 'iterdir', return_value=iter(entries)), \
                patch.object(guard, 'process_stat', side_effect=lambda pid: rows[pid]):
            result = guard.group_members(123)
        self.assertEqual([row['pid'] for row in result], [123, 124])
        self.assertEqual(sum(row['rss_bytes'] for row in result), 3 * guard.GIB)
        with patch.object(Path, 'iterdir', return_value=iter(entries)), \
                patch.object(guard, 'process_stat', side_effect=PermissionError('proc denied')):
            with self.assertRaises(PermissionError):
                guard.group_members(123)

    def test_owner_reparenting_reuse_and_zombie_are_not_alive(self):
        supervisor = guard.Guard(['python', 'job.py'], Path('unused-receipt.json'), 180)
        owner = member(42)
        with patch.object(guard.os, 'getppid', return_value=42), \
                patch.object(guard, 'process_stat', return_value=owner) as stat:
            self.assertTrue(supervisor.owner_alive(owner))
            stat.return_value = {**owner, 'start_ticks': owner['start_ticks'] + 1}
            self.assertFalse(supervisor.owner_alive(owner))
            stat.return_value = {**owner, 'state': 'Z'}
            self.assertFalse(supervisor.owner_alive(owner))
            stat.side_effect = FileNotFoundError('owner exited')
            self.assertFalse(supervisor.owner_alive(owner))
        with patch.object(guard.os, 'getppid', return_value=1):
            self.assertFalse(supervisor.owner_alive(owner))

    def test_cgroup_v2_path_must_identify_a_nonroot_boundary(self):
        with patch.object(Path, 'read_text', return_value='0::/system.slice/qwen.service\n'):
            self.assertEqual(guard.owned_cgroup(), Path('/sys/fs/cgroup/system.slice/qwen.service'))
        for text in ('', '1:memory:/job\n', '0::/\n', '0::relative\n',
                     '0::/../job\n', '0::/job\n0::/other\n'):
            with self.subTest(text=text), patch.object(Path, 'read_text', return_value=text):
                with self.assertRaises(ValueError):
                    guard.owned_cgroup()


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.events = []
        self.killed = False
        self.stubborn = False
        self.group_rows = deque([[member()], [member(state='Z', rss=0)]])
        self.child = Mock(pid=123)
        self.child.wait.side_effect = self.reap
        self.host = self.mock(guard, 'host_metrics', return_value=metrics())
        self.real_cgroup_metrics = guard.cgroup_metrics
        self.cgroup = self.mock(guard, 'cgroup_metrics', return_value=cgroup_metrics())
        self.diagnostic_mount = self.root / 'cgroup'
        self.diagnostic_root = self.diagnostic_mount / 'owned.service'
        self.diagnostic_root.mkdir(parents=True)
        collector = guard.pressure_diagnostics
        self.mock(guard, 'pressure_diagnostics', side_effect=lambda _root, **kwargs: collector(
            self.diagnostic_root, cgroup_mount=self.diagnostic_mount,
            vmstat_path=self.root / 'missing_vmstat', **kwargs))
        self.mock(guard, 'owned_cgroup', return_value=Path('/sys/fs/cgroup/owned.service'))
        self.mock(guard, 'process_stat', return_value=member(os.getppid()))
        self.mock(guard, 'group_members', side_effect=self.members)
        self.owner = self.mock(guard.Guard, 'owner_alive', return_value=True)
        self.spawn = self.mock(guard.subprocess, 'Popen', return_value=self.child)
        self.mock(guard.os, 'getpgid', return_value=123)
        self.mock(guard.os, 'killpg', side_effect=self.kill)
        self.release = self.mock(guard.os, 'write', side_effect=self.release_child)
        self.mock(guard.signal, 'signal')
        self.mock(guard.signal, 'getsignal', return_value=signal.SIG_DFL)
        self.timer = self.mock(guard.signal, 'setitimer')
        self.mock(guard.time, 'monotonic', side_effect=self.clock.monotonic)
        self.mock(guard.time, 'sleep', side_effect=self.clock.sleep)
        self.stack.enter_context(patch.object(guard.sys, 'platform', 'linux'))
        original_open = os.open

        def open_local_lock(path, flags, mode=0o777, **kwargs):
            if str(path) == f'/tmp/qwen-linux-guard-{os.getuid()}.lock':
                path = self.root / 'lock'
            return original_open(path, flags, mode, **kwargs)

        self.mock(guard.os, 'open', side_effect=open_local_lock)
        self.supervisor = guard.Guard(['python', '-B', 'job.py', 'literal;not-shell'], self.root / 'receipt.json', 30)

    def mock(self, target, name, **kwargs):
        return self.stack.enter_context(patch.object(target, name, **kwargs))

    def members(self, _pgid):
        if self.killed and not self.stubborn:
            return [member(state='Z', rss=0)]
        if len(self.group_rows) > 1:
            return self.group_rows.popleft()
        return self.group_rows[0]

    def kill(self, pgid, signum):
        self.assertEqual(pgid, 123)
        self.events.append((signal.Signals(signum).name, self.clock.now))
        if signum == signal.SIGKILL:
            self.killed = True

    def reap(self, timeout):
        self.assertEqual(timeout, guard.GRACE_SECONDS)
        self.assertTrue(self.killed, 'Leader must stay unreaped until final SIGKILL')
        self.assertFalse(self.supervisor.group_signals_enabled)
        self.events.append(('reap', self.clock.now))
        return 0

    def release_child(self, _fd, value):
        self.assertEqual(value, b'G')
        saved = json.loads(self.supervisor.receipt.read_text())
        self.assertEqual(saved['owned_pgid'], 123)
        self.assertFalse(saved['startup_released'])
        self.assertGreaterEqual(len(saved['samples']), 2)
        self.events.append(('exec_release', self.clock.now))
        return 1

    def run_guard(self):
        result = self.supervisor.run()
        self.assertEqual(json.loads(self.supervisor.receipt.read_text()), result)
        return result

    def test_clean_exit_has_literal_command_evidence_and_final_kill_before_reap(self):
        result = self.run_guard()
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertTrue(guard.clean_exit(result))
        self.assertEqual(result['argv'], ['python', '-B', 'job.py', 'literal;not-shell'])
        args, kwargs = self.spawn.call_args
        self.assertFalse(kwargs['shell'])
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(args[0][-4:], result['argv'])
        self.assertEqual(len(kwargs['pass_fds']), 2)
        self.assertEqual(result['rss_peak_bytes'], guard.GIB)
        self.assertEqual(result['live_members_before_reap'], [])
        self.assertEqual([event[0] for event in self.events], ['exec_release', 'SIGTERM', 'SIGKILL', 'reap'])
        count = len(self.events)
        self.supervisor.signal_group(signal.SIGKILL)
        self.supervisor.hard_deadline(None, None)
        self.assertEqual(len(self.events), count, 'No delayed callback may signal a reaped PGID')

    def test_missing_host_metrics_rejects_before_spawn(self):
        self.host.side_effect = FileNotFoundError('/proc/pressure/memory')
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.release.assert_not_called()
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['error']['type'], 'FileNotFoundError')
        self.assertFalse(result['startup_released'])

    def test_low_admission_memory_rejects_without_model_process(self):
        self.host.return_value = metrics(8 * guard.GIB - 1)
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.assertEqual(result['first_stop_reason'], 'admission_mem_available_below_8GiB')

    def test_pressure_change_while_bootstrap_waits_blocks_exec(self):
        self.host.side_effect = [metrics(), metrics(full=101)]
        result = self.run_guard()
        self.spawn.assert_called_once()
        self.release.assert_not_called()
        self.assertFalse(result['startup_released'])
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertTrue(result['reaped'])

    def test_runtime_pressure_term_grace_kill_preserves_first_reason(self):
        self.group_rows = deque([[member()]])
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['stop_reason'], 'memory_psi_full_increased')
        term = next(at for name, at in self.events if name == 'SIGTERM')
        kill = next(at for name, at in self.events if name == 'SIGKILL')
        self.assertEqual(kill - term, guard.GRACE_SECONDS)
        self.assertEqual(result['guard_exit_code'], 1)

    def test_actual_663us_host_increase_rejects_with_unchanged_cgroup_pressure(self):
        write_pressure_fixture(self.diagnostic_root, full=0)
        self.host.side_effect = [metrics(full=44507), metrics(full=44507), metrics(full=45170)]
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['samples'][-1]['host']['memory_psi']['full']['total']
                         - result['samples'][0]['host']['memory_psi']['full']['total'], 663)
        self.assertTrue(all(entry['observation']['own_cgroup']
                            ['memory.pressure']['value']['full']['total'] == 0
                            for entry in result['pressure_diagnostics']))
        self.assertEqual(result['samples'][-1]['cgroup']['events'],
                         result['samples'][0]['cgroup']['events'])
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])
        self.assertEqual([event[0] for event in self.events],
                         ['exec_release', 'SIGTERM', 'SIGKILL', 'reap'])

    def test_actual_663us_host_increase_rejects_with_unavailable_cgroup_pressure(self):
        self.host.side_effect = [metrics(full=44507), metrics(full=44507), metrics(full=45170)]
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])
        pressure = result['pressure_diagnostics'][-1]['observation']['own_cgroup']['memory.pressure']
        self.assertNotIn('value', pressure)
        self.assertEqual(pressure['unavailable']['type'], 'FileNotFoundError')

    def test_optional_evidence_cannot_mask_mandatory_swap_rejection(self):
        write_pressure_fixture(self.diagnostic_root)
        (self.diagnostic_root / 'memory.swap.current').write_text('1')
        self.cgroup.side_effect = lambda _root: self.real_cgroup_metrics(self.diagnostic_root)
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertFalse(result['startup_released'])
        self.spawn.assert_not_called()
        self.release.assert_not_called()

    def test_stop_during_first_diagnostic_read_cancels_remaining_optional_io(self):
        write_pressure_fixture(self.diagnostic_root)
        actual_read = guard.diagnostic_read

        def read_then_stop(*args, **kwargs):
            result = actual_read(*args, **kwargs)
            self.supervisor.request_stop(signal.SIGTERM, None)
            return result

        with patch.object(guard, 'diagnostic_read', side_effect=read_then_stop) as read:
            sample = self.supervisor.sample(Path('/unused'), phase='admission')
        read.assert_called_once()
        self.assertEqual(self.supervisor.record['first_stop_reason'], 'owner_signal_SIGTERM')
        diagnostics = self.supervisor.record['pressure_diagnostics'][-1]['observation']
        self.assertEqual(diagnostics['own_cgroup']['memory.current']['value'], 123)
        self.assertNotIn('memory.peak', diagnostics['own_cgroup'])
        self.assertEqual(diagnostics['own_cgroup']['unavailable']['reason'], 'collection_cancelled')
        self.assertEqual(diagnostics['unavailable']['reason'], 'collection_cancelled')
        self.assertEqual(diagnostics['ancestors'], [])
        self.assertEqual(diagnostics['omitted_visible_ancestors'], 1)
        self.assertEqual(diagnostics['host_vmstat']['unavailable']['reason'], 'collection_cancelled')
        self.host.assert_called_once()
        self.cgroup.assert_called_once()
        self.spawn.assert_not_called()
        self.release.assert_not_called()

    def test_pressure_stop_captures_late_cgroup_pressure_after_reap(self):
        write_pressure_fixture(self.diagnostic_root, full=0)
        self.host.side_effect = [metrics(full=48200), metrics(full=48200), metrics(full=49218)]
        self.group_rows = deque([[member()]])
        actual_read = guard.diagnostic_read

        def reap_then_update(timeout):
            code = self.reap(timeout)
            write_pressure_fixture(self.diagnostic_root, full=1018)
            return code

        def observe_read(*args, **kwargs):
            if self.supervisor.record['first_stop_reason'] is not None:
                self.assertTrue(self.supervisor.record['reaped'])
                self.assertIsNone(self.supervisor.pgid)
                self.assertEqual(self.events[-1][0], 'reap')
                saved = json.loads(self.supervisor.receipt.read_text())
                self.assertEqual(saved['cleanup_state'], 'completed')
                self.assertEqual(saved['first_stop_reason'], 'memory_psi_full_increased')
            return actual_read(*args, **kwargs)

        self.child.wait.side_effect = reap_then_update
        with patch.object(guard, 'diagnostic_read', side_effect=observe_read):
            result = self.run_guard()
        post = result['post_cleanup_pressure_diagnostics']
        self.assertEqual(post['observation']['own_cgroup']['memory.pressure']['value']['full']['total'],
                         1018)
        self.assertTrue(all(entry['observation']['own_cgroup']
                            ['memory.pressure']['value']['full']['total'] == 0
                            for entry in result['pressure_diagnostics']))
        self.assertGreaterEqual(post['started_monotonic'], self.events[-1][1])
        self.assertLessEqual(post['finished_monotonic'],
                             result['started_monotonic'] + result['elapsed_seconds'])
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertFalse(guard.clean_exit(result))

    def test_pressure_stop_skips_post_cleanup_io_after_original_deadline(self):
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        self.group_rows = deque([[member()]])

        def reap_at_deadline(timeout):
            code = self.reap(timeout)
            self.clock.now = self.supervisor.deadline
            return code

        self.child.wait.side_effect = reap_at_deadline
        result = self.run_guard()
        post = result['post_cleanup_pressure_diagnostics']
        self.assertEqual(post['unavailable']['reason'], 'collection_deadline')
        self.assertNotIn('observation', post)
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])

    def test_pressure_stop_requires_proven_reap_before_post_cleanup_io(self):
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        self.group_rows = deque([[member()]])
        self.child.wait.side_effect = subprocess.TimeoutExpired('child', guard.GRACE_SECONDS)
        result = self.run_guard()
        post = result['post_cleanup_pressure_diagnostics']
        self.assertEqual(post['unavailable']['reason'], 'cleanup_not_proven')
        self.assertNotIn('observation', post)
        self.assertFalse(result['reaped'])
        self.assertEqual(result['cleanup_state'], 'failed_or_unproven')
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_post_cleanup_collection_error_cannot_mask_pressure_failure(self):
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        self.group_rows = deque([[member()]])
        collect = guard.pressure_diagnostics.side_effect

        def collect_or_fail(*args, **kwargs):
            if self.supervisor.record['reaped']:
                self.clock.now += 0.125
                raise OSError(5, 'SECRET post-cleanup path')
            return collect(*args, **kwargs)

        guard.pressure_diagnostics.side_effect = collect_or_fail
        result = self.run_guard()
        post = result['post_cleanup_pressure_diagnostics']
        self.assertEqual(post['unavailable']['reason'], 'collection_error')
        self.assertEqual(post['unavailable']['type'], 'OSError')
        self.assertNotIn('SECRET', json.dumps(post))
        self.assertEqual(result['cleanup_state'], 'completed')
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertEqual(result['started_monotonic'] + result['elapsed_seconds'],
                         post['finished_monotonic'])
        self.assertEqual(post['finished_monotonic'] - post['started_monotonic'], 0.125)

    def test_post_cleanup_budget_cancels_remaining_reads_without_new_sample(self):
        write_pressure_fixture(self.diagnostic_root)
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        self.group_rows = deque([[member()]])
        actual_read = guard.diagnostic_read

        def read_then_spend_budget(*args, **kwargs):
            value = actual_read(*args, **kwargs)
            if self.supervisor.record['reaped']:
                self.clock.now += guard.SAMPLE_SECONDS
            return value

        with patch.object(guard, 'diagnostic_read', side_effect=read_then_spend_budget):
            result = self.run_guard()
        post = result['post_cleanup_pressure_diagnostics']
        observation = post['observation']
        self.assertEqual(observation['own_cgroup']['memory.current']['value'], 123)
        self.assertNotIn('memory.peak', observation['own_cgroup'])
        self.assertEqual(observation['unavailable']['reason'], 'collection_cancelled')
        self.assertEqual(observation['ancestors'], [])
        self.assertEqual(post['host_memory_psi']['unavailable']['reason'], 'collection_cancelled')
        self.assertEqual(result['samples'][-1]['host']['memory_psi']['full']['total'], 101)
        self.assertEqual(result['samples'][-1]['rss_bytes'], guard.GIB)
        self.assertEqual(result['first_stop_reason'], 'memory_psi_full_increased')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_post_cleanup_host_error_preserves_completed_cgroup_observation(self):
        write_pressure_fixture(self.diagnostic_root, full=1018)
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        self.group_rows = deque([[member()]])
        actual_read = guard.diagnostic_read

        def fail_host_read(path, *args, **kwargs):
            if self.supervisor.record['reaped'] and Path(path) == Path('/proc/pressure/memory'):
                self.clock.now += 0.125
                raise RuntimeError('SECRET host-read error')
            return actual_read(path, *args, **kwargs)

        with patch.object(guard, 'diagnostic_read', side_effect=fail_host_read):
            result = self.run_guard()
        post = result['post_cleanup_pressure_diagnostics']
        self.assertEqual(post['observation']['own_cgroup']['memory.pressure']['value']['full']['total'],
                         1018)
        self.assertEqual(post['host_memory_psi']['unavailable']['type'], 'RuntimeError')
        self.assertNotIn('SECRET', json.dumps(post))
        self.assertLessEqual(post['observation']['finished_monotonic'],
                             post['host_memory_psi']['started_monotonic'])
        self.assertEqual(post['host_memory_psi']['finished_monotonic'], post['finished_monotonic'])
        self.assertEqual(result['started_monotonic'] + result['elapsed_seconds'],
                         post['finished_monotonic'])
        self.assertEqual(result['guard_exit_code'], 1)

    def test_post_cleanup_rejects_absent_or_invalid_cleanup_proof(self):
        for field, value in (
                ('live_members_before_reap', None), ('live_members_before_reap', [member()]),
                ('reaped', 1), ('cleanup_errors', [{'stage': 'leader_reap'}]),
                ('pgid', 123), ('group_signals_enabled', True)):
            with self.subTest(field=field, value=value):
                supervisor = guard.Guard(['job'], self.root / 'unused.json', 30)
                supervisor.record.update(first_stop_reason='memory_psi_full_increased',
                                         cleanup_state='completed', reaped=True, cleanup_errors=[],
                                         live_members_before_reap=[], elapsed_seconds=0)
                supervisor.pgid = None
                supervisor.group_signals_enabled = False
                if field in ('pgid', 'group_signals_enabled'):
                    setattr(supervisor, field, value)
                elif value is None:
                    del supervisor.record[field]
                else:
                    supervisor.record[field] = value
                supervisor.capture_post_cleanup_pressure()
                post = supervisor.record['post_cleanup_pressure_diagnostics']
                self.assertEqual(post['unavailable']['reason'], 'cleanup_not_proven')
                self.assertNotIn('observation', post)

    def test_diagnostic_history_keeps_admission_and_latest_two_only(self):
        for index in range(7):
            write_pressure_fixture(self.diagnostic_root, full=index, current=123 + index)
            self.clock.now += 0.25
            self.supervisor.sample(Path('/unused'), phase='admission' if index == 0 else 'runtime')
            self.assertLessEqual(len(self.supervisor.record['pressure_diagnostics']), 3)
        record = self.supervisor.record
        self.assertEqual(len(record['samples']), 7)
        self.assertTrue(all('pressure_diagnostics' not in sample for sample in record['samples']))
        self.assertEqual([entry['sample_index'] for entry in record['pressure_diagnostics']], [0, 5, 6])
        for entry in record['pressure_diagnostics']:
            index, observation = entry['sample_index'], entry['observation']
            self.assertEqual(observation['own_cgroup']['memory.current']['value'], 123 + index)
            self.assertEqual(observation['own_cgroup']['memory.pressure']['value']['full']['total'],
                             index)
            self.assertLessEqual(observation['finished_monotonic'],
                                 record['samples'][index]['observation_times']['host']['started_monotonic'])

    def test_runtime_rss_sums_entire_owned_group(self):
        self.group_rows = deque([[member()], [member(rss=2 * guard.GIB), member(124, rss=4 * guard.GIB)]])
        result = self.run_guard()
        self.assertEqual(result['rss_peak_bytes'], 6 * guard.GIB)
        self.assertEqual(result['first_stop_reason'], 'group_rss_at_6GiB')

    def test_runtime_owner_disappearance_stops_child(self):
        self.owner.side_effect = [True, True, False]
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'owner_not_alive')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_runtime_missing_metrics_stops_instead_of_treating_as_zero(self):
        self.host.side_effect = [metrics(), metrics(), PermissionError('pressure unreadable')]
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertTrue(result['reaped'])
        self.assertIn('SIGKILL', [name for name, _at in self.events])

    def test_runtime_host_memory_drop_stops_child(self):
        self.host.side_effect = [metrics(), metrics(), metrics(6 * guard.GIB - 1)]
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'mem_available_below_6GiB')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_runtime_cgroup_swap_or_limit_read_failure_stops_child(self):
        self.cgroup.side_effect = [cgroup_metrics(), cgroup_metrics(), ValueError('swap detected')]
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])

    def test_admission_cgroup_failure_prevents_spawn(self):
        self.cgroup.side_effect = ValueError('unlimited memory.max')
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.release.assert_not_called()
        self.assertEqual(result['guard_exit_code'], 1)

    def test_deadline_reserves_grace_and_does_not_retry(self):
        self.supervisor.seconds = 4
        self.supervisor.deadline = self.clock.now + 4
        self.group_rows = deque([[member()]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'deadline')
        self.assertLessEqual(next(at for name, at in self.events if name == 'SIGKILL'), 104.0)
        self.spawn.assert_called_once()
        self.assertEqual(result['guard_exit_code'], 1)

    def test_cgroup_oom_event_change_fails_even_if_leader_exits_zero(self):
        initial = cgroup_metrics()
        failed = copy.deepcopy(initial)
        failed['events']['oom_kill'] = 1
        self.cgroup.side_effect = [initial, initial, failed]
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'cgroup_oom_counter_changed')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_nonzero_child_exit_is_not_success(self):
        self.child.wait.side_effect = lambda timeout: 23
        result = self.run_guard()
        self.assertEqual(result['child_exit_code'], 23)
        self.assertEqual(result['stop_reason'], 'child_exit')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_live_descendants_after_leader_exit_are_not_clean_success(self):
        self.group_rows = deque([[member()], [member(state='Z', rss=0), member(124)]])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'leader_exited_with_live_descendants')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertEqual(result['live_members_before_reap'], [])

    def test_unproven_cleanup_records_live_members_and_fails(self):
        self.stubborn = True
        self.group_rows = deque([[member()]])
        self.host.side_effect = [metrics(), metrics(), metrics(full=101)]
        result = self.run_guard()
        self.assertEqual(result['cleanup_state'], 'failed_or_unproven')
        self.assertEqual(result['live_members_before_reap'][0]['pid'], 123)
        self.assertEqual(result['guard_exit_code'], 1)

    def test_wait_timeout_is_not_reported_as_reaped(self):
        self.child.wait.side_effect = subprocess.TimeoutExpired('child', guard.GRACE_SECONDS)
        result = self.run_guard()
        self.assertFalse(result['reaped'])
        self.assertEqual(result['cleanup_state'], 'failed_or_unproven')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_signal_failure_is_recorded_not_hidden_by_zero_child_exit(self):
        self.mock(guard.os, 'killpg', side_effect=PermissionError('denied'))
        self.child.wait.side_effect = lambda timeout: 0
        result = self.run_guard()
        self.assertEqual(result['cleanup_state'], 'failed_or_unproven')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(all(item['result'] == 'error' for item in result['signals']))

    def test_external_signal_retains_first_reason_across_hard_deadline(self):
        self.supervisor.pgid = 123
        self.supervisor.request_stop(signal.SIGTERM, None)
        self.supervisor.hard_deadline(None, None)
        self.assertEqual(self.supervisor.record['first_stop_reason'], 'owner_signal_SIGTERM')
        self.assertEqual([name for name, _at in self.events], ['SIGTERM', 'SIGKILL'])

    def test_no_group_signal_in_kernel_reap_to_python_assignment_window(self):
        def reap_with_pending_alarm(timeout):
            self.reap(timeout)
            count = len(self.events)
            self.supervisor.hard_deadline(None, None)
            self.assertEqual(len(self.events), count)
            return 0

        self.child.wait.side_effect = reap_with_pending_alarm
        result = self.run_guard()
        self.assertTrue(result['reaped'])
        self.assertEqual(self.events[-1][0], 'reap')

    def assert_sigchld_owned_until_reap(self, inherited):
        dispositions = {signum: signal.SIG_DFL for signum in (
            signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGCHLD)}
        dispositions[signal.SIGCHLD] = inherited

        def set_disposition(signum, handler):
            previous = dispositions[signum]
            if signum == signal.SIGCHLD:
                if handler is signal.SIG_DFL:
                    self.spawn.assert_not_called()
                    self.events.append(('SIGCHLD_default', self.clock.now))
                else:
                    self.assertIs(handler, inherited)
                    self.assertTrue(self.supervisor.record['reaped'])
                    self.assertFalse(self.supervisor.group_signals_enabled)
                    self.assertEqual(self.events[-1][0], 'reap')
                    self.events.append(('SIGCHLD_restored', self.clock.now))
            dispositions[signum] = handler
            return previous

        def spawn_with_owned_disposition(*_args, **_kwargs):
            self.assertIs(dispositions[signal.SIGCHLD], signal.SIG_DFL)
            self.events.append(('spawn', self.clock.now))
            return self.child

        def signal_with_owned_disposition(pgid, signum):
            self.assertIs(dispositions[signal.SIGCHLD], signal.SIG_DFL)
            self.kill(pgid, signum)

        def reap_with_owned_disposition(timeout):
            self.assertIs(dispositions[signal.SIGCHLD], signal.SIG_DFL)
            return self.reap(timeout)

        self.mock(guard.signal, 'getsignal', side_effect=dispositions.__getitem__)
        self.mock(guard.signal, 'signal', side_effect=set_disposition)
        self.mock(guard.os, 'killpg', side_effect=signal_with_owned_disposition)
        self.spawn.side_effect = spawn_with_owned_disposition
        self.child.wait.side_effect = reap_with_owned_disposition
        result = self.run_guard()
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertIs(dispositions[signal.SIGCHLD], inherited)
        self.assertEqual([name for name, _at in self.events], [
            'SIGCHLD_default', 'spawn', 'exec_release', 'SIGTERM', 'SIGKILL',
            'reap', 'SIGCHLD_restored'])
        count = len(self.events)
        self.supervisor.hard_deadline(None, None)
        self.supervisor.signal_group(signal.SIGKILL)
        self.assertEqual(len(self.events), count)

    def test_inherited_ignored_sigchld_is_reset_before_spawn_until_reap(self):
        self.assert_sigchld_owned_until_reap(signal.SIG_IGN)

    def test_inherited_custom_sigchld_is_disabled_before_spawn_until_reap(self):
        inherited = Mock(name='inherited_reaper')
        self.assert_sigchld_owned_until_reap(inherited)
        inherited.assert_not_called()

    def test_failure_to_own_sigchld_rejects_before_spawn(self):
        self.mock(guard.signal, 'getsignal',
                  side_effect=lambda signum: signal.SIG_IGN if signum == signal.SIGCHLD else signal.SIG_DFL)

        def reject_default(signum, handler):
            if signum == signal.SIGCHLD and handler is signal.SIG_DFL:
                raise RuntimeError('Cannot establish SIGCHLD ownership')

        install = self.mock(guard.signal, 'signal', side_effect=reject_default)
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.release.assert_not_called()
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['error']['message'], 'Cannot establish SIGCHLD ownership')
        self.assertEqual(install.call_args.args, (signal.SIGCHLD, signal.SIG_IGN))

    def test_existing_receipt_is_never_overwritten(self):
        self.supervisor.receipt.write_text('existing evidence')
        with self.assertRaises(FileExistsError):
            self.supervisor.run()
        self.assertEqual(self.supervisor.receipt.read_text(), 'existing evidence')
        self.spawn.assert_not_called()

    def test_serialization_lock_contention_does_not_queue_or_launch(self):
        self.mock(guard.fcntl, 'flock', side_effect=BlockingIOError('busy'))
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertEqual(result['error']['type'], 'BlockingIOError')

    def enable_gpu(self, observations=None):
        self.supervisor = guard.Guard(
            ['python', '-B', 'job.py', 'literal;not-shell'], self.root / 'receipt.json',
            30, gpu_uuid=GPU_UUID)
        self.mock(guard, 'process_stat', side_effect=lambda pid: {
            **member(pid), 'pgid': 123 if pid in (123, 124) else pid})
        return self.mock(guard, 'gpu_metrics', side_effect=observations,
                         return_value=gpu_measurement())

    def test_cpu_mode_never_queries_gpu_or_changes_receipt_schema(self):
        query = self.mock(guard, 'gpu_metrics', side_effect=AssertionError('CPU queried GPU'))
        result = self.run_guard()
        query.assert_not_called()
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertNotIn('gpu', result)
        self.assertTrue(all('gpu' not in row and 'gpu_phase' not in row for row in result['samples']))
        with self.assertRaises(ValueError):
            guard.validate_gpu_receipt(result, GPU_UUID)

    def test_gpu_complete_receipt_validates_actual_peak_and_owned_descendant(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(),
                         gpu_measurement(guard.GPU_MEMORY_LIMIT - 1, pids=(123, 124)),
                         gpu_measurement()])
        self.group_rows = deque([[member()], [member(), member(124)], [member(state='Z', rss=0)]])
        result = self.run_guard()
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertEqual(result['gpu']['used_peak_bytes'], guard.GPU_MEMORY_LIMIT - 1)
        self.assertEqual(result['gpu']['identity'], guard.gpu_identity(gpu_measurement()))
        self.assertEqual([row['gpu_phase'] for row in result['samples']],
                         ['admission', 'pre_exec', 'runtime', 'runtime'])
        guard.validate_gpu_receipt(result, GPU_UUID)
        self.assertEqual([name for name, _at in self.events][-3:], ['SIGTERM', 'SIGKILL', 'reap'])

    def test_gpu_admission_user_prevents_spawn(self):
        self.enable_gpu([gpu_measurement(pids=(999,))])
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.release.assert_not_called()
        self.assertEqual(result['first_stop_reason'], 'gpu_compute_users_before_exec')
        with self.assertRaises(ValueError):
            guard.validate_gpu_receipt(result, GPU_UUID)

    def test_gpu_pre_exec_even_owned_user_prevents_release(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(pids=(123,))])
        result = self.run_guard()
        self.spawn.assert_called_once()
        self.release.assert_not_called()
        self.assertEqual(result['first_stop_reason'], 'gpu_compute_users_before_exec')
        self.assertTrue(result['reaped'])
        self.assertEqual([name for name, _at in self.events][-2:], ['SIGKILL', 'reap'])

    def test_gpu_runtime_foreign_user_stops_and_preserves_evidence(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(pids=(999,))])
        result = self.run_guard()
        self.assertTrue(result['startup_released'])
        self.assertEqual(result['first_stop_reason'], 'gpu_foreign_compute_user')
        self.assertEqual(result['samples'][-1]['gpu']['compute_processes'], [{'uuid': GPU_UUID, 'pid': 999}])
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])

    def test_gpu_runtime_exact_cap_stops_even_with_zero_host_rss(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(6 * guard.GIB)])
        self.group_rows = deque([[member(rss=0)], [member(state='Z', rss=0)]])
        result = self.run_guard()
        self.assertEqual(result['rss_peak_bytes'], 0)
        self.assertEqual(result['gpu']['used_peak_bytes'], 6 * guard.GIB)
        self.assertEqual(result['first_stop_reason'], 'gpu_memory_at_6GiB')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_gpu_runtime_driver_change_is_not_success(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(driver_version='580.1')])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'gpu_identity_changed')
        self.assertEqual(result['gpu']['identity']['driver_version'], '570.195.03')
        self.assertEqual(result['guard_exit_code'], 1)

    def test_gpu_missing_monitor_prevents_spawn(self):
        self.enable_gpu([FileNotFoundError('nvidia-smi')])
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['error']['type'], 'FileNotFoundError')

    def test_gpu_runtime_monitor_timeout_still_kills_before_reap(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(),
                         subprocess.TimeoutExpired('nvidia-smi', 1)])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['error']['type'], 'TimeoutExpired')
        self.assertTrue(result['reaped'])
        self.assertEqual([name for name, _at in self.events][-2:], ['SIGKILL', 'reap'])

    def test_gpu_monitor_timeout_is_clamped_to_guard_remaining_time(self):
        query = self.enable_gpu()
        self.supervisor.deadline = self.clock.now + 0.2
        self.supervisor.sample(Path('/unused'), phase='admission')
        self.assertAlmostEqual(query.call_args.kwargs['timeout_seconds'], 0.2)
        query.reset_mock()
        self.clock.now += 0.2
        with self.assertRaises(TimeoutError):
            self.supervisor.sample(Path('/unused'), phase='admission')
        query.assert_not_called()

    def test_host_window_is_not_gpu_completion_or_an_atomic_sample(self):
        query = self.enable_gpu()

        def read_host():
            self.clock.now += 0.25
            return metrics()

        def read_gpu(*_args, **_kwargs):
            self.clock.now += 0.5
            return gpu_measurement()

        self.host.side_effect = read_host
        query.side_effect = read_gpu
        sample = self.supervisor.sample(Path('/unused'), phase='admission')
        times = sample['observation_times']
        self.assertEqual(self.supervisor.record['pressure_diagnostics'][0]
                         ['observation']['finished_monotonic'], 100.0)
        self.assertEqual(times['host'],
                         {'started_monotonic': 100.0, 'finished_monotonic': 100.25})
        self.assertEqual(times['cgroup'],
                         {'started_monotonic': 100.25, 'finished_monotonic': 100.25})
        self.assertEqual(times['gpu'],
                         {'started_monotonic': 100.25, 'finished_monotonic': 100.75})
        self.assertEqual(sample['elapsed_seconds'], 0.75)
        self.assertEqual(self.supervisor.record['started_monotonic'], 100.0)

    def test_gpu_receipt_rejects_missing_altered_or_unsafe_evidence(self):
        self.enable_gpu()
        result = self.run_guard()
        guard.validate_gpu_receipt(result, GPU_UUID)
        mutations = [
            lambda r: r.pop('gpu'),
            lambda r: r['gpu'].update(expected_uuid=OTHER_GPU_UUID),
            lambda r: r['gpu'].update(schema='unversioned'),
            lambda r: r['gpu'].update(memory_limit_bytes=7 * guard.GIB),
            lambda r: r['gpu'].update(used_peak_bytes=0),
            lambda r: r['gpu']['identity'].update(name='other device'),
            lambda r: r.update(samples=r['samples'][:2]),
            lambda r: r['samples'][1].pop('gpu'),
            lambda r: r['samples'][1].update(gpu_phase='runtime'),
            lambda r: r['samples'][2].update(elapsed_seconds=math.nan),
            lambda r: r['samples'][2].update(elapsed_seconds=r['seconds']),
            lambda r: r['samples'][2]['gpu'].update(used_bytes=6 * guard.GIB),
            lambda r: r['samples'][2]['gpu'].update(used_bytes=True),
            lambda r: r['samples'][2]['gpu'].update(driver_version='580.1'),
            lambda r: r['samples'][0]['gpu'].update(compute_processes=[{'uuid': GPU_UUID, 'pid': 999}]),
            lambda r: r['samples'][2]['gpu'].update(compute_processes=[{'uuid': GPU_UUID, 'pid': 999}]),
            lambda r: r['samples'][2]['members'][0].update(pgid=999),
            lambda r: r.update(cleanup_state='failed_or_unproven'),
            lambda r: r.update(startup_released=False),
            lambda r: r.update(guard_exit_code=1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                changed = copy.deepcopy(result)
                mutate(changed)
                with self.assertRaises(ValueError):
                    guard.validate_gpu_receipt(changed, GPU_UUID)
        with self.assertRaises(ValueError):
            guard.validate_gpu_receipt(result, OTHER_GPU_UUID)

    def test_gpu_pre_exec_identity_change_blocks_application(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(name='NVIDIA A10G')])
        result = self.run_guard()
        self.release.assert_not_called()
        self.assertEqual(result['first_stop_reason'], 'gpu_identity_changed')
        self.assertTrue(result['reaped'])

    def test_gpu_admission_memory_cap_prevents_spawn(self):
        self.enable_gpu([gpu_measurement(guard.GPU_MEMORY_LIMIT)])
        result = self.run_guard()
        self.spawn.assert_not_called()
        self.assertEqual(result['first_stop_reason'], 'gpu_memory_at_6GiB')

    def test_gpu_runtime_uuid_mismatch_fails_closed_and_reaps(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(uuid=OTHER_GPU_UUID)])
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['error']['type'], 'ValueError')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])

    def test_owned_compute_descendant_started_during_probe_is_accepted(self):
        query = self.enable_gpu()
        processes = {123: member()}
        self.mock(guard, 'process_stat', side_effect=lambda pid: processes.get(pid, member(pid)))
        self.group_rows = deque([[member()], [member()], [member(state='Z', rss=0)]])
        calls = 0

        def observe(_uuid, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                # The already-captured group snapshot cannot include this child.
                processes[124] = member(124)
                return gpu_measurement(pids=(124,))
            return gpu_measurement()

        query.side_effect = observe
        result = self.run_guard()
        self.assertEqual(result['guard_exit_code'], 0)
        runtime = result['samples'][2]
        self.assertEqual([row['pid'] for row in runtime['members']], [123])
        self.assertEqual(runtime['gpu_process_observations'], [{'pid': 124, 'after': member(124)}])
        guard.validate_gpu_receipt(result, GPU_UUID)
        mutations = [
            lambda s: s.pop('gpu_process_observations'),
            lambda s: s.update(gpu_process_observations=[]),
            lambda s: s['gpu_process_observations'][0].update(after=None),
            lambda s: s['gpu_process_observations'][0].update(pid=125),
            lambda s: s['gpu_process_observations'][0]['after'].update(pgid=999),
            lambda s: s['gpu_process_observations'][0]['after'].update(start_ticks=0),
            lambda s: s['gpu_process_observations'].append(copy.deepcopy(s['gpu_process_observations'][0])),
            lambda s: s['members'].append({**member(124), 'start_ticks': 999}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                altered = copy.deepcopy(result)
                mutate(altered['samples'][2])
                with self.assertRaises(ValueError):
                    guard.validate_gpu_receipt(altered, GPU_UUID)

    def test_fresh_foreign_identity_overrides_pre_probe_owned_member(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(pids=(124,))])
        self.group_rows = deque([[member()], [member(), member(124)]])
        self.mock(guard, 'process_stat', side_effect=lambda pid: {
            **member(pid), 'pgid': 999 if pid == 124 else 123})
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'gpu_foreign_compute_user')
        self.assertEqual(result['samples'][2]['gpu_process_observations'][0]['after']['pgid'], 999)
        self.assertTrue(result['reaped'])

    def test_reused_pid_in_owned_group_cannot_inherit_pre_probe_identity(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(pids=(124,))])
        self.group_rows = deque([[member()], [member(), member(124)]])
        self.mock(guard, 'process_stat', side_effect=lambda pid: {
            **member(pid), 'start_ticks': 9999 if pid == 124 else member(pid)['start_ticks']})
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'gpu_foreign_compute_user')
        self.assertEqual(result['guard_exit_code'], 1)
        self.assertTrue(result['reaped'])

    def test_exited_owned_compute_pid_retains_pre_probe_evidence(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(),
                         gpu_measurement(pids=(124,)), gpu_measurement()])
        self.group_rows = deque([[member()], [member(), member(124)], [member(state='Z', rss=0)]])

        def process(pid):
            if pid == 124:
                raise ProcessLookupError('Exited after NVIDIA snapshot')
            return member(pid)

        self.mock(guard, 'process_stat', side_effect=process)
        result = self.run_guard()
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertEqual(result['samples'][2]['gpu_process_observations'], [{'pid': 124, 'after': None}])
        guard.validate_gpu_receipt(result, GPU_UUID)
        altered = copy.deepcopy(result)
        altered['samples'][2]['members'] = [member()]
        with self.assertRaises(ValueError):
            guard.validate_gpu_receipt(altered, GPU_UUID)

    def test_disappeared_unknown_compute_pid_is_not_assumed_owned(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(pids=(124,))])

        def process(pid):
            if pid == 124:
                raise FileNotFoundError('PID absent')
            return member(pid)

        self.mock(guard, 'process_stat', side_effect=process)
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'gpu_foreign_compute_user')
        self.assertTrue(result['reaped'])

    def test_unreadable_fresh_identity_cannot_use_old_ownership(self):
        self.enable_gpu([gpu_measurement(), gpu_measurement(), gpu_measurement(pids=(124,))])
        self.group_rows = deque([[member()], [member(), member(124)]])

        def process(pid):
            if pid == 124:
                raise PermissionError('proc denied')
            return member(pid)

        self.mock(guard, 'process_stat', side_effect=process)
        result = self.run_guard()
        self.assertEqual(result['first_stop_reason'], 'guard_error')
        self.assertEqual(result['error']['type'], 'PermissionError')
        self.assertTrue(result['reaped'])


if __name__ == '__main__':
    unittest.main()
