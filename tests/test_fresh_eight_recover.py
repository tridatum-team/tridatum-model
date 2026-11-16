"""Offline collection-only recovery boundaries; no cloud requests are issued."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import unicodedata
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import fresh_eight_recover as R

L = R.lifecycle
MIB = 1024**2


def registration_fixture():
    return dict(bucket='recovery-synthetic', key='output/recovery/export.tar',
        instance_id='i-0123456789abcdef0', root_volume_ids=['vol-0123456789abcdef0'],
        registered_at=100, deadline=200, export={'bytes': 16 * MIB},
        new_request_limit=128, new_retrieval_byte_limit=512 * MIB)


def plan_fixture():
    return {'identity': {'region': 'ap-northeast-2'}}


class ExportRangeTests(unittest.TestCase):
    def download(self, directory, raw, *, chunk_bytes=4, head_change=None, range_change=None,
                 body_change=None, version=True):
        identity = dict(bucket='recovery-synthetic', output_prefix='output/recovery/')
        pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        metadata = dict(ContentLength=len(raw), ETag='"immutable-export"')
        if version:
            metadata['VersionId'] = 'immutable-version'
        calls = []
        self.calls = calls

        def client(service, operation, *args):
            calls.append((service, operation, args))
            self.assertEqual(service, 's3api')
            self.assertEqual(args[:4], ('--bucket', identity['bucket'], '--key',
                                       identity['output_prefix'] + 'export.tar'))
            if operation == 'head-object':
                self.assertEqual(len(calls), 1)
                response = dict(metadata)
                if head_change:
                    head_change(response)
                return response
            self.assertEqual(operation, 'get-object')
            self.assertEqual(args[args.index('--if-match') + 1], metadata['ETag'])
            if version:
                self.assertEqual(args[args.index('--version-id') + 1], metadata['VersionId'])
            else:
                self.assertNotIn('--version-id', args)
            start, end = map(int, args[args.index('--range') + 1].removeprefix('bytes=').split('-'))
            body = raw[start:end + 1]
            Path(args[-1]).write_bytes(body_change(body) if body_change else body)
            response = dict(metadata, ContentLength=len(body),
                            ContentRange=f'bytes {start}-{end}/{len(raw)}')
            if range_change:
                range_change(response)
            return response

        output = Path(directory) / 'export.tar'
        result = L.download_export_ranges(client, identity, pin, output, chunk_bytes=chunk_bytes)
        return output, result

    def test_chunk_boundaries_and_final_short_range(self):
        for size, expected in ((1, ['bytes=0-0']), (4, ['bytes=0-3']),
                (5, ['bytes=0-3', 'bytes=4-4']), (8, ['bytes=0-3', 'bytes=4-7']),
                (9, ['bytes=0-3', 'bytes=4-7', 'bytes=8-8'])):
            for version in (False, True):
                with self.subTest(size=size, version=version), tempfile.TemporaryDirectory() as directory:
                    raw = bytes(range(size))
                    output, result = self.download(directory, raw, version=version)
                    self.assertEqual(output.read_bytes(), raw)
                    self.assertEqual(result['ContentLength'], size)
                    self.assertEqual(result['ETag'], '"immutable-export"')
                    self.assertEqual(result.get('VersionId'), 'immutable-version' if version else None)
                    self.assertEqual(self.calls[0][1], 'head-object')
                    ranges = [args[args.index('--range') + 1] for _, _, args in self.calls[1:]]
                    self.assertEqual(ranges, expected)

    def test_head_drift_or_missing_etag_blocks_all_gets(self):
        for mutation in (lambda r: r.update(ContentLength=8), lambda r: r.pop('ContentLength'),
                         lambda r: r.pop('ETag'), lambda r: r.update(ETag='')):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'Export object length/identity'):
                    self.download(directory, b'abcdefghi', head_change=mutation)
                self.assertEqual(len(self.calls), 1)
                self.assertFalse((Path(directory) / 'export.tar').exists())

    def test_ranged_response_identity_and_geometry_must_match(self):
        mutations = (
            lambda r: r.update(ETag='"changed"'),
            lambda r: r.update(VersionId='changed-version'),
            lambda r: r.pop('VersionId'),
            lambda r: r.update(ContentLength=3),
            lambda r: r.update(ContentRange='bytes 0-3/10'),
            lambda r: r.update(ContentRange='bytes 1-4/9'),
            lambda r: r.pop('ContentRange'),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'immutable object identity'):
                    self.download(directory, b'abcdefghi', range_change=mutation)
                self.assertEqual(len(self.calls), 2)
                self.assertEqual((Path(directory) / 'export.tar').read_bytes(), b'')

    def test_body_length_and_digest_are_authenticated(self):
        for change, message in ((lambda raw: raw[:-1], 'immutable object identity'),
                                (lambda raw: b'x' * len(raw), 'Export bytes differ')):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, message):
                    self.download(directory, b'abcd', body_change=change)
                self.assertEqual(len(self.calls), 2)

    def test_invalid_chunk_size_is_rejected_before_head(self):
        for size in (0, -1, True, 1.5, 8 * MIB + 1):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'per-request bound'):
                    self.download(directory, b'abcd', chunk_bytes=size)
                self.assertEqual(self.calls, [])


class RecoveryClientTests(unittest.TestCase):
    def get_args(self, registration, start=0, end=3):
        return ('--bucket', registration['bucket'], '--key', registration['key'],
                '--range', f'bytes={start}-{end}', '--if-match', '"immutable-export"', 'range-part')

    def test_failed_dispatch_is_durably_reserved_without_retry_or_refund(self):
        for failure in ('aws-error', 'timeout'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                registration = registration_fixture()
                ledger = Path(directory) / 'requests'
                client = R.RecoveryClient(plan_fixture(), registration, ledger)
                events = []
                real_fsync = R.os.fsync

                def fsync(fd):
                    real_fsync(fd)
                    events.append('fsync')

                def dispatch(command, **kwargs):
                    events.append('dispatch')
                    self.assertEqual(events, ['fsync', 'fsync', 'dispatch'])
                    record = json.loads((ledger / '0001.json').read_bytes())
                    self.assertEqual(record, client.reservations[0])
                    self.assertEqual(record['arguments'], list(self.get_args(registration)))
                    self.assertEqual(record['sequence'], 1)
                    self.assertEqual(record['reserved_bytes'], 4)
                    self.assertEqual(record['reserved_at'], 150)
                    self.assertEqual(client.reserved_bytes, 4)
                    self.assertEqual(client.requests, 1)
                    self.assertEqual(kwargs['env']['AWS_MAX_ATTEMPTS'], '1')
                    self.assertEqual(kwargs['env']['AWS_IGNORE_CONFIGURED_ENDPOINT_URLS'], 'true')
                    self.assertLessEqual(kwargs['timeout'], 45)
                    if failure == 'timeout':
                        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
                    return subprocess.CompletedProcess(command, 1, '',
                        'An error occurred (RequestTimeout) when calling the GetObject operation')

                with patch.object(R.time, 'time', return_value=150), \
                        patch.object(R.os, 'fsync', side_effect=fsync), \
                        patch.object(L.subprocess, 'run', side_effect=dispatch) as run:
                    with self.assertRaises((L.AWSCallError, subprocess.TimeoutExpired)):
                        client('s3api', 'get-object', *self.get_args(registration))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(client.reserved_bytes, 4)
                self.assertEqual(len(client.reservations), 1)
                self.assertTrue((ledger / '0001.json').is_file())
                with self.assertRaises(FileExistsError):
                    R.RecoveryClient(plan_fixture(), registration, ledger)

    def test_readonly_and_exact_identity_rejections_do_not_reserve_or_dispatch(self):
        r = registration_fixture()
        cases = [
            ('ec2', 'run-instances', (), {}),
            ('ec2', 'terminate-instances', ('--instance-ids', r['instance_id']), {}),
            ('ec2', 'delete-volume', ('--volume-id', r['root_volume_ids'][0]), {}),
            ('s3api', 'put-object', (), {}),
            ('s3api', 'list-objects-v2', ('--bucket', r['bucket']), {}),
            ('ec2', 'describe-instances', ('--instance-ids', 'i-foreign'), {}),
            ('ec2', 'describe-volumes', ('--volume-ids', 'vol-foreign'), {}),
            ('s3api', 'head-object', ('--bucket', 'foreign', '--key', r['key']), {}),
            ('s3api', 'head-object', ('--bucket', r['bucket'], '--key', 'foreign'), {}),
            ('ec2', 'describe-instances', ('--instance-ids', r['instance_id']), {'region': 'us-east-1'}),
            ('ec2', 'describe-instances', ('--instance-ids', r['instance_id']), {'endpoint_url': 'https://invalid'}),
        ]
        for service, operation, args, kwargs in cases:
            with self.subTest(operation=operation, args=args, kwargs=kwargs), \
                    tempfile.TemporaryDirectory() as directory:
                client = R.RecoveryClient(plan_fixture(), r, Path(directory) / 'requests')
                with patch.object(R.time, 'time', return_value=150), patch.object(L.subprocess, 'run') as run:
                    with self.assertRaises((RuntimeError, ValueError)):
                        client(service, operation, *args, **kwargs)
                run.assert_not_called()
                self.assertEqual(client.reservations, [])
                self.assertEqual(client.reserved_bytes, 0)
                self.assertEqual(list(client.directory.iterdir()), [])

    def test_budget_and_deadline_reject_before_reservation(self):
        for changes, now in (({'new_request_limit': 0}, 150),
                             ({'new_retrieval_byte_limit': 3}, 150), ({}, 200), ({}, 201)):
            with self.subTest(changes=changes, now=now), tempfile.TemporaryDirectory() as directory:
                r = dict(registration_fixture(), **changes)
                client = R.RecoveryClient(plan_fixture(), r, Path(directory) / 'requests')
                with patch.object(R.time, 'time', return_value=now), patch.object(L.subprocess, 'run') as run:
                    with self.assertRaises(RuntimeError):
                        client('s3api', 'get-object', *self.get_args(r))
                run.assert_not_called()
                self.assertEqual(client.reservations, [])
                self.assertEqual(client.reserved_bytes, 0)
                self.assertEqual(list(client.directory.iterdir()), [])

    def test_failed_attempt_consumes_last_request_or_byte_allowance(self):
        for changes in ({'new_request_limit': 1}, {'new_retrieval_byte_limit': 4}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                r = dict(registration_fixture(), **changes)
                client = R.RecoveryClient(plan_fixture(), r, Path(directory) / 'requests')
                with patch.object(R.time, 'time', return_value=150), patch.object(L.subprocess, 'run',
                        side_effect=subprocess.TimeoutExpired('aws', 30)) as run:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        client('s3api', 'get-object', *self.get_args(r))
                    with self.assertRaisesRegex(RuntimeError, 'allowance exhausted'):
                        client('s3api', 'get-object', *self.get_args(r))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(client.reserved_bytes, 4)
                self.assertEqual(len(client.reservations), 1)
                self.assertEqual([p.name for p in client.directory.iterdir()], ['0001.json'])

    def test_unbounded_out_of_object_and_oversized_ranges_are_rejected(self):
        r = registration_fixture()
        base = ('--bucket', r['bucket'], '--key', r['key'])
        cases = [base + ('range-part',),
                 base + ('--range', 'bytes=0-', '--if-match', '"etag"', 'range-part'),
                 base + ('--range', 'bytes=0-3', 'range-part'),
                 self.get_args(r, 4, 3), self.get_args(r, 0, 8 * MIB),
                 self.get_args(r, r['export']['bytes'], r['export']['bytes'])]
        for args in cases:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                client = R.RecoveryClient(plan_fixture(), r, Path(directory) / 'requests')
                with patch.object(R.time, 'time', return_value=150), patch.object(L.subprocess, 'run') as run:
                    with self.assertRaises((RuntimeError, ValueError)):
                        client('s3api', 'get-object', *args)
                run.assert_not_called()
                self.assertEqual(client.reservations, [])


class RecoveryCleanupTests(unittest.TestCase):
    def terminated(self, identifier=None, state='terminated'):
        return {'Reservations': [{'Instances': [{'InstanceId': identifier or registration_fixture()['instance_id'],
                                                 'State': {'Name': state}}]}]}

    def missing_volume(self):
        return L.AWSCallError('ec2', 'describe-volumes',
            'An error occurred (InvalidVolume.NotFound) when calling the DescribeVolumes operation')

    def test_exact_terminated_instance_and_positive_volume_absence_are_accepted(self):
        r = registration_fixture()
        client = Mock(side_effect=[self.terminated(), self.missing_volume()])
        with patch.object(R.time, 'time', return_value=150):
            observations = R.observe_cleanup(client, r)
        self.assertEqual(client.call_args_list[0].args,
                         ('ec2', 'describe-instances', '--instance-ids', r['instance_id']))
        self.assertEqual(client.call_args_list[1].args,
                         ('ec2', 'describe-volumes', '--volume-ids', r['root_volume_ids'][0]))
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[1]['response'], {'error_code': 'InvalidVolume.NotFound'})
        R.validate_cleanup(observations, r)

    def test_missing_empty_foreign_or_running_instance_is_not_cleanup_proof(self):
        invalid = [None, {}, {'Reservations': []}, {'Reservations': [{'Instances': []}]},
                   self.terminated('i-foreign'), self.terminated(state='running')]
        duplicate = self.terminated()
        duplicate['Reservations'] *= 2
        invalid.append(duplicate)
        for response in invalid:
            with self.subTest(response=response), patch.object(R.time, 'time', return_value=150):
                client = Mock(side_effect=[response, self.missing_volume()])
                with self.assertRaises((RuntimeError, ValueError, KeyError, TypeError)):
                    R.observe_cleanup(client, registration_fixture())

    def test_empty_volume_response_and_nonabsence_errors_are_not_cleanup_proof(self):
        invalid = [None, {}, {'Volumes': []}, {'Volumes': [{'VolumeId': 'vol-foreign'}]},
                   {'error_code': 'InvalidInstanceID.NotFound'},
                   L.AWSCallError('ec2', 'describe-volumes', 'An error occurred (AccessDenied)')]
        for response in invalid:
            with self.subTest(response=response), patch.object(R.time, 'time', return_value=150):
                with self.assertRaises((RuntimeError, ValueError, KeyError, TypeError)):
                    R.observe_cleanup(Mock(side_effect=[self.terminated(), response]), registration_fixture())

    def test_timeout_at_either_lookup_is_not_absence_and_is_never_retried(self):
        for index in (0, 1):
            with self.subTest(index=index), patch.object(R.time, 'time', return_value=150):
                responses = [self.terminated()] * index + [subprocess.TimeoutExpired('aws', 30)]
                client = Mock(side_effect=responses)
                with self.assertRaises(subprocess.TimeoutExpired):
                    R.observe_cleanup(client, registration_fixture())
                self.assertEqual(client.call_count, index + 1)

    def test_cleanup_requires_two_fresh_exact_observations(self):
        r = registration_fixture()
        with patch.object(R.time, 'time', return_value=150):
            observations = R.observe_cleanup(Mock(side_effect=[self.terminated(), self.missing_volume()]), r)
        invalid = [[], observations[:1], observations[::-1]]
        for field, value in (('identifier', 'i-foreign'), ('started_at', 99),
                             ('observed_at', 200), ('observed_at', 149)):
            altered = [dict(row) for row in observations]
            altered[0][field] = value
            invalid.append(altered)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises((RuntimeError, ValueError)):
                R.validate_cleanup(rows, r)


class RecoveryAcceptanceDeadlineTests(unittest.TestCase):
    def test_expiry_during_final_validation_or_hashing_never_accepts(self):
        for expiration_stage in ('registration', 'archive', 'candidate'):
            with self.subTest(stage=expiration_stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                state = root / 'cloud-state'
                state.mkdir()
                requests = state / 'collection-recovery-requests'
                requests.mkdir()
                registration_path = state / 'collection-recovery-registration.json'
                registration_path.write_text('{}')
                registration = dict(registered_at=1.0, deadline=20.0, export={},
                                    originals={}, sources=[])
                clock = [10.0]
                validations = [0]

                def registration_validation(*args, **kwargs):
                    validations[0] += 1
                    if expiration_stage == 'registration' and validations[0] == 2:
                        clock[0] = 21.0
                    return registration, {'identity': {}}

                def evidence_pin(root, path):
                    name = Path(path).name
                    if ((expiration_stage == 'archive' and name == 'recovered-export.tar')
                            or (expiration_stage == 'candidate' and name == 'candidate.json')):
                        clock[0] = 21.0
                    return dict(path=name, bytes=1, sha256='0' * 64)

                with patch.object(R, 'validate_registration', side_effect=registration_validation), \
                        patch.object(R, 'RecoveryClient', return_value=Mock(directory=requests)), \
                        patch.object(L, 'download_export_ranges', return_value={}), \
                        patch.object(R, 'validate_export', return_value={}), \
                        patch.object(R, 'observe_cleanup', return_value=[]), \
                        patch.object(R, 'validate_ledger', return_value={}), \
                        patch.object(R, 'pin', side_effect=evidence_pin), \
                        patch.object(R.time, 'time', side_effect=lambda: clock[0]):
                    with self.assertRaisesRegex((RuntimeError, ValueError), 'deadline'):
                        R.recover(root, registration_path)
                self.assertFalse((state / 'collection-recovery.json').exists())
                failure = json.loads((state / 'collection-recovery-failure.json').read_text())
                self.assertEqual(failure['state'], 'failed')
                self.assertIn('deadline', failure['error'])


class RecoveryModuleIdentityTests(unittest.TestCase):
    def test_exact_physical_module_accepts_but_same_bytes_elsewhere_do_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected = root / 'module.py'
            expected.write_text('value = 1\n')
            R.validate_executing_module(root, 'module.py', expected)
            foreign = root / 'foreign.py'
            foreign.write_bytes(expected.read_bytes())
            with self.assertRaisesRegex((RuntimeError, ValueError), 'registered checkout'):
                R.validate_executing_module(root, 'module.py', foreign)
            alias = root / 'alias.py'
            alias.symlink_to(expected)
            with self.assertRaisesRegex((RuntimeError, ValueError), 'registered checkout'):
                R.validate_executing_module(root, 'module.py', alias)
            hardlink = root / 'hardlink.py'
            R.os.link(expected, hardlink)
            with self.assertRaisesRegex((RuntimeError, ValueError), 'registered checkout'):
                R.validate_executing_module(root, 'module.py', hardlink)

    def test_unicode_alias_on_normalizing_filesystem_uses_physical_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            folder = root / '사업'
            folder.mkdir()
            expected = folder / 'module.py'
            expected.write_text('value = 1\n')
            alternate = Path(unicodedata.normalize('NFD', str(expected)))
            self.assertNotEqual(str(expected), str(alternate))
            if not alternate.exists():
                self.skipTest('Filesystem does not resolve NFC/NFD as the same file')
            self.assertTrue(R.os.path.samefile(expected, alternate))
            R.validate_executing_module(root, '사업/module.py', alternate)


def finalization_registration_fixture():
    return dict(registration_fixture(), account='123456789012', new_request_limit=3,
                new_retrieval_byte_limit=3 * MIB,
                prior_accounting={'new_reserved_bytes': 16 * MIB})


def prior_cleanup_fixture():
    r = registration_fixture()
    return dict(cleanup_state='complete', instance_id=r['instance_id'],
                instance_terminated=True, root_volume_ids=r['root_volume_ids'],
                absent_volume_ids=r['root_volume_ids'], recorded_volume_ids=r['root_volume_ids'],
                remaining_volumes=[])


class FinalizationCleanupTests(unittest.TestCase):
    def observations(self, instance=None, volume=None):
        r = finalization_registration_fixture()
        responses = (
            {'Reservations': []} if instance is None else instance,
            {'Volumes': []} if volume is None else volume,
        )
        return [dict(operation=operation, identifier=identifier, started_at=150,
                     observed_at=151, response=response)
                for operation, identifier, response in zip(
                    ('describe-instances', 'describe-volumes'),
                    (r['instance_id'], r['root_volume_ids'][0]), responses)]

    def test_exact_terminated_typed_absence_and_explicit_empty_are_accepted(self):
        r = finalization_registration_fixture()
        instances = [
            {'Reservations': [{'Instances': [{'InstanceId': r['instance_id'],
                                              'State': {'Name': 'terminated'}}]}]},
            {'error_code': 'InvalidInstanceID.NotFound'},
            {'Reservations': []},
        ]
        volumes = [{'error_code': 'InvalidVolume.NotFound'}, {'Volumes': []}]
        for instance in instances:
            for volume in volumes:
                with self.subTest(instance=instance, volume=volume):
                    R.validate_finalization_cleanup(
                        self.observations(instance, volume), r, prior_cleanup_fixture())

    def test_empty_reads_require_exact_positive_historical_cleanup(self):
        invalid = [{}]
        prior = prior_cleanup_fixture()
        for field in prior:
            invalid.append({key: value for key, value in prior.items() if key != field})
        for field, value in (
                ('cleanup_state', 'pending'), ('instance_id', 'i-foreign'),
                ('instance_terminated', False), ('instance_terminated', 1),
                ('root_volume_ids', []), ('root_volume_ids', ['vol-foreign']),
                ('absent_volume_ids', []), ('absent_volume_ids', ['vol-foreign']),
                ('recorded_volume_ids', []), ('recorded_volume_ids', ['vol-foreign']),
                ('remaining_volumes', ['vol-foreign']), ('remaining_volumes', None)):
            invalid.append(dict(prior, **{field: value}))
        for cleanup in invalid:
            with self.subTest(cleanup=cleanup), self.assertRaises((RuntimeError, ValueError)):
                R.validate_finalization_cleanup(
                    self.observations(), finalization_registration_fixture(), cleanup)

    def test_missing_malformed_foreign_live_and_duplicate_results_are_rejected(self):
        r = finalization_registration_fixture()
        exact = {'InstanceId': r['instance_id'], 'State': {'Name': 'terminated'}}
        invalid_instances = [
            None, {}, {'Reservations': None}, {'Reservations': {}},
            {'Reservations': [{'Instances': []}]},
            {'Reservations': [{'Instances': [dict(exact, InstanceId='i-foreign')]}]},
            {'Reservations': [{'Instances': [exact, exact]}]},
            {'error_code': 'InvalidVolume.NotFound'},
            {'error_code': 'AccessDenied'},
        ]
        invalid_instances.extend(
            {'Reservations': [{'Instances': [dict(exact, State={'Name': state})]}]}
            for state in ('pending', 'running', 'stopping', 'stopped', 'shutting-down'))
        invalid_volumes = [
            None, {}, {'Volumes': None}, {'Volumes': {}},
            {'Volumes': [{'VolumeId': r['root_volume_ids'][0], 'State': 'available'}]},
            {'Volumes': [{'VolumeId': 'vol-foreign'}]},
            {'error_code': 'InvalidInstanceID.NotFound'}, {'error_code': 'AccessDenied'},
        ]
        for index, responses in enumerate((invalid_instances, invalid_volumes)):
            for response in responses:
                rows = self.observations()
                rows[index]['response'] = response
                with self.subTest(index=index, response=response), \
                        self.assertRaises((RuntimeError, ValueError, KeyError, TypeError)):
                    R.validate_finalization_cleanup(rows, r, prior_cleanup_fixture())

    def test_observations_require_exact_order_identity_and_fresh_window(self):
        r = finalization_registration_fixture()
        rows = self.observations()
        invalid = [[], rows[:1], rows[::-1], rows + rows[:1]]
        for index in (0, 1):
            for field, value in (
                    ('operation', 'describe-tags'), ('identifier', 'foreign'),
                    ('started_at', 99), ('observed_at', 200), ('observed_at', 149)):
                altered = [dict(row) for row in rows]
                altered[index][field] = value
                invalid.append(altered)
        for observations in invalid:
            with self.subTest(observations=observations), \
                    self.assertRaises((RuntimeError, ValueError)):
                R.validate_finalization_cleanup(observations, r, prior_cleanup_fixture())
        rows[0]['started_at'] = r['registered_at']
        rows[1]['observed_at'] = r['deadline'] - 0.001
        R.validate_finalization_cleanup(rows, r, prior_cleanup_fixture())


class FinalizationClientTests(unittest.TestCase):
    def calls(self, r):
        return [('sts', 'get-caller-identity'),
                ('ec2', 'describe-instances', '--instance-ids', r['instance_id']),
                ('ec2', 'describe-volumes', '--volume-ids', r['root_volume_ids'][0])]

    def test_three_metadata_reads_are_write_ahead_reserved_and_cannot_repeat(self):
        r = finalization_registration_fixture()
        with tempfile.TemporaryDirectory() as directory:
            client = R.FinalizationClient(plan_fixture(), r, Path(directory) / 'requests')
            events = []
            real_fsync = R.os.fsync

            def fsync(fd):
                real_fsync(fd)
                events.append('fsync')

            def dispatch(command, **kwargs):
                events.append('dispatch')
                index = len(client.reservations)
                self.assertEqual(events, ['fsync', 'fsync', 'dispatch'] * index)
                row = json.loads((client.directory / f'{index:04d}.json').read_text())
                self.assertEqual(row, client.reservations[-1])
                service, operation, *args = self.calls(r)[index - 1]
                self.assertEqual(row, dict(sequence=index, service=service, operation=operation,
                                          arguments=args, reserved_bytes=MIB, reserved_at=150))
                self.assertEqual(client.requests, index)
                self.assertEqual(kwargs['env']['AWS_MAX_ATTEMPTS'], '1')
                self.assertEqual(kwargs['env']['AWS_IGNORE_CONFIGURED_ENDPOINT_URLS'], 'true')
                self.assertLessEqual(kwargs['timeout'], 45)
                return subprocess.CompletedProcess(command, 0, '{}', '')

            with patch.object(R.time, 'time', return_value=150), \
                    patch.object(R.os, 'fsync', side_effect=fsync), \
                    patch.object(L.subprocess, 'run', side_effect=dispatch) as run:
                for call in self.calls(r):
                    client(*call)
                for call in self.calls(r):
                    with self.assertRaises((RuntimeError, ValueError)):
                        client(*call)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sum(row['reserved_bytes'] for row in client.reservations), 3 * MIB)
            with self.assertRaises(FileExistsError):
                R.FinalizationClient(plan_fixture(), r, client.directory)

    def test_s3_mutations_foreign_ids_and_out_of_order_calls_never_dispatch(self):
        r = finalization_registration_fixture()
        valid = self.calls(r)
        invalid = [
            (('s3api', 'head-object', '--bucket', r['bucket'], '--key', r['key']), {}),
            (('s3api', 'get-object', '--bucket', r['bucket'], '--key', r['key'], 'part'), {}),
            (('ec2', 'terminate-instances', '--instance-ids', r['instance_id']), {}),
            (('ec2', 'delete-volume', '--volume-id', r['root_volume_ids'][0]), {}),
            (('ec2', 'describe-instances', '--instance-ids', 'i-foreign'), {}),
            (('ec2', 'describe-volumes', '--volume-ids', 'vol-foreign'), {}),
            (('sts', 'get-caller-identity', '--region', 'us-east-1'), {}),
        ]
        for stage in range(3):
            cases = invalid + [(call, {}) for index, call in enumerate(valid) if index != stage]
            cases += [(valid[stage], {'region': 'us-east-1'}),
                      (valid[stage], {'endpoint_url': 'https://invalid'})]
            for call, kwargs in cases:
                with self.subTest(stage=stage, call=call, kwargs=kwargs), \
                        tempfile.TemporaryDirectory() as directory:
                    client = R.FinalizationClient(plan_fixture(), r, Path(directory) / 'requests')
                    with patch.object(R.time, 'time', return_value=150), \
                            patch.object(L.subprocess, 'run',
                                         return_value=subprocess.CompletedProcess([], 0, '{}', '')) as run:
                        for preceding in valid[:stage]:
                            client(*preceding)
                        with self.assertRaises((RuntimeError, ValueError)):
                            client(*call, **kwargs)
                    self.assertEqual(run.call_count, stage)
                    self.assertEqual(len(client.reservations), stage)
                    self.assertEqual(len(list(client.directory.iterdir())), stage)

    def test_failed_dispatch_is_reserved_and_never_retried_or_refunded(self):
        r = finalization_registration_fixture()
        for stage in range(3):
            for timeout in (False, True):
                with self.subTest(stage=stage, timeout=timeout), tempfile.TemporaryDirectory() as directory:
                    client = R.FinalizationClient(plan_fixture(), r, Path(directory) / 'requests')
                    success = subprocess.CompletedProcess([], 0, '{}', '')
                    failure = (subprocess.TimeoutExpired('aws', 30) if timeout else
                               subprocess.CompletedProcess([], 1, '',
                                   'An error occurred (AccessDenied) when calling the operation'))
                    with patch.object(R.time, 'time', return_value=150), \
                            patch.object(L.subprocess, 'run', side_effect=[success] * stage + [failure]) as run:
                        for call in self.calls(r)[:stage]:
                            client(*call)
                        with self.assertRaises((L.AWSCallError, subprocess.TimeoutExpired)):
                            client(*self.calls(r)[stage])
                        with self.assertRaises((RuntimeError, ValueError)):
                            client(*self.calls(r)[stage])
                    self.assertEqual(run.call_count, stage + 1)
                    self.assertEqual(client.requests, stage + 1)
                    self.assertEqual(len(client.reservations), stage + 1)
                    self.assertEqual(json.loads(
                        (client.directory / f'{stage + 1:04d}.json').read_text()),
                        client.reservations[-1])

    def test_deadline_and_both_byte_budgets_fail_before_reservation(self):
        cases = [
            ({'new_retrieval_byte_limit': MIB - 1}, 150),
            ({'prior_accounting': {'new_reserved_bytes': R.PRIOR_BYTES - MIB + 1}}, 150),
            ({}, 200),
            ({}, 201),
        ]
        for changes, now in cases:
            with self.subTest(changes=changes, now=now), tempfile.TemporaryDirectory() as directory:
                r = dict(finalization_registration_fixture(), **changes)
                client = R.FinalizationClient(plan_fixture(), r, Path(directory) / 'requests')
                with patch.object(R.time, 'time', return_value=now), patch.object(L.subprocess, 'run') as run:
                    with self.assertRaises((RuntimeError, ValueError)):
                        client('sts', 'get-caller-identity')
                run.assert_not_called()
                self.assertEqual(client.reservations, [])
                self.assertEqual(list(client.directory.iterdir()), [])


class FinalizationObservationTests(unittest.TestCase):
    def observation_path(self, directory, index):
        return Path(directory) / f'collection-finalization-observation-{index}.json'

    def client(self, directory, responses):
        client = Mock(side_effect=responses)
        client.directory = Path(directory) / 'requests'
        return client

    def test_raw_responses_persist_before_semantic_validation_and_are_returned(self):
        r = finalization_registration_fixture()
        responses = [{'Account': r['account'], 'Arn': 'synthetic-caller'},
                     {'Reservations': [], 'ResponseMetadata': {'RequestId': 'instance-read'}},
                     {'Volumes': [], 'ResponseMetadata': {'RequestId': 'volume-read'}}]
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(directory, responses)
            validate_caller = R.validate_finalization_caller
            validate_cleanup = R.validate_finalization_cleanup

            def caller_validation(row, registration):
                self.assertEqual(json.loads(self.observation_path(directory, 0).read_text()), row)
                validate_caller(row, registration)

            def cleanup_validation(rows, registration, prior):
                for index, row in enumerate(rows, 1):
                    self.assertEqual(json.loads(self.observation_path(directory, index).read_text()), row)
                validate_cleanup(rows, registration, prior)

            with patch.object(R.time, 'time', return_value=150), \
                    patch.object(R, 'validate_finalization_caller', side_effect=caller_validation), \
                    patch.object(R, 'validate_finalization_cleanup', side_effect=cleanup_validation):
                rows = R.observe_finalization_cleanup(client, r, prior_cleanup_fixture())
            self.assertEqual([row['response'] for row in rows], responses[1:])
            self.assertEqual([call.args for call in client.call_args_list],
                             FinalizationClientTests().calls(r))
            for index, response in enumerate(responses):
                self.assertEqual(json.loads(
                    self.observation_path(directory, index).read_text())['response'], response)

    def test_foreign_or_missing_account_is_persisted_and_blocks_ec2(self):
        r = finalization_registration_fixture()
        for response in (None, {}, {'Account': '999999999999'}):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                client = self.client(directory, [response])
                with patch.object(R.time, 'time', return_value=150), \
                        self.assertRaises((RuntimeError, ValueError)):
                    R.observe_finalization_cleanup(client, r, prior_cleanup_fixture())
                self.assertEqual(client.call_count, 1)
                row = json.loads(self.observation_path(directory, 0).read_text())
                self.assertEqual(row['response'], response)
                self.assertFalse(self.observation_path(directory, 1).exists())

    def test_invalid_cleanup_raw_results_survive_semantic_failure(self):
        r = finalization_registration_fixture()
        for index, response in ((1, None), (1, {}),
                (1, {'Reservations': [{'Instances': [{'InstanceId': r['instance_id'],
                                                       'State': {'Name': 'running'}}]}]}),
                (2, None), (2, {'Volumes': [{'VolumeId': 'vol-foreign'}]})):
            with self.subTest(index=index, response=response), tempfile.TemporaryDirectory() as directory:
                responses = [{'Account': r['account']}, {'Reservations': []}, {'Volumes': []}]
                responses[index] = response
                client = self.client(directory, responses)
                with patch.object(R.time, 'time', return_value=150), \
                        self.assertRaises((RuntimeError, ValueError)):
                    R.observe_finalization_cleanup(client, r, prior_cleanup_fixture())
                self.assertEqual(client.call_count, 3)
                for position, raw in enumerate(responses):
                    self.assertEqual(json.loads(
                        self.observation_path(directory, position).read_text())['response'], raw)

    def test_typed_absence_preserves_original_error_diagnostics(self):
        r = finalization_registration_fixture()
        errors = [L.AWSCallError('ec2', operation,
                                f'An error occurred ({code}) ' + 'original diagnostic\n' * 100)
                  for operation, code in (('describe-instances', 'InvalidInstanceID.NotFound'),
                                          ('describe-volumes', 'InvalidVolume.NotFound'))]
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(directory, [{'Account': r['account']}, *errors])
            with patch.object(R.time, 'time', return_value=150):
                rows = R.observe_finalization_cleanup(client, r, prior_cleanup_fixture())
            self.assertEqual(client.call_count, 3)
            for index, (row, error) in enumerate(zip(rows, errors), 1):
                self.assertEqual(row['response'], {'error_code': error.code, 'error': error.stderr})
                self.assertEqual(json.loads(self.observation_path(directory, index).read_text()), row)

    def test_transport_errors_at_each_read_are_persisted_before_reraise_without_retry(self):
        r = finalization_registration_fixture()
        for index in range(3):
            for kind in ('timeout', 'denied'):
                with self.subTest(index=index, kind=kind), tempfile.TemporaryDirectory() as directory:
                    operation = ('get-caller-identity', 'describe-instances', 'describe-volumes')[index]
                    error = (subprocess.TimeoutExpired('aws', 30) if kind == 'timeout' else
                             L.AWSCallError('sts' if index == 0 else 'ec2', operation,
                                            'An error occurred (AccessDenied) diagnostic'))
                    responses = [{'Account': r['account']}, {'Reservations': []}, {'Volumes': []}]
                    client = self.client(directory, responses[:index] + [error])
                    with patch.object(R.time, 'time', return_value=150), \
                            self.assertRaises(type(error)) as raised:
                        R.observe_finalization_cleanup(client, r, prior_cleanup_fixture())
                    self.assertIs(raised.exception, error)
                    self.assertEqual(client.call_count, index + 1)
                    row = json.loads(self.observation_path(directory, index).read_text())
                    expected = ({'error_code': error.code, 'error': error.stderr}
                                if kind == 'denied' and index > 0 else
                                {'error_type': type(error).__name__, 'error': str(error)})
                    self.assertEqual(row['response'], expected)
                    for earlier in range(index):
                        self.assertTrue(self.observation_path(directory, earlier).exists())
                    self.assertFalse(self.observation_path(directory, index + 1).exists())


class RecoverySourceArchiveTests(unittest.TestCase):
    def archive_fixture(self, root, mutation=None):
        archive_path = root / 'historical-sources.tar'
        sources = []
        with tarfile.open(archive_path, 'w', format=tarfile.PAX_FORMAT) as archive:
            for index, name in enumerate(R.SOURCES):
                raw = f'# historical source {index}\n'.encode()
                sources.append(dict(path=name, bytes=len(raw),
                                    sha256=hashlib.sha256(raw).hexdigest()))
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                member.pax_headers = {'mtime': '1700000000.123456'}
                if index == 0:
                    if mutation == 'pax-path':
                        member.pax_headers['path'] = name
                    elif mutation == 'link':
                        member.type = tarfile.SYMTYPE
                        member.linkname = '../outside.py'
                        member.size = 0
                    elif mutation == 'changed-source':
                        raw = b'x' * len(raw)
                    elif mutation == 'traversal':
                        member.name = '../outside.py'
                    elif mutation in ('nan', 'inf', '-inf'):
                        member.pax_headers['mtime'] = mutation
                    elif mutation == 'missing':
                        continue
                archive.addfile(member, io.BytesIO(raw))
                if index == 0 and mutation == 'duplicate':
                    archive.addfile(member, io.BytesIO(raw))
        return R.pin(root, archive_path), sources

    def test_exact_source_closure_with_fractional_pax_mtime_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive_pin, sources = self.archive_fixture(root)
            with patch.object(tarfile.TarFile, 'extract') as extract, \
                    patch.object(tarfile.TarFile, 'extractall') as extractall:
                R.authenticate_recovery_sources(root, archive_pin, sources)
            extract.assert_not_called()
            extractall.assert_not_called()
            self.assertEqual([path.name for path in root.iterdir()], ['historical-sources.tar'])

    def test_pax_path_duplicate_link_traversal_and_missing_members_are_rejected(self):
        for mutation in ('pax-path', 'duplicate', 'link', 'traversal', 'missing'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                archive_pin, sources = self.archive_fixture(root, mutation)
                with self.assertRaises((RuntimeError, ValueError)):
                    R.authenticate_recovery_sources(root, archive_pin, sources)

    def test_changed_source_bytes_fail_even_with_valid_whole_archive_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive_pin, sources = self.archive_fixture(root, 'changed-source')
            with self.assertRaisesRegex((RuntimeError, ValueError), 'source bytes differ'):
                R.authenticate_recovery_sources(root, archive_pin, sources)

    def test_nonfinite_pax_mtime_is_rejected(self):
        for value in ('nan', 'inf', '-inf'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                archive_pin, sources = self.archive_fixture(root, value)
                with self.assertRaises((RuntimeError, ValueError)):
                    R.authenticate_recovery_sources(root, archive_pin, sources)

    def test_whole_archive_pin_and_exact_registered_source_closure_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive_pin, sources = self.archive_fixture(root)
            for altered in (sources[:-1], [dict(sources[0], path='foreign.py'), *sources[1:]]):
                with self.subTest(sources=altered), \
                        self.assertRaises((RuntimeError, ValueError)):
                    R.authenticate_recovery_sources(root, archive_pin, altered)
            with (root / archive_pin['path']).open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex((RuntimeError, ValueError), 'pin differs'):
                R.authenticate_recovery_sources(root, archive_pin, sources)


if __name__ == '__main__':
    unittest.main()
