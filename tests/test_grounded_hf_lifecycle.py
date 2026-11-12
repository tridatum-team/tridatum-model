"""Offline behavioral tests; synthetic fixtures are never launch admission receipts."""
import copy
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import grounded_hf_lifecycle as L
import grounded_hf_worker as W


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def identity():
    run = 'grounded-synthetic-test'
    return dict(run_id=run, experiment='synthetic-only', eval_tag='synthetic-only',
        account='148761639313', region='ap-northeast-2', bucket='synthetic-bucket',
        code_prefix=f'code/{run}/', output_prefix=f'output/{run}/',
        instance_profile='synthetic-instance', instance_role='synthetic-instance',
        scheduler_role='synthetic-scheduler', schedule_group='synthetic-group',
        tags={'trid:eval': 'synthetic-only', 'trid:experiment': 'synthetic-only', 'trid:run': run})


def synthetic_ebs_evidence(root, plan, mutate=None):
    ident = plan['identity']
    key_id = '11111111-2222-3333-4444-555555555555'
    arn = f'arn:aws:kms:{ident["region"]}:{ident["account"]}:key/{key_id}'
    plan['ebs_kms_key_arn'] = arn
    raw = dict(query=['aws', 'kms', 'describe-key', '--region', ident['region'], '--key-id', 'alias/aws/ebs'],
        exit_code=0, error_code=None, response={'KeyMetadata': dict(
            Arn=arn, KeyId=key_id, AWSAccountId=ident['account'], KeyManager='AWS', Origin='AWS_KMS',
            Enabled=True, KeyState='Enabled', KeyUsage='ENCRYPT_DECRYPT', KeySpec='SYMMETRIC_DEFAULT',
            EncryptionAlgorithms=['SYMMETRIC_DEFAULT'], MultiRegion=False)})
    if mutate:
        mutate(raw)
    path = Path(root) / 'synthetic-ebs-key.json'
    path.write_text(json.dumps(raw))
    pin = dict(path=path.name, sha256=L.digest(path), bytes=path.stat().st_size)
    plan.setdefault('bundle', {}).setdefault('files', []).append(pin)
    return pin


def instance(ident, state='running'):
    return dict(InstanceId='i-0123456789abcdef0', ClientToken=ident['run_id'],
        Tags=[dict(Key=k, Value=v) for k, v in ident['tags'].items()], State={'Name': state},
        ImageId='ami-0123456789abcdef0', InstanceType='g6e.2xlarge', SubnetId='subnet-synthetic',
        Placement={'AvailabilityZone': 'ap-northeast-2a'},
        MetadataOptions={'HttpTokens': 'required', 'HttpPutResponseHopLimit': 1},
        NetworkInterfaces=[{'Ipv6Addresses': []}],
        BlockDeviceMappings=[dict(DeviceName='/dev/sda1', Ebs={'VolumeId': 'vol-synthetic', 'DeleteOnTermination': True})],
        LaunchTime=dt.datetime.fromtimestamp(1000, dt.timezone.utc).isoformat())


def gpu():
    return dict(uuid='GPU-11111111-2222-3333-4444-555555555555', minor=3,
                name='NVIDIA L40S', total_mib=46000, driver='synthetic-driver')


def prerequisite_expectations():
    names = ['python3', 'aws', 'systemd-run', 'systemctl', 'iptables', 'ip6tables',
             'mount', 'useradd', 'shutdown', 'nvidia-smi', 'sha256sum', 'timeout',
             'modprobe', 'dpkg-deb', 'dpkg-query']
    return dict(prerequisite_expectations_bound=True, postboot_required_names=names,
        historical_evidence_levels={name: 'unobserved' for name in names},
        expected_commands={name: dict(path='/expected/' + name, sha256='c' * 64,
            probe_args=['--help'] if name in ('useradd', 'shutdown') else ['--version']) for name in names},
        python_version='3.12.3', venv_package_version='3.12.3-1ubuntu0.17', aws_cli_version='2.39.5')


def actual_prerequisites(expected):
    return dict(prerequisites_verified=True, commands={
        name: dict(found_path=pin['path'], resolved_path='/resolved/' + name, sha256='c' * 64,
                   probe_args=pin['probe_args'], stdout='Synthetic successful capability probe.')
        for name, pin in expected['expected_commands'].items()},
        **{key: expected[key] for key in ('python_version', 'venv_package_version', 'aws_cli_version')})


def release_plan():
    ident = identity()
    return dict(identity=ident, runtime=dict(gpu_name='NVIDIA L40S', gpu_catalog_mib=45776,
        gpu_nvml_min_mib=45776, gpu_nvml_max_mib=49152, driver_version='synthetic-driver'),
        iam={'scheduler_role': {'arn': 'arn:aws:iam::148761639313:role/synthetic-scheduler'}})


def release(plan):
    row = instance(plan['identity'])
    return dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64, instance_id=row['InstanceId'],
        launch_time=1000, deadline=15400, schedule_verified=True, gpu=gpu(),
        schedule=L.schedule_spec(plan, row['InstanceId'], 1000))


def archive_fixture(directory, members):
    path = Path(directory) / 'bundle.tar'
    rows = []
    with tarfile.open(path, 'w:', format=tarfile.USTAR_FORMAT) as archive:
        for name, payload, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                member.linkname = 'elsewhere'
                archive.addfile(member)
            rows.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return path, dict(bytes=path.stat().st_size, sha256=L.digest(path), unpacked_bytes=1024 * 1024, files=rows)


class ReconciliationTests(unittest.TestCase):
    def exercise(self, observations, volumes=None, *, exact=None, evidence=None, state=None):
        clock, calls = Clock(), []
        ident = identity()
        index = [0]
        evidence = evidence if evidence is not None else L.volume_identity(ident, 'a' * 64, '/dev/sda1')

        def transport(service, operation, *args, **kwargs):
            calls.append((service, operation, args))
            if operation == 'describe-instances':
                rows = observations[min(index[0], len(observations) - 1)]
                index[0] += 1
                return {'Reservations': [{'Instances': rows}]}
            if operation == 'describe-volumes':
                if '--volume-ids' not in args:
                    return {'Volumes': volumes or []}
                if isinstance(exact, dict):
                    value = exact[args[args.index('--volume-ids') + 1]]
                    if isinstance(value, Exception):
                        raise value
                    if value is None:
                        raise L.AWSCallError(service, operation, 'An error occurred (InvalidVolume.NotFound)')
                    return {'Volumes': [value]}
                if exact is not None:
                    return {'Volumes': exact}
                current = observations[min(max(index[0] - 1, 0), len(observations) - 1)]
                if current and current[0]['State']['Name'] != 'terminated':
                    return {'Volumes': [dict(VolumeId='vol-synthetic', Tags=current[0]['Tags'],
                        Attachments=[dict(InstanceId=current[0]['InstanceId'], Device='/dev/sda1')])]}
                raise L.AWSCallError(service, operation, 'An error occurred (InvalidVolume.NotFound)')
            if operation == 'terminate-instances':
                if state is not None:
                    persisted = json.loads((Path(state) / 'volume-identity.json').read_text())
                    self.assertEqual(persisted['root_volume_ids'], ['vol-synthetic'])
                    self.assertTrue(persisted['volumes']['vol-synthetic']['owned_attachments'])
                return {}
            raise AssertionError(operation)

        result = L.reconcile(transport, ident, None, 1020, evidence=evidence,
            evidence_path=Path(state) / 'volume-identity.json' if state is not None else None,
            clock=clock, pause=clock.sleep)
        return result, calls

    def test_empty_discovery_is_unknown_not_termination(self):
        result, calls = self.exercise([[]])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertFalse(any(op == 'terminate-instances' for _, op, _ in calls))

    def test_delayed_visibility_then_terminated_without_volumes_is_proof(self):
        result, calls = self.exercise([[], [instance(identity())], [instance(identity(), 'terminated')]])
        self.assertEqual(result['cleanup_state'], 'complete')
        self.assertEqual(sum(op == 'terminate-instances' for _, op, _ in calls), 1)
        self.assertTrue(any('Name=tag:trid:experiment,Values=synthetic-only' in args
                            for _, op, args in calls if op == 'describe-volumes'))

    def test_stopped_is_not_terminated(self):
        result, calls = self.exercise([[instance(identity(), 'stopped')]])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertTrue(any(op == 'terminate-instances' for _, op, _ in calls))

    def test_owned_volume_blocks_completion(self):
        result, _ = self.exercise([[instance(identity(), 'terminated')]], [{'VolumeId': 'vol-owned'}])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertEqual(result['remaining_volumes'], ['vol-owned'])

    def test_conflicting_token_never_terminated(self):
        row = instance(identity()); row['ClientToken'] = 'foreign'
        result, calls = self.exercise([[row]])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertFalse(any(op == 'terminate-instances' for _, op, _ in calls))

    def test_conflicting_tag_never_terminated(self):
        row = instance(identity()); row['Tags'][0]['Value'] = 'foreign'
        result, calls = self.exercise([[row]])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertFalse(any(op == 'terminate-instances' for _, op, _ in calls))

    def test_multiple_instances_never_terminated(self):
        result, calls = self.exercise([[instance(identity()), instance(identity())]])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertFalse(any(op == 'terminate-instances' for _, op, _ in calls))

    def test_expected_instance_is_checked_even_with_correct_tags(self):
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            L.owned(instance(identity()), identity(), 'i-other')

    def test_persisted_exact_root_survives_missing_terminated_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            self.exercise([[instance(identity())]], state=temp)
            plan = dict(identity=identity(), ami={'RootDeviceName': '/dev/sda1'})
            evidence = L.load_volume_identity(temp, plan, 'a' * 64)
            row = instance(identity(), 'terminated')
            row['BlockDeviceMappings'] = []
            result, _ = self.exercise([[row]], evidence=evidence, state=temp)
            self.assertEqual(result['cleanup_state'], 'complete')
            self.assertEqual(result['absent_volume_ids'], ['vol-synthetic'])
            self.assertTrue(evidence['volumes']['vol-synthetic']['owned_attachments'])

    def test_empty_tag_discovery_cannot_hide_exact_volume_tag_drift(self):
        for tags in ([], [dict(Key='trid:run', Value='foreign')]):
            with self.subTest(tags=tags), tempfile.TemporaryDirectory() as temp:
                self.exercise([[instance(identity())]], state=temp)
                plan = dict(identity=identity(), ami={'RootDeviceName': '/dev/sda1'})
                evidence = L.load_volume_identity(temp, plan, 'a' * 64)
                row = instance(identity(), 'terminated'); row['BlockDeviceMappings'] = []
                result, calls = self.exercise([[row]], evidence=evidence, exact=[
                    dict(VolumeId='vol-synthetic', Tags=tags, Attachments=[])])
                self.assertEqual(result['cleanup_state'], 'unresolved')
                self.assertTrue(any('Volume ownership conflict' in error for error in result['errors']))
                self.assertFalse(any(op in ('delete-volume', 'terminate-instances', 'delete-schedule')
                                     for _, op, _ in calls))
                with self.assertRaisesRegex(RuntimeError, 'No termination proof'):
                    L.cleanup_schedule(lambda *a: self.fail('Must retain protection'), identity(), temp, result)

    def test_exact_empty_response_is_not_absence(self):
        result, _ = self.exercise([[instance(identity(), 'terminated')]], exact=[])
        self.assertEqual(result['cleanup_state'], 'unresolved')

    def test_root_never_observed_cannot_prove_cleanup(self):
        row = instance(identity(), 'terminated'); row['BlockDeviceMappings'] = []
        result, _ = self.exercise([[row]])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertEqual(result['root_volume_ids'], [])

    def test_all_recorded_ids_must_be_absent_even_after_mapping_disappears(self):
        evidence = L.volume_identity(identity(), 'a' * 64, '/dev/sda1')
        evidence.update(instance_id=instance(identity())['InstanceId'], root_volume_ids=['vol-synthetic'],
                        volumes={'vol-synthetic': dict(instance_mappings=[], owned_attachments=[]),
                                 'vol-extra': dict(instance_mappings=[], owned_attachments=[])})
        row = instance(identity(), 'terminated'); row['BlockDeviceMappings'] = []
        result, calls = self.exercise([[row]], evidence=evidence)
        self.assertEqual(result['cleanup_state'], 'complete')
        self.assertEqual(result['absent_volume_ids'], ['vol-extra', 'vol-synthetic'])
        self.assertTrue(any('--volume-ids' in args and 'vol-extra' in args for _, _, args in calls))

    def test_persisted_identity_cannot_cross_plan_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            self.exercise([[instance(identity())]], state=temp)
            with self.assertRaisesRegex(RuntimeError, 'Persisted volume identity'):
                L.load_volume_identity(temp, dict(identity=identity(), ami={'RootDeviceName': '/dev/sda1'}), 'b' * 64)

    def test_one_surviving_recorded_volume_blocks_proof_despite_empty_tags(self):
        evidence = L.volume_identity(identity(), 'a' * 64, '/dev/sda1')
        row = instance(identity(), 'terminated')
        evidence.update(instance_id=row['InstanceId'], root_volume_ids=['vol-synthetic'],
                        volumes={name: dict(instance_mappings=[], owned_attachments=[])
                                 for name in ('vol-synthetic', 'vol-extra')})
        row['BlockDeviceMappings'] = []
        result, calls = self.exercise([[row]], evidence=evidence, exact={
            'vol-synthetic': None, 'vol-extra': dict(VolumeId='vol-extra', Tags=row['Tags'], Attachments=[])})
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertEqual(result['absent_volume_ids'], ['vol-synthetic'])
        self.assertEqual(result['remaining_volumes'], ['vol-extra'])
        self.assertFalse(any(op == 'delete-volume' for _, op, _ in calls))

    def test_permission_failure_is_not_exact_absence(self):
        result, _ = self.exercise([[instance(identity(), 'terminated')]], exact={
            'vol-synthetic': L.AWSCallError('ec2', 'describe-volumes', 'An error occurred (UnauthorizedOperation)')})
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertEqual(result['absent_volume_ids'], [])

    def test_reconcile_cli_loads_persisted_root_before_proving_absence(self):
        with tempfile.TemporaryDirectory() as temp:
            self.exercise([[instance(identity())]], state=temp)
            root = Path(temp)
            plan = dict(identity=identity(), ami={'RootDeviceName': '/dev/sda1'},
                        cost=dict(control_requests=1000, termination_grace_seconds=20))
            (root / 'approval-consumed.json').write_text(json.dumps(
                dict(run_id=identity()['run_id'], plan_sha256='a' * 64)))
            row = instance(identity(), 'terminated'); row['BlockDeviceMappings'] = []
            calls = []

            def transport(service, operation, *args):
                calls.append((operation, args))
                if operation == 'describe-instances':
                    return {'Reservations': [{'Instances': [row]}]}
                if operation == 'describe-volumes':
                    if '--volume-ids' in args:
                        raise L.AWSCallError(service, operation, 'An error occurred (InvalidVolume.NotFound)')
                    return {'Volumes': []}
                self.fail(operation)

            argv = ['controller', 'reconcile', '--plan', str(root / 'plan.json'),
                    '--plan-sha256', 'a' * 64, '--root', temp, '--state', temp]
            with patch.object(sys, 'argv', argv), patch.object(L, 'read_bound', return_value=plan), \
                 patch.object(L, 'Client', return_value=transport), patch('builtins.print'):
                L.main()
            result = json.loads((root / 'reconcile-result.json').read_text())
            self.assertEqual(result['cleanup_state'], 'complete')
            self.assertEqual(result['absent_volume_ids'], ['vol-synthetic'])
            self.assertTrue(any('--volume-ids' in args for _, args in calls))


class ArtifactTests(unittest.TestCase):
    def test_authenticated_extract_creates_fresh_regular_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pin = archive_fixture(temp, [('nested/code.py', b'content', tarfile.REGTYPE)])
            target = Path(temp) / 'new'
            result = L.authenticate_archive(path, pin, target)
            self.assertEqual(set(result), {'nested/code.py'})
            self.assertEqual((target / 'nested/code.py').read_bytes(), b'content')
            with self.assertRaises(FileExistsError):
                L.authenticate_archive(path, pin, target)

    def test_traversal_link_duplicate_and_pax_fail_before_extract(self):
        cases = [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                 ('a/../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE)]
        for name, kind in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path, pin = archive_fixture(temp, [(name, b'x', kind)])
                target = Path(temp) / 'new'
                with self.assertRaises(RuntimeError):
                    L.authenticate_archive(path, pin, target)
                self.assertFalse(target.exists())
        with tempfile.TemporaryDirectory() as temp:
            path, pin = archive_fixture(temp, [('same', b'x', tarfile.REGTYPE), ('same', b'x', tarfile.REGTYPE)])
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                L.authenticate_archive(path, pin)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bundle.tar'
            with tarfile.open(path, 'w:', format=tarfile.PAX_FORMAT) as archive:
                member = tarfile.TarInfo('data'); member.size = 1
                member.pax_headers = {'comment': 'unapproved metadata'}
                archive.addfile(member, io.BytesIO(b'x'))
            pin = dict(bytes=path.stat().st_size, sha256=L.digest(path), unpacked_bytes=1,
                       files=[dict(path='data', bytes=1, sha256=hashlib.sha256(b'x').hexdigest())])
            with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                L.authenticate_archive(path, pin)

    def test_member_hash_and_envelope_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pin = archive_fixture(temp, [('data', b'1234', tarfile.REGTYPE)])
            pin['files'][0]['sha256'] = 'f' * 64
            with self.assertRaisesRegex(RuntimeError, 'hash'):
                L.authenticate_archive(path, pin)
            pin['unpacked_bytes'] = 3
            with self.assertRaises(RuntimeError):
                L.authenticate_archive(path, pin)

    def test_symlink_or_hardlinked_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pin = archive_fixture(temp, [('data', b'x', tarfile.REGTYPE)])
            link = Path(temp) / 'link.tar'; link.symlink_to(path)
            with self.assertRaises(RuntimeError):
                L.authenticate_archive(link, pin)
            os.link(path, Path(temp) / 'hard.tar')
            with self.assertRaises(RuntimeError):
                L.authenticate_archive(path, pin)

    def test_export_overflow_is_error_not_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'output.jsonl'; path.write_bytes(b'12345')
            with self.assertRaisesRegex(RuntimeError, 'overflow'):
                W.export_inventory(temp, 4)
            self.assertEqual(path.read_bytes(), b'12345')

    def test_export_rejects_hardlinks_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'output.jsonl'; path.write_bytes(b'x')
            alias = Path(temp) / 'alias'; os.link(path, alias)
            with self.assertRaisesRegex(RuntimeError, 'hardlink'):
                W.export_inventory(temp, 10)
            alias.unlink(); alias.symlink_to(path)
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                W.export_inventory(temp, 10)

    def test_unregistered_output_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'output').mkdir()
            (base / 'output/unregistered.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Unregistered'):
                W.export_outputs(base, {'output_allowlist': []}, {}, 9999999999)
            self.assertFalse((base / 'export.tar').exists())

    def test_download_rejects_non_tls_and_unapproved_redirect_origin(self):
        for url in ('http://approved/file', 'https://foreign/file', 'https://user:pass@approved/file'):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                W.check_url(url, {'approved'})
        W.check_url('https://approved/file', {'approved'})


class BarrierTests(unittest.TestCase):
    def test_exact_verified_schedule_releases_only_workload_window(self):
        plan = release_plan(); token = release(plan)
        self.assertEqual(L.verify_release(token, plan, 'a' * 64, token['instance_id'], 1001), 13600)
        for field, value in [('schedule_verified', False), ('plan_sha256', 'b' * 64),
                             ('instance_id', 'foreign'), ('deadline', 15401)]:
            bad = copy.deepcopy(token); bad[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                L.verify_release(bad, plan, 'a' * 64, token['instance_id'], 1001)
        with self.assertRaises(RuntimeError):
            L.verify_release(token, plan, 'a' * 64, token['instance_id'], 13600)

    def test_schedule_target_and_deadline_mutation_prevents_release(self):
        plan = release_plan(); token = release(plan)
        for field, value in [('State', 'DISABLED'), ('ActionAfterCompletion', 'DELETE'),
                             ('ScheduleExpression', 'at(2099-01-01T00:00:00)')]:
            bad = copy.deepcopy(token); bad['schedule'][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                L.verify_release(bad, plan, 'a' * 64, token['instance_id'], 1001)
        token['schedule']['Target']['Input'] = json.dumps({'InstanceIds': ['foreign']})
        with self.assertRaises(RuntimeError):
            L.verify_release(token, plan, 'a' * 64, token['instance_id'], 1001)

    def test_scheduler_source_arn_is_group_not_schedule(self):
        ident = identity()
        trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow',
            'Principal': {'Service': 'scheduler.amazonaws.com'}, 'Action': 'sts:AssumeRole',
            'Condition': {'StringEquals': {'aws:SourceAccount': ident['account']}, 'ArnEquals': {
                'aws:SourceArn': 'arn:aws:scheduler:ap-northeast-2:148761639313:schedule-group/synthetic-group'}}}]}
        L.verify_scheduler_trust({'trust': trust}, ident)
        trust['Statement'][0]['Condition']['ArnEquals']['aws:SourceArn'] = 'arn:aws:scheduler:ap-northeast-2:148761639313:schedule/synthetic-group/run'
        with self.assertRaisesRegex(RuntimeError, 'schedule-GROUP'):
            L.verify_scheduler_trust({'trust': trust}, ident)

    def test_no_schedule_deletion_without_termination_proof(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, 'termination proof'):
            L.cleanup_schedule(lambda *args: events.append(args), identity(), '/unused',
                               {'cleanup_state': 'unresolved'})
        self.assertEqual(events, [])

    def test_expired_deadline_prevents_subprocess_dispatch(self):
        with patch.object(W.subprocess, 'run') as run, self.assertRaisesRegex(RuntimeError, 'deadline'):
            W.command(['must-not-run'], 0)
        run.assert_not_called()

    def test_foreign_unit_is_not_stopped(self):
        with patch.object(W, 'unit_state', return_value={'Environment': 'TRID_OWNED_RUN=foreign', 'InvocationID': 'x'}), \
             patch.object(W, 'command') as command, self.assertRaisesRegex(RuntimeError, 'unowned'):
            W.stop_unit('unit', 'ours', 9999999999)
        command.assert_not_called()

    def test_observed_nonempty_cgroup_blocks_stop_proof(self):
        state = dict(Environment='TRID_OWNED_RUN=ours', InvocationID='id', ActiveState='inactive', ControlGroup='/owned')
        with patch.object(W, 'unit_state', return_value=state), patch.object(W, 'command'), \
             patch.object(W, 'cgroup_empty', return_value=False), self.assertRaisesRegex(RuntimeError, 'reap'):
            W.stop_unit('unit', 'ours', 9999999999)

    def test_successfully_collected_unit_still_requires_empty_cgroup(self):
        running = dict(Environment='TRID_OWNED_RUN=ours', InvocationID='id', ControlGroup='/owned')
        with patch.object(W, 'unit_state', side_effect=[running, {'LoadState': 'not-found'}]), \
             patch.object(W, 'command'), patch.object(W, 'cgroup_empty', return_value=True):
            self.assertEqual(W.stop_unit('unit', 'ours', 9999999999), {'LoadState': 'not-found'})


class LaunchTests(unittest.TestCase):
    """Exercise the actual dispatcher against a deterministic transport boundary."""
    def fixture(self, root):
        plan = release_plan()
        path, bundle = archive_fixture(root, [('fixture.txt', b'synthetic', tarfile.REGTYPE)])
        bundle['local_path'] = path.name
        plan.update(bundle=bundle, sources={}, instance_type='g6e.2xlarge',
            ami={'ImageId': 'ami-0123456789abcdef0', 'RootDeviceName': '/dev/sda1'},
            network=dict(subnet_id='subnet-synthetic', security_group_id='sg-synthetic', availability_zone='ap-northeast-2a'),
            cost={'termination_grace_seconds': 20},
            ebs_kms_key_arn='arn:aws:kms:ap-northeast-2:148761639313:key/11111111-2222-3333-4444-555555555555')
        return plan

    def run_launch(self, temp, *, ambiguous=False, schedule_wrong=False, prerequisites_wrong=False, root_key_override=None):
        plan = self.fixture(temp); clock = Clock(); events = []
        row = instance(plan['identity'])
        cloud = dict(launched=False, terminated=False, released=False, schedule=None,
                     root_key='arn:aws:kms:ap-northeast-2:148761639313:key/ffffffff-ffff-ffff-ffff-ffffffffffff')
        expectations = prerequisite_expectations()
        prerequisites = actual_prerequisites(expectations)
        if prerequisites_wrong:
            prerequisites['commands']['aws']['found_path'] = '/unexpected/aws'

        def transport(service, operation, *args, **kwargs):
            events.append((operation, args))
            if operation == 'describe-instances':
                if not cloud['launched']:
                    return {'Reservations': []}
                current = copy.deepcopy(row)
                if cloud['terminated'] or cloud['released']:
                    current['State']['Name'] = 'terminated'
                return {'Reservations': [{'Instances': [current]}]}
            if operation == 'head-object':
                key = args[args.index('--key') + 1]
                pin = {'bytes': 100, 'sha256': 'a' * 64} if key.endswith('plan.json') else plan['bundle']
                return {'ContentLength': pin['bytes'], 'Metadata': {'sha256': pin['sha256']}}
            if operation == 'run-instances':
                request = json.loads(args[args.index('--cli-input-json') + 1])
                # EC2 otherwise uses the account default, which deliberately
                # differs from the approved key in this behavioral transport.
                cloud['root_key'] = request['BlockDeviceMappings'][0]['Ebs'].get('KmsKeyId', cloud['root_key'])
                cloud['launched'] = True
                if ambiguous:
                    raise TimeoutError('response lost after EC2 accepted request')
                return {'Instances': [copy.deepcopy(row)]}
            if operation == 'terminate-instances':
                cloud['terminated'] = True
                return {}
            if operation == 'describe-volumes':
                if '--volume-ids' in args:
                    if cloud['terminated'] or cloud['released']:
                        raise L.AWSCallError(service, operation, 'An error occurred (InvalidVolume.NotFound)')
                    volume = dict(VolumeId='vol-synthetic', Tags=row['Tags'], Size=120,
                        VolumeType='gp3', Iops=3000, Throughput=125, Encrypted=True,
                        Attachments=[{'InstanceId': row['InstanceId']}])
                    key = cloud['root_key'] if root_key_override is None else root_key_override
                    if key:
                        volume['KmsKeyId'] = key
                    return {'Volumes': [volume]}
                return {'Volumes': []}
            if operation == 'describe-instance-attribute':
                return {'InstanceInitiatedShutdownBehavior': {'Value': 'terminate'}}
            if operation == 'create-schedule':
                cloud['schedule'] = json.loads(args[args.index('--cli-input-json') + 1])
                return {}
            if operation == 'get-schedule':
                observed = copy.deepcopy(cloud['schedule'])
                if schedule_wrong:
                    observed['State'] = 'DISABLED'
                return observed
            if operation == 'delete-schedule':
                self.assertTrue(cloud['terminated'] or cloud['released'])
                return {}
            if operation == 'get-object':
                key = args[args.index('--key') + 1]
                if key.endswith('ready.json'):
                    Path(args[-1]).write_text(json.dumps(dict(run_id=plan['identity']['run_id'],
                        plan_sha256='a' * 64, instance_id=row['InstanceId'], gpu=gpu(),
                        prerequisites=prerequisites)))
                    return {}
                raise RuntimeError('Synthetic transport has no native completion evidence')
            if operation == 'put-object':
                key = args[args.index('--key') + 1]
                if key.endswith('release.json'):
                    self.assertIsNotNone(cloud['schedule'])
                    cloud['released'] = True
                return {}
            raise AssertionError((service, operation, args))

        state = Path(temp) / 'state'
        with patch.object(L, 'validate_plan', return_value={}), patch.object(L, 'validate_data', return_value=[]), \
             patch.object(L, 'validate_cost_admission', return_value={}), \
             patch.object(L, 'validate_bootstrap_admission', return_value=expectations), \
             patch.object(L, 'live_preflight', return_value={'synthetic': True}), \
             patch.object(L, 'bootstrap_text', return_value='#!/bin/bash\nexit 0\n'):
            result = L.launch(plan, {}, 'a' * 64, temp, state, transport, clock=clock, pause=clock.sleep)
            with self.assertRaises(FileExistsError):
                L.launch(plan, {}, 'a' * 64, temp, state, transport, clock=clock, pause=clock.sleep)
        return result, events

    def test_ambiguous_launch_is_never_resent_and_owned_resource_is_reconciled(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.run_launch(temp, ambiguous=True)
            persisted = json.loads((Path(temp) / 'state/volume-identity.json').read_text())
            self.assertEqual(persisted['root_volume_ids'], ['vol-synthetic'])
            self.assertTrue(persisted['volumes']['vol-synthetic']['owned_attachments'])
        self.assertEqual(sum(op == 'run-instances' for op, _ in events), 1)
        self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
        self.assertEqual(result['state'], 'failed')
        self.assertFalse(any(op == 'create-schedule' for op, _ in events))
        self.assertFalse(result['training_success'])

    def test_wrong_schedule_never_releases_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.run_launch(temp, schedule_wrong=True)
        released = [args for op, args in events if op == 'put-object' and any(str(a).endswith('release.json') for a in args)]
        self.assertEqual(released, [])
        self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
        self.assertFalse(result['training_success'])

    def test_actual_host_prerequisite_mismatch_prevents_model_release(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.run_launch(temp, prerequisites_wrong=True)
        self.assertEqual(result['state'], 'failed')
        self.assertIn('Actual postboot command capability proof differs: aws', result['error'])
        self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
        self.assertTrue(any(op == 'put-object' and any(str(a).endswith('termination.json') for a in args)
                            for op, args in events))
        self.assertFalse(any(op == 'put-object' and any(str(a).endswith('release.json') for a in args)
                             for op, args in events))

    def test_account_default_key_drift_cannot_change_explicit_root_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.run_launch(temp)
        self.assertTrue(any(op == 'put-object' and any(str(a).endswith('release.json') for a in args)
                            for op, args in events))
        self.assertEqual(result['cleanup']['cleanup_state'], 'complete')

    def test_mismatched_or_missing_created_root_key_denies_release_and_reconciles(self):
        for key in ('', 'arn:aws:kms:ap-northeast-2:148761639313:key/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                result, events = self.run_launch(temp, root_key_override=key)
                self.assertEqual(result['state'], 'failed')
                self.assertIn('root volume encryption key differs', result['error'])
                self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
                self.assertFalse(result['training_success'])
                self.assertTrue(any(op == 'terminate-instances' for op, _ in events))
                self.assertFalse(any(op == 'put-object' and any(str(a).endswith('release.json') for a in args)
                                     for op, args in events))
                self.assertFalse((Path(temp) / 'state/release-intent.json').exists())

    def test_release_follows_schedule_readback_and_termination_is_not_training_success(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.run_launch(temp)
        release_index = next(i for i, (op, args) in enumerate(events)
                             if op == 'put-object' and any(str(a).endswith('release.json') for a in args))
        self.assertTrue(any(op == 'get-schedule' for op, _ in events[:release_index]))
        self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
        self.assertIn('evidence_error', result)
        self.assertFalse(result['training_success'])

    def test_missing_bootstrap_proof_blocks_before_any_cloud_call(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp)
            plan['runtime']['bootstrap'] = dict(
                admission={'path': 'not-proven.json', 'sha256': '0' * 64})
            calls = []
            with patch.object(L, 'validate_plan', return_value={}), patch.object(L, 'validate_data', return_value=[]), \
                 patch.object(L, 'validate_cost_admission', return_value={}):
                result = L.launch(plan, {}, 'a' * 64, temp, Path(temp) / 'state',
                                  lambda *args, **kwargs: calls.append(args))
            self.assertEqual(calls, [])
            self.assertEqual(result['state'], 'failed')
            self.assertFalse((Path(temp) / 'state/launch-dispatched.json').exists())


class FiniteRecipeTests(unittest.TestCase):
    def test_old_identity_cannot_use_new_approval_path(self):
        plan = dict(schema='trid-grounded-hf-v1', identity={**identity(), 'run_id': 'qwen013-cuda-20261111030853'})
        with self.assertRaisesRegex(RuntimeError, 'Fresh'):
            L.validate_plan(plan, {}, 'a' * 64, now=1000)

    def test_changed_limits_cannot_use_fresh_identity(self):
        plan = dict(schema='trid-grounded-hf-v1', identity=identity(), limits={**L.LIMITS, 'rss_bytes': 49 * L.GIB},
                    instance_type='g6e.2xlarge', vcpus=8, host_mib=65536, architecture='x86_64')
        with self.assertRaisesRegex(RuntimeError, 'resource envelope'):
            L.validate_plan(plan, {}, 'a' * 64, now=1000)

    def test_wrapped_extra_examples_are_rejected(self):
        plan = dict(schema='trid-grounded-hf-v1', identity=identity(), limits=L.LIMITS,
            instance_type='g6e.2xlarge', vcpus=8, host_mib=65536, architecture='x86_64',
            recipe={**L.RECIPE, 'rows': 5, 'epochs': 3, 'steps': 4})
        with self.assertRaisesRegex(RuntimeError, 'exactly three epochs'):
            L.validate_plan(plan, {}, 'a' * 64, now=1000)


class DataAdmissionTests(unittest.TestCase):
    def fixture(self, directory, *, target_in_blind=False, admitted=True):
        root = Path(directory)
        messages = [{'role': 'system', 'content': 'Synthetic fixture only.'},
                    {'role': 'user', 'content': 'Synthetic training evidence.'}]
        rows = [dict(id=f'train-{i}', task='synthetic', split='train', messages=messages, answer='Synthetic')
                for i in range(4)]
        (root / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest = {'train': [dict(path='rows.jsonl', sha256=L.digest(root / 'rows.jsonl'),
                                   adapter='grounded_messages', epochs=3)]}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        laws = ['law:009916', 'law:005046', 'law:005048']
        cluster = 'local-government-service-compensation-connected'
        cases, case_pins = [], []
        for index, family in enumerate(laws):
            prompt = [messages[0], {'role': 'user', 'content': 'Synthetic blind evidence.'}]
            cases.append(dict(id=f'blind-{index + 1}', messages=prompt))
            case_pins.append(dict(id=cases[-1]['id'], kind='grounded', family=family,
                                  prompt_sha256=L.object_sha(prompt)))
        for index in range(6):
            prompt = [{'role': 'user', 'content': f'Synthetic general retention {index}.'}]
            cases.append(dict(id=f'general-{index + 1}', messages=prompt))
            case_pins.append(dict(id=cases[-1]['id'], kind='synthetic-general-retention',
                                  family='synthetic-general-retention', prompt_sha256=L.object_sha(prompt)))
        if target_in_blind:
            cases[0]['answer'] = 'Forbidden label'
        (root / 'blind.jsonl').write_text(''.join(json.dumps(case) + '\n' for case in cases))
        allocation = dict(schema_version=1, owner='Main', status='EVALUATION_ADMITTED',
            train_allocation_changed=False, prospective_evaluation_cluster=dict(
                id=cluster, families=laws, separate_validation_split=False),
            synthetic_general_retention_count=6, questions_authored_after_source_closure=True,
            blind_inputs_sha256=L.digest(root / 'blind.jsonl'), blind_cases_sha256=L.object_sha(case_pins))
        (root / 'allocation.json').write_text(json.dumps(allocation))
        admission = dict(status='admitted' if admitted else 'quarantined', run_id=identity()['run_id'],
            manifest_sha256=L.digest(root / 'manifest.json'), blind_inputs_sha256=L.digest(root / 'blind.jsonl'),
            evaluation_allocation_sha256=L.digest(root / 'allocation.json'), rights_reviewed=True,
            source_verified=True, overlap_reviewed=True, targets_reviewed=True, protected_sets_excluded=True)
        (root / 'admission.json').write_text(json.dumps(admission))
        files = [dict(path=name, sha256=L.digest(root / name), bytes=(root / name).stat().st_size)
                 for name in ('rows.jsonl', 'manifest.json', 'blind.jsonl', 'admission.json', 'allocation.json')]
        return dict(identity=identity(), recipe={**L.RECIPE, 'rows': 4, 'epochs': 3, 'steps': 6},
            evaluation=dict(cluster_id=cluster, grounded_families=laws, synthetic_general_retention_count=6,
                            checkpoint_selection='final_only', adaptive_selection=False),
            bundle={'files': files}, data=dict(manifest=files[1], blind_inputs=files[2], admission=files[3],
                allocation=files[4], blind_cases=case_pins))

    def test_sealed_synthetic_data_preserves_exact_blind_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp)
            cases = L.validate_data(temp, plan)
            self.assertEqual(cases[0]['messages'][1]['content'], 'Synthetic blind evidence.')

    def test_plain_chat_and_grounded_messages_preserve_verbatim_content(self):
        text = '  Plain retention prompt.\nKeep whitespace.  '
        expected = [{'role': 'user', 'content': text}]
        with self.assertRaisesRegex(RuntimeError, 'extra fields'):
            L.blind_messages({'id': 'plain', 'prompt': text})
        self.assertEqual(L.blind_messages({'id': 'plain', 'messages': expected}), expected)
        grounded = [{'role': 'system', 'content': 'Use the supplied JSON evidence.'},
                    {'role': 'user', 'content': '{\"documents\": [{\"id\": \"D1\", \"text\": \"evidence\"}]}'}]
        self.assertEqual(L.blind_messages({'id': 'grounded', 'messages': grounded}), grounded)
        for field in ('answer', 'target', 'gold', 'labels'):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'Blind labels'):
                L.blind_messages({'id': 'plain', 'messages': expected, field: 'not allowed'})

    def test_cluster_is_sealed_data_identity_not_resource_run(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp)
            L.validate_evaluation(plan)
            self.assertNotEqual(plan['evaluation']['cluster_id'], plan['identity']['run_id'])
            plan['evaluation']['cluster_id'] = plan['identity']['run_id']
            with self.assertRaisesRegex(RuntimeError, 'Sealed data cluster'):
                L.validate_evaluation(plan)

    def test_retention_is_separately_counted_and_never_assigned_to_a_law(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp)
            retention = plan['data']['blind_cases'][-1]
            retention['family'] = 'law:009916'
            with self.assertRaisesRegex(RuntimeError, 'masquerade'):
                L.validate_evaluation(plan)
            retention['family'] = 'synthetic-general-retention'
            plan['data']['blind_cases'].pop()
            with self.assertRaisesRegex(RuntimeError, 'exactly six'):
                L.validate_evaluation(plan)

    def test_source_only_allocation_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp)
            path = Path(temp) / 'allocation.json'
            allocation = json.loads(path.read_text())
            allocation['status'] = 'SOURCE_CLOSURE_PENDING_NO_QUESTIONS_AUTHORED'
            path.write_text(json.dumps(allocation))
            plan['data']['allocation']['sha256'] = L.digest(path)
            with self.assertRaisesRegex(RuntimeError, 'not sealed'):
                L.validate_data(temp, plan)

    def test_target_bearing_blind_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp, target_in_blind=True)
            with self.assertRaisesRegex(RuntimeError, 'Blind labels'):
                L.validate_data(temp, plan)

    def test_quarantined_manifest_is_not_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp, admitted=False)
            with self.assertRaisesRegex(RuntimeError, 'not sealed'):
                L.validate_data(temp, plan)

    def test_data_hash_and_approved_count_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(temp)
            plan['recipe']['rows'] = 8
            with self.assertRaisesRegex(RuntimeError, 'row count'):
                L.validate_data(temp, plan)
            plan['recipe']['rows'] = 4
            (Path(temp) / 'rows.jsonl').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                L.validate_data(temp, plan)


class DownloadBoundaryTests(unittest.TestCase):
    def exercise(self, directory, payload, expected):
        class Response(io.BytesIO):
            status = 200
            headers = {}
        class Opener:
            def open(self, *args, **kwargs):
                return Response(payload)
        row = dict(path='weight.safetensors', bytes=len(expected),
                   sha256=hashlib.sha256(expected).hexdigest(), url='https://approved/model')
        with patch.object(W.urllib.request, 'build_opener', return_value=Opener()):
            return W.download(row, directory, {'approved'}, 9999999999, [1])

    def test_exact_bytes_hash_verified_and_second_acquisition_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.exercise(temp, b'weights', b'weights')
            self.assertEqual(path.read_bytes(), b'weights')
            with self.assertRaises(FileExistsError):
                self.exercise(temp, b'weights', b'weights')

    def test_overflow_and_short_response_fail_without_retry(self):
        for actual in (b'weightsoverflow', b'we', b'WEIGHTS'):
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(RuntimeError):
                    self.exercise(temp, actual, b'weights')

    def test_request_cap_prevents_open(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(W.urllib.request, 'build_opener') as opened:
            with self.assertRaisesRegex(RuntimeError, 'request budget'):
                W.download(dict(path='weight', url='https://approved/file', bytes=1, sha256='a' * 64),
                           temp, {'approved'}, 9999999999, [0])
            opened.assert_not_called()


class CostClosureTests(unittest.TestCase):
    def fixture(self):
        return dict(bundle={'bytes': 10240}, network={'egress_byte_cap': 2 * L.GIB}, cost=dict(
            cap_usd=50, retention_days=1, ancillary_bound_usd=1,
            get_requests=512, put_requests=16, control_requests=1000, download_requests=100,
            network_egress_bytes=4 * L.GIB, retained_bytes=3 * L.GIB, termination_grace_seconds=300,
            rates=dict(ec2_hour=3, ipv4_hour=.01, ebs_gb_month=.1, s3_gb_month=.03,
                       egress_gb=.15, s3_get=.000001, s3_put=.000005, scheduler_invocation=.000001)))

    def test_retention_and_requests_enter_bound_without_claiming_hard_cap(self):
        plan = self.fixture()
        original = L.cost_bound(plan)
        plan['cost']['retention_days'] = 30
        increased = L.cost_bound(plan)
        self.assertGreater(increased['additional_pretax_bound_usd'], original['additional_pretax_bound_usd'])
        self.assertFalse(original['spending_hard_cap'])
        self.assertFalse(original['invoice'])

    def test_missing_get_or_unbounded_storage_fails_admission(self):
        plan = self.fixture(); del plan['cost']['rates']['s3_get']
        with self.assertRaisesRegex(RuntimeError, 'GET'):
            L.cost_bound(plan)
        plan = self.fixture(); plan['cost']['retention_days'] = 0
        with self.assertRaises(RuntimeError):
            L.cost_bound(plan)

    def test_above_cap_or_nonfinite_price_fails(self):
        for value in (100, float('nan'), float('inf')):
            plan = self.fixture(); plan['cost']['rates']['ec2_hour'] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                L.cost_bound(plan)


class LifecycleOverlapTests(unittest.TestCase):
    def fixture(self, extra=()):
        ident = identity()
        configuration = dict(TransitionDefaultMinimumObjectSize='all_storage_classes_128K',
            Rules=[dict(ID=prefix, Status='Enabled', Filter={'Prefix': prefix}, Expiration={'Days': 1})
                   for prefix in (ident['code_prefix'], ident['output_prefix'])] + list(extra))
        return dict(identity=ident, cost=dict(retention_days=1, bucket_lifecycle_configuration=configuration))

    def test_ancestor_global_and_descendant_transitions_are_unpriced(self):
        for selector in ({'Prefix': 'code/'}, {'Prefix': ''}, {},
                         {'Prefix': identity()['output_prefix'] + 'nested/'}):
            with self.subTest(selector=selector):
                plan = self.fixture([dict(Status='Enabled', Filter=selector,
                    Transitions=[dict(Days=0, StorageClass='GLACIER')])])
                with self.assertRaisesRegex(RuntimeError, 'Unpriced overlapping'):
                    L.validate_lifecycle_configuration(plan, plan['cost']['bucket_lifecycle_configuration'])

    def test_legacy_global_prefix_and_noncurrent_transition_rejected(self):
        plan = self.fixture([dict(Status='Enabled', Prefix='',
            NoncurrentVersionTransitions=[dict(NoncurrentDays=1, StorageClass='STANDARD_IA')])])
        with self.assertRaisesRegex(RuntimeError, 'Unpriced overlapping'):
            L.validate_lifecycle_configuration(plan, plan['cost']['bucket_lifecycle_configuration'])

    def test_ambiguous_tag_size_and_and_filters_fail_closed(self):
        for selector in ({'Tag': {'Key': 'class', 'Value': 'archive'}},
                         {'ObjectSizeGreaterThan': 100},
                         {'And': {'Prefix': 'code/', 'Tags': [{'Key': 'class', 'Value': 'archive'}]}},
                         {'Prefix': 'code/', 'ObjectSizeLessThan': 100}):
            with self.subTest(selector=selector):
                plan = self.fixture([dict(Status='Enabled', Filter=selector, Expiration={'Days': 1})])
                with self.assertRaisesRegex(RuntimeError, 'Ambiguous lifecycle'):
                    L.validate_lifecycle_configuration(plan, plan['cost']['bucket_lifecycle_configuration'])

    def test_unrelated_transition_and_disabled_global_rule_are_preserved(self):
        plan = self.fixture([dict(Status='Enabled', Filter={'Prefix': 'unrelated/'},
                                 Transitions=[dict(Days=1, StorageClass='GLACIER')]),
                             dict(Status='Disabled', Filter={}, Transitions=[dict(Days=1, StorageClass='GLACIER')])])
        observed = copy.deepcopy(plan['cost']['bucket_lifecycle_configuration'])
        self.assertEqual(L.validate_lifecycle_configuration(plan, observed), observed)
        self.assertEqual(observed, plan['cost']['bucket_lifecycle_configuration'])

    def test_full_observation_drift_rejected_even_for_unrelated_rules_or_defaults(self):
        for field in ('rule', 'default'):
            with self.subTest(field=field):
                plan = self.fixture()
                observed = copy.deepcopy(plan['cost']['bucket_lifecycle_configuration'])
                if field == 'rule':
                    observed['Rules'].append(dict(Status='Disabled', Filter={'Prefix': 'unrelated/'}))
                else:
                    observed['TransitionDefaultMinimumObjectSize'] = 'varies_by_storage_class'
                with self.assertRaisesRegex(RuntimeError, 'Full bucket lifecycle configuration changed'):
                    L.validate_lifecycle_configuration(plan, observed)

    def test_sealed_cost_receipt_binds_complete_lifecycle_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture()
            plan.update(ancillary_checks={}, price_queries={}, checked_at='2026-11-12T00:00:00Z')
            plan['cost']['ancillary_bound_usd'] = 1
            status_path = Path(temp) / 'policy-status.json'
            status_path.write_text(json.dumps(dict(
                query=['aws', 's3api', 'get-bucket-policy-status', '--bucket', plan['identity']['bucket'],
                       '--region', plan['identity']['region']],
                exit_code=254, response=None, error_code='NoSuchBucketPolicy',
                stderr='An error occurred (NoSuchBucketPolicy) when calling the GetBucketPolicyStatus operation')))
            status_pin = dict(path=status_path.name, sha256=L.digest(status_path), bytes=status_path.stat().st_size)
            plan['bundle'] = {'files': [status_pin]}
            plan['bucket_policy_status'] = dict(state='absent', error_code='NoSuchBucketPolicy')
            key_pin = synthetic_ebs_evidence(temp, plan)
            receipt = dict(schema='trid-grounded-hf-cost-admission-v1', status='admitted',
                run_id=identity()['run_id'], account=identity()['account'], region=identity()['region'],
                ancillary_bound_usd=1, ancillary_checks_sha256=L.object_sha({}), pricing_products_sha256={},
                bucket_lifecycle_configuration_sha256=L.object_sha(plan['cost']['bucket_lifecycle_configuration']),
                bucket_policy_status_sha256=L.object_sha(plan['bucket_policy_status']),
                bucket_policy_status_evidence=status_pin,
                ebs_kms_key_arn=plan['ebs_kms_key_arn'], ebs_kms_key_evidence=key_pin,
                no_unpriced_services=True, retention_closure_reviewed=True,
                network_request_closure_reviewed=True, checked_at=plan['checked_at'])
            path = Path(temp) / 'cost.json'; path.write_text(json.dumps(receipt))
            plan['ancillary_evidence'] = dict(path=path.name, sha256=L.digest(path))
            self.assertEqual(L.validate_cost_admission(temp, plan), receipt)
            plan['cost']['bucket_lifecycle_configuration']['Rules'].append(
                dict(Status='Disabled', Filter={'Prefix': 'unrelated/'}))
            with self.assertRaisesRegex(RuntimeError, 'cost admission is missing'):
                L.validate_cost_admission(temp, plan)


class BootstrapAdmissionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / 'evidence.txt').write_text('Synthetic unit-test evidence; not an operational receipt.')
        evidence = dict(path='evidence.txt', sha256=L.digest(root / 'evidence.txt'))
        bootstrap = dict(nvidia_modprobe={'path': 'nvidia-modprobe.deb', 'sha256': 'b' * 64})
        receipt = dict(schema='trid-grounded-hf-bootstrap-admission-v1', status='admitted',
            image_id='ami-040ebe879ae70cc30', owner_id='898082745236',
            no_unpinned_package_installation=True, evidence=[evidence],
            same_ami_preinstalled_venv_verified=True, ubuntu_24_04_amd64_abi_verified=True,
            artifact_inventory_sha256=L.object_sha(bootstrap['nvidia_modprobe']))
        receipt.update(prerequisite_expectations())
        (root / 'bootstrap-admission.json').write_text(json.dumps(receipt))
        bootstrap['admission'] = dict(path='bootstrap-admission.json', sha256=L.digest(root / 'bootstrap-admission.json'))
        plan = dict(ami={'ImageId': receipt['image_id'], 'OwnerId': receipt['owner_id']},
                    runtime={'python_version': '3.12.3', 'bootstrap': bootstrap},
                    bundle={'files': [evidence]})
        return plan, receipt

    def reseal(self, directory, plan, receipt):
        path = Path(directory) / 'bootstrap-admission.json'
        path.write_text(json.dumps(receipt))
        plan['runtime']['bootstrap']['admission']['sha256'] = L.digest(path)

    def test_unverified_stock_venv_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, receipt = self.fixture(temp)
            receipt['same_ami_preinstalled_venv_verified'] = False
            self.reseal(temp, plan, receipt)
            with self.assertRaisesRegex(RuntimeError, 'Known-unknown'):
                L.validate_bootstrap_admission(temp, plan)

    def test_same_ami_venv_proof_does_not_assume_modprobe_preinstalled(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            result = L.validate_bootstrap_admission(temp, plan)
            self.assertNotIn('nvidia-modprobe', result['expected_commands'])
            self.assertTrue(result['same_ami_preinstalled_venv_verified'])
            plan['runtime']['bootstrap']['nvidia_modprobe']['sha256'] = 'd' * 64
            with self.assertRaisesRegex(RuntimeError, 'changed historical artifact'):
                L.validate_bootstrap_admission(temp, plan)

    def test_unbound_expectations_or_changed_evidence_blocks_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, receipt = self.fixture(temp)
            receipt['prerequisite_expectations_bound'] = False
            self.reseal(temp, plan, receipt)
            with self.assertRaisesRegex(RuntimeError, 'Known-unknown'):
                L.validate_bootstrap_admission(temp, plan)
            receipt['prerequisite_expectations_bound'] = True
            self.reseal(temp, plan, receipt)
            (Path(temp) / 'evidence.txt').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'evidence differs'):
                L.validate_bootstrap_admission(temp, plan)

    def test_historical_unknown_paths_are_expectations_not_postboot_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, receipt = self.fixture(temp)
            admitted = L.validate_bootstrap_admission(temp, plan)
            self.assertEqual(admitted['historical_evidence_levels']['useradd'], 'unobserved')
            with self.assertRaisesRegex(RuntimeError, 'Actual postboot prerequisite'):
                L.verify_prerequisites(receipt, admitted)
            actual = actual_prerequisites(admitted)
            self.assertEqual(L.verify_prerequisites(actual, admitted), actual)

    def test_missing_or_unsafe_postboot_expectations_are_rejected(self):
        for mutation in ('missing_name', 'duplicate_name', 'unsafe_probe', 'unknown_level', 'relative_path'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                plan, receipt = self.fixture(temp)
                if mutation == 'missing_name':
                    receipt['postboot_required_names'].remove('aws')
                elif mutation == 'duplicate_name':
                    receipt['postboot_required_names'].append('aws')
                elif mutation == 'unsafe_probe':
                    receipt['expected_commands']['shutdown']['probe_args'] = ['now']
                elif mutation == 'unknown_level':
                    receipt['historical_evidence_levels']['aws'] = 'verified'
                else:
                    receipt['expected_commands']['aws']['path'] = 'aws'
                self.reseal(temp, plan, receipt)
                with self.assertRaisesRegex(RuntimeError, 'postboot command expectations'):
                    L.validate_bootstrap_admission(temp, plan)

    def test_actual_command_and_version_mutations_fail_release_barrier(self):
        expected = prerequisite_expectations()
        for mutation in ('missing', 'extra', 'false', 'found_path', 'resolved_path', 'hash', 'probe',
                         'stdout', 'python_version', 'venv_package_version', 'aws_cli_version'):
            with self.subTest(mutation=mutation):
                actual = actual_prerequisites(expected)
                command = actual['commands']['aws']
                if mutation == 'missing':
                    del actual['commands']['shutdown']
                elif mutation == 'extra':
                    actual['commands']['foreign'] = command
                elif mutation == 'false':
                    actual['prerequisites_verified'] = False
                elif mutation == 'found_path':
                    command['found_path'] = '/wrong/aws'
                elif mutation == 'resolved_path':
                    command['resolved_path'] = 'relative'
                elif mutation == 'hash':
                    command['sha256'] = 'd' * 64
                elif mutation == 'probe':
                    command['probe_args'] = ['configure']
                elif mutation == 'stdout':
                    del command['stdout']
                else:
                    actual[mutation] = 'different'
                with self.assertRaisesRegex(RuntimeError, 'Actual postboot'):
                    L.verify_prerequisites(actual, expected)

    def test_optional_expected_hash_still_requires_actual_hash_observation(self):
        expected = prerequisite_expectations()
        del expected['expected_commands']['aws']['sha256']
        actual = actual_prerequisites(expected)
        actual['commands']['aws']['sha256'] = 'd' * 64
        self.assertEqual(L.verify_prerequisites(actual, expected), actual)
        del actual['commands']['aws']['sha256']
        with self.assertRaisesRegex(RuntimeError, 'Actual postboot command'):
            L.verify_prerequisites(actual, expected)

    def test_termination_ticket_requires_actual_schedule_before_dependency_acquisition(self):
        plan = release_plan()
        ticket = release(plan)
        del ticket['gpu']
        self.assertEqual(L.verify_termination_ticket(ticket, plan, 'a' * 64, ticket['instance_id'], 1001), 15400)
        ticket['schedule_verified'] = False
        with self.assertRaisesRegex(RuntimeError, 'ticket rejected'):
            L.verify_termination_ticket(ticket, plan, 'a' * 64, ticket['instance_id'], 1001)

    def test_only_hash_verified_helper_is_extracted_from_deb_data_tar(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = b'synthetic-not-executable'
            path, _ = archive_fixture(temp, [('./usr/bin/nvidia-modprobe', binary, tarfile.REGTYPE)])
            pin = dict(max_unpacked_bytes=path.stat().st_size,
                member=dict(path='usr/bin/nvidia-modprobe', bytes=len(binary), sha256=hashlib.sha256(binary).hexdigest()))
            target = Path(temp) / 'helper'
            W.extract_bootstrap_member(path, pin, target)
            self.assertEqual(target.read_bytes(), binary)
            pin['member']['sha256'] = 'f' * 64
            with self.assertRaisesRegex(RuntimeError, 'hash differs'):
                W.extract_bootstrap_member(path, pin, Path(temp) / 'rejected')
            self.assertFalse((Path(temp) / 'rejected').exists())

    def test_bootstrap_archive_links_and_traversal_fail_before_writing(self):
        for name, kind in (('../escape', tarfile.REGTYPE), ('usr/bin/nvidia-modprobe', tarfile.SYMTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path, _ = archive_fixture(temp, [(name, b'x', kind)])
                pin = dict(max_unpacked_bytes=path.stat().st_size,
                    member=dict(path='usr/bin/nvidia-modprobe', bytes=1, sha256=hashlib.sha256(b'x').hexdigest()))
                target = Path(temp) / 'helper'
                with self.assertRaises(RuntimeError):
                    W.extract_bootstrap_member(path, pin, target)
                self.assertFalse(target.exists())


class EBSKeyAdmissionTests(unittest.TestCase):
    def test_actual_key_evidence_is_authenticated_and_bound_to_exact_arn(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = {'identity': identity()}
            pin = synthetic_ebs_evidence(temp, plan)
            self.assertEqual(L.validate_ebs_key_evidence(temp, plan, pin)['Arn'], plan['ebs_kms_key_arn'])
            plan['ebs_kms_key_arn'] = plan['ebs_kms_key_arn'].replace('11111111', 'aaaaaaaa')
            with self.assertRaisesRegex(RuntimeError, 'Observed EBS key differs'):
                L.validate_ebs_key_evidence(temp, plan, pin)

    def test_customer_disabled_other_account_or_failed_key_evidence_is_rejected(self):
        mutations = [
            lambda raw: raw['response']['KeyMetadata'].update(KeyManager='CUSTOMER'),
            lambda raw: raw['response']['KeyMetadata'].update(Enabled=False, KeyState='Disabled'),
            lambda raw: raw['response']['KeyMetadata'].update(AWSAccountId='999999999999'),
            lambda raw: raw.update(exit_code=254, error_code='AccessDenied'),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temp:
                plan = {'identity': identity()}
                pin = synthetic_ebs_evidence(temp, plan, mutate)
                with self.assertRaises(RuntimeError):
                    L.validate_ebs_key_evidence(temp, plan, pin)

    def test_changed_raw_key_evidence_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = {'identity': identity()}
            pin = synthetic_ebs_evidence(temp, plan)
            path = Path(temp) / pin['path']
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaises(RuntimeError):
                L.validate_ebs_key_evidence(temp, plan, pin)

    def test_alias_or_cross_region_key_cannot_replace_explicit_observed_arn(self):
        for value in ('alias/aws/ebs',
                      'arn:aws:kms:us-east-1:148761639313:key/11111111-2222-3333-4444-555555555555'):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'Explicit same-account/region'):
                L.ebs_key_arn(dict(identity=identity(), ebs_kms_key_arn=value))

    def test_approval_for_another_key_cannot_authorize_plan(self):
        plan = dict(schema='trid-grounded-hf-v1', identity=identity(), limits=L.LIMITS,
            instance_type='g6e.2xlarge', vcpus=8, host_mib=65536, architecture='x86_64',
            recipe={**L.RECIPE, 'rows': 34, 'epochs': 3, 'steps': 51},
            ebs_kms_key_arn='arn:aws:kms:ap-northeast-2:148761639313:key/11111111-2222-3333-4444-555555555555')
        approval = dict(schema='trid-grounded-hf-approval-v1', run_id=plan['identity']['run_id'],
            plan_sha256='a' * 64, instance_limit=1, additional_pretax_usd=50, recipe=plan['recipe'],
            limits=L.LIMITS, ebs_kms_key_arn=plan['ebs_kms_key_arn'].replace('11111111', 'aaaaaaaa'),
            automatic_retry=False, approved=True)
        with self.assertRaisesRegex(RuntimeError, 'Exact new final approval'):
            L.validate_plan(plan, approval, 'a' * 64, now=1000)


if __name__ == '__main__':
    unittest.main()
