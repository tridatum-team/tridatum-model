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

    def run_launch(self, temp, *, ambiguous=False, schedule_wrong=False, prerequisites_wrong=False,
                   root_key_override=None, ordered=False, state_name='state', dispatch_state=None,
                   preflight_error=None, remote_intent_error=False, remote_intent_delay=0):
        plan = self.fixture(temp); clock = Clock(); events = []
        if ordered:
            plan['schema'] = L.INFERENCE_SCHEMA
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
                if L.inference_mode(plan):
                    persisted = json.loads((state / 'controller-result.json').read_text())
                    self.assertTrue(persisted['dispatch_attempted'])
                    self.assertTrue(persisted['remote_intent_attempted'])
                    self.assertEqual(persisted['phase'], 'run_instances')
                    self.assertTrue((state / 'launch-dispatched.json').exists())
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
                if L.inference_mode(plan):
                    return {'ScheduleArn': (f"arn:aws:scheduler:{plan['identity']['region']}:"
                        f"{plan['identity']['account']}:schedule/{plan['identity']['schedule_group']}/"
                        f"{plan['identity']['run_id']}")}
                return {}
            if operation == 'get-schedule':
                observed = copy.deepcopy(cloud['schedule'])
                if L.inference_mode(plan):
                    observed.update(
                        Arn=(f"arn:aws:scheduler:{plan['identity']['region']}:{plan['identity']['account']}:"
                             f"schedule/{plan['identity']['schedule_group']}/{plan['identity']['run_id']}"),
                        CreationDate='2026-11-13T01:00:00Z', LastModificationDate='2026-11-13T01:00:00Z')
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
                if key.endswith('approval-consumed.json') and L.inference_mode(plan):
                    persisted = json.loads((state / 'controller-result.json').read_text())
                    self.assertTrue(persisted['remote_intent_attempted'])
                    self.assertFalse(persisted['dispatch_attempted'])
                    self.assertEqual(persisted['phase'], 'remote_launch_intent')
                    self.assertTrue(((Path(dispatch_state) if dispatch_state is not None else state)
                                     / 'dispatch-authorization.json').exists())
                    clock.sleep(remote_intent_delay)
                if key.endswith('approval-consumed.json') and remote_intent_error:
                    raise TimeoutError('Synthetic ambiguous remote intent response')
                if key.endswith('release.json'):
                    self.assertIsNotNone(cloud['schedule'])
                    cloud['released'] = True
                return {}
            raise AssertionError((service, operation, args))

        state = Path(temp) / state_name
        with patch.object(L, 'validate_plan', return_value={}), patch.object(L, 'validate_data', return_value=[]), \
             patch.object(L, 'validate_cost_admission', return_value={}), \
             patch.object(L, 'validate_bootstrap_admission', return_value=expectations), \
             patch.object(L, 'live_preflight', return_value={'synthetic': True}, side_effect=preflight_error), \
             patch.object(L, 'bootstrap_text', return_value='#!/bin/bash\nexit 0\n'):
            result = L.launch(plan, {}, 'a' * 64, temp, state, transport,
                              dispatch_state=dispatch_state, clock=clock, pause=clock.sleep)
            with self.assertRaises(FileExistsError):
                L.launch(plan, {}, 'a' * 64, temp, state, transport,
                         dispatch_state=dispatch_state, clock=clock, pause=clock.sleep)
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


class PriceProductFingerprintTests(unittest.TestCase):
    def fixture(self):
        return [dict(
            publicationDate='2026-11-12T00:00:00Z', version='20261112000000',
            serviceCode='AmazonEC2', formatVersion='aws_v1',
            product=dict(sku=sku, productFamily='Compute Instance',
                attributes=dict(instanceType='g6e.2xlarge', location='Asia Pacific (Seoul)',
                    publicationDate='nested-product-date', version='nested-product-version')),
            terms={'OnDemand': {sku + '.synthetic-term': dict(
                sku=sku, offerTermCode='synthetic-term', effectiveDate='2026-11-01T00:00:00Z',
                publicationDate='nested-term-date', version='nested-term-version',
                termAttributes={},
                priceDimensions={sku + '.synthetic-term.hour': dict(
                    rateCode=sku + '.synthetic-term.hour', description='Synthetic hourly rate',
                    beginRange='0', endRange='Inf', unit='Hrs', pricePerUnit={'USD': rate},
                    appliesTo=['synthetic-first', 'synthetic-second'],
                    publicationDate='nested-dimension-date', version='nested-dimension-version')})}})
            for sku, rate in (('synthetic-sku-a', '3.0000000000'), ('synthetic-sku-b', '4.0000000000'))]

    def test_top_level_catalog_metadata_changes_preserve_fingerprint_not_raw_hashes(self):
        for fields in (('publicationDate',), ('version',), ('publicationDate', 'version')):
            for operation in ('change', 'add', 'remove'):
                with self.subTest(fields=fields, operation=operation):
                    original = self.fixture()
                    if operation == 'add':
                        for product in original:
                            for field in fields:
                                del product[field]
                    changed = copy.deepcopy(original)
                    for product in changed:
                        for field in fields:
                            if operation == 'remove':
                                del product[field]
                            else:
                                product[field] = ('2026-11-14T00:00:00Z' if field == 'publicationDate'
                                                  else '20261114000000')
                    self.assertEqual(L.on_demand_price_products_sha256(original),
                                     L.on_demand_price_products_sha256(changed))
                    self.assertNotEqual(L.canonical(original), L.canonical(changed))
                    self.assertNotEqual(L.object_sha(original), L.object_sha(changed))
                    self.assertNotEqual(hashlib.sha256(L.canonical(original)).hexdigest(),
                                        hashlib.sha256(L.canonical(changed)).hexdigest())

    def test_reserved_offer_changes_preserve_on_demand_fingerprint_not_raw_hashes(self):
        for operation in ('add', 'remove', 'date', 'price', 'order', 'offer_order'):
            with self.subTest(operation=operation):
                original = self.fixture()
                reserved = copy.deepcopy(original[0]['terms']['OnDemand'])
                offer = reserved['synthetic-sku-a.synthetic-term']
                offer['termAttributes'] = {'LeaseContractLength': '1yr'}
                reserved['synthetic-second-reserved-term'] = copy.deepcopy(offer)
                original[0]['terms']['Reserved'] = reserved
                if operation == 'add':
                    del original[0]['terms']['Reserved']
                changed = copy.deepcopy(original)
                if operation == 'add':
                    changed[0]['terms']['Reserved'] = reserved
                elif operation == 'remove':
                    del changed[0]['terms']['Reserved']
                else:
                    changed_reserved = changed[0]['terms']['Reserved']
                    changed_offer = changed_reserved['synthetic-sku-a.synthetic-term']
                    dimension = changed_offer['priceDimensions']['synthetic-sku-a.synthetic-term.hour']
                    if operation == 'date':
                        changed_offer['effectiveDate'] = '2026-11-14T00:00:00Z'
                    elif operation == 'price':
                        dimension['pricePerUnit']['USD'] = '99.0000000000'
                    elif operation == 'order':
                        dimension['appliesTo'].reverse()
                    else:
                        changed[0]['terms']['Reserved'] = dict(reversed(list(changed_reserved.items())))
                self.assertEqual(L.on_demand_price_products_sha256(original),
                                 L.on_demand_price_products_sha256(changed))
                self.assertNotEqual(json.dumps(original), json.dumps(changed))
                if operation != 'offer_order':
                    self.assertNotEqual(L.object_sha(original), L.object_sha(changed))

    def test_only_on_demand_terms_are_projected(self):
        original = self.fixture()
        changed = copy.deepcopy(original)
        changed[0]['terms'].update(Reserved={'offer': {'effectiveDate': 'changed'}},
                                   SavingsPlan={'offer': {'price': 'changed'}})
        self.assertEqual(L.on_demand_price_products_sha256(original),
                         L.on_demand_price_products_sha256(changed))

    def test_invalid_product_lists_fail_closed(self):
        for products in (None, {}, (), '', 0, False, []):
            with self.subTest(products=products), self.assertRaises(RuntimeError):
                L.on_demand_price_products_sha256(products)
        for value in (None, [], '', 0, False):
            for index in range(2):
                with self.subTest(value=value, index=index), self.assertRaises(RuntimeError):
                    products = self.fixture()
                    products[index] = value
                    L.on_demand_price_products_sha256(products)

    def test_missing_or_malformed_on_demand_terms_fail_closed(self):
        for path in (('terms',), ('terms', 'OnDemand')):
            for operation in ('remove', 'replace'):
                for value in (None, {}, [], '', 0, False):
                    for index in range(2):
                        with self.subTest(path=path, operation=operation, value=value, index=index):
                            products = self.fixture()
                            products[index]['terms']['Reserved'] = {'offer': {'sku': 'reserved-only'}}
                            target = products[index]
                            for key in path[:-1]:
                                target = target[key]
                            if operation == 'remove':
                                del target[path[-1]]
                            else:
                                target[path[-1]] = value
                            with self.assertRaises(RuntimeError):
                                L.on_demand_price_products_sha256(products)
        products = self.fixture()
        terms = products[0]['terms']
        terms['OnDemand-changed'] = terms.pop('OnDemand')
        with self.assertRaises(RuntimeError):
            L.on_demand_price_products_sha256(products)

    def test_empty_or_nondict_on_demand_offers_fail_closed(self):
        for value in (None, {}, [], '', 0, False):
            for index in range(2):
                for operation in ('replace', 'add'):
                    with self.subTest(value=value, index=index, operation=operation):
                        products = self.fixture()
                        offers = products[index]['terms']['OnDemand']
                        key = next(iter(offers)) if operation == 'replace' else 'malformed-extra-offer'
                        offers[key] = value
                        with self.assertRaises(RuntimeError):
                            L.on_demand_price_products_sha256(products)

    def test_nonmetadata_values_remain_bound(self):
        term = ('terms', 'OnDemand', 'synthetic-sku-a.synthetic-term')
        dimension = term + ('priceDimensions', 'synthetic-sku-a.synthetic-term.hour')
        changes = [
            (('serviceCode',), 'AmazonS3'),
            (('formatVersion',), 'aws_v2'),
            (('product', 'sku'), 'different-sku'),
            (('product', 'productFamily'), 'Storage'),
            (('product', 'attributes', 'instanceType'), 'g6e.4xlarge'),
            (('product', 'attributes', 'location'), 'US East (N. Virginia)'),
            (term + ('sku',), 'different-sku'),
            (term + ('offerTermCode',), 'different-term'),
            (term + ('effectiveDate',), '2026-11-14T00:00:00Z'),
            (term + ('termAttributes',), {'LeaseContractLength': '1yr'}),
            (dimension + ('rateCode',), 'different-rate'),
            (dimension + ('description',), 'Different rate description'),
            (dimension + ('pricePerUnit', 'USD'), '3.1000000000'),
            (dimension + ('pricePerUnit',), {'KRW': '3.0000000000'}),
            (dimension + ('beginRange',), '1'),
            (dimension + ('endRange',), '100'),
            (dimension + ('unit',), 'GB-Mo'),
        ]
        original = self.fixture()
        expected = L.on_demand_price_products_sha256(original)
        for path, value in changes:
            with self.subTest(path=path):
                changed = copy.deepcopy(original)
                target = changed[0]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertNotEqual(expected, L.on_demand_price_products_sha256(changed))

    def test_term_and_dimension_mapping_identities_remain_bound(self):
        term = ('terms', 'OnDemand', 'synthetic-sku-a.synthetic-term')
        for path in (('terms', 'OnDemand'), term + ('priceDimensions',)):
            with self.subTest(path=path):
                original = self.fixture()
                changed = copy.deepcopy(original)
                target = changed[0]
                for key in path:
                    target = target[key]
                key = next(iter(target))
                target[key + '-changed'] = target.pop(key)
                self.assertNotEqual(L.on_demand_price_products_sha256(original),
                                    L.on_demand_price_products_sha256(changed))

    def test_nested_catalog_field_names_are_not_excluded(self):
        term = ('terms', 'OnDemand', 'synthetic-sku-a.synthetic-term')
        paths = [('product', 'attributes'), term,
                 term + ('priceDimensions', 'synthetic-sku-a.synthetic-term.hour')]
        original = self.fixture()
        expected = L.on_demand_price_products_sha256(original)
        for path in paths:
            for field in ('publicationDate', 'version'):
                for operation in ('change', 'remove'):
                    with self.subTest(path=path, field=field, operation=operation):
                        changed = copy.deepcopy(original)
                        target = changed[0]
                        for key in path:
                            target = target[key]
                        if operation == 'remove':
                            del target[field]
                        else:
                            target[field] = 'changed-nested-value'
                        self.assertNotEqual(expected, L.on_demand_price_products_sha256(changed))

    def test_product_and_nested_list_order_remain_bound(self):
        original = self.fixture()
        expected = L.on_demand_price_products_sha256(original)
        changed = copy.deepcopy(original)
        changed.reverse()
        self.assertNotEqual(expected, L.on_demand_price_products_sha256(changed))
        changed = copy.deepcopy(original)
        dimension = changed[0]['terms']['OnDemand']['synthetic-sku-a.synthetic-term'][
            'priceDimensions']['synthetic-sku-a.synthetic-term.hour']
        dimension['appliesTo'].reverse()
        self.assertNotEqual(expected, L.on_demand_price_products_sha256(changed))

    def test_product_insertion_removal_and_duplication_remain_bound(self):
        original = self.fixture()
        expected = L.on_demand_price_products_sha256(original)
        for operation in ('insert', 'remove', 'duplicate'):
            with self.subTest(operation=operation):
                changed = copy.deepcopy(original)
                if operation == 'remove':
                    changed.pop()
                else:
                    product = copy.deepcopy(original[0])
                    if operation == 'insert':
                        product['product']['sku'] = 'synthetic-sku-c'
                    changed.insert(1, product)
                self.assertNotEqual(expected, L.on_demand_price_products_sha256(changed))

    def test_fingerprinting_does_not_mutate_input(self):
        products = self.fixture()
        original = copy.deepcopy(products)
        original_bytes = L.canonical(products)
        first = L.on_demand_price_products_sha256(products)
        self.assertEqual(products, original)
        self.assertEqual(L.canonical(products), original_bytes)
        self.assertEqual(L.on_demand_price_products_sha256(products), first)
        self.assertEqual(products, original)


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

class OrderedInferenceAdmissionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        plan = release_plan()
        plan.update(schema=L.INFERENCE_SCHEMA, limits=L.LIMITS, instance_type='g6e.2xlarge',
            vcpus=8, host_mib=65536, architecture='x86_64',
            checked_at=dt.datetime.fromtimestamp(1000, dt.timezone.utc).isoformat(),
            ami=dict(ImageId='ami-040ebe879ae70cc30', OwnerId='898082745236', root_volume_gib=75),
            ebs_kms_key_arn='arn:aws:kms:ap-northeast-2:148761639313:key/11111111-2222-3333-4444-555555555555',
            disk_os_reserve_bytes=20 * L.GIB)
        plan.update(CostClosureTests().fixture())
        plan['cost'].update(cap_usd=20, ancillary_bound_usd=5, get_requests=579, put_requests=20,
            control_requests=1067, download_requests=200,
            network_egress_bytes=4 * L.GIB + 2 * L.CHECKPOINT_CHANNEL_BYTES)
        plan['network'].update(download_hosts=['huggingface.co'],
                               egress_byte_cap=2 * L.GIB + L.CHECKPOINT_CHANNEL_BYTES)
        rows = []

        def pin(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            row = dict(path=name, bytes=path.stat().st_size, sha256=L.digest(path))
            rows.append(row)
            return row

        profiles = [dict(name=f'qwen8-{"thinking" if thinking else "nonthinking"}-seed{seed}',
            thinking=thinking, temperature=.6 if thinking else .7, top_p=.95 if thinking else .8,
            top_k=20, min_p=0., presence_penalty=0., repetition_penalty=1., penalty_context_size=0,
            seed=seed, max_new_tokens=4096 if thinking else 1024, context_tokens=8192, prefill_step_size=128)
            for thinking in (False, True) for seed in (14, 29)]
        workload = dict(models={}, profiles=profiles, phase_seconds=dict(L.ORDERED_PHASES),
            backend='transformers', dtype='bfloat16', prefill='full_prompt', question_timeout_seconds=180,
            selection_policy=dict(metric='joint_correct', require_both_seeds=True,
                tie_breaks=['minimum_correct', 'combined_correct', 'mean_elapsed_seconds',
                            'peak_cuda_allocated_bytes', 'lexical_group'], validation_references_in_bundle=False),
            checkpoint_keys=dict(archive=plan['identity']['output_prefix'] + 'qwen8-checkpoint.tar',
                receipt=plan['identity']['output_prefix'] + 'qwen8-checkpoint.json',
                decision=plan['identity']['code_prefix'] + 'parent-decision.json'))
        ident = plan['identity']
        plan['iam']['instance_role'] = {'inline': {'ExactRunOnly': {'Statement': [
            dict(Effect='Allow', Action=['s3:GetObject'],
                 Resource=[f"arn:aws:s3:::{ident['bucket']}/{ident['code_prefix']}*"]),
            dict(Effect='Allow', Action=['s3:PutObject'],
                 Resource=[f"arn:aws:s3:::{ident['bucket']}/{ident['output_prefix']}*"]),
            dict(Effect='Allow', Action=['s3:ListBucket'], Resource=[f"arn:aws:s3:::{ident['bucket']}"],
                 Condition={'StringEquals': {'s3:prefix': workload['checkpoint_keys']['decision']}}),
        ]}}}
        for key, (model_id, revision) in L.ORDERED_MODELS.items():
            files = [dict(path=name, bytes=1, sha256='b' * 64,
                url=f'https://huggingface.co/{model_id}/resolve/{revision}/{name}')
                for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json',
                             'model.safetensors.index.json', 'LICENSE', 'model-00001.safetensors')]
            workload['models'][key] = dict(id=model_id, revision=revision, trust_remote_code=False,
                                           dtype='bfloat16', files=files, max_bytes=L.GIB)
        for cohort, count in (('development', 6), ('validation', 16)):
            workload[cohort + '_case_ids'] = [f'{cohort}-{index}' for index in range(count)]
            workload[cohort + '_inputs'] = pin(cohort + '-inputs.json', [
                dict(id=case_id, messages=[dict(role='user', content='Synthetic input, no labels.')])
                for case_id in workload[cohort + '_case_ids']])
        workload['development_references'] = pin('development-references.json',
            dict(cases=[dict(id=case_id, reference={},
                messages=[dict(role='user', content='Synthetic input, no labels.')])
                for case_id in workload['development_case_ids']]))
        workload['source_closure'] = pin('source-closure.json', dict(
            schema='trid-ordered-source-closure-v1', run_id=plan['identity']['run_id'],
            development_inputs_sha256=workload['development_inputs']['sha256'],
            validation_inputs_sha256=workload['validation_inputs']['sha256'],
            development_references_sha256=workload['development_references']['sha256'],
            development_case_ids=workload['development_case_ids'], validation_case_ids=workload['validation_case_ids'],
            validation_references_in_bundle=False, source_verified=True, protected_sets_excluded=True,
            scorer_sha256='0c25aaeee6075d49f90b06c06f2f4389c698c7a543ab6500e72c5a262c8c3e8a',
            provenance={key: dict(path=f'private/{key}.json', bytes=1,
                sha256='0c25aaeee6075d49f90b06c06f2f4389c698c7a543ab6500e72c5a262c8c3e8a'
                if key == 'scorer' else 'c' * 64)
                for key in ('comparison_protocol', 'case_review', 'reference_validation', 'scope', 'scorer')}))
        plan['workload'] = workload
        plan['sources'] = {}
        for name in L.ordered_sources():
            source_pin = dict(path=name, bytes=1, sha256='c' * 64)
            rows.append(source_pin)
            plan['sources'][name] = source_pin['sha256']
        admission = pin('bootstrap.json', {})
        plan['runtime'].update(python_version='3.12.3', torch_cuda='synthetic',
            packages={key: 'synthetic' for key in
                ('torch', 'transformers', 'tokenizers', 'safetensors', 'accelerate', 'psutil')},
            wheels=[dict(path='synthetic.whl', bytes=1, sha256='d' * 64, url='https://example.test/synthetic.whl')],
            max_bytes=L.GIB, download_hosts=['example.test'],
            bootstrap=dict(admission=admission, nvidia_modprobe=dict(
                path='nvidia-modprobe.deb', bytes=1, sha256='e' * 64, package='nvidia-modprobe',
                architecture='amd64', version='620.70.06-1ubuntu1', url='https://example.test/modprobe.deb',
                member=dict(path='usr/bin/nvidia-modprobe', bytes=1, sha256='f' * 64),
                max_unpacked_bytes=1024)))
        plan['bundle'].update(files=rows, unpacked_bytes=1024**2)
        plan['ancillary_evidence'] = pin('ancillary.json', {})
        region = plan['identity']['region']
        specs = {
            'ec2_hour': ('AmazonEC2', 'Hrs', dict(regionCode=region, instanceType='g6e.2xlarge',
                operatingSystem='Linux', tenancy='Shared', preInstalledSw='NA', capacitystatus='Used')),
            'ebs_gb_month': ('AmazonEC2', 'GB-Mo', dict(regionCode=region, volumeApiName='gp3', productFamily='Storage')),
            'ipv4_hour': ('AmazonVPC', 'Hrs', dict(regionCode=region, usagetype='APN2-PublicIPv4:InUseAddress')),
            'egress_gb': ('AWSDataTransfer', 'GB', dict(fromLocation='Asia Pacific (Seoul)',
                toLocation='External', transferType='AWS Outbound')),
            's3_gb_month': ('AmazonS3', 'GB-Mo', dict(regionCode=region, productFamily='Storage', volumeType='Standard')),
            's3_get': ('AmazonS3', 'Requests', dict(regionCode=region, group='S3-API-Tier2')),
            's3_put': ('AmazonS3', 'Requests', dict(regionCode=region, group='S3-API-Tier1')),
            'scheduler_invocation': ('synthetic', 'Requests', {})}
        plan['price_queries'] = {key: dict(service=service, unit=unit, filters=filters)
                                 for key, (service, unit, filters) in specs.items()}
        plan['output_allowlist'] = sorted(L.ordered_output_members(plan))
        approval = dict(schema='trid-ordered-hf-inference-approval-v1', run_id=plan['identity']['run_id'],
            plan_sha256='a' * 64, instance_limit=1, additional_pretax_usd=20,
            workload_sha256=L.object_sha(workload), limits=L.LIMITS,
            ebs_kms_key_arn=plan['ebs_kms_key_arn'], automatic_retry=False, approved=True)
        return plan, approval

    def test_sealed_inference_plan_admits_without_training_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, approval = self.fixture(temp)
            result = L.validate_plan(plan, approval, 'a' * 64, now=1000)
            self.assertLess(result['additional_pretax_bound_usd'], 20)
            self.assertEqual(len(L.validate_data(temp, plan)['validation_inputs']), 16)
            self.assertNotIn('recipe', plan)
            self.assertNotIn('peft', plan['runtime']['packages'])

    def test_ordered_plan_without_exact_decision_listing_permission_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            original, approval = self.fixture(temp)
            for prefix in (None, original['identity']['code_prefix'], '*'):
                with self.subTest(prefix=prefix):
                    plan = copy.deepcopy(original)
                    statements = plan['iam']['instance_role']['inline']['ExactRunOnly']['Statement']
                    if prefix is None:
                        statements.pop()
                    else:
                        statements[-1]['Condition']['StringEquals']['s3:prefix'] = prefix
                    with self.assertRaisesRegex(RuntimeError, 'exact-decision listing'):
                        L.validate_plan(plan, approval, 'a' * 64, now=1000)

    def test_old_grant_or_changed_workload_cannot_admit_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, approval = self.fixture(temp)
            for key, value in (('additional_pretax_usd', 50), ('schema', 'trid-grounded-hf-approval-v1'),
                               ('automatic_retry', True), ('workload_sha256', '0' * 64)):
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'USD20'):
                    L.validate_plan(plan, {**approval, key: value}, 'a' * 64, now=1000)
            plan['recipe'] = dict(L.RECIPE)
            with self.assertRaisesRegex(RuntimeError, 'cannot admit training'):
                L.validate_plan(plan, approval, 'a' * 64, now=1000)

    def test_changed_model_profile_allocation_or_checkpoint_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            original, approval = self.fixture(temp)
            changes = [
                lambda w: w['models']['qwen14'].update(revision='0' * 40),
                lambda w: w['models']['qwen8'].update(dtype='float16'),
                lambda w: w['profiles'][0].update(seed=15),
                lambda w: w['phase_seconds'].update(parent_decision=301),
                lambda w: w['checkpoint_keys'].update(decision='code/another-run/parent-decision.json'),
                lambda w: w['selection_policy'].update(require_both_seeds=False)]
            for change in changes:
                plan = copy.deepcopy(original)
                change(plan['workload'])
                current = {**approval, 'workload_sha256': L.object_sha(plan['workload'])}
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    L.validate_plan(plan, current, 'a' * 64, now=1000)

    def test_validation_labels_and_reference_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            pin = plan['workload']['validation_inputs']
            path = Path(temp) / pin['path']
            rows = json.loads(path.read_text())
            rows[0]['expected'] = 'forbidden'
            path.write_text(json.dumps(rows))
            pin.update(bytes=path.stat().st_size, sha256=L.digest(path))
            with self.assertRaisesRegex(RuntimeError, 'labels'):
                L.validate_data(temp, plan)
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            plan['workload']['development_case_ids'][0] = plan['workload']['validation_case_ids'][0]
            with self.assertRaisesRegex(RuntimeError, 'Disjoint'):
                L.validate_ordered_workload(plan, {r['path']: r for r in plan['bundle']['files']})

    def test_development_reference_required_fields_and_messages_are_admitted_before_generation(self):
        for change, error in (
                (lambda row: row.pop('reference'), 'reference fields'),
                (lambda row: row.pop('messages'), 'reference fields'),
                (lambda row: row['messages'][0].update(content='Different question.'), 'messages differ')):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                reference_pin = plan['workload']['development_references']
                path = Path(temp) / reference_pin['path']
                references = json.loads(path.read_text())
                change(references['cases'][0])
                path.write_text(json.dumps(references))
                reference_pin.update(bytes=path.stat().st_size, sha256=L.digest(path))
                with self.assertRaisesRegex(RuntimeError, error):
                    L.validate_data(temp, plan)

    def test_checkpoint_channel_quantities_and_usd20_are_mandatory(self):
        with tempfile.TemporaryDirectory() as temp:
            original, _ = self.fixture(temp)
            for key, value in (('cap_usd', 50), ('get_requests', 512), ('put_requests', 16),
                               ('control_requests', 1000), ('ancillary_bound_usd', 4),
                               ('network_egress_bytes', 4 * L.GIB)):
                plan = copy.deepcopy(original)
                plan['cost'][key] = value
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    L.cost_bound(plan)
            original['cost']['rates']['ec2_hour'] = 4
            with self.assertRaisesRegex(RuntimeError, 'USD20'):
                L.cost_bound(original)

    def model_artifacts(self, plan, model_key='qwen8', group='nonthinking'):
        workload = plan['workload']
        content = {name: b'{}' for name in L.ordered_output_members(plan, model_key, group)}
        identity_fields = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64)
        profiles = [{**p, 'name': p['name'].replace('qwen8-', model_key + '-', 1)}
                    for p in workload['profiles']]
        dev_outputs = []
        for cohort, input_key in (('dev', 'development'), ('val', 'validation')):
            for profile in profiles:
                profile_group = 'thinking' if profile['thinking'] else 'nonthinking'
                if cohort == 'val' and profile_group != group:
                    continue
                stem = f'{model_key}-{cohort}-{profile_group}-seed{profile["seed"]}'
                output = dict(path=stem + '-outputs.jsonl', bytes=2, sha256=hashlib.sha256(b'{}').hexdigest())
                receipt = dict(**identity_fields, state='complete', phase=stem, model_key=model_key,
                    model_sha256=L.object_sha(workload['models'][model_key]), profile=profile,
                    input_sha256=workload[input_key + '_inputs']['sha256'],
                    completed_case_ids=workload[input_key + '_case_ids'], question_deadline_epoch=None,
                    output_sha256=output['sha256'], output_bytes=output['bytes'])
                content[stem + '-receipt.json'] = L.canonical(receipt)
                if cohort == 'dev':
                    dev_outputs.append(output)
        selection = dict(**identity_fields, state='complete', model_key=model_key,
            model_sha256=L.object_sha(workload['models'][model_key]), profiles_sha256=L.object_sha(profiles),
            development_inputs_sha256=workload['development_inputs']['sha256'],
            development_references_sha256=workload['development_references']['sha256'],
            selection_policy=workload['selection_policy'], selected_group=group,
            selected_profiles=[f'{model_key}-{group}-seed{seed}' for seed in (14, 29)], dev_outputs=dev_outputs)
        content[model_key + '-selection.json'] = L.canonical(selection)
        suffix = '8' if model_key == 'qwen8' else '14'
        for phase in ('acquire' + suffix, 'dev' + suffix, 'val' + suffix):
            content[phase + '-receipt.json'] = L.canonical(dict(**identity_fields, phase=phase, state='complete'))
        return content

    def checkpoint(self, plan, directory):
        content = self.model_artifacts(plan)
        archive, pin = archive_fixture(directory, [
            (name, payload, tarfile.REGTYPE) for name, payload in sorted(content.items())])
        workload = plan['workload']
        receipt = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
            instance_id=instance(plan['identity'])['InstanceId'], phase='val8', selected_group='nonthinking',
            model_sha256=L.object_sha(workload['models']['qwen8']),
            profiles_sha256=L.object_sha(workload['profiles']),
            development_inputs_sha256=workload['development_inputs']['sha256'],
            validation_inputs_sha256=workload['validation_inputs']['sha256'],
            export={key: pin[key] for key in ('bytes', 'sha256')}, files=pin['files'])
        return archive, receipt

    def test_checkpoint_binds_exact_members_and_every_identity_dimension(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            _, receipt = self.checkpoint(plan, temp)
            instance_id = receipt['instance_id']
            self.assertEqual(L.validate_checkpoint_receipt(receipt, plan, 'a' * 64, instance_id), receipt['export'])
            for key in ('run_id', 'plan_sha256', 'instance_id', 'model_sha256', 'profiles_sha256',
                        'development_inputs_sha256', 'validation_inputs_sha256', 'phase'):
                altered = copy.deepcopy(receipt)
                altered[key] = 'wrong'
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    L.validate_checkpoint_receipt(altered, plan, 'a' * 64, instance_id)
            for change in (lambda r: r['files'].pop(),
                           lambda r: r['files'].append(r['files'][0]),
                           lambda r: r['export'].update(bytes=L.CHECKPOINT_MAX + 1)):
                altered = copy.deepcopy(receipt)
                change(altered)
                with self.assertRaises(RuntimeError):
                    L.validate_checkpoint_receipt(altered, plan, 'a' * 64, instance_id)

    def test_readonly_checkpoint_watcher_returns_while_protection_remains_armed(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            archive, receipt = self.checkpoint(plan, temp)
            payload = archive.read_bytes()
            state = Path(temp) / 'state'
            state.mkdir()
            token = release(plan)
            (state / 'release.json').write_text(json.dumps(token))
            calls = []
            clock = Clock()

            def transport(service, operation, *args):
                calls.append((service, operation))
                self.assertEqual((service, operation), ('s3api', 'get-object'))
                if len(calls) == 1:
                    raise L.AWSCallError(service, operation, 'An error occurred (NoSuchKey)')
                if args[args.index('--key') + 1].endswith('.json'):
                    Path(args[-1]).write_text(json.dumps(receipt))
                else:
                    Path(args[-1]).write_bytes(payload)

            result = L.collect_checkpoint(transport, plan, 'a' * 64, state, clock=clock, pause=clock.sleep)
            self.assertEqual(result['state'], 'checkpoint_collected')
            self.assertEqual(json.loads((state / 'release.json').read_text()), token)
            self.assertEqual(len(calls), 3)
            self.assertEqual(clock.now, 1120)
            self.assertNotIn('inference_success', result)

    def test_checkpoint_watcher_does_not_retry_malformed_or_denied_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            state = Path(temp) / 'state'
            state.mkdir()
            (state / 'release.json').write_text(json.dumps(release(plan)))
            calls = []

            def denied(*args):
                calls.append(args)
                raise L.AWSCallError('s3api', 'get-object', 'An error occurred (AccessDenied)')

            with self.assertRaisesRegex(RuntimeError, 'only missing'):
                L.collect_checkpoint(denied, plan, 'a' * 64, state, clock=Clock())
            self.assertEqual(len(calls), 1)

    def test_parent_decision_never_accepts_wrong_run_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            decision = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                            checkpoint_sha256='b' * 64, decision='complete', reason='Independent review.')
            self.assertEqual(L.validate_parent_decision(decision, plan, 'a' * 64, 'b' * 64), decision)
            for key, value in (('run_id', 'wrong'), ('checkpoint_sha256', 'c' * 64),
                               ('decision', 'retry'), ('reason', '')):
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    L.validate_parent_decision({**decision, key: value}, plan, 'a' * 64, 'b' * 64)

    def test_native_completion_requires_exact_conditional_members_and_bound_profile_receipts(self):
        for decision_name in ('complete', 'compare14'):
            with self.subTest(decision=decision_name), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                _, checkpoint = self.checkpoint(plan, temp)
                content = self.model_artifacts(plan)
                phases = ['acquire8', 'dev8', 'val8', 'parent_decision']
                if decision_name == 'compare14':
                    content.update(self.model_artifacts(plan, 'qwen14', 'thinking'))
                    phases += ['acquire14', 'dev14', 'val14']
                decision = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                    checkpoint_sha256=checkpoint['export']['sha256'], decision=decision_name,
                    reason='Independent synthetic review.')
                content['parent-decision.json'] = L.canonical(decision)
                content['parent_decision-receipt.json'] = b'{}'
                completion = dict(inference_success=True, checkpoint=checkpoint, decision=decision,
                    stages=[dict(phase=phase, state='complete', run_id=plan['identity']['run_id'],
                                 plan_sha256='a' * 64) for phase in phases])
                path, archive_pin = archive_fixture(temp, [
                    (name, payload, tarfile.REGTYPE) for name, payload in sorted(content.items())])
                L.validate_ordered_completion(completion, plan, 'a' * 64, checkpoint['instance_id'],
                                              path, archive_pin)
                incomplete = copy.deepcopy(completion)
                incomplete['stages'][-1]['state'] = 'incomplete'
                with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                    L.validate_ordered_completion(incomplete, plan, 'a' * 64,
                                                  checkpoint['instance_id'], path, archive_pin)
                altered = copy.deepcopy(archive_pin)
                altered['files'].append(dict(path='qwen14-val-unregistered.json', bytes=1, sha256='a' * 64))
                with self.assertRaisesRegex(RuntimeError, 'exact conditional'):
                    L.validate_ordered_completion(completion, plan, 'a' * 64,
                                                  checkpoint['instance_id'], path, altered)
                bad_profile = copy.deepcopy(content)
                name = 'qwen8-val-nonthinking-seed14-receipt.json'
                receipt = json.loads(bad_profile[name])
                receipt['output_sha256'] = 'f' * 64
                bad_profile[name] = L.canonical(receipt)
                path, bad_pin = archive_fixture(temp, [
                    (name, payload, tarfile.REGTYPE) for name, payload in sorted(bad_profile.items())])
                with self.assertRaisesRegex(RuntimeError, 'Profile completion'):
                    L.validate_ordered_completion(completion, plan, 'a' * 64,
                                                  checkpoint['instance_id'], path, bad_pin)

    def test_collect_export_authenticates_parent_owned_decision_and_uses_inference_success(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            _, checkpoint = self.checkpoint(plan, temp)
            content = self.model_artifacts(plan)
            decision = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                checkpoint_sha256=checkpoint['export']['sha256'], decision='complete',
                reason='Independent synthetic review.')
            content['parent-decision.json'] = L.canonical(decision)
            content['parent_decision-receipt.json'] = b'{}'
            files = [dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                     for name, payload in sorted(content.items())]
            manifest = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64, files=files)
            content['export-manifest.json'] = L.canonical(manifest)
            archive, pin = archive_fixture(temp, [
                (name, payload, tarfile.REGTYPE) for name, payload in sorted(content.items())])
            payload = archive.read_bytes()
            completion = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                instance_id=checkpoint['instance_id'], inference_success=True, stop_errors=[],
                checkpoint=checkpoint, decision=decision, runtime_unit=dict(gpu=gpu()),
                export={key: pin[key] for key in ('bytes', 'sha256')},
                stages=[dict(phase=phase, state='complete', run_id=plan['identity']['run_id'],
                             plan_sha256='a' * 64)
                        for phase in ('acquire8', 'dev8', 'val8', 'parent_decision')])
            state = Path(temp) / 'state'
            state.mkdir()
            (state / 'release.json').write_text(json.dumps(release(plan)))
            objects = {
                plan['identity']['output_prefix'] + 'completion.json': L.canonical(completion),
                plan['identity']['output_prefix'] + 'export.tar': payload,
                plan['workload']['checkpoint_keys']['receipt']: L.canonical(checkpoint),
                plan['workload']['checkpoint_keys']['decision']: L.canonical(decision)}
            calls = []

            def transport(service, operation, *args):
                self.assertEqual((service, operation), ('s3api', 'get-object'))
                key = args[args.index('--key') + 1]
                calls.append(key)
                Path(args[-1]).write_bytes(objects[key])

            result = L.collect_export(transport, plan, 'a' * 64, state, checkpoint['instance_id'])
            self.assertTrue(result['native_pipeline_success'])
            self.assertEqual(result['semantic_acceptance'], 'not_evaluated')
            self.assertIn(plan['workload']['checkpoint_keys']['decision'], calls)
            objects[plan['workload']['checkpoint_keys']['decision']] = L.canonical({**decision, 'run_id': 'wrong'})
            with self.assertRaisesRegex(RuntimeError, 'independently retrieved'):
                L.collect_export(transport, plan, 'a' * 64, state, checkpoint['instance_id'])

    def test_ordered_mode_uses_same_one_shot_release_and_cleanup_path(self):
        case = LaunchTests()
        fixture = case.fixture

        def ordered_fixture(root):
            plan = fixture(root)
            plan['schema'] = L.INFERENCE_SCHEMA
            return plan

        with tempfile.TemporaryDirectory() as temp, patch.object(case, 'fixture', side_effect=ordered_fixture):
            result, events = case.run_launch(temp)
        self.assertEqual(sum(op == 'run-instances' for op, _ in events), 1)
        self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
        self.assertFalse(result['inference_success'])
        self.assertNotIn('training_success', result)
        self.assertTrue(any(op == 'put-object' and any(str(a).endswith('release.json') for a in args)
                            for op, args in events))

    def test_transport_pins_standard_s3_endpoint_and_ignores_configured_overrides(self):
        response = L.subprocess.CompletedProcess(['aws'], 0, stdout='{}', stderr='')
        with patch.dict(os.environ, {'AWS_ENDPOINT_URL': 'https://unpriced.example',
                                    'AWS_ENDPOINT_URL_S3': 'https://unpriced-s3.example'}), \
             patch.object(L.subprocess, 'run', return_value=response) as run:
            L.Client(identity())('s3api', 'head-object', '--bucket', 'synthetic', '--key', 'key')
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--endpoint-url') + 1],
                             'https://s3.ap-northeast-2.amazonaws.com')
            self.assertEqual(run.call_args.kwargs['env']['AWS_IGNORE_CONFIGURED_ENDPOINT_URLS'], 'true')
            L.Client(identity())('ec2', 'describe-instances')
            self.assertNotIn('--endpoint-url', run.call_args.args[0])
            self.assertEqual(run.call_args.kwargs['env']['AWS_IGNORE_CONFIGURED_ENDPOINT_URLS'], 'true')

    def test_ordered_schedule_rejects_extra_target_configuration_before_release(self):
        plan = release_plan()
        spec = L.schedule_spec(plan, 'i-0123456789abcdef0', 1000)
        observed = dict(spec, Arn=(f"arn:aws:scheduler:{plan['identity']['region']}:"
            f"{plan['identity']['account']}:schedule/{plan['identity']['schedule_group']}/"
            f"{plan['identity']['run_id']}"),
            CreationDate='2026-11-13T01:00:00Z', LastModificationDate='2026-11-13T01:00:00Z')
        self.assertEqual(L.ordered_schedule_readback(observed, spec, plan['identity']), observed)
        observed = copy.deepcopy(observed)
        observed['Target']['DeadLetterConfig'] = {'Arn': 'arn:aws:sqs:ap-northeast-2:148761639313:unapproved'}
        with self.assertRaisesRegex(RuntimeError, 'complete configuration'):
            L.ordered_schedule_readback(observed, spec, plan['identity'])

    def test_ordered_schedule_cleanup_requires_unchanged_actual_creation_and_readback(self):
        for mutation in ('none', 'created_date', 'modified_date', 'extra_target', 'extra_top_level',
                         'missing_created', 'missing_readback', 'wrong_create_arn'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                state = Path(temp)
                plan = release_plan()
                plan['schema'] = L.INFERENCE_SCHEMA
                ident = plan['identity']
                instance_id = 'i-0123456789abcdef0'
                spec = L.schedule_spec(plan, instance_id, 1000)
                snapshot = dict(spec, Arn=(f"arn:aws:scheduler:{ident['region']}:{ident['account']}:"
                    f"schedule/{ident['schedule_group']}/{ident['run_id']}"),
                    CreationDate='2026-11-13T01:00:00Z', LastModificationDate='2026-11-13T01:00:00Z')
                L.save(state / 'schedule-intent.json', spec)
                L.save(state / 'approval-consumed.json', dict(run_id=ident['run_id'], plan_sha256='a' * 64))
                if mutation != 'missing_created':
                    L.save(state / 'schedule-created.json', {'ScheduleArn':
                        'arn:wrong' if mutation == 'wrong_create_arn' else snapshot['Arn']})
                if mutation != 'missing_readback':
                    L.save(state / 'schedule-readback.json', snapshot)
                observed = copy.deepcopy(snapshot)
                if mutation == 'created_date':
                    observed['CreationDate'] = '2026-11-13T02:00:00Z'
                    observed['LastModificationDate'] = '2026-11-13T02:00:00Z'
                elif mutation == 'modified_date':
                    observed['LastModificationDate'] = '2026-11-13T02:00:00Z'
                elif mutation == 'extra_target':
                    observed['Target']['DeadLetterConfig'] = {'Arn': 'arn:unapproved'}
                elif mutation == 'extra_top_level':
                    observed['Description'] = 'Changed after creation'
                observed['ResponseMetadata'] = {'RequestId': 'fresh-read'}
                cleanup = dict(cleanup_state='complete', instance_terminated=True, instance_id=instance_id,
                    root_volume_ids=['vol-synthetic'], recorded_volume_ids=['vol-synthetic'],
                    absent_volume_ids=['vol-synthetic'], remaining_volumes=[])
                calls = []

                def transport(service, operation, *args):
                    calls.append(operation)
                    return observed if operation == 'get-schedule' else {}

                if mutation == 'none':
                    L.cleanup_schedule(transport, ident, state, cleanup, plan=plan, plan_sha='a' * 64)
                    self.assertEqual(calls, ['get-schedule', 'delete-schedule'])
                else:
                    with self.assertRaises((RuntimeError, FileNotFoundError)):
                        L.cleanup_schedule(transport, ident, state, cleanup, plan=plan, plan_sha='a' * 64)
                    self.assertNotIn('delete-schedule', calls)

    def test_put_json_caps_actual_serialized_bytes_and_retains_oversized_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'oversized-control.json'
            value = {'rows': [0] * 180000}
            self.assertLess(len(L.canonical(value)), 1024**2)
            calls = []
            with self.assertRaisesRegex(RuntimeError, 'Serialized control document'):
                L.put_json(lambda *args: calls.append(args), identity(), 'code/control.json', value, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1024**2)
            self.assertEqual(calls, [])
            small = Path(temp) / 'small-control.json'
            L.put_json(lambda *args: calls.append(args), identity(), 'code/control.json',
                       {'decision': 'complete'}, small)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:2], ('s3api', 'put-object'))
            self.assertIn('--if-none-match', calls[0])

    def test_reconcile_caps_each_client_call_and_restores_previous_deadline(self):
        for previous, expected_timeouts in ((None, [12, 5]), (1007, [7]), (1020, [12, 5])):
            with self.subTest(previous=previous):
                clock = Clock()
                client = L.Client(identity(), deadline=previous)
                timeouts = []
                effective = min(previous, 1012) if previous is not None else 1012

                def transport(command, **kwargs):
                    self.assertEqual(client.deadline, effective)
                    self.assertEqual(kwargs['timeout'], effective - clock())
                    timeouts.append(kwargs['timeout'])
                    clock.sleep(2)
                    return L.subprocess.CompletedProcess(command, 0, stdout='{"Reservations":[]}', stderr='')

                evidence = L.volume_identity(identity(), 'a' * 64, '/dev/sda1')
                with patch.object(L.time, 'time', side_effect=clock), \
                     patch.object(L.subprocess, 'run', side_effect=transport), \
                     patch.object(L, 'Client', return_value=client):
                    result = L.reconcile(client, identity(), None, 1012, evidence=evidence,
                                         clock=clock, pause=clock.sleep)
                self.assertEqual(result['cleanup_state'], 'unresolved')
                self.assertEqual(timeouts, expected_timeouts)
                self.assertEqual(client.deadline, previous)
                self.assertEqual(clock(), effective)

    def test_reconcile_restores_client_deadline_when_identity_validation_raises(self):
        client = L.Client(identity(), deadline=1030)
        evidence = L.volume_identity(identity(), 'a' * 64, '/dev/sda1')
        evidence['instance_id'] = 'i-foreign'
        with self.assertRaisesRegex(RuntimeError, 'identity conflict'):
            L.reconcile(client, identity(), 'i-expected', 1012, evidence=evidence, clock=Clock())
        self.assertEqual(client.deadline, 1030)

    def test_ordered_cleanup_cutoff_never_renews_grace_for_ambiguous_or_late_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            plan = release_plan()
            plan.update(schema=L.INFERENCE_SCHEMA, cost={'termination_grace_seconds': 1800})
            L.save(state / 'approval-consumed.json', dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64))
            L.save(state / 'launch-dispatched.json', dict(run_id=plan['identity']['run_id'],
                plan_sha256='a' * 64, request_sha256='b' * 64, at=1000))
            self.assertEqual(L.cleanup_cutoff(plan, 'a' * 64, state), 17200)
            # Later successful visibility must not move the already registered cutoff.
            observed = release(plan)
            observed.update(launch_time=1005, deadline=15405,
                schedule=L.schedule_spec(plan, observed['instance_id'], 1005))
            L.save(state / 'termination.json', observed)
            self.assertEqual(L.cleanup_cutoff(plan, 'a' * 64, state), 17200)
            clock = Clock()
            clock.now = 17201
            calls = []
            evidence = L.volume_identity(plan['identity'], 'a' * 64, '/dev/sda1')
            result = L.reconcile(lambda *args: calls.append(args), plan['identity'], None,
                min(clock() + 1800, L.cleanup_cutoff(plan, 'a' * 64, state)),
                evidence=evidence, clock=clock, pause=clock.sleep)
            self.assertEqual(result['cleanup_state'], 'unresolved')
            self.assertEqual(calls, [])
            observed['plan_sha256'] = 'c' * 64
            L.save(state / 'termination.json', observed)
            with self.assertRaisesRegex(RuntimeError, 'launch evidence'):
                L.cleanup_cutoff(plan, 'a' * 64, state)
            with self.assertRaisesRegex(RuntimeError, 'consumed approval'):
                L.cleanup_cutoff(plan, 'c' * 64, state)

    def test_catalog_workload_permutation_is_semantic_and_does_not_mutate_evidence(self):
        plan = release_plan()
        device = dict(Name='L40S', Manufacturer='NVIDIA', Count=1, LogicalGpuCount=1,
            GpuPartitionSize=1.0, MemoryInfo={'SizeInMiB': 45776}, Workloads=['graphics', 'ml-ai'])
        plan['gpu_type_observation'] = [device]
        observation = dict(Gpus=[{**device, 'Workloads': ['ml-ai', 'graphics']}], TotalGpuMemoryInMiB=45776)
        original_plan, original_observation = copy.deepcopy(plan), copy.deepcopy(observation)
        L.verify_gpu_catalog(plan, observation)
        self.assertEqual(plan, original_plan)
        self.assertEqual(observation, original_observation)

    def test_catalog_still_rejects_capacity_values_multiplicity_presence_and_other_fields(self):
        plan = release_plan()
        plan['gpu_type_observation'] = [dict(Name='L40S', Manufacturer='NVIDIA', Count=1,
            LogicalGpuCount=1, GpuPartitionSize=1.0, MemoryInfo={'SizeInMiB': 45776},
            Workloads=['graphics', 'ml-ai'])]
        observation = dict(Gpus=copy.deepcopy(plan['gpu_type_observation']), TotalGpuMemoryInMiB=45776)
        mutations = (
            lambda value: value['Gpus'][0].update(Workloads=['graphics']),
            lambda value: value['Gpus'][0].update(Workloads=['graphics', 'ml-ai', 'ml-ai']),
            lambda value: value['Gpus'][0].update(Workloads=['graphics', 'other']),
            lambda value: value['Gpus'][0].pop('Workloads'),
            lambda value: value['Gpus'][0].update(Workloads='graphics'),
            lambda value: value['Gpus'][0].update(Name='L4'),
            lambda value: value['Gpus'][0].update(Count=2),
            lambda value: value['Gpus'][0].update(LogicalGpuCount=2),
            lambda value: value['Gpus'][0].update(GpuPartitionSize=.5),
            lambda value: value['Gpus'][0]['MemoryInfo'].update(SizeInMiB=48000),
            lambda value: value['Gpus'][0].update(Unexpected=True),
            lambda value: value.update(TotalGpuMemoryInMiB=48000))
        for mutate in mutations:
            altered = copy.deepcopy(observation)
            mutate(altered)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                L.verify_gpu_catalog(plan, altered)

    def test_ordered_readonly_failure_does_not_consume_dispatch_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = LaunchTests().run_launch(temp, ordered=True,
                preflight_error=RuntimeError('Synthetic read-only preflight rejection'))
            state = Path(temp) / 'state'
            self.assertEqual(events, [])
            self.assertEqual(result['state'], 'failed')
            self.assertEqual(result['phase'], 'readonly_preflight')
            self.assertFalse(result['dispatch_attempted'])
            self.assertFalse(result['remote_intent_attempted'])
            self.assertTrue((state / 'controller-invocation.json').exists())
            for name in ('approval-consumed.json', 'dispatch-authorization.json',
                         'remote-approval-intent.json', 'launch-dispatched.json'):
                self.assertFalse((state / name).exists(), name)
            L.require_no_dispatch_evidence(state)

    def test_generation_continuation_preserves_history_and_blocks_after_remote_intent_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            historical_approval = b'{"historical":true}'
            (root / 'approval-consumed.json').write_bytes(historical_approval)
            (root / 'controller-result.json').write_text('{"state":"failed"}')
            L.require_no_dispatch_evidence(root)
            result, events = LaunchTests().run_launch(temp, ordered=True, state_name='generation-3',
                dispatch_state=root, remote_intent_error=True)
            self.assertEqual((root / 'approval-consumed.json').read_bytes(), historical_approval)
            self.assertTrue((root / 'dispatch-authorization.json').exists())
            self.assertTrue((root / 'generation-3/approval-consumed.json').exists())
            self.assertEqual(result['phase'], 'remote_launch_intent')
            self.assertTrue(result['remote_intent_attempted'])
            self.assertFalse(result['dispatch_attempted'])
            self.assertFalse(any(operation == 'run-instances' for operation, _ in events))
            with self.assertRaises(FileExistsError):
                LaunchTests().run_launch(temp, ordered=True, state_name='generation-4', dispatch_state=root)

    def test_possible_run_instances_dispatch_is_marked_and_cannot_cross_generations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, events = LaunchTests().run_launch(temp, ordered=True, state_name='generation-3',
                dispatch_state=root, ambiguous=True, remote_intent_delay=7)
            self.assertTrue(result['dispatch_attempted'])
            self.assertTrue(result['remote_intent_attempted'])
            self.assertTrue((root / 'generation-3/launch-dispatched.json').exists())
            global_record = json.loads((root / 'dispatch-authorization.json').read_text())
            local_record = json.loads((root / 'generation-3/launch-dispatched.json').read_text())
            self.assertEqual(local_record, global_record)
            self.assertEqual(local_record['at'], 1000)
            plan = dict(schema=L.INFERENCE_SCHEMA, identity=identity(),
                        cost={'termination_grace_seconds': 20})
            self.assertEqual(L.cleanup_cutoff(plan, 'a' * 64, root / 'generation-3'), 15420)
            self.assertEqual(sum(operation == 'run-instances' for operation, _ in events), 1)
            with self.assertRaises(FileExistsError):
                LaunchTests().run_launch(temp, ordered=True, state_name='generation-4', dispatch_state=root)



class FollowupInferenceAdmissionTests(unittest.TestCase):
    """Synthetic new-lease evidence, never recycled ordered execution receipts."""

    def fixture(self, directory):
        plan, approval = OrderedInferenceAdmissionTests().fixture(directory)
        plan['schema'] = L.FOLLOWUP_SCHEMA
        workload = plan['workload']
        del workload['models']['qwen8']
        del workload['checkpoint_keys']
        workload['phase_seconds'] = dict(L.FOLLOWUP_PHASES)
        history = dict(schema='trid-hf14-followup-history-v1',
            source_run_id='historical-synthetic-test', source_plan_sha256='1' * 64,
            source_checkpoint_sha256='2' * 64, source_result_sha256='3' * 64,
            source_model_key='qwen8', qualified=False, continuation_reason='frozen_gate_failed')
        path = Path(directory) / 'historical-evidence.json'
        path.write_bytes(L.canonical(history))
        workload['historical_evidence'] = dict(
            path=path.name, bytes=path.stat().st_size, sha256=L.digest(path))
        plan['bundle']['files'].append(workload['historical_evidence'])
        plan['iam']['instance_role']['inline']['ExactRunOnly']['Statement'].pop()
        plan['output_allowlist'] = sorted(L.ordered_output_members(plan))
        approval.update(schema='trid-hf14-followup-approval-v1',
                        workload_sha256=L.object_sha(workload))
        return plan, approval

    def reseal_data(self, directory, plan, key, value):
        pin = plan['workload'][key]
        path = Path(directory) / pin['path']
        path.write_bytes(L.canonical(value))
        pin.update(bytes=path.stat().st_size, sha256=L.digest(path))

    def completion_fixture(self, plan, group='nonthinking'):
        content = OrderedInferenceAdmissionTests().model_artifacts(plan, 'qwen14', group)
        bound = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                     instance_id=instance(plan['identity'])['InstanceId'])
        runtime = dict(**bound, packages=plan['runtime']['packages'],
            torch=plan['runtime']['packages']['torch'], cuda=plan['runtime']['torch_cuda'],
            count=1, bf16=True, name=plan['runtime']['gpu_name'])
        content['runtime.json'] = L.canonical(runtime)
        stages = []
        for index, phase in enumerate(L.FOLLOWUP_PHASES):
            stage = dict(run_id=bound['run_id'], plan_sha256=bound['plan_sha256'],
                phase=phase, state='complete', started_epoch=1100 + index * 100,
                finished_epoch=1190 + index * 100)
            stages.append(stage)
            content[phase + '-receipt.json'] = L.canonical(stage)
        completion = dict(**bound, inference_success=True, stop_errors=[], runtime=runtime,
            runtime_unit=dict(**bound, phase='runtime', state='complete',
                unit=bound['run_id'] + '-runtime', invocation='new-synthetic-invocation', gpu=gpu()),
            stages=stages)
        return completion, content

    def export_archive(self, directory, plan, completion, content):
        manifest = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
            status=copy.deepcopy({k: v for k, v in completion.items() if k != 'export'}),
            files=[dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                   for name, payload in sorted(content.items())])
        members = {**content, 'export-manifest.json': L.canonical(manifest)}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:', format=tarfile.USTAR_FORMAT) as archive:
            for name, payload in sorted(members.items()):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        payload = buffer.getvalue()
        path = Path(directory) / 'followup-export.tar'
        path.write_bytes(payload)
        completion['export'] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        return path, manifest

    def validate_completion(self, plan, completion, path, manifest):
        return L.validate_followup_completion(completion, plan, 'a' * 64,
            instance(plan['identity'])['InstanceId'], path, manifest)

    def test_new_fourteen_only_plan_and_blind_data_admit_with_exact_read_write_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, approval = self.fixture(temp)
            result = L.validate_plan(plan, approval, 'a' * 64, now=1000)
            loaded = L.validate_data(temp, plan)
            self.assertLessEqual(result['additional_pretax_bound_usd'], 20)
            self.assertEqual(set(plan['workload']['models']), {'qwen14'})
            self.assertEqual(plan['workload']['phase_seconds'],
                             dict(acquire14=900, dev14=2400, val14=2400))
            self.assertEqual({key: len(rows) for key, rows in loaded.items()},
                             dict(development_inputs=6, validation_inputs=16))
            self.assertTrue(L.inference_mode(plan))
            self.assertTrue(L.followup_mode(plan))
            self.assertFalse(L.ordered_mode(plan))
            self.assertNotIn('checkpoint_keys', plan['workload'])
            self.assertNotIn('recipe', plan)
            self.assertNotIn('parent-decision.json', plan['output_allowlist'])
            statements = plan['iam']['instance_role']['inline']['ExactRunOnly']['Statement']
            self.assertEqual([row['Action'] for row in statements],
                             [['s3:GetObject'], ['s3:PutObject']])

    def test_old_or_expanded_approval_cannot_authorize_new_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, approval = self.fixture(temp)
            for key, value in (
                    ('schema', 'trid-ordered-hf-inference-approval-v1'),
                    ('schema', 'trid-grounded-hf-approval-v1'),
                    ('run_id', 'historical-synthetic-test'), ('plan_sha256', '1' * 64),
                    ('workload_sha256', '2' * 64), ('instance_limit', 2),
                    ('additional_pretax_usd', 50), ('automatic_retry', True), ('approved', False),
                    ('limits', {**L.LIMITS, 'lifetime_seconds': 14401})):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(RuntimeError, 'USD20'):
                    L.validate_plan(plan, {**approval, key: value}, 'a' * 64, now=1000)

    def test_followup_rejects_reintroduced_eight_model_phase_and_decision_channel(self):
        changes = [
            lambda w: w['models'].update(qwen8=copy.deepcopy(w['models']['qwen14'])),
            lambda w: w['models']['qwen14'].update(revision='0' * 40),
            lambda w: w.update(checkpoint_keys={}),
            lambda w: w.pop('historical_evidence'),
            lambda w: w['phase_seconds'].update(acquire8=900),
            lambda w: w['phase_seconds'].update(parent_decision=300),
            lambda w: w['phase_seconds'].update(dev14=2401),
            lambda w: w['profiles'][0].update(seed=15),
            lambda w: w['selection_policy'].update(require_both_seeds=False),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as temp:
                plan, approval = self.fixture(temp)
                change(plan['workload'])
                approval['workload_sha256'] = L.object_sha(plan['workload'])
                with self.assertRaises(RuntimeError):
                    L.validate_plan(plan, approval, 'a' * 64, now=1000)
        for field in ('recipe', 'data', 'evaluation', 'phase_seconds', 'model'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                plan, approval = self.fixture(temp)
                plan[field] = {}
                with self.assertRaises(RuntimeError):
                    L.validate_plan(plan, approval, 'a' * 64, now=1000)

    def test_followup_worker_rejects_listing_missing_write_and_foreign_prefix(self):
        for mutation in ('listing', 'missing_write', 'foreign_read', 'broad_write'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                plan, approval = self.fixture(temp)
                statements = plan['iam']['instance_role']['inline']['ExactRunOnly']['Statement']
                if mutation == 'listing':
                    ident = plan['identity']
                    statements.append(dict(Effect='Allow', Action=['s3:ListBucket'],
                        Resource=[f"arn:aws:s3:::{ident['bucket']}"],
                        Condition={'StringEquals': {'s3:prefix': ident['code_prefix'] + 'parent-decision.json'}}))
                elif mutation == 'missing_write':
                    statements.pop()
                elif mutation == 'foreign_read':
                    statements[0]['Resource'] = ['arn:aws:s3:::synthetic-bucket/code/historical/*']
                else:
                    statements[1]['Resource'] = ['*']
                with self.assertRaisesRegex(RuntimeError, 'Worker role'):
                    L.validate_plan(plan, approval, 'a' * 64, now=1000)

    def test_history_requires_exact_failed_eight_provenance_from_different_run(self):
        mutations = [
            lambda h, p: h.update(schema='trid-ordered-hf-inference-v1'),
            lambda h, p: h.update(source_run_id=p['identity']['run_id']),
            lambda h, p: h.update(source_run_id='../historical'),
            lambda h, p: h.update(source_model_key='qwen14'),
            lambda h, p: h.update(qualified=True),
            lambda h, p: h.update(qualified=0),
            lambda h, p: h.update(continuation_reason='passed'),
            lambda h, p: h.update(runtime={'state': 'complete'}),
            lambda h, p: h.pop('source_result_sha256'),
        ]
        for field in ('source_plan_sha256', 'source_checkpoint_sha256', 'source_result_sha256'):
            mutations.append(lambda h, p, field=field: h.update({field: 'not-a-sha256'}))
        for index, change in enumerate(mutations):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                history = json.loads((Path(temp) / plan['workload']['historical_evidence']['path']).read_text())
                change(history, plan)
                self.reseal_data(temp, plan, 'historical_evidence', history)
                with self.assertRaisesRegex(RuntimeError, 'historical provenance|Historical evidence'):
                    L.validate_data(temp, plan)

    def test_history_file_hash_size_bundle_pin_and_bound_are_required(self):
        for mutation in ('hash', 'size', 'bundle_pin', 'over_bound'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                pin = plan['workload']['historical_evidence']
                path = Path(temp) / pin['path']
                if mutation == 'hash':
                    path.write_bytes(path.read_bytes().replace(b'1' * 64, b'9' * 64))
                elif mutation == 'size':
                    path.write_bytes(path.read_bytes() + b' ')
                elif mutation == 'bundle_pin':
                    plan['workload']['historical_evidence'] = {**pin, 'sha256': '9' * 64}
                else:
                    pin['bytes'] = 1024**2 + 1
                with self.assertRaises(RuntimeError):
                    L.validate_data(temp, plan)

    def test_followup_data_still_rejects_labels_overlap_and_changed_reference_messages(self):
        for mutation in ('labels', 'overlap', 'reference_messages', 'closure'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                key = {'labels': 'validation_inputs', 'reference_messages': 'development_references',
                       'closure': 'source_closure'}.get(mutation)
                if mutation == 'overlap':
                    plan['workload']['development_case_ids'][0] = plan['workload']['validation_case_ids'][0]
                else:
                    value = json.loads((Path(temp) / plan['workload'][key]['path']).read_text())
                    if mutation == 'labels':
                        value[0]['expected'] = 'Never export a private answer.'
                    elif mutation == 'reference_messages':
                        value['cases'][0]['messages'][0]['content'] = 'Different development input.'
                    else:
                        value['run_id'] = 'historical-synthetic-test'
                    self.reseal_data(temp, plan, key, value)
                with self.assertRaises(RuntimeError):
                    L.validate_data(temp, plan)

    def test_actual_tar_completion_admits_both_independently_selected_profile_groups(self):
        for group in ('nonthinking', 'thinking'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                completion, content = self.completion_fixture(plan, group)
                path, manifest = self.export_archive(temp, plan, completion, content)
                self.validate_completion(plan, completion, path, manifest)
                self.assertEqual([s['phase'] for s in completion['stages']], ['acquire14', 'dev14', 'val14'])
                self.assertEqual(len([name for name in content if name.startswith('qwen14-dev-')
                                      and name.endswith('-receipt.json')]), 4)
                self.assertEqual(len([name for name in content if name.startswith('qwen14-val-')
                                      and name.endswith('-receipt.json')]), 2)
                self.assertFalse(any(name.startswith('qwen8') or 'parent' in name for name in content))

    def test_forged_historical_runtime_or_runtime_unit_is_not_new_execution(self):
        for target, field, value in (
                ('runtime', 'run_id', 'historical-synthetic-test'),
                ('runtime', 'plan_sha256', '1' * 64),
                ('runtime', 'instance_id', 'i-11111111111111111'),
                ('runtime', 'bf16', False), ('runtime', 'count', 2),
                ('runtime', 'packages', {}), ('runtime', 'cuda', 'historical-cuda'),
                ('runtime_unit', 'run_id', 'historical-synthetic-test'),
                ('runtime_unit', 'plan_sha256', '1' * 64),
                ('runtime_unit', 'instance_id', 'i-11111111111111111'),
                ('runtime_unit', 'unit', 'historical-synthetic-test-runtime'),
                ('runtime_unit', 'invocation', ''), ('runtime_unit', 'state', 'failed')):
            with self.subTest(target=target, field=field), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                completion, content = self.completion_fixture(plan)
                completion[target][field] = value
                content['runtime.json'] = L.canonical(completion['runtime'])
                path, manifest = self.export_archive(temp, plan, completion, content)
                with self.assertRaisesRegex(RuntimeError, 'runtime'):
                    self.validate_completion(plan, completion, path, manifest)

    def test_old_checkpoint_decision_training_and_eight_phases_cannot_supply_completion(self):
        for mutation in ('checkpoint', 'decision', 'training_success', 'acquire8', 'dev8',
                         'val8', 'parent_decision', 'incomplete', 'stop_error', 'stage_identity',
                         'stage_time', 'archived_stage'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                completion, content = self.completion_fixture(plan)
                if mutation in ('checkpoint', 'decision', 'training_success'):
                    completion[mutation] = True if mutation == 'training_success' else {'run_id': 'historical-synthetic-test'}
                elif mutation in ('acquire8', 'dev8', 'val8', 'parent_decision'):
                    completion['stages'].insert(0, dict(phase=mutation, state='complete'))
                elif mutation == 'stop_error':
                    completion['stop_errors'] = ['synthetic unreaped child']
                else:
                    stage = completion['stages'][-1]
                    if mutation == 'incomplete':
                        stage['state'] = 'incomplete'
                    elif mutation == 'stage_identity':
                        stage['run_id'] = 'historical-synthetic-test'
                    elif mutation == 'stage_time':
                        stage['finished_epoch'] = stage['started_epoch'] - 1
                    else:
                        stage['finished_epoch'] += 1
                    if mutation != 'archived_stage':
                        content['val14-receipt.json'] = L.canonical(stage)
                path, manifest = self.export_archive(temp, plan, completion, content)
                with self.assertRaises(RuntimeError):
                    self.validate_completion(plan, completion, path, manifest)

    def test_archive_bytes_manifest_and_exact_member_binding_reject_tampering(self):
        for mutation in ('bytes', 'manifest_status', 'supplied_manifest', 'member_bytes',
                         'missing_log', 'old_member', 'extra_selected_group'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                completion, content = self.completion_fixture(plan)
                if mutation == 'missing_log':
                    del content['runtime.log']
                elif mutation == 'old_member':
                    content['qwen8-checkpoint.json'] = b'{}'
                elif mutation == 'extra_selected_group':
                    content['qwen14-val-thinking-seed14.log'] = b'synthetic extra group'
                path, manifest = self.export_archive(temp, plan, completion, content)
                if mutation == 'bytes':
                    payload = bytearray(path.read_bytes())
                    payload[-1] ^= 1
                    path.write_bytes(payload)
                elif mutation == 'manifest_status':
                    completion['runtime_unit']['invocation'] = 'different-terminal-unit'
                elif mutation == 'supplied_manifest':
                    manifest['files'][0]['sha256'] = 'f' * 64
                elif mutation == 'member_bytes':
                    with tarfile.open(path, 'r:') as archive:
                        offset = archive.getmember('runtime.log').offset_data
                    payload = bytearray(path.read_bytes())
                    payload[offset] ^= 1
                    path.write_bytes(payload)
                    completion['export']['sha256'] = L.digest(path)
                with self.assertRaises(RuntimeError):
                    self.validate_completion(plan, completion, path, manifest)

    def test_all_six_profile_receipts_bind_current_identity_model_input_output_and_cases(self):
        fields = dict(run_id='historical-synthetic-test', plan_sha256='1' * 64,
            model_sha256='2' * 64, input_sha256='3' * 64, output_sha256='4' * 64,
            output_bytes=3, completed_case_ids=[], question_deadline_epoch=1234, profile={},
            model_key='qwen8', state='incomplete', phase='val8')
        for cohort, group, seed in (
                ('dev', 'nonthinking', 14), ('dev', 'nonthinking', 29),
                ('dev', 'thinking', 14), ('dev', 'thinking', 29),
                ('val', 'nonthinking', 14), ('val', 'nonthinking', 29)):
            for field, value in fields.items():
                with self.subTest(cohort=cohort, group=group, seed=seed, field=field), \
                        tempfile.TemporaryDirectory() as temp:
                    plan, _ = self.fixture(temp)
                    completion, content = self.completion_fixture(plan)
                    name = f'qwen14-{cohort}-{group}-seed{seed}-receipt.json'
                    receipt = json.loads(content[name])
                    receipt[field] = value
                    content[name] = L.canonical(receipt)
                    path, manifest = self.export_archive(temp, plan, completion, content)
                    with self.assertRaisesRegex(RuntimeError, 'Profile completion'):
                        self.validate_completion(plan, completion, path, manifest)

    def test_selection_cannot_reuse_historical_or_validation_evidence(self):
        for field, value in (
                ('run_id', 'historical-synthetic-test'), ('model_key', 'qwen8'),
                ('profiles_sha256', '1' * 64), ('development_inputs_sha256', '2' * 64),
                ('development_references_sha256', '3' * 64), ('selected_profiles', []),
                ('dev_outputs', []), ('selection_policy', {})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                plan, _ = self.fixture(temp)
                completion, content = self.completion_fixture(plan)
                selection = json.loads(content['qwen14-selection.json'])
                selection[field] = value
                content['qwen14-selection.json'] = L.canonical(selection)
                path, manifest = self.export_archive(temp, plan, completion, content)
                with self.assertRaises(RuntimeError):
                    self.validate_completion(plan, completion, path, manifest)

    def test_collect_export_success_fetches_only_new_completion_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(temp)
            completion, content = self.completion_fixture(plan)
            path, _ = self.export_archive(temp, plan, completion, content)
            state = Path(temp) / 'state'
            state.mkdir()
            (state / 'release.json').write_bytes(L.canonical(release(plan)))
            prefix = plan['identity']['output_prefix']
            objects = {prefix + 'completion.json': L.canonical(completion),
                       prefix + 'export.tar': path.read_bytes()}
            calls = []

            def transport(service, operation, *args):
                self.assertEqual((service, operation), ('s3api', 'get-object'))
                key = args[args.index('--key') + 1]
                self.assertIn(key, objects, 'Followup must never fetch checkpoint or parent decision')
                calls.append(key)
                Path(args[-1]).write_bytes(objects[key])

            result = L.collect_export(transport, plan, 'a' * 64, state, completion['instance_id'])
            self.assertTrue(result['native_pipeline_success'])
            self.assertEqual(result['semantic_acceptance'], 'not_evaluated')
            self.assertEqual(result['export_sha256'], completion['export']['sha256'])
            self.assertEqual(calls, [prefix + 'completion.json', prefix + 'export.tar'])

    def test_followup_shared_launch_release_cleanup_and_error_boundaries_are_one_shot(self):
        for boundary in ('release', 'ambiguous', 'schedule_wrong', 'prerequisites_wrong',
                         'preflight_error', 'remote_intent_error'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temp:
                case = LaunchTests()
                fixture = case.fixture

                def followup_fixture(root):
                    plan = fixture(root)
                    plan['schema'] = L.FOLLOWUP_SCHEMA
                    return plan

                options = {} if boundary == 'release' else {boundary: True}
                if boundary == 'preflight_error':
                    options[boundary] = RuntimeError('Synthetic read-only rejection')
                with patch.object(case, 'fixture', side_effect=followup_fixture):
                    result, events = case.run_launch(temp, **options)
                dispatched = boundary not in ('preflight_error', 'remote_intent_error')
                self.assertEqual(sum(op == 'run-instances' for op, _ in events), int(dispatched))
                released = [args for op, args in events if op == 'put-object'
                            and any(str(arg).endswith('release.json') for arg in args)]
                self.assertEqual(bool(released), boundary == 'release')
                self.assertFalse(result['inference_success'])
                self.assertNotIn('training_success', result)
                if dispatched:
                    self.assertEqual(result['cleanup']['cleanup_state'], 'complete')
                    self.assertTrue(result['cleanup']['instance_terminated'])
                    self.assertEqual(result['cleanup']['remaining_volumes'], [])
                if boundary == 'preflight_error':
                    self.assertEqual(events, [])
                    self.assertFalse(result['dispatch_attempted'])
                    self.assertFalse(result['remote_intent_attempted'])
                    L.require_no_dispatch_evidence(Path(temp) / 'state')
                elif boundary == 'remote_intent_error':
                    self.assertTrue(result['remote_intent_attempted'])
                    self.assertFalse(result['dispatch_attempted'])
                    self.assertTrue((Path(temp) / 'state/dispatch-authorization.json').exists())


if __name__ == '__main__':
    unittest.main()
