"""Offline only: no subprocesses, credentials, AWS requests or paid resources."""
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.aws_eval import fresh_eight_launch as fresh


SHA = 'a' * 64


def pin(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content if isinstance(content, bytes) else json.dumps(content).encode()
    path.write_bytes(raw)
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def registration(root):
    run = 'fresh-eight-offline-20261115'
    ident = dict(run_id=run, account='148761639313', region='ap-northeast-2',
        bucket='trid-training-148761639313', experiment='fresh-eight', eval_tag='fresh-eight-test',
        instance_role=run + '-worker', scheduler_role=run + '-terminator', instance_profile=run + '-profile',
        schedule_group=run + '-group', schedule_name=run + '-deadline', code_prefix='code/' + run + '/',
        output_prefix='output/' + run + '/')
    ident['tags'] = {'trid:eval': ident['eval_tag'], 'trid:experiment': ident['experiment'], 'trid:run': run}
    instance_trust = {'Version': '2012-10-17', 'Statement': [
        dict(Effect='Allow', Principal={'Service': 'ec2.amazonaws.com'}, Action='sts:AssumeRole')]}
    scheduler_trust = {'Version': '2012-10-17', 'Statement': [dict(Effect='Allow',
        Principal={'Service': 'scheduler.amazonaws.com'}, Action='sts:AssumeRole',
        Condition={'StringEquals': {'aws:SourceAccount': ident['account']}, 'ArnEquals': {
            'aws:SourceArn': f"arn:aws:scheduler:{ident['region']}:{ident['account']}:schedule-group/{ident['schedule_group']}"}})]}
    code = f"arn:aws:s3:::{ident['bucket']}/{ident['code_prefix']}"
    statements = [dict(Effect='Allow', Action=['s3:GetObject'], Resource=[code + name for name in
        ('plan.json', 'bundle.tar', 'approval-consumed.json', 'termination.json', 'release.json')]),
        dict(Effect='Allow', Action=['s3:PutObject'], Resource=[f"arn:aws:s3:::{ident['bucket']}/{ident['output_prefix']}*"]),
        dict(Effect='Allow', Action=['s3:ListBucket'], Resource=[f"arn:aws:s3:::{ident['bucket']}"],
            Condition={'StringEquals': {'s3:prefix': [ident['code_prefix'] + name for name in ('termination.json', 'release.json')]}})]
    termination = [dict(Effect='Allow', Action='ec2:TerminateInstances',
        Resource=f"arn:aws:ec2:{ident['region']}:{ident['account']}:instance/*",
        Condition={'StringEquals': {f'ec2:ResourceTag/{k}': v for k, v in ident['tags'].items()}})]
    iam = {key: dict(arn=f"arn:aws:iam::{ident['account']}:role/{ident[key]}", trust=trust,
                     inline={'ExactRunOnly': {'Version': '2012-10-17', 'Statement': policy}})
           for key, trust, policy in (('instance_role', instance_trust, statements),
                                      ('scheduler_role', scheduler_trust, termination))}
    authorization = Path(__file__).resolve().parents[1] / 'runs/mois-eight-training-recovery-20261116T014906Z/authorization.json'
    auth = pin(root, 'authorization.json', authorization.read_bytes())
    ledger = pin(root, 'aggregate-ledger.json', dict(schema='trid-fresh-eight-aggregate-budget-v2',
        run_id=run, authorization=auth, previous_run_id='mois-fresh-eight-gpu-20261115-094150',
        previous_authorization_consumed=True, prior_conditional_all_in_bound_usd='5.89',
        current_run_cap_usd='4.11', aggregate_budget_usd='10.00', additional_gpu_executions=1,
        automatic_retry=False, automatic_extension=False))
    prices = pin(root, 'cost-evidence.json', {'offline_fixture': True})
    envelope = pin(root, 'resource-envelope.json', dict(schema='trid-fresh-eight-training-resource-envelope-v2',
        identity=ident, authorization=auth, aggregate_ledger=ledger,
        budget_usd='10.00', prior_conditional_all_in_bound_usd='5.89', current_run_cap_usd='4.11',
        billing_hard_cap=False, cost_evidence=prices, instance_type='g6e.2xlarge',
        root_gib=120, root_encrypted=True, root_delete_on_termination=True,
        gpu_execution_lease_seconds_max=1800, export_reserve_seconds=300,
        termination_grace_seconds_max=900, total_billable_lifetime_seconds_max=2700,
        whole_host_egress_bytes_max=2 * 1024**3, controller_retrieval_bytes_max=1024**3,
        priced_egress_bytes=3 * 1024**3, combined_retained_s3_bytes_max=1024**3,
        output_archive_bytes_max=512 * 1024**2, scheduler_invocations_max=10000,
        tax_assumption='0.10', contingency_before_tax_usd='1.00',
        s3_expiration_days=30, multipart_abort_days=1, bundle_bytes_max=32 * 1024**2))
    sources = [pin(root, name, Path(module.__file__).read_bytes()) for name, module in (
        ('scripts/aws_eval/fresh_eight_launch.py', fresh),
        ('scripts/aws_eval/grounded_hf_lifecycle.py', fresh.lifecycle))]
    data = pin(root, 'inputs/data.json', {'input': 'sealed'})
    return dict(schema=fresh.SCHEMA, approved=True, automatic_retry=False, identity=ident,
        controller_arn='arn:aws:iam::148761639313:user/parent', authorization=auth,
        aggregate_ledger=ledger, resource_envelope=envelope,
        sources=sources, iam=iam, network=dict(vpc_id='vpc-123', subnet_id='subnet-123',
            availability_zone='ap-northeast-2a', security_group_name=run + '-sg', security_group_description='Fresh owned SG'),
        bucket_lifecycle_before={'Rules': [dict(ID='historical', Status='Enabled', Filter={'Prefix': 'old/'},
            Expiration={'Days': 30})], 'TransitionDefaultMinimumObjectSize': 'all_storage_classes_128K'},
        retention_days=30, lifecycle_rule_ids=dict(code_prefix=run + '-code', output_prefix=run + '-output'),
        bundle=dict(local_path='bundle.tar', files=[*sources, data]))


def missing(service, operation, code):
    return fresh.lifecycle.AWSCallError(service, operation, f'An error occurred ({code}) when calling operation')


def evaluation_registration(root):
    reg = registration(root)
    old_run = reg['identity']['run_id']
    run = 'mois-eight-cuda-eval-20261116-134815'
    reg = json.loads(json.dumps(reg).replace(old_run, run))
    reg['schema'] = fresh.EVALUATION_SCHEMA
    del reg['aggregate_ledger']
    auth_name = 'runs/mois-eight-cuda-evaluation-20261116T134815Z/authorization.json'
    reg['authorization'] = pin(root, auth_name, (Path(__file__).resolve().parents[1] / auth_name).read_bytes())
    envelope = json.loads((root / reg['resource_envelope']['path']).read_bytes())
    for key in ('aggregate_ledger', 'prior_conditional_all_in_bound_usd', 'current_run_cap_usd'):
        del envelope[key]
    envelope.update(schema='trid-cuda-evaluation-resource-envelope-v1', identity=reg['identity'],
        authorization=reg['authorization'], budget_usd='20', gpu_execution_lease_seconds_max=18900,
        total_billable_lifetime_seconds_max=19800, combined_retained_s3_bytes_max=2 * 1024**3,
        bundle_bytes_max=128 * 1024**2)
    reg['resource_envelope'] = pin(root, 'evaluation-envelope.json', envelope)
    reg['lineage_binding'] = pin(root, 'lineage-binding.json', {'synthetic_lineage': True})
    reg['bundle']['files'] += [reg[key] for key in ('authorization', 'resource_envelope', 'lineage_binding')]
    return reg


class CudaEvaluationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.state = self.root / 'state'
        self.state.mkdir()
        self.reg = evaluation_registration(self.root)

    def test_new_registration_reauthenticates_parent_lineage(self):
        with patch.object(fresh, 'validate_lineage') as authenticate:
            self.assertIs(fresh.validate_registration(self.reg, self.root), self.reg)
            authenticate.assert_called_once_with(self.root, self.reg['lineage_binding'])
        with patch.object(fresh, 'validate_lineage', side_effect=RuntimeError('original lineage changed')):
            with self.assertRaisesRegex(RuntimeError, 'original lineage changed'):
                fresh.validate_registration(self.reg, self.root)

    def test_inactive_registration_still_authenticates_archived_bytes(self):
        with patch.object(fresh, 'validate_lineage', side_effect=AssertionError('remote provenance is controller-only')):
            fresh.validate_registration(self.reg, self.root, active=False)
            (self.root / self.reg['authorization']['path']).write_text('{}')
            with self.assertRaises(RuntimeError):
                fresh.validate_registration(self.reg, self.root, active=False)

    def test_new_registration_rejects_old_grant_and_history_upload(self):
        with patch.object(fresh, 'validate_lineage'):
            self.reg['aggregate_ledger'] = {'sha256': SHA}
            with self.assertRaisesRegex(RuntimeError, 'Exact parent prerequisite'):
                fresh.validate_registration(self.reg, self.root)
            del self.reg['aggregate_ledger']
            self.reg['bundle']['files'].append(pin(self.root, 'recovered-export.tar', b'old archive'))
            with self.assertRaisesRegex(RuntimeError, 'Historical recovery archives'):
                fresh.validate_registration(self.reg, self.root)

    def test_new_registration_requires_sealed_lineage_and_keeps_shared_rules(self):
        before = copy.deepcopy(self.reg['bucket_lifecycle_before'])
        after = fresh.lifecycle_merge(self.reg)
        self.assertEqual(after['Rules'][:-2], before['Rules'])
        self.assertEqual(self.reg['bucket_lifecycle_before'], before)
        self.reg['bundle']['files'] = [p for p in self.reg['bundle']['files']
                                      if p != self.reg['lineage_binding']]
        with patch.object(fresh, 'validate_lineage'):
            with self.assertRaisesRegex(RuntimeError, 'bundle must bind'):
                fresh.validate_registration(self.reg, self.root)

    def test_evaluation_bundle_accepts_generic_cap_without_training_cap(self):
        bundle = fresh.prepare_bundle(self.reg, self.root, self.state)
        self.assertLess(bundle['bytes'], 128 * 1024**2)
        with tarfile.open(self.root / bundle['local_path']) as archive:
            self.assertEqual(set(archive.getnames()), {p['path'] for p in self.reg['bundle']['files']})
        envelope = json.loads((self.root / self.reg['resource_envelope']['path']).read_bytes())
        envelope['bundle_bytes_max'] = 256 * 1024**2
        self.reg['resource_envelope'] = pin(self.root, 'oversized-envelope.json', envelope)
        self.reg['bundle']['local_path'] = 'oversized.tar'
        with self.assertRaisesRegex(RuntimeError, 'Bundle too large'):
            fresh.prepare_bundle(self.reg, self.root, self.state)


class FreshPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.state = self.root / 'state'
        self.state.mkdir()
        self.reg = registration(self.root)

    def test_registration_authenticates_actual_inputs(self):
        self.assertIs(fresh.validate_registration(self.reg, self.root), self.reg)
        (self.root / 'authorization.json').write_text('{}')
        with self.assertRaises(RuntimeError):
            fresh.validate_registration(self.reg, self.root)

    def test_registration_rejects_consumed_or_repriced_aggregate_ledger(self):
        original = copy.deepcopy(self.reg)
        for key, value in (('previous_authorization_consumed', False),
                           ('prior_conditional_all_in_bound_usd', '0.00'),
                           ('current_run_cap_usd', '10.00'), ('additional_gpu_executions', 2)):
            with self.subTest(key=key):
                self.reg = copy.deepcopy(original)
                ledger = json.loads((self.root / self.reg['aggregate_ledger']['path']).read_text())
                ledger.update({key: value})
                self.reg['aggregate_ledger'] = pin(self.root, 'mutated-ledger.json', ledger)
                with self.assertRaisesRegex(RuntimeError, 'Parent aggregate ledger'):
                    fresh.validate_registration(self.reg, self.root)

    def test_registration_rejects_historical_schema_and_resource_extensions(self):
        historical = copy.deepcopy(self.reg)
        historical['schema'] = 'trid-fresh-eight-prerequisites-v1'
        with self.assertRaisesRegex(RuntimeError, 'Exact parent prerequisite'):
            fresh.validate_registration(historical, self.root)
        original = json.loads((self.root / self.reg['resource_envelope']['path']).read_text())
        for key, value in (('gpu_execution_lease_seconds_max', 7200),
                           ('export_reserve_seconds', 600), ('root_encrypted', False),
                           ('termination_grace_seconds_max', 1800),
                           ('total_billable_lifetime_seconds_max', 9000)):
            with self.subTest(key=key):
                envelope = dict(original, **{key: value})
                self.reg['resource_envelope'] = pin(self.root, 'mutated-envelope.json', envelope)
                with self.assertRaisesRegex(RuntimeError, 'resource envelope differs'):
                    fresh.validate_registration(self.reg, self.root)

    def test_registration_never_reuses_old_one_shot_grant(self):
        self.reg['authorization'] = pin(self.root, 'old-authorization.json', dict(
            schema='trid-fresh-eight-gpu-authorization-v1', additional_budget_usd='10.00'))
        with self.assertRaisesRegex(RuntimeError, 'original aggregate grant'):
            fresh.validate_registration(self.reg, self.root)

    def test_registration_rejects_expanded_worker_policy(self):
        self.reg['iam']['instance_role']['inline']['ExactRunOnly']['Statement'][0]['Resource'] = ['*']
        with self.assertRaises(RuntimeError):
            fresh.validate_registration(self.reg, self.root)

    def test_registration_rejects_unbound_historical_identity(self):
        self.reg['identity']['instance_role'] = 'historical-worker'
        with self.assertRaises(RuntimeError):
            fresh.validate_registration(self.reg, self.root)

    def test_registration_rejects_missing_helper_seal(self):
        self.reg['sources'] = self.reg['sources'][:1]
        with self.assertRaises(RuntimeError):
            fresh.validate_registration(self.reg, self.root)

    def test_lifecycle_append_preserves_entire_snapshot(self):
        before = copy.deepcopy(self.reg['bucket_lifecycle_before'])
        merged = fresh.lifecycle_merge(self.reg)
        self.assertEqual(self.reg['bucket_lifecycle_before'], before)
        self.assertEqual(merged['Rules'][:-2], before['Rules'])
        self.assertEqual(merged['TransitionDefaultMinimumObjectSize'], before['TransitionDefaultMinimumObjectSize'])
        for key, rule in zip(self.reg['lifecycle_rule_ids'], merged['Rules'][-2:]):
            self.assertEqual(rule['Filter'], {'Prefix': self.reg['identity'][key]})
            self.assertEqual(rule['Expiration'], {'Days': 30})
            self.assertEqual(rule['AbortIncompleteMultipartUpload'], {'DaysAfterInitiation': 1})

    def test_lifecycle_rejects_rule_id_collision(self):
        self.reg['bucket_lifecycle_before']['Rules'][0]['ID'] = self.reg['lifecycle_rule_ids']['code_prefix']
        with self.assertRaises(RuntimeError):
            fresh.lifecycle_merge(self.reg)

    def test_lifecycle_rejects_overlapping_transition(self):
        self.reg['bucket_lifecycle_before']['Rules'][0].update(Filter={'Prefix': ''}, Transitions=[{'Days': 1, 'StorageClass': 'GLACIER'}])
        with self.assertRaises(RuntimeError):
            fresh.lifecycle_merge(self.reg)

    def test_bundle_is_deterministic_explicit_only(self):
        (self.root / 'unsealed.txt').write_text('not in tar')
        one = fresh.prepare_bundle(self.reg, self.root, self.state)
        other_state = self.root / 'state-two'
        other_state.mkdir()
        self.reg['bundle']['local_path'] = 'second.tar'
        two = fresh.prepare_bundle(self.reg, self.root, other_state)
        self.assertEqual(one['sha256'], two['sha256'])
        with tarfile.open(self.root / 'bundle.tar') as archive:
            self.assertEqual(archive.getnames(), sorted(p['path'] for p in self.reg['bundle']['files']))
            self.assertTrue(all(m.mtime == m.uid == m.gid == 0 and m.isfile() for m in archive.getmembers()))
        with self.assertRaises(FileExistsError):
            fresh.prepare_bundle(self.reg, self.root, other_state)

    def test_bundle_rejects_credentials_before_output(self):
        self.reg['bundle']['files'].append(pin(self.root, '.aws/credentials', b'not real'))
        with self.assertRaises(RuntimeError):
            fresh.prepare_bundle(self.reg, self.root, self.state)
        self.assertFalse((self.root / 'bundle.tar').exists())

    def test_bundle_rejects_symlink_components(self):
        (self.root / 'inputs' / 'data.json').unlink()
        (self.root / 'inputs' / 'data.json').symlink_to(self.root / 'authorization.json')
        with self.assertRaises(RuntimeError):
            fresh.prepare_bundle(self.reg, self.root, self.state)

    def test_bundle_rejects_drift_before_output(self):
        (self.root / 'inputs' / 'data.json').write_text('changed')
        with self.assertRaises(RuntimeError):
            fresh.prepare_bundle(self.reg, self.root, self.state)
        self.assertFalse((self.root / 'bundle.tar').exists())

    def test_only_exact_service_error_proves_absence(self):
        client = Mock(side_effect=missing('iam', 'get-role', 'AccessDenied'))
        with self.assertRaises(RuntimeError):
            fresh.absent(client, 'iam', 'get-role', 'NoSuchEntity', '--role-name', 'fresh')
        client.side_effect = missing('iam', 'get-role', 'NoSuchEntity')
        self.assertTrue(fresh.absent(client, 'iam', 'get-role', 'NoSuchEntity')['absent'])
        client.side_effect = None
        client.return_value = {}
        with self.assertRaises(RuntimeError):
            fresh.absent(client, 'iam', 'get-role', 'NoSuchEntity')

    def test_mutation_intent_is_durable_before_call_and_ambiguity_not_retried(self):
        def fail(*args):
            self.assertTrue((self.state / 'create-intent.json').is_file())
            raise TimeoutError('ambiguous')
        client = Mock(side_effect=fail)
        with self.assertRaises(TimeoutError):
            fresh.mutate(client, self.state, SHA, 'create', 'iam', 'create-role', ['--role-name', 'fresh'], lambda r: r)
        with self.assertRaises(FileExistsError):
            fresh.mutate(client, self.state, SHA, 'create', 'iam', 'create-role', ['--role-name', 'fresh'], lambda r: r)
        self.assertEqual(client.call_count, 1)
        self.assertFalse((self.state / 'create-response.json').exists())

    def test_successful_mutation_saves_response_and_readback(self):
        client = Mock(return_value={'Role': {'RoleId': 'AROA123'}})
        observed = fresh.mutate(client, self.state, SHA, 'create', 'iam', 'create-role', [], lambda r: r['Role'])
        self.assertEqual(observed, {'RoleId': 'AROA123'})
        self.assertEqual(fresh.load(self.state / 'create-response.json')['response'], client.return_value)
        self.assertEqual(fresh.load(self.state / 'create-readback.json')['observation'], observed)

    def test_readback_failure_still_forbids_repeated_mutation(self):
        client = Mock(return_value={})
        def reject(response):
            raise RuntimeError('readback not established')
        with self.assertRaises(RuntimeError):
            fresh.mutate(client, self.state, SHA, 'create', 'iam', 'create-role', [], reject)
        with self.assertRaises(FileExistsError):
            fresh.mutate(client, self.state, SHA, 'create', 'iam', 'create-role', [], reject)
        self.assertEqual(client.call_count, 1)
        self.assertTrue((self.state / 'create-response.json').exists())

    def test_live_preparation_access_denial_never_mutates(self):
        def client(service, operation, *args):
            if operation == 'get-caller-identity':
                return {'Account': self.reg['identity']['account'], 'Arn': self.reg['controller_arn']}
            self.assertEqual(operation, 'get-role')
            raise missing(service, operation, 'AccessDenied')
        with self.assertRaises(RuntimeError):
            fresh.prepare_prerequisites(self.reg, SHA, self.state, client)
        self.assertEqual(list(self.state.iterdir()), [])

    def test_ambiguous_dispatch_blocks_prerequisite_cleanup(self):
        fresh.record(self.state / 'launch-dispatched.json', {'at': 1})
        client = Mock()
        with self.assertRaises(RuntimeError):
            fresh.verify_compute_cleanup(self.reg, self.state, client)
        client.assert_not_called()

    def test_profile_cross_region_references_block_cleanup(self):
        client = Mock(side_effect=[{'Regions': [{'RegionName': 'ap-northeast-2', 'OptInStatus': 'opt-in-not-required'},
            {'RegionName': 'us-east-1', 'OptInStatus': 'opt-in-not-required'}]}, {'IamInstanceProfileAssociations': []},
            {'Reservations': []}, {'IamInstanceProfileAssociations': [{'IamInstanceProfile': {'Arn': 'owned-profile'}}]}])
        with self.assertRaises(RuntimeError):
            fresh.verify_profile_unused(client, 'owned-profile')
        self.assertEqual(client.call_args.kwargs['region'], 'us-east-1')

    def test_security_group_references_block_cleanup(self):
        client = Mock(side_effect=[{'NetworkInterfaces': []}, {'SecurityGroupReferenceSet': [{'VpcId': 'unrelated'}]}])
        with self.assertRaises(RuntimeError):
            fresh.verify_sg_unused(client, 'sg-owned')
        self.assertEqual(client.call_count, 2)

    def test_lifecycle_write_uses_preserved_config_and_exact_readback(self):
        after = fresh.lifecycle_merge(self.reg)
        client = Mock(side_effect=[self.reg['bucket_lifecycle_before'], {}, after])
        result = fresh.append_lifecycle_configuration(self.reg, SHA, self.state, client)
        self.assertEqual(result, after)
        args = client.call_args_list[1].args
        self.assertEqual(args[:2], ('s3api', 'put-bucket-lifecycle-configuration'))
        configuration = json.loads(args[args.index('--lifecycle-configuration') + 1])
        self.assertEqual(configuration, {'Rules': after['Rules']})
        self.assertEqual(args[args.index('--transition-default-minimum-object-size') + 1],
                         self.reg['bucket_lifecycle_before']['TransitionDefaultMinimumObjectSize'])
        self.assertTrue((self.state / 'append-lifecycle-intent.json').is_file())

    def test_lifecycle_immediate_drift_does_not_write(self):
        changed = copy.deepcopy(self.reg['bucket_lifecycle_before'])
        changed['Rules'].append(dict(ID='unrelated-concurrent', Status='Disabled', Filter={'Prefix': 'other/'}))
        client = Mock(return_value=changed)
        with self.assertRaises(RuntimeError):
            fresh.append_lifecycle_configuration(self.reg, SHA, self.state, client)
        self.assertEqual(client.call_count, 1)
        self.assertFalse((self.state / 'append-lifecycle-intent.json').exists())

    def test_lifecycle_readback_drift_never_rolls_back_or_repeats(self):
        client = Mock(side_effect=[self.reg['bucket_lifecycle_before'], {}, {'Rules': []}])
        with self.assertRaises(RuntimeError):
            fresh.append_lifecycle_configuration(self.reg, SHA, self.state, client)
        self.assertEqual(client.call_count, 3)
        client.side_effect = None
        client.return_value = self.reg['bucket_lifecycle_before']
        with self.assertRaises(FileExistsError):
            fresh.append_lifecycle_configuration(self.reg, SHA, self.state, client)
        self.assertEqual(client.call_count, 4)

    def test_bundle_rejects_private_key_material_even_under_plain_name(self):
        self.reg['bundle']['files'].append(pin(self.root, 'inputs/plain.txt', b'-----BEGIN PRIVATE KEY-----'))
        with self.assertRaises(RuntimeError):
            fresh.prepare_bundle(self.reg, self.root, self.state)
        self.assertFalse((self.root / 'bundle.tar').exists())

    def test_partial_cleanup_deletes_proven_role_preserves_ambiguous_other_role(self):
        key = 'instance_role'
        role_identity = dict(Arn=self.reg['iam'][key]['arn'], RoleId='AROA-owned',
            RoleName=self.reg['identity'][key], Path='/', CreateDate='2026-11-15T00:00:00Z')
        observed = dict(identity=role_identity, inventory=dict(self.reg['iam'][key], inline={}),
                        tags=fresh.tags(self.reg['identity']))
        fresh.mutate(Mock(return_value={'Role': role_identity}), self.state, SHA, 'create-instance_role',
                     'iam', 'create-role', ['--role-name', self.reg['identity'][key]], lambda response: observed)
        fresh.record(self.state / 'create-scheduler_role-intent.json',
            dict(registration_sha256=SHA, service='iam', operation='create-role', args=[]))
        calls = []
        def client(service, operation, *args):
            calls.append((service, operation, args))
            if operation == 'list-instance-profiles-for-role':
                return {'InstanceProfiles': []}
            if operation == 'delete-role':
                return {}
            if operation == 'get-role':
                raise missing(service, operation, 'NoSuchEntity')
            self.fail('Unapproved cleanup call: ' + operation)
        with patch.object(fresh, 'verify_compute_cleanup'), patch.object(fresh, 'role_snapshot', return_value=observed):
            result = fresh.partial_prerequisite_cleanup(self.reg, SHA, self.state, client)
        self.assertEqual(result['cleanup_state'], 'unresolved')
        self.assertEqual(result['deleted_resources'], ['instance_role'])
        self.assertIn('scheduler_role', result['preserved_unresolved'])
        self.assertEqual(result['s3_mutations'], 0)
        self.assertTrue(all(service == 'iam' for service, _, _ in calls))
        self.assertEqual(sum(op == 'delete-role' for _, op, _ in calls), 1)

    def test_partial_cleanup_rejects_replaced_role_without_deleting(self):
        identity = dict(Arn=self.reg['iam']['instance_role']['arn'], RoleId='AROA-old')
        observed = dict(identity=identity, inventory=dict(self.reg['iam']['instance_role'], inline={}))
        fresh.mutate(Mock(return_value={'Role': identity}), self.state, SHA, 'create-instance_role',
                     'iam', 'create-role', [], lambda response: observed)
        changed = copy.deepcopy(observed)
        changed['identity']['RoleId'] = 'AROA-replaced'
        client = Mock()
        with patch.object(fresh, 'verify_compute_cleanup'), patch.object(fresh, 'role_snapshot', return_value=changed):
            result = fresh.partial_prerequisite_cleanup(self.reg, SHA, self.state, client)
        self.assertEqual(result['deleted_resources'], [])
        self.assertEqual(result['cleanup_state'], 'unresolved')
        client.assert_not_called()

    def test_journal_rejects_wrong_registration(self):
        fresh.record(self.state / 'create-intent.json',
            dict(registration_sha256='b' * 64, service='iam', operation='create-role', args=[]))
        with self.assertRaises(RuntimeError):
            fresh.journal_observation(self.state, 'create', SHA)

    def test_reconcile_delegates_exact_volume_and_schedule_cleanup(self):
        plan = {'identity': self.reg['identity'], 'cost': {'termination_grace_seconds': 300}}
        fresh.record(self.state / 'approval-consumed.json', dict(run_id=self.reg['identity']['run_id'], plan_sha256=SHA))
        evidence = {'instance_id': 'i-owned'}
        result = {'cleanup_state': 'complete'}
        with patch.object(fresh.lifecycle, 'load_volume_identity', return_value=evidence), \
             patch.object(fresh.lifecycle, 'cleanup_cutoff', return_value=10**20), \
             patch.object(fresh.lifecycle, 'reconcile', return_value=result) as reconcile, \
             patch.object(fresh.lifecycle, 'cleanup_schedule') as cleanup:
            self.assertEqual(fresh.reconcile(plan, SHA, self.state, Mock()), result)
        self.assertEqual(reconcile.call_args.args[2], 'i-owned')
        self.assertEqual(cleanup.call_args.kwargs, {'plan': plan, 'plan_sha': SHA})

class ScheduleGroupDeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / 'state'
        self.state.mkdir()
        self.reg = registration(self.root)
        ident = self.reg['identity']
        self.expected = dict(Name=ident['schedule_group'],
            Arn=f"arn:aws:scheduler:{ident['region']}:{ident['account']}:schedule-group/{ident['schedule_group']}",
            State='ACTIVE', CreationDate='2026-11-15T00:00:00Z',
            LastModificationDate='2026-11-15T00:00:00Z')
        self.live = dict(self.expected)
        self.deleted = False
        self.disappear = False
        self.deletes = 0
        self.calls = []
        self.now = 0

    def client(self, service, operation, *args):
        self.calls.append(operation)
        self.assertEqual(service, 'scheduler')
        if operation == 'list-tags-for-resource':
            return {'Tags': fresh.tags(self.reg['identity'])}
        self.assertEqual(args, ('--name', self.reg['identity']['schedule_group']))
        if operation == 'delete-schedule-group':
            self.deletes += 1
            self.deleted = True
            self.live['State'] = 'DELETING'
            self.live['LastModificationDate'] = '2026-11-15T00:01:00Z'
            return {}
        self.assertEqual(operation, 'get-schedule-group')
        if self.deleted and self.disappear:
            raise missing(service, operation, 'ResourceNotFoundException')
        return dict(self.live)

    def pause(self, seconds):
        self.now += seconds

    def delete(self, pause=None):
        return fresh.delete_schedule_group(self.client, self.reg, SHA, self.state, self.expected,
            timeout=2, clock=lambda: self.now, pause=pause or self.pause)

    def test_accepted_delete_waits_for_async_absence_with_one_mutation(self):
        def finish(seconds):
            self.pause(seconds)
            self.disappear = True
        result = self.delete(finish)
        self.assertTrue(result['absent'])
        self.assertEqual(self.deletes, 1)
        self.assertIn('DELETING', self.live.values())
        self.assertEqual(fresh.load(self.state / 'cleanup-schedule-group-readback.json')['observation'], result)
        self.assertEqual(self.calls[-2:], ['get-schedule-group', 'get-schedule-group'])

    def test_timeout_preserves_acceptance_and_resume_only_reads(self):
        with self.assertRaisesRegex(RuntimeError, 'still pending'):
            self.delete()
        self.assertEqual(self.now, 2)
        names = ['cleanup-schedule-group-intent.json', 'cleanup-schedule-group-response.json']
        original = {name: (self.state / name).read_bytes() for name in names}
        self.assertFalse((self.state / 'cleanup-schedule-group-readback.json').exists())
        self.disappear = True
        self.calls.clear()
        result = self.delete()
        self.assertTrue(result['absent'])
        self.assertEqual(self.deletes, 1)
        self.assertEqual(self.calls, ['get-schedule-group'])
        self.assertEqual({name: (self.state / name).read_bytes() for name in names}, original)
        # A subsequent readback is also read-only, never another delete.
        self.delete()
        self.assertEqual(self.deletes, 1)

    def test_recreated_same_name_group_blocks_accepted_delete_reconciliation(self):
        with self.assertRaises(RuntimeError):
            self.delete()
        self.live['CreationDate'] = '2026-11-15T01:00:00Z'
        self.live['State'] = 'ACTIVE'
        with self.assertRaisesRegex(RuntimeError, 'replaced/modified'):
            self.delete()
        self.assertEqual(self.deletes, 1)
        self.assertFalse((self.state / 'cleanup-schedule-group-readback.json').exists())

    def test_ambiguous_delete_without_acceptance_never_replays_or_reads(self):
        fresh.record(self.state / 'cleanup-schedule-group-intent.json',
            dict(registration_sha256=SHA, service='scheduler', operation='delete-schedule-group',
                 args=['--name', self.reg['identity']['schedule_group']]))
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous group deletion'):
            self.delete()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.deletes, 0)

    def test_foreign_acceptance_and_access_denial_never_prove_absence(self):
        with self.assertRaises(RuntimeError):
            self.delete()
        def denied(*args):
            raise missing('scheduler', 'get-schedule-group', 'AccessDeniedException')
        with self.assertRaisesRegex(RuntimeError, 'Group absence unresolved'):
            fresh.delete_schedule_group(denied, self.reg, SHA, self.state, self.expected)
        self.assertFalse((self.state / 'cleanup-schedule-group-readback.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous group deletion'):
            fresh.delete_schedule_group(self.client, self.reg, 'b' * 64, self.state, self.expected)
        self.assertEqual(self.deletes, 1)

    def owned_cleanup_journal(self):
        resources = dict(schedule_group=self.expected,
            instance_profile=dict(Arn='arn:owned-profile', InstanceProfileId='AIPA-owned', CreateDate='original'),
            security_group=dict(GroupId='sg-owned'))
        for key in fresh.ROLE_KEYS:
            resources[key] = dict(identity=dict(Arn=self.reg['iam'][key]['arn'],
                RoleId='AROA-' + key, CreateDate='original'))
        fresh.record(self.state / 'owned-prerequisites.json',
            dict(registration_sha256=SHA, identity=self.reg['identity'], resources=resources))
        fresh.record(self.state / 'owned-cleanup-started.json',
            dict(registration_sha256=SHA, resources=resources))
        for key, value in resources.items():
            if key in fresh.ROLE_KEYS:
                response = {'Role': value['identity']}
            elif key == 'instance_profile':
                response = {'InstanceProfile': value}
            elif key == 'schedule_group':
                response = {'ScheduleGroupArn': value['Arn']}
            else:
                response = {'GroupId': value['GroupId']}
            request = dict(registration_sha256=SHA, service='fixture', operation='create-' + key, args=[])
            fresh.record(self.state / ('create-' + key + '-intent.json'), request)
            fresh.record(self.state / ('create-' + key + '-response.json'), dict(request=request, response=response))
            fresh.record(self.state / ('create-' + key + '-readback.json'), dict(request=request, observation=value))

    def test_owned_cleanup_resumes_group_readback_but_compute_uncertainty_blocks_other_deletes(self):
        with self.assertRaisesRegex(RuntimeError, 'still pending'):
            self.delete()
        self.owned_cleanup_journal()
        self.disappear = True
        operations = []

        def transport(service, operation, *args, **kwargs):
            operations.append(operation)
            if operation == 'get-caller-identity':
                return dict(Account=self.reg['identity']['account'], Arn=self.reg['controller_arn'])
            if service == 'scheduler':
                return self.client(service, operation, *args)
            raise missing(service, operation, 'AccessDenied')

        with self.assertRaisesRegex(RuntimeError, 'AccessDenied'):
            fresh.owned_prerequisite_cleanup(self.reg, SHA, self.state, transport)
        self.assertEqual(operations, ['get-caller-identity', 'get-schedule-group', 'describe-instances'])
        self.assertEqual(self.deletes, 1)
        self.assertTrue((self.state / 'cleanup-schedule-group-readback.json').exists())
        self.assertFalse((self.state / 'cleanup-profile-detach-intent.json').exists())

    def test_owned_cleanup_never_replays_beyond_the_accepted_group_boundary(self):
        with self.assertRaisesRegex(RuntimeError, 'still pending'):
            self.delete()
        self.owned_cleanup_journal()
        fresh.record(self.state / 'cleanup-profile-detach-intent.json', {'ambiguous': True})
        calls = []

        def transport(service, operation, *args):
            calls.append(operation)
            self.assertEqual(operation, 'get-caller-identity')
            return dict(Account=self.reg['identity']['account'], Arn=self.reg['controller_arn'])

        with self.assertRaisesRegex(RuntimeError, 'explicit parent recovery'):
            fresh.owned_prerequisite_cleanup(self.reg, SHA, self.state, transport)
        self.assertEqual(calls, ['get-caller-identity'])
        self.assertEqual(self.deletes, 1)



if __name__ == '__main__':
    unittest.main()
