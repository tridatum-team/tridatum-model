"""Offline fresh600 lease regressions; fixtures are not live launch evidence."""
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import grounded_hf_lifecycle as L


def plan_fixture():
    run = 'fresh-eight-synthetic'
    identity = dict(run_id=run, account='148761639313', region='ap-northeast-2',
        schedule_group=run, schedule_name=run + '-deadline', bucket='fresh-synthetic', code_prefix=f'code/{run}/',
        output_prefix=f'output/{run}/')
    return dict(schema=L.FRESH_SCHEMA, identity=identity,
        limits=dict(L.LIMITS, lifetime_seconds=1800, reserve_seconds=300, output_bytes=512 * 1024**2),
        lifetime=dict(total_seconds=2700, termination_grace_seconds=900, automatic_extension=False),
        iam={'scheduler_role': {'arn': 'arn:aws:iam::148761639313:role/fresh-synthetic'}},
        runtime=dict(gpu_name='NVIDIA L40S', gpu_catalog_mib=45776,
            gpu_nvml_min_mib=45776, gpu_nvml_max_mib=49152, driver_version='synthetic'),
        bundle={'bytes': 1024}, network={'egress_byte_cap': 2 * L.GIB},
        cost=dict(cap_usd=4.11, retention_days=30, ancillary_bound_usd=0.25,
            get_requests=4608, put_requests=4608, control_requests=4608, download_requests=100,
            network_egress_bytes=3 * L.GIB, retained_bytes=L.GIB,
            export_bytes=512 * 1024**2, controller_retrieval_bytes=L.GIB,
            tax_rate=0.10, contingency_usd=1, termination_grace_seconds=900,
            scheduler_invocations=10000,
            rates=dict(ec2_hour=2, ipv4_hour=0.005, ebs_gb_month=0.1,
                s3_gb_month=0.025, egress_gb=0.1, s3_get=0.0000004,
                s3_put=0.000005, scheduler_invocation=0.000001)))


def release_fixture(plan):
    return dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
        instance_id='i-0123456789abcdef0', launch_time=1000, deadline=2800,
        schedule_verified=True,
        gpu=dict(uuid='GPU-11111111-2222-3333-4444-555555555555', minor=0,
            name='NVIDIA L40S', total_mib=46000, driver='synthetic'),
        schedule=L.schedule_spec(plan, 'i-0123456789abcdef0', 1000))


def evaluation_plan_fixture():
    plan = plan_fixture()
    plan.update(schema=L.EVALUATION_SCHEMA,
        limits=dict(L.LIMITS, lifetime_seconds=18900, reserve_seconds=300, output_bytes=512 * 1024**2),
        lifetime=dict(total_seconds=19800, termination_grace_seconds=900, automatic_extension=False))
    plan['cost'].update(cap_usd=20, retained_bytes=2 * L.GIB)
    return plan


class CudaEvaluationLifecycleTests(unittest.TestCase):
    def test_evaluation_uses_new_lease_and_never_training_success(self):
        plan = evaluation_plan_fixture()
        self.assertTrue(L.bounded_mode(plan))
        self.assertFalse(L.fresh_mode(plan))
        self.assertEqual(L.success_field(plan), 'evaluation_success')
        self.assertEqual(L.lifetime_seconds(plan), 18900)
        self.assertEqual(L.reserve_seconds(plan), 300)
        self.assertEqual(L.export_limit(plan), 512 * 1024**2)
        spec = L.schedule_spec(plan, 'i-evaluation', 1000)
        cutoff = dt.datetime.fromtimestamp(19840, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        self.assertEqual(spec['ScheduleExpression'], 'at(' + cutoff + ')')
        self.assertEqual(spec['Target']['RetryPolicy']['MaximumRetryAttempts'], 0)

    def test_new_grant_prices_all_five_and_half_hours(self):
        plan = evaluation_plan_fixture()
        original = L.cost_bound(plan)
        self.assertLessEqual(float(original['all_in_bound_usd']), 20)
        self.assertNotIn('aggregate_budget_usd', original)
        plan['cost']['rates']['ec2_hour'] += 0.1
        self.assertAlmostEqual(L.cost_bound(plan)['additional_pretax_bound_usd']
                               - original['additional_pretax_bound_usd'], 0.55)

    def test_old_grants_deadline_extensions_and_unpriced_egress_fail(self):
        for section, key, value in [('cost', 'cap_usd', 4.11), ('cost', 'cap_usd', 10),
                ('cost', 'termination_grace_seconds', 1800),
                ('cost', 'network_egress_bytes', 2 * L.GIB),
                ('network', 'egress_byte_cap', 3 * L.GIB),
                ('lifetime', 'total_seconds', 19801), ('limits', 'lifetime_seconds', 18901)]:
            with self.subTest(section=section, key=key, value=value):
                plan = evaluation_plan_fixture()
                plan[section][key] = value
                with self.assertRaises((RuntimeError, ValueError)):
                    L.cost_bound(plan)
        plan = evaluation_plan_fixture()
        plan['cost']['rates']['ec2_hour'] = 100
        with self.assertRaisesRegex(RuntimeError, 'new USD20'):
            L.cost_bound(plan)

    def test_evaluation_terminal_requires_exact_owned_reaped_phases(self):
        plan = evaluation_plan_fixture()
        stages = [dict(phase=phase, state='complete', unit=plan['identity']['run_id'] + '-' + phase,
            invocation='new-' + phase, gpu={'uuid': 'GPU-owned'}, run_id=plan['identity']['run_id'],
            plan_sha256='a' * 64, instance_id='i-owned', cleanup_verified=True,
            stopped_state={'ActiveState': 'inactive'}) for phase in ('runtime', 'evaluation')]
        completion = dict(evaluation_success=True, quality_accepted=False, stop_errors=[],
            completion_scope='cuda_evaluation_only_existing_final150', runtime_unit=stages[0], stages=stages[1:])
        L.validate_evaluation_terminal(plan, 'a' * 64, 'i-owned', completion)
        for key, value in [('instance_id', 'i-other'), ('cleanup_verified', False),
                           ('unit', 'other-evaluation'), ('phase', 'train')]:
            old = stages[1][key]
            stages[1][key] = value
            with self.assertRaises((RuntimeError, ValueError)):
                L.validate_evaluation_terminal(plan, 'a' * 64, 'i-owned', completion)
            stages[1][key] = old
        completion['quality_accepted'] = True
        with self.assertRaises((RuntimeError, ValueError)):
            L.validate_evaluation_terminal(plan, 'a' * 64, 'i-owned', completion)

    def test_evaluation_cleanup_cutoff_includes_grace_once(self):
        plan = evaluation_plan_fixture()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            L.save(state / 'approval-consumed.json', dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64))
            L.save(state / 'launch-dispatched.json', dict(run_id=plan['identity']['run_id'],
                plan_sha256='a' * 64, request_sha256='b' * 64, at=900))
            self.assertEqual(L.cleanup_cutoff(plan, 'a' * 64, state), 20700)

    def test_failed_evaluation_collection_stays_failed_and_is_one_shot(self):
        plan = evaluation_plan_fixture()
        completion = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                          instance_id='i-owned', evaluation_success=False, error='smoke failed')
        def client(service, operation, *args):
            self.assertEqual((service, operation), ('s3api', 'get-object'))
            Path(args[-1]).write_text(json.dumps(completion))
        with tempfile.TemporaryDirectory() as directory:
            result = L.collect_export(client, plan, 'a' * 64, directory, 'i-owned')
            self.assertFalse(result['native_pipeline_success'])
            self.assertEqual(result['semantic_acceptance'], 'not_evaluated')
            self.assertEqual(client.retrieval_byte_limit, L.GIB)
            with self.assertRaises(FileExistsError):
                L.collect_export(client, plan, 'a' * 64, directory, 'i-owned')

    def test_training_success_cannot_authorize_evaluation_collection(self):
        plan = evaluation_plan_fixture()
        def client(service, operation, *args):
            Path(args[-1]).write_text(json.dumps(dict(run_id=plan['identity']['run_id'],
                plan_sha256='a' * 64, instance_id='i-owned', training_success=True)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'explicit terminal failure'):
                L.collect_export(client, plan, 'a' * 64, directory, 'i-owned')

    def test_full_polling_and_export_ranges_fit_controller_reservation(self):
        plan = evaluation_plan_fixture()
        reserved = 120 * 1024**2 + 1024**2 + plan['cost']['export_bytes']
        self.assertEqual(reserved, 633 * 1024**2)
        self.assertLessEqual(reserved, plan['cost']['controller_retrieval_bytes'])
        # The paid5.5h bound never turns30s monitoring into unbounded requests.
        self.assertLessEqual((L.lifetime_seconds(plan) + 29) // 30, 630)
        plan['cost']['controller_retrieval_bytes'] = reserved - 1
        with self.assertRaises((RuntimeError, ValueError)):
            L.cost_bound(plan)

    def test_request_and_attempted_retrieval_caps_stop_before_another_call(self):
        from types import SimpleNamespace
        plan = evaluation_plan_fixture()
        client = L.Client(plan['identity'], request_limit=10000)
        client.requests = 10000
        with patch.object(L.subprocess, 'run') as subprocess_run:
            with self.assertRaisesRegex(RuntimeError, 'request envelope exhausted'):
                client('ec2', 'describe-instances')
            subprocess_run.assert_not_called()
        client = L.Client(plan['identity'], request_limit=10000)
        client.retrieval_byte_limit = L.GIB
        client.retrieval_bytes = L.GIB - 1024**2
        with patch.object(L.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stderr='ambiguous failed transfer', stdout='')) as subprocess_run:
            with self.assertRaises(L.AWSCallError):
                client('s3api', 'get-object', '--range', 'bytes=0-1048575', 'unused')
            self.assertEqual(client.retrieval_bytes, L.GIB)
            with self.assertRaisesRegex(RuntimeError, 'retrieval byte envelope exhausted'):
                client('s3api', 'get-object', '--range', 'bytes=0-0', 'unused')
            subprocess_run.assert_called_once()


class FreshLifetimeTests(unittest.TestCase):
    def test_total_includes_cleanup_and_keeps_legacy_unchanged(self):
        plan = plan_fixture()
        L.validate_fresh_lifetime(plan)
        self.assertEqual(L.lifetime_seconds(plan), 1800)
        self.assertEqual(L.reserve_seconds(plan), 300)
        self.assertEqual(L.lifetime_seconds({'schema': L.EIGHT_SCHEMA}), 14400)
        self.assertEqual(L.reserve_seconds({}), 1800)
        self.assertTrue(L.bounded_mode(plan))
        self.assertFalse(L.eight_mode(plan))

    def test_rejects_unsealed_or_extended_lifetime(self):
        mutations = [
            lambda p: p['lifetime'].update(total_seconds=2701),
            lambda p: p['lifetime'].update(automatic_extension=True),
            lambda p: p['lifetime'].update(termination_grace_seconds=300),
            lambda p: p['limits'].update(lifetime_seconds=1801),
            lambda p: p['limits'].update(reserve_seconds=301),
            lambda p: p['limits'].update(root_gib=100),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                plan = plan_fixture()
                mutate(plan)
                with self.assertRaises((ValueError, RuntimeError)):
                    L.validate_fresh_lifetime(plan)

    def test_schedule_uses_compute_cutoff_not_legacy_or_total(self):
        plan = plan_fixture()
        spec = L.schedule_spec(plan, 'i-owned', 1000)
        expected = dt.datetime.fromtimestamp(2740, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        self.assertEqual(spec['ScheduleExpression'], 'at(' + expected + ')')
        self.assertEqual(json.loads(spec['Target']['Input']), {'InstanceIds': ['i-owned']})
        self.assertEqual(spec['Target']['RetryPolicy']['MaximumRetryAttempts'], 0)
        self.assertEqual(spec['ActionAfterCompletion'], 'NONE')

    def test_release_and_ticket_reject_old_four_hour_deadline(self):
        plan = plan_fixture()
        release = release_fixture(plan)
        self.assertEqual(L.verify_termination_ticket(release, plan, 'a' * 64, release['instance_id'], 1100), 2800)
        self.assertEqual(L.verify_release(release, plan, 'a' * 64, release['instance_id'], 1100), 2500)
        release['deadline'] = 15400
        for check in (L.verify_release, L.verify_termination_ticket):
            with self.assertRaises((ValueError, RuntimeError)):
                check(release, plan, 'a' * 64, release['instance_id'], 1100)

    def test_cleanup_cutoff_uses_earlier_ambiguous_dispatch(self):
        plan = plan_fixture()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            L.save(state / 'approval-consumed.json', dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64))
            L.save(state / 'launch-dispatched.json', dict(run_id=plan['identity']['run_id'],
                plan_sha256='a' * 64, request_sha256='b' * 64, at=900))
            L.save(state / 'release.json', release_fixture(plan))
            self.assertEqual(L.cleanup_cutoff(plan, 'a' * 64, state), 3600)

    def test_remaining_budget_debits_consumed_prior_execution(self):
        plan = plan_fixture()
        result = L.cost_bound(plan)
        self.assertLessEqual(float(result['all_in_bound_usd']), 4.11)
        self.assertAlmostEqual(float(result['aggregate_all_in_bound_usd']),
                               5.89 + float(result['all_in_bound_usd']))
        self.assertFalse(result['spending_hard_cap'])
        plan['cost']['cap_usd'] = 10
        with self.assertRaises((ValueError, RuntimeError)):
            L.cost_bound(plan)

    def test_cost_prices_total_billable_lifetime(self):
        plan = plan_fixture()
        original = L.cost_bound(plan)['additional_pretax_bound_usd']
        plan['cost']['rates']['ec2_hour'] += 0.1
        self.assertAlmostEqual(L.cost_bound(plan)['additional_pretax_bound_usd'] - original, 0.075)

    def test_new_run_under_ten_but_over_remaining_budget_is_rejected(self):
        plan = plan_fixture()
        plan['cost']['rates']['ec2_hour'] = 4
        with self.assertRaisesRegex(RuntimeError, 'remaining USD4.11'):
            L.cost_bound(plan)

    def test_active_schema_and_status_do_not_alias_old_fresh_profile(self):
        self.assertEqual(L.success_field(plan_fixture()), 'training_success')
        self.assertFalse(L.fresh_mode({'schema': 'trid-fresh-eight-gpu-experiment-v1'}))
        self.assertEqual(L.success_field({'schema': L.EIGHT_SCHEMA}), 'experimental_success')

    def test_worker_policy_excludes_checkpoint_and_final_input_channels(self):
        plan = plan_fixture()
        identity = plan['identity']
        code = f"arn:aws:s3:::{identity['bucket']}/{identity['code_prefix']}"
        statements = [dict(Effect='Allow', Action=['s3:GetObject'], Resource=[code + name for name in
            ('plan.json', 'bundle.tar', 'approval-consumed.json', 'termination.json', 'release.json')]),
            dict(Effect='Allow', Action=['s3:PutObject'], Resource=[
                f"arn:aws:s3:::{identity['bucket']}/{identity['output_prefix']}*"]),
            dict(Effect='Allow', Action=['s3:ListBucket'], Resource=[f"arn:aws:s3:::{identity['bucket']}"],
                Condition={'StringEquals': {'s3:prefix': [identity['code_prefix'] + name
                    for name in ('termination.json', 'release.json')]}})]
        plan['iam']['instance_role'] = {'inline': {'fresh': {'Version': '2012-10-17', 'Statement': statements}}}
        L.validate_worker_policy(plan)
        statements[0]['Resource'].append(code + 'final-inputs.json')
        with self.assertRaises((ValueError, RuntimeError)):
            L.validate_worker_policy(plan)

    def test_missing_schedule_proof_blocks_cleanup(self):
        plan = plan_fixture()
        cleanup = dict(cleanup_state='complete', instance_terminated=True,
            instance_id='i-owned', root_volume_ids=['vol-owned'], remaining_volumes=[],
            recorded_volume_ids=['vol-owned'], absent_volume_ids=['vol-owned'])
        with tempfile.TemporaryDirectory() as directory:
            def forbidden(*args, **kwargs):
                raise AssertionError('No cloud call permitted without schedule provenance')
            with self.assertRaises((ValueError, RuntimeError)):
                L.cleanup_schedule(forbidden, plan['identity'], directory, cleanup, plan=plan, plan_sha='a' * 64)

    def test_retention_requires_exact_prefix_and_multipart_abort(self):
        plan = plan_fixture()
        rules = [dict(ID='historical', Status='Enabled', Filter={'Prefix': 'old/'},
                      Expiration={'Days': 30})]
        rules.extend(dict(ID=name, Status='Enabled', Filter={'Prefix': plan['identity'][name]},
            Expiration={'Days': 30}, AbortIncompleteMultipartUpload={'DaysAfterInitiation': 1})
            for name in ('code_prefix', 'output_prefix'))
        plan['cost']['bucket_lifecycle_configuration'] = {'Rules': rules}
        L.validate_lifecycle_configuration(plan, {'Rules': rules})
        del rules[-1]['AbortIncompleteMultipartUpload']
        with self.assertRaises(RuntimeError):
            L.validate_lifecycle_configuration(plan, {'Rules': rules})

    def test_ambiguous_schedule_deletion_is_not_repeated(self):
        plan = plan_fixture()
        identity = plan['identity']
        spec = L.schedule_spec(plan, 'i-owned', 1000)
        arn = (f"arn:aws:scheduler:{identity['region']}:{identity['account']}:schedule/"
               f"{identity['schedule_group']}/{identity['schedule_name']}")
        snapshot = dict(spec, Arn=arn, CreationDate='2026-11-15T00:00:00Z',
                        LastModificationDate='2026-11-15T00:00:00Z')
        cleanup = dict(cleanup_state='complete', instance_terminated=True,
            instance_id='i-owned', root_volume_ids=['vol-owned'], remaining_volumes=[],
            recorded_volume_ids=['vol-owned'], absent_volume_ids=['vol-owned'])
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            L.save(state / 'approval-consumed.json', dict(run_id=identity['run_id'], plan_sha256='a' * 64))
            L.save(state / 'schedule-intent.json', spec)
            L.save(state / 'schedule-created.json', {'ScheduleArn': arn})
            L.save(state / 'schedule-readback.json', snapshot)
            L.save(state / 'schedule-delete-intent.json', {'ambiguous': True})
            def forbidden(*args, **kwargs):
                raise AssertionError('Ambiguous deletion must not call AWS')
            with self.assertRaisesRegex(RuntimeError, 'Ambiguous prior schedule deletion'):
                L.cleanup_schedule(forbidden, identity, state, cleanup, plan=plan, plan_sha='a' * 64)

    def test_all_in_cost_cannot_omit_tax_or_contingency(self):
        for key, value in (('tax_rate', 0), ('contingency_usd', 0),
                           ('retained_bytes', 2 * L.GIB), ('network_egress_bytes', 2 * L.GIB),
                           ('export_bytes', L.GIB), ('put_requests', 10001)):
            with self.subTest(key=key):
                plan = plan_fixture()
                plan['cost'][key] = value
                with self.assertRaises(RuntimeError):
                    L.cost_bound(plan)

    def test_controller_retrieval_limit_blocks_before_cloud_call(self):
        client = L.Client(plan_fixture()['identity'])
        client.retrieval_byte_limit = 1024
        with self.assertRaisesRegex(RuntimeError, 'retrieval byte envelope'):
            client('s3api', 'get-object', '--range', 'bytes=0-1024', 'never-created')
        self.assertEqual(client.retrieval_bytes, 0)

    def test_terminal_requires_two_owned_clean_training_stages(self):
        plan = plan_fixture()
        completion = dict(training_success=True, stop_errors=[],
            completion_scope='cuda_capacity_train_export_only', quality_accepted=False, cloud_evaluation=False, stages=[
            dict(phase=phase, state='complete', cleanup_verified=True,
                run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                instance_id='i-owned', unit=phase, invocation='synthetic',
                gpu=release_fixture(plan)['gpu'])
            for phase in ('capacity', 'train')])
        L.validate_fresh_terminal(plan, 'a' * 64, 'i-owned', completion)
        for key, value in (('quality_accepted', True), ('cloud_evaluation', True),
                           ('completion_scope', 'paired_evaluation')):
            with self.subTest(key=key):
                altered = dict(completion, **{key: value})
                with self.assertRaises(RuntimeError):
                    L.validate_fresh_terminal(plan, 'a' * 64, 'i-owned', altered)
        with self.assertRaises(RuntimeError):
            L.validate_fresh_terminal(plan, 'a' * 64, 'i-owned',
                dict(completion, stages=completion['stages'] + [dict(phase='evaluate')]))
        completion['export_error'] = 'synthetic failure'
        with self.assertRaises(RuntimeError):
            L.validate_fresh_terminal(plan, 'a' * 64, 'i-owned', completion)
        del completion['export_error']
        completion['stages'][-1]['cleanup_verified'] = False
        with self.assertRaises(RuntimeError):
            L.validate_fresh_terminal(plan, 'a' * 64, 'i-owned', completion)
        completion['stages'].pop()
        with self.assertRaises(RuntimeError):
            L.validate_fresh_terminal(plan, 'a' * 64, 'i-owned', completion)


class FreshExportFailureTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_fixture()
        self.plan['output_allowlist'] = ['train-receipt.json']
        self.completion = dict(run_id=self.plan['identity']['run_id'], plan_sha256='a' * 64,
            instance_id='i-owned', training_success=False, stop_errors=[],
            stages=[dict(phase='train', state='complete', cleanup_verified=True)],
            error='RuntimeError: Absolute deadline exhausted',
            export_error="Unregistered output member: ['tmp/generated.py']")

    def archive(self, *, extra=False, bad_member=False, foreign_status=False):
        # A real tar with authenticated receipt bytes, not a mocked validator.
        content = b'{"phase":"train","state":"complete"}'
        status = {k: v for k, v in self.completion.items() if k not in ('error', 'export_error', 'export')}
        if foreign_status:
            status['instance_id'] = 'i-foreign'
        manifest = dict(run_id=self.completion['run_id'], plan_sha256=self.completion['plan_sha256'],
            status=status, files=[dict(path='train-receipt.json', bytes=len(content),
                sha256=('b' * 64 if bad_member else hashlib.sha256(content).hexdigest()))])
        rows = {'train-receipt.json': content, 'export-manifest.json': json.dumps(manifest).encode()}
        if extra:
            rows['undeclared.bin'] = b'not in authenticated inventory'
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:') as stream:
            for name, raw in rows.items():
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                stream.addfile(member, io.BytesIO(raw))
        raw = output.getvalue()
        self.completion['export'] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw

    def collect(self, directory, archive=None):
        completion = self.completion
        prefix = self.plan['identity']['output_prefix']
        calls = []

        class Transport:
            def __call__(self, service, operation, *args):
                if service != 's3api' or operation not in ('head-object', 'get-object'):
                    raise AssertionError('Unexpected cloud operation')
                key = args[args.index('--key') + 1]
                calls.append(key)
                if key == prefix + 'completion.json' and operation == 'get-object':
                    Path(args[-1]).write_text(json.dumps(completion))
                    return {}
                if key != prefix + 'export.tar' or archive is None:
                    raise AssertionError('Unexpected archive retrieval after metadata-only failure')
                metadata = dict(ContentLength=len(archive), ETag='"synthetic-export"')
                if operation == 'head-object':
                    return metadata
                start, end = map(int, args[args.index('--range') + 1].removeprefix('bytes=').split('-'))
                if args[args.index('--if-match') + 1] != metadata['ETag']:
                    raise AssertionError('Missing immutable export condition')
                part = archive[start:end + 1]
                Path(args[-1]).write_bytes(part)
                return dict(metadata, ContentLength=len(part),
                            ContentRange=f'bytes {start}-{end}/{len(archive)}')

        self.calls = calls
        return L.collect_export(Transport(), self.plan, 'a' * 64, directory, 'i-owned')

    def test_missing_archive_preserves_worker_export_and_cleanup_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.collect(directory)
        self.assertFalse(result['native_pipeline_success'])
        self.assertEqual(result['semantic_acceptance'], 'not_evaluated')
        self.assertEqual(result['evidence_kind'], 'metadata_only')
        self.assertEqual(result['export_state'], 'missing')
        self.assertEqual(result['completion'], self.completion)
        self.assertEqual(result['worker_error'], self.completion['error'])
        self.assertEqual(result['export_error'], self.completion['export_error'])
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('export_sha256', result)

    def test_missing_archive_never_bypasses_completion_identity(self):
        for key in ('run_id', 'plan_sha256', 'instance_id'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                original = self.completion[key]
                self.completion[key] = 'foreign'
                with self.assertRaisesRegex(RuntimeError, 'Completion ownership'):
                    self.collect(directory)
                self.completion[key] = original
                self.assertEqual(len(self.calls), 1)

    def test_success_or_nonterminal_metadata_without_archive_is_protocol_error(self):
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as directory:
                self.completion['training_success'] = success
                self.completion.pop('error', None)
                self.completion.pop('export_error', None)
                with self.assertRaisesRegex(RuntimeError, 'explicit terminal failure'):
                    self.collect(directory)
                self.assertEqual(len(self.calls), 1)

    def test_failed_terminal_recovers_authenticated_earlier_train_snapshot(self):
        archive = self.archive()
        with tempfile.TemporaryDirectory() as directory:
            result = self.collect(directory, archive)
        self.assertFalse(result['native_pipeline_success'])
        self.assertEqual(result['export_state'], 'authenticated_partial')
        self.assertEqual(result['completion']['error'], self.completion['error'])
        self.assertEqual(result['export_sha256'], hashlib.sha256(archive).hexdigest())
        self.assertEqual(len(self.calls), 3)

    def test_failed_terminal_still_rejects_corrupt_archive_bytes(self):
        archive = self.archive()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Export bytes differ'):
                self.collect(directory, archive[:-1] + b'x')

    def test_failed_terminal_still_authenticates_exact_members_and_status_identity(self):
        for option in ('extra', 'bad_member', 'foreign_status'):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                archive = self.archive(**{option: True})
                with self.assertRaises(RuntimeError):
                    self.collect(directory, archive)

    def test_present_malformed_export_is_not_metadata_only_fallback(self):
        self.completion['export'] = None
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Export envelope'):
                self.collect(directory)
        self.assertEqual(len(self.calls), 1)

    def test_training_only_export_authenticates_without_cuda_answers(self):
        plan = self.plan
        self.completion = dict(run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
            instance_id='i-owned', training_success=True, stop_errors=[],
            completion_scope='cuda_capacity_train_export_only', quality_accepted=False, cloud_evaluation=False,
            runtime_unit=dict(gpu=release_fixture(plan)['gpu']),
            stages=[dict(phase=phase, state='complete', cleanup_verified=True,
                run_id=plan['identity']['run_id'], plan_sha256='a' * 64,
                instance_id='i-owned', unit=phase, invocation='synthetic', gpu=release_fixture(plan)['gpu'])
                for phase in ('capacity', 'train')])
        archive = self.archive()
        # Full training receipt semantics are covered by the experiment tests;
        # this regression exercises the real transport, tar and terminal gates.
        with tempfile.TemporaryDirectory() as directory, patch.object(
                L.fresh_module(plan), 'validate_completion') as validate_training:
            release = dict(release_fixture(plan), instance_id='i-owned')
            L.save(Path(directory) / 'release.json', release)
            result = self.collect(directory, archive)
            validate_training.assert_called_once_with(Path(directory) / 'fresh-export', plan)
        self.assertTrue(result['native_pipeline_success'])
        self.assertEqual(result['semantic_acceptance'], 'not_evaluated')
        self.assertFalse(result['completion']['quality_accepted'])
        self.assertEqual(result['export_state'], 'authenticated_terminal')


if __name__ == '__main__':
    unittest.main()
