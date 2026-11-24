"""Offline only: no subprocesses, credentials, AWS requests or paid resources."""
import copy
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aws_eval import fresh_eight_launch as fresh
from scripts.aws_eval import public_work_operator_inputs as operator


SHA = 'a' * 64

# Synthetic unit-test inputs, not operator approvals. Production trust anchors
# deliberately reject these bytes; tests substitute only their expected digests.
TRAINING_AUTHORIZATION = dict(
    synthetic_unit_test=True, schema='trid-eight-training-recovery-authorization-v2',
    aggregate_budget_usd='10.00', additional_gpu_executions=1,
    previous_authorization_consumed=True, execution_started=False,
    automatic_retry=False, automatic_extension=False,
    previous_run_id='mois-fresh-eight-gpu-20261115-094150')
EVALUATION_AUTHORIZATION = dict(
    synthetic_unit_test=True, schema='trid-cuda-evaluation-authorization-v1',
    run_id='mois-eight-cuda-eval-20261116-134815', cap_usd='20',
    additional_to_prior_spending=True, maximum_instance_executions=1,
    maximum_total_lifetime_seconds_including_shutdown_grace=19800,
    work_lease_seconds=18900, shutdown_grace_seconds=900, evaluation_only=True,
    retraining_allowed=False, automatic_instance_retry_allowed=False,
    prior_training_authorization_reused=False)
SAMPLING_AUTHORIZATION = dict(
    synthetic_unit_test=True, schema='trid-overnight-gpu-authorization-v1',
    maximum_concurrent_instances=1, instance_type='g6e.2xlarge', gpu='NVIDIA L40S',
    fixed_dollar_cap_explicitly_selected_by_user=False,
    conservative_operational_deadline=fresh.lifecycle.SAMPLING_DEADLINE)


def patch_authorization_digest(test, validator, original_sha, fixture_sha):
    """Replace one literal trust anchor, retaining every real validation check."""
    code = validator.__code__
    test.assertEqual(code.co_consts.count(original_sha), 1)
    replacement = code.replace(co_consts=tuple(
        fixture_sha if value == original_sha else value for value in code.co_consts))
    # Historical digests are literals, not patchable module constants. Clone
    # only the validator so its original code stays intact through test cleanup.
    synthetic_validator = FunctionType(replacement, validator.__globals__,
        validator.__name__, validator.__defaults__, validator.__closure__)
    synthetic_validator.__kwdefaults__ = validator.__kwdefaults__
    patcher = patch.object(fresh.lifecycle, validator.__name__, synthetic_validator)
    patcher.start()
    test.addCleanup(patcher.stop)


def pin(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content if isinstance(content, bytes) else json.dumps(content).encode()
    path.write_bytes(raw)
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class PublicWorkOperatorSafetyTests(unittest.TestCase):
    """Fresh synthetic receipts only; no published run artifacts or AWS process."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / 'fresh'
        self.reference = self.root / 'reference'
        self.run.mkdir()
        self.reference.mkdir()
        self.old_run = 'public-work-historical'
        self.new_run = 'public-work-fresh'
        self.identity = dict(run_id=self.new_run, region='ap-northeast-2',
                             account='123456789012')
        self.started = '2030-01-01T00:00:00Z'
        self.completed = '2030-01-01T00:00:01Z'
        self.clock = datetime(2030, 1, 1, 0, 5, tzinfo=timezone.utc)
        pin(self.root, 'reference/plan.json',
            {'approved': True, 'authorization': 'historical-grant'})
        pin(self.root, 'fresh/projection.json', {'cloud_authorized': False})
        self.args = SimpleNamespace(
            allow_read_only_aws=True, profile='fresh-operator',
            reference_run_dir=self.reference, run_dir=self.run,
            projection=self.run / 'projection.json', run_id=self.new_run,
            observations_sha256=None, error_review=None,
            approve_fresh_one_run=False,
            approve_ancillary_retention_network_cost_review=False,
            absolute_deadline_utc='2030-01-01T08:00:00Z')
        self.queries = {
            'instances': ['aws', 'ec2', 'describe-instances', '--filters',
                'Name=tag:trid:run,Values=' + self.old_run,
                'Name=instance-state-name,Values=pending,running,stopping,stopped',
                '--region', 'ap-northeast-2', '--profile', 'historical-profile',
                '--output', 'json'],
            'new_security_group_absent': ['aws', 'ec2', 'describe-security-groups',
                '--filters', 'Name=group-name,Values=' + self.old_run + '-sg',
                'Name=vpc-id,Values=vpc-synthetic', '--region', 'ap-northeast-2',
                '--output', 'json'],
            'code_prefix_empty': ['aws', 's3api', 'list-objects-v2',
                '--bucket', 'synthetic-bucket', '--prefix', 'code/' + self.old_run + '/',
                '--region', 'ap-northeast-2', '--output', 'json'],
            'instance_role_absent': ['aws', 'iam', 'get-role',
                '--role-name', self.old_run + '-worker', '--output', 'json'],
        }
        historical = {}
        for name, query in self.queries.items():
            historical[name] = pin(self.root, 'reference/' + name + '.json',
                dict(query=query, started_at='2020-01-01T00:00:00Z',
                     completed_at='2020-01-01T00:00:01Z', exit_code=0,
                     response={'approved': True, 'authorization': 'historical-grant'}))
        self.ctx = (self.root, self.run, {'approved': True}, {'approved': True},
                    self.identity, {self.old_run: self.new_run},
                    {'cloud_authorized': False}, {'observations': historical})

    def collect(self, runner=None):
        runner = runner or Mock(return_value=SimpleNamespace(
            returncode=0, stdout='{"current": true}', stderr=''))
        count = len(self.queries) + 1
        with patch.object(operator.subprocess, 'run', runner), \
                patch.object(operator, 'now', side_effect=[self.started, self.completed] * count), \
                patch('builtins.print'):
            operator.observe(self.args, self.ctx)
        self.args.observations_sha256 = operator.pin(
            self.root, self.run / 'operator-observations.json')['sha256']
        return runner, fresh.load(self.run / 'operator-observations.json')

    def load_observations(self):
        with patch.object(operator, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = self.clock
            return operator.observations(self.args, self.root, self.run, self.identity)

    def rewrite_receipt(self, name, **changes):
        ledger = fresh.load(self.run / 'operator-observations.json')
        relative = ledger['observations'][name]['path']
        receipt = fresh.load(self.root / relative)
        receipt.update(changes)
        ledger['observations'][name] = pin(self.root, relative, receipt)
        self.args.observations_sha256 = pin(
            self.root, 'fresh/operator-observations.json', ledger)['sha256']

    def test_readonly_rejects_mutation_shell_and_indirect_inputs(self):
        forbidden = (
            ['aws', 'ec2', 'run-instances'],
            ['aws', 'ec2', 'terminate-instances'],
            ['aws', 'iam', 'create-role'],
            ['aws', 's3api', 'put-object'],
            ['aws', 'scheduler', 'create-schedule'],
            ['aws', 'sts', 'get-session-token'],
            ['aws', 'sts', 'get-federation-token'],
            ['aws', 'ecr', 'get-authorization-token'],
            ['aws', 'ecr', 'get-login-password'],
            ['aws', 'sso', 'get-role-credentials'],
            ['aws', 'redshift-serverless', 'get-credentials'],
            ['sh', '-c', 'aws ec2 describe-instances'],
            ['aws', 'ec2', 'describe-instances', '--cli-input-json', 'file://request.json'],
            ['aws', 'ec2', 'describe-instances', '--cli-input-json', 'fileb://request.json'],
            ['aws', 'ec2', 'describe-instances', '--endpoint-url', 'https://example.invalid'],
            ['aws', 'ec2', 'describe-instances', '--no-sign-request'],
            ['aws', 'ec2', 'describe-instances', '--endpoint-url=https://example.invalid'],
            ['aws', 'ec2', 'describe-instances', '--no-sign-request=true'],
            ['aws', 'ec2', 'describe-instances', '--cli-input-json=file://request.json'],
            ['aws', 'ec2', 'describe-instances', '--cli-input-json=fileb://request.json'],
        )
        with patch.object(operator.subprocess, 'run') as runner:
            for query in forbidden:
                with self.subTest(query=query), self.assertRaises(RuntimeError):
                    operator.readonly(query)
            runner.assert_not_called()

    def test_inspector_batch_read_exception_is_exact_and_never_allows_batch_mutations(self):
        query = ['aws', 'inspector2', 'batch-get-account-status',
                 '--account-ids', '123456789012', '--region', 'ap-northeast-2',
                 '--profile', 'fresh-operator', '--output', 'json']
        with patch.object(operator.subprocess, 'run') as runner:
            self.assertIs(operator.readonly(query), query)
            for service, operation in (
                    ('inspector', 'batch-get-account-status'),
                    ('ec2', 'batch-get-account-status'),
                    ('inspector2', 'batch-update-member-ec2-deep-inspection-status'),
                    ('inspector2', 'batch-get-account-status-extra'),
                    ('dynamodb', 'batch-write-item'),
                    ('codebuild', 'batch-delete-builds'),
                    ('glue', 'batch-delete-table')):
                with self.subTest(service=service, operation=operation):
                    with self.assertRaises(RuntimeError):
                        operator.readonly(['aws', service, operation])
            runner.assert_not_called()

    def test_exact_critical_filters_and_transport_are_preserved(self):
        def run(query, **kwargs):
            inventory = fresh.load(self.run / 'readonly-calls.json')
            self.assertIn(query, inventory['calls'].values())
            self.assertFalse(inventory['cloud_mutations'])
            self.assertEqual(kwargs['env']['AWS_PROFILE'], 'fresh-operator')
            self.assertEqual(kwargs['env']['AWS_MAX_ATTEMPTS'], '1')
            self.assertEqual(kwargs['env']['AWS_PAGER'], '')
            self.assertEqual(kwargs['env']['AWS_IGNORE_CONFIGURED_ENDPOINT_URLS'], 'true')
            self.assertEqual({key: value for key, value in kwargs.items() if key != 'env'},
                             dict(capture_output=True, text=True, timeout=180, check=False))
            return SimpleNamespace(returncode=0, stdout='{"current": true}', stderr='')

        runner, ledger = self.collect(Mock(side_effect=run))
        expected = [
            ['aws', 'ec2', 'describe-instances', '--filters',
             'Name=tag:trid:run,Values=public-work-fresh',
             'Name=instance-state-name,Values=pending,running,stopping,stopped',
             '--region', 'ap-northeast-2', '--profile', 'fresh-operator', '--output', 'json'],
            ['aws', 'ec2', 'describe-security-groups', '--filters',
             'Name=group-name,Values=public-work-fresh-sg', 'Name=vpc-id,Values=vpc-synthetic',
             '--region', 'ap-northeast-2', '--output', 'json', '--profile', 'fresh-operator'],
            ['aws', 's3api', 'list-objects-v2', '--bucket', 'synthetic-bucket',
             '--prefix', 'code/public-work-fresh/', '--region', 'ap-northeast-2',
             '--output', 'json', '--profile', 'fresh-operator'],
            ['aws', 'iam', 'get-role', '--role-name', 'public-work-fresh-worker',
             '--output', 'json', '--profile', 'fresh-operator'],
            ['aws', 'sts', 'get-caller-identity', '--profile', 'fresh-operator',
             '--region', 'ap-northeast-2', '--output', 'json', '--no-cli-pager',
             '--cli-connect-timeout', '5', '--cli-read-timeout', '30'],
        ]
        self.assertEqual([call.args[0] for call in runner.call_args_list], expected)
        self.assertEqual(len(ledger['observations']), len(expected))
        self.assertFalse(ledger['old_observations_reused'])
        self.assertFalse(ledger['cloud_mutations'])
        for receipt_pin in ledger['observations'].values():
            receipt = fresh.bound_json(self.root, receipt_pin)
            self.assertEqual(receipt['started_at'], self.started)
            self.assertEqual(receipt['completed_at'], self.completed)
            self.assertEqual(receipt['response'], {'current': True})
            self.assertEqual(receipt['exit_code'], 0)
            self.assertIsNone(receipt['error_code'])
        for name in ('authorization.json', 'approval.json', 'plan.json', 'registration.json'):
            self.assertFalse((self.run / name).exists())

    def test_permission_and_entire_inventory_are_checked_before_any_call(self):
        with patch.object(operator.subprocess, 'run') as runner:
            self.args.allow_read_only_aws = False
            with self.assertRaisesRegex(RuntimeError, 'allow-read-only-aws'):
                operator.observe(self.args, self.ctx)
            self.assertFalse((self.run / 'observations').exists())
            self.args.allow_read_only_aws = True
            self.ctx[-1]['observations']['unsafe_last'] = pin(
                self.root, 'reference/unsafe.json', {'query': ['aws', 'iam', 'delete-role']})
            with self.assertRaisesRegex(RuntimeError, 'read-only'):
                operator.observe(self.args, self.ctx)
            runner.assert_not_called()
        self.assertFalse((self.run / 'operator-observations.json').exists())

    def test_observation_replay_never_calls_aws_or_changes_receipts(self):
        self.collect()
        before = (self.run / 'operator-observations.json').read_bytes()
        with patch.object(operator.subprocess, 'run') as runner:
            with self.assertRaisesRegex(RuntimeError, 'never replay'):
                operator.observe(self.args, self.ctx)
            runner.assert_not_called()
        self.assertEqual((self.run / 'operator-observations.json').read_bytes(), before)

    def test_successful_blank_bucket_acceleration_response_is_an_empty_object(self):
        query = ['aws', 's3api', 'get-bucket-accelerate-configuration',
                 '--bucket', 'synthetic-bucket', '--region', 'ap-northeast-2',
                 '--output', 'json']
        self.queries['bucket_acceleration'] = query
        self.ctx[-1]['observations']['bucket_acceleration'] = pin(
            self.root, 'reference/bucket-acceleration.json', {'query': query})

        def run(argv, **kwargs):
            stdout = '' if argv[1:3] == ['s3api', 'get-bucket-accelerate-configuration'] else '{}'
            return SimpleNamespace(returncode=0, stdout=stdout, stderr='')

        runner, ledger = self.collect(Mock(side_effect=run))
        receipt = fresh.bound_json(self.root, ledger['observations']['bucket_acceleration'])
        self.assertEqual(receipt['stdout'], '')
        self.assertEqual(receipt['response'], {})
        self.assertEqual(receipt['exit_code'], 0)
        self.assertIsNone(receipt['error_code'])
        self.assertEqual(runner.call_count, len(self.queries) + 1)
        _, _, rows, _, _ = self.load_observations()
        self.assertEqual(rows['bucket_acceleration']['response'], {})

    def test_timeout_and_invalid_json_never_publish_complete_observation_ledger(self):
        for failure in (subprocess.TimeoutExpired(['aws'], 180), None):
            with self.subTest(failure=type(failure).__name__):
                self.run = self.root / ('timeout' if failure else 'invalid-json')
                self.run.mkdir()
                ctx = (self.root, self.run, *self.ctx[2:])
                runner = Mock(side_effect=failure) if failure else Mock(
                    return_value=SimpleNamespace(returncode=0, stdout='not JSON', stderr=''))
                with patch.object(operator.subprocess, 'run', runner), \
                        patch.object(operator, 'now', return_value=self.started):
                    with self.assertRaises(subprocess.TimeoutExpired if failure else json.JSONDecodeError):
                        operator.observe(self.args, ctx)
                self.assertEqual(runner.call_count, 1)
                self.assertFalse((self.run / 'operator-observations.json').exists())
                self.assertFalse((self.run / 'authorization.json').exists())

    def test_paginated_response_is_retained_but_never_completes_collection(self):
        response = {'Reservations': [], 'NextToken': 'unread-page'}
        runner = Mock(return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(response), stderr=''))
        with patch.object(operator.subprocess, 'run', runner), \
                patch.object(operator, 'now', side_effect=[self.started, self.completed]):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete AWS observation'):
                operator.observe(self.args, self.ctx)
        self.assertEqual(runner.call_count, 1)
        receipt = fresh.load(self.run / 'observations' / 'instances.json')
        self.assertEqual(receipt['response'], response)
        self.assertEqual(receipt['exit_code'], 0)
        self.assertFalse((self.run / 'operator-observations.json').exists())
        self.assertFalse((self.run / 'authorization.json').exists())

    def test_reviewed_receipts_cannot_hide_incomplete_pagination(self):
        self.collect()
        for key, value in (
                ('NextToken', 'next'), ('next_token', 'next'), ('NextMarker', 'next'),
                ('NextContinuationToken', 'next'), ('nextPageToken', 'next'),
                ('IsTruncated', True)):
            with self.subTest(key=key):
                self.rewrite_receipt('instances', response={key: value})
                with self.assertRaisesRegex(RuntimeError, 'Incomplete AWS observation'):
                    self.load_observations()
        self.rewrite_receipt('instances', response={'IsTruncated': False, 'NextToken': None})
        self.load_observations()

    def test_successful_receipts_require_json_objects_without_error_codes(self):
        self.collect()
        for response in (None, [], 'not an object'):
            with self.subTest(response=response):
                self.rewrite_receipt('instances', response=response)
                with self.assertRaisesRegex(RuntimeError, 'JSON object'):
                    self.load_observations()
        self.rewrite_receipt('instances', response={}, error_code='AccessDenied')
        with self.assertRaisesRegex(RuntimeError, 'Successful observation contains an error'):
            self.load_observations()

    def test_error_receipts_preserve_denial_and_never_treat_it_as_absence(self):
        def run(query, **kwargs):
            if query[1:3] == ['iam', 'get-role']:
                return SimpleNamespace(returncode=254, stdout='', stderr=(
                    'An error occurred (AccessDenied) when calling the GetRole operation: denied'))
            return SimpleNamespace(returncode=0, stdout='{}', stderr='')
        runner, ledger = self.collect(Mock(side_effect=run))
        receipt = fresh.bound_json(self.root, ledger['observations']['instance_role_absent'])
        self.assertEqual(receipt['error_code'], 'AccessDenied')
        self.assertEqual(receipt['exit_code'], 254)
        self.assertIsNone(receipt['response'])
        self.assertEqual(runner.call_count, len(self.queries) + 1)
        with self.assertRaisesRegex(RuntimeError, 'Unresolved read-only observation failure'):
            self.load_observations()
        self.rewrite_receipt('instance_role_absent', error_code='NoSuchEntity')
        with self.assertRaisesRegex(RuntimeError, 'does not match actual stderr'):
            self.load_observations()

    def test_recognized_absence_requires_exact_service_error(self):
        self.collect()
        self.rewrite_receipt('instance_role_absent', exit_code=254, response=None,
            error_code='NoSuchEntity', stderr=(
                'An error occurred (NoSuchEntity) when calling the GetRole operation: absent'))
        _, _, rows, earliest, latest = self.load_observations()
        self.assertEqual(rows['instance_role_absent']['error_code'], 'NoSuchEntity')
        self.assertEqual((earliest, latest), (self.started, self.completed))

    def test_reviewed_digest_projection_and_original_time_window_are_required(self):
        self.collect()
        actual_sha = self.args.observations_sha256
        self.args.observations_sha256 = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'reviewed observation SHA'):
            self.load_observations()
        self.args.observations_sha256 = actual_sha
        original_clock = self.clock
        self.clock = datetime(2030, 1, 1, 1, tzinfo=timezone.utc)
        with self.assertRaisesRegex(RuntimeError, 'window expired'):
            self.load_observations()
        self.clock = datetime(2029, 12, 31, 23, 59, tzinfo=timezone.utc)
        with self.assertRaisesRegex(RuntimeError, 'Invalid observation timing'):
            self.load_observations()
        self.clock = original_clock
        self.rewrite_receipt('instances', completed_at='2029-12-31T23:59:59Z')
        with self.assertRaisesRegex(RuntimeError, 'Invalid observation timing'):
            self.load_observations()
        self.rewrite_receipt('instances', completed_at='2030-01-01T00:06:00Z')
        with self.assertRaisesRegex(RuntimeError, 'Invalid observation timing'):
            self.load_observations()
        self.rewrite_receipt('instances', completed_at=self.completed)
        pin(self.root, 'fresh/projection.json', {'cloud_authorized': False, 'changed': True})
        with self.assertRaisesRegex(RuntimeError, 'run/projection differs'):
            self.load_observations()

    def test_historical_run_identity_is_rejected_before_projection_or_cloud_access(self):
        pin(self.root, 'reference/registration.json',
            dict(schema=fresh.PUBLIC_WORK_SCHEMA, identity={'run_id': self.old_run}))
        pin(self.root, 'reference/plan.json', dict(schema=operator.public.SCHEMA, approved=True))
        self.args.run_id = self.old_run
        with patch.object(operator, '__file__', str(
                self.root / 'scripts' / 'aws_eval' / 'public_work_operator_inputs.py')), \
                patch.object(operator.subprocess, 'run') as runner:
            with self.assertRaisesRegex(RuntimeError, 'Historical run authorization is not reusable'):
                operator.context(self.args)
            runner.assert_not_called()

    def test_historical_observation_marker_cannot_be_reviewed_into_freshness(self):
        _, ledger = self.collect()
        ledger['old_observations_reused'] = True
        self.args.observations_sha256 = pin(
            self.root, 'fresh/operator-observations.json', ledger)['sha256']
        with self.assertRaisesRegex(RuntimeError, 'run/projection differs'):
            self.load_observations()

    def test_historical_approvals_do_not_replace_explicit_fresh_grant(self):
        for fresh_grant, cost_review in ((False, False), (False, True), (True, False)):
            with self.subTest(fresh_grant=fresh_grant, cost_review=cost_review):
                self.args.approve_fresh_one_run = fresh_grant
                self.args.approve_ancillary_retention_network_cost_review = cost_review
                with patch.object(operator, 'observations') as observations, \
                        patch.object(operator.subprocess, 'run') as runner:
                    with self.assertRaisesRegex(RuntimeError, 'Fresh one-run authorization'):
                        operator.authorize(self.args, self.ctx)
                    observations.assert_not_called()
                    runner.assert_not_called()
        self.assertFalse((self.run / 'authorization.json').exists())

    def test_existing_grant_cannot_be_overwritten_even_with_fresh_flags(self):
        self.args.approve_fresh_one_run = True
        self.args.approve_ancillary_retention_network_cost_review = True
        self.args.observations_sha256 = SHA
        pin(self.root, 'fresh/authorization.json', {'approved': True, 'historical': True})
        before = (self.run / 'authorization.json').read_bytes()
        with patch.object(operator, 'observations') as observations:
            with self.assertRaisesRegex(RuntimeError, 'never overwrite or replay a grant'):
                operator.authorize(self.args, self.ctx)
            observations.assert_not_called()
        self.assertEqual((self.run / 'authorization.json').read_bytes(), before)


class PublicEvidenceRegistrationBindingTests(unittest.TestCase):
    def test_only_registered_schemas_select_an_execution_profile(self):
        for schema, expected in (
                (fresh.PUBLIC_EVIDENCE_SCHEMA, 'trid-public-evidence-gpu-plan-v2'),
                (fresh.PUBLIC_WORK_SCHEMA, 'trid-public-work-gpu-plan-v1'),
                (fresh.SCHEMA, fresh.lifecycle.FRESH_SCHEMA),
                (fresh.EVALUATION_SCHEMA, fresh.lifecycle.EVALUATION_SCHEMA)):
            with self.subTest(schema=schema):
                self.assertEqual(fresh.registration_plan_schema({'schema': schema}), expected)
        for schema in ('trid-public-evidence-prerequisites-v1', 'unregistered-prerequisites', None):
            reg = {'schema': schema}
            with self.subTest(schema=schema):
                self.assertFalse(fresh.public_evidence_registration(reg))
                with self.assertRaisesRegex(RuntimeError, 'generic fallback forbidden'):
                    fresh.registration_plan_schema(reg)
                with self.assertRaisesRegex(RuntimeError, 'Exact parent prerequisite registration'):
                    fresh.validate_registration(reg, '/missing')
                with patch.object(fresh.lifecycle, 'validate_plan') as validate_plan:
                    with self.assertRaisesRegex(RuntimeError, 'generic fallback forbidden'):
                        fresh.validate_final(reg, SHA, '/missing', '/missing',
                                             {'schema': fresh.lifecycle.FRESH_SCHEMA}, {}, SHA)
                    validate_plan.assert_not_called()

    def test_old_public_plan_is_rejected_before_final_dispatch(self):
        reg = {'schema': fresh.PUBLIC_EVIDENCE_SCHEMA}
        with patch.object(fresh.lifecycle, 'validate_plan') as validate_plan:
            with self.assertRaisesRegex(RuntimeError, 'Final plan not bound'):
                fresh.validate_final(reg, SHA, '/missing', '/missing',
                    {'schema': 'trid-public-evidence-gpu-plan-v1'}, {}, SHA)
            validate_plan.assert_not_called()

    def test_final_plan_cannot_change_registered_deadline_cost_or_inputs(self):
        reg = dict(schema=fresh.PUBLIC_EVIDENCE_SCHEMA, identity={'run_id': 'public-evidence-fixture'},
            iam={}, controller_arn='synthetic-controller', authorization={'sha256': 'a' * 64},
            resource_envelope={'sha256': 'b' * 64}, workload={'evaluation_inputs': {'sha256': 'c' * 64}},
            recipe={'rows': 1230, 'steps': 615}, limits={'lifetime_seconds': 27900},
            lifetime={'total_seconds': 28800}, cost={'cap_usd': 30},
            absolute_termination_deadline='2030-01-02T00:00:00Z',
            data={'train': {'sha256': 'd' * 64}}, phase_seconds={'train': 3600},
            ancillary_evidence={'sha256': 'e' * 64}, bootstrap_admission={'sha256': 'f' * 64})
        for field in ('workload', 'recipe', 'limits', 'lifetime', 'cost', 'absolute_termination_deadline',
                      'data', 'phase_seconds', 'ancillary_evidence'):
            plan = copy.deepcopy(reg)
            plan['schema'] = fresh.lifecycle.PUBLIC_EVIDENCE_SCHEMA
            plan['runtime'] = {'bootstrap': {'admission': reg['bootstrap_admission']}}
            plan[field] = {'unregistered': True} if field != 'absolute_termination_deadline' else '2030-01-03T00:00:00Z'
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'differs from registration'):
                fresh.validate_final(reg, 'a' * 64, '/missing', '/missing', plan, {}, 'b' * 64)


class PublicWorkPurgeTests(unittest.TestCase):
    def test_explicit_retrieval_preserves_launch_state_and_requires_cleanup(self):
        fresh.record(self.state / 'approval-consumed.json',
                     dict(run_id=self.identity['run_id'], plan_sha256=SHA))
        release = dict(run_id=self.identity['run_id'], plan_sha256=SHA, instance_id='i-fixture')
        fresh.record(self.state / 'release.json', release)
        fresh.record(self.state / 'export-retrieval-intent.json', {'interrupted_launch': True})
        events = []
        with patch.object(fresh, 'verify_compute_cleanup', side_effect=lambda *args: events.append('cleanup')), \
                patch.object(fresh.lifecycle, 'collect_export',
                    side_effect=lambda *args: events.append('collect') or {'export_state': 'authenticated_terminal'}) as collect:
            result = fresh.retrieve_run_export(self.reg, SHA, self.state, self.client, self.plan, SHA)
        self.assertEqual(events, ['cleanup', 'collect'])
        collect.assert_called_once_with(self.client, self.plan, SHA,
                                        self.state / 'retrieved-export', 'i-fixture')
        self.assertEqual(result['export_state'], 'authenticated_terminal')
        self.assertEqual(fresh.load(self.state / 'retrieved-export' / 'release.json'), release)
        self.assertEqual(fresh.load(self.state / 'export-retrieval-intent.json'), {'interrupted_launch': True})
        with patch.object(fresh.lifecycle, 'collect_export') as collect:
            with self.assertRaises(FileExistsError):
                fresh.retrieve_run_export(self.reg, SHA, self.state, self.client, self.plan, SHA)
            collect.assert_not_called()

    def test_retrieval_rejects_wrong_profile_and_consumed_approval(self):
        with patch.object(fresh.lifecycle, 'collect_export') as collect:
            self.reg['schema'] = fresh.PUBLIC_EVIDENCE_SCHEMA
            with self.assertRaisesRegex(RuntimeError, 'distinct bound public-work'):
                fresh.retrieve_run_export(self.reg, SHA, self.state, self.client, self.plan, SHA)
            self.reg['schema'] = fresh.PUBLIC_WORK_SCHEMA
            fresh.record(self.state / 'approval-consumed.json',
                         dict(run_id=self.identity['run_id'], plan_sha256='b' * 64))
            with self.assertRaisesRegex(RuntimeError, 'exact consumed approval'):
                fresh.retrieve_run_export(self.reg, SHA, self.state, self.client, self.plan, SHA)
            collect.assert_not_called()

    def test_purge_accepts_authenticated_explicit_retrieval(self):
        destination = self.state / 'retrieved-export'
        destination.mkdir()
        for name in ('completion.json', 'export.tar', 'local-export-authenticated.json'):
            (destination / name).write_bytes((self.state / name).read_bytes())
        (self.state / 'export.tar').write_bytes(b'interrupted launch archive')
        self.assertEqual(self.purge()['cleanup_state'], 'complete')

    def test_work_and_evidence_dispatch_remain_distinct(self):
        work = {'schema': fresh.lifecycle.PUBLIC_WORK_SCHEMA}
        evidence = {'schema': fresh.lifecycle.PUBLIC_EVIDENCE_SCHEMA}
        self.assertTrue(fresh.lifecycle.public_contract_mode(work))
        self.assertFalse(fresh.lifecycle.public_evidence_mode(work))
        self.assertFalse(fresh.lifecycle.public_work_mode(evidence))
        self.assertEqual(fresh.lifecycle.success_field(work), 'public_work_success')
        self.assertEqual(fresh.lifecycle.success_field(evidence), 'public_evidence_success')
        with self.assertRaisesRegex(RuntimeError, 'distinct schema'):
            fresh.lifecycle.public_evidence_module(work)
        with self.assertRaisesRegex(RuntimeError, 'generic fallback forbidden'):
            fresh.lifecycle.public_contract_module({'schema': 'unknown'})

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)
        self.identity = dict(run_id='public-work-fixture', bucket='fixture-bucket',
            code_prefix='code/public-work-fixture/', output_prefix='outputs/public-work-fixture/')
        self.reg = dict(schema=fresh.PUBLIC_WORK_SCHEMA, identity=self.identity)
        self.plan = dict(schema=fresh.lifecycle.PUBLIC_WORK_SCHEMA, identity=self.identity,
                         cost={'export_bytes': 1024})
        archive = self.state / 'export.tar'
        archive.write_bytes(b'authenticated archive fixture')
        self.pin = dict(bytes=archive.stat().st_size, sha256=fresh.lifecycle.digest(archive))
        completion = dict(run_id=self.identity['run_id'], instance_id='i-fixture',
                          plan_sha256=SHA, export=self.pin)
        fresh.record(self.state / 'completion.json', completion)
        fresh.record(self.state / 'local-export-authenticated.json',
            dict(schema='trid-public-work-local-export-authentication-v1',
                 run_id=self.identity['run_id'], plan_sha256=SHA, instance_id='i-fixture',
                 completion_sha256=fresh.lifecycle.digest(self.state / 'completion.json'),
                 export=self.pin, export_state='authenticated_terminal'))
        self.keys = {self.identity['code_prefix'] + 'plan.json',
                     self.identity['output_prefix'] + 'export.tar'}
        self.calls = []
        self.versioning = {}
        self.multipart = {}
        self.retain_deleted = False
        for module, name, value in (
                (fresh, 'caller_matches', None),
                (fresh, 'verify_compute_cleanup', None),
                (fresh.lifecycle, 'load_volume_identity', {'instance_id': 'i-fixture'}),
                (fresh.lifecycle, 'validate_lifecycle_configuration', None)):
            patcher = patch.object(module, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def client(self, service, operation, *args):
        self.calls.append((service, operation, args))
        if operation == 'get-bucket-versioning':
            return self.versioning
        if operation == 'get-bucket-lifecycle-configuration':
            return {'Rules': [{'ID': 'preserved-unrelated-rule'}]}
        if operation == 'list-multipart-uploads':
            return self.multipart
        if operation == 'list-objects-v2':
            prefix = args[args.index('--prefix') + 1]
            return {'Contents': [{'Key': key} for key in sorted(self.keys) if key.startswith(prefix)]}
        if operation in ('delete-object', 'head-object'):
            key = args[args.index('--key') + 1]
            if operation == 'delete-object':
                if not self.retain_deleted:
                    self.keys.remove(key)
                return {}
            if key not in self.keys:
                raise missing(service, operation, '404')
            return {}
        self.fail('Unexpected AWS operation: ' + operation)

    def purge(self):
        return fresh.purge_run_s3(self.reg, SHA, self.state, self.client, self.plan, SHA)

    def test_purge_journals_exact_keys_and_preserves_lifecycle(self):
        expected = sorted(self.keys)
        result = self.purge()
        self.assertEqual(result['deleted_keys'], expected)
        self.assertEqual(result['s3_mutations'], 2)
        self.assertTrue(result['retained_prefix_lifecycle_rules'])
        self.assertFalse(self.keys)
        for index in range(2):
            for suffix in ('intent', 'response', 'readback'):
                self.assertTrue((self.state / f's3-purge-object-{index}-{suffix}.json').is_file())
        self.assertFalse(any(op == 'put-bucket-lifecycle-configuration' for _, op, _ in self.calls))
        with self.assertRaises(FileExistsError):
            self.purge()
        self.assertEqual(sum(op == 'delete-object' for _, op, _ in self.calls), 2)

    def test_changed_local_export_blocks_all_cloud_access(self):
        (self.state / 'export.tar').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'missing or changed'):
            self.purge()
        self.assertEqual(self.calls, [])

    def test_compute_cleanup_failure_prevents_s3_mutation(self):
        with patch.object(fresh, 'verify_compute_cleanup', side_effect=RuntimeError('compute remains')):
            with self.assertRaisesRegex(RuntimeError, 'compute remains'):
                self.purge()
        self.assertEqual(self.calls, [])

    def test_rejects_evidence_profile_and_nonexact_prefix(self):
        self.reg['schema'] = fresh.PUBLIC_EVIDENCE_SCHEMA
        with self.assertRaisesRegex(RuntimeError, 'distinct bound public-work'):
            self.purge()
        self.reg['schema'] = fresh.PUBLIC_WORK_SCHEMA
        self.identity['output_prefix'] = 'outputs/public-work-fixture/extra/'
        with self.assertRaisesRegex(RuntimeError, 'only exact run'):
            self.purge()
        self.assertEqual(self.calls, [])

    def test_versioning_and_incomplete_uploads_block_deletion(self):
        self.versioning = {'Status': 'Suspended'}
        with self.assertRaisesRegex(RuntimeError, 'Versioned/suspended'):
            self.purge()
        self.versioning = {}
        self.multipart = {'Uploads': [{'Key': self.identity['code_prefix'] + 'bundle.tar'}]}
        with self.assertRaisesRegex(RuntimeError, 'multipart run inventory'):
            self.purge()
        self.assertFalse(any(op == 'delete-object' for _, op, _ in self.calls))

    def test_absence_failure_is_not_retried(self):
        self.retain_deleted = True
        with self.assertRaisesRegex(RuntimeError, 'Object remains'):
            self.purge()
        self.assertTrue((self.state / 's3-purge-object-0-intent.json').is_file())
        self.assertFalse((self.state / 's3-purge-object-0-readback.json').exists())
        with self.assertRaises(FileExistsError):
            self.purge()
        self.assertEqual(sum(op == 'delete-object' for _, op, _ in self.calls), 1)


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
    auth = pin(root, 'authorization.json', TRAINING_AUTHORIZATION)
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
    reg['authorization'] = pin(root, auth_name, EVALUATION_AUTHORIZATION)
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


def sampling_registration(root):
    """Actual JSON shape, synthetic resources; never an executable cloud approval."""
    reg = evaluation_registration(root)
    old = reg['identity']['run_id']
    reg = json.loads(json.dumps(reg).replace(old, 'mois-eight-sampling-offline'))
    reg.update(decoder_profile=fresh.lifecycle.SAMPLING_PROFILE,
               absolute_termination_deadline=fresh.lifecycle.SAMPLING_DEADLINE)
    name = fresh.lifecycle.OVERNIGHT_AUTHORIZATION
    previous_pins = {reg[key]['path'] for key in ('authorization', 'resource_envelope')}
    reg['authorization'] = pin(root, name, SAMPLING_AUTHORIZATION)
    envelope = json.loads((root / reg['resource_envelope']['path']).read_bytes())
    envelope.update(schema='trid-cuda-evaluation-sampling-resource-envelope-v1',
        identity=reg['identity'], authorization=reg['authorization'],
        decoder_profile=reg['decoder_profile'], absolute_termination_deadline=reg['absolute_termination_deadline'],
        operator_imposed_cap_usd='20', automatic_retry=False, automatic_extension=False,
        gpu_execution_lease_seconds_max=13500, total_billable_lifetime_seconds_max=14400)
    reg['resource_envelope'] = pin(root, 'sampling-envelope.json', envelope)
    reg['bundle']['files'] = [row for row in reg['bundle']['files'] if row['path'] not in previous_pins]
    reg['bundle']['files'] += [reg['authorization'], reg['resource_envelope']]
    return reg


class SamplingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.reg = sampling_registration(self.root)
        evaluation = fresh.lifecycle.evaluation_module({'schema': fresh.lifecycle.EVALUATION_SCHEMA})
        for module, name in ((fresh.lifecycle, 'OVERNIGHT_AUTHORIZATION_SHA'),
                             (evaluation, 'SAMPLING_AUTHORIZATION_SHA')):
            patcher = patch.object(module, name, self.reg['authorization']['sha256'])
            patcher.start()
            self.addCleanup(patcher.stop)
        self.now = fresh.lifecycle.timestamp(fresh.lifecycle.SAMPLING_DEADLINE) - 20000

    def test_sampling_registration_binds_separate_grant_and_deadline(self):
        with patch.object(fresh.time, 'time', return_value=self.now), patch.object(fresh, 'validate_lineage'):
            self.assertIs(fresh.validate_registration(self.reg, self.root), self.reg)
        with patch.object(fresh.time, 'time', return_value=self.now + 20001):
            with self.assertRaisesRegex(RuntimeError, 'remaining overnight'):
                fresh.validate_registration(self.reg, self.root)
        with patch.object(fresh.time, 'time', return_value=self.now + 20001), patch.object(fresh, 'validate_lineage'):
            self.assertIs(fresh.validate_registration(self.reg, self.root, launch_window=False), self.reg)

    def test_sampling_cannot_reuse_consumed_greedy_authorization(self):
        name = 'runs/mois-eight-cuda-evaluation-20261116T134815Z/authorization.json'
        self.reg['authorization'] = pin(self.root, name, EVALUATION_AUTHORIZATION)
        with patch.object(fresh.time, 'time', return_value=self.now), patch.object(fresh, 'validate_lineage'):
            with self.assertRaisesRegex(RuntimeError, 'Separate overnight authorization'):
                fresh.validate_registration(self.reg, self.root)

    def test_sampling_envelope_cannot_extend_deadline_or_lease(self):
        for key, value in (('absolute_termination_deadline', '2026-11-17T00:00:00Z'),
                           ('gpu_execution_lease_seconds_max', 18900),
                           ('total_billable_lifetime_seconds_max', 19800)):
            with self.subTest(key=key):
                reg = copy.deepcopy(self.reg)
                envelope = json.loads((self.root / reg['resource_envelope']['path']).read_bytes())
                envelope[key] = value
                reg['resource_envelope'] = pin(self.root, 'mutated-envelope.json', envelope)
                with patch.object(fresh.time, 'time', return_value=self.now), patch.object(fresh, 'validate_lineage'):
                    with self.assertRaises(RuntimeError):
                        fresh.validate_registration(reg, self.root)

    def test_final_plan_must_match_registration_deadline(self):
        reg = self.reg
        plan = {key: reg[key] for key in ('identity', 'iam', 'controller_arn', 'authorization', 'resource_envelope')}
        plan.update(schema=fresh.lifecycle.EVALUATION_SCHEMA,
                    workload=dict(decoder_profile=reg['decoder_profile'], lineage_binding=reg['lineage_binding']),
                    absolute_termination_deadline='2026-11-17T00:00:00Z')
        with self.assertRaisesRegex(RuntimeError, 'Final absolute deadline differs'):
            fresh.validate_final(reg, SHA, self.root, self.root, plan, {}, SHA, active=False)

    def test_exact_sampling_approval_rejects_changed_deadline_and_prior_grant(self):
        E = fresh.lifecycle.evaluation_module({'schema': fresh.lifecycle.EVALUATION_SCHEMA})
        plan = dict(schema=fresh.lifecycle.EVALUATION_SCHEMA, identity=self.reg['identity'],
                    workload={'decoder_profile': E.SAMPLING_PROFILE}, authorization=self.reg['authorization'],
                    absolute_termination_deadline=self.reg['absolute_termination_deadline'])
        approval = dict(schema=E.SAMPLING_APPROVAL_SCHEMA, run_id=plan['identity']['run_id'],
            plan_sha256=SHA, instance_limit=1, operator_imposed_cap_usd='20',
            fixed_dollar_cap_explicitly_selected_by_user=False,
            workload_sha256=fresh.lifecycle.object_sha(plan['workload']),
            authorization_sha256=plan['authorization']['sha256'], limits=E.SAMPLING_LIMITS,
            ebs_kms_key_arn='synthetic-key', automatic_retry=False, automatic_extension=False,
            retraining=False, deployment=False, approved=True, decoder_profile=E.SAMPLING_PROFILE,
            absolute_termination_deadline=plan['absolute_termination_deadline'])
        E.validate_approval(plan, approval, SHA, 'synthetic-key')
        for key, value in (('absolute_termination_deadline', '2026-11-17T00:00:00Z'),
                           ('authorization_sha256', '4967ea79e40e1f5df25004460642d80f8832c190d8d785323250cf2428fafd98'),
                           ('fixed_dollar_cap_explicitly_selected_by_user', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                E.validate_approval(plan, dict(approval, **{key: value}), SHA, 'synthetic-key')



class CudaEvaluationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.state = self.root / 'state'
        self.state.mkdir()
        self.reg = evaluation_registration(self.root)
        patch_authorization_digest(self, fresh.lifecycle.validate_evaluation_resources,
            '4967ea79e40e1f5df25004460642d80f8832c190d8d785323250cf2428fafd98',
            self.reg['authorization']['sha256'])

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
        patch_authorization_digest(self, fresh.lifecycle.validate_fresh_budget,
            'a305a416c12f4038a78078cb5201d567c6e3d1d34b5dd44839e086b049e90967',
            self.reg['authorization']['sha256'])

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
