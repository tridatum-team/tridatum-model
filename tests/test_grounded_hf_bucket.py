"""Private S3 admission behavior with synthetic transport responses; no AWS calls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import grounded_hf_lifecycle as L


ABSENT = dict(state='absent', error_code='NoSuchBucketPolicy')
PRESENT = dict(state='present', is_public=False)


def aws_error(operation, code):
    return L.AWSCallError('s3api', operation, f'An error occurred ({code}) when calling the operation')


def private_fixture():
    plan = dict(identity=dict(bucket='synthetic-private-bucket', account='123456789012', region='ap-northeast-2'),
                bucket_policy_status=copy.deepcopy(ABSENT))
    responses = {
        'get-public-access-block': {'PublicAccessBlockConfiguration': dict(
            BlockPublicAcls=True, IgnorePublicAcls=True, BlockPublicPolicy=True, RestrictPublicBuckets=True)},
        'get-bucket-ownership-controls': {'OwnershipControls': {'Rules': [{'ObjectOwnership': 'BucketOwnerEnforced'}]}},
        'get-bucket-acl': {'Owner': {'ID': 'synthetic-canonical-owner'}, 'Grants': [
            {'Permission': 'FULL_CONTROL', 'Grantee': {'Type': 'CanonicalUser', 'ID': 'synthetic-canonical-owner'}}]},
        'get-bucket-policy-status': aws_error('get-bucket-policy-status', 'NoSuchBucketPolicy'),
    }

    def client(service, operation, *args):
        result = responses[operation]
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    return plan, responses, client


class PrivateBucketAdmissionTests(unittest.TestCase):
    def test_no_policy_is_valid_absence_when_all_private_owner_controls_hold(self):
        plan, _, client = private_fixture()
        observed = L.observe_private_bucket(plan, client)
        self.assertEqual(observed, ABSENT)
        self.assertNotIn('is_public', observed)

    def test_access_denied_or_other_status_errors_are_not_policy_absence(self):
        for code in ('AccessDenied', 'NoSuchBucket', 'RequestTimeout', 'NoSuchLifecycleConfiguration'):
            with self.subTest(code=code):
                plan, responses, client = private_fixture()
                responses['get-bucket-policy-status'] = aws_error('get-bucket-policy-status', code)
                with self.assertRaises(L.AWSCallError) as caught:
                    L.observe_private_bucket(plan, client)
                self.assertEqual(caught.exception.code, code)

    def test_unclassified_transport_error_is_not_policy_absence(self):
        plan, responses, client = private_fixture()
        responses['get-bucket-policy-status'] = L.AWSCallError('s3api', 'get-bucket-policy-status', 'transport failed')
        with self.assertRaises(L.AWSCallError):
            L.observe_private_bucket(plan, client)

    def test_policy_absence_does_not_replace_public_access_blocks(self):
        plan, responses, client = private_fixture()
        responses['get-public-access-block']['PublicAccessBlockConfiguration']['BlockPublicPolicy'] = False
        with self.assertRaisesRegex(RuntimeError, 'public-access blocks'):
            L.observe_private_bucket(plan, client)

    def test_policy_absence_does_not_replace_bucket_owner_enforced_control(self):
        plan, responses, client = private_fixture()
        responses['get-bucket-ownership-controls']['OwnershipControls']['Rules'][0]['ObjectOwnership'] = 'ObjectWriter'
        with self.assertRaisesRegex(RuntimeError, 'Bucket-owner-enforced'):
            L.observe_private_bucket(plan, client)

    def test_public_or_different_owner_acl_is_rejected_even_without_policy(self):
        for grantee in ({'Type': 'Group', 'URI': 'http://acs.amazonaws.com/groups/global/AllUsers'},
                        {'Type': 'CanonicalUser', 'ID': 'different-owner'}):
            with self.subTest(grantee=grantee):
                plan, responses, client = private_fixture()
                responses['get-bucket-acl']['Grants'][0]['Grantee'] = grantee
                with self.assertRaisesRegex(RuntimeError, 'Non-owner or public ACL'):
                    L.observe_private_bucket(plan, client)

    def test_failed_acl_ownership_observation_is_not_private_proof(self):
        plan, responses, client = private_fixture()
        responses['get-bucket-acl'] = aws_error('get-bucket-acl', 'AccessDenied')
        with self.assertRaises(L.AWSCallError):
            L.observe_private_bucket(plan, client)

    def test_public_policy_is_rejected_and_private_presence_requires_rebinding(self):
        plan, responses, client = private_fixture()
        responses['get-bucket-policy-status'] = {'PolicyStatus': {'IsPublic': True}}
        with self.assertRaisesRegex(RuntimeError, 'Public or unknown bucket policy'):
            L.observe_private_bucket(plan, client)
        responses['get-bucket-policy-status'] = {'PolicyStatus': {'IsPublic': False}}
        with self.assertRaisesRegex(RuntimeError, 'policy status changed'):
            L.observe_private_bucket(plan, client)
        plan['bucket_policy_status'] = copy.deepcopy(PRESENT)
        self.assertEqual(L.observe_private_bucket(plan, client), PRESENT)

    def test_ancillary_policy_absence_must_be_registered_and_status_consistent(self):
        plan, _, _ = private_fixture()
        check = dict(service='s3api', operation='get-bucket-policy', args=['--bucket', plan['identity']['bucket']],
                     expected_absence='NoSuchBucketPolicy')

        def absent(*args):
            raise aws_error('get-bucket-policy', 'NoSuchBucketPolicy')

        L.observe_ancillary_check('bucket_policy', check, plan['identity'], ABSENT, absent)
        with self.assertRaisesRegex(RuntimeError, 'absence disagrees'):
            L.observe_ancillary_check('bucket_policy', check, plan['identity'], PRESENT, absent)
        check['args'] = ['--bucket', 'another-bucket']
        with self.assertRaisesRegex(RuntimeError, 'not evidence of absence'):
            L.observe_ancillary_check('bucket_policy', check, plan['identity'], ABSENT, absent)

    def test_access_denied_cannot_be_registered_as_ancillary_absence(self):
        plan, _, _ = private_fixture()
        check = dict(service='s3api', operation='get-bucket-policy', args=['--bucket', plan['identity']['bucket']],
                     expected_absence='AccessDenied')

        def denied(*args):
            raise aws_error('get-bucket-policy', 'AccessDenied')

        with self.assertRaisesRegex(RuntimeError, 'not evidence of absence'):
            L.observe_ancillary_check('bucket_policy', check, plan['identity'], ABSENT, denied)


class PolicyEvidenceTests(unittest.TestCase):
    def fixture(self, root, mutate=None):
        plan, _, _ = private_fixture()
        raw = dict(query=['aws', 's3api', 'get-bucket-policy-status', '--bucket', plan['identity']['bucket'],
                          '--region', plan['identity']['region']], exit_code=254, response=None,
                   error_code='NoSuchBucketPolicy', stderr='An error occurred (NoSuchBucketPolicy) when calling the operation')
        if mutate:
            mutate(raw)
        path = Path(root) / 'status.json'
        path.write_text(json.dumps(raw))
        pin = dict(path=path.name, bytes=path.stat().st_size, sha256=L.digest(path))
        plan['bundle'] = {'files': [pin]}
        return plan, pin

    def test_raw_absence_receipt_is_preserved_and_authenticated(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, pin = self.fixture(temp)
            before = (Path(temp) / pin['path']).read_bytes()
            self.assertEqual(L.validate_policy_status_evidence(temp, plan, pin), ABSENT)
            self.assertEqual((Path(temp) / pin['path']).read_bytes(), before)
            (Path(temp) / pin['path']).write_bytes(before + b' ')
            with self.assertRaises(RuntimeError):
                L.validate_policy_status_evidence(temp, plan, pin)

    def test_access_denied_or_mislabeled_stderr_cannot_prove_absence(self):
        mutations = [lambda raw: raw.update(error_code='AccessDenied', stderr='An error occurred (AccessDenied)'),
                     lambda raw: raw.update(stderr='An error occurred (AccessDenied)'),
                     lambda raw: raw.update(response={'PolicyStatus': {'IsPublic': False}})]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temp:
                plan, pin = self.fixture(temp, mutate)
                with self.assertRaisesRegex(RuntimeError, 'failure is not absence'):
                    L.validate_policy_status_evidence(temp, plan, pin)

    def test_other_bucket_evidence_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, pin = self.fixture(temp, lambda raw: raw['query'].__setitem__(4, 'another-bucket'))
            with self.assertRaisesRegex(RuntimeError, 'another operation or resource'):
                L.validate_policy_status_evidence(temp, plan, pin)


class BucketEncryptionTests(unittest.TestCase):
    def test_actual_aes256_security_fields_are_supported_without_mutation(self):
        encryption = {'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'},
                                'BucketKeyEnabled': False, 'BlockedEncryptionTypes': {'EncryptionType': ['SSE-C']}}]}
        before = copy.deepcopy(encryption)
        L.validate_bucket_encryption(encryption)
        self.assertEqual(encryption, before)
        L.validate_bucket_encryption({'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}]})

    def test_unknown_or_cost_changing_encryption_fields_are_rejected(self):
        updates = [dict(BucketKeyEnabled=True), dict(UnknownEncryptionFeature=True),
                   dict(ApplyServerSideEncryptionByDefault={'SSEAlgorithm': 'aws:kms'}),
                   dict(BlockedEncryptionTypes={'EncryptionType': ['unknown']})]
        for update in updates:
            with self.subTest(update=update):
                rule = {'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}, **update}
                with self.assertRaisesRegex(RuntimeError, 'known cost-closed SSE-S3'):
                    L.validate_bucket_encryption({'Rules': [rule]})


if __name__ == '__main__':
    unittest.main()
