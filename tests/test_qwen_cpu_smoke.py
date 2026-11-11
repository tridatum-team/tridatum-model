"""Stdlib-only command/audit boundaries; no cloud, models, or repository artifacts.

These tests do not establish real Linux guard, cache compatibility, or smoke success.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.aws_eval import qwen_cpu_smoke as smoke
from scripts.aws_eval import qwen_cpu_pipeline as pipeline


def frozen_plan():
    pins = [{'path': f'/audit/shard-{n:03d}.json', 'sha256': f'{n:064x}'}
            for n in range(1, 7)]
    child = ['scripts/train_qwen_preferences.py',
             '--model', '/immutable/model', '--model-id', 'Qwen/Qwen3-1.7B',
             '--model-revision', 'frozen-revision', '--base-gguf', '/immutable/base.gguf',
             '--base-gguf-sha256', 'a' * 64, '--manifest', '/immutable/manifest.json',
             '--rank', '16', '--alpha', '32', '--source-dropout', '0.05',
             '--max-tokens', '1024', '--beta', '0.1', '--lr', '0.00005',
             '--batch-pairs', '4', '--seed', '20261030', '--replay-micro-batch', '4',
             '--mode', 'train', '--out', '/old/train', '--checkpoint-every', '2',
             '--valid-every', '2', '--max-steps', '2']
    for pin in pins:
        child += ['--cache', pin['path'], '--cache-sha256', pin['sha256']]
    return {'argv': ['/python', '-B', '/old/guard.py', '--seconds', '180', '--', *child],
            'verified_cache_shards': pins}


def audit_shards():
    bindings = {
        'kind': 'preference_reference_cache', 'manifest_sha256': 'a' * 64,
        'reference': {'checkpoint': '/source011/step-000024', 'source_steps': 24,
                      'source_cumulative_steps': 96},
        'base': {'gguf_sha256': 'b' * 64},
        'tokenizer': {'files': {'tokenizer.json': 'c' * 64}, 'eos_token_id': 151645},
        'family': 'qwen3', 'max_tokens': 1024,
        'objective': 'summed_answer_logprob_including_eos',
        'replay': {'rows_admitted': 288, 'weight': 1.0, 'epochs': 1,
                   'token_label_sha256': 'd' * 64, 'rows_by_file': {'train.jsonl': 288}},
        'base_weights': {'representation': 'q8_0', 'quantized_tensors': 197},
        'init_adapter': {'adapter_weights_sha256': 'e' * 64, 'tensors_loaded': 392},
        'admitted_pairs': 91, 'seed': 20261030, 'device': 'cpu', 'dtype': 'torch.float32',
        'manifest': '/immutable/manifest.json', 'manifest_id': 'preferences-with-replay',
    }
    shards = []
    start = 0
    for size in [16] * 5 + [11]:
        ids = [f'pair-{i:03d}' for i in range(start, start + size)]
        shard = copy.deepcopy(bindings)
        shard.update(ids=ids, scores={pid: {'pair_sha256': f'{i + 1:064x}',
                                          'chosen': -1000.25 - i, 'rejected': -2000.75 - i}
                                      for i, pid in enumerate(ids, start)})
        shards.append(shard)
        start += size
    return shards


def write_audit(directory, shards):
    rows = []
    for n, shard in enumerate(shards, 1):
        path = Path(directory) / f'shard-{n:03d}.json'
        data = json.dumps(shard).encode()
        path.write_bytes(data)
        rows.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()})
    return {'verified_cache_shards': rows}


def reuse_fixture(root, shards=None):
    origin = root / 'runs' / smoke.EXPERIMENT / smoke.ORIGIN_RUN_ID
    directory = origin / 'remote-output/reference-cache-linux'
    directory.mkdir(parents=True)
    plan = write_audit(directory, audit_shards() if shards is None else shards)
    acceptance = origin / 'remote-output/linux-cache-acceptance.json'
    acceptance.write_text(json.dumps({'accepted': True, 'pairs': 91, 'replay_rows': 288,
        'cache_recomputed': True, 'old_scores_reused': False, 'shards': plan['verified_cache_shards']}))
    recovery = origin / 'output-recovery-verification.json'
    recovery.write_text(json.dumps({'run_id': smoke.ORIGIN_RUN_ID,
        'all_retained_files_hash_verified': True, 'export_complete': False,
        'export_errors': ['Historical export manifest unavailable']}))
    plan['cache_acceptance'] = {'path': str(acceptance), 'sha256': smoke.digest(acceptance)}
    plan['cache_reuse'] = {'origin_run_id': smoke.ORIGIN_RUN_ID, 'recompute_on_failure': False,
        'recovery_verification': {'path': str(recovery), 'sha256': smoke.digest(recovery)}}
    return plan


class UserDataBoundaryTests(unittest.TestCase):
    def launch_request(self, bootstrap=None, error='offline launch boundary'):
        """Capture the real CLI JSON without executing any subprocess or cloud call."""
        requests = []

        def command(argv, **kwargs):
            self.assertEqual(argv[0], 'aws')
            self.assertEqual(kwargs['env']['AWS_MAX_ATTEMPTS'], '1')
            operation = tuple(argv[1:3])
            if operation == ('s3api', 'head-object'):
                result = {'ContentLength': 1, 'ServerSideEncryption': 'AES256'}
            elif operation == ('iam', 'simulate-principal-policy'):
                action = argv[argv.index('--action-names') + 1]
                start = argv.index('--resource-arns') + 1
                resources = []
                for value in argv[start:]:
                    if value.startswith('--'):
                        break
                    resources.append(value)
                result = {'EvaluationResults': [{
                    'EvalActionName': action, 'EvalDecision': 'allowed', 'EvalResourceName': '*',
                    'ResourceSpecificResults': [
                        {'EvalResourceName': resource, 'EvalResourceDecision': 'allowed'}
                        for resource in resources]}]}
            elif operation == ('ec2', 'run-instances'):
                self.assertEqual(argv[3], '--cli-input-json')
                requests.append(json.loads(argv[4]))
                raise RuntimeError('offline launch boundary')
            else:
                self.fail(f'Unexpected command: {operation}')
            return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                run_id=smoke.RUN_ID, bundle_sha256='a' * 64,
                bundle_s3_uri=f's3://{smoke.BUCKET}/{smoke.PREFIX}/code/{smoke.RUN_ID}/inputs.tar',
                provenance=smoke.PROVENANCE, state_dir=directory,
                subnet_id='subnet-offline', max_seconds=7200, approved_usd=10)
            fixed_now = smoke.dt.datetime(2026, 11, 8, 0, 0, 0, 123456,
                                          tzinfo=smoke.dt.timezone.utc)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(smoke, 'preflight', return_value={
                    'ami': {'ImageId': 'ami-offline', 'RootDeviceName': '/dev/xvda'},
                    'security_group': {'GroupId': 'sg-offline'}}))
                stack.enter_context(mock.patch.object(smoke, 'instances', return_value=[]))
                stack.enter_context(mock.patch.object(smoke.subprocess, 'run', side_effect=command))
                clock = stack.enter_context(mock.patch.object(
                    smoke.dt, 'datetime', wraps=smoke.dt.datetime))
                clock.now.return_value = fixed_now
                stack.enter_context(mock.patch.object(smoke.time, 'time', return_value=fixed_now.timestamp()))
                stack.enter_context(mock.patch.object(smoke, 'reconcile_cleanup', return_value={
                    'cleanup_state': 'cleanup_unresolved', 'launch_outcome': 'launch_outcome_unknown'}))
                if bootstrap is not None:
                    stack.enter_context(mock.patch.object(Path, 'read_text', return_value=bootstrap))
                with self.assertRaisesRegex(RuntimeError, error):
                    smoke.launch(args)
            if requests:
                self.assertEqual(len(requests), 1)
        return requests

    def test_serialized_cli_user_data_is_raw_executable_utf8(self):
        request, = self.launch_request()
        raw = request['UserData'].encode('utf-8')
        self.assertLessEqual(len(raw), 16384)
        self.assertTrue(raw.startswith(b'#!'))

    def test_limit_applies_to_raw_utf8_not_base64_text(self):
        minimal = '# body\n'
        request, = self.launch_request(bootstrap=minimal)
        prefix_bytes = len(request['UserData'].encode('utf-8')) - len(minimal.encode('utf-8'))
        # 13381 raw bytes became 17844 bytes under the failed double encoding;
        # 16384 exercises the exact EC2 boundary, not merely a small fixture.
        for size in (13381, 16384):
            with self.subTest(raw_bytes=size):
                body = '#' + 'x' * (size - prefix_bytes - 2) + '\n'
                request, = self.launch_request(bootstrap=body)
                raw = request['UserData'].encode('utf-8')
                self.assertEqual(len(raw), size)
        self.assertEqual(self.launch_request(bootstrap=body + 'x',
                                            error='EC2 user data exceeds 16KiB'), [])
        self.assertEqual(self.launch_request(bootstrap='é' * 8192,
                                            error='EC2 user data exceeds 16KiB'), [])


class CommandBoundaryTests(unittest.TestCase):
    def test_frozen_child_rejects_changed_update_or_cache_pin_contract(self):
        plan = frozen_plan()
        expected = plan['argv'][plan['argv'].index('--') + 1:]
        child = smoke.frozen_child(plan)
        self.assertEqual(child, expected)
        child[-1] = 'caller mutation'
        self.assertEqual(plan, frozen_plan())
        for option, value in (('--max-steps', '3'), ('--mode', 'cache'),
                              ('--batch-pairs', '8'), ('--checkpoint-every', '1'),
                              ('--valid-every', '1'), ('--cache', '/different/shard.json'),
                              ('--cache-sha256', 'f' * 64)):
            changed = copy.deepcopy(plan)
            changed['argv'][changed['argv'].index(option) + 1] = value
            with self.subTest(option=option), self.assertRaises(RuntimeError):
                smoke.frozen_child(changed)
        for extra in (['--max-steps', '2'], ['--out', '/second/out'], ['--ids', 'pair-000']):
            changed = copy.deepcopy(plan)
            changed['argv'].extend(extra)
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                smoke.frozen_child(changed)



class CacheAuditBoundaryTests(unittest.TestCase):
    def test_audit_returns_only_partition_bindings_and_pair_digests(self):
        shards = audit_shards()
        with tempfile.TemporaryDirectory() as directory:
            plan = write_audit(directory, shards)
            partitions, bindings, digests = smoke.cache_audit(plan)
            self.assertEqual(partitions, [shard['ids'] for shard in shards])
            self.assertEqual([len(ids) for ids in partitions], [16] * 5 + [11])
            self.assertEqual(len(digests), 91)
            self.assertEqual(digests, {pid: entry['pair_sha256'] for shard in shards
                                       for pid, entry in shard['scores'].items()})
            self.assertEqual(bindings, {key: value for key, value in shards[0].items()
                                        if key not in ('ids', 'scores')})
            # Different numeric score values cannot enter the returned audit contract.
            for shard in shards:
                for entry in shard['scores'].values():
                    entry.update(chosen=-0.125, rejected=-0.25)
            repinned = write_audit(directory, shards)
            self.assertEqual(smoke.cache_audit(repinned), (partitions, bindings, digests))

    def test_changed_file_hash_is_rejected_before_metadata_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = write_audit(directory, audit_shards())
            path = Path(plan['verified_cache_shards'][2]['path'])
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                smoke.cache_audit(plan)

    def test_partition_requires_six_shards_and_exactly_91_unique_ids(self):
        variants = []
        duplicate = audit_shards()
        duplicate[-1]['ids'][0] = duplicate[0]['ids'][0]
        variants.append(duplicate)
        missing = audit_shards()
        missing[-1]['ids'].pop()
        variants.append(missing)
        variants.append(audit_shards()[:-1])
        for n, shards in enumerate(variants):
            with self.subTest(variant=n), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    smoke.cache_audit(write_audit(directory, shards))

    def test_full_immutable_bindings_must_agree_across_shards(self):
        fields = set(audit_shards()[0]) - {'ids', 'scores'}
        for key in sorted(fields):
            shards = audit_shards()
            shards[-1][key] = {'changed': key}
            with self.subTest(field=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'bindings disagree'):
                    smoke.cache_audit(write_audit(directory, shards))
        for key in ('admitted_pairs', 'replay'):
            shards = audit_shards()
            for shard in shards:
                if key == 'replay':
                    shard[key]['rows_admitted'] = 287
                else:
                    shard[key] = 90
            with self.subTest(population=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'population differs'):
                    smoke.cache_audit(write_audit(directory, shards))


class ApprovalBoundaryTests(unittest.TestCase):
    def test_approval_refused_before_any_cloud_read(self):
        for usd, seconds in ((5, 7200), (10, 3600), (8, 7200), (10, 7201)):
            with self.subTest(usd=usd, seconds=seconds), mock.patch.object(smoke, 'aws') as call:
                with self.assertRaisesRegex(RuntimeError, 'Approval must'):
                    smoke.preflight(SimpleNamespace(approved_usd=usd, max_seconds=seconds))
                call.assert_not_called()

    def test_bound_covers_lifecycle_cleanup_storage_and_both_caps(self):
        result = smoke.budget_bound(.3, .1, .005, .1, .025, .000005)
        self.assertGreaterEqual(result['compute_hours'], 2 + 300 / 3600)
        self.assertGreaterEqual(result['ebs_hours_conservative'], result['compute_hours'])
        self.assertEqual(result['output_cap_bytes'], 512 * 1024**2)
        self.assertAlmostEqual(result['bound_usd'],
                               result['prior_bound_usd'] + result['additional_bound_usd'])
        self.assertEqual(result['prior_bound_usd'], 6.860792723718095)
        self.assertLessEqual(result['bound_usd'], 10)
        self.assertLessEqual(result['additional_bound_usd'], 3)
        for cap, prior in ((.01, smoke.PRIOR_BOUND_USD), (3, 9.99)):
            with self.subTest(cap=cap, prior=prior), \
                    mock.patch.object(smoke, 'ADDITIONAL_CAP_USD', cap), \
                    mock.patch.object(smoke, 'PRIOR_BOUND_USD', prior):
                with self.assertRaisesRegex(RuntimeError, 'bound exceeds approval'):
                    smoke.budget_bound(.3, .1, .005, .1, .025, .000005)
        with self.assertRaisesRegex(RuntimeError, 'price outside'):
            smoke.budget_bound(.51, .1, .005, .1, .025, .000005)

    def test_failed_preflight_consumes_approval_and_cannot_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_id=smoke.RUN_ID, bundle_sha256='a' * 64,
                bundle_s3_uri=f's3://{smoke.BUCKET}/{smoke.PREFIX}/code/{smoke.RUN_ID}/inputs.tar',
                provenance=smoke.PROVENANCE, state_dir=directory)
            with mock.patch.object(smoke, 'preflight', side_effect=RuntimeError('refused')) as check, \
                    mock.patch.object(smoke, 'aws') as call:
                with self.assertRaisesRegex(RuntimeError, 'refused'):
                    smoke.launch(args)
                with self.assertRaises(FileExistsError):
                    smoke.launch(args)
                self.assertEqual(check.call_count, 1)
                args.run_id = 'qwen013-alternate-id'
                with self.assertRaisesRegex(RuntimeError, 'exact approved'):
                    smoke.launch(args)
                call.assert_not_called()
            intent = json.loads((Path(directory) / f'{smoke.RUN_ID}-approval.intent.json').read_text())
            self.assertEqual(intent['client_token'], smoke.RUN_ID)

    def test_exact_new_run_approval_rejects_old_caps_retries_and_identity(self):
        approval = {'run_id': smoke.RUN_ID, 'instance_limit': 1, 'region': smoke.REGION,
            'instance_type': smoke.TYPE, 'max_seconds': 7200, 'additional_cap_usd': 3.0,
            'cumulative_cap_usd': 10.0, 'prior_conservative_bound_usd': 6.860792723718095,
            'automatic_model_or_launch_retry': False, 'instance_or_az_fallback': False,
            'fresh_pricing_and_owned_cleanup_required': True}
        smoke.verify_approval(approval)
        for key, value in (('run_id', smoke.ORIGIN_RUN_ID), ('instance_limit', 2),
                           ('max_seconds', 7201), ('additional_cap_usd', 4),
                           ('cumulative_cap_usd', 8), ('prior_conservative_bound_usd', 0),
                           ('automatic_model_or_launch_retry', True),
                           ('instance_or_az_fallback', True),
                           ('fresh_pricing_and_owned_cleanup_required', False)):
            with self.subTest(field=key), self.assertRaisesRegex(RuntimeError, 'approval mismatch'):
                smoke.verify_approval({**approval, key: value})


class AuthenticationBoundaryTests(unittest.TestCase):
    def test_source_manifest_and_current_bytes_must_all_match(self):
        pins = {path: smoke.ENGINE_PINS.get(path, 'a' * 64) for path in smoke.REQUIRED_SOURCES}
        files = {path: {'sha256': sha} for path, sha in pins.items()}
        with mock.patch.object(smoke, 'digest',
                               side_effect=lambda path: pins[path.relative_to(smoke.ROOT).as_posix()]):
            smoke.verify_sources({'source_pins': pins}, files)
            changed = copy.deepcopy(files)
            changed['tests/test_train_open_weight_lora.py']['sha256'] = 'f' * 64
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                smoke.verify_sources({'source_pins': pins}, changed)
            missing = dict(pins)
            del missing['tests/test_train_qwen_preferences.py']
            with self.assertRaisesRegex(RuntimeError, 'authentication missing'):
                smoke.verify_sources({'source_pins': missing}, files)
            wrong_engine = dict(pins)
            wrong_engine['scripts/train_qwen_preferences.py'] = 'f' * 64
            with self.assertRaisesRegex(RuntimeError, 'engine identity'):
                smoke.verify_sources({'source_pins': wrong_engine}, files)
        with mock.patch.object(smoke, 'digest', return_value='0' * 64):
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                smoke.verify_sources({'source_pins': pins}, files)

    def test_exact_allowlist_rejects_extra_missing_wrong_run_and_rebinding(self):
        row = {'path': 'scripts/example.py', 'bytes': 12, 'sha256': 'a' * 64}
        allowlist = {'required_root': str(smoke.ROOT), 'run_id': smoke.RUN_ID,
                     'file_count': 1, 'total_bytes': 12, 'files': [row]}
        files = {row['path']: row, smoke.ALLOWLIST: {'path': smoke.ALLOWLIST}}
        smoke.verify_allowlist(allowlist, files)
        for variant in ({**files, 'extra': {}}, {smoke.ALLOWLIST: {}},
                        {**files, row['path']: {**row, 'sha256': 'b' * 64}}):
            with self.subTest(files=variant), self.assertRaises(RuntimeError):
                smoke.verify_allowlist(allowlist, variant)
        for key, value in (('run_id', 'qwen013-other'), ('total_bytes', 13), ('file_count', 2)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                smoke.verify_allowlist({**allowlist, key: value}, files)

    def test_compatibility_selectors_cannot_expand_to_unbounded_test_modules(self):
        expected = list(smoke.COMPATIBILITY_TESTS)
        self.assertEqual(smoke.compatibility_selectors({'compatibility_tests': expected}), expected)
        for selectors in (expected[:1], expected + ['test_train_open_weight_lora'],
                          expected + [expected[0]], ['test_train_qwen_preferences']):
            with self.subTest(selectors=selectors), self.assertRaises(RuntimeError):
                smoke.compatibility_selectors({'compatibility_tests': selectors})

    def test_simulation_must_allow_every_exact_resource_without_missing_context(self):
        resources = ['arn:one', 'arn:two']
        response = {'EvaluationResults': [{'EvalActionName': 's3:GetObject',
            'EvalDecision': 'allowed', 'EvalResourceName': '*', 'ResourceSpecificResults': [
                {'EvalResourceName': name, 'EvalResourceDecision': 'allowed'} for name in resources]}]}
        with mock.patch.object(smoke, 'aws', return_value=response):
            smoke.simulate(smoke.PROFILE, 's3:GetObject', resources)
        variants = [{'EvaluationResults': []}, {**response, 'IsTruncated': True}]
        for mutation in ('wrong_resource', 'deny_resource', 'missing_context', 'wrong_action'):
            value = copy.deepcopy(response)
            evaluation = value['EvaluationResults'][0]
            if mutation == 'wrong_resource':
                evaluation['ResourceSpecificResults'][1]['EvalResourceName'] = 'arn:other'
            elif mutation == 'deny_resource':
                evaluation['ResourceSpecificResults'][1]['EvalResourceDecision'] = 'explicitDeny'
            elif mutation == 'missing_context':
                evaluation['ResourceSpecificResults'][0]['MissingContextValues'] = ['required']
            else:
                evaluation['EvalActionName'] = 's3:PutObject'
            variants.append(value)
        for value in variants:
            with self.subTest(value=value), mock.patch.object(smoke, 'aws', return_value=value):
                with self.assertRaises(RuntimeError):
                    smoke.simulate(smoke.PROFILE, 's3:GetObject', resources)


class PriorLinuxReuseTests(unittest.TestCase):
    def test_authenticated_source_bytes_are_reused_without_relabeling_export(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(smoke, 'ROOT', Path(directory)):
            plan = reuse_fixture(Path(directory))
            pins = [plan['cache_acceptance'], plan['cache_reuse']['recovery_verification'],
                    *plan['verified_cache_shards']]
            files = {Path(p['path']).relative_to(smoke.ROOT).as_posix(): dict(p) for p in pins}
            before = {p['path']: Path(p['path']).read_bytes() for p in pins}
            result = smoke.verify_cache_reuse(plan, files)
            self.assertFalse(result['cache_recomputed'])
            self.assertTrue(result['old_scores_reused'])
            self.assertEqual(result['origin_run_id'], smoke.ORIGIN_RUN_ID)
            self.assertFalse(result['origin_export_complete'])
            self.assertTrue(result['origin_export_errors'])
            self.assertEqual(before, {p['path']: Path(p['path']).read_bytes() for p in pins})
            for pin in pins:
                missing = dict(files)
                del missing[Path(pin['path']).relative_to(smoke.ROOT).as_posix()]
                with self.assertRaisesRegex(RuntimeError, 'not authenticated'):
                    smoke.verify_cache_reuse(plan, missing)
                path = Path(pin['path'])
                path.write_bytes(before[str(path)] + b'\n')
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    smoke.verify_cache_reuse(plan)
                path.write_bytes(before[str(path)])

    def test_extra_missing_duplicate_and_reordered_shards_fail_closed(self):
        for mutation in ('extra', 'missing', 'duplicate', 'reordered', 'wrong_path'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(smoke, 'ROOT', Path(directory)):
                plan = reuse_fixture(Path(directory))
                paths = [Path(row['path']) for row in plan['verified_cache_shards']]
                if mutation == 'extra':
                    (paths[0].parent / 'shard-007.json').write_text('{}')
                elif mutation == 'missing':
                    paths[-1].unlink()
                elif mutation == 'duplicate':
                    plan['verified_cache_shards'][-1] = plan['verified_cache_shards'][0]
                elif mutation == 'reordered':
                    plan['verified_cache_shards'].reverse()
                else:
                    plan['verified_cache_shards'][0]['path'] = str(paths[0].parent / 'other.json')
                with self.assertRaises(RuntimeError):
                    smoke.verify_cache_reuse(plan)

    def test_recovery_and_acceptance_semantics_cannot_be_repinned_away(self):
        variants = [('recovery', 'run_id', smoke.RUN_ID),
                    ('recovery', 'all_retained_files_hash_verified', False),
                    ('recovery', 'export_complete', True), ('recovery', 'export_errors', []),
                    ('acceptance', 'accepted', False), ('acceptance', 'pairs', 90),
                    ('acceptance', 'replay_rows', 287), ('acceptance', 'cache_recomputed', False),
                    ('acceptance', 'old_scores_reused', True), ('acceptance', 'shards', [])]
        for source, key, value in variants:
            with self.subTest(source=source, field=key), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(smoke, 'ROOT', Path(directory)):
                plan = reuse_fixture(Path(directory))
                pin = (plan['cache_acceptance'] if source == 'acceptance'
                       else plan['cache_reuse']['recovery_verification'])
                path = Path(pin['path'])
                record = json.loads(path.read_text())
                record[key] = value
                path.write_text(json.dumps(record))
                pin['sha256'] = smoke.digest(path)
                with self.assertRaises(RuntimeError):
                    smoke.verify_cache_reuse(plan)

    def test_wrong_reuse_policy_and_provenance_paths_are_rejected(self):
        for mutation in ('origin', 'fallback', 'acceptance_path', 'recovery_path'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(smoke, 'ROOT', Path(directory)):
                plan = reuse_fixture(Path(directory))
                if mutation == 'origin':
                    plan['cache_reuse']['origin_run_id'] = smoke.RUN_ID
                elif mutation == 'fallback':
                    plan['cache_reuse']['recompute_on_failure'] = True
                elif mutation == 'acceptance_path':
                    plan['cache_acceptance']['path'] = '/other/linux-cache-acceptance.json'
                else:
                    plan['cache_reuse']['recovery_verification']['path'] = '/other/recovery.json'
                with self.assertRaises(RuntimeError):
                    smoke.verify_cache_reuse(plan)

    def test_actual_runtime_and_engine_mismatch_stop_before_frozen_verifier(self):
        engine = {'answer_head': 'packed_supervised_tokens_v1', 'versions': smoke.RUNTIME_VERSIONS,
                  'device': 'cpu', 'dtype': 'torch.float32',
                  'code': {Path(p).name: sha for p, sha in smoke.ENGINE_PINS.items()}}
        for mutation in ('torch', 'transformers', 'peft', 'cuda', 'engine', 'versions',
                         'platform', 'architecture', 'python'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(smoke, 'ROOT', Path(directory)):
                shards = audit_shards()
                for shard in shards:
                    shard.update(engine=copy.deepcopy(engine), versions=dict(smoke.RUNTIME_VERSIONS))
                if mutation == 'engine':
                    shards[-1]['engine']['code']['train_qwen_preferences.py'] = '0' * 64
                elif mutation == 'versions':
                    shards[-1]['versions']['torch'] = 'wrong'
                plan = reuse_fixture(Path(directory), shards)
                plan_path = smoke.ROOT / smoke.PLAN
                plan_path.parent.mkdir(parents=True)
                plan_path.write_text(json.dumps(plan))
                trainer = SimpleNamespace(engine_identity=mock.Mock(return_value=engine),
                                          verify_cache_shards=mock.Mock())
                torch = SimpleNamespace(__version__=smoke.RUNTIME_VERSIONS['torch'],
                    version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False),
                    float32='torch.float32')
                modules = {'train_qwen_preferences': trainer, 'torch': torch,
                    'transformers': SimpleNamespace(__version__=smoke.RUNTIME_VERSIONS['transformers']),
                    'peft': SimpleNamespace(__version__=smoke.RUNTIME_VERSIONS['peft'])}
                if mutation in smoke.RUNTIME_VERSIONS:
                    modules[mutation].__version__ = 'wrong'
                elif mutation == 'cuda':
                    torch.version.cuda = '12.0'
                with mock.patch.dict(smoke.sys.modules, modules), \
                        mock.patch.object(smoke.sys, 'path', list(smoke.sys.path)), \
                        mock.patch.object(smoke.sys, 'platform', 'darwin' if mutation == 'platform' else 'linux'), \
                        mock.patch.object(smoke.platform, 'machine',
                                          return_value='x86_64' if mutation == 'architecture' else 'aarch64'), \
                        mock.patch.object(smoke.sys, 'version_info', (3, 13) if mutation == 'python' else (3, 12)):
                    with self.assertRaisesRegex(RuntimeError, 'runtime'):
                        smoke.validate_linux_cache(Path(plan['verified_cache_shards'][0]['path']).parent)
                trainer.verify_cache_shards.assert_not_called()


class TracedSmokeDispatchTests(unittest.TestCase):
    def test_actual_wrapper_is_guarded_recorded_and_never_retried(self):
        for mismatch in (False, True):
            with self.subTest(receipt_mismatch=mismatch), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.dict(smoke.sys.modules, {'qwen_cpu_pipeline': pipeline}):
                output = Path(directory)
                canonical = smoke.frozen_child(frozen_plan())
                outcome = {'training_children_requested': 0, 'cache_children_requested': 0,
                           'invocations': []}
                dispatched = ['/python', '-B', 'scripts/trace_qwen_preferences.py', '--trace',
                              str(output / 'model-smoke-phases.jsonl'), '--', *canonical[1:]]
                calls = []

                def unit(name, command, seconds):
                    calls.append(command)
                    self.assertEqual(name, 'model-smoke')
                    self.assertEqual(seconds, 180)
                    self.assertEqual(command, ['/python', '-B', '/guard', '--receipt',
                        str(output / 'model-smoke-supervisor.json'), '--seconds', '180',
                        '--', *dispatched])
                    receipt = {'argv': ['/python', '-B', *canonical] if mismatch else dispatched,
                        'seconds': 180, 'startup_released': True, 'stop_reason': 'child_exit',
                        'child_exit_code': 0, 'first_stop_reason': None, 'reaped': True,
                        'cleanup_state': 'completed', 'cleanup_errors': [],
                        'live_members_before_reap': [], 'elapsed_seconds': 170, 'rss_peak_bytes': 1024,
                        'samples': [{'rss_bytes': 1024, 'host': {'memory_psi': {}},
                            'cgroup': {'memory_max_bytes': 6 * 1024**3, 'swap_max_bytes': 0,
                                       'swap_current_bytes': 0, 'events': {}}}]}
                    (output / 'model-smoke-supervisor.json').write_text(json.dumps(receipt))
                    (output / 'model-smoke.log').write_text('{"weight_loading": true}\n')
                    return 0

                if mismatch:
                    with self.assertRaisesRegex(RuntimeError, 'receipt incomplete'):
                        smoke.run_model_smoke(output, canonical, unit, outcome, '/python', '/guard')
                else:
                    smoke.run_model_smoke(output, canonical, unit, outcome, '/python', '/guard')
                recorded = json.loads((output / 'smoke-outcome.json').read_text())
                self.assertEqual(recorded['training_children_requested'], 1)
                self.assertEqual(recorded['cache_children_requested'], 0)
                self.assertEqual(recorded['invocations'][0]['canonical_argv'], canonical)
                self.assertEqual(recorded['invocations'][0]['dispatched_argv'], dispatched)
                with self.assertRaisesRegex(RuntimeError, 'no retry'):
                    smoke.run_model_smoke(output, canonical, unit, outcome, '/python', '/guard')
                self.assertEqual(len(calls), 1)


class CleanupBoundaryTests(unittest.TestCase):
    def row(self, state='running'):
        return {'InstanceId': 'i-owned', 'ClientToken': smoke.RUN_ID,
                'Tags': smoke.tags(smoke.RUN_ID), 'State': {'Name': state}}

    def reconcile(self, observations, volumes=(), expected=None):
        index = 0
        now = 0
        calls = []

        def clock():
            return now

        def pause(seconds):
            nonlocal now
            now += seconds

        def discover(run):
            nonlocal index
            result = observations[min(index, len(observations) - 1)]
            index += 1
            return result

        def call(service, operation, *args):
            calls.append((service, operation, args))
            return {'Volumes': list(volumes)} if operation == 'describe-volumes' else {}

        result = smoke.reconcile_cleanup(smoke.RUN_ID, expected, 16, discover=discover,
                                         call=call, clock=clock, pause=pause)
        return result, calls

    def test_empty_eventual_discovery_never_proves_absence(self):
        for expected in (None, 'i-owned'):
            result, calls = self.reconcile([[]], expected=expected)
            self.assertEqual(result['cleanup_state'], 'cleanup_unresolved')
            self.assertIsNone(result['remaining_owned_volumes'])
            self.assertEqual(calls, [])

    def test_delayed_visibility_terminates_once_then_requires_no_ebs(self):
        result, calls = self.reconcile([[], [self.row()], [self.row('shutting-down')],
                                        [self.row('terminated')]])
        self.assertEqual(result['cleanup_state'], 'complete')
        self.assertEqual(result['launch_outcome'], 'instance_found')
        self.assertEqual(result['termination_attempts'], 1)
        self.assertEqual([operation for _, operation, _ in calls],
                         ['terminate-instances', 'describe-volumes'])
        result, _ = self.reconcile([[self.row('terminated')]], volumes=[{'VolumeId': 'vol-owned'}])
        self.assertEqual(result['cleanup_state'], 'cleanup_unresolved')

    def test_conflicting_tags_token_or_instance_never_authorizes_termination(self):
        variants = []
        row = self.row()
        row['ClientToken'] = 'another-token'
        variants.append((row, None))
        row = self.row()
        row['Tags'] = row['Tags'][:-1]
        variants.append((row, None))
        variants.append((self.row(), 'i-other'))
        for row, expected in variants:
            with self.subTest(row=row, expected=expected):
                result, calls = self.reconcile([[row]], expected=expected)
                self.assertEqual(result['cleanup_state'], 'cleanup_unresolved')
                self.assertTrue(result['errors'])
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
