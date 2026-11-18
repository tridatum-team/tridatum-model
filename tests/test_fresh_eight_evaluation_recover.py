"""Synthetic pure-helper regressions, never authenticated recovery receipts."""
import copy
import hashlib
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import fresh_eight_evaluation as E
import fresh_eight_evaluation_recover as R


KNOWN_FAILURE = dict(
    automatic_retry=False,
    completed_rows_retained=True,
    elapsed_seconds=5787.645240746,
    message="[Errno 13] Permission denied: '/opt/trid-mois-eight-cuda-eval-20261116-134815/output/model-acquisition.json'",
    quality_accepted=False,
    status='failed',
    type='PermissionError',
)


def record(name, body):
    return dict(path=name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def write_archive(path, members):
    """Build tiny adversarial archives, not an evaluation export."""
    with tarfile.open(path, 'w') as archive:
        for member, body in members:
            if isinstance(member, str):
                member = tarfile.TarInfo(member)
                member.size = len(body)
            archive.addfile(member, io.BytesIO(body))


class RecoveryArchiveTests(unittest.TestCase):
    def test_reads_exact_member_bytes_without_extracting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'synthetic.tar'
            files = {'nested/data.bin': b'\x00\xff\n', 'empty.json': b''}
            write_archive(path, files.items())
            self.assertEqual(R.archive_files(path), files)
            self.assertEqual(R.archive_files(path, [record(n, b) for n, b in files.items()]), files)
            self.assertEqual(set(root.iterdir()), {path})

    def test_rejects_unsafe_and_noncanonical_member_names(self):
        names = ('../escape', '/absolute', 'nested/../../escape', './entry',
                 'nested/../entry', 'nested/./entry', 'nested//entry', 'entry/')
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'synthetic.tar'
                write_archive(path, [(name, b'not evidence')])
                with self.assertRaises(ValueError):
                    R.archive_files(path)

    def test_rejects_duplicate_member_even_when_bytes_match(self):
        for second in (b'first', b'changed'):
            with self.subTest(second=second), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'synthetic.tar'
                write_archive(path, [('entry', b'first'), ('entry', second)])
                with self.assertRaises(ValueError):
                    R.archive_files(path)

    def test_rejects_links_devices_fifo_and_sparse_members(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                     tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'synthetic.tar'
                member = tarfile.TarInfo('unsafe')
                member.type = kind
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = '../outside'
                write_archive(path, [(member, b'')])
                with self.assertRaises(ValueError):
                    R.archive_files(path)

    def test_requires_exact_record_names_sizes_hashes_and_coverage(self):
        body = b'synthetic bytes'
        exact = record('entry', body)
        bad_records = (
            [],
            [dict(exact, path='other')],
            [dict(exact, bytes=len(body) + 1)],
            [dict(exact, sha256='0' * 64)],
            [exact, record('extra', b'')],
            [exact, exact],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.tar'
            write_archive(path, [('entry', body)])
            for records in bad_records:
                with self.subTest(records=records), self.assertRaises(ValueError):
                    R.archive_files(path, records)


class RecoveryFailureTests(unittest.TestCase):
    def test_accepts_only_the_known_post_generation_failure_without_mutating_it(self):
        failure = copy.deepcopy(KNOWN_FAILURE)
        R.validate_failure(failure)
        self.assertEqual(failure, KNOWN_FAILURE)

    def test_rejects_changed_failure_class_path_status_or_flags(self):
        changes = (
            ('type', 'RuntimeError'), ('type', 'OSError'),
            ('message', KNOWN_FAILURE['message'].replace('model-acquisition.json', 'evaluation-worker.json')),
            ('message', KNOWN_FAILURE['message'] + ' '),
            ('status', 'complete'), ('automatic_retry', True),
            ('completed_rows_retained', False), ('quality_accepted', True),
            ('automatic_retry', 0), ('completed_rows_retained', 1),
            ('quality_accepted', 0),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                R.validate_failure(dict(KNOWN_FAILURE, **{key: value}))

    def test_rejects_missing_and_additional_failure_fields(self):
        for key in KNOWN_FAILURE:
            failure = dict(KNOWN_FAILURE)
            del failure[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                R.validate_failure(failure)
        with self.assertRaises(ValueError):
            R.validate_failure(dict(KNOWN_FAILURE, recovered=True))

    def test_requires_exact_finite_numeric_elapsed_time(self):
        for value in (float('nan'), float('inf'), -float('inf'), True, False,
                      None, '5787.645240746', 5787, 5787.645240747, -1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                R.validate_failure(dict(KNOWN_FAILURE, elapsed_seconds=value))


class RecoveryPinTests(unittest.TestCase):
    def test_binds_portable_record_to_unchanged_local_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            path = root / 'nested/data.bin'
            path.parent.mkdir()
            body = b'synthetic local bytes'
            path.write_bytes(body)
            pinned = R.pin(path, root)
            self.assertEqual(pinned, record('nested/data.bin', body))
            self.assertEqual(R.bound(root, pinned), path)
            path.write_bytes(b'changed local content')
            with self.assertRaises(ValueError):
                R.bound(root, pinned)

    def test_rejects_parent_traversal_and_absolute_record_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'root'
            root.mkdir()
            outside = Path(temp) / 'outside'
            body = b'not inside root'
            outside.write_bytes(body)
            for name in ('../outside', str(outside)):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    R.bound(root, record(name, body))

    def test_rejects_wrong_size_and_digest_even_for_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            path = root / 'data.bin'
            body = b'synthetic bytes'
            path.write_bytes(body)
            exact = record(path.name, body)
            for pinned in (dict(exact, bytes=len(body) + 1), dict(exact, sha256='0' * 64)):
                with self.subTest(pinned=pinned), self.assertRaises(ValueError):
                    R.bound(root, pinned)

    def test_rejects_symlinked_evidence_and_parent_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            directory = root / 'real'
            directory.mkdir()
            path = directory / 'entry'
            path.write_bytes(b'synthetic')
            file_link = root / 'linked-entry'
            file_link.symlink_to(path)
            parent_link = root / 'linked-directory'
            parent_link.symlink_to(directory, target_is_directory=True)
            for linked in (file_link, parent_link / 'entry'):
                with self.subTest(path=linked), self.assertRaises(ValueError):
                    R.pin(linked, root)


class RecoveryJsonTests(unittest.TestCase):
    def test_loads_plain_json_without_tolerating_duplicate_or_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.json'
            path.write_bytes(b'{"synthetic": true}')
            self.assertEqual(R.load(path), {'synthetic': True})
            for raw in (b'{"a": 1, "a": 2}', b'{"nested": {"a": 1, "a": 2}}',
                        b'{"elapsed": NaN}', b'{"elapsed": Infinity}',
                        b'{"elapsed": -Infinity}', b'{'):
                path.write_bytes(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    R.load(path)


def failed_unit_fixture():
    """Only the parser's inputs: not a launch, admission, or recovered receipt."""
    plan = dict(identity=dict(run_id=E.RUN_ID), limits=dict(E.LIMITS))
    release = dict(launch_time=1000, deadline=19900, gpu=dict(uuid='GPU-synthetic'))
    base = '/opt/trid-' + E.RUN_ID
    unit = dict(Result='exit-code', ExecMainStatus='1', ControlGroup='',
        MemoryMax=str(E.LIMITS['rss_bytes']), MemorySwapMax='0', DevicePolicy='closed',
        Environment=' '.join((
            'CUDA_VISIBLE_DEVICES=GPU-synthetic', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
            'HF_DATASETS_OFFLINE=1', 'AWS_EC2_METADATA_DISABLED=true', 'PYTHONDONTWRITEBYTECODE=1',
            'TOKENIZERS_PARALLELISM=false', 'OMP_NUM_THREADS=6', 'MKL_NUM_THREADS=6',
            'HF_HOME=' + base + '/scratch/cache', 'TMPDIR=' + base + '/scratch/tmp',
            'TRID_OWNED_RUN=' + E.RUN_ID)),
        User='995', Group='985', LoadState='loaded', ActiveState='failed', SubState='failed',
        InvocationID='a' * 32)
    completion = dict(error='RuntimeError: Native stage failed: evaluation: ' + repr(unit),
        runtime_unit=dict(invocation='b' * 32),
        phase_attempt=dict(phase='evaluation', unit=E.RUN_ID + '-evaluation',
            work_deadline=19600, started_at=1100, registered_seconds=18600,
            effective_seconds=6000, phase_deadline=7100))
    return completion, plan, release, unit


class RecoveryFailedUnitTests(unittest.TestCase):
    def test_parses_bounded_failed_unit_without_mutating_inputs(self):
        completion, plan, release, _ = failed_unit_fixture()
        before = copy.deepcopy((completion, plan, release))
        R.parse_failed_unit(completion, plan, release)
        self.assertEqual((completion, plan, release), before)

    def test_rejects_untyped_wrong_phase_or_nonliteral_error(self):
        completion, plan, release, unit = failed_unit_fixture()
        errors = (
            'Native stage failed: evaluation: ' + repr(unit),
            'PermissionError: Native stage failed: evaluation: ' + repr(unit),
            'RuntimeError: Native stage failed: smoke: ' + repr(unit),
            'RuntimeError: Native stage failed: evaluation: dict()',
            'RuntimeError: Native stage failed: evaluation: []',
            'RuntimeError: Native stage failed: evaluation: None',
            'RuntimeError: Native stage failed: evaluation: {',
            'RuntimeError: Native stage failed: evaluation: ' + repr(unit) + '\nextra',
        )
        for error in errors:
            with self.subTest(error=error), self.assertRaises(ValueError):
                R.parse_failed_unit(dict(completion, error=error), plan, release)

    def test_rejects_changed_unit_state_limits_identity_or_environment(self):
        completion, plan, release, original = failed_unit_fixture()
        changes = (
            ('Result', 'success'), ('ExecMainStatus', '0'), ('ExecMainStatus', 1),
            ('ControlGroup', '/still-running'), ('MemoryMax', 'infinity'),
            ('MemorySwapMax', '1'), ('DevicePolicy', 'auto'),
            ('User', '0'), ('Group', '0'), ('LoadState', 'not-found'),
            ('ActiveState', 'active'), ('SubState', 'running'),
            ('InvocationID', ''), ('InvocationID', 'A' * 32),
            ('InvocationID', 'a' * 31), ('InvocationID', 'b' * 32),
            ('Environment', original['Environment'].replace('GPU-synthetic', 'GPU-other')),
            ('Environment', original['Environment'].replace('HF_HUB_OFFLINE=1', 'HF_HUB_OFFLINE=0')),
            ('Environment', original['Environment'] + ' EXTRA_SETTING=1'),
            ('Environment', original['Environment'] + ' HF_HUB_OFFLINE=1'),
        )
        for key, value in changes:
            unit = dict(original, **{key: value})
            changed = dict(completion, error='RuntimeError: Native stage failed: evaluation: ' + repr(unit))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                R.parse_failed_unit(changed, plan, release)

    def test_rejects_missing_or_additional_unit_properties(self):
        completion, plan, release, original = failed_unit_fixture()
        units = [dict(original, Unexpected='value')]
        for key in original:
            unit = dict(original)
            del unit[key]
            units.append(unit)
        for unit in units:
            changed = dict(completion, error='RuntimeError: Native stage failed: evaluation: ' + repr(unit))
            with self.subTest(keys=sorted(unit)), self.assertRaises(ValueError):
                R.parse_failed_unit(changed, plan, release)

    def test_rejects_unbounded_mistyped_or_misattributed_phase_attempt(self):
        completion, plan, release, _ = failed_unit_fixture()
        changes = (
            ('phase', 'runtime'), ('unit', 'unowned-evaluation'),
            ('registered_seconds', 18601), ('registered_seconds', '18600'),
            ('effective_seconds', 0), ('effective_seconds', True),
            ('effective_seconds', 5787), ('effective_seconds', 18601),
            ('effective_seconds', 6000.5),
            ('started_at', 999), ('started_at', float('nan')), ('started_at', True),
            ('phase_deadline', 1100), ('phase_deadline', 7099),
            ('phase_deadline', 7131), ('phase_deadline', 19601),
            ('work_deadline', 19601), ('work_deadline', float('inf')),
        )
        for key, value in changes:
            changed = copy.deepcopy(completion)
            changed['phase_attempt'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                R.parse_failed_unit(changed, plan, release)
        changed = copy.deepcopy(completion)
        changed['phase_attempt']['extra'] = 'not registered'
        with self.assertRaises(ValueError):
            R.parse_failed_unit(changed, plan, release)


class RecoveryFactoringTests(unittest.TestCase):
    ORIGINAL = b"""from pathlib import Path
EXPECTED = 2

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_mode(root, mode, plan):
    folder = Path(root) / mode
    require(not (folder / 'failure.json').exists(), 'Failure evidence forbids completion')
    require(plan['count'] == EXPECTED, 'Content differs')
    return plan['count']

def validate_completion(root, plan):
    return validate_mode(root, 'evaluation', plan)
"""
    RECOVERY = b"""from pathlib import Path
EXPECTED = 2

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_mode_artifacts(root, mode, plan):
    \"\"\"Content authentication does not assert native success.\"\"\"
    folder = Path(root) / mode
    require(plan['count'] == EXPECTED, 'Content differs')
    return plan['count']

def validate_completion_artifacts(root, plan):
    return validate_mode_artifacts(root, 'evaluation', plan)

def validate_mode(root, mode, plan):
    require(not (Path(root) / mode / 'failure.json').exists(), 'Failure evidence forbids completion')
    return validate_mode_artifacts(root, mode, plan)

def validate_completion(root, plan):
    require(not (Path(root) / 'evaluation' / 'failure.json').exists(), 'Failure evidence forbids completion')
    return validate_completion_artifacts(root, plan)
"""

    def test_accepts_only_content_preserving_artifact_factoring(self):
        R.authenticate_factoring(self.ORIGINAL, self.RECOVERY)

    def test_rejects_content_constant_import_and_shared_helper_drift(self):
        changes = (
            (b"plan['count'] == EXPECTED", b"plan['count'] >= EXPECTED"),
            (b"require(plan['count'] == EXPECTED, 'Content differs')", b'pass'),
            (b'EXPECTED = 2', b'EXPECTED = 3'),
            (b'from pathlib import Path', b'from pathlib import PurePath as Path'),
            (b'raise ValueError(message)', b'raise RuntimeError(message)'),
            (b"return validate_mode_artifacts(root, 'evaluation', plan)",
             b"return validate_mode(root, 'evaluation', plan)"),
            (b'def validate_completion_artifacts(', b'def omitted_completion_artifacts('),
        )
        for before, after in changes:
            with self.subTest(change=before), self.assertRaises(ValueError):
                R.authenticate_factoring(self.ORIGINAL, self.RECOVERY.replace(before, after))

    def test_rejects_missing_duplicate_or_changed_historical_failure_gate(self):
        gate = b"    require(not (folder / 'failure.json').exists(), 'Failure evidence forbids completion')\n"
        for replacement in (b'', gate + gate, gate.replace(b'failure.json', b'other.json')):
            with self.subTest(gate=replacement), self.assertRaises(ValueError):
                R.authenticate_factoring(self.ORIGINAL.replace(gate, replacement), self.RECOVERY)


class RecoveryTemporaryPathTests(unittest.TestCase):
    """Exercise real extraction/import with a symlinked OS temporary directory.

    Admission is deliberately stopped by sentinel exceptions, never represented
    by synthetic successful recovery receipts.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.real_tmp = self.root / 'real-tmp'
        self.real_tmp.mkdir()
        self.link_tmp = self.root / 'linked-tmp'
        self.link_tmp.symlink_to(self.real_tmp, target_is_directory=True)

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(R.canonical(value))
        return R.pin(path, self.root)

    def test_authentication_materializes_all_views_under_resolved_tmpdir(self):
        source = b"""from pathlib import Path
p = Path(__file__)
assert p == p.resolve()
temporary = p.parents[3]
assert (temporary / 'output/marker').read_bytes() == b'unchanged output'
assert (temporary / 'recovery/scripts/aws_eval/marker').read_bytes() == b'unchanged code'
raise RuntimeError('synthetic original import reached')
"""
        bundle_path = self.root / 'bundle.tar'
        write_archive(bundle_path, [(R.SOURCES[1], source)])
        bundle_pin = R.pin(bundle_path, self.root)
        plan = dict(bundle=dict(local_path=bundle_pin['path'], bytes=bundle_pin['bytes'],
            sha256=bundle_pin['sha256'], files=[record(R.SOURCES[1], source)]))
        export_path = self.root / 'export.tar'
        write_archive(export_path, [('marker', b'unchanged output')])
        registration = dict(
            anchors={'plan.json': self.save('plan.json', plan),
                     'registration.json': self.save('registration.json', {}),
                     'approval.json': self.save('approval.json', {}),
                     'cloud-state/export.tar': R.pin(export_path, self.root)},
            source_bundle=bundle_pin)
        code = {R.SOURCES[1]: b'# synthetic unused profile\n',
                'scripts/aws_eval/marker': b'unchanged code'}
        with patch.dict(os.environ, TMPDIR=str(self.link_tmp)), patch.object(tempfile, 'tempdir', None), \
             patch.object(R, 'source_bound_failure'), patch.object(R, 'authenticate_factoring'):
            with self.assertRaisesRegex(RuntimeError, 'synthetic original import reached'):
                R.authenticate(self.root, registration, self.root, code)
        self.assertEqual(list(self.real_tmp.iterdir()), [])

    def test_registered_dispatch_imports_actual_archived_bytes_under_resolved_tmpdir(self):
        source = b"""from pathlib import Path
def _validate_receipt(root, output, plan, plan_sha):
    p = Path(__file__)
    assert p == p.resolve()
    assert p.read_bytes().endswith(b'# archived dispatch marker\\n')
    assert (p.parent / 'marker').read_bytes() == b'unchanged code'
    raise RuntimeError('synthetic archived dispatch reached')
# archived dispatch marker
"""
        archive = self.root / 'source.tar'
        members = [(R.SOURCES[0], source), ('scripts/aws_eval/marker', b'unchanged code')]
        write_archive(archive, members)
        code = R.archive_files(archive, [record(name, body) for name, body in members])
        registration_pin = self.save('state/registration.json', {'synthetic': 'not admitted'})
        self.save('state/' + R.RECEIPT_NAME, dict(registration=registration_pin))
        with patch.dict(os.environ, TMPDIR=str(self.link_tmp)), patch.object(tempfile, 'tempdir', None), \
             patch.object(R, 'validate_registration', return_value=({}, self.root / 'state', code)):
            with self.assertRaisesRegex(RuntimeError, 'synthetic archived dispatch reached'):
                R.validate_recovery(self.root, self.root / 'state/output', {}, 'unused')
        self.assertEqual(list(self.real_tmp.iterdir()), [])

    def test_external_symlink_parent_remains_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe extraction parent'):
            R.materialize(self.link_tmp / 'external-output', {'entry': b'unchanged'})


if __name__ == '__main__':
    unittest.main()
