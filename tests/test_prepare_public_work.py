"""Portable pins, exclusive output and strict synthetic suite preparation."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import prepare_public_work as prep
from trid.service.public_work import TASKS

SOURCE = Path(__file__).resolve().parents[1] / 'data/public-work-v1/diagnostics.json'


def source_value():
    return json.loads(SOURCE.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def repin(manifest_path, key, artifact):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    data = artifact.read_bytes()
    manifest[key] = {'path': artifact.name, 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest()}
    write_json(manifest_path, manifest)


def development_source(count=None):
    cases = []
    for index in range(len(TASKS) if count is None else count):
        task = TASKS[index % len(TASKS)]
        general = task.startswith('general_')
        cases.append({
            'id': f'development-{index}', 'task': task, 'domain': 'fixture',
            'institution': '가상 기관', 'family_ids': [f'fixture-{index}'],
            'documents': [] if general else [{
                'id': 'D1', 'title': '가상 안내', 'institution': '가상 기관',
                'text': '접수는 월요일입니다.', 'source_kind': 'synthetic_development',
            }],
            'history': [], 'question': f'사례 {index}: 요청에 답하세요.',
            'output_contract': {'format': 'plain_text', 'citations_required': not general},
            'target': '안녕하세요.' if general else '접수는 월요일입니다. [D1]',
            'rubric': {'required_claims': ['응답'], 'forbidden_claims': [],
                       'required_citations': [] if general else ['D1'], 'calculation': None},
            'provenance': {'kind': 'synthetic_development', 'author': 'test-fixture',
                           'historically_unseen': False, 'expert_reviewed': False},
        })
    return {'schema': prep.SOURCE_SCHEMA, 'scope': 'development_holdout', 'cases': cases}


class PreparePublicWorkTests(unittest.TestCase):
    def test_development_holdout_prepare_and_verify_preserve_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = development_source()
            source_path = root / 'source.json'
            write_json(source_path, source)
            manifest = prep.prepare(source_path, root / 'suite')
            self.assertEqual(manifest['scope'], 'development_holdout')
            self.assertEqual(manifest['case_count'], len(TASKS))
            self.assertEqual(manifest['expected_tasks'], list(TASKS))
            self.assertEqual(prep.verify_manifest(root / 'suite/manifest.json'), manifest)
            self.assertTrue(all(manifest[key] is False for key in (
                'training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen')))

    def test_development_provenance_flags_and_coverage_reject_before_output(self):
        mutations = [
            lambda s: s['cases'].clear(),
            lambda s: s['cases'].pop(),
            lambda s: s['cases'][0]['provenance'].update(kind='synthetic_diagnostic'),
            lambda s: s['cases'][0]['documents'][0].update(source_kind='synthetic_diagnostic'),
            lambda s: s['cases'][0]['provenance'].update(expert_reviewed=True),
            lambda s: s['cases'][0]['provenance'].update(historically_unseen=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    source = development_source()
                    mutate(source)
                    path, output = root / 'source.json', root / f'suite-{index}'
                    write_json(path, source)
                    with self.assertRaises(ValueError):
                        prep.prepare(path, output)
                    self.assertFalse(output.exists())

    def test_development_manifest_rejects_flags_and_source_scope_count_mismatch(self):
        mutations = [lambda m: m.update(scope='diagnostic'),
                     lambda m: m.update(case_count=25),
                     lambda m: m.update(expected_tasks=list(TASKS)[:-1])]
        mutations.extend(lambda m, key=key: m.update({key: True}) for key in (
            'training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A 24-case development source isolates scope binding from count checks.
            source_path = root / 'source.json'
            write_json(source_path, development_source(24))
            manifest = prep.prepare(source_path, root / 'suite')
            path = root / 'suite/manifest.json'
            for mutate in mutations:
                changed = deepcopy(manifest)
                mutate(changed)
                write_json(path, changed)
                with self.subTest(manifest=changed), self.assertRaises(ValueError):
                    prep.verify_manifest(path)

    def test_cli_writes_verified_portable_suite_without_mutating_source(self):
        before = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'suite'
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = prep.main(['--source', str(SOURCE), '--output', str(output)])
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary['case_count'], 24)
            self.assertFalse(summary['quality_accepted'])
            manifest = prep.verify_manifest(output / 'manifest.json')
            self.assertEqual(manifest['expected_tasks'], list(TASKS))
            self.assertEqual(manifest['scope'], 'diagnostic')
            for key in ('training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen'):
                self.assertIs(manifest[key], False)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'source.private.json', 'inputs.json', 'references.private.json', 'manifest.json'})
            self.assertEqual((output / 'source.private.json').read_bytes(), before)
            for key in ('source', 'inputs', 'references'):
                pin = manifest[key]
                self.assertFalse(Path(pin['path']).is_absolute())
                data = (output / pin['path']).read_bytes()
                self.assertEqual(pin['bytes'], len(data))
                self.assertEqual(pin['sha256'], hashlib.sha256(data).hexdigest())
            rows = json.loads((output / 'inputs.json').read_text(encoding='utf-8'))
            refs = json.loads((output / 'references.private.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), len(refs))
            self.assertTrue(all(set(row) == {'id', 'messages'} for row in rows))
            self.assertEqual([row['id'] for row in rows], [ref['id'] for ref in refs])
            # The entire suite remains valid after moving it away from its source.
            moved = Path(directory) / 'relocated'
            output.rename(moved)
            self.assertEqual(prep.verify_manifest(moved / 'manifest.json'), manifest)
        self.assertEqual(SOURCE.read_bytes(), before)

    def test_bytes_are_deterministic_and_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'first', root / 'second'
            prep.prepare(SOURCE, first)
            prep.prepare(SOURCE, second)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            snapshot = {path.name: path.read_bytes() for path in first.iterdir()}
            with self.assertRaises(FileExistsError):
                prep.prepare(SOURCE, first)
            self.assertEqual(snapshot, {path.name: path.read_bytes() for path in first.iterdir()})
            empty = root / 'existing-empty'
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                prep.prepare(SOURCE, empty)
            self.assertEqual(list(empty.iterdir()), [])

    def test_duplicate_ids_prompts_counts_and_missing_task_coverage_rejected_before_output(self):
        mutations = [
            lambda s: s['cases'][1].update(id=s['cases'][0]['id']),
            lambda s: s['cases'].__setitem__(1, {**deepcopy(s['cases'][0]), 'id': 'different-id'}),
            lambda s: s['cases'].pop(),
            lambda s: s['cases'][0].update(task='scope_conditions'),
            lambda s: s.update(scope='heldout'),
            lambda s: s.update(training_ready=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    source = source_value()
                    mutate(source)
                    path, output = root / f'source-{index}.json', root / f'out-{index}'
                    write_json(path, source)
                    with self.assertRaises(ValueError):
                        prep.prepare(path, output)
                    self.assertFalse(output.exists())

    def test_private_reference_corruption_fails_even_after_file_is_repinned(self):
        mutations = [
            lambda refs: refs[0].update(id='unknown'),
            lambda refs: refs[0].update(prompt_sha256='0' * 64),
            lambda refs: refs[0]['rubric'].update(required_citations=['D999']),
            lambda refs: refs[0].update(target='Unreviewed replacement [D1]'),
            lambda refs: refs.append(deepcopy(refs[0])),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    output = root / str(index)
                    prep.prepare(SOURCE, output)
                    path = output / 'references.private.json'
                    refs = json.loads(path.read_text(encoding='utf-8'))
                    mutate(refs)
                    write_json(path, refs)
                    repin(output / 'manifest.json', 'references', path)
                    with self.assertRaisesRegex(ValueError, 'authored projections'):
                        prep.verify_manifest(output / 'manifest.json')

    def test_input_tampering_and_pin_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'suite'
            prep.prepare(SOURCE, output)
            inputs = output / 'inputs.json'
            rows = json.loads(inputs.read_text(encoding='utf-8'))
            rows[0]['messages'][-1]['content'] += ' appended'
            write_json(inputs, rows)
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                prep.verify_manifest(output / 'manifest.json')
            repin(output / 'manifest.json', 'inputs', inputs)
            with self.assertRaisesRegex(ValueError, 'authored projections'):
                prep.verify_manifest(output / 'manifest.json')
            for index, mutation in enumerate((
                lambda m: m['source'].update(path='../diagnostics.json'),
                lambda m: m['references'].update(bytes=True),
                lambda m: m.update(quality_accepted=True),
                lambda m: m.update(historically_unseen=True),
                lambda m: m.update(expected_tasks=list(TASKS)[:-1]),
                lambda m: m.update(case_count=True),
            )):
                destination = root / f'pin-{index}'
                prep.prepare(SOURCE, destination)
                path = destination / 'manifest.json'
                manifest = json.loads(path.read_text(encoding='utf-8'))
                mutation(manifest)
                write_json(path, manifest)
                with self.assertRaises(ValueError):
                    prep.verify_manifest(path)

    def test_source_pin_and_nonordinary_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'suite'
            prep.prepare(SOURCE, output)
            source = output / 'source.private.json'
            source.write_bytes(source.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                prep.verify_manifest(output / 'manifest.json')
            source.unlink()
            source.symlink_to(SOURCE)
            with self.assertRaisesRegex(ValueError, 'local file'):
                prep.verify_manifest(output / 'manifest.json')
            with self.assertRaisesRegex(ValueError, 'ordinary file'):
                prep.prepare(source, root / 'linked-input')

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for text in ('{"schema":"a","schema":"b"}', '{"cases":NaN}', '{"cases":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prep.strict_json(text)


if __name__ == '__main__':
    unittest.main()
