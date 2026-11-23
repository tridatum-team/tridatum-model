"""Offline sealed public-work GPU admission; never calls AWS."""
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.aws_eval import public_work_experiment as P
from scripts.aws_eval import public_evidence_experiment as evidence
from scripts.aws_eval import fresh_eight_launch as launch
from scripts import evaluate_public_model as cpu

ROOT = Path(__file__).resolve().parents[1]
ADMITTED = ROOT / 'runs/public-work-development-20261123/admitted/manifest.json'
HOLDOUT = ROOT / 'runs/public-work-development-20261123/prepared-v2/holdout/manifest.json'
DIAGNOSTIC = ROOT / 'runs/public-model-development-20261121T060754Z/suite-reviewed/manifest.json'
ACQUISITION = ROOT / 'runs/mois-ordered-models-20261113T062501Z/qwen8-native-acquisition.json'


ARTIFACT_FREE = {'test_generation_is_exact_cpu_requested_policy',
                 'test_separate_whitelist_never_changes_evidence'}


class PublicWorkContractTests(unittest.TestCase):
    def setUp(self):
        # The sealed inputs are local run artifacts that are never published to Git.
        if (self._testMethodName not in ARTIFACT_FREE
                and not all(path.is_file() for path in (ADMITTED, HOLDOUT, DIAGNOSTIC, ACQUISITION))):
            self.skipTest('Local admitted run artifacts are not present')

    def projection(self, output, admitted=ADMITTED):
        return P.prepare_projection(admitted, output, root=ROOT, sources=sorted(P.SOURCE_NAMES),
            acquisition_manifest=ACQUISITION, holdout_manifest=HOLDOUT, diagnostic_manifest=DIAGNOSTIC)

    def test_generation_is_exact_cpu_requested_policy(self):
        self.assertEqual(P.GENERATION, cpu.requested_policy(512, 4096))
        self.assertFalse(P.GENERATION['do_sample'])
        self.assertFalse(P.GENERATION['template_kwargs']['enable_thinking'])

    def test_separate_whitelist_never_changes_evidence(self):
        self.assertTrue(P.is_public({'schema': P.SCHEMA}))
        self.assertFalse(evidence.is_public({'schema': P.SCHEMA}))
        self.assertFalse(P.is_public({'schema': evidence.SCHEMA}))
        self.assertEqual(evidence.RECIPE['steps'], 615)
        self.assertEqual(evidence.RECIPE['epochs'], 1)
        self.assertEqual(launch.registration_plan_schema({'schema': 'trid-public-work-prerequisites-v1'}), P.SCHEMA)
        with self.assertRaises((ValueError, RuntimeError)):
            launch.registration_plan_schema({'schema': 'unregistered-public-work'})

    def test_admission_rejects_state_profile_count_and_review_changes(self):
        manifest = P.read_json(ADMITTED)
        train = {k: manifest['train'][0][k] for k in ('path', 'bytes', 'sha256')}
        P.validate_admitted(manifest, train)
        for field, bad in [('state', 'CANDIDATE'), ('training_ready', False), ('rows', 529),
                           ('schema', 'other'), ('profile', 'public_evidence_v1'), ('reviews', [])]:
            altered = dict(manifest, **{field: bad})
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                P.validate_admitted(altered, train)

    def test_modified_admitted_manifest_refused_before_projection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            path = Path(tmp) / 'changed.json'
            manifest = P.read_json(ADMITTED)
            manifest['rows'] = 529
            P.write_json(path, manifest)
            with self.assertRaises((ValueError, RuntimeError)):
                self.projection(Path(tmp) / 'projection', path)
            self.assertFalse((Path(tmp) / 'projection').exists())

    def test_modified_training_rows_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_bytes(ADMITTED.read_bytes())
            lines = (ADMITTED.parent / 'train.jsonl').read_bytes().splitlines(keepends=True)
            (Path(tmp) / 'train.jsonl').write_bytes(b''.join(lines[:-1]))
            with self.assertRaises((ValueError, RuntimeError)):
                self.projection(Path(tmp) / 'projection', path)
            self.assertFalse((Path(tmp) / 'projection').exists())


    def test_projection_and_tar_exclude_all_private_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            output = Path(tmp) / 'projection'
            result = self.projection(output)
            self.assertFalse(result['cloud_authorized'])
            self.assertEqual(result['recipe'], P.RECIPE)
            self.assertEqual(P.load(ROOT, result['data']['manifest'])['train'][0]['epochs'], 2)
            self.assertEqual(len(P.load(ROOT, result['workload']['evaluation_inputs'])), 194)
            self.assertEqual(P.pin(output / 'admitted-manifest.json')['sha256'], P.ADMITTED_SHA)
            files = [*result['data'].values(), *P.workload_pins(result['workload'])]
            files.extend(P.load(ROOT, result['workload']['source_manifest'])['files'])
            names = {r['path'] for r in files}
            self.assertFalse(any('private' in n or '/reviews/' in n for n in names))
            self.assertEqual({n for n in names if n.endswith('.jsonl')}, {result['data']['train']['path']})
            archive = Path(tmp) / 'public.tar'
            with tarfile.open(archive, 'w') as stream:
                for row in files:
                    stream.add(P.bound(ROOT, row), arcname=row['path'], recursive=False)
            with tarfile.open(archive) as stream:
                self.assertEqual(set(stream.getnames()), names)
                self.assertFalse(any('references.private' in n or 'source.private' in n for n in stream.getnames()))
            for name in ('holdout', 'diagnostic'):
                suite = P.load(ROOT, result['workload'][name + '_manifest'])
                self.assertIn('references', suite)  # metadata pins allowed, never private file bytes
                public = P.load(ROOT, result['workload'][name + '_inputs'])
                self.assertTrue(all(set(r) == {'id', 'messages'} for r in public))

    def test_private_fields_and_wrong_suite_counts_refused(self):
        row = {'id': 'one', 'messages': [{'role': 'user', 'content': 'question'}]}
        with self.assertRaises((ValueError, RuntimeError)):
            P.validate_public_rows([dict(row, target='private')], 1)
        manifest = P.read_json(HOLDOUT)
        rows = P.read_json(HOLDOUT.parent / 'inputs.json')
        with self.assertRaises((ValueError, RuntimeError)):
            P.validate_suite(manifest, rows[:-1], 'holdout', manifest['inputs'])
        with self.assertRaises((ValueError, RuntimeError)):
            P.validate_suite(dict(manifest, targets=['private']), rows, 'holdout', manifest['inputs'])

    def test_modified_suite_manifest_cannot_smuggle_private_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            changed = Path(tmp) / 'manifest.json'
            manifest = dict(P.read_json(HOLDOUT), targets=['private'])
            P.write_json(changed, manifest)
            with self.assertRaises((ValueError, RuntimeError)):
                P.prepare_projection(ADMITTED, Path(tmp) / 'projection', root=ROOT,
                    sources=sorted(P.SOURCE_NAMES), acquisition_manifest=ACQUISITION,
                    holdout_manifest=changed, diagnostic_manifest=DIAGNOSTIC)
            self.assertFalse((Path(tmp) / 'projection').exists())

    def test_export_allowlist_has_adapter_not_base(self):
        members = P.output_members()
        self.assertIn('training/adapter/adapter_model.safetensors', members)
        self.assertIn('training/provenance.json', members)
        self.assertNotIn('model/model-00001-of-00005.safetensors', members)
        self.assertFalse(any('private' in p for p in members))
        self.assertEqual({n for n in members if n.endswith('.safetensors')},
                         {'training/adapter/adapter_model.safetensors'})
        for arm in ('baseline', 'adapter'):
            for suite in ('holdout', 'diagnostic'):
                self.assertIn(f'{arm}/{suite}/responses.jsonl', members)
                self.assertIn(f'{arm}/{suite}/runtime.json', members)
                self.assertIn(f'{arm}/{suite}/completion.json', members)

    def test_recipe_seals_two_complete_epochs(self):
        self.assertEqual(P.RECIPE, dict(rows=530, epochs=2, steps=530, batch_size=2,
            micro_batch_size=1, lr=5e-5, rank=16, alpha=32, dropout=0.0,
            context_tokens=4096, answer_tokens=768, seed=14, checkpoint_every=0, valid_every=0))
        self.assertEqual(P.RECIPE['rows'] * P.RECIPE['epochs'], P.RECIPE['steps'] * P.RECIPE['batch_size'])
        changed = dict(schema=P.SCHEMA, recipe=dict(P.RECIPE, epochs=1))
        with self.assertRaises((ValueError, RuntimeError)):
            P.generation_binding(changed, 'baseline')


if __name__ == '__main__':
    unittest.main()
