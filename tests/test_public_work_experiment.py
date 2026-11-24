"""Offline sealed public-work GPU admission; never calls AWS."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tarfile
import tempfile
import unittest
import sys
import unicodedata
from unittest.mock import patch

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
                 'test_separate_whitelist_never_changes_evidence',
                 'test_export_allowlist_has_adapter_not_base', 'test_recipe_seals_two_complete_epochs',
                 'test_per_run_recipe_rejects_tampering', 'test_original_plan_recipe_and_workload',
                 'test_finalizer_requires_run_directory', 'test_finalizer_preserves_seals_and_refuses_replay',
                 'test_original_sealed_bundle_still_validates'}


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
        for field, bad in [('state', 'CANDIDATE'), ('training_ready', False), ('rows', 0),
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
            (Path(tmp) / 'train.jsonl').write_bytes((ADMITTED.parent / 'train.jsonl').read_bytes())
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
            self.assertEqual(result['recipe'], P.recipe_for_rows(P.read_json(ADMITTED)['rows']))
            self.assertEqual(P.load(ROOT, result['data']['manifest'])['train'][0]['epochs'], 2)
            self.assertEqual(len(P.load(ROOT, result['workload']['evaluation_inputs'])), 194)
            self.assertEqual(P.pin(output / 'admitted-manifest.json')['sha256'], P.pin(ADMITTED)['sha256'])
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
            (Path(tmp) / 'inputs.json').write_bytes((HOLDOUT.parent / 'inputs.json').read_bytes())
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
        recipe = P.recipe_for_rows(530)
        self.assertEqual(recipe, dict(rows=530, epochs=2, steps=530, batch_size=2,
            micro_batch_size=1, lr=5e-5, rank=16, alpha=32, dropout=0.0,
            context_tokens=4096, answer_tokens=768, seed=14, checkpoint_every=0, valid_every=0))
        self.assertEqual(recipe['rows'] * recipe['epochs'], recipe['steps'] * recipe['batch_size'])
        changed = dict(schema=P.SCHEMA, recipe=dict(recipe, epochs=1))
        with self.assertRaises((ValueError, RuntimeError)):
            P.generation_binding(changed, 'baseline')

    def test_per_run_recipe_rejects_tampering(self):
        recipe = P.recipe_for_rows(731)
        self.assertEqual(P.validate_recipe(recipe)['steps'], 731)
        self.assertEqual(recipe['rows'] * recipe['epochs'], 1462)
        for field, value in [('rows', True), ('rows', 0), ('rows', 731.0), ('steps', 530),
                             ('steps', 731.0), ('epochs', 2.0), ('batch_size', 2.0),
                             ('epochs', 1), ('batch_size', 1), ('lr', 1e-4), ('rank', 32)]:
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, RuntimeError)):
                P.validate_recipe(dict(recipe, **{field: value}))

    def test_new_admitted_rows_and_revised_private_suite_pins(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            directory = Path(tmp)
            admitted = P.read_json(ADMITTED)
            lines = (ADMITTED.parent / 'train.jsonl').read_text().splitlines()
            extra = dict(json.loads(lines[0]), id='new-run-only-training-row')
            (directory / 'train.jsonl').write_text('\n'.join(lines + [json.dumps(extra)]) + '\n')
            admitted['rows'] += 1
            train = P.pin(directory / 'train.jsonl')
            admitted['train'][0].update(bytes=train['bytes'], sha256=train['sha256'])
            P.write_json(directory / 'manifest.json', admitted)
            suite_dir = directory / 'holdout'
            suite_dir.mkdir()
            suite = P.read_json(HOLDOUT)
            suite['references']['sha256'] = 'a' * 64
            P.write_json(suite_dir / 'manifest.json', suite)
            (suite_dir / 'inputs.json').write_bytes((HOLDOUT.parent / 'inputs.json').read_bytes())
            result = P.prepare_projection(directory / 'manifest.json', directory / 'projection',
                root=ROOT, sources=sorted(P.SOURCE_NAMES), acquisition_manifest=ACQUISITION,
                holdout_manifest=suite_dir / 'manifest.json', diagnostic_manifest=DIAGNOSTIC)
            self.assertEqual(result['recipe'], P.recipe_for_rows(admitted['rows']))
            self.assertEqual(P.load(ROOT, result['data']['admission'])['training_rows'], admitted['rows'])
            self.assertEqual(P.load(ROOT, result['data']['manifest'])['rows'], admitted['rows'])
            self.assertEqual(result['workload']['holdout_manifest']['sha256'],
                             P.pin(suite_dir / 'manifest.json')['sha256'])
            plan = dict(result, schema=P.SCHEMA, identity={'run_id': 'new-run'})
            adapter = dict(config={}, weights={}, step=result['recipe']['steps'])
            self.assertEqual(P.generation_binding(plan, 'adapter', adapter)['adapter'], adapter)
            with self.assertRaises((ValueError, RuntimeError)):
                P.generation_binding(plan, 'adapter', dict(adapter, step=530))

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS Unicode filesystem normalization')
    def test_absolute_unicode_review_pins_accept_equivalent_root_spelling(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            directory = Path(tmp)
            manifest = P.read_json(ADMITTED)
            for review in manifest['reviews']:
                path = Path(review['path'])
                if not path.is_absolute():
                    path = ROOT / path
                review['path'] = unicodedata.normalize('NFC', str(path.resolve()))
            P.write_json(directory / 'manifest.json', manifest)
            (directory / 'train.jsonl').write_bytes((ADMITTED.parent / 'train.jsonl').read_bytes())
            result = P.prepare_projection(directory / 'manifest.json', directory / 'projection',
                root=Path(unicodedata.normalize('NFC', str(ROOT))), sources=sorted(P.SOURCE_NAMES),
                acquisition_manifest=Path(unicodedata.normalize('NFC', str(ACQUISITION))),
                holdout_manifest=HOLDOUT, diagnostic_manifest=DIAGNOSTIC)
            self.assertEqual(result['recipe']['rows'], manifest['rows'])

    def test_original_plan_recipe_and_workload(self):
        path = ROOT / 'runs/public-work-gpu-20261123/plan.json'
        if not path.is_file():
            self.skipTest('Local original plan is not present')
        plan = P.read_json(path)
        self.assertEqual(P.validate_recipe(plan['recipe']), P.recipe_for_rows(530))
        P.validate_workload(plan, {p['path']: p for p in plan['bundle']['files']})
        binding = P.generation_binding(plan, 'baseline')
        self.assertEqual(binding['recipe_sha256'], P.object_sha(plan['recipe']))

    def test_original_sealed_bundle_still_validates(self):
        path = ROOT / 'runs/public-work-gpu-20261123/plan.json'
        if not path.is_file():
            self.skipTest('Local original plan is not present')
        plan = P.read_json(path)
        bundle = ROOT / plan['bundle']['local_path']
        if not bundle.is_file():
            self.skipTest('Local original sealed bundle is not present')
        self.assertEqual(P.pin(bundle)['sha256'], plan['bundle']['sha256'])
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(bundle) as archive:
            root = Path(tmp)
            expected = {record['path'] for record in plan['bundle']['files']}
            self.assertEqual(set(archive.getnames()), expected)
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                target = P.safe_path(root, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    target.write_bytes(stream.read())
            # Historical bytes remain authenticated. Expiry is disabled only for
            # this offline compatibility check, never for launch authorization.
            P.validate_resources(root, plan, active=False)
            result = P.validate_data(root, plan)
            self.assertEqual(len(result['rows']), 530)

    def test_finalizer_requires_run_directory(self):
        from scripts.aws_eval import public_work_finalize_plan as finalize
        with self.assertRaises(SystemExit):
            finalize.main([])

    def test_finalizer_preserves_seals_and_refuses_replay(self):
        from scripts.aws_eval import public_work_finalize_plan as finalize
        (ROOT / 'runs').mkdir(exist_ok=True)  # absent in a clean checkout
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
            run = Path(tmp)
            state = run / 'state'
            state.mkdir()
            reg = dict(schema='trid-public-work-prerequisites-v1', identity={'run_id': 'synthetic'},
                       iam={key: {'sealed': key} for key in launch.ROLE_KEYS})
            P.write_json(run / 'registration.json', reg)
            reg_pin = P.pin(run / 'registration.json', ROOT)
            P.write_json(run / 'cloud-evidence.json', {'observations': {}})
            fields = dict(schema=P.SCHEMA, network={}, cost={'cap_usd': 30},
                          workload={}, authorization={'sha256': 'a' * 64}, limits={},
                          ebs_kms_key_arn='synthetic-key', absolute_termination_deadline='2099-01-01T00:00:00Z',
                          checked_at='2026-01-01T00:00:00Z')
            provisional = dict(registration=reg_pin, plan_fields=fields,
                cloud_evidence=P.pin(run / 'cloud-evidence.json', ROOT),
                observations_expire_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
            P.write_json(run / 'provisional-plan.json', provisional)
            resources = {key: {'inventory': reg['iam'][key]} for key in launch.ROLE_KEYS}
            resources.update(bucket_lifecycle_configuration={'Rules': []},
                             security_group={'GroupId': 'sg-synthetic'})
            P.write_json(state / 'owned-prerequisites.json',
                         dict(registration_sha256=reg_pin['sha256'], identity=reg['identity'], resources=resources))
            P.write_json(state / 'bundle-prepared.json', {'sealed': 'bundle'})
            with patch.object(launch, 'validate_registration'), patch.object(launch, 'validate_final') as validate, \
                    patch.object(launch, 'lifecycle_merge', return_value={'Rules': []}):
                with self.assertRaises((ValueError, RuntimeError)):
                    finalize.finalize(run, state, 'b' * 64, root=ROOT)
                provisional['observations_expire_at'] = '2000-01-01T00:00:00Z'
                P.write_json(run / 'provisional-plan.json', provisional)
                with self.assertRaises((ValueError, RuntimeError)):
                    finalize.finalize(run, state, reg_pin['sha256'], root=ROOT)
                self.assertFalse((run / 'plan.json').exists())
                provisional['observations_expire_at'] = (
                    datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
                P.write_json(run / 'provisional-plan.json', provisional)
                result = finalize.finalize(run, state, reg_pin['sha256'], root=ROOT)
                validate.assert_called_once()
                self.assertEqual(result['cloud_mutations'], 0)
                plan = P.read_json(run / 'plan.json')
                self.assertEqual(plan['checked_at'], fields['checked_at'])
                self.assertEqual(plan['bundle'], {'sealed': 'bundle'})
                self.assertEqual(plan['network']['security_group_id'], 'sg-synthetic')
                self.assertEqual(P.read_json(run / 'approval.json')['plan_sha256'], result['plan']['sha256'])
                with self.assertRaises((ValueError, RuntimeError)):
                    finalize.finalize(run, state, reg_pin['sha256'], root=ROOT)


if __name__ == '__main__':
    unittest.main()
