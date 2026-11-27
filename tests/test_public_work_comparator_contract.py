"""Artifact-free sealed comparator and public-suite behavior regressions."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.aws_eval import public_work_experiment as P
from trid.service.public_work import TASKS


class ComparatorContractTests(unittest.TestCase):
    def prior_spec(self):
        files = {name: dict(path='bundle/prior/' + name, bytes=1, sha256=sha)
                 for name, sha in P.PRIOR_RECEIPT_SHA.items()}
        files[P.PRIOR_WEIGHT_PATH] = dict(path=P.PRIOR_WEIGHT_PATH,
            bytes=P.PRIOR_WEIGHTS_BYTES, sha256=P.PRIOR_WEIGHTS_SHA)
        return dict(run_id=P.PRIOR_RUN_ID, step=530,
            model=dict(id=P.MODEL_ID, revision=P.MODEL_REVISION), files=files)

    def plan(self):
        record = dict(path='public.json', bytes=1, sha256='1' * 64)
        return dict(schema=P.SCHEMA, identity=dict(run_id='new-run'), recipe=P.recipe_for_rows(674),
            workload=dict(generation=P.GENERATION, evaluation_inputs=record, source_manifest=record,
                suites=dict(fresh=dict(manifest=record, inputs=record, case_count=180,
                    scope='development_holdout', expected_tasks=list(TASKS))),
                prior_adapter=self.prior_spec()))

    def test_prior_identity_is_original_not_current_training_step(self):
        plan = self.plan()
        prior = P.prior_adapter_identity(plan)
        self.assertEqual(prior['step'], 530)
        self.assertEqual(P.evaluation_rows(plan), 180)
        self.assertEqual(P.evaluation_arms(plan), ('baseline', 'prior_adapter', 'adapter'))
        self.assertEqual(P.generation_binding(plan, 'prior_adapter', prior)['adapter'], prior)
        with self.assertRaises((ValueError, RuntimeError)):
            P.generation_binding(plan, 'adapter', prior)
        for key in ('config', 'weights'):
            changed = copy.deepcopy(prior)
            changed[key]['sha256'] = '0' * 64
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                P.generation_binding(plan, 'prior_adapter', changed)

    def test_prior_receipts_and_model_are_anchored(self):
        spec = self.prior_spec()
        P.validate_prior_spec(spec)
        for name in P.PRIOR_FILES:
            changed = copy.deepcopy(spec)
            changed['files'][name]['sha256'] = '0' * 64
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                P.validate_prior_spec(changed)
        for key, value in [('step', 674), ('run_id', 'replacement'),
                           ('model', dict(id=P.MODEL_ID, revision='replacement'))]:
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                P.validate_prior_spec(dict(spec, **{key: value}))

    def test_bundle_and_export_include_original_raw_weights_once(self):
        plan = self.plan()
        spec = plan['workload']['prior_adapter']
        pins = P.workload_pins(plan['workload'])
        self.assertEqual(pins.count(spec['files'][P.PRIOR_WEIGHT_PATH]), 1)
        members = P.output_members(plan)
        self.assertIn('prior_adapter_artifact/' + P.PRIOR_WEIGHT_PATH, members)
        self.assertIn('prior_adapter/fresh/responses.jsonl', members)
        self.assertFalse(any(name.endswith('.gz') for name in members))
        self.assertFalse(any('/holdout/' in name or '/diagnostic/' in name for name in members))

    def test_comparator_allowance_is_narrow_and_requires_original_identity(self):
        plan = self.plan()
        self.assertEqual(P.bundle_bytes_max(plan), 256 * 1024**2)
        plain = copy.deepcopy(plan)
        del plain['workload']['prior_adapter']
        self.assertEqual(P.bundle_bytes_max(plain), 128 * 1024**2)
        for schema in ('other', 'trid-public-evidence-plan-v2'):
            with self.subTest(schema=schema), self.assertRaises((ValueError, RuntimeError)):
                P.bundle_bytes_max(dict(plan, schema=schema))
        for name in P.PRIOR_FILES:
            changed = copy.deepcopy(plan)
            changed['workload']['prior_adapter']['files'][name]['sha256'] = '0' * 64
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                P.bundle_bytes_max(changed)
        changed = copy.deepcopy(plan)
        changed['workload']['prior_adapter']['files']['extra.bin'] = dict(path='extra.bin', bytes=1, sha256='0' * 64)
        with self.assertRaises((ValueError, RuntimeError)):
            P.bundle_bytes_max(changed)

    def test_suite_count_scope_and_inputs_are_frozen(self):
        rows = [dict(id='fresh-1', messages=[dict(role='user', content='2 + 2?')])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            P.write_json(root / 'inputs.json', rows)
            record = P.pin(root / 'inputs.json', root)
            contract = dict(inputs=record, manifest=record, case_count=1,
                scope='development_holdout', expected_tasks=list(TASKS))
            manifest = dict(schema='trid-public-work-suite-v1', profile=P.PROFILE,
                case_count=1, expected_tasks=list(TASKS), scope='development_holdout',
                inputs=record, source=record, references=record, expert_reviewed=False,
                historically_unseen=False, quality_accepted=False, training_ready=False)
            P.validate_suite(manifest, rows, 'fresh', record, contract)
            for key, value in [('case_count', 2), ('scope', 'diagnostic'),
                               ('expected_tasks', ['general_dialogue']),
                               ('inputs', dict(record, sha256='0' * 64))]:
                with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                    P.validate_suite(dict(manifest, **{key: value}), rows, 'fresh', record, contract)
            for updates in (dict(expected_tasks=['general_arithmetic']), dict(scope='diagnostic')):
                changed_manifest = dict(manifest, **updates)
                changed_contract = dict(contract, **updates)
                with self.subTest(updates=updates), self.assertRaises((ValueError, RuntimeError)):
                    P.validate_suite(changed_manifest, rows, 'fresh', record, changed_contract)
                with self.subTest(updates=updates), self.assertRaises((ValueError, RuntimeError)):
                    P.suite_contracts(dict(workload=dict(suites=dict(fresh=changed_contract))))
            P.write_json(root / 'inputs.json', [dict(rows[0], id='changed')])
            with self.assertRaises((ValueError, RuntimeError)):
                P.load(root, record)

    def test_generation_binding_rejects_recipe_before_missing_workload(self):
        plan = dict(schema=P.SCHEMA, recipe=dict(P.recipe_for_rows(674), steps=1))
        with self.assertRaises((ValueError, RuntimeError)):
            P.generation_binding(plan, 'adapter', dict(config={}, weights={}, step=1))

    def test_legacy_contract_requires_both_original_input_pins(self):
        legacy = dict(generation=P.GENERATION, evaluation_inputs={}, source_manifest={}, acquisition_manifest={})
        for name, (_, _, sha) in P.LEGACY_SUITES.items():
            legacy[name + '_manifest'] = {}
            legacy[name + '_inputs'] = dict(sha256=sha)
        self.assertEqual(P.evaluation_rows(dict(workload=legacy)), 194)
        changed = copy.deepcopy(legacy)
        changed['holdout_inputs']['sha256'] = '0' * 64
        with self.assertRaises((ValueError, RuntimeError)):
            P.suite_contracts(dict(workload=changed))
        changed = dict(legacy, prior_adapter=self.prior_spec())
        with self.assertRaises((ValueError, RuntimeError)):
            P.suite_contracts(dict(workload=changed))


if __name__ == '__main__':
    unittest.main()
