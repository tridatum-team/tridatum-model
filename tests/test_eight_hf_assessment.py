"""Synthetic-only boundaries for local blinded assessment; no protected suites."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aws_eval import eight_hf_assessment as assessment


GENERATION = {'enable_thinking': False, 'do_sample': False, 'num_beams': 1, 'seed': 14,
              'max_new_tokens': 768, 'context_tokens': 3072, 'dtype': 'bfloat16',
              'backend': 'transformers-cuda'}


def case():
    packet = {'documents': [{'doc_id': 'd', 'title': '규정'}],
              'evidence': [{'id': 'E1', 'doc_id': 'd', 'quote': '기한은 3일이다.'}]}
    return {'id': 'synthetic', 'messages': [{'role': 'system', 'content': 'Return JSON.'},
            {'role': 'user', 'content': json.dumps({'question': '기한?', 'source_data': packet}, ensure_ascii=False)}]}


def output(item):
    return {'id': item['id'], 'messages': item['messages'],
            'prompt_sha256': assessment.object_sha(item['messages']),
            'rendered_prompt_sha256': 'a' * 64, 'prompt_token_ids': [1, 2],
            'output_token_ids': [3, 4], 'generated_tokens': 2,
            'terminal_stop_token_id': 4, 'termination_reason': 'eos',
            'generation': GENERATION,
            'response_text': '{"paragraphs":[{"text":"기한은 3일이다.","sources":["E1"]}]}'}


def judgment(passed=True):
    return {'semantic_pass': passed, 'format_pass': True, 'critical_errors': [],
            'reason': 'Synthetic supported statement.'}


class PathContainmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def report(self, root):
        path = root / 'reviews' / 'reviewer.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"synthetic":true}\n')
        return path

    def assert_round_trip(self, path, root):
        entry = assessment.pin(path, root)
        self.assertEqual(entry, {'path': 'reviews/reviewer.json',
                                'sha256': assessment.digest(path), 'bytes': path.stat().st_size})
        self.assertTrue((root / entry['path']).samefile(path))
        self.assertTrue(assessment.bound(root, entry).samefile(path))
        return entry

    def test_ordinary_containment(self):
        root = self.base / 'root'
        self.assert_round_trip(self.report(root), root)

    def test_unicode_spellings_of_same_directory(self):
        nfc, nfd = self.base / '\uac00', self.base / '\u1100\u1161'
        self.assertNotEqual(str(nfc), str(nfd))
        original = self.report(nfc)
        if not nfd.exists() or not nfc.samefile(nfd):
            self.skipTest('temporary filesystem does not alias NFC/NFD directory spellings')
        alias = nfd / 'reviews' / 'reviewer.json'
        self.assertTrue(original.samefile(alias))
        first = self.assert_round_trip(original.resolve(), nfd.resolve())
        second = self.assert_round_trip(alias.resolve(), nfc.resolve())
        self.assertEqual(first, second)

    def test_distinct_unicode_directories_are_not_conflated(self):
        nfc, nfd = self.base / '\uac00', self.base / '\u1100\u1161'
        original = self.report(nfc)
        if nfd.exists() and nfc.samefile(nfd):
            self.skipTest('temporary filesystem aliases NFC/NFD directory spellings')
        other = self.report(nfd)
        self.assertFalse(nfc.samefile(nfd))
        self.assertEqual(original.read_bytes(), other.read_bytes())
        for path, own_root, wrong_root in ((original, nfc, nfd), (other, nfd, nfc)):
            entry = self.assert_round_trip(path, own_root)
            with self.assertRaisesRegex(ValueError, 'outside repository'):
                assessment.pin(path, wrong_root)
            outside = dict(entry, path=str(Path('..') / own_root.name / entry['path']))
            with self.assertRaisesRegex(ValueError, 'outside repository'):
                assessment.bound(wrong_root, outside)

    def test_siblings_and_prefix_sharing_roots_remain_outside(self):
        root = self.base / 'root'
        self.report(root)
        for sibling in (self.base / 'sibling', self.base / 'root-other'):
            outside = self.report(sibling)
            entry = assessment.pin(outside, sibling)
            with self.assertRaisesRegex(ValueError, 'outside repository'):
                assessment.pin(outside, root)
            escaped = dict(entry, path=str(Path('..') / sibling.name / entry['path']))
            with self.assertRaisesRegex(ValueError, 'outside repository'):
                assessment.bound(root, escaped)

    def test_symlink_escape_is_rejected(self):
        root = self.base / 'root'
        self.report(root)
        outside = self.report(self.base / 'outside')
        link = root / 'escaped.json'
        link.symlink_to(outside)
        entry = dict(assessment.pin(outside, self.base / 'outside'), path=link.name)
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            assessment.pin(link, root)
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            assessment.bound(root, entry)

    def test_same_leaf_inode_does_not_prove_root_ancestry(self):
        root = self.base / 'root'
        original = self.report(root)
        outside = self.base / 'outside-hardlink.json'
        outside.hardlink_to(original)
        self.assertTrue(original.samefile(outside))
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            assessment.pin(outside, root)
        entry = dict(assessment.pin(original, root), path='../outside-hardlink.json')
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            assessment.bound(root, entry)


class AmendmentTests(unittest.TestCase):
    def test_effective_generation_keeps_training_and_answer_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            design = {'run_id': 'synthetic', 'generation': GENERATION,
                      'recipe': {'context_tokens': 3072, 'answer_tokens': 768}}
            assessment.save(run / 'experiment-design.json', design)
            amendment = {'schema': 'trid-eight-execution-amendment-v1', 'run_id': 'synthetic',
                         'base_design': assessment.pin(run / 'experiment-design.json', root),
                         'frozen_before_outputs': True, 'training_recipe_unchanged': True,
                         'training_context_tokens': 3072, 'answer_tokens_unchanged': 768,
                         'input_bytes_unchanged': True, 'no_trimming_or_drops': True,
                         'generation': dict(GENERATION, context_tokens=4096)}
            path = run / 'amendment.json'
            assessment.save(path, amendment)
            self.assertEqual(assessment.effective_generation(root, run, assessment.pin(path, root)),
                             dict(GENERATION, context_tokens=4096))
            altered = copy.deepcopy(amendment)
            altered['generation']['max_new_tokens'] = 512
            bad = run / 'invalid-answer-change.json'
            assessment.save(bad, altered)
            with self.assertRaisesRegex(ValueError, 'pre-output scope'):
                assessment.effective_generation(root, run, assessment.pin(bad, root))
            stale = dict(assessment.pin(path, root), sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                assessment.effective_generation(root, run, stale)


class StructuralTests(unittest.TestCase):
    def test_actual_production_citation_and_numeric_contract(self):
        item = case()
        generated = output(item)
        self.assertTrue(assessment.structural_result(item, generated, GENERATION)['pass'])
        for text, reason in [('{"paragraphs":[{"text":"기한은 9일이다.","sources":["E1"]}]}',
                              'unsupported_numeric_value'),
                             ('{"paragraphs":[{"text":"기한은 3일이다.","sources":["E2"]}]}',
                              'invalid_citations')]:
            generated['response_text'] = text
            actual = assessment.structural_result(item, generated, GENERATION)
            self.assertFalse(actual['pass'])
            self.assertEqual(actual['reason'], reason)

    def test_missing_length_and_changed_prompt_fail(self):
        item = case()
        self.assertFalse(assessment.structural_result(item, None, GENERATION)['pass'])
        for updates in ({'termination_reason': 'length'}, {'context_truncated': True},
                        {'terminal_stop_token_id': None}, {'prompt_sha256': '0' * 64},
                        {'generated_tokens': 9}, {'response_text': ''}):
            self.assertFalse(assessment.structural_result(item, dict(output(item), **updates), GENERATION)['pass'])

    def test_structure_does_not_claim_entailment(self):
        item = case()
        generated = output(item)
        generated['response_text'] = '{"paragraphs":[{"text":"아무 조치도 허용되지 않는다.","sources":["E1"]}]}'
        actual = assessment.structural_result(item, generated, GENERATION)
        self.assertTrue(actual['pass'])
        self.assertIn('not_entailment', actual['reason'])


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapping = {'rows': {'opaque1': {'case_id': 'one', 'arm': 'base'},
                                 'opaque2': {'case_id': 'one', 'arm': 'step-000012'}},
                        'output_pins': {}, 'arms': ['base', 'step-000012'], 'case_ids': ['one']}
        self.packet = {'schema': 'trid-eight-blind-packet-v1', 'stage': 'dev', 'instructions': 'Review.',
                       'mapping_commitment': assessment.object_sha(self.mapping),
                       'rows': [{'sample_id': 'opaque1', 'structural': {'pass': False}},
                                {'sample_id': 'opaque2', 'structural': {'pass': True}}]}
        self.packet_path = self.root / 'packet.json'
        assessment.save(self.packet_path, self.packet)
        self.mapping_path = self.root / 'mapping.json'
        assessment.save(self.mapping_path, dict(self.mapping, packet_sha256=assessment.digest(self.packet_path)))
        self.reports = []
        for reviewer in ('A', 'B'):
            report = {'schema': 'trid-eight-blind-review-v1', 'packet_sha256': assessment.digest(self.packet_path),
                      'reviewer_id': reviewer, 'independent': True, 'blinded': True,
                      'judgments': {'opaque1': judgment(), 'opaque2': judgment()}}
            path = self.root / (reviewer + '.json')
            assessment.save(path, report)
            self.reports.append(path)

    def test_review_cannot_override_structural_failure(self):
        _, _, resolved, disagreements = assessment.resolve_reviews(self.packet_path, self.mapping_path, self.reports)
        self.assertFalse(resolved[0]['joint_pass'])
        self.assertTrue(resolved[1]['joint_pass'])
        self.assertEqual(disagreements, [])

    def test_missing_denominator_and_non_boolean_fail(self):
        report = assessment.load(self.reports[0])
        del report['judgments']['opaque1']
        with self.assertRaisesRegex(ValueError, 'denominator'):
            assessment.validate_report(report, self.packet, assessment.digest(self.packet_path))
        report = assessment.load(self.reports[0])
        report['judgments']['opaque1']['semantic_pass'] = 1
        with self.assertRaisesRegex(ValueError, 'boolean'):
            assessment.validate_report(report, self.packet, assessment.digest(self.packet_path))
        report = assessment.load(self.reports[0])
        report['judgments']['opaque1']['arm'] = 'step-000012'
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            assessment.validate_report(report, self.packet, assessment.digest(self.packet_path))

    def test_disagreement_only_third_and_mapping_commitment(self):
        report = assessment.load(self.reports[1])
        report['judgments']['opaque2'] = judgment(False)
        changed = self.root / 'B-changed.json'
        assessment.save(changed, report)
        reports = [self.reports[0], changed]
        with self.assertRaisesRegex(ValueError, 'third reviewer'):
            assessment.resolve_reviews(self.packet_path, self.mapping_path, reports)
        third = dict(report, reviewer_id='C', judgments={'opaque2': judgment(True)})
        third_path = self.root / 'third.json'
        assessment.save(third_path, third)
        _, _, resolved, disagreements = assessment.resolve_reviews(self.packet_path, self.mapping_path, reports, third_path)
        self.assertEqual(disagreements, ['opaque2'])
        self.assertTrue(resolved[1]['joint_pass'])
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            assessment.resolve_reviews(self.packet_path, self.mapping_path, self.reports, third_path)
        tampered = assessment.load(self.mapping_path)
        tampered['rows']['opaque1']['arm'] = 'step-000012'
        tampered_path = self.root / 'tampered.json'
        assessment.save(tampered_path, tampered)
        with self.assertRaisesRegex(ValueError, 'commitment'):
            assessment.resolve_reviews(self.packet_path, tampered_path, self.reports)

    def test_third_packet_excludes_agreed_rows_and_votes(self):
        report = assessment.load(self.reports[1])
        report['judgments']['opaque2'] = judgment(False)
        changed = self.root / 'B-disagreement.json'
        assessment.save(changed, report)
        result = assessment.adjudication(self.packet_path, [self.reports[0], changed], self.root / 'third')
        self.assertEqual(result['disagreements'], 1)
        packet = assessment.load(self.root / 'third/adjudication-packet.json')
        self.assertEqual([r['sample_id'] for r in packet['rows']], ['opaque2'])
        self.assertNotIn('judgments', packet)
        self.assertNotIn('arm', packet['rows'][0])


class SelectionTests(unittest.TestCase):
    def fixture(self):
        results = [{'case_id': str(i), 'arm': arm, 'behavior': assessment.BEHAVIORS[i // 2],
                    'joint_pass': i < 6, 'critical_errors': []}
                   for arm in assessment.ARMS for i in range(8)]
        ledger = {'checkpoints': [{'step': step, 'valid_loss_by_task': {'mois_grounded_explanation': loss}}
                                 for step, loss in [(12, 0.2), (24, 0.1), (36, 0.1)]]}
        return results, ledger

    def test_loss_then_step_tie_break_and_no_selection(self):
        results, ledger = self.fixture()
        _, selected = assessment.dev_aggregate(results, ledger)
        self.assertEqual(selected, 24)
        for row in results:
            if row['arm'] != 'base':
                row['critical_errors'] = ['Invented obligation.']
        _, selected = assessment.dev_aggregate(results, ledger)
        self.assertIsNone(selected)

    def test_behavior_regression_disqualifies_higher_total(self):
        results, ledger = self.fixture()
        for row in results:
            if row['arm'] == 'step-000024':
                row['joint_pass'] = row['case_id'] != '0'
        aggregate, selected = assessment.dev_aggregate(results, ledger)
        self.assertEqual(aggregate['candidates'][1]['joint_pass_count'], 7)
        self.assertEqual(selected, 36)

    def test_nan_loss_rejected(self):
        results, ledger = self.fixture()
        ledger['checkpoints'][0]['valid_loss_by_task']['mois_grounded_explanation'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid dev loss'):
            assessment.dev_aggregate(results, ledger)


class PairedTests(unittest.TestCase):
    gates = {'frozen_before_outputs': True, 'supported_refusal_count': 3,
             'required_supported_refusal_case_ids': ['mois-0', 'mois-1', 'mois-11']}

    def fixture(self):
        return [{'case_id': f'{cohort}-{i}', 'arm': arm, 'cohort': cohort,
                 'joint_pass': not (cohort == 'mois' and arm == 'base' and i == 2),
                 'critical_errors': [], 'supported_refusal': cohort == 'mois' and i in (0, 1, 11),
                 'category': str(i // 3), 'source_clusters': ['family'] if cohort == 'mois' else []}
                for cohort in ('general', 'mois') for arm in ('base', 'step-000024') for i in range(12)]

    def test_paired_gain_with_separate_gates(self):
        result = assessment.paired_aggregate(self.fixture(), ['base', 'step-000024'], self.gates)
        self.assertTrue(result['diagnostic_improvement'])
        self.assertEqual(result['cohorts']['mois']['paired_gains'], ['mois-2'])
        self.assertEqual(result['cohorts']['general']['denominator'], 12)
        self.assertFalse(result['broad_accuracy_claim'])

    def test_regression_and_refusal_failure_block_claim(self):
        results = self.fixture()
        for row in results:
            if row['case_id'] == 'mois-0' and row['arm'] != 'base':
                row['joint_pass'] = False
        result = assessment.paired_aggregate(results, ['base', 'step-000024'], self.gates)
        self.assertFalse(result['diagnostic_improvement'])
        self.assertFalse(result['cohorts']['mois']['gate_pass'])
        self.assertEqual(result['cohorts']['mois']['denominator'], 12)

    def test_missing_row_and_arm_dependent_reference_fail(self):
        results = self.fixture()
        with self.assertRaisesRegex(ValueError, 'denominator'):
            assessment.paired_aggregate(results[:-1], ['base', 'step-000024'], self.gates)
        results[-1]['supported_refusal'] = False
        with self.assertRaisesRegex(ValueError, 'cannot differ'):
            assessment.paired_aggregate(results, ['base', 'step-000024'], self.gates)

    def test_all_three_frozen_gap_cases_required(self):
        results = self.fixture()
        for row in results:
            if row['case_id'] == 'mois-11':
                row['joint_pass'] = False
        result = assessment.paired_aggregate(results, ['base', 'step-000024'], self.gates)
        self.assertEqual(result['cohorts']['mois']['after_pass'], 11)
        self.assertFalse(result['cohorts']['mois']['gate_pass'])
        self.assertFalse(result['diagnostic_improvement'])


if __name__ == '__main__':
    unittest.main()
