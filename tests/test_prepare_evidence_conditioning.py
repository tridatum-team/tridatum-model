"""Behavioral tests for source-preserving evidence dataset preparation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_evidence_conditioning as prep


def packet(institution='가상 새봄청', title='가상 접수 지침'):
    return {'documents': [{'doc_id': 'D1', 'institution': institution, 'title': title,
                           'currency': 'FICTIONAL_SCENARIO', 'scope': '가상 절차'}],
            'evidence': [{'id': 'E1', 'doc_id': 'D1', 'kind': 'quotation', 'subject': title,
                          'quote': '담당자는 서면 신청을 접수한다. 구두 신청은 접수하지 않는다.'},
                         {'id': 'E2', 'doc_id': 'D1', 'kind': 'quotation', 'subject': title,
                          'quote': '접수한 서면 신청은 기록 담당자에게 전달한다.'}],
            'scope': '가상 수집본만; 현행성 판단 없음', 'warnings': ['가상 기관의 가상 절차이며 실제 법령이 아니다.']}


def source(ident, split, institution):
    data = packet(institution, institution + ' 접수 지침')
    target = {'paragraphs': [{'text': '담당자는 서면 신청을 접수하고 구두 신청은 접수하지 않습니다.', 'sources': ['E1']},
                             {'text': '접수한 서면 신청은 기록 담당자에게 전달합니다.', 'sources': ['E2']}]}
    return {'id': ident, 'family_id': 'family-' + ident, 'split': split, 'cohort': 'mois',
            'behavior': 'conditions', 'messages': [{'role': 'system', 'content': 'original frozen prompt'},
                {'role': 'user', 'content': prep.canonical({'question': '접수 대상과 후속 절차는 무엇인가요?', 'source_data': data})}],
            'reference_answer': prep.canonical(target), 'rubric': {'required': ['서면 신청만 접수한다.'], 'forbidden': ['구두 신청도 접수한다고 말한다.']},
            'provenance': {'institution': institution, 'doc_id': 'persistent-' + ident,
                           'family_id': 'family-' + ident}}


def originals():
    result = [source('train-a', 'train', '가상 새봄청'), source('train-b', 'train', '가상 들녘청'),
              source('dev-a', 'evaluation', '가상 마루청'), source('dev-b', 'evaluation', '가상 온빛청')]
    for split in ('train', 'evaluation'):
        result.append({'id': 'general-' + split, 'family_id': 'general-family-' + split,
                       'split': split, 'cohort': 'general', 'behavior': 'conversation',
                       'messages': [{'role': 'system', 'content': '친절하게 답하세요.'},
                                    {'role': 'user', 'content': '안녕하세요?'}],
                       'reference_answer': '안녕하세요. 무엇을 도와드릴까요?',
                       'rubric': {'required': ['인사한다.']}, 'provenance': {}})
    return result


def annotations(records):
    parent = records[0]
    return {'schema': 'trid-evidence-annotations-v1', 'author_identity': 'EvidenceCurator', 'facts': [{
        'parent_id': parent['id'], 'parent_case_sha256': prep.case_sha(parent), 'split': 'train',
        'review_status': 'needs_independent_review', 'facts': [
            {'id': 'F1', 'subject': '담당자', 'predicate': '접수한다', 'object': '서면 신청',
             'qualifiers': {'condition': '구두 신청은 접수하지 않는다.'},
             'supporting_sources': ['E1'], 'support_quote': '담당자는 서면 신청을 접수한다. 구두 신청은 접수하지 않는다.'}]}],
            'synthetic': []}


def scenario(family='fictional-review', split='development'):
    clean = packet()
    target = {'paragraphs': [{'text': '담당자는 서면 신청을 접수합니다.', 'sources': ['E1']}]}
    changed = deepcopy(clean)
    changed['evidence'][0]['quote'] = '담당자는 구두 신청을 접수한다. 서면 신청은 접수하지 않는다.'
    conflict = deepcopy(clean)
    conflict['evidence'][1]['quote'] = '담당자는 서면 신청을 접수하지 않는다.'
    missing = {'documents': [], 'evidence': [], 'scope': clean['scope'], 'warnings': clean['warnings']}
    variants = []
    for condition, data, answer, relation in [
        ('clean', clean, target, 'invariant'),
        ('changed_value_or_condition', changed, {'paragraphs': [{'text': '담당자는 구두 신청을 접수합니다.', 'sources': ['E1']}]}, 'changed_value_or_condition'),
        ('distractor', clean, target, 'invariant'),
        ('missing', missing, prep.MISSING_TARGET, 'support_removed'),
        ('conflict', conflict, {'paragraphs': [{'text': '서면 신청 접수에 관한 근거가 서로 충돌하므로 접수 여부를 단정할 수 없습니다.', 'sources': ['E1', 'E2']}]}, 'conflict'),
        ('structured', clean, target, 'invariant'),
    ]:
        variants.append({'condition': condition, 'question': '서면 신청을 접수하나요?', 'source_data': deepcopy(data),
                         'target': deepcopy(answer), 'rubric': ['제공된 가상 절차에 근거해서만 설명한다.'], 'expected_relation': relation})
    return {'family_id': family, 'split': split, 'fictional': True,
            'review_status': 'needs_independent_review', 'variants': variants}


class PreparationBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.records = originals()
        self.annotations = annotations(self.records)

    def build(self):
        return prep.build_candidate(self.records, self.annotations)

    def test_originals_immutable_and_deterministic_matrix(self):
        before = deepcopy(self.records)
        annotation_before = deepcopy(self.annotations)
        first = self.build()
        self.assertEqual(first, self.build())
        self.assertEqual(self.records, before)
        self.assertEqual(self.annotations, annotation_before)
        self.assertEqual(len(first[0]), 19)  # four four-condition parents + one overlay + two general
        self.assertEqual({m['reference']['parent_id'] for m in first[0]}, {r['id'] for r in self.records})

    def test_citation_remapping_preserves_text_and_semantic_quote_identity(self):
        matrix, _, _ = self.build()
        by_id = prep.indexed(matrix, 'matrix')
        clean, reordered = by_id['train-a::clean'], by_id['train-a::reordered']
        before = clean['reference']['target']['paragraphs']
        after = reordered['reference']['target']['paragraphs']
        self.assertEqual([p['text'] for p in before], [p['text'] for p in after])
        self.assertEqual([p['sources'] for p in after], [['E2'], ['E1']])
        old_packet = json.loads(clean['messages'][1]['content'])['source_data']
        new_packet = json.loads(reordered['messages'][1]['content'])['source_data']
        self.assertEqual(old_packet['evidence'][0]['quote'], new_packet['evidence'][1]['quote'])

    def test_distractors_same_partition_different_family_and_explicit_scope(self):
        matrix, _, _ = self.build()
        sources = prep.indexed(self.records, 'source')
        for item in matrix:
            ref = item['reference']
            if ref['condition'] != 'distractor':
                continue
            parent = sources[ref['parent_id']]
            donor = sources[ref['transformation']['donor_parent_id']]
            self.assertEqual(parent['split'], donor['split'])
            self.assertNotEqual(parent['family_id'], donor['family_id'])
            public = json.loads(item['messages'][1]['content'])
            self.assertIn(parent['provenance']['institution'], public['question'])
            self.assertIn('다른 기관이나 다른 문서', public['question'])
            self.assertEqual(len(public['source_data']['documents']), 2)
            self.assertEqual(ref['target'], json.loads(parent['reference_answer']))
            quotes = {e['quote'] for e in public['source_data']['evidence']}
            original_quotes = {e['quote'] for r in (parent, donor) for e in prep.original_packet(r)[1]['evidence']}
            self.assertLessEqual(quotes, original_quotes)

    def test_single_institution_development_uses_distinct_document_donors(self):
        development = [r for r in self.records if r['split'] == 'evaluation' and r['cohort'] != 'general']
        for index, record in enumerate(development):
            record['provenance']['institution'] = '가상 마루청'
            payload = json.loads(record['messages'][1]['content'])
            document = payload['source_data']['documents'][0]
            document['institution'] = '가상 마루청'
            document['title'] = ('가상 마루청 민원 접수 지침', '가상 마루청 기록 전달 지침')[index]
            for evidence in payload['source_data']['evidence']:
                evidence['subject'] = document['title']
            record['messages'][1]['content'] = prep.canonical(payload)
        before = deepcopy(self.records)
        matrix, _, _ = self.build()
        self.assertEqual(self.records, before)
        descendants = [m for m in matrix if m['reference']['split'] == 'development'
                       and m['reference']['row_type'] == 'evidence']
        self.assertEqual(len(descendants), len(development) * 4)
        parents = prep.indexed(development, 'development')
        for item in descendants:
            ref = item['reference']
            if ref['condition'] != 'distractor':
                continue
            donor = parents[ref['transformation']['donor_parent_id']]
            parent = parents[ref['parent_id']]
            self.assertNotEqual(parent['family_id'], donor['family_id'])
            self.assertNotEqual(parent['provenance']['doc_id'], donor['provenance']['doc_id'])
            payload = json.loads(item['messages'][1]['content'])
            documents = payload['source_data']['documents']
            self.assertEqual({d['institution'] for d in documents}, {'가상 마루청'})
            self.assertEqual(len({d['title'] for d in documents}), 2)
            self.assertIn(documents[0]['title'], payload['question'])
            self.assertNotIn(documents[1]['title'], payload['question'])
            self.assertEqual(ref['target'], json.loads(parent['reference_answer']))

    def test_donor_rejects_ambiguous_scope_even_with_different_family(self):
        parent, donor = self.records[:2]
        payload = json.loads(donor['messages'][1]['content'])
        source_document = prep.original_packet(parent)[1]['documents'][0]
        payload['source_data']['documents'][0] = deepcopy(source_document)
        donor['messages'][1]['content'] = prep.canonical(payload)
        with self.assertRaisesRegex(ValueError, 'same-split distractor'):
            self.build()

    def test_donor_rejects_same_persistent_document_with_different_labels(self):
        self.records[1]['provenance']['doc_id'] = self.records[0]['provenance']['doc_id']
        with self.assertRaisesRegex(ValueError, 'same-split distractor'):
            self.build()

    def test_no_cross_split_donor_fallback(self):
        self.records = [r for r in self.records if r['id'] != 'train-b']
        with self.assertRaisesRegex(ValueError, 'same-split distractor'):
            self.build()

    def test_missing_support_removes_all_evidence_and_citations(self):
        matrix, groups, _ = self.build()
        for item in matrix:
            if item['reference']['condition'] == 'missing':
                data = json.loads(item['messages'][1]['content'])['source_data']
                self.assertEqual(data['documents'], [])
                self.assertEqual(data['evidence'], [])
                self.assertEqual(item['reference']['target'], prep.MISSING_TARGET)
                self.assertEqual(item['reference']['review_status'], 'needs_independent_review')
        self.assertEqual(sum(g['relation'] == 'support_removed' for g in groups), 4)

    def test_fact_edges_remain_unreviewed_and_full_quotes_are_reviewable(self):
        matrix, _, review = self.build()
        item = prep.indexed(matrix, 'matrix')['train-a::structured']
        data = json.loads(item['messages'][1]['content'])['source_data']
        fact = data['evidence'][-1]
        self.assertEqual(fact['id'], 'E3')
        self.assertEqual(fact['supporting_sources'], ['E1'])
        self.assertNotIn('support_quote', fact)
        self.assertEqual(item['reference']['review_status'], 'needs_independent_review')
        self.assertEqual(review[0]['source_data'], prep.original_packet(self.records[0])[1])
        self.assertEqual(review[0]['proposed_source_data'], data)

    def test_facts_reject_stale_hash_split_bad_support_and_claimed_review(self):
        for field, value in [('parent_case_sha256', '0' * 64), ('split', 'development'),
                             ('review_status', 'independently_reviewed')]:
            with self.subTest(field=field):
                edited = deepcopy(self.annotations)
                edited['facts'][0][field] = value
                with self.assertRaises(ValueError):
                    prep.build_candidate(self.records, edited)
        for field, value in [('support_quote', '원문에 없는 문장'), ('supporting_sources', ['E99'])]:
            with self.subTest(field=field):
                edited = deepcopy(self.annotations)
                edited['facts'][0]['facts'][0][field] = value
                with self.assertRaises(ValueError):
                    prep.build_candidate(self.records, edited)

    def test_general_rows_exact_and_replay_at_least_twenty_percent(self):
        self.annotations['synthetic'] = [scenario('fictional-train', 'train')]
        matrix, _, _ = self.build()
        train, metadata = prep.training_view(matrix)
        originals_by_id = prep.indexed(self.records, 'source')
        general_metadata = [m for m in metadata if m['row_type'] == 'general']
        self.assertGreaterEqual(len(general_metadata) * 5, len(train))
        train_by_id = prep.indexed(train, 'train')
        for item in general_metadata:
            original = originals_by_id[item['parent_id']]
            row = train_by_id[item['id']]
            self.assertEqual(row['messages'], original['messages'])
            self.assertEqual(row['answer'], original['reference_answer'])
        self.assertIn('train-a::structured', train_by_id)
        self.assertEqual(sum(m['parent_id'] == 'train-b' for m in metadata), 1)
        self.assertEqual(sum(m['fictional'] for m in metadata), 6)
        self.assertEqual(prep.training_view(matrix), (train, metadata))

    def test_synthetic_whole_family_partition_and_explicit_fiction(self):
        self.annotations['synthetic'] = [scenario()]
        matrix, groups, review = self.build()
        synthetic = [m for m in matrix if m['reference']['fictional']]
        self.assertEqual(len(synthetic), 6)
        self.assertEqual({m['reference']['split'] for m in synthetic}, {'development'})
        self.assertEqual(len([g for g in groups if g['family_id'] == 'fictional-review']), 4)
        self.assertEqual(review[-1]['kind'], 'synthetic')
        self.annotations['synthetic'].append(scenario(split='train'))
        with self.assertRaisesRegex(ValueError, 'duplicate/cross-split'):
            self.build()

    def test_synthetic_without_public_fiction_label_is_rejected(self):
        self.annotations['synthetic'] = [scenario()]
        self.annotations['synthetic'][0]['variants'][0]['source_data']['warnings'] = ['실제 규정']
        with self.assertRaisesRegex(ValueError, 'fictional status'):
            self.build()

    def test_preparation_outputs_strict_public_pins_and_honest_candidate_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            annotation_path = root / 'annotations.json'
            prep.write_json(annotation_path, self.annotations)
            output = root / 'prepared'
            with patch.object(prep, 'load_sources', return_value=(self.records, {'source_caveats': ['snapshot only']}, {})):
                manifest = prep.prepare(root / 'unused-frozen.json', annotation_path, output, root=root)
            self.assertFalse(manifest['training_ready'])
            self.assertFalse(manifest['new_annotations_independently_reviewed'])
            self.assertEqual(manifest['token_budget_status'], 'not_measured')
            public = prep.read_json(output / 'evaluation-inputs.json')
            refs = prep.read_json(output / 'evaluation-references.private.json')
            self.assertTrue(all(set(item) == {'id', 'messages'} for item in public))
            self.assertEqual({r['id'] for r in public}, {r['id'] for r in refs})
            self.assertEqual(prep.read_json(output / 'source-records.private.json'), self.records)
            evaluation = prep.read_json(output / 'evaluation-manifest.json')
            for field in ('inputs', 'references'):
                bound = evaluation[field]
                self.assertEqual(prep.file_sha(output / bound['path']), bound['sha256'])
            for item in public:
                ref = next(r for r in refs if r['id'] == item['id'])
                self.assertEqual(prep.case_sha(item['messages']), ref['prompt_sha256'])
            with self.assertRaisesRegex(ValueError, 'output already exists'):
                prep.prepare(root / 'unused-frozen.json', annotation_path, output, root=root)

    def test_unknown_freeze_schema_rejected_before_author_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / 'freeze.json'
            prep.write_json(path, {'schema': 'untrusted'})
            with self.assertRaisesRegex(ValueError, 'unsupported source freeze'):
                prep.load_sources(path, root=root)

    def review_fixture(self):
        self.annotations['synthetic'] = [scenario()]
        _, _, items = self.build()
        packet = {'items': items, 'author_identity': self.annotations['author_identity']}
        digest = prep.case_sha(packet)
        judgments = []
        for item in items:
            judgment = {'id': item['id'], 'pass': True, 'reason': '주체·조건·예외·출처와 모든 변형의 정답 및 관계를 각각 검토했다.'}
            if item['kind'] == 'synthetic':
                judgment['members'] = [v['condition'] for v in item['annotation']['variants']]
            judgments.append(judgment)
        reports = [{'schema': 'trid-evidence-annotation-review-v1', 'packet_sha256': digest,
                    'reviewer': reviewer, 'judgments': deepcopy(judgments)}
                   for reviewer in ('independent-a', 'independent-b')]
        return packet, digest, reports

    def test_admission_requires_two_complete_distinct_reviewers(self):
        packet, digest, reports = self.review_fixture()
        self.assertEqual(prep.validate_annotation_reviews(packet, digest, reports),
                         ['independent-a', 'independent-b'])
        for bad in ([reports[0]], [reports[0], reports[0]]):
            with self.assertRaises(ValueError):
                prep.validate_annotation_reviews(packet, digest, bad)
        with self.assertRaisesRegex(ValueError, 'CLI author identity differs'):
            prep.validate_annotation_reviews(packet, digest, reports, author_identity='independent-a')

    def test_bound_author_self_review_rejected_without_cli_identity(self):
        packet, digest, reports = self.review_fixture()
        reports[0]['reviewer'] = packet['author_identity']
        with self.assertRaisesRegex(ValueError, 'author cannot'):
            prep.validate_annotation_reviews(packet, digest, reports)
        with self.assertRaisesRegex(ValueError, 'author cannot'):
            prep.validate_annotation_reviews(packet, digest, reports, author_identity=packet['author_identity'])
        del packet['author_identity']
        with self.assertRaisesRegex(ValueError, 'author identity missing'):
            prep.validate_annotation_reviews(packet, digest, reports)

    def test_annotations_cannot_omit_recorded_author_identity(self):
        del self.annotations['author_identity']
        with self.assertRaisesRegex(ValueError, 'recorded author identity'):
            self.build()

    def test_forged_smaller_token_proof_rejected_against_live_measurements(self):
        measurement = {'id': 'real-case', 'prompt_tokens': 900, 'answer_tokens_including_eos': 100,
                       'total_tokens': 1000, 'split': 'development', 'within_budget': True}
        recomputed = {'status': 'pass', 'context_tokens': 4096, 'max_answer_tokens': 768,
                      'enable_thinking': False, 'truncation': False, 'trainer_execution_verified': True,
                      'measurements': [measurement], 'failed_ids': []}
        stored = {**deepcopy(recomputed), 'trainer_execution_verified': False}
        prep.validate_recomputed_token_proof(stored, recomputed)
        stored['measurements'][0].update(prompt_tokens=9, answer_tokens_including_eos=1, total_tokens=10)
        with self.assertRaisesRegex(ValueError, 'authenticated live recomputation'):
            prep.validate_recomputed_token_proof(stored, recomputed)

    def test_live_proof_cannot_claim_readiness_without_trainer_or_with_overflow(self):
        for live in ({'status': 'pass', 'trainer_execution_verified': False},
                     {'status': 'blocked', 'trainer_execution_verified': True}):
            with self.subTest(live=live):
                with self.assertRaisesRegex(ValueError, 'live tokenizer and trainer'):
                    prep.validate_recomputed_token_proof({}, live)

    def test_admission_blocks_stale_partial_unknown_failed_and_duplicate_judgments(self):
        packet, digest, reports = self.review_fixture()
        for mutation in ('hash', 'missing', 'unknown', 'failed', 'duplicate', 'blank_reason'):
            with self.subTest(mutation=mutation):
                bad = deepcopy(reports)
                if mutation == 'hash':
                    bad[0]['packet_sha256'] = '0' * 64
                elif mutation == 'missing':
                    bad[0]['judgments'].pop()
                elif mutation == 'unknown':
                    bad[0]['judgments'][0]['id'] = 'unknown'
                elif mutation == 'failed':
                    bad[0]['judgments'][0]['pass'] = False
                elif mutation == 'duplicate':
                    bad[0]['judgments'].append(deepcopy(bad[0]['judgments'][0]))
                else:
                    bad[0]['judgments'][0]['reason'] = ''
                with self.assertRaises(ValueError):
                    prep.validate_annotation_reviews(packet, digest, bad)

    def test_admission_reviews_every_synthetic_member_not_passing_subset(self):
        packet, digest, reports = self.review_fixture()
        synthetic = next(j for j in reports[0]['judgments'] if 'members' in j)
        synthetic['members'].remove('conflict')
        with self.assertRaisesRegex(ValueError, 'synthetic member review incomplete'):
            prep.validate_annotation_reviews(packet, digest, reports)


if __name__ == '__main__':
    unittest.main()
