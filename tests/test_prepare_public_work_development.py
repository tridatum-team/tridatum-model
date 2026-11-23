"""Behavioural development admission tests using fictional in-memory fixtures only."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import unicodedata

from scripts import prepare_public_work_development as development
from trid.service.public_work import GENERAL_TASKS, TASKS, public_row


def put(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def case(batch, task, index, *, kind='synthetic_development'):
    general = task in GENERAL_TASKS
    institution = development.GENERAL_INSTITUTION if general else '가상 ' + batch + '기관'
    calculation = ({'expression': 'min(600 * 2, 1500)', 'result': '1200', 'unit': '원'}
                   if task in ('table_calculation', 'general_arithmetic') else None)
    return {'id': f'dev-{batch}-{index:02d}', 'task': task, 'domain': batch + '업무',
            'institution': institution, 'family_ids': [batch + '-' + task],
            'documents': [] if general else [{'id': 'D1', 'title': '가상 ' + batch + ' ' + task + '안내',
                'institution': institution, 'text': batch + ' ' + task + ' 전용 근거입니다.', 'source_kind': kind}],
            'history': [], 'question': batch + ' ' + task + ' 전용 질문은 무엇인가요?',
            'output_contract': {'format': 'plain_text', 'citations_required': not general},
            'target': batch + ' ' + task + ' 전용 제안 답변입니다. '
                      + ('600원 × 2와 1,500원 중 작은 금액은 1,200원입니다. ' if calculation else '')
                      + ('' if general else '[D1]'),
            'rubric': {'required_claims': [batch + ' ' + task + ' 전용 필수 주장'],
                       'forbidden_claims': [batch + ' ' + task + ' 전용 금지 주장'],
                       'required_citations': [] if general else ['D1'], 'calculation': calculation},
            'provenance': {'kind': kind, 'author': 'fixture-author-' + batch,
                           'historically_unseen': False, 'expert_reviewed': False}}


def batch(name, split):
    return {'schema': development.BATCH_SCHEMA, 'batch': name, 'split': split,
            'business_group': name + '업무군', 'institutions': ['가상 ' + name + '기관'],
            'cases': [case(name, task, index) for index, task in enumerate(TASKS, 1)]}


class DevelopmentPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.batches = self.root / 'batches'
        self.batches.mkdir()
        self.train = batch('Train', 'train')
        self.holdout = batch('Holdout', 'holdout')
        self.diagnostics = {'schema': development.SOURCE_SCHEMA, 'scope': 'diagnostic',
                            'cases': [case('Diagnostic', 'missing_evidence', 1, kind='synthetic_diagnostic')]}
        self.diagnostics_path = self.root / 'diagnostics.json'
        self.output = self.root / 'prepared'

    def save(self):
        put(self.batches / 'train.json', self.train)
        put(self.batches / 'holdout.json', self.holdout)
        put(self.diagnostics_path, self.diagnostics)

    def prepare(self):
        self.save()
        return development.prepare(self.batches, self.diagnostics_path, self.output)

    def rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.prepare()
        self.assertFalse(self.output.exists())

    def reviews(self):
        manifest = development.read_json(self.output / 'manifest.json')
        paths = []
        for reviewer in ('reviewer-a', 'reviewer-b'):
            review = {'schema': development.REVIEW_SCHEMA, 'packet_sha256': manifest['packet_sha256'],
                      'reviewer': reviewer, 'model_outputs_seen': False, 'expert_certified': False,
                      'cases': [{'id': item['id'], 'pass': True, 'reason': '문서와 제안 답변의 일치를 독립적으로 확인했습니다.'}
                                for item in self.train['cases'] + self.holdout['cases']]}
            path = self.root / (reviewer + '.json')
            put(path, review)
            paths.append(path)
        return paths

    def test_prepare_counts_pins_packet_and_human_review(self):
        manifest = self.prepare()
        self.assertEqual(manifest['counts']['train']['rows'], 15)
        self.assertEqual(manifest['counts']['holdout']['task'], dict.fromkeys(TASKS, 1))
        self.assertEqual(manifest['batch_task_counts']['Train'], dict.fromkeys(TASKS, 1))
        self.assertEqual(manifest['counts']['train']['business_group'], {'Train업무군': 15})
        self.assertEqual(manifest['counts']['train']['general_task_share'], 2 / 15)
        lengths = [len(document['text']) for item in self.train['cases'] for document in item['documents']]
        self.assertEqual(manifest['counts']['train']['document_length_chars'],
                         {'count': 13, 'min': min(lengths), 'max': max(lengths),
                          'mean': sum(lengths) / 13, 'median': sorted(lengths)[6]})
        for name, pin in manifest['artifacts'].items():
            self.assertEqual(pin, development._pin(name, (self.output / name).read_bytes()))
        packet = development.read_json(self.output / 'review-packet.json')
        self.assertEqual(manifest['packet_sha256'], development.canonical_sha(packet))
        self.assertEqual(packet['batches'], [self.holdout, self.train])
        human = (self.output / 'expert-review/holdout.md').read_text()
        template = development.read_json(self.output / 'expert-review/verdicts.template.json')
        self.assertIn('아직 검토되지 않았습니다', human)
        self.assertEqual(template['status'], 'NOT_REVIEWED')
        self.assertFalse(template['expert_reviewed'])
        for item, verdict in zip(self.holdout['cases'], template['cases']):
            self.assertIn(item['question'], human)
            self.assertIn(item['target'], human)
            self.assertEqual(verdict, {'id': item['id'], 'verdict': '', 'issue': '',
                                      'corrected_answer': '', 'reviewer': '', 'date': ''})
        for artifact in manifest['artifacts']:
            content = (self.output / artifact).read_text()
            for key in ('question', 'target'):
                self.assertNotIn(self.diagnostics['cases'][0][key], content)
        self.assertEqual(manifest['builder'], development._pin(str(Path(development.__file__).resolve()),
                                                              Path(development.__file__).read_bytes()))

    def test_holdout_private_content_never_enters_train_rows(self):
        self.prepare()
        text = (self.output / 'train.jsonl').read_text()
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual([row['id'] for row in rows], [item['id'] for item in self.train['cases']])
        for item, row in zip(self.train['cases'], rows):
            self.assertEqual(row, {'id': item['id'], 'task': 'public_work_' + item['task'], 'split': 'train',
                                   'messages': public_row(item)['messages'], 'answer': item['target']})
            self.assertNotIn('rubric', row)
        for item in self.holdout['cases']:
            self.assertNotIn(item['question'], text)
            self.assertNotIn(item['target'], text)
            for claim in item['rubric']['required_claims'] + item['rubric']['forbidden_claims']:
                self.assertNotIn(claim, text)
        suite = development.read_json(self.output / 'holdout/source.private.json')
        self.assertEqual(suite['scope'], 'development_holdout')
        self.assertEqual(suite['cases'], self.holdout['cases'])

    def test_real_trainer_loader_accepts_prepared_rows_without_model_loading(self):
        from scripts.train_open_weight_lora import load_rows
        self.train['cases'][0]['history'] = [{'role': 'user', 'content': '이전 질문입니다.'},
                                            {'role': 'assistant', 'content': '이전 응답입니다.'}]
        self.prepare()
        manifest = development.read_json(self.output / 'training-manifest.json')
        rows = load_rows(manifest['train'][0], self.output)
        self.assertEqual(len(rows), len(self.train['cases']))
        for row, item in zip(rows, self.train['cases']):
            self.assertEqual(row['messages'], public_row(item)['messages'])
            self.assertEqual(row['answer'], item['target'])
            self.assertEqual(row['adapter'], 'grounded_messages')
        self.assertEqual(manifest['valid'], [])
        self.assertFalse(manifest['training_ready'])
        self.assertFalse(manifest['expert_certified'])
        self.assertFalse(manifest['semantic_evaluated'])

    def test_shared_declared_institution_refused_even_when_unused(self):
        self.holdout['institutions'].append(self.train['institutions'][0])
        self.rejected('Split overlap \\(institutions\\)')

    def test_diagnostic_institution_refused(self):
        self.diagnostics['cases'][0]['institution'] = self.train['institutions'][0]
        self.rejected('Split overlap \\(institutions\\)')

    def test_shared_family_refused(self):
        self.diagnostics['cases'][0]['family_ids'] = self.holdout['cases'][0]['family_ids']
        self.rejected('Split overlap \\(family_ids\\)')

    def test_duplicate_case_identity_refused(self):
        self.diagnostics['cases'][0]['id'] = self.train['cases'][0]['id']
        self.rejected('Split overlap \\(case_ids\\)')

    def test_normalized_document_duplicate_refused(self):
        original = self.train['cases'][0]['documents'][0]['text']
        self.holdout['cases'][0]['documents'][0]['text'] = unicodedata.normalize('NFD', original.replace(' ', '\n\t'))
        self.rejected('Split overlap \\(documents\\)')

    def test_normalized_question_duplicate_refused(self):
        original = self.holdout['cases'][0]['question']
        self.diagnostics['cases'][0]['question'] = original.replace(' ', '\n\t')
        self.rejected('Split overlap \\(questions\\)')

    def test_document_near_duplicates_refused_for_every_split_pair(self):
        for left, right in (('train', 'holdout'), ('train', 'diagnostics'), ('holdout', 'diagnostics')):
            with self.subTest(left=left, right=right):
                sources = {'train': self.train, 'holdout': self.holdout, 'diagnostics': self.diagnostics}
                a = sources[left]['cases'][0]['documents'][0]
                b = sources[right]['cases'][0]['documents'][0]
                before = a['text'], b['text']
                shared = ' '.join('항목' + str(index) for index in range(50))
                a['text'], b['text'] = '서두 ' + shared, shared + ' 끝부분'
                self.rejected('Near-duplicate documents')
                a['text'], b['text'] = before

    def test_short_common_phrases_do_not_trigger_near_duplicate(self):
        self.train['cases'][0]['documents'][0]['text'] = '훈련 전용 공통 안내 문구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열'
        self.holdout['cases'][0]['documents'][0]['text'] = '홀드아웃 전용 공통 안내 문구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열'
        self.prepare()

    def test_calculation_mismatch_refused(self):
        item = next(item for item in self.train['cases'] if item['task'] == 'table_calculation')
        item['rubric']['calculation']['result'] = '1201'
        self.rejected('Calculation mismatch')

    def test_required_calculation_and_target_result_refused(self):
        item = next(item for item in self.holdout['cases'] if item['task'] == 'general_arithmetic')
        original = deepcopy(item)
        item['rubric']['calculation'] = None
        self.rejected('Calculation required')
        item.update(deepcopy(original))
        item['target'] = '잘못된 제안 답변은 21200원입니다.'
        self.rejected('Calculation result missing')

    def test_arithmetic_whitelist_and_exact_decimal_arithmetic(self):
        self.assertEqual(development.arithmetic('max(0.1 + 0.2, 0.25) / 3'), development.Fraction('0.1'))
        self.assertEqual(development.arithmetic('min(-2, +3) * (4 - 1)'), -6)
        for expression in ('__import__("os")', 'a + 1', '1 ** 2', '(1).__class__', 'sum(1, 2)',
                           'min(x=1)', '1e2', '1_000', '[1]', 'True', '1 / 0', 'min()', '1 // 2'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                development.arithmetic(expression)

    def test_task_coverage_is_per_split_with_general_and_long_document_batches(self):
        for current in (self.train, self.holdout):
            current['cases'] = [item for item in current['cases'] if item['task'] not in GENERAL_TASKS]
            current['cases'][0]['documents'][0]['text'] = current['batch'] + '의긴가상안내문' * 400
            general = batch(current['batch'] + 'General', current['split'])
            general['institutions'] = []
            general['cases'] = [item for item in general['cases'] if item['task'] in GENERAL_TASKS]
            put(self.batches / (general['batch'] + '.json'), general)
        manifest = self.prepare()
        self.assertEqual(manifest['batch_task_counts']['TrainGeneral'], dict.fromkeys(GENERAL_TASKS, 1))
        self.assertEqual(len(manifest['batch_task_counts']['Train']), 13)
        self.assertGreater(manifest['counts']['train']['document_length_chars']['max'], 2000)
        self.assertEqual(manifest['counts']['holdout']['general_task_share'], 2 / 15)

    def test_missing_task_refused(self):
        self.train['cases'].pop()
        self.rejected('Missing task coverage')

    def test_batch_membership_prefix_and_provenance_refused(self):
        mutations = (
            lambda value: value['cases'][0].update(id='other-prefix'),
            lambda value: value['cases'][0].update(family_ids=['other-family']),
            lambda value: value['cases'][0].update(institution='가상 미선언기관'),
            lambda value: value['cases'][0]['documents'][0].update(institution='가상 미선언기관'),
            lambda value: value['cases'][0]['provenance'].update(kind='synthetic_diagnostic'),
            lambda value: value['cases'][12].update(institution='가상 일반과제오류기관'),
        )
        original = deepcopy(self.train)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.train = deepcopy(original)
                mutation(self.train)
                self.rejected('.')

    def test_prepare_is_deterministic_and_never_overwrites(self):
        self.prepare()
        snapshot = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob('*') if path.is_file()}
        second = self.root / 'second'
        development.prepare(self.batches, self.diagnostics_path, second)
        self.assertEqual(snapshot, {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()})
        with self.assertRaises(FileExistsError):
            development.prepare(self.batches, self.diagnostics_path, self.output)
        self.assertEqual(snapshot, {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob('*') if path.is_file()})
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            development.prepare(self.batches, self.diagnostics_path, empty)

    def test_admit_copies_train_only_and_keeps_expert_status_false(self):
        self.prepare()
        reviews = self.reviews()
        before = {path: path.read_bytes() for path in (self.output / 'holdout').iterdir()}
        admitted_dir = self.root / 'admitted'
        manifest = development.admit(self.output / 'manifest.json', reviews, admitted_dir)
        self.assertEqual((admitted_dir / 'train.jsonl').read_bytes(), (self.output / 'train.jsonl').read_bytes())
        self.assertEqual(manifest['state'], 'ADMITTED_FOR_RESEARCH')
        self.assertTrue(manifest['training_ready'])
        self.assertFalse(manifest['expert_certified'])
        self.assertFalse(manifest['semantic_evaluated'])
        self.assertEqual(manifest['valid'], [])
        self.assertEqual(set(path.name for path in admitted_dir.iterdir()), {'manifest.json', 'train.jsonl'})
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(manifest['candidate_manifest'], development._pin(str((self.output / 'manifest.json').resolve()),
                                                                        (self.output / 'manifest.json').read_bytes()))
        self.assertEqual(manifest['reviews'], [development._pin(str(path.resolve()), path.read_bytes()) for path in reviews])
        from scripts.train_open_weight_lora import load_rows
        self.assertEqual(len(load_rows(manifest['train'][0], admitted_dir)), 15)

    def test_admit_refuses_incomplete_failed_duplicate_or_unbound_reviews(self):
        self.prepare()
        mutations = {
            'incomplete': lambda value: value['cases'].pop(),
            'failed': lambda value: value['cases'][0].update({'pass': False}),
            'empty reason': lambda value: value['cases'][0].update(reason='  '),
            'duplicate case': lambda value: value['cases'].append(deepcopy(value['cases'][0])),
            'duplicate reviewer': lambda value: value.update(reviewer=' reviewer-a '),
            'wrong packet': lambda value: value.update(packet_sha256='0' * 64),
            'outputs seen': lambda value: value.update(model_outputs_seen=True),
            'expert claim': lambda value: value.update(expert_certified=True),
            'not bool': lambda value: value['cases'][0].update({'pass': 1}),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                reviews = self.reviews()
                value = development.read_json(reviews[1])
                mutation(value)
                put(reviews[1], value)
                destination = self.root / 'rejected'
                with self.assertRaises(ValueError):
                    development.admit(self.output / 'manifest.json', reviews, destination)
                self.assertFalse(destination.exists())
        with self.assertRaisesRegex(ValueError, 'Exactly two'):
            development.admit(self.output / 'manifest.json', self.reviews()[:1], self.root / 'one')

    def test_admit_rejects_tampered_candidate_artifacts(self):
        self.prepare()
        reviews = self.reviews()
        with (self.output / 'train.jsonl').open('ab') as handle:
            handle.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'pin mismatch'):
            development.admit(self.output / 'manifest.json', reviews, self.root / 'admitted')
        self.assertFalse((self.root / 'admitted').exists())

    def test_cli_reports_exact_counts_and_admission_state(self):
        self.save()
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.assertEqual(development.main(['prepare', '--batches', str(self.batches), '--diagnostics',
                                               str(self.diagnostics_path), '--output', str(self.output)]), 0)
        report = json.loads(capture.getvalue())
        self.assertEqual(report['batch_task_counts']['Holdout'], dict.fromkeys(TASKS, 1))
        reviews = self.reviews()
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.assertEqual(development.main(['admit', '--prepared', str(self.output / 'manifest.json'),
                                               '--review', str(reviews[0]), '--review', str(reviews[1]),
                                               '--output', str(self.root / 'admitted')]), 0)
        self.assertEqual(json.loads(capture.getvalue())['state'], 'ADMITTED_FOR_RESEARCH')


if __name__ == '__main__':
    unittest.main()
