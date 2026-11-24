"""Consumer tests with synthetic receipts only; no inference or semantic oracle."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_public_work as evaluator
from trid.service.public_work import TASKS, public_row, reference_row


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return evaluator.artifact_pin(path)


def case(identifier, task):
    general = task.startswith('general_')
    return {'id': identifier, 'task': task, 'domain': 'general' if general else 'synthetic_work',
            'institution': '가상 기관', 'family_ids': ['synthetic-' + identifier],
            'documents': [] if general else [{'id': 'D1', 'title': '가상 진단 안내', 'institution': '가상 기관',
                                             'text': '접수는 월요일입니다.', 'source_kind': 'synthetic_diagnostic'}],
            'history': [], 'question': identifier + (': 인사해 주세요.' if general else ': 접수일을 알려 주세요.'),
            'output_contract': {'format': 'plain_text', 'citations_required': not general},
            'target': '안녕하세요.' if general else '접수는 월요일입니다. [D1]',
            'rubric': {'required_claims': ['자연스러운 인사' if general else '월요일 접수'],
                       'forbidden_claims': [], 'required_citations': [] if general else ['D1'], 'calculation': None},
            'provenance': {'kind': 'synthetic_diagnostic', 'author': 'test-fixture',
                           'historically_unseen': False, 'expert_reviewed': False}}


class PublicWorkEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.authored = [case('work', 'fact_extraction'), case('general', 'general_dialogue'),
                         case('work2', 'grounded_explanation')]
        covered = {item['task'] for item in self.authored}
        self.authored.extend(case(task, task) for task in TASKS if task not in covered)
        self.authored.extend(case(f'extra-{index}', 'fact_extraction') for index in range(9))
        self.set_suite('diagnostic')
        self.paths = {name: self.root / name for name in ('base', 'candidate')}
        for name in self.paths:
            self.make_arm(name)

    def set_suite(self, scope):
        kind = 'synthetic_development' if scope == 'development_holdout' else 'synthetic_diagnostic'
        for item in self.authored:
            item['provenance']['kind'] = kind
            for document in item['documents']:
                document['source_kind'] = kind
        self.rows = [public_row(item) for item in self.authored]
        self.refs = [reference_row(item, row) for item, row in zip(self.authored, self.rows)]
        source = self.suite_pin('source.private.json', {'schema': 'trid-public-work-source-v1',
                               'scope': scope, 'cases': self.authored})
        inputs = self.suite_pin('inputs.json', self.rows)
        refs = self.suite_pin('references.private.json', self.refs)
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = {'schema': 'trid-public-work-suite-v1', 'profile': 'public_work_v1',
                         'scope': scope, 'source': source, 'inputs': inputs, 'references': refs,
                         'case_count': len(self.rows), 'expected_tasks': list(TASKS), 'training_ready': False,
                         'quality_accepted': False, 'expert_reviewed': False, 'historically_unseen': False}
        put(self.manifest_path, self.manifest)

    def suite_pin(self, name, value):
        return dict(put(self.root / name, value), path=name)

    def make_arm(self, name, model='fixture/model', token=1):
        directory = self.paths[name]
        directory.mkdir(exist_ok=True)
        tokenizer_pin = put(directory / 'tokenizer.json', {'fixture_token': token})
        tokenizer_config_pin = put(directory / 'tokenizer_config.json', {'chat_template': str(token)})
        model_pin = put(directory / 'model-manifest.json', {'repo_id': model, 'revision': 'fixture-revision',
                        'files': [tokenizer_pin, tokenizer_config_pin]})
        requested = {'profile': 'public_work_greedy_v1', 'max_new_tokens': 16, 'context_limit': 128,
                     'do_sample': False, 'num_beams': 1, 'seed': 0, 'seed_policy': 'constant',
                     'mode': 'evaluation', 'template_kwargs': {'enable_thinking': False}}
        runtime = {'schema': 'public_work_hf_runtime_v1', 'suite_manifest': evaluator.artifact_pin(self.manifest_path),
                   'suite_scope': self.manifest['scope'],
                   'source': dict(self.manifest['source'], path=str(self.root / 'source.private.json')),
                   'inputs': dict(self.manifest['inputs'], path=str(self.root / 'inputs.json')),
                   'model_manifest': model_pin,
                   'model_identity': {'repo_id': model, 'revision': 'fixture-revision'},
                   'tokenizer_identity': {'files': [tokenizer_pin, tokenizer_config_pin],
                                          'template_sha256': hashlib.sha256(str(token).encode()).hexdigest()},
                   'generation_policy': requested, 'adapter': None, 'packages': {}, 'resources': {}}
        put(directory / 'runtime.json', runtime)
        records = []
        for row, reference in zip(self.rows, self.refs):
            rendered = 'official fixture template ' + str(token) + json.dumps(row['messages'], ensure_ascii=False)
            config = {'max_new_tokens': 16, 'do_sample': False, 'num_beams': 1, 'eos_token_id': 9}
            records.append({'id': row['id'], 'generation_receipt': 'public_work_hf_v1',
                            'prompt_sha256': evaluator.object_sha(row['messages']), 'response': reference['target'],
                            'raw_output': reference['target'], 'finish_reason': 'eos', 'error': None,
                            'input_tokens': 2, 'output_tokens': 2, 'prompt_token_ids': [token, 2],
                            'output_token_ids': [3, 9], 'rendered_prompt': rendered,
                            'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                            'template_kwargs': {'enable_thinking': False, 'add_generation_prompt': True},
                            'generation_config': config,
                            'generation_policy': {'effective_config_sha256': evaluator.object_sha(config),
                                                  'generation_mode': 'greedy_search'},
                            'context_limit': 128, 'context_truncated': False, 'truncated_output': False,
                            'seed': 0, 'mode': 'evaluation', 'wall_seconds': 0.25})
        self.replace_records(name, records)

    def records(self, name):
        return [json.loads(line) for line in (self.paths[name] / 'responses.jsonl').read_text().splitlines()]

    def replace_records(self, name, records, **completion_changes):
        directory = self.paths[name]
        responses = directory / 'responses.jsonl'
        responses.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in records))
        expected = [row['id'] for row in self.rows]
        observed = list(dict.fromkeys(item['id'] for item in records))
        missing = [identifier for identifier in expected if identifier not in observed]
        completion = {'schema': 'public_work_hf_completion_v1', 'expected_ids': expected,
                      'observed_ids': observed, 'missing_ids': missing,
                      'complete': not missing and all(item['error'] is None for item in records),
                      'worker_exit_code': 0, 'guard_failure': None,
                      'responses': evaluator.artifact_pin(responses),
                      'runtime': evaluator.artifact_pin(directory / 'runtime.json')}
        completion.update(completion_changes)
        put(directory / 'completion.json', completion)

    def change_runtime(self, name, change):
        path = self.paths[name] / 'runtime.json'
        value = json.loads(path.read_text())
        change(value)
        put(path, value)
        self.replace_records(name, self.records(name))

    def packet(self):
        return evaluator.build_blind_packet(self.manifest_path, self.paths, seed='fixed',
                    manifest_sha256=evaluator.artifact_pin(self.manifest_path)['sha256'],
                    arm_sha256={name: evaluator.artifact_pin(path / 'completion.json')['sha256']
                                for name, path in self.paths.items()})

    def judgments(self, packet, mapping):
        return {'schema': evaluator.JUDGMENT_SCHEMA, 'packet_sha256': mapping['packet_sha256'],
                'cases': [{'response_id': answer['response_id'], 'semantic_pass': True,
                           'critical_error': False, 'reason': 'The answer addresses the request and does not invent unsupported facts.'}
                          for item in packet['cases'] for answer in item['responses'] if answer['judgment_required']]}

    def test_development_holdout_load_packet_and_report_preserve_scope(self):
        self.authored = self.authored[:len(TASKS)]
        self.set_suite('development_holdout')
        for name in self.paths:
            self.make_arm(name)
        manifest, rows, refs, _ = evaluator.load_suite(self.manifest_path)
        self.assertEqual(manifest['scope'], 'development_holdout')
        self.assertEqual(len(rows), len(TASKS))
        self.assertEqual({ref['task'] for ref in refs.values()}, set(TASKS))
        packet, mapping = self.packet()
        report = evaluator.score_packet(packet, mapping)
        for artifact in (packet, mapping, report):
            self.assertEqual(artifact['suite_scope'], 'development_holdout')
        self.assertEqual(report['scope'], 'observed_development_holdout_not_fresh_gold')
        self.assertFalse(report['quality_accepted'])
        self.assertFalse(report['historically_unseen'])
        self.assertFalse(report['expert_reviewed'])
        self.change_runtime('candidate', lambda value: value.update(suite_scope='diagnostic'))
        with self.assertRaisesRegex(ValueError, 'Runtime suite scope mismatch'):
            self.packet()

    def test_development_evaluator_authenticates_flags_scope_count_and_inventory(self):
        self.set_suite('development_holdout')
        original = copy.deepcopy(self.manifest)
        mutations = [{'scope': 'diagnostic'}, {'case_count': 25},
                     {'expected_tasks': list(TASKS)[:-1]}]
        mutations.extend({key: True} for key in (
            'training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen'))
        for change in mutations:
            put(self.manifest_path, {**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluator.load_suite(self.manifest_path)

    def test_development_evaluator_rejects_repinned_provenance_and_missing_coverage(self):
        self.set_suite('development_holdout')
        source = json.loads((self.root / 'source.private.json').read_text())
        mutations = [
            lambda s: s['cases'][0]['provenance'].update(kind='synthetic_diagnostic'),
            lambda s: s['cases'][0]['documents'][0].update(source_kind='synthetic_diagnostic'),
            lambda s: s['cases'][0]['provenance'].update(expert_reviewed=True),
            lambda s: s['cases'][0]['provenance'].update(historically_unseen=True),
            lambda s: s.update(cases=[item for item in s['cases'] if item['task'] != 'clarification']),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(source)
            mutate(changed)
            self.manifest['source'] = self.suite_pin('source.private.json', changed)
            self.manifest['case_count'] = len(changed['cases'])
            put(self.manifest_path, self.manifest)
            with self.subTest(source=changed), self.assertRaises(ValueError):
                evaluator.load_suite(self.manifest_path)

    def test_structural_only_is_not_semantic_acceptance(self):
        packet, mapping = self.packet()
        report = evaluator.score_packet(packet, mapping)
        self.assertFalse(report['semantic_evaluated'])
        self.assertFalse(report['quality_accepted'])
        self.assertTrue(report['research_only'])
        self.assertNotIn('paired', report)
        for summary in report['arms'].values():
            self.assertEqual(summary['expected'], 24)
            self.assertNotIn('accepted', summary)
            self.assertEqual(set(summary['by']['task']), set(TASKS))
            self.assertEqual(summary['by']['task']['general_arithmetic']['expected'], 1)
        self.assertNotIn('arm_alias', json.dumps(packet))
        self.assertNotIn('model_identity', json.dumps(packet))
        self.assertEqual(len(packet['cases']), 24)
        self.assertEqual(packet['scope'], 'observed_diagnostic_not_fresh_gold')
        self.assertEqual(report['suite_scope'], 'diagnostic')

    def test_missing_rows_are_not_invented_or_judged(self):
        self.replace_records('candidate', self.records('candidate')[:1])
        packet, mapping = self.packet()
        self.assertEqual(sum(len(item['responses']) for item in packet['cases']), 25)
        verdicts = self.judgments(packet, mapping)
        self.assertEqual(len(verdicts['cases']), 25)
        report = evaluator.score_packet(packet, mapping, verdicts)
        candidate = report['arms']['candidate']
        self.assertEqual((candidate['expected'], candidate['missing'], candidate['accepted']), (24, 23, 1))
        self.assertEqual(candidate['accepted_rate_fixed_denominator'], 1 / 24)
        self.assertEqual(candidate['semantic_judged'], 1)
        self.assertEqual(report['paired']['by_task_group']['general']['regressions'], 2)

    def test_empty_unexecuted_arm_keeps_denominator(self):
        self.replace_records('candidate', [], complete=False, worker_exit_code=137, guard_failure='memory guard')
        packet, mapping = self.packet()
        report = evaluator.score_packet(packet, mapping, self.judgments(packet, mapping))
        self.assertEqual(report['arms']['candidate']['expected'], 24)
        self.assertEqual(report['arms']['candidate']['observed'], 0)
        self.assertEqual(report['arms']['candidate']['semantic_judged'], 0)
        self.assertEqual(report['arms']['candidate']['termination']['worker_exit_code'], 137)

    def initial_failed_arm(self):
        import evaluate_public_model as runner
        previous = json.loads((self.paths['candidate'] / 'runtime.json').read_text())
        directory = self.root / 'failed-preflight'
        directory.mkdir()
        # Exact initial supervisor runtime, before tokenizer or model admission.
        runtime = {'schema': 'public_work_hf_runtime_v1',
                   **{key: previous[key] for key in ('suite_manifest', 'suite_scope', 'source', 'inputs',
                                                     'model_manifest', 'model_identity')},
                   'tokenizer_identity': {'files': [], 'template_sha256': None}, 'adapter': None,
                   'packages': {}, 'resources': {}, 'generation_policy': runner.requested_policy(16, 128),
                   'status': 'not_started', 'research_only': True, 'quality_accepted': False}
        put(directory / 'runtime.json', runtime)
        # Use the production finalizer: it creates the empty journal and pins
        # actual runtime/response bytes rather than echoing a consumer fixture.
        runner.finalize(directory, [row['id'] for row in self.rows], {}, 'resource admission failed')
        self.paths['candidate'] = directory

    def test_producer_preflight_failure_is_all_missing_not_verified_inference(self):
        self.initial_failed_arm()
        packet, mapping = self.packet()
        report = evaluator.score_packet(packet, mapping, self.judgments(packet, mapping))
        summary = report['arms']['candidate']
        self.assertEqual((summary['expected'], summary['missing'], summary['observed']), (24, 24, 0))
        self.assertEqual((summary['semantic_judged'], summary['accepted'], summary['completion_failures']), (0, 0, 24))
        self.assertEqual(sum(len(case['responses']) for case in packet['cases']), 24)
        authentication = report['generation_authentication']
        self.assertEqual(authentication['arms']['candidate']['status'], 'failed_before_tokenizer_preflight')
        self.assertFalse(authentication['arms']['candidate']['tokenizer_identity_authenticated'])
        self.assertFalse(authentication['arms']['candidate']['inference_observed'])
        self.assertIsNone(authentication['cross_tokenizer_comparison'])
        self.assertEqual(report['paired']['by_task_group']['general']['regressions'], 2)
        self.assertEqual(summary['termination']['guard_failure'], 'resource admission failed')

    def test_missing_tokenizer_cannot_admit_nonempty_or_unfailed_arm(self):
        records = self.records('candidate')
        self.initial_failed_arm()
        self.replace_records('candidate', records, complete=False, worker_exit_code=1, guard_failure='failed')
        with self.assertRaisesRegex(ValueError, 'failed zero-response preflight'):
            self.packet()
        self.replace_records('candidate', [], complete=False, worker_exit_code=0, guard_failure=None)
        with self.assertRaisesRegex(ValueError, 'failed zero-response preflight'):
            self.packet()

    def test_preflight_failure_still_authenticates_pins_and_common_policy(self):
        self.initial_failed_arm()
        runtime_path = self.paths['candidate'] / 'runtime.json'
        runtime = json.loads(runtime_path.read_text())
        runtime['generation_policy']['max_new_tokens'] = 17
        put(runtime_path, runtime)
        self.replace_records('candidate', [], complete=False, worker_exit_code=1, guard_failure='failed')
        with self.assertRaisesRegex(ValueError, 'Paired requested policy mismatch'):
            self.packet()
        runtime['generation_policy']['max_new_tokens'] = 16
        put(runtime_path, runtime)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.packet()

    def test_producer_interrupted_unicode_journal_retains_only_observed_rows(self):
        import evaluate_public_model as runner
        runtime = json.loads((self.paths['candidate'] / 'runtime.json').read_text())
        first = json.dumps(self.records('candidate')[0], ensure_ascii=False).encode() + b'\n'
        for index, suffix in enumerate((b'{"id":"general","response":"' + '한'.encode()[:2],
                                        b'{"id":"general","response":')):
            with self.subTest(suffix=suffix):
                directory = self.root / f'interrupted-{index}'
                directory.mkdir()
                put(directory / 'runtime.json', {**runtime, 'status': 'loaded'})
                journal = first + suffix
                (directory / 'responses.jsonl').write_bytes(journal)
                runner.finalize(directory, [row['id'] for row in self.rows],
                                {'child_exit_code': 137, 'first_stop_reason': 'interrupted'})
                self.paths['candidate'] = directory
                packet, mapping = self.packet()
                report = evaluator.score_packet(packet, mapping, self.judgments(packet, mapping))
                summary = report['arms']['candidate']
                self.assertEqual((summary['expected'], summary['observed'], summary['missing']), (24, 1, 23))
                self.assertEqual(summary['semantic_judged'], 1)
                self.assertTrue(summary['termination']['journal_error'])
                self.assertEqual(sum(len(case['responses']) for case in packet['cases']), 25)
                self.assertEqual((directory / 'responses.jsonl').read_bytes(), journal)

    def test_unattested_malformed_journal_is_rejected(self):
        path = self.paths['candidate'] / 'responses.jsonl'
        path.write_bytes(b'{"id":"work"')
        completion_path = self.paths['candidate'] / 'completion.json'
        completion = json.loads(completion_path.read_text())
        completion.update(responses=evaluator.artifact_pin(path), complete=False,
                          worker_exit_code=1, guard_failure='failure without journal attestation')
        put(completion_path, completion)
        with self.assertRaisesRegex(ValueError, 'authenticated interrupted final write'):
            self.packet()

    def test_duplicate_unknown_and_malformed_ids_rejected(self):
        original = self.records('candidate')
        for records in (original + [original[0]], [{**original[0], 'id': 'unknown'}],
                        [{**original[0], 'id': 4}]):
            with self.subTest(records=records):
                self.replace_records('candidate', records)
                with self.assertRaises(ValueError):
                    self.packet()

    def test_tampered_artifact_pins_and_mapping_rejected(self):
        packet, mapping = self.packet()
        changed = copy.deepcopy(mapping)
        changed['response_map'][next(iter(changed['response_map']))]['arm'] = 'candidate'
        # Force a definite map change regardless of deterministic first arm.
        changed['case_map'][next(iter(changed['case_map']))] = 'not-a-case'
        with self.assertRaises(ValueError):
            evaluator.score_packet(packet, changed)
        responses = self.paths['candidate'] / 'responses.jsonl'
        responses.write_text(responses.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluator.score_packet(packet, mapping)

    def test_model_and_source_pins_are_authenticated(self):
        packet, mapping = self.packet()
        model_path = self.paths['candidate'] / 'model-manifest.json'
        model_path.write_text(model_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluator.score_packet(packet, mapping)
        self.make_arm('candidate')
        source_path = self.root / 'source.private.json'
        source_path.write_text(source_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'pin mismatch'):
            self.packet()

    def test_source_projection_and_reference_coverage_are_authenticated(self):
        references_path = self.root / 'references.private.json'
        self.manifest['references'] = self.suite_pin(references_path.name, self.refs[:-1])
        put(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'authored projections'):
            self.packet()
        changed = copy.deepcopy(self.refs)
        changed[0]['target'] = '접수는 화요일입니다. [D1]'
        self.manifest['references'] = self.suite_pin(references_path.name, changed)
        put(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'authored projections'):
            self.packet()

    def test_false_complete_receipt_rejected(self):
        self.replace_records('candidate', self.records('candidate')[:1], complete=True)
        with self.assertRaisesRegex(ValueError, 'False complete'):
            self.packet()

    def test_wrong_manifest_and_completion_hashes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Manifest hash'):
            evaluator.build_blind_packet(self.manifest_path, self.paths, manifest_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'Completion hash'):
            evaluator.build_blind_packet(self.manifest_path, self.paths,
                                         arm_sha256={'base': '0' * 64, 'candidate': '0' * 64})

    def test_prompt_mismatch_rejected(self):
        records = self.records('candidate')
        records[0]['prompt_sha256'] = '0' * 64
        self.replace_records('candidate', records)
        with self.assertRaisesRegex(ValueError, 'prompt hash'):
            self.packet()

    def test_common_requested_policy_is_required(self):
        self.change_runtime('candidate', lambda value: value['generation_policy'].update(seed=1))
        with self.assertRaisesRegex(ValueError, 'policy mismatch'):
            self.packet()

    def test_same_model_cannot_change_native_template_or_tokens(self):
        records = self.records('candidate')
        records[0]['prompt_token_ids'] = [5, 2]
        self.replace_records('candidate', records)
        with self.assertRaisesRegex(ValueError, 'native prompt/policy'):
            self.packet()

    def test_native_generation_prompt_flag_is_mandatory(self):
        records = self.records('candidate')
        records[0]['template_kwargs']['add_generation_prompt'] = False
        self.replace_records('candidate', records)
        with self.assertRaisesRegex(ValueError, 'template policy mismatch'):
            self.packet()

    def test_distinct_recorded_models_allow_official_tokenizer_differences(self):
        self.make_arm('candidate', model='fixture/other-model', token=7)
        packet, mapping = self.packet()
        report = evaluator.score_packet(packet, mapping)
        self.assertTrue(report['generation_authentication']['cross_tokenizer_comparison'])
        self.assertEqual(report['arms']['candidate']['completed'], 24)
        self.assertEqual(report['arms']['candidate']['candidate_source']['model_identity']['repo_id'],
                         'fixture/other-model')

    def test_external_outputs_cannot_claim_native_verification(self):
        records = self.records('candidate')
        records[0]['generation_receipt'] = 'external_unverified'
        self.replace_records('candidate', records)
        with self.assertRaisesRegex(ValueError, 'not verified'):
            self.packet()

    def test_complete_packet_bound_meaningful_judgments_required(self):
        packet, mapping = self.packet()
        original = self.judgments(packet, mapping)
        variants = []
        partial = copy.deepcopy(original)
        partial['cases'].pop()
        variants.append(partial)
        for key, value in [('semantic_pass', 1), ('reason', ''), ('reason', 'correct'),
                           ('critical_error', True), ('response_id', 'unknown')]:
            altered = copy.deepcopy(original)
            altered['cases'][0][key] = value
            variants.append(altered)
        duplicate = copy.deepcopy(original)
        duplicate['cases'].append(duplicate['cases'][0])
        variants.append(duplicate)
        stale = copy.deepcopy(original)
        stale['packet_sha256'] = '0' * 64
        variants.append(stale)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.score_packet(packet, mapping, value)

    def test_length_error_and_truncation_remain_failures(self):
        records = self.records('candidate')
        records[0].update(finish_reason='length', truncated_output=True, output_token_ids=[3] * 16, output_tokens=16)
        records[1].update(response='', raw_output='', finish_reason='exception', error='native failure',
                          output_token_ids=[], output_tokens=0, generation_policy=None)
        records[2]['context_truncated'] = True
        self.replace_records('candidate', records)
        packet, mapping = self.packet()
        judgments = self.judgments(packet, mapping)
        self.assertEqual(len(judgments['cases']), 47)
        report = evaluator.score_packet(packet, mapping, judgments)
        summary = report['arms']['candidate']
        self.assertEqual(summary['expected'], 24)
        self.assertEqual(summary['length'], 1)
        self.assertEqual(summary['exception'], 1)
        self.assertEqual(summary['truncated'], 2)
        self.assertEqual(summary['accepted'], 21)
        self.assertEqual(summary['semantic_pass'], 23)
        self.assertEqual(summary['completion_failures'], 3)
        error_id = next(key for key, value in mapping['response_map'].items()
                        if value == {'case_id': 'general', 'arm': 'candidate'})
        judgments['cases'].append({'response_id': error_id, 'semantic_pass': False,
                                   'critical_error': False, 'reason': 'No generated answer exists for semantic review.'})
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            evaluator.score_packet(packet, mapping, judgments)

    def test_general_regression_is_visible_despite_work_gain(self):
        packet, mapping = self.packet()
        judgments = self.judgments(packet, mapping)
        for verdict in judgments['cases']:
            identity = mapping['response_map'][verdict['response_id']]
            if identity in ({'case_id': 'general', 'arm': 'candidate'}, {'case_id': 'work', 'arm': 'base'}):
                verdict.update(semantic_pass=False, reason='The answer omits a required request-specific substantive claim.')
        report = evaluator.score_packet(packet, mapping, judgments)
        self.assertEqual(report['paired']['overall']['gains'], 1)
        self.assertEqual(report['paired']['overall']['regressions'], 1)
        self.assertEqual(report['paired']['by_task_group']['general']['regression_ids'], ['general'])
        self.assertEqual(report['paired']['by_task']['fact_extraction']['gains'], 1)
        self.assertFalse(report['quality_accepted'])

    def test_cli_exclusive_output_and_judge_schema(self):
        output = self.root / 'blind'
        argv = ['blind-packet', '--manifest', str(self.manifest_path), '--manifest-sha256',
                evaluator.artifact_pin(self.manifest_path)['sha256'], '--output', str(output), '--seed', 'fixed']
        for label, name in (('a', 'base'), ('b', 'candidate')):
            argv += ['--arm-' + label, str(self.paths[name]), '--arm-' + label + '-name', name,
                     '--arm-' + label + '-sha256', evaluator.artifact_pin(self.paths[name] / 'completion.json')['sha256']]
        self.assertEqual(evaluator.main(argv), 0)
        schema = json.loads((output / 'judgments.schema.json').read_text())
        mapping = json.loads((output / 'mapping.private.json').read_text())
        self.assertEqual(schema['properties']['packet_sha256']['const'], mapping['packet_sha256'])
        before = (output / 'packet.private.json').read_bytes()
        with self.assertRaises(ValueError):
            evaluator.main(argv)
        self.assertEqual((output / 'packet.private.json').read_bytes(), before)
        report_path = self.root / 'report.json'
        score = ['score', '--packet', str(output / 'packet.private.json'), '--mapping',
                 str(output / 'mapping.private.json'), '--output', str(report_path)]
        self.assertEqual(evaluator.main(score), 0)
        report_before = report_path.read_bytes()
        with self.assertRaises(FileExistsError):
            evaluator.main(score)
        self.assertEqual(report_path.read_bytes(), report_before)


if __name__ == '__main__':
    unittest.main()
