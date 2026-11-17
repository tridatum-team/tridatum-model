"""Offline artifact/assessment contracts; fixture answers are not model inference.

No fake models, remote calls or downloads. The real HF/PEFT transport is exercised
separately by the parent's explicitly bounded tiny-CPU integration smoke.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_evidence_conditioning as E
from trid.service.evidence_conditioning import build_evidence_messages


def answer(text='신청자는 담당 부서에 서류를 제출해야 합니다.', sources=None):
    return json.dumps({'paragraphs': [{'text': text, 'sources': ['E1'] if sources is None else sources}]}, ensure_ascii=False)


class EvidenceEvaluationContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        packet = {'documents': [{'doc_id': 'D1', 'title': '가상기관 접수 규정', 'institution': '가상기관'}],
                  'evidence': [{'id': 'E1', 'kind': 'quotation', 'doc_id': 'D1', 'subject': '접수',
                                'quote': '신청자는 담당 부서에 서류를 제출해야 한다.'}],
                  'scope': '가상기관 접수 절차', 'warnings': ['명시적 허구 시나리오']}
        empty = {'documents': [], 'evidence': [], 'scope': '가상기관 접수 절차', 'warnings': []}
        self.rows = [
            {'id': 'PRIVATE-CLEAN-ID', 'messages': build_evidence_messages('신청자는 무엇을 제출해야 하나요?', packet)},
            {'id': 'PRIVATE-ORDER-ID', 'messages': build_evidence_messages('접수할 때 무엇을 제출해야 하나요?', packet)},
            {'id': 'PRIVATE-MISSING-ID', 'messages': build_evidence_messages('신청자는 무엇을 제출해야 하나요?', empty)},
            {'id': 'PRIVATE-GENERAL-ID', 'messages': [{'role': 'user', 'content': '정중하게 인사해 주세요.'}]},
        ]
        self.groups = [{'id': 'PRIVATE-INVARIANT-GROUP', 'family_id': 'PRIVATE-FAMILY', 'relation': 'invariant',
                        'members': [self.rows[0]['id'], self.rows[1]['id']]},
                       {'id': 'PRIVATE-REMOVED-GROUP', 'family_id': 'PRIVATE-FAMILY', 'relation': 'support_removed',
                        'members': [self.rows[0]['id'], self.rows[2]['id']]}]
        self.references = []
        conditions = ['clean', 'reordered', 'missing', 'general']
        for index, row in enumerate(self.rows):
            self.references.append({'id': row['id'], 'family_id': 'PRIVATE-FAMILY',
                                    'institution': 'PRIVATE-INSTITUTION', 'parent_id': 'PRIVATE-PARENT',
                                    'condition': conditions[index], 'row_type': 'general' if index == 3 else 'evidence',
                                    'prompt_sha256': E.object_sha(row['messages']), 'target': 'PRIVATE-GOLD',
                                    'rubric': ['PRIVATE-RUBRIC'], 'relation_groups': [group['id'] for group in self.groups
                                                                                 if row['id'] in group['members']]})
        self.input_path = self.write('inputs.json', self.rows)
        self.reference_path = self.write('references.private.json', self.references)
        self.manifest = {'schema': E.MANIFEST_SCHEMA, 'profile': E.PROFILE,
                         'inputs': {'path': self.input_path.name, 'sha256': E.digest(self.input_path)},
                         'references': {'path': self.reference_path.name, 'sha256': E.digest(self.reference_path)},
                         'groups': self.groups}
        self.manifest_path = self.write('manifest.json', self.manifest)
        self.records = [{'id': row['id'], 'prompt_sha256': E.object_sha(row['messages']),
                         'generation_receipt': 'external_unverified',
                         'response': answer() if index < 2 else answer('제공된 근거만으로 확인할 수 없습니다.', []) if index == 2 else '안녕하세요.',
                         'finish_reason': 'eos', 'error': None, 'input_tokens': 100, 'output_tokens': 20,
                         'generation_config': {'max_new_tokens': 32, 'do_sample': True}}
                        for index, row in enumerate(self.rows)]
        self.arm_paths = {'PRIVATE-BASE-ADAPTER': self.write('base-private.json', self.records),
                          'PRIVATE-CANDIDATE-ADAPTER': self.write('candidate-private.json', self.records)}

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return path

    def blind(self, **kwargs):
        return E.build_blind_packet(self.manifest_path, self.arm_paths, seed='private-random-seed', **kwargs)

    def native_records(self, *, profile=E.SAMPLING_PROFILE, mode='smoke', budget=8):
        records = copy.deepcopy(self.records)
        for record in records:
            config = dict(E.SAMPLING_PARAMETERS, max_new_tokens=budget, eos_token_id=2)
            config.update({key: value for key, value in E.generation_dispatch(profile).items()
                           if key != 'use_model_defaults'})
            prompt = 'Artifact-only fixture<think>\n\n</think>\n\n'
            record.update(generation_receipt='hf_native_v1', generation_config=config,
                          generation_profile=profile, mode=mode, seed=E.case_seed(mode, record['id']),
                          dispatch_kwargs=E.generation_dispatch(profile), context_limit=4096,
                          template_kwargs={'add_generation_prompt': True, 'enable_thinking': False},
                          rendered_prompt=prompt, rendered_prompt_sha256=E.hashlib.sha256(prompt.encode()).hexdigest(),
                          prompt_token_ids=[1, 5], input_tokens=2, output_token_ids=[17, 2], output_tokens=2,
                          terminal_stop_token_id=2, raw_output=record['response'],
                          context_truncated=False, truncated_output=False,
                          generation_policy={'effective_config_sha256': E.object_sha(config),
                                             'generation_mode': 'sample' if profile == E.SAMPLING_PROFILE else 'greedy_search'})
        return records

    def repin_references(self):
        self.write(self.reference_path.name, self.references)
        self.manifest['references']['sha256'] = E.digest(self.reference_path)
        self.write(self.manifest_path.name, self.manifest)

    def judgments(self, packet, mapping):
        return {'schema': E.JUDGMENT_SCHEMA, 'packet_sha256': mapping['packet_sha256'],
                'cases': [{'response_id': response_id, 'pass': True, 'critical_error': False}
                          for response_id in mapping['response_map']],
                'groups': [{'group_id': group['group_id'], 'arm_alias': arm, 'pass': True}
                           for group in packet['groups'] for arm in packet['arm_aliases']]}

    def test_blind_identities_and_private_metadata_never_enter_packet(self):
        self.records[0].update(arm='PRIVATE-BASE-ADAPTER', adapter_path=str(self.root / 'PRIVATE-ADAPTER'),
                               finish_reason='PRIVATE-UNRECOGNIZED-FINISH')
        self.write('base-private.json', self.records)
        packet, mapping = self.blind()
        payload = json.dumps(packet, ensure_ascii=False)
        for secret in ['PRIVATE-', str(self.root), 'base-private', 'candidate-private',
                       'private-random-seed', 'target', 'rubric', 'generation_config', 'prompt_sha256']:
            self.assertNotIn(secret, payload)
        self.assertEqual(set(mapping['arm_map'].values()), set(self.arm_paths))
        self.assertEqual(set(mapping['case_map'].values()), {row['id'] for row in self.rows})
        self.assertEqual(len(mapping['response_map']), 8)
        self.assertEqual(mapping['packet_sha256'], E.object_sha(packet))

    def test_blinding_is_stable_for_private_seed_and_changes_with_seed(self):
        first = self.blind()
        self.assertEqual(first, self.blind())
        other = E.build_blind_packet(self.manifest_path, self.arm_paths, seed='different-private-seed')
        self.assertNotEqual(first[0], other[0])

    def test_duplicate_missing_unknown_output_ids_rejected_for_either_arm(self):
        mutations = [self.records + [self.records[0]], self.records[:-1],
                     [dict(self.records[0], id='unknown')] + self.records[1:]]
        for arm in self.arm_paths:
            for changed, expected in zip(mutations, ['Duplicate', 'Missing', 'Unknown']):
                with self.subTest(arm=arm, expected=expected):
                    self.write(self.arm_paths[arm].name, changed)
                    with self.assertRaisesRegex(ValueError, expected):
                        self.blind()
                self.write(self.arm_paths[arm].name, self.records)

    def test_duplicate_missing_unknown_reference_ids_rejected(self):
        original = copy.deepcopy(self.references)
        for changed, expected in [(original + [original[0]], 'Duplicate'), (original[:-1], 'Missing'),
                                  ([dict(original[0], id='unknown')] + original[1:], 'Unknown')]:
            with self.subTest(expected=expected):
                self.references = changed
                self.repin_references()
                with self.assertRaisesRegex(ValueError, expected):
                    self.blind()

    def test_wrong_reference_prompt_hash_rejected(self):
        self.references[0]['prompt_sha256'] = '0' * 64
        self.repin_references()
        with self.assertRaisesRegex(ValueError, 'Reference prompt hash mismatch'):
            self.blind()

    def test_wrong_output_prompt_hash_rejected(self):
        self.records[0]['prompt_sha256'] = '0' * 64
        self.write('base-private.json', self.records)
        with self.assertRaisesRegex(ValueError, 'Response prompt hash mismatch'):
            self.blind()

    def test_manifest_inputs_references_and_output_hashes_are_authenticated(self):
        with self.assertRaisesRegex(ValueError, 'Manifest hash mismatch'):
            self.blind(manifest_sha256='0' * 64)
        hashes = {arm: E.digest(path) for arm, path in self.arm_paths.items()}
        hashes['PRIVATE-BASE-ADAPTER'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Response artifact hash mismatch'):
            self.blind(arm_sha256=hashes)
        for field in ('inputs', 'references'):
            with self.subTest(field=field):
                original = self.manifest[field]['sha256']
                self.manifest[field]['sha256'] = '0' * 64
                self.write('manifest.json', self.manifest)
                with self.assertRaisesRegex(ValueError, 'Artifact hash mismatch'):
                    self.blind()
                self.manifest[field]['sha256'] = original

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for text in ['[{"id":"first","id":"second"}]', '[{"id":NaN}]']:
            self.arm_paths['PRIVATE-BASE-ADAPTER'].write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                self.blind()

    def test_jsonl_raw_outputs_preserve_complete_records(self):
        path = self.root / 'base-private.jsonl'
        path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in self.records) + '\n', encoding='utf-8')
        self.arm_paths['PRIVATE-BASE-ADAPTER'] = path
        packet, mapping = self.blind()
        self.assertEqual(E.score_packet(packet, mapping)['denominators']['responses'], 8)

    def test_no_semantic_review_emits_diagnostics_not_accuracy_or_success(self):
        packet, mapping = self.blind()
        report = E.score_packet(packet, mapping)
        self.assertFalse(report['semantic_evaluated'])
        self.assertEqual(report['denominators'], {'unique_cases': 4, 'arms': 2, 'responses': 8,
                                                  'relation_groups': 2, 'relation_group_arm_verdicts': 4})
        self.assertEqual(report['groups'], [])
        for forbidden in ('accuracy', 'success', 'accepted', 'semantic_pass'):
            self.assertNotIn('"' + forbidden + '"', json.dumps(report))
        for arm in report['arms'].values():
            self.assertEqual(arm['cases'], 4)
            self.assertEqual(arm['completed'], 4)
            self.assertEqual(arm['citation_applicable'], 3)
            self.assertEqual(set(arm['by']), {'institution', 'family_id', 'condition', 'parent_id'})

    def test_truncation_empty_exception_unknown_finish_are_failures_not_exclusions(self):
        variants = [dict(finish_reason='length', truncated_output=True),
                    dict(response='', finish_reason='empty'),
                    dict(error={'type': 'RuntimeError', 'message': 'PRIVATE-BASE-ADAPTER failed'}),
                    dict(finish_reason='unexplained'), dict(context_truncated=True)]
        for changed in variants:
            with self.subTest(changed=changed):
                rows = copy.deepcopy(self.records)
                rows[0].update(changed)
                self.write('base-private.json', rows)
                packet, mapping = self.blind()
                self.assertNotIn('PRIVATE-BASE-ADAPTER failed', json.dumps(packet))
                report = E.score_packet(packet, mapping, self.judgments(packet, mapping))
                base_alias = next(alias for alias, name in mapping['arm_map'].items() if name == 'PRIVATE-BASE-ADAPTER')
                result = next(item for item in report['cases'] if item['arm_alias'] == base_alias and item['id'] == self.rows[0]['id'])
                self.assertFalse(result['completion_valid'])
                self.assertFalse(result['accepted'])
                self.assertEqual(report['denominators']['responses'], 8)
                self.assertEqual(report['arms'][base_alias]['cases'], 4)
                self.assertEqual(report['arms'][base_alias]['completed'], 3)

    def test_invalid_citations_fail_but_empty_evidence_allows_bounded_uncertainty(self):
        self.records[0]['response'] = answer(sources=['E999'])
        self.write('base-private.json', self.records)
        packet, mapping = self.blind()
        report = E.score_packet(packet, mapping)
        base = next(alias for alias, name in mapping['arm_map'].items() if name == 'PRIVATE-BASE-ADAPTER')
        results = {item['id']: item for item in report['cases'] if item['arm_alias'] == base}
        self.assertFalse(results[self.rows[0]['id']]['citation_valid'])
        self.assertFalse(results[self.rows[0]['id']]['structural_valid'])
        self.assertTrue(results[self.rows[2]['id']]['citation_valid'])
        self.assertTrue(results[self.rows[2]['id']]['structural_valid'])
        self.assertIsNone(results[self.rows[3]['id']]['citation_valid'])

    def test_general_rows_require_explicit_private_type_not_parser_fallback(self):
        self.references[-1]['row_type'] = 'evidence'
        self.repin_references()
        with self.assertRaises(ValueError):
            self.blind()

    def test_unknown_duplicate_and_missing_relation_members_rejected(self):
        for members in [['unknown', self.rows[0]['id']], [self.rows[0]['id']] * 2, [self.rows[0]['id']]]:
            self.manifest['groups'][0]['members'] = members
            self.write('manifest.json', self.manifest)
            with self.assertRaisesRegex(ValueError, 'group members'):
                self.blind()

    def test_private_group_membership_must_match_manifest(self):
        self.references[0]['relation_groups'] = []
        self.repin_references()
        with self.assertRaisesRegex(ValueError, 'membership differs'):
            self.blind()

    def test_complete_semantic_review_has_case_and_relation_denominators(self):
        packet, mapping = self.blind()
        judgments = self.judgments(packet, mapping)
        judgments['cases'][0].update({'pass': False, 'critical_error': True})
        judgments['groups'][0]['pass'] = False
        report = E.score_packet(packet, mapping, judgments)
        self.assertTrue(report['semantic_evaluated'])
        self.assertEqual(sum(arm['semantic_judged'] for arm in report['arms'].values()), 8)
        self.assertEqual(sum(arm['critical_errors'] for arm in report['arms'].values()), 1)
        self.assertEqual(sum(arm['relation_groups_judged'] for arm in report['arms'].values()), 4)
        self.assertEqual(sum(arm['relation_groups_pass'] for arm in report['arms'].values()), 3)

    def test_group_verdicts_are_semantic_not_answer_string_equality(self):
        self.records[1]['response'] = answer('서류는 담당 부서에 제출해야 합니다.')
        for path in self.arm_paths.values():
            self.write(path.name, self.records)
        packet, mapping = self.blind()
        judgments = self.judgments(packet, mapping)
        report = E.score_packet(packet, mapping, judgments)
        self.assertTrue(all(group['pass'] for group in report['groups']))
        for judgment in judgments['groups']:
            judgment['pass'] = False
        report = E.score_packet(packet, mapping, judgments)
        self.assertFalse(any(group['pass'] for group in report['groups']))

    def test_incomplete_unknown_duplicate_semantic_cases_rejected(self):
        packet, mapping = self.blind()
        original = self.judgments(packet, mapping)
        cases = original['cases']
        changes = [(cases[:-1], 'Incomplete'), (cases + [cases[0]], 'Duplicate'),
                   ([dict(cases[0], response_id='unknown')] + cases[1:], 'Unknown'),
                   ([dict(cases[0], critical_error=None)] + cases[1:], 'explicit'),
                   ([dict(cases[0], critical_error=True)] + cases[1:], 'Critical error')]
        for changed, message in changes:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    E.score_packet(packet, mapping, dict(original, cases=changed))

    def test_incomplete_unknown_duplicate_semantic_groups_rejected(self):
        packet, mapping = self.blind()
        original = self.judgments(packet, mapping)
        groups = original['groups']
        changes = [(groups[:-1], 'Incomplete'), (groups + [groups[0]], 'Duplicate'),
                   ([dict(groups[0], group_id='unknown')] + groups[1:], 'Unknown'),
                   ([dict(groups[0], arm_alias='unknown')] + groups[1:], 'Unknown'),
                   ([dict(groups[0], **{'pass': 'yes'})] + groups[1:], 'explicit')]
        for changed, message in changes:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    E.score_packet(packet, mapping, dict(original, groups=changed))

    def test_judgments_and_packet_are_hash_bound(self):
        packet, mapping = self.blind()
        judgments = self.judgments(packet, mapping)
        judgments['packet_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Judgment packet hash mismatch'):
            E.score_packet(packet, mapping, judgments)
        packet['cases'][0]['responses'][0]['response'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'Blind packet hash mismatch'):
            E.score_packet(packet, mapping)

    def test_mapping_and_artifact_mutations_rejected_after_blinding(self):
        packet, mapping = self.blind()
        modified = copy.deepcopy(mapping)
        aliases = list(modified['arm_map'])
        modified['arm_map'][aliases[0]], modified['arm_map'][aliases[1]] = modified['arm_map'][aliases[1]], modified['arm_map'][aliases[0]]
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            E.score_packet(packet, modified)
        self.records[0]['response'] = 'changed after review'
        self.write('base-private.json', self.records)
        with self.assertRaisesRegex(ValueError, 'Artifact hash mismatch'):
            E.score_packet(packet, mapping)

    def test_token_and_rendered_prompt_receipts_cannot_contradict_their_counts(self):
        for extra, message in [({'prompt_token_ids': [1]}, 'Prompt token count'),
                               ({'output_token_ids': [1]}, 'Output token count'),
                               ({'rendered_prompt': 'actual', 'rendered_prompt_sha256': '0' * 64}, 'Rendered prompt hash')]:
            with self.subTest(message=message):
                records = copy.deepcopy(self.records)
                records[0].update(extra)
                self.write('base-private.json', records)
                with self.assertRaisesRegex(ValueError, message):
                    self.blind()

    def test_external_receipts_are_explicitly_unverified_and_cannot_be_partial(self):
        packet, mapping = self.blind()
        report = E.score_packet(packet, mapping)
        self.assertEqual(report['generation_authentication'],
                         {'receipt_kind': 'external_unverified', 'matched_native_policy': False,
                          'declared_config_matched': True, 'observed_responses': 0})
        for changed in [{'generation_receipt': None}, {'prompt_token_ids': [1] * 100}]:
            records = copy.deepcopy(self.records)
            records[0].update(changed)
            self.write('base-private.json', records)
            with self.assertRaisesRegex(ValueError, 'receipt'):
                self.blind()

    def test_native_paired_generation_rejects_policy_drift_across_or_within_arms(self):
        baseline = self.native_records()
        for drift in [self.native_records(profile='greedy'), self.native_records(budget=768),
                      self.native_records(mode='evaluation'),
                      [dict(record, context_limit=8192) for record in baseline]]:
            for across in (True, False):
                with self.subTest(across=across, drift=drift[0]['generation_profile']):
                    self.write('base-private.json', baseline)
                    self.write('candidate-private.json', drift if across else [drift[0]] + baseline[1:])
                    with self.assertRaisesRegex(ValueError, 'generation policy mismatch'):
                        self.blind()

    def test_native_seed_and_prompt_tokenization_are_matched_per_case(self):
        baseline = self.native_records()
        self.write('base-private.json', baseline)
        for extra, expected in [({'seed': 0}, 'seed mismatch'),
                                ({'prompt_token_ids': [1, 6]}, 'tokenization mismatch')]:
            records = copy.deepcopy(baseline)
            records[0].update(extra)
            self.write('candidate-private.json', records)
            with self.assertRaisesRegex(ValueError, expected):
                self.blind()

    def test_native_finish_is_recomputed_from_tokens_not_output_claims(self):
        baseline = self.native_records()
        self.write('base-private.json', baseline)
        cases = [({'output_token_ids': [17], 'output_tokens': 1}, 'Unexplained'),
                 ({'output_token_ids': [17] * 8, 'output_tokens': 8}, 'finish/token'),
                 ({'finish_reason': 'length', 'truncated_output': True}, 'finish/token'),
                 ({'truncated_output': True}, 'truncation'),
                 ({'terminal_stop_token_id': 17}, 'finish/token'),
                 ({'output_token_ids': [2, 17, 2], 'output_tokens': 3}, 'Nonterminal'),
                 ({'output_token_ids': [17] * 8 + [2], 'output_tokens': 9}, 'exceeds token budget')]
        for changed, expected in cases:
            with self.subTest(expected=expected):
                records = copy.deepcopy(baseline)
                records[0].update(changed)
                self.write('candidate-private.json', records)
                with self.assertRaisesRegex(ValueError, expected):
                    self.blind()

    def test_native_one_token_non_eos_cannot_claim_eos(self):
        records = self.native_records(budget=1)
        for record in records:
            record.update(output_token_ids=[17], output_tokens=1)
        self.write('base-private.json', records)
        with self.assertRaisesRegex(ValueError, 'finish/token'):
            self.blind()

    def test_native_length_and_pregeneration_exceptions_remain_failures(self):
        baseline = self.native_records()
        self.write('base-private.json', baseline)
        records = copy.deepcopy(baseline)
        records[0].update(output_token_ids=[17] * 8, output_tokens=8, terminal_stop_token_id=None,
                          finish_reason='length', truncated_output=True)
        records[1].update(rendered_prompt=None, rendered_prompt_sha256=None, prompt_token_ids=[],
                          input_tokens=0, output_token_ids=[], output_tokens=0, terminal_stop_token_id=None,
                          generation_policy=None, response='', error={'type': 'ValueError', 'message': 'preflight failed'},
                          finish_reason='exception')
        records[2].update(prompt_token_ids=[1] * 4096, input_tokens=4096, output_token_ids=[], output_tokens=0,
                          terminal_stop_token_id=None, generation_policy=None, response='',
                          error={'type': 'ValueError', 'message': 'context overflow'}, finish_reason='exception')
        # The exception reached a different diagnostic token count, so retain
        # the same count on both arms instead of presenting mismatched prompts.
        baseline[2] = copy.deepcopy(records[2])
        self.write('base-private.json', baseline)
        self.write('candidate-private.json', records)
        packet, mapping = self.blind()
        report = E.score_packet(packet, mapping, self.judgments(packet, mapping))
        candidate = next(alias for alias, arm in mapping['arm_map'].items() if arm == 'PRIVATE-CANDIDATE-ADAPTER')
        self.assertEqual(report['arms'][candidate]['completed'], 1)
        self.assertEqual(report['arms'][candidate]['accepted'], 1)
        self.assertEqual(report['generation_authentication']['observed_responses'], 5)

    def test_explicit_smoke_scope_is_preserved_in_report(self):
        self.manifest['scope'] = 'smoke_only'
        self.write('manifest.json', self.manifest)
        packet, mapping = self.blind()
        report = E.score_packet(packet, mapping)
        self.assertEqual(report['suite_scope'], 'smoke_only')
        self.assertEqual(report['manifest']['sha256'], E.digest(self.manifest_path))

    def test_zero_or_unavailable_token_counts_cannot_count_as_complete(self):
        for counts in [{'input_tokens': 0}, {'output_tokens': 0}, {'output_tokens': None}]:
            with self.subTest(counts=counts):
                record = dict(self.records[0], **counts)
                checks = E.structural_checks(record, self.rows[0], self.references[0])
                self.assertFalse(checks['completion_valid'])

    def test_cli_creates_separate_exclusive_private_mapping_and_scores(self):
        packet_path, mapping_path, report_path = [self.root / name for name in ('blind.json', 'mapping.private.json', 'score.private.json')]
        argv = ['blind-packet', '--manifest', str(self.manifest_path), '--manifest-sha256', E.digest(self.manifest_path),
                '--arm-a', str(self.arm_paths['PRIVATE-BASE-ADAPTER']), '--arm-a-name', 'PRIVATE-BASE-ADAPTER',
                '--arm-a-sha256', E.digest(self.arm_paths['PRIVATE-BASE-ADAPTER']),
                '--arm-b', str(self.arm_paths['PRIVATE-CANDIDATE-ADAPTER']), '--arm-b-name', 'PRIVATE-CANDIDATE-ADAPTER',
                '--arm-b-sha256', E.digest(self.arm_paths['PRIVATE-CANDIDATE-ADAPTER']),
                '--packet', str(packet_path), '--mapping', str(mapping_path), '--seed', 'private-seed']
        self.assertEqual(E.main(argv), 0)
        self.assertEqual(mapping_path.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, 'new artifacts'):
            E.main(argv)
        self.assertEqual(E.main(['score', '--packet', str(packet_path), '--mapping', str(mapping_path), '--output', str(report_path)]), 0)
        self.assertFalse(E.load_json(report_path)['semantic_evaluated'])
        with self.assertRaises(FileExistsError):
            E.main(['score', '--packet', str(packet_path), '--mapping', str(mapping_path), '--output', str(report_path)])


if __name__ == '__main__':
    unittest.main()
