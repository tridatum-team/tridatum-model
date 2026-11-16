"""Full-denominator synthetic assessment tests; never read experiment artifacts."""
import copy
import hashlib
import json
import tarfile
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import assess_expanded_experiment as assessment
from scripts.evaluate_local_eight import mois_schema


GENERATION = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
                  max_new_tokens=768, context_tokens=4096, dtype='bfloat16',
                  backend='transformers-mps')
ARMS = ('base', 'step-000024')
BEHAVIORS = ('direct', 'condition', 'exception', 'comparison', 'refusal',
             'multi_turn', 'multi_item', 'multi_clause_synthesis')
ADAPTER_SHA = hashlib.sha256(b'synthetic adapter identity only').hexdigest()
SUPPORTED = '{"paragraphs":[{"text":"기한은 3일이다.","sources":["E1"]}]}'
UNSUPPORTED = '{"paragraphs":[{"text":"아무 조치도 허용되지 않는다.","sources":["E1"]}]}'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
                    encoding='utf-8')
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_pin(path, root=None):
    path = Path(path)
    return dict(path=str(path.relative_to(root) if root else path.resolve()),
                sha256=sha(path), bytes=path.stat().st_size)


def vote(semantic=True, format_pass=True, errors=(), reason='The supplied evidence supports the deadline.'):
    return dict(semantic_pass=semantic, format_pass=format_pass,
                critical_errors=list(errors), reason=reason)


class SyntheticExperiment:
    """Author local-evaluator-shaped files, without weights, runners, or real data."""

    def __init__(self, root):
        self.root = Path(root)
        self.inputs = []
        self.metadata = []
        self.references = []
        self.secondary_ids = []
        evidence = dict(documents=[dict(doc_id='d', title='합성 규정')],
                        evidence=[dict(id='E1', doc_id='d', quote='기한은 3일이다.')],
                        scope='synthetic fixture only', warnings=[])
        for family in range(20):
            for behavior in BEHAVIORS:
                cid = f'mois-{family:02d}-{behavior}'
                messages = [dict(role='system', content='Answer from supplied evidence as paragraph JSON.')]
                if behavior == 'multi_turn':
                    messages.extend([dict(role='user', content='앞으로 제시하는 합성 규정만 사용하세요.'),
                                     dict(role='assistant', content='제시된 규정에 근거하겠습니다.')])
                messages.append(dict(role='user', content=json.dumps(
                    dict(question=f'합성 사례 {family}: 기한은? ({behavior})', source_data=evidence),
                    ensure_ascii=False)))
                self.add_case(cid, messages, 'mois', f'synthetic-source-{family:02d}', behavior,
                              SUPPORTED, behavior in BEHAVIORS[-2:])
        for index in range(40):
            self.add_case(f'general-{index:02d}',
                          [dict(role='user', content=f'Synthetic exercise {index}: say hello.')],
                          'general', f'author-synthetic-family-{index // 4:02d}', 'greeting', 'Hello.', False)
        self.design = dict(primary=dict(evaluation_questions=200, planned_answers=400,
                                       arms=['base', 'step24'], decoder_profile='unconstrained'),
                           format_secondary=dict(questions=40, planned_answers=80,
                                                 decoder_profile='json-schema', scores_combined_with_primary=False),
                           generation=GENERATION,
                           model=dict(repo_id='synthetic/model', revision='f' * 40,
                                      adapter_step=24, adapter_sha256=ADAPTER_SHA))
        documents = dict(design=self.design, evaluation_inputs=self.inputs,
                         evaluation_metadata=self.metadata, evaluation_references=self.references,
                         source_allocation={'synthetic': True}, training_manifest={'synthetic': True},
                         semantic_review_resolution={'synthetic': True})
        self.paths = {name: dump(self.root / 'dataset' / (name + '.json'), value)
                      for name, value in documents.items()}
        self.frozen_path = self.root / 'frozen-dataset.json'
        self.repin_dataset()
        self.primary = self.root / 'primary'
        self.secondary = self.root / 'secondary'
        self.write_run(self.primary, self.inputs, 'unconstrained', self.paths['evaluation_inputs'])

    def add_case(self, cid, messages, cohort, family, behavior, reference, selected):
        item = dict(id=cid, messages=messages)
        case_sha = assessment.object_sha(item)
        self.inputs.append(item)
        self.metadata.append(dict(id=cid, cohort=cohort, family_id=family, behavior=behavior,
                                  case_sha256=case_sha, secondary_selected=selected))
        self.references.append(dict(id=cid, case_sha256=case_sha, reference_answer=reference,
                                    rubric=dict(requirements=['Answer only the supplied synthetic question.'],
                                                requested_format='paragraph JSON' if cohort == 'mois' else 'plain text')))
        if selected:
            self.secondary_ids.append(cid)

    def repin_dataset(self):
        dump(self.frozen_path, dict(schema='trid-expanded-frozen-dataset-v1',
                                   pins={name: file_pin(path, self.root) for name, path in self.paths.items()},
                                   secondary_case_ids=self.secondary_ids))

    def add_secondary(self):
        cases = [item for item in self.inputs if item['id'] in self.secondary_ids]
        path = dump(self.root / 'secondary-inputs.json', cases)
        self.write_run(self.secondary, cases, 'json-schema', path)

    def write_run(self, directory, cases, profile, input_path):
        prompts = [dict(item, prompt_sha256=assessment.object_sha(item['messages']),
                        rendered_prompt_sha256=hashlib.sha256(('rendered:' + item['id']).encode()).hexdigest(),
                        prompt_token_ids=[1, 2]) for item in cases]
        decoder = dict(profile=profile, semantic_truth_guarantee=False)
        if profile == 'json-schema':
            schemas = {item['id']: mois_schema(item) for item in cases}
            decoder.update(package='lm-format-enforcer', version='0.11.3',
                           integration='build_transformers_prefix_allowed_tokens_fn',
                           schemas=schemas, schema_sha256={cid: assessment.object_sha(s) for cid, s in schemas.items()},
                           eos_violation_policy='fatal; never substitute EOS, repair, or retry',
                           production_structural_validator='unchanged',
                           tokenizer_path_preflight=[dict(id=item['id'], citation_paths=1) for item in cases])
        dump(directory / 'decoder-profile.json', decoder)
        manifest = dict(schema='trid-local-eight-paired-v1', mode='evaluation',
                        arm_order=list(ARMS), grading_access=False, retries=False, resume=False,
                        selected_ids=[item['id'] for item in cases], input=file_pin(input_path),
                        generation=GENERATION, decoder_profile=decoder,
                        model=dict(id=self.design['model']['repo_id'], revision=self.design['model']['revision'],
                                   body_hashes_authenticated=True, representation='hf-safetensors-bfloat16'),
                        adapter=dict(step=24, files={'adapter_model.safetensors': {'sha256': ADAPTER_SHA}}),
                        runtime=dict(device='mps', dtype='torch.bfloat16', cpu_fallback=False,
                                     cuda_equivalence_claim=False), code={})
        dump(directory / 'run-manifest.json', manifest)
        dump(directory / 'prompt-preflight.json', dict(status='passed', prompts=prompts, truncation=False,
                                                      all_selected_preflighted_before_weights=True))
        config = dict(do_sample=False, num_beams=1, max_new_tokens=768, return_dict_in_generate=True,
                      use_cache=True, eos_token_id=[4], pad_token_id=4, bos_token_id=1)
        policy = dict(dispatch_kwargs=dict(use_model_defaults=False, do_sample=False, num_beams=1),
                      generation_mode='greedy_search', effective_config_sha256=assessment.object_sha(config),
                      preparation_implementation=dict(path='synthetic/generation/utils.py', bytes=1,
                                                      sha256=hashlib.sha256(b'synthetic preparation').hexdigest()))
        dump(directory / 'loaded-runtime.json', dict(parameter_dtypes=['torch.bfloat16'],
                                                     buffer_dtypes=['torch.bfloat16'], generation_config=config,
                                                     requested_generation_config=copy.deepcopy(config),
                                                     generation_policy=policy,
                                                     elapsed_seconds=0.1, loader={'synthetic': True}, code={}))
        for arm in ARMS:
            rows = [dict(item, arm=arm,
                         run_manifest_sha256=sha(directory / 'run-manifest.json'),
                         loaded_runtime_sha256=sha(directory / 'loaded-runtime.json'),
                         model_id=manifest['model']['id'], model_revision=manifest['model']['revision'],
                         input_sha256=manifest['input']['sha256'], nominated_adapter_weights_sha256=ADAPTER_SHA,
                         adapter_weights_sha256=None if arm == 'base' else ADAPTER_SHA,
                         mode='evaluation', decoder_profile=profile,
                         decoder_profile_sha256=sha(directory / 'decoder-profile.json'),
                         generation=GENERATION, generation_config=config,
                         generation_policy_sha256=assessment.object_sha(policy),
                         effective_generation_mode='greedy_search',
                         response_text=SUPPORTED if item['id'].startswith('mois-') else 'Hello.',
                         output_token_ids=[3, 4], generated_tokens=2, terminal_stop_token_id=4,
                         termination_reason='eos', exception=None, context_truncated=False,
                         skipped_context=False, truncated_output=False,
                         elapsed_seconds=(1.0 if arm == 'base' else 1.5) + (0.25 if profile == 'json-schema' else 0.0))
                    for item in prompts]
            self.write_rows(directory, arm, rows, refresh=False)
        dump(directory / 'completion.json', dict(status='complete', mode='evaluation', rows_per_arm=len(cases),
                                                run_manifest_sha256=sha(directory / 'run-manifest.json'),
                                                elapsed_seconds=10.0,
                                                outputs={arm: file_pin(directory / (arm + '-outputs.jsonl')) for arm in ARMS}))

    def rows(self, directory, arm='base'):
        return [json.loads(line) for line in (directory / (arm + '-outputs.jsonl')).read_text().splitlines()]

    def write_rows(self, directory, arm, rows, refresh=True):
        path = directory / (arm + '-outputs.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        if refresh:
            self.refresh_output(directory, arm)

    def refresh_output(self, directory, arm):
        completion = load(directory / 'completion.json')
        completion['outputs'][arm] = file_pin(directory / (arm + '-outputs.jsonl'))
        dump(directory / 'completion.json', completion)

    def packet(self, secondary=False, name='blind'):
        out = self.root / name
        assessment.make_packet(self.root, self.frozen_path, sha(self.frozen_path), self.primary, out,
                               self.secondary if secondary else None)
        return out / 'review-packet.json', out / 'private-mapping.json'

    def reports(self, packet_path, votes=None):
        packet = load(packet_path)
        votes = votes or {row['sample_id']: vote() for row in packet['rows']}
        return [dump(self.root / f'review-{index}.json',
                     dict(schema='trid-expanded-blind-review-v1', packet_sha256=sha(packet_path),
                          reviewer_id=f'synthetic-reviewer-{index}', independent=True, blinded=True,
                          judgments=copy.deepcopy(votes))) for index in (1, 2)]

    def freeze(self, packet_path, reports, **kwargs):
        path = self.root / 'frozen-judgments.json'
        assessment.freeze(self.root, packet_path, reports, path, **kwargs)
        return path

    def aggregate(self, judgments, mapping, expected_sha=None):
        return assessment.aggregate(self.root, judgments, expected_sha or sha(judgments), mapping,
                                    self.root / 'assessment.json')


class ExpandedAssessmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = SyntheticExperiment(self.root)

    def assert_blinded(self, packet):
        forbidden = {'arm', 'profile', 'decoder_profile', 'elapsed_seconds', 'timing', 'votes',
                     'reviewer_id', 'case_id', 'family_id', 'suite', 'model_id', 'adapter_weights_sha256'}
        def walk(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(packet)
        for row in packet['rows']:
            self.assertEqual(set(row), {'sample_id', 'messages', 'response', 'reference_answer', 'rubric', 'structural'})

    def test_historical_greedy_receipts_need_no_sampling_fields_or_new_policy_validator(self):
        from scripts.aws_eval import fresh_eight_evaluation as evaluation
        with patch.object(evaluation, 'validate_generation_policy',
                          side_effect=AssertionError('historical policy must remain independently validated')), \
             patch.object(evaluation, 'case_seed',
                          side_effect=AssertionError('historical greedy has no per-case sampling seed')):
            packet, mapping = self.fixture.packet()
            self.assertNotIn('generation_profile', load(mapping)['runs']['primary'])
            frozen = self.fixture.freeze(packet, self.fixture.reports(packet))
            result = self.fixture.aggregate(frozen, mapping)
        self.assertEqual(result['primary']['answers'], 400)
        self.assertNotIn('generation_profile', result)
        self.assertIsNone(result['selected_step'])
        self.assertFalse(result['deployment_qualified'])

    def test_full_primary_pipeline_preserves_families_and_anonymity(self):
        packet_path, mapping_path = self.fixture.packet()
        packet, mapping = load(packet_path), load(mapping_path)
        self.assertEqual(len(packet['rows']), 400)
        self.assertEqual(len({row['sample_id'] for row in packet['rows']}), 400)
        self.assert_blinded(packet)
        self.assertTrue(all(row['structural']['pass'] for row in packet['rows']))
        self.assertTrue(all('not_entailment' in row['structural']['reason'] for row in packet['rows']
                            if mapping['rows'][row['sample_id']]['cohort'] == 'mois'))
        reports = self.fixture.reports(packet_path)
        second = load(reports[1])
        for judgment in second['judgments'].values():
            judgment['reason'] = 'Independent wording: the answer meets the supplied rubric.'
        dump(reports[1], second)
        judgments = self.fixture.freeze(packet_path, reports)
        frozen = load(judgments)
        self.assertFalse(frozen['identities_joined'])
        self.assertEqual(frozen['disagreements'], [])
        result = self.fixture.aggregate(judgments, mapping_path)
        primary = result['primary']
        self.assertEqual((primary['cases'], primary['answers']), (200, 400))
        self.assertEqual(primary['by_cohort']['mois']['cases'], 160)
        self.assertEqual(primary['by_cohort']['general']['cases'], 40)
        self.assertEqual(primary['by_behavior']['multi_item']['cases'], 20)
        self.assertEqual(primary['by_behavior']['multi_clause_synthesis']['cases'], 20)
        self.assertEqual(len(primary['by_family_id']), 30)
        for index in range(10):
            self.assertEqual(primary['by_family_id'][f'general:author-synthetic-family-{index:02d}']['cases'], 4)
        for index in range(20):
            self.assertEqual(primary['by_family_id'][f'mois:synthetic-source-{index:02d}']['cases'], 8)
        self.assertEqual(primary['cluster_uncertainty']['clusters'], 30)
        self.assertEqual(primary['cluster_uncertainty']['equal_family_weight_joint_net_fraction'], 0)
        self.assertTrue(primary['cluster_uncertainty']['synthetic_general_clusters_preserved'])
        for arm in ARMS:
            self.assertEqual(primary['arms'][arm]['joint_pass'], 200)
            self.assertEqual(primary['arms'][arm]['latency']['n'], 200)
        self.assertEqual(primary['paired']['latency_delta_seconds'], 0.5)
        self.assertIsNone(result['secondary'])
        self.assertIsNone(result['selected_step'])
        self.assertFalse(result['deployment_qualified'])
        self.assertFalse(result['broad_accuracy_claim'])
        self.assertTrue(result['original_no_selection_unchanged'])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.fixture.packet()

    def test_immediate_eos_empty_answer_survives_packet_freeze_and_aggregation(self):
        rows = self.fixture.rows(self.fixture.primary, ARMS[1])
        rows[0].update(response_text='', output_token_ids=[4], generated_tokens=1,
                       terminal_stop_token_id=4, termination_reason='eos')
        cid = rows[0]['id']
        self.fixture.write_rows(self.fixture.primary, ARMS[1], rows)
        packet_path, mapping_path = self.fixture.packet()
        packet, mapping = load(packet_path), load(mapping_path)
        self.assertEqual(len(packet['rows']), 400)
        self.assertEqual(len(mapping['rows']), 400)
        sid = next(sid for sid, item in mapping['rows'].items()
                   if item['case_id'] == cid and item['arm'] == ARMS[1])
        selected = next(row for row in packet['rows'] if row['sample_id'] == sid)
        self.assertEqual(selected['response'], '')
        self.assertFalse(selected['structural']['pass'])
        self.assertEqual(selected['structural']['reason'], 'empty_output')
        self.assertEqual(sum(not row['structural']['pass'] for row in packet['rows']), 1)
        votes = {row['sample_id']: vote() for row in packet['rows']}
        votes[sid] = vote(False, False, reason='No answer or requested paragraph JSON was produced.')
        judgments = self.fixture.freeze(packet_path, self.fixture.reports(packet_path, votes))
        frozen = load(judgments)
        self.assertEqual(len(frozen['judgments']), 400)
        self.assertEqual(frozen['judgments'][sid], votes[sid])
        primary = self.fixture.aggregate(judgments, mapping_path)['primary']
        self.assertEqual((primary['cases'], primary['answers']), (200, 400))
        for arm in ARMS:
            self.assertEqual(primary['arms'][arm]['latency']['n'], 200)
        for metric in ('structural_pass', 'semantic_pass', 'format_pass', 'joint_pass'):
            self.assertEqual(primary['arms']['base'][metric], 200)
            self.assertEqual(primary['arms'][ARMS[1]][metric], 199)
            self.assertEqual(primary['paired'][metric]['regressions'], 1)
            self.assertEqual(primary['paired'][metric]['net_fraction'], -1 / 200)

    def test_structurally_valid_false_answer_is_not_semantic_success(self):
        rows = self.fixture.rows(self.fixture.primary, ARMS[1])
        rows[0]['response_text'] = UNSUPPORTED
        cid = rows[0]['id']
        self.fixture.write_rows(self.fixture.primary, ARMS[1], rows)
        packet_path, mapping_path = self.fixture.packet()
        packet, mapping = load(packet_path), load(mapping_path)
        sid = next(sid for sid, item in mapping['rows'].items() if item['case_id'] == cid and item['arm'] == ARMS[1])
        selected = next(row for row in packet['rows'] if row['sample_id'] == sid)
        self.assertTrue(selected['structural']['pass'])
        votes = {row['sample_id']: vote() for row in packet['rows']}
        votes[sid] = vote(False, True, ['Invents a blanket prohibition absent from E1.'],
                          'E1 states a deadline, not a prohibition.')
        judgments = self.fixture.freeze(packet_path, self.fixture.reports(packet_path, votes))
        primary = self.fixture.aggregate(judgments, mapping_path)['primary']
        adapted = primary['arms'][ARMS[1]]
        self.assertEqual(adapted['structural_pass'], 200)
        self.assertEqual(adapted['format_pass'], 200)
        self.assertEqual(adapted['semantic_pass'], 199)
        self.assertEqual(adapted['joint_pass'], 199)
        self.assertEqual(adapted['critical_case_count'], 1)
        self.assertEqual(primary['paired']['semantic_pass']['regressions'], 1)
        family = primary['by_family_id']['mois:synthetic-source-00']
        self.assertEqual(family['paired']['joint_pass']['net_fraction'], -1 / 8)
        self.assertAlmostEqual(primary['cluster_uncertainty']['equal_family_weight_joint_net_fraction'], -1 / 240)

    def test_secondary_grammar_benefit_and_semantic_regression_remain_separate(self):
        self.fixture.add_secondary()
        cid = self.fixture.secondary_ids[0]
        primary_rows = self.fixture.rows(self.fixture.primary)
        next(row for row in primary_rows if row['id'] == cid)['response_text'] = '기한은 3일이다.'
        self.fixture.write_rows(self.fixture.primary, 'base', primary_rows)
        secondary_rows = self.fixture.rows(self.fixture.secondary)
        next(row for row in secondary_rows if row['id'] == cid)['response_text'] = UNSUPPORTED
        self.fixture.write_rows(self.fixture.secondary, 'base', secondary_rows)
        packet_path, mapping_path = self.fixture.packet(secondary=True)
        packet, mapping = load(packet_path), load(mapping_path)
        self.assertEqual(len(packet['rows']), 480)
        self.assert_blinded(packet)
        votes = {row['sample_id']: vote() for row in packet['rows']}
        for sid, item in mapping['rows'].items():
            if item['case_id'] == cid and item['arm'] == 'base':
                votes[sid] = (vote(True, False, reason='Correct deadline, but not the requested JSON.')
                              if item['suite'] == 'primary' else
                              vote(False, True, ['Invents a prohibition.'], 'Valid JSON does not entail a true claim.'))
        judgments = self.fixture.freeze(packet_path, self.fixture.reports(packet_path, votes))
        result = self.fixture.aggregate(judgments, mapping_path)
        self.assertEqual((result['primary']['cases'], result['primary']['answers']), (200, 400))
        secondary = result['secondary']
        self.assertEqual((secondary['cases'], secondary['answers']), (40, 80))
        self.assertFalse(secondary['combined_with_primary'])
        self.assertEqual(len(secondary['by_family_id']), 20)
        benefits = secondary['grammar_vs_unconstrained_same_cases']
        self.assertEqual(benefits['base']['cases'], 40)
        for metric in ('format_gains', 'structural_gains', 'semantic_regressions', 'introduced_critical_cases'):
            self.assertEqual(benefits['base'][metric], 1)
        for metric in ('format_regressions', 'structural_regressions', 'semantic_gains'):
            self.assertEqual(benefits['base'][metric], 0)
        self.assertEqual(benefits['base']['latency_delta_seconds'], 0.25)
        self.assertEqual(benefits[ARMS[1]]['format_gains'], 0)
        self.assertEqual(result['primary']['arms']['base']['semantic_pass'], 200)
        self.assertEqual(secondary['arms']['base']['semantic_pass'], 39)

    def test_disagreement_only_third_review_resolves_all_flagged_dimensions(self):
        packet_path, mapping_path = self.fixture.packet()
        reports = self.fixture.reports(packet_path)
        second = load(reports[1])
        ids = list(second['judgments'])[:3]
        second['judgments'][ids[0]] = vote(False, reason='Meaning is unsupported.')
        second['judgments'][ids[1]] = vote(True, False, reason='Requested format is not satisfied.')
        second['judgments'][ids[2]] = vote(True, True, ['Material invented exception.'])
        dump(reports[1], second)
        with self.assertRaisesRegex(ValueError, 'require.*third'):
            self.fixture.freeze(packet_path, reports)
        info = assessment.adjudication(self.root, packet_path, reports, self.root / 'third-packet')
        adjudication_path = Path(info['packet'])
        self.assertEqual(info['disagreements'], 3)
        self.assertTrue(info['third_required'])
        third_packet = load(adjudication_path)
        self.assert_blinded(third_packet)
        self.assertEqual({row['sample_id'] for row in third_packet['rows']}, set(ids))
        self.assertEqual(third_packet['parent_packet_sha256'], sha(packet_path))
        third_votes = {sid: vote(False, True, ['Third review confirms unsupported content.'],
                                 'The reference does not support this claim.') for sid in ids}
        third = dump(self.root / 'third-review.json',
                     dict(schema='trid-expanded-blind-review-v1', packet_sha256=sha(packet_path),
                          adjudication_packet_sha256=sha(adjudication_path), reviewer_id='third-independent',
                          independent=True, blinded=True, judgments=third_votes))
        judgments = self.fixture.freeze(packet_path, reports, third=third, adjudication_path=adjudication_path)
        frozen = load(judgments)
        self.assertEqual(set(frozen['disagreements']), set(ids))
        for sid in ids:
            self.assertEqual(frozen['judgments'][sid], third_votes[sid])
        self.assertEqual(len(frozen['judgments']), 400)
        primary = self.fixture.aggregate(judgments, mapping_path)['primary']
        self.assertEqual(sum(primary['arms'][arm]['semantic_pass'] for arm in ARMS), 397)

    def test_third_review_rejects_unflagged_missing_reused_or_stale_reports(self):
        packet_path, _ = self.fixture.packet()
        reports = self.fixture.reports(packet_path)
        second = load(reports[1])
        sid = next(iter(second['judgments']))
        second['judgments'][sid] = vote(False)
        dump(reports[1], second)
        info = assessment.adjudication(self.root, packet_path, reports, self.root / 'third-packet')
        adjudication_path = Path(info['packet'])
        valid = dict(schema='trid-expanded-blind-review-v1', packet_sha256=sha(packet_path),
                     adjudication_packet_sha256=sha(adjudication_path), reviewer_id='third-independent',
                     independent=True, blinded=True, judgments={sid: vote(False)})
        another = next(row['sample_id'] for row in load(packet_path)['rows'] if row['sample_id'] != sid)
        variants = [dict(valid, judgments={}), dict(valid, judgments={sid: vote(), another: vote()}),
                    dict(valid, reviewer_id='synthetic-reviewer-1'),
                    dict(valid, adjudication_packet_sha256='0' * 64),
                    dict(valid, packet_sha256='0' * 64), dict(valid, blinded=False)]
        for index, report in enumerate(variants):
            with self.subTest(index=index):
                third = dump(self.root / f'invalid-third-{index}.json', report)
                with self.assertRaises(ValueError):
                    self.fixture.freeze(packet_path, reports, third=third, adjudication_path=adjudication_path)
        mutated = load(adjudication_path)
        mutated['rows'][0]['response'] = 'Changed adjudication answer'
        dump(adjudication_path, mutated)
        third = dump(self.root / 'third-review.json', dict(valid, adjudication_packet_sha256=sha(adjudication_path)))
        with self.assertRaisesRegex(ValueError, 'adjudication'):
            self.fixture.freeze(packet_path, reports, third=third, adjudication_path=adjudication_path)

    def test_agreement_forbids_third_review(self):
        packet_path, _ = self.fixture.packet()
        reports = self.fixture.reports(packet_path)
        info = assessment.adjudication(self.root, packet_path, reports, self.root / 'empty-third')
        self.assertEqual(info['disagreements'], 0)
        self.assertFalse(info['third_required'])
        self.assertEqual(load(info['packet'])['rows'], [])
        with self.assertRaisesRegex(ValueError, 'third forbidden'):
            self.fixture.freeze(packet_path, reports, third=reports[0], adjudication_path=Path(info['packet']))

    def test_review_denominator_identity_and_strict_vote_types(self):
        packet_path, _ = self.fixture.packet()
        reports = self.fixture.reports(packet_path)
        original = load(reports[1])
        sid = next(iter(original['judgments']))
        variants = []
        missing = copy.deepcopy(original)
        del missing['judgments'][sid]
        variants.append(missing)
        extra = copy.deepcopy(original)
        extra['judgments']['unknown-sample'] = vote()
        variants.append(extra)
        variants.extend([dict(original, packet_sha256='0' * 64),
                         dict(original, reviewer_id='synthetic-reviewer-1'),
                         dict(original, independent=False), dict(original, blinded=False)])
        for field, value in [('semantic_pass', 1), ('format_pass', 'true'), ('reason', ''),
                             ('critical_errors', ['duplicate', 'duplicate'])]:
            invalid = copy.deepcopy(original)
            invalid['judgments'][sid][field] = value
            variants.append(invalid)
        for index, report in enumerate(variants):
            with self.subTest(index=index):
                dump(reports[1], report)
                with self.assertRaises(ValueError):
                    self.fixture.freeze(packet_path, reports)
        dump(reports[1], original)
        text = reports[1].read_text()
        reports[1].write_text(text.replace('"blinded": true', '"blinded": false, "blinded": true', 1))
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.fixture.freeze(packet_path, reports)

    def test_output_denominator_duplicates_missing_and_unknown_ids(self):
        original = self.fixture.rows(self.fixture.primary)
        variants = [original[:-1], original + [copy.deepcopy(original[0])],
                    [dict(original[0], id='unknown')] + original[1:]]
        for index, rows in enumerate(variants):
            with self.subTest(index=index):
                self.fixture.write_rows(self.fixture.primary, 'base', rows)
                with self.assertRaisesRegex(ValueError, 'denominator|duplicate IDs'):
                    self.fixture.packet(name=f'bad-rows-{index}')

    def test_answer_bindings_reject_rehashed_wrong_content(self):
        original = self.fixture.rows(self.fixture.primary)
        mutations = dict(arm=ARMS[1], run_manifest_sha256='0' * 64, loaded_runtime_sha256='0' * 64,
                         input_sha256='0' * 64, model_id='wrong/model', model_revision='wrong-revision',
                         nominated_adapter_weights_sha256='0' * 64, adapter_weights_sha256=ADAPTER_SHA,
                         mode='smoke', decoder_profile='json-schema', decoder_profile_sha256='0' * 64,
                         generation_policy_sha256='0' * 64, effective_generation_mode='sample',
                         messages=[dict(role='user', content='Changed question')], prompt_sha256='0' * 64,
                         prompt_token_ids=[99], rendered_prompt_sha256='0' * 64,
                         generation=dict(GENERATION, seed=99),
                         generation_config=dict(do_sample=True))
        for field, value in mutations.items():
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[0][field] = value
                self.fixture.write_rows(self.fixture.primary, 'base', rows)
                with self.assertRaises(ValueError):
                    self.fixture.packet(name='wrong-' + field)

    def test_invalid_response_exception_and_latency_records_are_rejected(self):
        original = self.fixture.rows(self.fixture.primary)
        variants = {
            'missing-response': {},
            'null-response': {'response_text': None},
            'number-response': {'response_text': 0},
            'list-response': {'response_text': []},
            'object-response': {'response_text': {}},
            'exception': {'response_text': '', 'exception': 'generation failed'},
            'negative-latency': {'response_text': '', 'elapsed_seconds': -1},
            'missing-latency-value': {'response_text': '', 'elapsed_seconds': None},
        }
        for name, fields in variants.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(original)
                del rows[0]['response_text']
                rows[0].update(fields)
                self.fixture.write_rows(self.fixture.primary, 'base', rows)
                with self.assertRaisesRegex(ValueError, 'missing answer or invalid latency'):
                    self.fixture.packet(name='invalid-' + name)

    def test_output_hash_duplicate_keys_and_nonfinite_json_are_rejected(self):
        path = self.fixture.primary / 'base-outputs.jsonl'
        original = path.read_text()
        path.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
            self.fixture.packet(name='stale-output')
        path.write_text(original.replace('"arm": "base"', '"arm": "other", "arm": "base"', 1))
        self.fixture.refresh_output(self.fixture.primary, 'base')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.fixture.packet(name='duplicate-key')
        path.write_text(original.replace('"elapsed_seconds": 1.0', '"elapsed_seconds": NaN', 1))
        self.fixture.refresh_output(self.fixture.primary, 'base')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.fixture.packet(name='nonfinite')

    def test_frozen_dataset_rejects_wrong_hash_and_denominator_changes(self):
        with self.assertRaisesRegex(ValueError, 'frozen dataset hash'):
            assessment.make_packet(self.root, self.fixture.frozen_path, '0' * 64,
                                   self.fixture.primary, self.root / 'wrong-frozen-hash')
        cases = copy.deepcopy(self.fixture.inputs)
        for label, changed in [('missing', cases[:-1]), ('duplicate', cases + [cases[0]])]:
            with self.subTest(label=label):
                dump(self.fixture.paths['evaluation_inputs'], changed)
                self.fixture.repin_dataset()
                with self.assertRaisesRegex(ValueError, 'denominator|duplicate IDs'):
                    self.fixture.packet(name='frozen-' + label)
        dump(self.fixture.paths['evaluation_inputs'], cases)
        metadata = copy.deepcopy(self.fixture.metadata)
        variants = []
        wrong_family = copy.deepcopy(metadata)
        wrong_family[0]['family_id'] = 'unexpected-family'
        variants.append(wrong_family)
        duplicate_behavior = copy.deepcopy(metadata)
        duplicate_behavior[0]['behavior'] = duplicate_behavior[1]['behavior']
        variants.append(duplicate_behavior)
        wrong_hash = copy.deepcopy(metadata)
        wrong_hash[0]['case_sha256'] = '0' * 64
        variants.append(wrong_hash)
        wrong_selection = copy.deepcopy(metadata)
        wrong_selection[0]['secondary_selected'] = True
        variants.append(wrong_selection)
        for index, changed in enumerate(variants):
            with self.subTest(index=index):
                dump(self.fixture.paths['evaluation_metadata'], changed)
                self.fixture.repin_dataset()
                with self.assertRaises(ValueError):
                    self.fixture.packet(name=f'frozen-meta-{index}')

    def test_run_manifest_preflight_runtime_and_completion_contracts(self):
        directory = self.fixture.primary
        variants = [
            ('run-manifest.json', lambda d: d.update(mode='smoke')),
            ('run-manifest.json', lambda d: d.update(selected_ids=d['selected_ids'][:-1])),
            ('run-manifest.json', lambda d: d['runtime'].update(cpu_fallback=True)),
            ('run-manifest.json', lambda d: d['runtime'].update(dtype='torch.float32')),
            ('run-manifest.json', lambda d: d['model'].update(revision='wrong')),
            ('run-manifest.json', lambda d: d['adapter'].update(step=25)),
            ('run-manifest.json', lambda d: d['generation'].update(seed=15)),
            ('decoder-profile.json', lambda d: d.update(profile='json-schema')),
            ('prompt-preflight.json', lambda d: d.update(truncation=True)),
            ('prompt-preflight.json', lambda d: d.update(prompts=d['prompts'][:-1])),
            ('prompt-preflight.json', lambda d: d.update(all_selected_preflighted_before_weights=False)),
            ('loaded-runtime.json', lambda d: d.update(parameter_dtypes=['torch.float32'])),
            ('loaded-runtime.json', lambda d: d['generation_config'].update(do_sample=True)),
            ('loaded-runtime.json', lambda d: d.pop('requested_generation_config')),
            ('loaded-runtime.json', lambda d: d.pop('generation_policy')),
            ('loaded-runtime.json', lambda d: d['generation_policy']['dispatch_kwargs'].update(use_model_defaults=True)),
            ('loaded-runtime.json', lambda d: d['generation_policy']['dispatch_kwargs'].update(do_sample=True)),
            ('loaded-runtime.json', lambda d: d['generation_policy'].update(generation_mode='sample')),
            ('loaded-runtime.json', lambda d: d['generation_policy'].update(effective_config_sha256='0' * 64)),
            ('loaded-runtime.json', lambda d: d['generation_policy']['preparation_implementation'].update(path='')),
            ('loaded-runtime.json', lambda d: d['generation_policy']['preparation_implementation'].update(bytes=0)),
            ('completion.json', lambda d: d.update(status='partial')),
            ('completion.json', lambda d: d.update(rows_per_arm=199)),
            ('completion.json', lambda d: d.update(run_manifest_sha256='0' * 64)),
            ('completion.json', lambda d: d['outputs'].pop(ARMS[1])),
        ]
        originals = {name: load(directory / name) for name, _ in variants}
        completion = copy.deepcopy(originals['completion.json'])
        for index, (name, mutate) in enumerate(variants):
            with self.subTest(name=name, index=index):
                changed = copy.deepcopy(originals[name])
                mutate(changed)
                dump(directory / name, changed)
                if name == 'run-manifest.json':
                    # Keep the completion hash current so the intended manifest gate is reached.
                    dump(directory / 'completion.json', dict(completion, run_manifest_sha256=sha(directory / name)))
                with self.assertRaises(ValueError):
                    self.fixture.packet(name=f'bad-contract-{index}')
                dump(directory / name, originals[name])
                dump(directory / 'completion.json', completion)
        dump(directory / 'failure.json', dict(error='synthetic run failure'))
        with self.assertRaisesRegex(ValueError, 'failed local run'):
            self.fixture.packet(name='failed-run')

    def test_stale_reports_frozen_mapping_dataset_and_run_artifacts(self):
        packet_path, mapping_path = self.fixture.packet()
        reports = self.fixture.reports(packet_path)
        judgments = self.fixture.freeze(packet_path, reports)
        original_sha = sha(judgments)
        targets = [reports[0], packet_path, mapping_path, self.fixture.frozen_path,
                   self.fixture.paths['evaluation_references'],
                   self.fixture.primary / 'base-outputs.jsonl',
                   self.fixture.primary / 'loaded-runtime.json']
        for path in targets:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path == mapping_path:
                    mapping = load(path)
                    next(iter(mapping['rows'].values()))['arm'] = 'altered'
                    dump(path, mapping)
                else:
                    path.write_bytes(original + b'\n')
                with self.assertRaises(ValueError):
                    self.fixture.aggregate(judgments, mapping_path, original_sha)
                path.write_bytes(original)
        frozen = load(judgments)
        sid = next(iter(frozen['judgments']))
        frozen['judgments'][sid]['semantic_pass'] = False
        dump(judgments, frozen)
        with self.assertRaisesRegex(ValueError, 'frozen judgments hash'):
            self.fixture.aggregate(judgments, mapping_path, original_sha)
        with self.assertRaisesRegex(ValueError, 'resolution differs'):
            self.fixture.aggregate(judgments, mapping_path)


    def test_secondary_rejects_rebound_grammar_identity_and_schema_changes(self):
        self.fixture.add_secondary()
        directory = self.fixture.secondary
        original_decoder = load(directory / 'decoder-profile.json')
        original_manifest = load(directory / 'run-manifest.json')
        original_completion = load(directory / 'completion.json')
        variants = [dict(original_decoder, version='0.0.0'),
                    dict(original_decoder, semantic_truth_guarantee=True),
                    dict(original_decoder, eos_violation_policy='repair and retry'),
                    dict(original_decoder, production_structural_validator='disabled'),
                    dict(original_decoder, schemas={}),
                    dict(original_decoder, schema_sha256={})]
        for index, decoder in enumerate(variants):
            with self.subTest(index=index):
                dump(directory / 'decoder-profile.json', decoder)
                dump(directory / 'run-manifest.json', dict(original_manifest, decoder_profile=decoder))
                dump(directory / 'completion.json',
                     dict(original_completion, run_manifest_sha256=sha(directory / 'run-manifest.json')))
                with self.assertRaisesRegex(ValueError, 'decoder|grammar'):
                    self.fixture.packet(secondary=True, name=f'bad-grammar-{index}')

    def test_rebound_public_input_messages_still_must_match_frozen_cases(self):
        changed = copy.deepcopy(self.fixture.inputs)
        changed[0]['messages'][-1]['content'] = 'A different question.'
        replacement = dump(self.root / 'wrong-public-input.json', changed)
        directory = self.fixture.primary
        manifest = load(directory / 'run-manifest.json')
        manifest['input'] = file_pin(replacement)
        dump(directory / 'run-manifest.json', manifest)
        completion = load(directory / 'completion.json')
        completion['run_manifest_sha256'] = sha(directory / 'run-manifest.json')
        dump(directory / 'completion.json', completion)
        with self.assertRaisesRegex(ValueError, 'run input messages differ'):
            self.fixture.packet()

    def test_production_structure_and_finish_failures_remain_in_denominator(self):
        rows = self.fixture.rows(self.fixture.primary, ARMS[1])
        rows[0]['response_text'] = '{"paragraphs":[{"text":"기한은 9일이다.","sources":["E1"]}]}'
        rows[1]['response_text'] = '{"paragraphs":[{"text":"기한은 3일이다.","sources":["E2"]}]}'
        rows[2]['termination_reason'] = 'length'
        rows[2]['truncated_output'] = True
        rows[3]['response_text'] = ' \t\n'
        self.fixture.write_rows(self.fixture.primary, ARMS[1], rows)
        packet_path, mapping_path = self.fixture.packet()
        packet = load(packet_path)
        failures = [row for row in packet['rows'] if not row['structural']['pass']]
        self.assertEqual(len(packet['rows']), 400)
        self.assertEqual(len(failures), 4)
        self.assertEqual({row['structural']['reason'] for row in failures},
                         {'unsupported_numeric_value', 'invalid_citations', 'invalid_finish_or_input_metadata',
                          'empty_output'})
        judgments = self.fixture.freeze(packet_path, self.fixture.reports(packet_path))
        primary = self.fixture.aggregate(judgments, mapping_path)['primary']
        self.assertEqual(primary['arms'][ARMS[1]]['semantic_pass'], 200)
        self.assertEqual(primary['arms'][ARMS[1]]['structural_pass'], 196)
        self.assertEqual(primary['arms'][ARMS[1]]['joint_pass'], 196)
        self.assertEqual(primary['paired']['structural_pass']['regressions'], 4)
        self.assertEqual((primary['cases'], primary['answers']), (200, 400))

    def test_packet_rejects_identity_or_votes_injected_before_review(self):
        packet_path, _ = self.fixture.packet()
        original = load(packet_path)
        for field, value in [('arm', 'base'), ('decoder_profile', 'unconstrained'),
                             ('elapsed_seconds', 1.0), ('votes', {'reviewer': True})]:
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed['rows'][0][field] = value
                dump(packet_path, changed)
                reports = self.fixture.reports(packet_path)
                with self.assertRaises(ValueError):
                    self.fixture.freeze(packet_path, reports)


class CudaEvaluationAssessmentTests(unittest.TestCase):
    """The CUDA admission seam feeds exactly the existing anonymous200 pipeline."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = SyntheticExperiment(self.root)
        self.directory = self.root / 'run/cloud-state/cuda-evaluation-export/evaluation'
        self.directory.parent.mkdir(parents=True)
        self.fixture.primary.rename(self.directory)
        self.fixture.primary = self.directory
        self.registration = dump(self.root / 'candidate.json', {'synthetic_original_final150': True})
        self.protocol = dict(instructions='Judge the synthetic whole pairs independently.',
            paired_consistency=True, critical_errors_are_codes=True,
            critical_error_codes={'wrong_actor': 'Wrong actor.'})
        protocol_path = dump(self.root / 'protocol.json', self.protocol)
        manifest = load(self.directory / 'run-manifest.json')
        manifest.update(schema='trid-cuda-eight-paired-v1', arm_order=['base', 'candidate'])
        manifest['adapter']['step'] = 150
        manifest['runtime']['device'] = 'cuda'
        manifest['generation'] = dict(GENERATION, backend='transformers-cuda')
        self.contract = dict(arms=['base', 'candidate'], adapter_sha256=ADAPTER_SHA,
            adapter=manifest['adapter'], registration=file_pin(self.registration),
            training_runtime='cuda', inference_runtime='mps',
            producer='trid-fresh-eight-cuda-training-candidate-v2',
            review_protocol=file_pin(protocol_path),
            policy_document=dict(data=dict(frozen_dataset=file_pin(self.fixture.frozen_path)),
                                 evaluation=dict(development_gate={'synthetic': 'six unchanged gates'})))
        manifest['evaluation_contract'] = assessment.public_contract(self.contract)
        dump(self.directory / 'run-manifest.json', manifest)
        loaded = load(self.directory / 'loaded-runtime.json')
        loaded['parameter_dtypes'] = ['torch.bfloat16', 'torch.float32']
        loaded['parameter_inventory'] = {
            group: {dtype + '@cuda:0': dict(tensors=1, elements=1, bytes=size)}
            for group, dtype, size in [('base', 'torch.bfloat16', 2), ('adapter', 'torch.float32', 4)]}
        dump(self.directory / 'loaded-runtime.json', loaded)
        for old, arm in zip(ARMS, self.contract['arms']):
            rows = self.fixture.rows(self.directory, old)
            for row in rows:
                row.update(arm=arm, generation=manifest['generation'],
                    run_manifest_sha256=sha(self.directory / 'run-manifest.json'),
                    loaded_runtime_sha256=sha(self.directory / 'loaded-runtime.json'),
                    candidate_manifest_sha256=sha(self.registration))
            self.fixture.write_rows(self.directory, arm, rows, refresh=False)
        (self.directory / 'step-000024-outputs.jsonl').unlink()
        self.refresh_completion()
        for target, value in (
            ('resolve_candidate', self.contract), ('review_protocol', self.protocol),
            ('authenticate_cuda_evaluation', dict(pins={}, public_input=self.fixture.paths['evaluation_inputs'])),
        ):
            mocked = patch.object(assessment, target, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        protocol_patch = patch.object(assessment, 'ANONYMOUS_PROTOCOL_SHA', assessment.object_sha(self.protocol))
        protocol_patch.start()
        self.addCleanup(protocol_patch.stop)

    def refresh_completion(self):
        completion = load(self.directory / 'completion.json')
        completion.update(run_manifest_sha256=sha(self.directory / 'run-manifest.json'),
            outputs={arm: file_pin(self.directory / (arm + '-outputs.jsonl'), self.directory)
                     for arm in ('base', 'candidate')})
        dump(self.directory / 'completion.json', completion)

    def packet(self, name='cuda-blind'):
        out = self.root / name
        assessment.make_packet(self.root, self.fixture.frozen_path, sha(self.fixture.frozen_path),
            self.directory, out, candidate_manifest=self.registration, candidate_sha=sha(self.registration))
        return out / 'review-packet.json', out / 'private-mapping.json'

    def test_full_cuda200_uses_same_blind_pairs_and_six_development_gates_without_mps_files(self):
        with patch('scripts.evaluate_local_eight.model_identity', side_effect=AssertionError('no Mac model read')), \
             patch('scripts.evaluate_local_eight.accepted_local_supervision', side_effect=AssertionError('no MPS receipt')):
            packet, mapping = self.packet()
            rows = load(packet)['rows']
            self.assertEqual(len(rows), 400)
            self.assertEqual(len({row['pair_id'] for row in rows}), 200)
            self.assertTrue(all('arm' not in row for row in rows))
            frozen = self.fixture.freeze(packet, self.fixture.reports(packet))
            result = self.fixture.aggregate(frozen, mapping)
        self.assertEqual((result['primary']['cases'], result['primary']['answers']), (200, 400))
        self.assertEqual(len(result['development_gate']['checks']), 6)
        self.assertFalse(result['deployment_qualified'])
        self.assertIsNone(result['selected_step'])
        self.assertTrue(result['historical_scores_unchanged'])
        self.assertEqual(result['evaluation_contract']['inference_runtime'], 'mps')

    def test_cuda_empty_and_length_limited_answers_remain_structural_failures_in_all400(self):
        rows = self.fixture.rows(self.directory, 'candidate')
        rows[0]['response_text'] = ' '
        rows[1].update(termination_reason='length', truncated_output=True)
        self.fixture.write_rows(self.directory, 'candidate', rows, refresh=False)
        self.refresh_completion()
        packet, mapping = self.packet()
        self.assertEqual(len(load(packet)['rows']), 400)
        self.assertEqual(sum(not row['structural']['pass'] for row in load(packet)['rows']), 2)
        frozen = self.fixture.freeze(packet, self.fixture.reports(packet))
        result = self.fixture.aggregate(frozen, mapping)
        self.assertEqual(result['primary']['arms']['candidate']['structural_pass'], 198)
        self.assertEqual(result['primary']['answers'], 400)

    def test_cuda_partial_smoke_mislabeled_or_cross_run_rows_cannot_shrink_denominator(self):
        original = self.fixture.rows(self.directory, 'candidate')
        variants = [original[:-1], original + [original[0]],
                    [dict(original[0], mode='smoke'), *original[1:]],
                    [dict(original[0], run_manifest_sha256='0' * 64), *original[1:]],
                    [dict(original[0], candidate_manifest_sha256='0' * 64), *original[1:]],
                    [dict(original[0], loaded_runtime_sha256='0' * 64), *original[1:]]]
        for index, rows in enumerate(variants):
            self.fixture.write_rows(self.directory, 'candidate', rows, refresh=False)
            self.refresh_completion()
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.packet('invalid-' + str(index))

    def test_cuda_output_paths_cannot_claim_remote_absolute_endpoints(self):
        completion = load(self.directory / 'completion.json')
        completion['outputs']['base']['path'] = '/opt/trid/output/evaluation/base-outputs.jsonl'
        dump(self.directory / 'completion.json', completion)
        with self.assertRaisesRegex(ValueError, 'portable'):
            self.packet()

    def test_cuda_schema_never_admits_without_original_candidate_or_with_smoke_mode(self):
        cases = assessment.indexed(self.fixture.inputs, 'synthetic')
        with self.assertRaises((ValueError, FileNotFoundError)):
            assessment.completed_run(self.root, self.directory, cases, self.fixture.design, 'unconstrained')
        manifest = load(self.directory / 'run-manifest.json')
        manifest['mode'] = 'smoke'
        dump(self.directory / 'run-manifest.json', manifest)
        self.refresh_completion()
        with self.assertRaisesRegex(ValueError, 'identity/mode'):
            self.packet()


class CudaSamplingAssessmentTests(CudaEvaluationAssessmentTests):
    """Fresh sampling receipts retain the same200 pairs and inherited gate checks."""

    generation_profile = 'qwen3_nonthinking_sampling'

    def setUp(self):
        super().setUp()
        from scripts.aws_eval import fresh_eight_evaluation as evaluation
        self.evaluation = evaluation
        manifest = load(self.directory / 'run-manifest.json')
        manifest.update(generation_profile=self.generation_profile,
                        generation=evaluation.generation_for_profile(self.generation_profile))
        dump(self.directory / 'run-manifest.json', manifest)
        loaded = load(self.directory / 'loaded-runtime.json')
        config = dict(loaded['generation_config'], **evaluation.SAMPLING_PARAMETERS)
        policy = dict(loaded['generation_policy'],
            dispatch_kwargs=dict(use_model_defaults=False, **evaluation.SAMPLING_PARAMETERS),
            generation_profile=self.generation_profile, generation_mode='sample',
            effective_config_sha256=assessment.object_sha(config))
        loaded.update(generation_profile=self.generation_profile, generation_config=config,
                      requested_generation_config=copy.deepcopy(config), generation_policy=policy)
        dump(self.directory / 'loaded-runtime.json', loaded)
        for arm in self.contract['arms']:
            rows = self.fixture.rows(self.directory, arm)
            for row in rows:
                row.update(generation_profile=self.generation_profile,
                           case_seed=evaluation.case_seed('evaluation', row['id']),
                           effective_generation_mode='sample')
            self.fixture.write_rows(self.directory, arm, rows, refresh=False)
        self.rebind_receipts()

    def rebind_receipts(self):
        manifest = load(self.directory / 'run-manifest.json')
        loaded = load(self.directory / 'loaded-runtime.json')
        for arm in self.contract['arms']:
            rows = self.fixture.rows(self.directory, arm)
            for row in rows:
                row.update(generation=manifest['generation'],
                    run_manifest_sha256=sha(self.directory / 'run-manifest.json'),
                    loaded_runtime_sha256=sha(self.directory / 'loaded-runtime.json'),
                    generation_config=loaded['generation_config'],
                    generation_policy_sha256=assessment.object_sha(loaded['generation_policy']))
            self.fixture.write_rows(self.directory, arm, rows, refresh=False)
        self.refresh_completion()

    def packet(self, name='cuda-blind'):
        out = self.root / name
        assessment.make_packet(self.root, self.fixture.frozen_path, sha(self.fixture.frozen_path),
            self.directory, out, candidate_manifest=self.registration, candidate_sha=sha(self.registration),
            generation_profile=self.generation_profile)
        return out / 'review-packet.json', out / 'private-mapping.json'

    def test_sampling_requires_explicit_opt_in_and_never_relabels_greedy(self):
        with self.assertRaisesRegex(ValueError, 'greedy generation profile'):
            CudaEvaluationAssessmentTests.packet(self)
        cases = assessment.indexed(self.fixture.inputs, 'synthetic')
        for profile in ('unknown', 'greedy'):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                assessment.completed_run(self.root, self.directory, cases, self.fixture.design,
                    'unconstrained', contract=self.contract, generation_profile=profile)
        with self.assertRaisesRegex(ValueError, 'without secondary'):
            assessment.make_packet(self.root, self.fixture.frozen_path, sha(self.fixture.frozen_path),
                self.directory, self.root / 'secondary-search', secondary=self.directory,
                candidate_manifest=self.registration, candidate_sha=sha(self.registration),
                generation_profile=self.generation_profile)

    def test_sampling_profile_and_exact_parameters_bind_manifest_runtime_and_policy(self):
        mutations = [
            ('run-manifest.json', lambda x: x.pop('generation_profile')),
            ('run-manifest.json', lambda x: x.update(generation_profile='greedy')),
            ('run-manifest.json', lambda x: x['generation'].update(temperature=0.8)),
            ('run-manifest.json', lambda x: x['generation'].update(seed=15)),
            ('loaded-runtime.json', lambda x: x.pop('generation_profile')),
            ('loaded-runtime.json', lambda x: x.update(generation_profile='greedy')),
            ('loaded-runtime.json', lambda x: x['generation_policy'].pop('generation_profile')),
            ('loaded-runtime.json', lambda x: x['generation_policy'].update(generation_mode='greedy_search')),
            ('loaded-runtime.json', lambda x: x['generation_policy']['dispatch_kwargs'].update(top_k=21)),
            ('loaded-runtime.json', lambda x: x['generation_config'].update(top_p=0.9)),
            ('loaded-runtime.json', lambda x: x['requested_generation_config'].update(min_p=0.1)),
        ]
        originals = {name: load(self.directory / name) for name, _ in mutations}
        for index, (name, mutate) in enumerate(mutations):
            changed = copy.deepcopy(originals[name])
            mutate(changed)
            dump(self.directory / name, changed)
            self.rebind_receipts()
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.packet('invalid-profile-' + str(index))
            dump(self.directory / name, originals[name])
            self.rebind_receipts()

    def test_sampling_rows_bind_profile_and_seed_to_mode_case_not_arm_or_tuning(self):
        original = self.fixture.rows(self.directory, 'candidate')
        variants = [
            lambda x: x.pop('generation_profile'),
            lambda x: x.update(generation_profile='greedy'),
            lambda x: x.pop('case_seed'),
            lambda x: x.update(case_seed=True),
            lambda x: x.update(case_seed=x['case_seed'] + 1),
            lambda x: x.update(case_seed=self.evaluation.case_seed('smoke', x['id'])),
            lambda x: x.update(case_seed=original[1]['case_seed']),
            lambda x: x.update(effective_generation_mode='greedy_search'),
        ]
        for index, mutate in enumerate(variants):
            rows = copy.deepcopy(original)
            mutate(rows[0])
            self.fixture.write_rows(self.directory, 'candidate', rows, refresh=False)
            self.refresh_completion()
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.packet('invalid-seed-' + str(index))

    def test_sampling_panels_pair_third_and_distinct_unpooled_aggregate(self):
        packet, mapping = self.packet()
        document = load(packet)
        for row in document['rows']:
            self.assertFalse({'generation_profile', 'case_seed', 'arm'} & row.keys())
        self.assertEqual(load(mapping)['runs']['primary']['generation_profile'], self.generation_profile)
        base_rows = self.fixture.rows(self.directory, 'base')
        candidate_rows = self.fixture.rows(self.directory, 'candidate')
        self.assertEqual([r['case_seed'] for r in base_rows], [r['case_seed'] for r in candidate_rows])
        index = ExpandedAssessmentPanelTests.shard(self, packet_path=packet)
        sid = document['rows'][0]['sample_id']
        panels, _ = ExpandedAssessmentPanelTests.panels(self, index,
            overrides={('second', sid): vote(semantic=False)})
        reports = ExpandedAssessmentPanelTests.collect(self, index, panels)
        adjudication_dir = self.root / 'pair-adjudication'
        assessment.adjudication(self.root, packet, reports, adjudication_dir)
        adjudication = adjudication_dir / 'adjudication-packet.json'
        self.assertEqual(len(load(adjudication)['rows']), 2)
        third_index = ExpandedAssessmentPanelTests.shard(
            self, packet_path=adjudication, name='third-shards')
        third_panels, _ = ExpandedAssessmentPanelTests.panels(self, third_index, panel_ids=('third',))
        third = ExpandedAssessmentPanelTests.collect(
            self, third_index, third_panels, name='third-collected', third=True)[0]
        frozen = self.fixture.freeze(packet, reports, third=third, adjudication_path=adjudication)
        result = self.fixture.aggregate(frozen, mapping)
        self.assertEqual(result['generation_profile'], self.generation_profile)
        self.assertEqual(result['generation_condition'], 'separate_inference_only_fixed_seed_sampling')
        self.assertFalse(result['historical_pooling'])
        self.assertFalse(result['stochastic_robustness_established'])
        self.assertEqual(len(result['development_gate']['checks']), 6)
        self.assertEqual(result['adjudicated_samples'], 2)
        self.assertEqual(result['primary']['answers'], 400)
        self.assertIsNone(result['selected_step'])
        self.assertFalse(result['deployment_qualified'])

    def test_aggregate_reauthenticates_sampling_seed_even_when_file_pins_are_refreshed(self):
        packet, mapping_path = self.packet()
        frozen_path = self.fixture.freeze(packet, self.fixture.reports(packet))
        rows = self.fixture.rows(self.directory, 'candidate')
        rows[0]['case_seed'] += 1
        self.fixture.write_rows(self.directory, 'candidate', rows, refresh=False)
        self.refresh_completion()
        mapping = load(mapping_path)
        for name in ('candidate-outputs.jsonl', 'completion.json'):
            mapping['runs']['primary']['pins'][name] = file_pin(self.directory / name, self.root)
        packet_document = load(packet)
        packet_document['mapping_commitment'] = assessment.object_sha(assessment.mapping_body(mapping))
        dump(packet, packet_document)
        mapping['packet_sha256'] = sha(packet)
        dump(mapping_path, mapping)
        # Re-freeze synthetic reviews so rejection reaches output seed validation,
        # not a stale report or mapping digest.
        frozen_path.unlink()
        frozen = self.fixture.freeze(packet, self.fixture.reports(packet))
        with self.assertRaisesRegex(ValueError, 'case seed binding'):
            self.fixture.aggregate(frozen, mapping_path)


class CudaEvaluationEvidenceTests(unittest.TestCase):
    """Exercise consumer custody/collection checks with synthetic component contracts."""

    def setUp(self):
        from scripts import local_eight_candidate as C
        from scripts.aws_eval import fresh_eight_evaluation as E
        from scripts.aws_eval import fresh_eight_launch as F
        from scripts.aws_eval import grounded_hf_lifecycle as L
        self.C, self.E, self.F, self.L = C, E, F, L
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        self.state = self.run / 'cloud-state'
        self.exported = self.state / 'cuda-evaluation-export'
        self.directory = self.exported / 'evaluation'
        self.directory.mkdir(parents=True)
        self.sources = self.root / 'staged'
        self.sources.mkdir()
        self.code_name = 'scripts/aws_eval/synthetic_evaluation.py'
        code = dump(self.sources / self.code_name, {'historical_execution': True})
        self.code = {self.code_name: file_pin(code, self.sources)}
        candidate_path = dump(self.root / 'original-candidate.json', {'original': 'final150'})
        self.contract = dict(registration=file_pin(candidate_path), adapter_sha256=ADAPTER_SHA)
        self.lineage = dict(schema='synthetic-parent-lineage', provenance={'original': 'full-closure'},
                            adapter_sha256=ADAPTER_SHA)
        lineage = dump(self.sources / 'lineage-binding.json', self.lineage)
        source_manifest = dump(self.sources / 'source-manifest.json', dict(files=list(self.code.values())))
        inputs = dump(self.root / 'public.json', [{'id': 'synthetic', 'messages': []}])
        self.manifest = dict(code=self.code, evaluation_contract=self.lineage['provenance'],
                             input=file_pin(inputs), runtime=dict(gpu_uuid='GPU-synthetic', instance_id='i-123'))
        self.loaded = dict(code=self.code, gpu_uuid='GPU-synthetic')
        bundle_path = self.run / 'source-bundle.tar'
        bundle_files = [*self.code.values(), file_pin(lineage, self.sources), file_pin(source_manifest, self.sources)]
        self.archive(bundle_path, self.sources, bundle_files)
        bundle = dict(file_pin(bundle_path, self.root), local_path=str(bundle_path.relative_to(self.root)),
                      files=bundle_files, unpacked_bytes=1024**2)
        del bundle['path']
        self.plan = dict(schema=E.SCHEMA, identity=dict(run_id='synthetic-run'),
            ami=dict(RootDeviceName='/dev/root'), bundle=bundle,
            sources={self.code_name: self.code[self.code_name]['sha256']},
            output_allowlist=['plan.json', 'runtime.json', 'lineage-binding.json', 'source-manifest.json'],
            workload=dict(lineage_binding=file_pin(lineage, self.sources),
                          source_manifest=file_pin(source_manifest, self.sources)))
        for path in (self.run / 'plan.json', self.exported / 'plan.json'):
            dump(path, self.plan)
        self.plan_sha = sha(self.run / 'plan.json')
        identity = dict(run_id='synthetic-run', schedule_group='synthetic-group',
            instance_profile='synthetic-profile', **{key: 'synthetic-' + key for key in F.ROLE_KEYS})
        registration = dump(self.run / 'registration.json', dict(identity=identity))
        self.registration_sha = sha(registration)
        dump(self.run / 'approval.json', {'synthetic': 'approved'})
        runtime = dict(packages={'synthetic': 'runtime'})
        dump(self.exported / 'runtime.json', runtime)
        dump(self.exported / 'lineage-binding.json', self.lineage)
        dump(self.exported / 'source-manifest.json', load(source_manifest))
        self.status = dict(run_id='synthetic-run', plan_sha256=self.plan_sha, instance_id='i-123',
            evaluation_success=True, stop_errors=[], runtime=runtime,
            runtime_unit=dict(gpu={'synthetic': True}), stages=[dict(gpu={'synthetic': True})])
        members = [file_pin(self.exported / name, self.exported) for name in self.plan['output_allowlist']]
        export_manifest = dump(self.exported / 'export-manifest.json',
            dict(run_id='synthetic-run', plan_sha256=self.plan_sha, status=self.status,
                 files=members, total_bytes=sum(p['bytes'] for p in members)))
        self.archive(self.state / 'export.tar', self.exported,
                     [*members, file_pin(export_manifest, self.exported)])
        export_pin = file_pin(self.state / 'export.tar')
        del export_pin['path']
        completion = dict(self.status, export=export_pin)
        dump(self.state / 'completion.json', completion)
        cleanup = dict(cleanup_state='complete', instance_terminated=True, instance_id='i-123',
            root_volume_ids=['vol-123'], recorded_volume_ids=['vol-123'], absent_volume_ids=['vol-123'],
            remaining_volumes=[])
        self.controller = dict(state='complete', evaluation_success=True, instance_id='i-123', cleanup=cleanup,
            evidence=dict(native_pipeline_success=True, evidence_kind='authenticated_archive',
                export_state='authenticated_terminal', completion=completion, export_sha256=export_pin['sha256']))
        dump(self.state / 'controller-result.json', self.controller)
        dump(self.state / 'volume-identity.json',
            dict(L.volume_identity(self.plan['identity'], self.plan_sha, '/dev/root'),
                 instance_id='i-123', root_volume_ids=['vol-123'], volumes={'vol-123': {}}))
        dump(self.state / 'approval-consumed.json', dict(run_id='synthetic-run', plan_sha256=self.plan_sha))
        dump(self.state / 'release.json', dict(run_id='synthetic-run', plan_sha256=self.plan_sha,
                                              instance_id='i-123', gpu={'uuid': 'GPU-synthetic'}))
        dump(self.state / 'registration-consumed.json',
             dict(registration_sha256=self.registration_sha, identity=identity))
        owned = dict(registration_sha256=self.registration_sha, resources={'security_group': {'GroupId': 'sg-123'}})
        dump(self.state / 'owned-prerequisites.json', owned)
        dump(self.state / 'owned-cleanup-started.json', owned)
        dump(self.state / 'owned-cleanup-complete.json', dict(registration_sha256=self.registration_sha,
            cleanup_state='complete', retained_prefix_lifecycle_rules=True, s3_mutations=0))
        deletions = {
            'schedule-group': ('scheduler', 'delete-schedule-group', ['--name', identity['schedule_group']], 'ResourceNotFoundException'),
            'profile': ('iam', 'delete-instance-profile', ['--instance-profile-name', identity['instance_profile']], 'NoSuchEntity'),
            'security-group': ('ec2', 'delete-security-group', ['--group-id', 'sg-123'], 'InvalidGroup.NotFound'),
            **{key: ('iam', 'delete-role', ['--role-name', identity[key]], 'NoSuchEntity') for key in F.ROLE_KEYS}}
        for name, (service, operation, args, error) in deletions.items():
            intent = dict(registration_sha256=self.registration_sha, service=service, operation=operation, args=args)
            dump(self.state / ('cleanup-' + name + '-intent.json'), intent)
            dump(self.state / ('cleanup-' + name + '-response.json'), dict(request=intent, response={}))
            dump(self.state / ('cleanup-' + name + '-readback.json'),
                 dict(request=intent, observation=dict(absent=True, error_code=error)))
        # Component validators have separate tests; here their synthetic contracts
        # isolate archive custody and consumer-owned evidence cross-bindings.
        for module, name, value in (
            (F, 'validate_registration', None), (F, 'validate_final', None),
            (E, 'validate_completion', None),
            (E, 'validate_supervision', dict(unit=self.status['stages'][0])),
            (L, 'validate_evaluation_terminal', None), (L, 'verify_gpu_observation', None),
            (L, 'output_limit', 1024**2), (C, 'cuda_evaluation_lineage', self.lineage),
        ):
            mocked = patch.object(module, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        source_patch = patch.object(E, 'REQUIRED_SOURCES', {self.code_name})
        source_patch.start()
        self.addCleanup(source_patch.stop)

    @staticmethod
    def archive(path, root, members):
        with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as stream:
            for member in members:
                stream.add(root / member['path'], arcname=member['path'], recursive=False)

    def admit(self):
        return assessment.authenticate_cuda_evaluation(self.root, self.directory,
            self.contract, self.manifest, self.loaded)

    def test_archived_source_admission_is_independent_of_later_checkout(self):
        dump(self.root / self.code_name, {'later_checkout': True})
        result = self.admit()
        self.assertEqual(result['public_input'], (self.root / 'public.json').resolve())
        self.assertGreater(len(result['pins']), 10)

    def test_plan_generation_profile_cannot_be_relabelled_by_assessor(self):
        profile = 'qwen3_nonthinking_sampling'
        with self.assertRaisesRegex(ValueError, 'plan generation profile'):
            assessment.authenticate_cuda_evaluation(self.root, self.directory,
                self.contract, self.manifest, self.loaded, generation_profile=profile)
        plan = copy.deepcopy(self.plan)
        plan['workload']['decoder_profile'] = profile
        dump(self.exported / 'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'plan generation profile'):
            self.admit()

    def test_sampling_cannot_reinterpret_original_archive_recovery(self):
        from scripts.aws_eval import fresh_eight_evaluation_recover as recovery
        profile = 'qwen3_nonthinking_sampling'
        plan = copy.deepcopy(self.plan)
        plan['workload']['decoder_profile'] = profile
        dump(self.exported / 'plan.json', plan)
        dump(self.state / recovery.RECEIPT_NAME, {})
        with patch.object(recovery, 'validate_recovery',
                          side_effect=AssertionError('sampling never enters historical recovery')):
            with self.assertRaisesRegex(ValueError, 'recovery is greedy only'):
                assessment.authenticate_cuda_evaluation(self.root, self.directory,
                    self.contract, self.manifest, self.loaded, generation_profile=profile)

    def test_source_archive_or_original_lineage_tampering_is_rejected(self):
        with patch.object(self.C, 'cuda_evaluation_lineage', return_value=dict(self.lineage, adapter_sha256='0' * 64)):
            with self.assertRaisesRegex(ValueError, 'original candidate lineage'):
                self.admit()
        path = self.run / 'source-bundle.tar'
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaises(ValueError):
            self.admit()

    def test_partial_export_incomplete_cleanup_and_foreign_deletion_are_rejected(self):
        for change in (
            lambda value: value.update(evaluation_success=False),
            lambda value: value['evidence'].update(export_state='authenticated_partial'),
            lambda value: value['cleanup'].update(absent_volume_ids=[]),
            lambda value: value['cleanup'].update(instance_id='i-foreign'),
        ):
            controller = copy.deepcopy(self.controller)
            change(controller)
            dump(self.state / 'controller-result.json', controller)
            with self.assertRaises(ValueError):
                self.admit()
        dump(self.state / 'controller-result.json', self.controller)
        readback = self.state / 'cleanup-profile-readback.json'
        value = load(readback)
        value['request']['args'] = ['--instance-profile-name', 'foreign']
        dump(readback, value)
        with self.assertRaisesRegex(ValueError, 'deletion readback'):
            self.admit()

    def test_metadata_recovery_marker_cannot_bypass_original_authentication(self):
        from scripts.aws_eval import fresh_eight_evaluation_recover as recovery
        # This intentionally invalid marker is not a fixture claiming real CUDA
        # acceptance. The actual recovery admission must reject it, unmocked.
        registration = dump(self.state / recovery.REG_NAME,
            dict(schema=recovery.REGISTRATION_SCHEMA, approved=True))
        dump(self.state / recovery.RECEIPT_NAME,
            dict(registration=file_pin(registration, self.root)))
        with self.assertRaisesRegex(ValueError, 'registration fields'):
            self.admit()

    def test_recovery_branch_still_requires_original_candidate_lineage(self):
        from scripts.aws_eval import fresh_eight_evaluation_recover as recovery
        dump(self.state / recovery.RECEIPT_NAME, {})
        with patch.object(self.C, 'cuda_evaluation_lineage',
                          return_value=dict(self.lineage, adapter_sha256='0' * 64)):
            with self.assertRaisesRegex(ValueError, 'original candidate lineage'):
                self.admit()


class ExpandedAssessmentPanelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = SyntheticExperiment(self.root)
        self.fixture.add_secondary()
        self.packet_path, self.mapping_path = self.fixture.packet(secondary=True)

    def shard(self, packet_path=None, name='shards', size=16):
        packet_path = packet_path or self.packet_path
        directory = self.root / name
        assessment.shard(self.root, packet_path, sha(packet_path), size, directory)
        return directory / 'shard-index.json'

    def panels(self, index_path, panel_ids=('first', 'second'), overrides=None, reviewer=None):
        index = load(index_path)
        manifests = []
        leaves = []
        for panel_id in panel_ids:
            paths = []
            for number, entry in enumerate(index['children']):
                child_path = self.root / entry['packet']['path']
                child = load(child_path)
                judgments = {row['sample_id']: vote() for row in child['rows']}
                for sid in judgments:
                    if overrides and (panel_id, sid) in overrides:
                        judgments[sid] = copy.deepcopy(overrides[(panel_id, sid)])
                identity = reviewer(panel_id, number) if reviewer else f'{panel_id}-actual-reviewer-{number:02d}'
                paths.append(dump(self.root / panel_id / f'leaf-{number:02d}.json',
                                  dict(schema='trid-expanded-blind-review-v1',
                                       packet_sha256=sha(child_path), reviewer_id=identity,
                                       independent=True, blinded=True, judgments=judgments)))
            manifest = dump(self.root / panel_id / 'panel.json',
                            dict(schema='trid-expanded-review-panel-v1', panel_id=panel_id,
                                 index_sha256=sha(index_path),
                                 reports=[file_pin(path, self.root) for path in paths]))
            manifests.append(manifest)
            leaves.append(paths)
        return manifests, leaves

    def collect(self, index_path, panels, name='collected', third=False):
        directory = self.root / name
        assessment.collect(self.root, index_path, sha(index_path), panels, directory, third=third)
        return ([directory / 'third-panel.json'] if third else
                [directory / 'panel-1.json', directory / 'panel-2.json'])

    def test_full_shards_and_truthful_panels_complete_assessment(self):
        index_path = self.shard()
        index = load(index_path)
        self.assertEqual(index['schema'], 'trid-expanded-review-shards-v1')
        self.assertEqual(index['parent'], file_pin(self.packet_path, self.root))
        self.assertEqual(len(index['children']), 30)
        parent_rows = {row['sample_id']: row for row in load(self.packet_path)['rows']}
        covered = []
        for entry in index['children']:
            child_path = self.root / entry['packet']['path']
            self.assertEqual(entry['packet'], file_pin(child_path, self.root))
            child = load(child_path)
            self.assertEqual(set(child), {'schema', 'parent_packet_sha256', 'rows', 'instructions'})
            self.assertEqual(child['schema'], 'trid-expanded-blind-shard-v1')
            self.assertEqual(child['parent_packet_sha256'], sha(self.packet_path))
            self.assertEqual(len(child['rows']), 16)
            ids = [row['sample_id'] for row in child['rows']]
            self.assertEqual(entry['sample_ids'], ids)
            for row in child['rows']:
                self.assertEqual(row, parent_rows[row['sample_id']])
            covered.extend(ids)
        self.assertEqual(len(covered), 480)
        self.assertEqual(set(covered), set(parent_rows))
        self.assertEqual(len(set(covered)), len(covered))
        panels, leaves = self.panels(index_path)
        reports = self.collect(index_path, panels)
        for panel_number, report_path in enumerate(reports):
            report = load(report_path)
            self.assertEqual(report['reviewer_kind'], 'panel')
            self.assertEqual(report['panel_id'], ('first', 'second')[panel_number])
            self.assertEqual(report['reviewer_id'], 'panel:' + report['panel_id'])
            self.assertEqual(report['packet_sha256'], sha(self.packet_path))
            self.assertEqual(len(report['judgments']), 480)
            composition = report['composition']
            self.assertEqual(composition['index'], file_pin(index_path, self.root))
            self.assertEqual(composition['panel'], file_pin(panels[panel_number], self.root))
            self.assertEqual(set(composition['samples']), set(parent_rows))
            actual_reviewers = set()
            for sid, provenance in composition['samples'].items():
                leaf_path = self.root / provenance['report']['path']
                self.assertIn(leaf_path, leaves[panel_number])
                leaf = load(leaf_path)
                actual_reviewers.add(provenance['reviewer_id'])
                self.assertEqual(provenance['reviewer_id'], leaf['reviewer_id'])
                self.assertNotEqual(report['reviewer_id'], leaf['reviewer_id'])
                self.assertEqual(report['judgments'][sid], leaf['judgments'][sid])
                child_path = self.root / provenance['child_packet']['path']
                self.assertEqual(provenance['child_packet'], file_pin(child_path, self.root))
                self.assertEqual(leaf['packet_sha256'], sha(child_path))
            self.assertEqual(len(actual_reviewers), 30)
        judgments = self.fixture.freeze(self.packet_path, reports)
        result = self.fixture.aggregate(judgments, self.mapping_path)
        self.assertEqual((result['primary']['cases'], result['primary']['answers']), (200, 400))
        self.assertEqual((result['secondary']['cases'], result['secondary']['answers']), (40, 80))
        self.assertEqual(result['primary']['arms']['base']['joint_pass'], 200)

    def test_sharding_rejects_wrong_parent_hash_and_invalid_sizes(self):
        with self.assertRaises(ValueError):
            assessment.shard(self.root, self.packet_path, '0' * 64, 16, self.root / 'wrong-parent')
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    assessment.shard(self.root, self.packet_path, sha(self.packet_path), size,
                                     self.root / f'invalid-size-{size}')

    def test_shard_index_requires_exact_disjoint_parent_coverage(self):
        index_path = self.shard()
        original = load(index_path)
        panels, _ = self.panels(index_path)
        mutations = []
        missing = copy.deepcopy(original)
        missing['children'].pop()
        mutations.append(missing)
        duplicate = copy.deepcopy(original)
        duplicate['children'].append(copy.deepcopy(duplicate['children'][0]))
        mutations.append(duplicate)
        unknown = copy.deepcopy(original)
        unknown['children'][0]['sample_ids'][0] = 'not-a-parent-sample'
        mutations.append(unknown)
        overlap = copy.deepcopy(original)
        overlap['children'][1]['sample_ids'][0] = overlap['children'][0]['sample_ids'][0]
        mutations.append(overlap)
        stale_pin = copy.deepcopy(original)
        stale_pin['children'][0]['packet']['sha256'] = '0' * 64
        mutations.append(stale_pin)
        for number, changed in enumerate(mutations):
            with self.subTest(number=number):
                dump(index_path, changed)
                for panel_path in panels:
                    panel = load(panel_path)
                    panel['index_sha256'] = sha(index_path)
                    dump(panel_path, panel)
                with self.assertRaises(ValueError):
                    self.collect(index_path, panels, name=f'invalid-index-{number}')

    def test_child_content_is_bound_to_parent_even_when_its_pin_is_refreshed(self):
        index_path = self.shard()
        index = load(index_path)
        child_path = self.root / index['children'][0]['packet']['path']
        child = load(child_path)
        child['rows'][0]['response'] = 'A substituted response, not the parent response.'
        dump(child_path, child)
        index['children'][0]['packet'] = file_pin(child_path, self.root)
        dump(index_path, index)
        panels, _ = self.panels(index_path)
        with self.assertRaises(ValueError):
            self.collect(index_path, panels)

    def test_panels_reject_missing_extra_duplicate_and_stale_leaf_pins(self):
        index_path = self.shard()
        panels, leaves = self.panels(index_path)
        original = load(panels[1])
        missing = copy.deepcopy(original)
        missing['reports'].pop()
        duplicate = copy.deepcopy(original)
        duplicate['reports'][1] = copy.deepcopy(duplicate['reports'][0])
        extra = copy.deepcopy(original)
        extra['reports'].append(file_pin(leaves[0][0], self.root))
        stale = copy.deepcopy(original)
        stale['reports'][0]['sha256'] = '0' * 64
        for number, changed in enumerate((missing, duplicate, extra, stale)):
            with self.subTest(number=number):
                dump(panels[1], changed)
                with self.assertRaises(ValueError):
                    self.collect(index_path, panels, name=f'invalid-panel-{number}')

    def test_collect_requires_expected_panel_count_and_index_hash(self):
        index_path = self.shard()
        panels, _ = self.panels(index_path)
        for number, selected in enumerate(([], panels[:1], panels + panels[:1])):
            with self.subTest(number=number):
                with self.assertRaises(ValueError):
                    self.collect(index_path, selected, name=f'wrong-panel-count-{number}')
        with self.assertRaises(ValueError):
            self.collect(index_path, panels, name='wrong-third-count', third=True)
        with self.assertRaises(ValueError):
            assessment.collect(self.root, index_path, '0' * 64, panels, self.root / 'wrong-index-hash')

    def test_distinct_panel_labels_do_not_hide_same_actual_reviewer(self):
        index_path = self.shard()
        panels, _ = self.panels(index_path, reviewer=lambda panel, number:
                                'same-person' if number == 0 else f'{panel}-reviewer-{number}')
        with self.assertRaises(ValueError):
            self.collect(index_path, panels)

    def test_leaf_report_must_cover_its_own_child_and_bind_its_hash(self):
        index_path = self.shard()
        panels, leaves = self.panels(index_path)
        original_report = load(leaves[0][0])
        original_panel = load(panels[0])
        sid = next(iter(original_report['judgments']))
        unknown = next(iter(load(leaves[0][1])['judgments']))
        missing = copy.deepcopy(original_report)
        del missing['judgments'][sid]
        extra = copy.deepcopy(original_report)
        extra['judgments'][unknown] = vote()
        wrong_hash = dict(original_report, packet_sha256='0' * 64)
        for number, changed in enumerate((missing, extra, wrong_hash)):
            with self.subTest(number=number):
                dump(leaves[0][0], changed)
                panel = copy.deepcopy(original_panel)
                panel['reports'][0] = file_pin(leaves[0][0], self.root)
                dump(panels[0], panel)
                with self.assertRaises(ValueError):
                    self.collect(index_path, panels, name=f'wrong-leaf-{number}')

    def test_composition_is_reauthenticated_at_freeze_and_aggregate(self):
        index_path = self.shard()
        panels, leaves = self.panels(index_path)
        reports = self.collect(index_path, panels)
        judgments = self.fixture.freeze(self.packet_path, reports)
        child_path = self.root / load(index_path)['children'][0]['packet']['path']
        for number, path in enumerate((index_path, child_path, panels[0], leaves[0][0])):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.assertRaises(ValueError):
                    assessment.freeze(self.root, self.packet_path, reports, self.root / f'stale-freeze-{number}.json')
                with self.assertRaises(ValueError):
                    self.fixture.aggregate(judgments, self.mapping_path)
                path.write_bytes(original)
        original = load(reports[0])
        sid = next(iter(original['judgments']))
        changed_vote = copy.deepcopy(original)
        changed_vote['judgments'][sid]['semantic_pass'] = False
        false_identity = copy.deepcopy(original)
        false_identity['composition']['samples'][sid]['reviewer_id'] = 'invented-reviewer'
        false_single_reviewer = copy.deepcopy(original)
        false_single_reviewer['reviewer_kind'] = 'individual'
        false_single_reviewer['reviewer_id'] = 'one-person-reviewed-everything'
        dropped_composition = {key: value for key, value in original.items()
                               if key not in ('composition', 'reviewer_kind', 'panel_id')}
        for number, changed in enumerate((changed_vote, false_identity, false_single_reviewer,
                                           dropped_composition)):
            with self.subTest(number=number):
                dump(reports[0], changed)
                with self.assertRaises(ValueError):
                    assessment.freeze(self.root, self.packet_path, reports, self.root / f'false-composition-{number}.json')

    def test_disagreement_only_third_panel_pipeline_and_actual_independence(self):
        index_path = self.shard()
        ids = [row['sample_id'] for row in load(self.packet_path)['rows'][:20]]
        overrides = {('second', sid): vote(False, reason='This claim is unsupported.') for sid in ids}
        panels, _ = self.panels(index_path, overrides=overrides)
        reports = self.collect(index_path, panels)
        with self.assertRaises(ValueError):
            self.fixture.freeze(self.packet_path, reports)
        adjudication_dir = self.root / 'adjudication'
        assessment.adjudication(self.root, self.packet_path, reports, adjudication_dir)
        adjudication_path = adjudication_dir / 'adjudication-packet.json'
        adjudication_packet = load(adjudication_path)
        self.assertEqual({row['sample_id'] for row in adjudication_packet['rows']}, set(ids))
        third_index = self.shard(adjudication_path, name='third-shards')
        self.assertEqual([len(entry['sample_ids']) for entry in load(third_index)['children']], [16, 4])
        third_overrides = {('third', sid): vote(False, True, ['Unsupported assertion.'],
                                               'The evidence does not entail this claim.') for sid in ids}
        third_panels, third_leaves = self.panels(third_index, ('third',), overrides=third_overrides)
        third = self.collect(third_index, third_panels, name='third-collected', third=True)[0]
        composed = load(third)
        self.assertEqual(composed['packet_sha256'], sha(self.packet_path))
        self.assertEqual(composed['adjudication_packet_sha256'], sha(adjudication_path))
        self.assertEqual(set(composed['judgments']), set(ids))
        for path in third_leaves[0]:
            self.assertNotIn('adjudication_packet_sha256', load(path))
        # A different panel name cannot conceal reuse of either original person.
        for number, primary_report in enumerate(reports):
            first_sid = load(third_index)['children'][0]['sample_ids'][0]
            reused_id = load(primary_report)['composition']['samples'][first_sid]['reviewer_id']
            panel_name = f'reused-third-{number}'
            reused_panels, _ = self.panels(third_index, (panel_name,),
                                          reviewer=lambda panel, child, identity=reused_id: identity)
            reused = self.collect(third_index, reused_panels, name=panel_name + '-collected', third=True)[0]
            with self.assertRaises(ValueError):
                assessment.freeze(self.root, self.packet_path, reports, self.root / f'reused-third-{number}.json',
                                  third=reused, adjudication_path=adjudication_path)
        judgments = self.fixture.freeze(self.packet_path, reports, third=third,
                                         adjudication_path=adjudication_path)
        frozen = load(judgments)
        self.assertEqual(set(frozen['disagreements']), set(ids))
        self.assertEqual(len(frozen['judgments']), 480)
        for sid in ids:
            self.assertEqual(frozen['judgments'][sid], composed['judgments'][sid])
        result = self.fixture.aggregate(judgments, self.mapping_path)
        semantic_passes = sum(result[suite]['arms'][arm]['semantic_pass']
                              for suite in ('primary', 'secondary') for arm in ARMS)
        self.assertEqual(semantic_passes, 460)


if __name__ == '__main__':
    unittest.main()
