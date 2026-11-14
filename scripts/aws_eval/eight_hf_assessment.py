#!/usr/bin/env python3
"""Parent-local blinded assessment. Never run inside the GPU training bundle.

prepare --root ROOT --run T
packets --root ROOT --run T --plan PLAN --outputs EXPORT --out LOCAL --stage dev|final
adjudication --packet PACKET --reports REVIEW_A REVIEW_B --out DIRECTORY
aggregate --packet PACKET --mapping PRIVATE_MAP --reports A B [--third C]
          --out DIRECTORY [--plan PLAN --export-sha256 SHA --ledger LEDGER]

Review reports use trid-eight-blind-review-v1, packet_sha256, reviewer_id,
independent:true, blinded:true, and judgments keyed by opaque sample_id. Each
judgment has semantic_pass:boolean, format_pass:boolean, critical_errors:[str],
reason:nonempty str. Final gate membership comes only from pre-output frozen
reference-only gate metadata, never from grader-selected labels. Never give a
reviewer the private mapping, arm names, selection scores or another review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREPARATION = 'runs/mois-eight-experiment-prep-20261114T053819Z'
PREPARATION_SHA = '8aa882deeb7bda1c618108663f60c956b11613dad17d943df5700a981910a4ea'
SUITES = {
    'general': ('runs/mois-independent-acceptance-20261113T120545Z-v3/readiness.json',
                '6191c60f9982b5cfed5d8d99f27cf971924d243123cc4f95403f1fb628643c5a'),
    'mois': ('runs/mois-independent-acceptance-20261113T120545Z/mois-evaluation-readiness.json',
             'e8b76b3a9d831c015d9df4811e14915af65697dae8eda03e97432c117977d3f6'),
}
ARMS = ('base', 'step-000012', 'step-000024', 'step-000036')
BEHAVIORS = ('conditions', 'exceptions', 'normative_force', 'information_boundary')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _contained_path(path, root):
    """Return resolved file and relative name; spelling alone cannot prove aliasing."""
    path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
    try:
        return path, path.relative_to(root)
    except ValueError:
        # Normalization-insensitive filesystems can resolve one directory under
        # different Unicode spellings. Accept only a same-directory ancestor,
        # never normalized strings, matching bytes, or a same-file leaf elsewhere.
        for ancestor in path.parents:
            try:
                same_directory = ancestor.samefile(root)
            except OSError:
                continue
            if same_directory:
                return path, path.relative_to(ancestor)
        raise ValueError('pin outside repository') from None


def pin(path, root):
    path, relative = _contained_path(path, root)
    return {'path': str(relative), 'sha256': digest(path), 'bytes': path.stat().st_size}


def bound(root, entry):
    path, _ = _contained_path(Path(root) / entry['path'], root)
    require(digest(path) == entry['sha256'], 'pin digest mismatch: ' + entry['path'])
    if 'bytes' in entry:
        require(path.stat().st_size == entry['bytes'], 'pin size mismatch')
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(path.read_text() == content, 'refusing to replace frozen artifact: ' + str(path))
    else:
        path.write_text(content)


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if destination.exists():
        require(destination.read_bytes() == content, 'frozen copy mismatch')
    else:
        destination.write_bytes(content)


def prepare(root, run):
    """Only closed prep metadata and final public input bytes are opened here."""
    root, run = Path(root).resolve(), Path(run).resolve()
    design_path = run / 'experiment-design.json'
    design = load(design_path)
    prep = root / PREPARATION
    result_path = prep / 'preparation-result.json'
    result = load(result_path)
    require(result['state'] == 'PREPARATION_COMPLETE_NOT_TRAINED', 'preparation not closed')
    require(result['rows'] == {'train': 24, 'dev': 8}, 'preparation count')
    require(result['manifest']['sha256'] == PREPARATION_SHA, 'preparation manifest identity')
    source_manifest_path = bound(root, result['manifest'])
    source_manifest = load(source_manifest_path)
    # Hash existing closed reports; do not reopen protected reservation/gold sources.
    closure_names = ('semantic_review', 'source_and_reservation_admission', 'source_attribution',
                     'configured_reservation_checks', 'fixed_evaluation_fingerprint_check',
                     'native_input_and_random_cpu_smoke', 'code_review_and_tests')
    closure_pins = {name: result[name] for name in closure_names}
    for entry in closure_pins.values():
        bound(root, entry)
    semantic = load(bound(root, result['semantic_review']))
    require(semantic['rows'] == 32 and semantic['all_message_and_answer_hashes_verified'] is True,
            'semantic closure missing')
    require(len(semantic['reviews']) == 2 and all(r['pass'] == 32 and r['fail'] == 0
            for r in semantic['reviews'].values()), 'two independent preparation reviews required')
    for review in semantic['reviews'].values():
        bound(root, review['report'])
    admission = load(bound(root, result['source_and_reservation_admission']))
    require(admission['decision'] == 'PASS_EXACT32_SOURCE_RIGHTS_AND_RESERVATION_GATE_WITH_EXPLICIT_LIMITATIONS'
            and not admission['blockers'], 'source admission not closed')
    require(load(bound(root, result['configured_reservation_checks']))['all_commands_completed'] is True,
            'reservation checks incomplete')
    data = run / 'execution-data'
    split_rows = {}
    for split, key, epochs, count in [('train', 'train', 3, 24), ('dev', 'valid', 1, 8)]:
        require(len(source_manifest[key]) == 1, 'unexpected manifest entries')
        entry = source_manifest[key][0]
        source = source_manifest_path.parent / entry['path']
        require(digest(source) == entry['sha256'], 'candidate pin mismatch')
        values = rows(source)
        require(len(values) == count and all(r['split'] == split for r in values), 'split mismatch')
        split_rows[split] = values
        copy_exact(source, data / source.name)
    train, dev = split_rows['train'], split_rows['dev']
    require(not ({r['id'] for r in train} & {r['id'] for r in dev}), 'cross-split identity leak')
    require(not ({r['provenance']['family_id'] for r in train} &
                 {r['provenance']['family_id'] for r in dev}), 'cross-split source leak')
    for split, values, expected in [('train', train, 6), ('dev', dev, 2)]:
        require(len({r['id'] for r in values}) == len(values), 'duplicate ' + split)
        require({b: sum(r['behavior'] == b for r in values) for b in BEHAVIORS} ==
                {b: expected for b in BEHAVIORS}, 'behavior allocation changed')
    manifest = {k: source_manifest[k] for k in ('model_id', 'model_revision', 'family',
                'enable_thinking', 'max_tokens', 'max_answer_tokens')}
    manifest.update(schema='trid-eight-sft-execution-manifest-v1', state='DERIVED_FOR_NEW_AUTHORIZATION',
        lineage={'preparation_manifest_sha256': PREPARATION_SHA,
                 'preparation_manifest': result['manifest'], 'preparation_result': pin(result_path, root),
                 'design': pin(design_path, root), 'closure_reports': closure_pins,
                 'messages_and_answers_unchanged': True, 'historical_dev_train_exposure': True},
        train=[dict(source_manifest['train'][0], epochs=3, rows=24)],
        valid=[dict(source_manifest['valid'][0], epochs=1, rows=8)])
    save(data / 'native-train-manifest.json', manifest)
    save(data / 'dev-inputs.json', [{'id': r['id'], 'messages': r['messages']} for r in dev])
    final_inputs, final_metadata, suites = [], {}, {}
    for cohort, (ready_name, expected_sha) in SUITES.items():
        ready_path = root / ready_name
        ready = load(ready_path)
        require(ready['freeze_manifest']['sha256'] == expected_sha, 'suite freeze changed')
        freeze_path = bound(root, ready['freeze_manifest'])
        freeze = load(freeze_path)
        if cohort == 'general':
            require(ready['independent_review']['combined_general_reference_pass_count'] == 12 and
                    ready['independent_review']['combined_general_reference_fail_count'] == 0, 'general not ready')
            name = 'general-inputs.jsonl'
            input_pin = dict(freeze['members'][name], path=str((freeze_path.parent / name).relative_to(root)))
            refs = [dict(v, path=str((freeze_path.parent / k).relative_to(root)))
                    for k, v in freeze['members'].items() if k in freeze['private_members']]
        else:
            require(ready['unique_cases'] == 12 and ready['all_manifest_file_and_input_pins_verified'] is True,
                    'MOIS not ready')
            name = 'public-prompts.jsonl'
            input_pin = next(dict(e, path=str((freeze_path.parent / e['path']).relative_to(root)))
                             for e in freeze['files'] if e['path'] == name)
            refs = [dict(e, path=str((freeze_path.parent / e['path']).relative_to(root)))
                    for e in freeze['files'] if e['visibility'] == 'reviewer_only']
        public = rows(bound(root, input_pin))
        require(len(public) == 12 and len({r['id'] for r in public}) == 12, 'final suite count')
        for row in public:
            if cohort == 'general':
                require(set(row) == {'id', 'messages'}, 'general transport fields')
                item = row
                metadata = {'cohort': cohort, 'category': 'general', 'source_clusters': []}
            else:
                item = {'id': row['id'], 'messages': [
                    {'role': 'system', 'content': row['instructions']},
                    {'role': 'user', 'content': canonical({k: row[k] for k in ('question', 'sources', 'answer_schema')}).decode()}]}
                metadata = {'cohort': cohort, 'category': row['metadata']['task_family'],
                            'source_clusters': sorted({s['family_id'] for s in row['sources']})}
            final_inputs.append(item)
            final_metadata[item['id']] = metadata
        suites[cohort] = {'readiness': pin(ready_path, root), 'freeze': ready['freeze_manifest'],
                         'public_inputs': input_pin, 'private_reference_pins': refs,
                         'case_ids': [r['id'] for r in public]}
    require(len(final_metadata) == 24, 'duplicate cross-suite final IDs')
    private = run / 'final-private'
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    save(private / 'final-inputs.json', final_inputs)
    save(private / 'final-case-metadata.json', final_metadata)
    protocol = {'schema': 'trid-eight-experimental-paired-protocol-v1', 'run_id': design['run_id'],
        'design': pin(design_path, root), 'suites': suites, 'final_inputs': pin(private / 'final-inputs.json', root),
        'metadata': pin(private / 'final-case-metadata.json', root),
        'input_order': [r['id'] for r in final_inputs],
        'transport': 'general unchanged id/messages; MOIS system=instructions; user=canonical UTF8 JSON of question,sources,answer_schema',
        'final_policy': design['final_policy'], 'development_selection': design['development_selection'],
        'private_references_opened_by_preparer': False, 'initial_bundle_contains_final_inputs': False,
        'historical_failed_gate_unchanged': True, 'denominators': {'general': 12, 'mois': 12},
        'missing_invalid_truncated_fail': True, 'semantic_grading': 'two independent blinded local reviewers; disagreement-only third',
        'interpretation': 'Small diagnostic sample only; no broad accuracy or deployment qualification.'}
    save(run / 'experimental-paired-protocol.json', protocol)
    closure = {'schema': 'trid-eight-sft-source-closure-v1', 'run_id': design['run_id'],
        'preparation_manifest_sha256': PREPARATION_SHA,
        'manifest_sha256': digest(data / 'native-train-manifest.json'),
        'train_sha256': digest(data / 'train-candidates.jsonl'), 'dev_sha256': digest(data / 'dev-candidates.jsonl'),
        'development_inputs_sha256': digest(data / 'dev-inputs.json'),
        'final_inputs_sha256': digest(private / 'final-inputs.json'),
        'private_references_in_bundle': False, 'final_inputs_in_bundle': False}
    save(run / 'source-closure.json', closure)
    save(run / 'assessment-plan-fields.json', {
        'data': {key: pin(path, root) for key, path in {
            'manifest': data / 'native-train-manifest.json', 'preparation_manifest': source_manifest_path,
            'train': data / 'train-candidates.jsonl', 'dev': data / 'dev-candidates.jsonl'}.items()},
        'workload': {'development_inputs': pin(data / 'dev-inputs.json', root),
            'development_case_ids': [r['id'] for r in dev],
            'development_behaviors': {r['id']: r['behavior'] for r in dev},
            'final_inputs': pin(private / 'final-inputs.json', root),
            'final_case_ids': {k: v['case_ids'] for k, v in suites.items()},
            'source_closure': pin(run / 'source-closure.json', root)}})
    return {'train': 24, 'dev': 8, 'final': 24, 'plan_fields': str(run / 'assessment-plan-fields.json')}


def effective_generation(root, run, amendment_pin):
    """Authenticate the pre-output amendment without changing the training recipe."""
    root, run = Path(root).resolve(), Path(run).resolve()
    design_path = run / 'experiment-design.json'
    design = load(design_path)
    amendment = load(bound(root, amendment_pin))
    require(amendment['schema'] == 'trid-eight-execution-amendment-v1'
            and amendment['run_id'] == design['run_id']
            and amendment['base_design'] == pin(design_path, root)
            and amendment['frozen_before_outputs'] is True
            and amendment['training_recipe_unchanged'] is True
            and amendment['training_context_tokens'] == design['recipe']['context_tokens'] == 3072
            and amendment['answer_tokens_unchanged'] == design['recipe']['answer_tokens'] == 768
            and amendment['input_bytes_unchanged'] is True
            and amendment['no_trimming_or_drops'] is True
            and amendment['generation'] == dict(design['generation'], context_tokens=4096),
            'effective generation amendment differs from frozen pre-output scope')
    return amendment['generation']


def audit_prompts(root, run, tokenizer_path, out, *, execution_amendment):
    """Offline rendering only. Parent runs after integration; never prints text."""
    from transformers import AutoTokenizer
    root, run, tokenizer_path = map(lambda p: Path(p).resolve(), (root, run, tokenizer_path))
    fields = load(run / 'assessment-plan-fields.json')['workload']
    protocol_path = run / 'experimental-paired-protocol.json'
    protocol = load(protocol_path)
    generation = effective_generation(root, run, execution_amendment)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False, local_files_only=True)
    records = []
    for cohort, entry, expected in [('dev', fields['development_inputs'], 8),
                                     ('final', fields['final_inputs'], 24)]:
        cases = load(bound(root, entry))
        require(len(cases) == expected, 'prompt audit denominator')
        for item in cases:
            rendered = tokenizer.apply_chat_template(item['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
            require(len(tokens) + generation['max_new_tokens'] <= generation['context_tokens'],
                    'prompt plus complete answer budget exceeds frozen context; no trimming allowed')
            records.append({'id': item['id'], 'cohort': cohort,
                'messages_sha256': object_sha(item['messages']),
                'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                'prompt_token_ids_sha256': object_sha(tokens), 'prompt_tokens': len(tokens),
                'reserved_answer_tokens': generation['max_new_tokens'],
                'total_budget_tokens': len(tokens) + generation['max_new_tokens']})
    token_files = [p for p in tokenizer_path.iterdir() if p.is_file()]
    receipt = {'schema': 'trid-eight-native-prompt-audit-v1',
        'run_id': load(run / 'experiment-design.json')['run_id'], 'generation': generation,
        'execution_amendment': execution_amendment,
        'development_inputs': fields['development_inputs'], 'final_inputs': fields['final_inputs'],
        'protocol': pin(protocol_path, root), 'transport_version': 'trid-eight-public-input-transport-v1',
        'source_input_pins': {key: value['public_inputs'] for key, value in protocol['suites'].items()},
        'tokenizer_files': [pin(p, root) for p in sorted(token_files)],
        'rows': records, 'ordered_ids_sha256': object_sha([r['id'] for r in records]),
        'matched_final_arms_required': True, 'truncated_rows': 0, 'dropped_rows': 0,
        'all_prompt_plus_answer_budgets_fit': True}
    save(out, receipt)
    return {'rows': len(records), 'dev': 8, 'final': 24,
            'max_budget_tokens': max(r['total_budget_tokens'] for r in records),
            'audit_sha256': digest(out)}


def structural_result(case, output, generation, *, validate_output=None):
    """Production checks for MOIS; general rubric format remains human-assessed."""
    if output is None:
        return {'pass': False, 'reason': 'missing_output'}
    if (output.get('termination_reason') != 'eos' or output.get('terminal_stop_token_id') is None
            or output.get('generation') != generation or output.get('exception')
            or output.get('context_truncated') or output.get('skipped_context')
            or output.get('truncated_output') or not output.get('output_token_ids')
            or output.get('generated_tokens') != len(output['output_token_ids'])
            or not output.get('prompt_token_ids')
            or len(output['prompt_token_ids']) + len(output['output_token_ids']) > generation['context_tokens']
            or len(output['output_token_ids']) > generation['max_new_tokens']
            or not isinstance(output.get('rendered_prompt_sha256'), str)
            or len(output['rendered_prompt_sha256']) != 64
            or output['output_token_ids'][-1] != output['terminal_stop_token_id']
            or output.get('prompt_sha256') != object_sha(case['messages'])
            or output.get('messages') != case['messages']):
        return {'pass': False, 'reason': 'invalid_finish_or_input_metadata'}
    raw = output.get('response_text')
    if not isinstance(raw, str) or not raw.strip():
        return {'pass': False, 'reason': 'empty_output'}
    try:
        payload = json.loads(case['messages'][-1]['content'])
    except (ValueError, TypeError):
        payload = {}
    if 'source_data' in payload:
        packet = payload['source_data']
        evidence = {e['id']: e for e in packet['evidence']}
    elif 'sources' in payload and 'answer_schema' in payload:
        # Adapt supplied public source text to the existing production validator;
        # source IDs stay unchanged and only cited text/metadata licenses numbers.
        packet = {'documents': [{'doc_id': s['id'], 'title': s['title'], 'date': s['version_date']}
                                for s in payload['sources']],
                  'evidence': [{'id': s['id'], 'doc_id': s['id'], 'quote': s['text']}
                               for s in payload['sources']]}
        evidence = {e['id']: e for e in packet['evidence']}
    else:
        return {'pass': True, 'reason': 'finish_valid_general_format_requires_frozen_rubric_review'}
    try:
        if validate_output is None:
            from trid.service.mois_graph_generation import _validate_output
            validate_output = _validate_output
        validate_output(raw, evidence, packet)
    except (ValueError, KeyError, TypeError) as error:
        return {'pass': False, 'reason': str(error)}
    return {'pass': True, 'reason': 'production_structure_only_not_entailment'}


def make_packets(root, run, plan_path, outputs, out, stage):
    root, run, outputs, out = map(Path, (root, run, outputs, out))
    plan = load(plan_path)
    plan_sha = digest(plan_path)
    design = load(run / 'experiment-design.json')
    amendment_pin = plan['workload']['execution_amendment']
    generation = effective_generation(root, run, amendment_pin)
    require(generation == plan['workload']['generation'], 'plan effective generation mismatch')
    audit_pin = plan['workload']['prompt_audit']
    audit = load(bound(root, audit_pin))
    require(audit['schema'] == 'trid-eight-native-prompt-audit-v1'
            and audit['run_id'] == plan['identity']['run_id']
            and audit['generation'] == generation
            and audit['execution_amendment'] == amendment_pin
            and audit['development_inputs'] == plan['workload']['development_inputs']
            and audit['final_inputs'] == plan['workload']['final_inputs']
            and audit['all_prompt_plus_answer_budgets_fit'] is True,
            'prompt audit pin mismatch')
    audited = {row['id']: row for row in audit['rows']}
    require(not (out / 'private-mapping.json').exists(), 'packet already frozen; never rerandomize')
    if stage == 'dev':
        candidates = rows(run / 'execution-data/dev-candidates.jsonl')
        cases = [{'id': r['id'], 'messages': r['messages']} for r in candidates]
        metadata = {r['id']: {'cohort': 'dev', 'behavior': r['behavior'], 'reference': r['answer']} for r in candidates}
        files = dict(zip(ARMS, ['baseline-dev-outputs.jsonl'] +
                     [f'checkpoint-step-{s:06d}-dev-outputs.jsonl' for s in (12, 24, 36)]))
        reference_pins = []
    else:
        selection = load(outputs / 'checkpoint-selection.json')
        require(selection['run_id'] == plan['identity']['run_id'] and selection['plan_sha256'] == plan_sha
                and selection['selected_step'] in (12, 24, 36), 'final requires selected checkpoint')
        cases = load(bound(root, plan['workload']['final_inputs']))
        protocol = load(run / 'experimental-paired-protocol.json')
        metadata = load(bound(root, protocol['metadata']))
        gate_pin = plan['workload']['final_gate_metadata']
        gates = load(bound(root, gate_pin))
        require(gates['schema'] == 'trid-eight-final-gate-metadata-v1'
                and gates['run_id'] == plan['identity']['run_id']
                and gates['final_inputs_sha256'] == plan['workload']['final_inputs']['sha256']
                and gates['frozen_before_outputs'] is True
                and gates['public_input_conversion_review'] == 'PASS'
                and gates['suite_freeze_sha256'] == {k: v[1] for k, v in SUITES.items()},
                'final gate freeze mismatch')
        refusal_ids = gates['required_supported_refusal_case_ids']
        require(len(refusal_ids) == len(set(refusal_ids)) == gates['supported_refusal_count'] == 3
                and set(refusal_ids).issubset(plan['workload']['final_case_ids']['mois']),
                'frozen supported-refusal gate differs from pre-output three-case clarification')
        metadata = {key: dict(value, supported_refusal=key in refusal_ids)
                    for key, value in metadata.items()}
        files = {'base': 'final-base-outputs.jsonl', f"step-{selection['selected_step']:06d}": 'final-adapter-outputs.jsonl'}
        reference_pins = [pin for suite in protocol['suites'].values() for pin in suite['private_reference_pins']]
    packet_rows, mapping, output_pins = [], {}, {}
    ids = [c['id'] for c in cases]
    for arm, name in files.items():
        path = outputs / name
        observed = rows(path) if path.exists() else []
        require(len({r['id'] for r in observed}) == len(observed), 'duplicate output IDs')
        require(all(r['id'] in ids for r in observed), 'unexpected output ID')
        require(all(r['run_id'] == plan['identity']['run_id'] and r['plan_sha256'] == plan_sha and r['arm'] == arm
                    for r in observed), 'output identity mismatch')
        output_pins[arm] = {'sha256': digest(path), 'bytes': path.stat().st_size, 'path': name} if path.exists() else None
        by_id = {r['id']: r for r in observed}
        for case in cases:
            sample = secrets.token_hex(16)
            output = by_id.get(case['id'])
            meta = metadata[case['id']]
            packet_row = {'sample_id': sample, 'case_id': case['id'], 'messages': case['messages'],
                          'response': output.get('response_text') if output else None,
                          'structural': structural_result(case, output, generation)}
            expected = audited[case['id']]
            if output is not None and (
                    expected['messages_sha256'] != output.get('prompt_sha256')
                    or expected['rendered_prompt_sha256'] != output.get('rendered_prompt_sha256')
                    or expected['prompt_token_ids_sha256'] != object_sha(output.get('prompt_token_ids'))
                    or expected['prompt_tokens'] != len(output.get('prompt_token_ids', []))):
                packet_row['structural'] = {'pass': False, 'reason': 'frozen_native_prompt_audit_mismatch'}
            if stage == 'dev':
                packet_row['local_reference'] = meta['reference']
            packet_rows.append(packet_row)
            mapping[sample] = {'case_id': case['id'], 'arm': arm,
                               **{k: v for k, v in meta.items() if k != 'reference'}}
    secrets.SystemRandom().shuffle(packet_rows)
    packet = {'schema': 'trid-eight-blind-packet-v1', 'stage': stage,
        'run_id': plan['identity']['run_id'], 'plan_sha256': plan_sha, 'rows': packet_rows,
        'mapping_commitment': object_sha({'rows': mapping, 'output_pins': output_pins,
                                         'arms': list(files), 'case_ids': ids}),
        'reference_pins': reference_pins, 'semantic_policy': design['development_selection']['semantic_pass'],
        'critical_error_policy': design['development_selection']['critical_errors'],
        'prompt_audit_pin': audit_pin,
        'instructions': 'Independent blinded judgments. Semantic paraphrases allowed; never exact target match. '
            'False structural pass cannot be overridden. All missing/truncated outputs fail. '
            'Final assessors alone open the pinned unchanged private references/rubrics locally; '
            'Gate membership is fixed by pre-output reference-only metadata; never label or change it.'}
    if stage == 'final':
        packet['final_gate_metadata'] = {'pin': gate_pin, 'metadata': gates}
    save(out / 'review-packet.json', packet)
    save(out / 'private-mapping.json', {'schema': 'trid-eight-blind-mapping-v1',
        'packet_sha256': digest(out / 'review-packet.json'), 'rows': mapping,
        'output_pins': output_pins, 'arms': list(files), 'case_ids': ids})
    (out / 'private-mapping.json').chmod(0o600)
    return {'packet': str(out / 'review-packet.json'), 'rows': len(packet_rows)}


def validate_report(report, packet, packet_sha, expected_ids=None):
    expected = set(expected_ids if expected_ids is not None else [r['sample_id'] for r in packet['rows']])
    require(report.get('schema') == 'trid-eight-blind-review-v1' and report.get('packet_sha256') == packet_sha,
            'review packet identity')
    require(isinstance(report.get('reviewer_id'), str) and report['reviewer_id'].strip()
            and report.get('independent') is True and report.get('blinded') is True, 'independent reviewer required')
    judgments = report.get('judgments')
    require(isinstance(judgments, dict) and set(judgments) == expected, 'review denominator mismatch')
    for judgment in judgments.values():
        require(isinstance(judgment, dict) and set(judgment) ==
                {'semantic_pass', 'format_pass', 'critical_errors', 'reason'},
                'review judgment has undeclared fields')
        require(type(judgment.get('semantic_pass')) is bool and type(judgment.get('format_pass')) is bool,
                'review pass fields must be boolean')
        errors = judgment.get('critical_errors')
        require(isinstance(errors, list) and all(isinstance(e, str) and e.strip() for e in errors)
                and len(errors) == len(set(errors)), 'critical errors must be unique nonempty strings')
        require(isinstance(judgment.get('reason'), str) and judgment['reason'].strip(), 'review reason missing')
        require('supported_refusal' not in judgment, 'reviewers cannot choose frozen gate membership')
    return judgments


def signature(judgment, final):
    # Reasons may differ without changing the decision; critical error descriptions
    # are retained and disagreements in their content require adjudication.
    return (judgment['semantic_pass'], judgment['format_pass'], tuple(sorted(judgment['critical_errors'])))


def review_pair(packet_path, report_paths):
    require(len(report_paths) == 2, 'exactly two primary reports required')
    packet = load(packet_path)
    reports = [load(p) for p in report_paths]
    require(reports[0]['reviewer_id'] != reports[1]['reviewer_id'], 'reviewers must differ')
    judgments = [validate_report(r, packet, digest(packet_path)) for r in reports]
    disagreements = sorted(k for k in judgments[0] if signature(judgments[0][k], packet['stage'] == 'final') !=
                            signature(judgments[1][k], packet['stage'] == 'final'))
    return packet, reports, judgments, disagreements


def adjudication(packet_path, report_paths, out):
    packet, reports, judgments, disagreements = review_pair(packet_path, report_paths)
    # Do not expose primary votes, arm identities, or scores to the third reviewer.
    save(Path(out) / 'adjudication-packet.json', {**packet,
        'parent_packet_sha256': digest(packet_path),
        'rows': [r for r in packet['rows'] if r['sample_id'] in disagreements],
        'report_packet_sha256': digest(packet_path),
        'instructions': packet['instructions'] + ' Report against parent packet SHA; only these flagged sample IDs.'})
    save(Path(out) / 'disagreements.json', {'packet_sha256': digest(packet_path), 'sample_ids': disagreements,
        'primary_report_sha256': [digest(p) for p in report_paths]})
    return {'disagreements': len(disagreements), 'third_required': bool(disagreements)}


def resolve_reviews(packet_path, mapping_path, report_paths, third_path=None):
    packet, reports, judgments, disagreements = review_pair(packet_path, report_paths)
    mapping = load(mapping_path)
    require(mapping['packet_sha256'] == digest(packet_path) and
            set(mapping['rows']) == set(judgments[0]), 'private mapping pin mismatch')
    require(packet['mapping_commitment'] == object_sha({k: mapping[k] for k in
            ('rows', 'output_pins', 'arms', 'case_ids')}), 'mapping commitment changed')
    if disagreements:
        require(third_path is not None, 'disagreements require blinded third reviewer')
        third = load(third_path)
        require(third['reviewer_id'] not in {r['reviewer_id'] for r in reports}, 'third must be independent')
        third_judgments = validate_report(third, packet, digest(packet_path), disagreements)
    else:
        require(third_path is None, 'third review forbidden without disagreements')
        third_judgments = {}
    results = []
    for row in packet['rows']:
        sid = row['sample_id']
        judgment = third_judgments.get(sid, judgments[0][sid])
        results.append({'sample_id': sid, **mapping['rows'][sid], **judgment,
            'structural_pass': row['structural']['pass'],
            'joint_pass': bool(row['structural']['pass'] and judgment['semantic_pass']
                               and judgment['format_pass'] and not judgment['critical_errors'])})
    return packet, mapping, results, disagreements


def dev_aggregate(results, ledger):
    checkpoints = {c['step']: c for c in ledger['checkpoints']}
    require(set(checkpoints) == {12, 24, 36}, 'all checkpoint losses required')
    aggregate = {'candidates': []}
    for arm in ARMS:
        subset = [r for r in results if r['arm'] == arm]
        require(len(subset) == 8 and len({r['case_id'] for r in subset}) == 8, 'dev denominator')
        value = {'joint_pass_count': sum(r['joint_pass'] for r in subset),
            'critical_errors': sum(bool(r['critical_errors']) for r in subset),
            'behavior_pass_counts': {b: sum(r['joint_pass'] for r in subset if r['behavior'] == b) for b in BEHAVIORS}}
        if arm == 'base':
            aggregate['base'] = value
        else:
            step = int(arm.split('-')[1])
            losses = checkpoints[step]['valid_loss_by_task']
            require(set(losses) == {'mois_grounded_explanation'}, 'unexpected validation tasks')
            loss = losses['mois_grounded_explanation']
            require(type(loss) in (float, int) and math.isfinite(loss) and loss >= 0, 'invalid dev loss')
            aggregate['candidates'].append(dict(value, step=step, teacher_forced_dev_loss=loss))
    base = aggregate['base']
    eligible = [c for c in aggregate['candidates'] if c['joint_pass_count'] >= 6 and c['critical_errors'] == 0
                and all(c['behavior_pass_counts'][b] >= base['behavior_pass_counts'][b] for b in BEHAVIORS)]
    eligible.sort(key=lambda c: (-c['joint_pass_count'], c['teacher_forced_dev_loss'], c['step']))
    return aggregate, eligible[0]['step'] if eligible else None


def paired_aggregate(results, arms, gates):
    require(len(arms) == 2 and 'base' in arms, 'paired arm identities')
    require(gates['frozen_before_outputs'] is True and gates['supported_refusal_count'] == 3
            and len(set(gates['required_supported_refusal_case_ids'])) == 3,
            'pre-output three-case refusal freeze required')
    after_arm = next(a for a in arms if a != 'base')
    cohorts = {}
    for cohort in ('general', 'mois'):
        before = {r['case_id']: r for r in results if r['cohort'] == cohort and r['arm'] == 'base'}
        after = {r['case_id']: r for r in results if r['cohort'] == cohort and r['arm'] == after_arm}
        require(len(before) == len(after) == 12 and set(before) == set(after), 'final denominator must stay twelve')
        require(all(before[k]['supported_refusal'] == after[k]['supported_refusal'] for k in before),
                'reference refusal classification cannot differ by arm')
        gains = [k for k in before if not before[k]['joint_pass'] and after[k]['joint_pass']]
        regressions = [k for k in before if before[k]['joint_pass'] and not after[k]['joint_pass']]
        categories = {c: {'total': sum(r['category'] == c for r in after.values()),
                          'before_pass': sum(r['category'] == c and r['joint_pass'] for r in before.values()),
                          'after_pass': sum(r['category'] == c and r['joint_pass'] for r in after.values())}
                      for c in sorted({r['category'] for r in after.values()})}
        clusters = {c: {'case_ids': [k for k, r in after.items() if c in r['source_clusters']],
                       'before_pass': sum(r['joint_pass'] for r in before.values() if c in r['source_clusters']),
                       'after_pass': sum(r['joint_pass'] for r in after.values() if c in r['source_clusters'])}
                    for c in sorted({c for r in after.values() for c in r['source_clusters']})}
        count = sum(r['joint_pass'] for r in after.values())
        critical = sum(len(r['critical_errors']) for r in after.values())
        refusals = list(gates['required_supported_refusal_case_ids']) if cohort == 'mois' else []
        if cohort == 'general':
            gate = count >= 11 and not regressions
        else:
            require(set(refusals).issubset(after), 'frozen refusal case missing from denominator')
            gate = count >= 10 and critical == 0 and not regressions and all(after[k]['joint_pass'] for k in refusals) \
                   and all(c['after_pass'] > 0 for c in categories.values())
        cohorts[cohort] = {'denominator': 12, 'before_pass': sum(r['joint_pass'] for r in before.values()),
            'after_pass': count, 'after_critical_errors': critical, 'paired_gains': gains,
            'paired_regressions': regressions, 'categories': categories, 'source_clusters': clusters,
            'supported_refusal_cases': refusals, 'gate_pass': bool(gate)}
    return {'schema': 'trid-eight-paired-diagnostic-result-v1', 'cohorts': cohorts,
        'diagnostic_improvement': bool(cohorts['general']['gate_pass'] and cohorts['mois']['gate_pass']
                                       and cohorts['mois']['paired_gains']),
        'broad_accuracy_claim': False, 'deployment_qualified': False,
        'historical_failed_qualification_unchanged': True}


def aggregate(packet_path, mapping_path, report_paths, out, third_path=None,
              plan_path=None, export_sha=None, ledger_path=None):
    packet, mapping, results, disagreements = resolve_reviews(packet_path, mapping_path, report_paths, third_path)
    out = Path(out)
    review_paths = [*map(Path, report_paths)] + ([Path(third_path)] if third_path else [])
    review_pins = [pin(p.resolve(), ROOT) for p in review_paths]
    # Write/freeze blinded decisions before joining arm identities in any public result.
    save(out / 'frozen-blind-judgments.json', {'schema': 'trid-eight-frozen-blind-judgments-v1',
        'packet_sha256': digest(packet_path),
        'mapping_sha256': digest(mapping_path), 'review_pins': review_pins,
        'judgments': {r['sample_id']: {k: r[k] for k in
                      ('semantic_pass', 'format_pass', 'critical_errors', 'reason') if k in r}
                      for r in results}, 'disagreements': disagreements})
    if packet['stage'] == 'dev':
        require(plan_path and ledger_path and export_sha and len(export_sha) == 64, 'selection bindings required')
        plan, ledger = load(plan_path), load(ledger_path)
        require(packet['run_id'] == plan['identity']['run_id'] and packet['plan_sha256'] == digest(plan_path), 'plan mismatch')
        values, selected = dev_aggregate(results, ledger)
        decision = {'schema': 'trid-eight-sft-selection-v1', 'run_id': plan['identity']['run_id'],
            'plan_sha256': digest(plan_path), 'dev_export_sha256': export_sha,
            'checkpoint_hashes': {f"step-{c['step']:06d}": c['adapter_weights_sha256'] for c in ledger['checkpoints']},
            'review_pins': review_pins + [
                pin(p.resolve(), ROOT)
                for p in (Path(packet_path), Path(mapping_path), out / 'frozen-blind-judgments.json')],
            'aggregate': values, 'selected_step': selected}
        from scripts.aws_eval.eight_hf_experiment import validate_selection
        validate_selection(plan, digest(plan_path), decision, export_sha, decision['checkpoint_hashes'])
        save(out / 'checkpoint-selection.json', decision)
        result = {'selected_step': selected, 'status': 'SELECTED' if selected is not None else 'NO_ELIGIBLE_CHECKPOINT_TERMINATE'}
    else:
        result = paired_aggregate(results, mapping['arms'], packet['final_gate_metadata']['metadata'])
        result['final_gate_metadata_pin'] = packet['final_gate_metadata']['pin']
        save(out / 'paired-diagnostic-result.json', result)
    save(out / 'assessment-audit.json', {'packet_sha256': digest(packet_path),
        'mapping_sha256': digest(mapping_path), 'review_pins': review_pins,
        'output_pins': mapping['output_pins'], 'all_denominator_rows': results,
        'frozen_blind_judgments_sha256': digest(out / 'frozen-blind-judgments.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--root', type=Path, default=ROOT)
    prep.add_argument('--run', type=Path, required=True)
    audit = sub.add_parser('audit-prompts')
    audit.add_argument('--root', type=Path, default=ROOT)
    audit.add_argument('--run', type=Path, required=True)
    audit.add_argument('--tokenizer', type=Path, required=True)
    audit.add_argument('--out', type=Path, required=True)
    audit.add_argument('--execution-amendment-sha256', required=True)
    packets = sub.add_parser('packets')
    packets.add_argument('--root', type=Path, default=ROOT)
    for name in ('run', 'plan', 'outputs', 'out'):
        packets.add_argument('--' + name, type=Path, required=True)
    packets.add_argument('--stage', choices=('dev', 'final'), required=True)
    for name in ('adjudication', 'aggregate'):
        command = sub.add_parser(name)
        command.add_argument('--packet', type=Path, required=True)
        command.add_argument('--reports', nargs=2, type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        if name == 'aggregate':
            command.add_argument('--mapping', type=Path, required=True)
            command.add_argument('--third', type=Path)
            command.add_argument('--plan', type=Path)
            command.add_argument('--ledger', type=Path)
            command.add_argument('--export-sha256')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.run)
    elif args.command == 'audit-prompts':
        amendment_path = args.run.resolve() / 'execution-pre-output-amendment.json'
        amendment_pin = pin(amendment_path, args.root.resolve())
        require(amendment_pin['sha256'] == args.execution_amendment_sha256, 'execution amendment SHA mismatch')
        result = audit_prompts(args.root, args.run, args.tokenizer, args.out,
                               execution_amendment=amendment_pin)
    elif args.command == 'packets':
        result = make_packets(args.root, args.run, args.plan, args.outputs, args.out, args.stage)
    elif args.command == 'adjudication':
        result = adjudication(args.packet, args.reports, args.out)
    else:
        result = aggregate(args.packet, args.mapping, args.reports, args.out, args.third,
                           args.plan, args.export_sha256, args.ledger)
    # No prompts, references or model answers are printed by this CLI.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
