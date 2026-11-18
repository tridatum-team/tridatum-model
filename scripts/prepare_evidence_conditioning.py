"""Prepare and independently admit a separately versioned offline research corpus.

The old admitted 600/200 corpus is an immutable ancestor, not a semantic-review
receipt for new prompts, facts, fictional targets, or evidence-removal answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This module is metadata-only; importing the trainer would force torch even when
# the caller only wants to prepare a corpus. The parent separately exercises it.
from scripts.prepare_expanded_experiment import (
    SOURCE_GATES, TOKENIZER_ACQUISITION_SHA256, canonical, case_sha, file_sha,
    load_tokenizer, pin, read_json, trainer_row, validate_partitions, verified,
    write_json, write_jsonl,
)
from trid.service.evidence_conditioning import (
    PROFILE, build_evidence_messages, remap_packet, validate_evidence_answer,
    validate_packet,
)

MISSING_TARGET = {'paragraphs': [{'text': '제공된 자료에 근거가 없어 질문의 내용을 확인할 수 없습니다. 관련 원문이 제공되어야 판단할 수 있습니다.', 'sources': []}]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def indexed(values, label):
    result = {}
    for value in values:
        require(isinstance(value, dict) and isinstance(value.get('id'), str), f'{label}: missing id')
        require(value['id'] not in result, f'{label}: duplicate id {value["id"]}')
        result[value['id']] = value
    return result


def local_pin(path):
    path = Path(path)
    return {'path': path.name, 'bytes': path.stat().st_size, 'sha256': file_sha(path)}


def original_packet(record):
    require(len(record['messages']) == 2, 'source evidence messages must be system/user')
    payload = json.loads(record['messages'][1]['content'])
    require(set(payload) == {'question', 'source_data'}, 'unexpected source message payload')
    validate_packet(payload['source_data'])
    return payload['question'], deepcopy(payload['source_data'])


def load_sources(source_frozen, root=ROOT):
    """Authenticate every current author against the admitted, frozen projection.

    Only the declared train/development author pins are opened. Source receipts
    retain protected-exclusion commitments; protected final payloads are not read.
    """
    source_frozen, root = Path(source_frozen).resolve(), Path(root).resolve()
    frozen = read_json(source_frozen)
    require(frozen.get('schema') == 'trid-expanded-frozen-dataset-v1', 'unsupported source freeze')
    pins = frozen['pins']
    training_path = verified(pins['training_manifest'], root)
    training = read_json(training_path)
    require(training.get('state') == 'ADMITTED', 'source training manifest is not admitted')
    require(training['input_pins'] == frozen['input_pins'], 'frozen/current author pins disagree')
    audit_path = source_frozen.parent / 'admission-audit.json'
    audit = read_json(audit_path)
    require(audit.get('state') == 'ADMITTED' and not audit.get('blockers'), 'source audit not admitted')
    require(audit['input_pins'] == training['input_pins'], 'source audit input pins disagree')
    inputs = training['input_pins']
    require(inputs['source_allocation'] == pins['source_allocation'], 'allocation pins disagree')
    allocation = read_json(verified(pins['source_allocation'], root))
    design = read_json(verified(pins['design'], root))
    contract = read_json(verified(inputs['authoring_contract'], root))
    records = []
    for author_pin in inputs['authors']:
        name = Path(author_pin['path']).name.lower()
        require(('train' in name or 'evaluation' in name or 'development' in name)
                and 'final' not in name, 'author pin is not a declared train/development source')
        records.extend(rows(verified(author_pin, root)))
    by_id = indexed(records, 'source author')
    require(all(r['split'] in ('train', 'evaluation') for r in records), 'protected/unknown source split')
    validate_partitions(records, design, contract, allocation)
    hashes = {r['id']: case_sha(r) for r in records}
    require(hashes == audit['case_hashes'], 'source case hash coverage differs from admitted audit')
    for gate in ('source_binding', 'native_tokenizer_and_target_masking'):
        checks = audit['case_checks'][gate]
        require(len(checks) == len(records) and {c['id'] for c in checks} == set(by_id)
                and all(c['status'] == 'pass' for c in checks), f'incomplete original audit gate: {gate}')
    frozen_train = []
    for entry in training['train']:
        require(entry['adapter'] == 'grounded_messages', 'unexpected original training adapter')
        frozen_train.extend(rows(verified(entry, training_path.parent)))
    require(indexed(frozen_train, 'frozen training') ==
            {r['id']: trainer_row(r) for r in records if r['split'] == 'train'}, 'frozen training projection mismatch')
    public = indexed(read_json(verified(pins['evaluation_inputs'], root)), 'frozen development')
    refs = indexed(read_json(verified(pins['evaluation_references'], root)), 'frozen development reference')
    metadata = indexed(read_json(verified(pins['evaluation_metadata'], root)), 'frozen development metadata')
    development = {r['id']: r for r in records if r['split'] == 'evaluation'}
    require(set(public) == set(refs) == set(metadata) == set(development), 'development coverage mismatch')
    for ident, record in development.items():
        require(public[ident] == {'id': ident, 'messages': record['messages']}
                and refs[ident]['reference_answer'] == record['reference_answer']
                and refs[ident]['case_sha256'] == hashes[ident]
                and metadata[ident]['case_sha256'] == hashes[ident], 'frozen development projection mismatch')
    receipt = read_json(verified(inputs['source_admission'], root))
    require(receipt.get('schema') == 'trid-expanded-source-admission-v1'
            and receipt['source_allocation'] == pins['source_allocation']
            and receipt['case_hashes'] == hashes, 'source admission receipt binding mismatch')
    require(receipt.get('historically_unseen') is False
            and receipt.get('checkpoint_consumption') == 'UNKNOWN_NOT_CERTIFIED'
            and receipt.get('currentness') == 'OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED', 'source caveats lost')
    require(receipt.get('protected_case_review', {}).get('verdict') == 'pass', 'source protected review absent')
    admitted = {f['family_id']: f for f in receipt['families']}
    allocated = {f['family_id']: f for f in allocation['families']}
    require(set(admitted) == set(allocated), 'source family admission coverage differs')
    for family in admitted.values():
        require(all(family.get(g, {}).get('verdict') == 'pass' for g in SOURCE_GATES), 'source family not independently admitted')
    # Reauthenticate selected source packets and exact quotations, not a glob of
    # stale author files. Only admitted allocation families can supply donors.
    packet_cache = {}
    for record in records:
        if record['cohort'] == 'general':
            continue
        family = allocated[record['family_id']]
        provenance = record['provenance']
        packet_pin = provenance['source_packet']
        key = canonical(packet_pin)
        if key not in packet_cache:
            packet_cache[key] = read_json(verified(packet_pin, root))
            verified(family['raw_source'], root)
        source_packet = packet_cache[key]
        for field in ('family_id', 'doc_id', 'version', 'body_sha256'):
            require(source_packet.get(field) == family.get(field) == provenance.get(field), 'persistent source identity mismatch')
        _, packet = original_packet(record)
        passages = provenance['passage_pins']
        exact = []
        for passage in passages:
            matches = [p for p in source_packet['articles'] if all(p.get(k) == v for k, v in passage.items())]
            require(len(matches) == 1, 'admitted passage absent/ambiguous')
            text = matches[0]['text']
            require(hashlib.sha256(text.encode()).hexdigest() == passage['quote_sha256'], 'quotation digest mismatch')
            exact.append(text)
        require(set(exact) == {e['quote'] for e in packet['evidence'] if e['kind'] == 'quotation'}, 'source quotation coverage mismatch')
        require(provenance['institution'] == family['issuer'], 'actual institution mismatch')
    return records, {
        'source_frozen': pin(source_frozen, root), 'source_audit': pin(audit_path, root),
        'source_training_manifest': pins['training_manifest'], 'original_input_pins': inputs,
        'source_caveats': frozen['source_caveats'],
        'historically_unseen': False, 'checkpoint_consumption': 'UNKNOWN_NOT_CERTIFIED',
    }, design


def rubric_lines(value):
    if isinstance(value, list):
        require(all(isinstance(v, str) and v.strip() for v in value), 'invalid rubric')
        return list(value)
    require(isinstance(value, dict), 'invalid source rubric')
    return [f'{key}: {text}' for key, texts in value.items() for text in texts]


def remap_target(target, mapping):
    result = deepcopy(target)
    for paragraph in result['paragraphs']:
        paragraph['sources'] = [mapping[s] for s in paragraph['sources']]
    return result


def scoped_question(question, packet):
    documents = packet['documents']
    require(len(documents) == 1, 'original real-law packet must have one document')
    document = documents[0]
    require(document.get('institution') and document.get('title'), 'explicit institution/document scope required')
    return f'질문 대상은 {document["institution"]}의 「{document["title"]}」 수집본입니다. 다른 기관이나 다른 문서의 규정은 이 질문의 대상이 아닙니다. {question}'


def add_distractor(packet, donor):
    """Append one complete admitted quotation; never edit a real-law excerpt."""
    result = deepcopy(packet)
    _, extra = original_packet(donor)
    quotation = min(extra['evidence'], key=lambda e: (len(e['quote']), e['id']))
    document = next(d for d in extra['documents'] if d['doc_id'] == quotation['doc_id'])
    require(all((d['institution'], d['title']) != (document['institution'], document['title'])
                for d in result['documents']), 'ambiguous distractor scope')
    doc_id, evidence_id = f'D{len(result["documents"]) + 1}', f'E{len(result["evidence"]) + 1}'
    result['documents'].append({**deepcopy(document), 'doc_id': doc_id})
    result['evidence'].append({**deepcopy(quotation), 'doc_id': doc_id, 'id': evidence_id})
    validate_packet(result)
    return result, {evidence_id: {'parent_id': donor['id'], 'parent_case_sha256': case_sha(donor),
                               'original_evidence_id': quotation['id'], 'provenance': donor['provenance']}}


def overlay_facts(packet, annotation):
    result = deepcopy(packet)
    by_id = {e['id']: e for e in result['evidence']}
    facts = annotation['facts']
    require(isinstance(facts, list) and facts, 'empty fact annotation')
    require(len({f['id'] for f in facts}) == len(facts), 'duplicate annotation fact ID')
    mapping = {f['id']: f'E{len(by_id) + i + 1}' for i, f in enumerate(facts)}
    require(not set(mapping) & set(by_id), 'fact annotation ID collides with quotation')
    for fact in facts:
        sources = fact['supporting_sources']
        require(isinstance(sources, list) and sources and all(s in by_id and by_id[s]['kind'] == 'quotation' for s in sources), 'unknown fact supporting quote')
        quote = fact['support_quote']
        require(isinstance(quote, str) and quote.strip() and any(quote in by_id[s]['quote'] for s in sources), 'fact support_quote is not an exact supporting substring')
        owners = {by_id[s]['doc_id'] for s in sources}
        require(len(owners) == 1, 'fact cannot merge different document sources')
        item = {k: deepcopy(v) for k, v in fact.items() if k not in ('support_quote', 'id')}
        item.update(id=mapping[fact['id']], kind='fact', doc_id=next(iter(owners)))
        for container in (item, item.get('qualifiers', {})):
            for field in ('exception_to', 'source_qualifications'):
                if field in container:
                    value = container[field]
                    container[field] = ([mapping.get(v, v) for v in value] if isinstance(value, list)
                                        else mapping.get(value, value))
        result['evidence'].append(item)
    validate_packet(result)
    return result


def build_candidate(records, annotations):
    require(annotations.get('schema') == 'trid-evidence-annotations-v1', 'unsupported annotations schema')
    require(isinstance(annotations.get('author_identity'), str) and annotations['author_identity'].strip(),
            'annotations must bind their recorded author identity')
    parents = indexed(records, 'source')
    facts = {}
    for annotation in annotations['facts']:
        ident = annotation['parent_id']
        require(ident in parents and ident not in facts, 'unknown/duplicate annotated parent')
        parent = parents[ident]
        split = 'train' if parent['split'] == 'train' else 'development'
        require(parent['cohort'] != 'general' and annotation['split'] == split
                and annotation['parent_case_sha256'] == case_sha(parent), 'fact annotation parent binding mismatch')
        require(annotation['review_status'] == 'needs_independent_review', 'new annotations cannot inherit review approval')
        facts[ident] = annotation
    matrix, groups, review_items = [], [], []

    def emit(ident, family, split, condition, messages, target, rubric, provenance, **extra):
        row_type = extra.pop('row_type', 'evidence')
        if row_type == 'evidence':
            packet = json.loads(messages[1]['content'])['source_data']
            verdict = validate_evidence_answer(canonical(target), packet)
            require(verdict['valid'], f'{ident}: invalid authored target: {verdict["errors"]}')
        reference = dict(id=ident, family_id=family, split=split, condition=condition,
                         row_type=row_type, prompt_sha256=case_sha(messages), target=target,
                         rubric=rubric_lines(rubric), provenance=deepcopy(provenance), relation_groups=[],
                         review_status='needs_independent_review', **extra)
        matrix.append({'id': ident, 'messages': messages, 'reference': reference})
        return ident

    def group(family, relation, members):
        ident = f'{family}::{relation}'
        source_family = next(item['reference']['family_id'] for item in matrix if item['id'] == members[0])
        groups.append({'id': ident, 'family_id': source_family, 'relation': relation, 'members': members})
        for item in matrix:
            if item['id'] in members:
                item['reference']['relation_groups'].append(ident)

    real = [r for r in records if r['cohort'] != 'general']
    donor_order = sorted(real, key=lambda r: (
        min(len(e['quote']) for e in original_packet(r)[1]['evidence']), r['id']))
    document_scopes = {}
    for record in real:
        _, source_data = original_packet(record)
        require(len(source_data['documents']) == 1, 'original real-law packet must have one document')
        document = source_data['documents'][0]
        document_scopes[record['id']] = (document['institution'], document['title'])
    for parent in records:
        split = 'train' if parent['split'] == 'train' else 'development'
        shared = dict(parent_id=parent['id'], parent_case_sha256=case_sha(parent),
                      institution=parent.get('provenance', {}).get('institution', 'not_applicable_general'), fictional=False)
        if parent['cohort'] == 'general':
            emit(parent['id'], parent['family_id'], split, 'original_general', deepcopy(parent['messages']),
                 parent['reference_answer'], parent['rubric'], parent.get('provenance', {}),
                 row_type='general', **shared)
            matrix[-1]['reference']['review_status'] = 'original_admitted_unchanged'
            continue
        question, packet = original_packet(parent)
        question = scoped_question(question, packet)
        target = json.loads(parent['reference_answer'])
        variants = [('clean', packet, target, {})]
        reordered, mapping = remap_packet(packet, [e['id'] for e in reversed(packet['evidence'])],
                                         document_order=[d['doc_id'] for d in reversed(packet['documents'])])
        variants.append(('reordered', reordered, remap_target(target, mapping), {'evidence_id_remap': mapping, 'permutation_is_identity': reordered == packet}))
        donor = next((r for r in donor_order if r['split'] == parent['split']
                      and r['family_id'] != parent['family_id']
                      and r['provenance']['doc_id'] != parent['provenance']['doc_id']
                      and document_scopes[r['id']] != document_scopes[parent['id']]), None)
        require(donor is not None, 'no independently admitted same-split distractor donor')
        mixed, donor_sources = add_distractor(packet, donor)
        variants.append(('distractor', mixed, target, {'donor_parent_id': donor['id'], 'donor_family_id': donor['family_id'], 'composite_sources': donor_sources}))
        empty = {'documents': [], 'evidence': [], 'scope': packet['scope'], 'warnings': list(packet['warnings'])}
        variants.append(('missing', empty, deepcopy(MISSING_TARGET), {}))
        if parent['id'] in facts:
            annotated = overlay_facts(packet, facts[parent['id']])
            variants.append(('structured', annotated, target, {'annotation_sha256': case_sha(facts[parent['id']])}))
            review_items.append({'id': parent['id'], 'kind': 'facts', 'parent_case_sha256': case_sha(parent),
                                 'question': question, 'source_data': packet, 'proposed_source_data': annotated,
                                 'annotation': facts[parent['id']], 'target': target, 'rubric': parent['rubric']})
        ids = {}
        for condition, source_data, answer, transformation in variants:
            ident = parent['id'] + '::' + condition
            rubric = parent['rubric'] if condition != 'missing' else ['근거가 전혀 없음을 설명하고 원래 규정·수치·기관 의무를 추측하지 않는다.', '출처를 만들지 않고 sources를 빈 목록으로 둔다.']
            ids[condition] = emit(ident, parent['family_id'], split, condition,
                                  build_evidence_messages(question, source_data), deepcopy(answer), rubric,
                                  parent['provenance'], transformation=transformation, **shared)
        group(parent['id'], 'invariant', [ids[c] for c in ('clean', 'reordered', 'distractor', 'structured') if c in ids])
        group(parent['id'], 'support_removed', [ids['clean'], ids['missing']])

    seen_families = {r['family_id'] for r in records}
    required = {'clean', 'changed_value_or_condition', 'distractor', 'missing', 'conflict', 'structured'}
    for scenario in annotations['synthetic']:
        family, split = scenario['family_id'], scenario['split']
        require(family not in seen_families and split in ('train', 'development'), 'duplicate/cross-split synthetic family')
        seen_families.add(family)
        require(scenario.get('fictional') is True and scenario['review_status'] == 'needs_independent_review', 'fictional/review status missing')
        variants = scenario['variants']
        require({v['condition'] for v in variants} == required and len(variants) == len(required), 'synthetic condition coverage mismatch')
        ids = {}
        for variant in variants:
            packet = variant['source_data']
            validate_packet(packet)
            require(any('가상' in w or '허구' in w or 'fiction' in w.lower() for w in packet['warnings']), 'fictional status must be explicit in public warnings')
            for doc in packet['documents']:
                require('가상' in canonical(doc) or 'fiction' in canonical(doc).lower(), 'fictional document metadata must be explicit')
            condition = variant['condition']
            require(condition != 'missing' or not packet['evidence'], 'synthetic missing retains evidence')
            ids[condition] = emit(f'{family}::{condition}', family, split, condition,
                                  build_evidence_messages(variant['question'], packet), deepcopy(variant['target']),
                                  variant['rubric'], {'annotation_sha256': case_sha(scenario)},
                                  parent_id=family, parent_case_sha256=case_sha(scenario), fictional=True,
                                  institution=next(v for v in variants if v['condition'] == 'clean')['source_data']['documents'][0]['institution'])
        group(family, 'invariant', [ids[c] for c in ('clean', 'distractor', 'structured')])
        for condition, relation in [('missing', 'support_removed'), ('changed_value_or_condition', 'changed_value_or_condition'), ('conflict', 'conflict')]:
            group(family, relation, [ids['clean'], ids[condition]])
        review_items.append({'id': family, 'kind': 'synthetic', 'annotation': scenario})
    indexed(matrix, 'derived matrix')
    # Family assignment is invariant, including all donor source identities.
    partitions = {}
    for item in matrix:
        ref = item['reference']
        family = ref['family_id']
        require(family not in partitions or partitions[family] == ref['split'], 'derived family crosses split')
        partitions[family] = ref['split']
    return matrix, groups, review_items


def training_view(matrix):
    """One deterministic condition per real parent, all authored synthetic variants.

    Fact overlays are always represented. General originals remain byte-equivalent
    message/answer projections and explicit replay keeps >=20% general exposures.
    """
    selected, by_parent, general = [], {}, []
    for item in matrix:
        ref = item['reference']
        if ref['split'] != 'train':
            continue
        if ref['row_type'] == 'general':
            general.append(item)
        elif ref['fictional']:
            selected.append(item)
        else:
            by_parent.setdefault(ref['parent_id'], []).append(item)
    for parent, choices in sorted(by_parent.items()):
        structured = [c for c in choices if c['reference']['condition'] == 'structured']
        selected.append(structured[0] if structured else choices[int(hashlib.sha256(parent.encode()).hexdigest(), 16) % len(choices)])
    selected.extend(general)
    require(general, 'general preservation rows absent')
    replay = max(0, (len(selected) - len(general) + 3) // 4 - len(general))
    for index in range(replay):
        item = deepcopy(general[index % len(general)])
        item['id'] = item['id'] + f'::replay-{index + 1}'
        item['reference']['id'] = item['id']
        item['reference']['replay_of'] = general[index % len(general)]['id']
        selected.append(item)
    selected.sort(key=lambda item: case_sha({'id': item['id'], 'profile': PROFILE}))
    result = []
    for item in selected:
        ref = item['reference']
        answer = ref['target'] if ref['row_type'] == 'general' else canonical(ref['target'])
        result.append({'id': item['id'], 'task': 'general_preservation' if ref['row_type'] == 'general' else 'public_evidence_' + ref['condition'],
                       'split': 'train', 'messages': item['messages'], 'answer': answer})
    require(sum(i['reference']['row_type'] == 'general' for i in selected) * 5 >= len(selected), 'general exposure ratio below 20%')
    return result, [i['reference'] for i in selected]


def token_budget_proof(tokenizer, matrix, train, context, answer_budget, *, trainer=None):
    """Measure every intended row; optionally exercise actual trainer masking.

    Metadata preparation does not import torch. Explicit research admission
    supplies the actual trainer and checks IDs, EOS and answer-only labels.
    """
    measurements = []
    intended = [(m['id'], m['messages'], m['reference']['target'], m['reference']['split']) for m in matrix]
    known = {i[0] for i in intended}
    intended.extend((r['id'], r['messages'], r['answer'], 'train') for r in train if r['id'] not in known)
    for ident, messages, target, split in intended:
        answer = target if isinstance(target, str) else canonical(target)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        require(answer_ids and answer_ids[-1] == tokenizer.eos_token_id, f'{ident}: native EOS missing')
        measurement = {'id': ident, 'prompt_tokens': len(prompt_ids), 'answer_tokens_including_eos': len(answer_ids),
                       'total_tokens': len(prompt_ids) + len(answer_ids), 'split': split}
        measurement['within_budget'] = (len(answer_ids) <= answer_budget and measurement['total_tokens'] <= context
                                        and (split == 'train' or len(prompt_ids) + answer_budget <= context))
        if trainer is not None:
            require(measurement['within_budget'], f'{ident}: live token proof exceeds budget')
            row = {'id': ident, 'task': 'evidence_admission', 'adapter': 'grounded_messages',
                   'messages': messages, 'answer': answer, 'file': 'derived-matrix.private.json'}
            encoded = trainer.encode(tokenizer, row, 'qwen3', context)
            require(encoded is not None and encoded['ids'] == prompt_ids + answer_ids
                    and encoded['labels'] == [-100] * len(prompt_ids) + answer_ids,
                    f'{ident}: actual trainer answer-only masking differs')
        measurements.append(measurement)
    failures = [m['id'] for m in measurements if not m['within_budget']]
    return {'status': 'pass' if not failures else 'blocked', 'context_tokens': context,
            'max_answer_tokens': answer_budget, 'enable_thinking': False, 'truncation': False,
            'trainer_execution_verified': trainer is not None, 'measurements': measurements, 'failed_ids': failures}


def validate_recomputed_token_proof(stored, recomputed):
    require(recomputed.get('status') == 'pass' and recomputed.get('trainer_execution_verified') is True,
            'live tokenizer and trainer proof required')
    expected = {**recomputed, 'trainer_execution_verified': False}
    require(stored == expected, 'stored token proof differs from authenticated live recomputation')


def prepare(source_frozen, annotations_path, output, *, root=ROOT, tokenizer=None,
            tokenizer_pins=None, tokenizer_binding=None, context=4096, answer_budget=768):
    output = Path(output)
    require(not output.exists(), 'output already exists; old runs must remain immutable')
    require(context > 0 and 0 < answer_budget < context, 'invalid token budgets')
    require(tokenizer is None or isinstance(tokenizer_binding, dict),
            'actual tokenizer requires its directory/acquisition binding')
    records, source_pins, _ = load_sources(source_frozen, root)
    annotations = read_json(annotations_path)
    matrix, groups, review_items = build_candidate(records, annotations)
    train, metadata = training_view(matrix)
    development = [m for m in matrix if m['reference']['split'] == 'development']
    development_ids = {m['id'] for m in development}
    evaluation_groups = [g for g in groups if set(g['members']) <= development_ids]
    proof = (token_budget_proof(tokenizer, matrix, train, context, answer_budget) if tokenizer is not None
             else {'status': 'not_measured', 'truncation': False, 'trainer_execution_verified': False})
    output.mkdir(parents=True)
    write_jsonl(output / 'train.jsonl', train)
    write_json(output / 'evaluation-inputs.json', [{'id': m['id'], 'messages': m['messages']} for m in development])
    write_json(output / 'evaluation-references.private.json', [m['reference'] for m in development])
    write_json(output / 'training-metadata.json', metadata)
    write_json(output / 'source-records.private.json', records)
    write_json(output / 'derived-matrix.private.json', {'cases': matrix, 'groups': groups})
    write_json(output / 'token-budget.json', proof)
    review_packet = {'schema': 'trid-evidence-annotation-review-packet-v1', 'profile': PROFILE,
                     'annotations': pin(annotations_path, root), 'items': review_items,
                     'author_identity': annotations['author_identity'],
                     'review_status': 'needs_independent_review',
                     'instructions': ['Review each new fact against exact supporting quotations, including actor, action, conditions, exceptions and provenance.',
                                      'Review every fictional variant target/rubric and its declared relation. Source links alone do not establish entailment.',
                                      'Old source receipts do not independently review these annotations.']}
    write_json(output / 'annotation-review-packet.private.json', review_packet)
    counts = {'original_cases': len(records), 'original_train': sum(r['split'] == 'train' for r in records),
              'original_development': sum(r['split'] == 'evaluation' for r in records),
              'unique_real_evidence_parents': sum(r['cohort'] != 'general' for r in records),
              'unique_real_source_families': len({r['family_id'] for r in records if r['cohort'] != 'general'}),
              'derived_matrix_cases': len(matrix), 'training_exposures': len(train),
              'general_training_exposures': sum(m['row_type'] == 'general' for m in metadata),
              'evaluation_cases': len(development), 'evaluation_groups': len(evaluation_groups),
              'synthetic_families': len(annotations['synthetic']), 'fact_overlays': len(annotations['facts']),
              'training_conditions': dict(Counter(m['condition'] for m in metadata))}
    common = {'profile': PROFILE, 'state': 'CANDIDATE_NEEDS_INDEPENDENT_REVIEW', 'training_ready': False,
              'semantic_evaluated': False, 'new_annotations_independently_reviewed': False}
    training_manifest = {'schema': 'trid-evidence-training-manifest-v1', **common,
                         'train': [{**local_pin(output / 'train.jsonl'), 'adapter': 'grounded_messages', 'epochs': 1}],
                         'valid': [], 'rows': len(train), 'max_tokens': context, 'max_answer_tokens': answer_budget,
                         'metadata': local_pin(output / 'training-metadata.json'), 'token_budget': local_pin(output / 'token-budget.json')}
    evaluation_manifest = {'schema': 'trid-evidence-evaluation-manifest-v1', **common,
                           'inputs': local_pin(output / 'evaluation-inputs.json'),
                           'references': local_pin(output / 'evaluation-references.private.json'),
                           'groups': evaluation_groups, 'counts': counts, 'context_tokens': context,
                           'max_new_tokens': answer_budget, 'enable_thinking': False}
    write_json(output / 'training-manifest.json', training_manifest)
    write_json(output / 'evaluation-manifest.json', evaluation_manifest)
    artifacts = {p.name: local_pin(p) for p in output.iterdir() if p.is_file()}
    manifest = {'schema': 'trid-evidence-preparation-manifest-v1', **common,
                'source_pins': source_pins, 'annotations': pin(annotations_path, root),
                'builder': pin(Path(__file__), ROOT), 'tokenizer_pins': tokenizer_pins or [],
                'tokenizer_binding': tokenizer_binding,
                'artifacts': artifacts, 'counts': counts, 'token_budget_status': proof['status'],
                'admission_limitations': ['No original source partition or historical exposure claim changes.',
                    'All descendants remain in the parent family split; distractors are same-split admitted sources, not new independent documents.',
                    'Real official quotation text is unchanged; only explicitly fictional families contain counterfactuals.',
                    'New prompts, missing-support targets, ontology edges and synthetic targets require independent review.',
                    'Old semantic receipts bind original cases only. No structural validator proves semantic correctness.',
                    'Trainer integration must be separately exercised with explicit manifest token budgets.',
                    'This evaluation protocol is separate from historical native/cloud sealed-run dispatchers.']}
    write_json(output / 'manifest.json', manifest)
    return manifest


def validate_annotation_reviews(packet, packet_sha256, reports, *, author_identity=None):
    """Require two complete independent reports; one failure blocks the corpus."""
    require(len(reports) == 2, 'exactly two independent annotation reports required')
    bound_author = packet.get('author_identity')
    require(isinstance(bound_author, str) and bound_author.strip(), 'review packet author identity missing')
    require(author_identity is None or author_identity == bound_author,
            'CLI author identity differs from bound annotation author')
    expected = indexed(packet['items'], 'review item')
    reviewers = set()
    for report in reports:
        require(report.get('schema') == 'trid-evidence-annotation-review-v1'
                and report.get('packet_sha256') == packet_sha256, 'annotation report packet binding mismatch')
        reviewer = report.get('reviewer')
        require(isinstance(reviewer, str) and reviewer.strip() and reviewer not in reviewers,
                'distinct independent reviewer identities required')
        require(reviewer != bound_author, 'annotation author cannot independently review own work')
        reviewers.add(reviewer)
        judgments = indexed(report.get('judgments', []), 'annotation judgment')
        require(set(judgments) == set(expected), 'annotation judgment coverage incomplete/unknown')
        for ident, item in expected.items():
            judgment = judgments[ident]
            require(judgment.get('pass') is True and isinstance(judgment.get('reason'), str)
                    and judgment['reason'].strip(), f'annotation review failed/disputed: {ident}')
            if item['kind'] == 'synthetic':
                members = judgment.get('members')
                conditions = {v['condition'] for v in item['annotation']['variants']}
                require(isinstance(members, list) and len(members) == len(conditions)
                        and set(members) == conditions, f'synthetic member review incomplete: {ident}')
    return sorted(reviewers)


def admit(prepared_path, report_paths, output, *, root=ROOT, author_identity=None):
    """Publish a NEW research-admitted copy, retaining all source/review boundaries."""
    import shutil

    root = Path(root).resolve()
    prepared_path = Path(prepared_path).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'output already exists; admission never mutates candidate runs')
    prepared = read_json(prepared_path)
    require(prepared.get('schema') == 'trid-evidence-preparation-manifest-v1'
            and prepared.get('state') == 'CANDIDATE_NEEDS_INDEPENDENT_REVIEW'
            and prepared.get('profile') == PROFILE, 'not an evidence candidate manifest')
    artifacts = prepared['artifacts']
    required_artifacts = {'train.jsonl', 'training-metadata.json', 'evaluation-inputs.json',
                          'evaluation-references.private.json', 'source-records.private.json',
                          'derived-matrix.private.json', 'annotation-review-packet.private.json',
                          'token-budget.json', 'training-manifest.json', 'evaluation-manifest.json'}
    require(required_artifacts <= set(artifacts), 'candidate artifact coverage incomplete')
    for name, artifact in artifacts.items():
        require(name == artifact['path'] and Path(name).name == name, 'candidate artifact must be local basename')
        verified(artifact, prepared_path.parent)
    verified(prepared['builder'], root)
    annotation_path = verified(prepared['annotations'], root)
    annotations = read_json(annotation_path)
    original_path = verified(prepared['source_pins']['source_frozen'], root)
    originals, source_pins, design = load_sources(original_path, root)
    require(source_pins == prepared['source_pins'], 'current source provenance differs from candidate')
    require(read_json(prepared_path.parent / 'source-records.private.json') == originals, 'source history changed')
    matrix, groups, review_items = build_candidate(originals, annotations)
    train, metadata = training_view(matrix)
    require(read_json(prepared_path.parent / 'derived-matrix.private.json') == {'cases': matrix, 'groups': groups}
            and rows(prepared_path.parent / 'train.jsonl') == train
            and read_json(prepared_path.parent / 'training-metadata.json') == metadata,
            'deterministic source-bound derivation contract differs')
    development = [m for m in matrix if m['reference']['split'] == 'development']
    development_ids = {m['id'] for m in development}
    expected_groups = [g for g in groups if set(g['members']) <= development_ids]
    require(read_json(prepared_path.parent / 'evaluation-inputs.json') ==
            [{'id': m['id'], 'messages': m['messages']} for m in development]
            and read_json(prepared_path.parent / 'evaluation-references.private.json') ==
            [m['reference'] for m in development], 'public/reference derivation differs')
    evaluation = read_json(prepared_path.parent / 'evaluation-manifest.json')
    training = read_json(prepared_path.parent / 'training-manifest.json')
    require(evaluation['groups'] == expected_groups
            and evaluation['inputs'] == artifacts['evaluation-inputs.json']
            and evaluation['references'] == artifacts['evaluation-references.private.json'],
            'evaluation manifest binding differs')
    require(training['train'] == [{**artifacts['train.jsonl'], 'adapter': 'grounded_messages', 'epochs': 1}]
            and training['rows'] == len(train), 'training manifest binding differs')
    packet_path = prepared_path.parent / 'annotation-review-packet.private.json'
    packet = read_json(packet_path)
    require(packet.get('schema') == 'trid-evidence-annotation-review-packet-v1'
            and packet['items'] == review_items and packet['annotations'] == prepared['annotations']
            and packet.get('author_identity') == annotations['author_identity'],
            'annotation review context differs')
    reports = [read_json(path) for path in report_paths]
    reviewers = validate_annotation_reviews(packet, file_sha(packet_path), reports, author_identity=author_identity)
    proof = read_json(prepared_path.parent / 'token-budget.json')
    require(prepared.get('token_budget_status') == 'pass' and proof.get('status') == 'pass'
            and proof.get('truncation') is False and proof.get('enable_thinking') is False
            and proof.get('failed_ids') == [] and prepared['tokenizer_pins'], 'complete native token proof required')
    for tokenizer_pin in prepared['tokenizer_pins']:
        verified(tokenizer_pin, root)
    expected_ids = {m['id'] for m in matrix} | {r['id'] for r in train}
    measurements = indexed(proof['measurements'], 'token measurement')
    require(set(measurements) == expected_ids, 'token proof does not cover every intended row')
    require(proof['context_tokens'] == training['max_tokens'] == evaluation['context_tokens']
            and proof['max_answer_tokens'] == training['max_answer_tokens'] == evaluation['max_new_tokens'],
            'token proof and manifest budgets disagree')
    for measurement in measurements.values():
        prompt, answer = measurement['prompt_tokens'], measurement['answer_tokens_including_eos']
        require(type(prompt) is int and type(answer) is int and prompt > 0 and answer > 0
                and answer <= proof['max_answer_tokens']
                and measurement['total_tokens'] == prompt + answer <= proof['context_tokens']
                and measurement.get('within_budget') is True
                and (measurement['split'] == 'train' or prompt + proof['max_answer_tokens'] <= proof['context_tokens']),
                'invalid/overflowing token measurement')
    binding = prepared.get('tokenizer_binding')
    require(isinstance(binding, dict) and isinstance(binding.get('directory'), str),
            'authenticated native tokenizer directory/acquisition binding required')
    directory = root / binding['directory']
    require(directory.resolve().is_relative_to(root), 'tokenizer directory outside repository')
    acquisition = verified(binding['acquisition'], root)
    require(binding['acquisition']['sha256'] == TOKENIZER_ACQUISITION_SHA256,
            'original authoritative tokenizer acquisition required')
    tokenizer, live_pins = load_tokenizer(directory, acquisition, binding['acquisition']['sha256'], design)
    require(live_pins + [binding['acquisition']] == prepared['tokenizer_pins'], 'native tokenizer pins differ')
    # Explicit admission is allowed to load trainer dependencies; preparation
    # and metadata-only imports never require torch.
    from scripts import train_open_weight_lora as trainer
    loaded = trainer.load_rows(training['train'][0], prepared_path.parent)
    require([(r['id'], r['messages'], r['answer']) for r in loaded] ==
            [(r['id'], r['messages'], r['answer']) for r in train], 'actual trainer projection differs')
    recomputed = token_budget_proof(tokenizer, matrix, train, proof['context_tokens'],
                                   proof['max_answer_tokens'], trainer=trainer)
    validate_recomputed_token_proof(proof, recomputed)
    output.mkdir(parents=True)
    for name in artifacts:
        shutil.copyfile(prepared_path.parent / name, output / name)
    write_json(output / 'admission-token-budget.json', recomputed)
    report_pins = []
    for index, report_path in enumerate(report_paths):
        name = f'annotation-review-{index + 1}.private.json'
        shutil.copyfile(report_path, output / name)
        report_pins.append(local_pin(output / name))
    receipt = {'schema': 'trid-evidence-research-admission-v1', 'profile': PROFILE,
               'prepared_manifest': pin(prepared_path, root), 'annotation_packet': artifacts[packet_path.name],
               'reports': report_pins, 'reviewers': reviewers, 'author_identity': packet['author_identity'],
               'all_annotation_items_passed': True, 'deterministic_derivation_contract': 'pass',
               'token_budget': artifacts['token-budget.json'], 'expert_certified': False,
               'live_tokenizer_and_trainer_proof': local_pin(output / 'admission-token-budget.json'),
               'review_method': 'independent_llm_annotation_review',
               'review_scope': 'Two independent annotation reviews; source-bound transformations mechanically audited.',
               'limitations': ['Research readiness only, not model semantic performance or expert/legal certification.',
                              'LLM-only annotation review is not independent domain-expert certification.',
                              'Reviewer identity is asserted by hash-bound report provenance, not a credential authority.',
                              'Original history/currentness limits remain; no claim of historically unseen sources.',
                              'Tokenizer/target-mask integration was exercised; no model training, GPU or cloud execution is authorized.']}
    write_json(output / 'research-admission.json', receipt)
    readiness = {'state': 'ADMITTED_FOR_RESEARCH', 'training_ready': True,
                 'new_annotations_independently_reviewed': True, 'semantic_evaluated': False,
                 'research_admission': local_pin(output / 'research-admission.json'), 'expert_certified': False}
    # Private row statuses remain their historical preparation-time statuses;
    # this receipt separately records exactly which new annotations were reviewed.
    for name, value in [('training-manifest.json', training), ('evaluation-manifest.json', evaluation)]:
        value.update(readiness)
        write_json(output / name, value)
    result = {**prepared, **readiness, 'schema': 'trid-evidence-admitted-manifest-v1',
              'candidate_manifest': pin(prepared_path, root),
              'admission_limitations': receipt['limitations'],
              'artifacts': {p.name: local_pin(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'manifest.json', result)
    return result


def admit_main(argv):
    parser = argparse.ArgumentParser(description='Admit a complete independently reviewed research candidate into a new directory.')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--review', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--author-identity')
    args = parser.parse_args(argv)
    result = admit(args.prepared, args.review, args.output, author_identity=args.author_identity)
    print(canonical({'state': result['state'], 'output': str(args.output), 'training_ready': result['training_ready']}))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'admit':
        return admit_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-frozen', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path)
    parser.add_argument('--tokenizer-acquisition', type=Path)
    parser.add_argument('--tokenizer-acquisition-sha256', default=TOKENIZER_ACQUISITION_SHA256)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--max-answer-tokens', type=int, default=768)
    args = parser.parse_args(argv)
    tokenizer, tokenizer_pins, tokenizer_binding = None, [], None
    require(bool(args.tokenizer_dir) == bool(args.tokenizer_acquisition), 'tokenizer directory/acquisition must be supplied together')
    if args.tokenizer_dir:
        frozen = read_json(args.source_frozen)
        design = read_json(verified(frozen['pins']['design']))
        tokenizer, tokenizer_pins = load_tokenizer(args.tokenizer_dir, args.tokenizer_acquisition,
                                                  args.tokenizer_acquisition_sha256, design)
        tokenizer_pins.append(pin(args.tokenizer_acquisition))
        tokenizer_binding = {'directory': str(args.tokenizer_dir.resolve().relative_to(ROOT.resolve())),
                             'acquisition': pin(args.tokenizer_acquisition)}
    manifest = prepare(args.source_frozen, args.annotations, args.output, tokenizer=tokenizer,
                       tokenizer_pins=tokenizer_pins, tokenizer_binding=tokenizer_binding,
                       context=args.max_tokens, answer_budget=args.max_answer_tokens)
    print(canonical({'state': manifest['state'], 'counts': manifest['counts'],
                     'token_budget_status': manifest['token_budget_status'], 'output': str(args.output)}))
    return 1 if manifest['token_budget_status'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
