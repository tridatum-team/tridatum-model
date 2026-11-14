"""Audit/freeze the registered expanded experiment, without inference or training.

All inputs are local. A failed gate produces a BLOCKED audit, never a runnable
manifest. Candidate source statuses are not clearance: an explicit, hash-bound
source admission receipt and independent semantic case reviews are mandatory.
See --help for the audit/freeze interface. No historical evaluation text is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ACQUISITION_SHA256 = 'e0dea2b77e7d9108d58786af3d9210c6ba0600ee29733c144a8e7efd1ff55300'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.aws_eval.eight_hf_assessment import _contained_path, pin as assessment_pin
MOIS_BEHAVIORS = ('direct_lookup', 'multi_item', 'conditions', 'exceptions',
                  'normative_force', 'information_boundary', 'multi_clause_synthesis', 'comparison')
SECONDARY = {'multi_item', 'multi_clause_synthesis'}
PIN_FIELDS = ('article_id', 'raw_body_node_ordinal', 'char_start', 'char_end', 'quote_sha256')
SOURCE_GATES = ('rights', 'evidence', 'bounded_copy_review', 'question_capacity',
                'protected_exclusions', 'signature_review', 'history_review')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def case_sha(record):
    return hashlib.sha256(canonical(record).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def path_in_repo(path, root=ROOT):
    root, path = Path(root), Path(path)
    candidate = (root / path).resolve() if not path.is_absolute() else path.resolve()
    missing = []
    ancestor = candidate
    while not ancestor.exists():
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    contained, _ = _contained_path(ancestor, root)
    return contained.joinpath(*reversed(missing))


def pin(path, root=ROOT):
    return assessment_pin(path_in_repo(path, root), root)


def verified(value, root=ROOT):
    if not isinstance(value, dict) or not {'path', 'bytes', 'sha256'} <= value.keys():
        raise ValueError('malformed file pin')
    path = path_in_repo(value['path'], root)
    if type(value['bytes']) is not int or path.stat().st_size != value['bytes'] or file_sha(path) != value['sha256']:
        raise ValueError(f'file pin mismatch: {path}')
    return path


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_record(record, contract):
    if not isinstance(record, dict) or not set(contract['record_required']) <= record.keys():
        raise ValueError('missing authored record fields')
    for key in ('id', 'family_id', 'behavior', 'reference_answer'):
        if not nonempty(record[key]):
            raise ValueError(f'{key} must be a nonempty string')
    if record['split'] not in ('train', 'evaluation') or record['cohort'] not in ('general', 'mois'):
        raise ValueError('invalid split/cohort')
    allowed = contract['general_behaviors'] if record['cohort'] == 'general' else MOIS_BEHAVIORS
    if record['behavior'] not in allowed:
        raise ValueError('unregistered behavior')
    messages = record['messages']
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    offset = int(isinstance(messages[0], dict) and messages[0].get('role') == 'system')
    for index, message in enumerate(messages):
        role = 'system' if offset and index == 0 else ('user' if (index - offset) % 2 == 0 else 'assistant')
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message.get('role') != role or not nonempty(message.get('content'))):
            raise ValueError('invalid native message role/content or alternation')
        if any(marker in message['content'] for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
            raise ValueError('native control token inside authored content')
    if messages[-1]['role'] != 'user':
        raise ValueError('current target must not be included in input messages')
    if any(marker in record['reference_answer'] for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
        raise ValueError('native control token inside target')
    if record['cohort'] == 'mois' and [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('MOIS grounded_messages requires exact system/user pair')
    rubric = record['rubric']
    if not isinstance(rubric, dict):
        raise ValueError('rubric must be an object')
    for key in ('required', 'forbidden', 'critical_errors', 'accepted_variants'):
        if not isinstance(rubric.get(key), list) or not rubric[key] or not all(nonempty(x) for x in rubric[key]):
            raise ValueError(f'rubric {key} must contain substantive strings')
    provenance = record['provenance']
    if not isinstance(provenance, dict) or provenance.get('model_outputs_seen_during_authoring') is not False:
        raise ValueError('author must attest no model outputs seen')
    if record['cohort'] == 'general':
        if provenance.get('kind') != 'synthetic':
            raise ValueError('general provenance must be synthetic')
        if provenance.get('surface_domain') not in contract['general_surface_partition'][record['split']]:
            raise ValueError('general surface partition violation')
    # Semantic copying/entailment is independently reviewed, not inferred from these checks.
    case_sha(record)


def validate_partitions(records, design, contract, allocation):
    ids, inputs, families = set(), {}, {}
    counts = Counter()
    behavior_counts = Counter()
    per_family = defaultdict(Counter)
    allocated = {f['family_id']: f for f in allocation['families']}
    if len(allocated) != len(allocation['families']):
        raise ValueError('duplicate allocated family')
    source_identities = {}
    for family in allocated.values():
        for field in ('group_key', 'doc_id', 'source_id', 'body_sha256'):
            value = family.get(field)
            if value is None:
                continue
            key = (field, value)
            if key in source_identities and source_identities[key] != family['split']:
                raise ValueError('source identity/body crosses family partitions')
            source_identities[key] = family['split']
    for row in records:
        validate_record(row, contract)
        if row['id'] in ids:
            raise ValueError('duplicate case ID')
        ids.add(row['id'])
        public_hash = case_sha(row['messages'])
        if public_hash in inputs:
            raise ValueError(f'identical public inputs under different IDs: {inputs[public_hash]}, {row["id"]}')
        inputs[public_hash] = row['id']
        family = row['family_id']
        if family in families and families[family] != row['split']:
            raise ValueError('whole-family train/evaluation partition overlap')
        families[family] = row['split']
        counts[row['split'], row['cohort']] += 1
        behavior_counts[row['split'], row['cohort'], row['behavior']] += 1
        if row['cohort'] == 'mois':
            source = allocated.get(family)
            expected = 'TRAIN_CANDIDATE' if row['split'] == 'train' else 'EVAL_CANDIDATE'
            if source is None or source['split'] != expected:
                raise ValueError('MOIS family absent or allocated to other partition')
            per_family[family][row['behavior']] += 1
    expected_counts = {('train', 'mois'): design['training_preparation']['grounded_rows'],
                       ('train', 'general'): design['training_preparation']['general_rows'],
                       ('evaluation', 'mois'): design['primary']['mois_questions'],
                       ('evaluation', 'general'): design['primary']['general_questions']}
    if counts != Counter(expected_counts):
        raise ValueError(f'exact registered counts required: observed {dict(counts)} expected {expected_counts}')
    if sum(counts[k] for k in counts if k[0] == 'train') != design['training_preparation']['rows']:
        raise ValueError('registered total train count mismatch')
    if sum(counts[k] for k in counts if k[0] == 'evaluation') != design['primary']['evaluation_questions']:
        raise ValueError('registered total evaluation count mismatch')
    for split, family_count, repetitions in (
            ('train', design['training_preparation']['grounded_families'], 2),
            ('evaluation', design['primary']['mois_families'], 1)):
        selected = [f for f in per_family if families[f] == split]
        if len(selected) != family_count:
            raise ValueError('registered MOIS family count mismatch')
        for family in selected:
            if per_family[family] != Counter({b: repetitions for b in MOIS_BEHAVIORS}):
                raise ValueError('every MOIS family requires exact registered behavior coverage')
            if allocated[family].get('planned_rows') != repetitions * len(MOIS_BEHAVIORS):
                raise ValueError('allocated planned_rows disagrees with registered behavior coverage')
        total = expected_counts[split, 'general']
        if total % len(contract['general_behaviors']):
            raise ValueError('general behavior allocation is not integral')
        if any(behavior_counts[split, 'general', b] != total // len(contract['general_behaviors'])
               for b in contract['general_behaviors']):
            raise ValueError('general behavior count mismatch')
    if set(per_family) != set(allocated):
        raise ValueError('unused or missing allocated source family')
    secondary = [r['id'] for r in records if r['split'] == 'evaluation' and r['cohort'] == 'mois' and r['behavior'] in SECONDARY]
    if len(secondary) != design['format_secondary']['questions']:
        raise ValueError('secondary fixed behavior count mismatch')
    return secondary


def validate_reviews(records, packets, reports, diagnostics=None):
    bound = {}
    current = {(record['id'], case_sha(record)) for record in records}
    current_ids = {identity for identity, _ in current}
    ignored_stale = []
    for digest, packet in packets.items():
        if packet.get('schema') != 'trid-expanded-data-review-packet-v1' or not isinstance(packet.get('source_file'), dict):
            raise ValueError('invalid data review packet')
        cases = {}
        for case in packet['cases']:
            if case['id'] in cases or case['id'] != case['record']['id'] or case_sha(case['record']) != case['case_sha256']:
                raise ValueError('review packet case hash/identity mismatch')
            cases[case['id']] = case
        bound[digest] = cases
    judgments = defaultdict(dict)
    for report in reports:
        if report.get('schema') != 'trid-expanded-data-review-v1' or not nonempty(report.get('reviewer_id')):
            raise ValueError('invalid semantic review report')
        reviewer = report['reviewer_id']
        cases = bound.get(report.get('packet_sha256'))
        if cases is None:
            raise ValueError('review report is not bound to an exact packet')
        seen = set()
        for judgment in report['judgments']:
            identity = judgment['id']
            if identity in seen or identity not in cases or judgment['case_sha256'] != cases[identity]['case_sha256']:
                raise ValueError('review judgment hash/identity mismatch')
            seen.add(identity)
            if (judgment.get('verdict') not in ('pass', 'fix', 'hold') or not nonempty(judgment.get('reason'))
                    or not isinstance(judgment.get('issues'), list) or 'suggested_fix' not in judgment):
                raise ValueError('semantic judgment lacks a verdict or substantive reason')
            key = (identity, judgment['case_sha256'])
            if key not in current:
                ignored_stale.append({'id': identity, 'case_sha256': judgment['case_sha256'],
                                      'reviewer_id': reviewer, 'packet_sha256': report['packet_sha256'],
                                      'reason': 'superseded_case_hash' if identity in current_ids else 'case_not_in_current_dataset'})
                continue  # Validly bound superseded judgments cannot veto a revised case.
            if reviewer in judgments[key] and judgments[key][reviewer] != judgment:
                raise ValueError('conflicting duplicate reviewer judgment')
            judgments[key][reviewer] = judgment
    ignored_stale.sort(key=lambda item: (item['id'], item['case_sha256'], item['reviewer_id'], item['packet_sha256']))
    if diagnostics is not None:
        diagnostics.update(current_case_count=len(current),
                           current_case_judgment_count=sum(len(reviews) for reviews in judgments.values()),
                           ignored_stale_count=len(ignored_stale), ignored_stale_judgments=ignored_stale)
    resolution = []
    for record in records:
        digest = case_sha(record)
        reviews = judgments.get((record['id'], digest), {})
        authors = set(record['provenance'].get('author_ids', []))
        if record['provenance'].get('author_id'):
            authors.add(record['provenance']['author_id'])
        if authors.intersection(reviews):
            raise ValueError('author cannot independently review their own case')
        passes = [name for name, j in reviews.items() if j['verdict'] == 'pass']
        nonpasses = [name for name, j in reviews.items() if j['verdict'] != 'pass']
        if len(passes) < 2:
            raise ValueError(f'{record["id"]}: two independent passes for current case hash required')
        if nonpasses:
            resolved = any(set(j.get('adjudicates', [])) >= set(reviews) - {name}
                           for name, j in reviews.items() if j['verdict'] == 'pass')
            if len(reviews) < 3 or not resolved:
                raise ValueError(f'{record["id"]}: disagreement requires a third independent adjudicator')
        resolution.append({'id': record['id'], 'case_sha256': digest,
                           'reviewer_ids': sorted(reviews), 'verdict': 'pass'})
    return {'schema': 'trid-expanded-semantic-review-resolution-v1', 'cases': resolution,
            'ignored_stale_count': len(ignored_stale), 'ignored_stale_judgments': ignored_stale}


def validate_source_binding(record, family, packet, body):
    provenance = record['provenance']
    for key in ('family_id', 'doc_id', 'version', 'body_sha256'):
        if packet.get(key) != family.get(key) or provenance.get(key) != family.get(key):
            raise ValueError(f'source binding mismatch: {key}')
    if hashlib.sha256(body.encode()).hexdigest() != family['body_sha256']:
        raise ValueError('canonical source body hash mismatch')
    supplied = provenance.get('passage_pins')
    if not isinstance(supplied, list) or not supplied:
        raise ValueError('exact source passage pins required')
    texts = []
    seen = set()
    for passage in supplied:
        if not isinstance(passage, dict) or not set(PIN_FIELDS) <= passage.keys():
            raise ValueError('malformed passage pin')
        if (not nonempty(passage['article_id']) or not nonempty(passage['quote_sha256'])
                or type(passage['raw_body_node_ordinal']) is not int or passage['raw_body_node_ordinal'] < 1
                or type(passage['char_start']) is not int or type(passage['char_end']) is not int):
            raise ValueError('malformed passage pin types')
        signature = tuple(passage[key] for key in PIN_FIELDS)
        if signature in seen:
            raise ValueError('duplicate passage binding')
        seen.add(signature)
        if not any(all(p.get(k) == passage[k] for k in PIN_FIELDS) for p in family['passage_pins']):
            raise ValueError('passage absent from selected allocation')
        matches = [p for p in packet['articles'] if all(p.get(k) == passage[k] for k in PIN_FIELDS)]
        if len(matches) != 1:
            raise ValueError('passage absent/ambiguous in source packet')
        start, end = passage['char_start'], passage['char_end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(body):
            raise ValueError('invalid source character span')
        text = body[start:end]
        if text != matches[0]['text'] or hashlib.sha256(text.encode()).hexdigest() != passage['quote_sha256']:
            raise ValueError('source excerpt bytes/hash mismatch')
        texts.append(text)
    evidence = provenance.get('evidence_reply')
    if not isinstance(evidence, dict):
        raise ValueError('production evidence_reply required')
    graph = evidence.get('graph', {})
    # Expanded authors use raw quotations, not unchecked synthetic graph facts.
    if graph.get('paths'):
        raise ValueError('unbound graph facts are not admitted; use pinned quotations')
    sources = graph.get('sources', [])
    if len(sources) != 1 or sources[0].get('doc_id') != family['doc_id']:
        raise ValueError('evidence source document mismatch')
    for key in ('title', 'source_url', 'issuer'):
        if key in sources[0] and sources[0][key] != family.get(key):
            raise ValueError(f'evidence source metadata mismatch: {key}')
    if (sources[0].get('institution') != family.get('issuer')
            or provenance.get('institution') != family.get('issuer')
            or sources[0].get('currency') != 'OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED'):
        raise ValueError('evidence institution/currentness binding mismatch')
    date = family.get('effective_date')
    expected_date = f'{date[:4]}-{date[4:6]}-{date[6:]}' if isinstance(date, str) and len(date) == 8 else date
    if 'effective_date' in sources[0] and sources[0]['effective_date'] != expected_date:
        raise ValueError('evidence effective date mismatch')
    quotes = []
    for candidate in evidence.get('candidates', []):
        prediction = candidate.get('prediction', {})
        quote = prediction.get('answer')
        text = candidate.get('text')
        span = prediction.get('source_character_span')
        if (candidate.get('doc_id') != family['doc_id'] or text not in texts or quote != text
                or span != [0, len(text)] or prediction.get('abstains')):
            raise ValueError('evidence quotation is not an exact admitted source excerpt')
        quotes.append(quote)
    if Counter(quotes) != Counter(texts):
        raise ValueError('evidence/passage binding coverage mismatch')
    return texts


def trainer_row(record):
    task = 'mois_grounded_explanation' if record['cohort'] == 'mois' else 'general_' + record['behavior']
    return {'id': record['id'], 'task': task, 'split': record['split'],
            'messages': record['messages'], 'answer': record['reference_answer']}


class TokenAdmissionError(ValueError):
    def __init__(self, reason, measurement):
        super().__init__(reason)
        self.details = measurement


def audit_cases(records, gate, check, blockers):
    """Retain every case and failure; successful neighbors never waive a gate."""
    results = []
    for index, record in enumerate(records):
        identity = record.get('id') if isinstance(record, dict) else None
        result = {'id': identity, 'record_index': index}
        try:
            result.update(status='pass', result=check(record))
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError, ImportError, ET.ParseError) as exc:
            result.update(status='blocked', reason=str(exc))
            if hasattr(exc, 'details'):
                result['details'] = exc.details
            blockers.append({'gate': gate, **result})
        results.append(result)
    return results


def audit_evaluation_reserve(measurement, context, max_new_tokens):
    if measurement['prompt_tokens'] + max_new_tokens > context:
        raise TokenAdmissionError('evaluation prompt leaves insufficient full generation budget', measurement)


def audit_encoded(tok, row, context, max_answer_tokens, trainer):
    prompt = tok.apply_chat_template(trainer.messages_of(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(prompt, add_special_tokens=False)['input_ids']
    answer_ids = tok(row['answer'] + tok.eos_token, add_special_tokens=False)['input_ids']
    measurement = {'id': row['id'], 'prompt_tokens': len(prompt_ids),
                   'answer_tokens_including_eos': len(answer_ids), 'total_tokens': len(prompt_ids) + len(answer_ids)}
    if not answer_ids or answer_ids[-1] != tok.eos_token_id:
        raise TokenAdmissionError('answer must end in native EOS', measurement)
    if len(answer_ids) > max_answer_tokens:
        raise TokenAdmissionError(f'over_answer_budget ({len(answer_ids)} > {max_answer_tokens}, EOS included)', measurement)
    if len(prompt_ids) + len(answer_ids) > context:
        raise TokenAdmissionError(f'over_context_budget ({measurement["total_tokens"]} > {context}); truncation is forbidden', measurement)
    item = trainer.encode(tok, row, 'qwen3', context)
    if (item is None or item['ids'] != prompt_ids + answer_ids
            or item['labels'] != [-100] * len(prompt_ids) + answer_ids):
        raise TokenAdmissionError('current answer-only target masking invariant failed', measurement)
    return measurement


def verify_tokenizer_metadata(directory, acquisition_path, expected_sha, design):
    if file_sha(acquisition_path) != expected_sha:
        raise ValueError('tokenizer acquisition hash mismatch')
    acquisition = read_json(acquisition_path)
    if acquisition.get('schema') != 'trid-immutable-model-metadata-v1':
        raise ValueError('original immutable tokenizer acquisition metadata required')
    if (acquisition['repo_id'], acquisition['revision']) != (design['model']['repo_id'], design['model']['revision']):
        raise ValueError('tokenizer model/revision mismatch')
    records = {row['path']: row for row in acquisition['files']}
    required = {'LICENSE', 'config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json'}
    if len(records) != len(acquisition['files']) or not required <= records.keys():
        raise ValueError('immutable acquisition lacks required native tokenizer metadata')
    if {path.name for path in directory.iterdir()} != set(records):
        raise ValueError('native model directory contains unpinned or missing files')
    pinned = []
    for name, row in records.items():
        if name not in required and not row.get('body_downloaded'):
            continue  # Weight bodies are not opened by tokenizer admission.
        if row.get('body_downloaded') is not True:
            raise ValueError(f'original native metadata was not downloaded: {name}')
        path = directory / name
        data = path.read_bytes()
        if len(data) != row['bytes']:
            raise ValueError(f'official metadata size mismatch: {name}')
        if hashlib.sha256(data).hexdigest() != row.get('sha256'):
            raise ValueError(f'official metadata content mismatch: {name}')
        pinned.append(pin(path))
    return pinned


def load_tokenizer(directory, acquisition_path, expected_sha, design):
    pinned = verify_tokenizer_metadata(directory, acquisition_path, expected_sha, design)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    if not tok.chat_template or not tok.eos_token:
        raise ValueError('native tokenizer/template/EOS missing')
    return tok, pinned


def source_receipt(receipt, allocation_pin, records, source_inputs, overlap_sha):
    if not receipt or receipt.get('schema') != 'trid-expanded-source-admission-v1':
        raise ValueError('explicit source admission receipt missing; candidates remain NOT_ADMITTED')
    if receipt.get('source_allocation') != allocation_pin or receipt.get('source_inputs') != source_inputs:
        raise ValueError('source receipt input pins stale')
    if receipt.get('authored_overlap_sha256') != overlap_sha:
        raise ValueError('source receipt does not review current authored overlap screen')
    expected = {r['id']: case_sha(r) for r in records}
    if receipt.get('case_hashes') != expected:
        raise ValueError('source receipt case hashes stale/incomplete')
    protected = receipt.get('protected_case_review', {})
    if (protected.get('verdict') != 'pass' or not nonempty(protected.get('reviewer_id'))
            or not nonempty(protected.get('reason')) or not protected.get('evidence_pins')):
        raise ValueError('all-case protected behavioral/domain exclusion review required')
    for evidence in protected['evidence_pins']:
        verified(evidence)
    if receipt.get('historically_unseen') is not False or receipt.get('checkpoint_consumption') != 'UNKNOWN_NOT_CERTIFIED':
        raise ValueError('source receipt must retain unknown historical exposure')
    if receipt.get('currentness') != 'OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED':
        raise ValueError('source receipt must retain snapshot-currentness caveat')
    families = receipt.get('families', [])
    expected_families = {r['family_id'] for r in records if r['cohort'] == 'mois'}
    if len(families) != len(expected_families) or {f['family_id'] for f in families} != expected_families:
        raise ValueError('source admission family coverage mismatch')
    for family in families:
        for gate in SOURCE_GATES:
            decision = family.get(gate, {})
            if (decision.get('verdict') != 'pass' or not nonempty(decision.get('reviewer_id'))
                    or not nonempty(decision.get('reason')) or not decision.get('evidence_pins')):
                raise ValueError(f'{family["family_id"]}: pending source gate {gate}')
            for evidence in decision['evidence_pins']:
                verified(evidence)
    return receipt


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, values):
    path.write_text(''.join(canonical(v) + '\n' for v in values), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--design-sha256', required=True)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--allocation-sha256', required=True)
    parser.add_argument('--authors', type=Path, nargs='+', required=True)
    parser.add_argument('--review-packet', type=Path, action='append', default=[])
    parser.add_argument('--review', type=Path, action='append', default=[])
    parser.add_argument('--source-admission', type=Path)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--tokenizer-acquisition', type=Path, required=True)
    parser.add_argument('--tokenizer-acquisition-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrite a prior audit/freeze')
    parser.add_argument('--freeze', action='store_true', help='write executable manifests only if every gate passes')
    args = parser.parse_args(argv)
    output = path_in_repo(args.output)
    if output.exists():
        raise ValueError('output directory already exists; preserve previous audit/freeze')
    output.mkdir(parents=True)
    blockers = []
    report = {'schema': 'trid-expanded-admission-audit-v1', 'state': 'BLOCKED', 'blockers': blockers,
              'inference_performed': False, 'training_performed': False, 'structural_validity_is_semantics': False}

    def stage(name, fn):
        try:
            return fn()
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError, ImportError, ET.ParseError) as exc:
            blockers.append({'gate': name, 'reason': str(exc)})
            return None

    def inputs():
        if args.tokenizer_acquisition_sha256 != TOKENIZER_ACQUISITION_SHA256:
            raise ValueError('use the original registered immutable eleven-file acquisition, not a fresh API inventory')
        if file_sha(args.design) != args.design_sha256 or file_sha(args.allocation) != args.allocation_sha256:
            raise ValueError('registered design/source allocation SHA mismatch')
        design = read_json(args.design)
        if design.get('schema') != 'trid-expanded-local-development-plan-v1':
            raise ValueError('unrecognized registered design schema')
        contract_path = args.design.parent / 'authoring-contract.json'
        if file_sha(contract_path) != design['authoring_contract_sha256']:
            raise ValueError('registered authoring contract SHA mismatch')
        if (design['generation']['context_tokens'], design['generation']['max_new_tokens'],
                design['generation']['context_truncation'], design['metrics']['independent_reviews']) != (4096, 768, False, 2):
            raise ValueError('unsupported registered generation/review contract')
        if (design['generation']['enable_thinking'] is not False or design['generation']['do_sample'] is not False
                or design['generation']['response_repair'] is not False
                or design['primary']['decoder_profile'] != 'unconstrained'
                or design['format_secondary']['decoder_profile'] != 'json-schema'
                or design['primary']['arms'] != ['base', 'step24']
                or design['format_secondary']['arms'] != ['base', 'step24']):
            raise ValueError('registered native generation/paired comparison profile mismatch')
        if ((design['primary']['evaluation_questions'], design['primary']['mois_questions'],
             design['primary']['general_questions'], design['primary']['mois_families'],
             design['training_preparation']['rows'], design['training_preparation']['grounded_rows'],
             design['training_preparation']['general_rows'], design['training_preparation']['grounded_families'],
             design['format_secondary']['questions']) != (200, 160, 40, 20, 600, 480, 120, 30, 40)):
            raise ValueError('expanded CLI requires the full registered 200/600/40 design, never a tiny fixture')
        records, author_groups = [], []
        for path in args.authors:
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            author_groups.append((path, len(records), len(rows)))
            records.extend(rows)
        hash_results = audit_cases(records, 'case_hash', case_sha, blockers)
        report['case_checks'] = {'case_hash': hash_results}
        report['case_hashes'] = {r['id']: result['result'] for r, result in zip(records, hash_results)
                                 if isinstance(r, dict) and nonempty(r.get('id')) and result['status'] == 'pass'}
        for path, start, count in author_groups:
            group_hashes = hash_results[start:start + count]
            if any(result['status'] != 'pass' for result in group_hashes):
                continue  # An invalid source file must not yield a reduced review packet.
            packet = {'schema': 'trid-expanded-data-review-packet-v1', 'source_file': pin(path),
                      'cases': [{'id': result['id'], 'case_sha256': result['result'], 'record': record}
                                for record, result in zip(records[start:start + count], group_hashes)]}
            write_json(output / (path.stem + '.review-packet.json'), packet)
        domain_contract = args.design.parent / 'domain-authoring-contract.json'
        report['input_pins'] = {'design': pin(args.design), 'authoring_contract': pin(contract_path),
                                'source_allocation': pin(args.allocation), 'authors': [pin(p) for p in args.authors],
                                'review_packets': [pin(p) for p in args.review_packet], 'reviews': [pin(p) for p in args.review],
                                'source_admission': pin(args.source_admission) if args.source_admission else None,
                                'tokenizer_acquisition': pin(args.tokenizer_acquisition)}
        if domain_contract.exists():
            report['input_pins']['domain_authoring_contract'] = pin(domain_contract)
        return design, read_json(contract_path), read_json(args.allocation), records

    loaded = stage('registered_inputs', inputs)
    if loaded:
        design, contract, allocation, records = loaded
        report['total_records'] = len(records)
        counts = Counter((str(r.get('split')), str(r.get('cohort'))) if isinstance(r, dict)
                         else ('INVALID_RECORD', 'INVALID_RECORD') for r in records)
        report['observed_counts'] = [{'split': split, 'cohort': cohort, 'records': count}
                                     for (split, cohort), count in sorted(counts.items())]
        report['case_checks']['record_contract'] = audit_cases(
            records, 'record_contract', lambda row: validate_record(row, contract), blockers)
        secondary = stage('counts_partitions_records', lambda: validate_partitions(records, design, contract, allocation))
        report['source_caveats'] = design['source_caveats']

        def production_reference(row):
            if row['cohort'] != 'mois':
                return {'applicable': False}
            from scripts.prepare_grounded_experiment import validate_authored_target
            return validate_authored_target(row['provenance']['question'], row['provenance']['evidence_reply'],
                                            row['messages'], row['reference_answer'])

        report['case_checks']['production_reference'] = audit_cases(
            records, 'production_reference', production_reference, blockers)
        report['semantic_review_summary'] = {}
        resolution = stage('independent_semantic_reviews', lambda: validate_reviews(
            records, {file_sha(p): read_json(p) for p in args.review_packet}, [read_json(p) for p in args.review],
            diagnostics=report['semantic_review_summary']))

        def sources():
            from scripts.prepare_expanded_source_inventory import normalized_body, SIGNATURES, FIXED_EXCLUDED, HISTORY, PREP
            from scripts.expanded_source_overlap import scan_units
            listing_path = args.design.parent / 'source-file-list.json'
            listing = read_json(listing_path)
            all_pins = listing['metadata_inputs']
            signature_pins = {p['path']: p for p in listing['signature_inputs']}
            for signature_path in SIGNATURES:
                if file_sha(ROOT / signature_path) != signature_pins[signature_path]['sha256']:
                    raise ValueError('authoritative reserved signature hash mismatch')
            metadata_pins = {p['path']: p for p in all_pins}
            safe = contract['safe_exclusion_metadata'] + [HISTORY + '/source-family-allocation.json', PREP + '/source-family-allocation.json']
            source_inputs = []
            excluded = set(FIXED_EXCLUDED)
            for path in safe:
                checked = verified(metadata_pins[path])
                source_inputs.append(pin(checked))
                if path.endswith('source-family-allocation.json'):
                    excluded.update(f['family_id'] for f in read_json(checked)['families'])
            # Fingerprint helpers reuse check_reserved_overlap, never its eval readers.
            unit_results = audit_cases(records, 'overlap_input', lambda r: {
                'unit_id': r['id'], 'family_id': r['family_id'], 'doc_id': '', 'source_url': '',
                'text': '\n'.join(m['content'] for m in r['messages']) + '\n' + r['reference_answer']}, blockers)
            report['case_checks']['overlap_input'] = [
                {key: value for key, value in result.items() if key != 'result'} for result in unit_results]
            units = [result['result'] for result in unit_results if result['status'] == 'pass']
            overlap = scan_units(ROOT, units, SIGNATURES)
            overlap['coverage'] = {'total_cases': len(records), 'screened_cases': len(units),
                                   'blocked_record_indices': [r['record_index'] for r in unit_results if r['status'] != 'pass']}
            overlap_path = output / 'authored-overlap-screen.json'
            write_json(overlap_path, overlap)
            source_inputs += [pin(ROOT / p) for p in SIGNATURES]
            report['authored_overlap'] = pin(overlap_path)
            for identity, unit in overlap['units'].items():
                if unit['hits']:
                    blockers.append({'gate': 'reserved_signature', 'id': identity,
                                     'reason': 'authored content hits protected reserved signature thresholds'})
            families = {f['family_id']: f for f in allocation['families']}
            if excluded.intersection(families):
                raise ValueError('reserved or historical source family selected')
            components = {}
            for family in families.values():
                component = family.get('copy_component')
                if component is not None:
                    if component in components and components[component] != family['split']:
                        raise ValueError('source copy component crosses train/evaluation partitions')
                    components[component] = family['split']
            cache = {}
            def source_binding(row):
                if row['cohort'] != 'mois':
                    return {'applicable': False}
                family = families[row['family_id']]
                if family['family_id'] not in cache:
                    raw_path = verified(family['raw_source'])
                    tree = ET.fromstring(raw_path.read_bytes())
                    nodes = tree.findall('./조문내용') or tree.findall('./조문/조문내용')
                    body = '\n\n'.join(normalized_body(''.join(node.itertext())) for node in nodes)
                    cache[family['family_id']] = body
                packet_path = verified(row['provenance']['source_packet'])
                texts = validate_source_binding(row, family, read_json(packet_path), cache[family['family_id']])
                if not texts or not family.get('license') or family['rights']['declared_basis'] != family['license']:
                    raise ValueError('source rights basis missing/mismatched')
                return {'applicable': True, 'passage_count': len(texts)}

            report['case_checks']['source_binding'] = audit_cases(records, 'source_binding', source_binding, blockers)
            source_inputs = sorted(source_inputs, key=lambda p: p['path'])
            report['source_review_inputs'] = source_inputs
            receipt = read_json(args.source_admission) if args.source_admission else None
            return source_receipt(receipt, pin(args.allocation), records, source_inputs, file_sha(overlap_path))

        stage('source_and_protected_admission', sources)

        def tokens():
            from scripts import train_open_weight_lora as trainer
            tok, metadata = load_tokenizer(args.tokenizer_dir, args.tokenizer_acquisition, args.tokenizer_acquisition_sha256, design)
            report['tokenizer_metadata'] = metadata
            entries = defaultdict(list)
            with tempfile.TemporaryDirectory(prefix='token-audit-', dir=output) as temporary:
                temporary = Path(temporary)

                def token_case(record):
                    if record['split'] not in ('train', 'evaluation'):
                        raise ValueError('invalid token-audit split')
                    path = temporary / 'current-case.jsonl'
                    write_jsonl(path, [trainer_row(record)])
                    entry = {**pin(path), 'path': path.name, 'adapter': 'grounded_messages', 'epochs': 1}
                    loaded_rows = trainer.load_rows(entry, temporary)
                    if len(loaded_rows) != 1:
                        raise ValueError('trainer changed single-case row count')
                    row = loaded_rows[0]
                    if trainer.messages_of(row) != record['messages']:
                        raise ValueError('trainer adapter altered authored public input')
                    measurement = audit_encoded(tok, row, 3072 if record['split'] == 'train' else 4096, 768, trainer)
                    if record['split'] == 'evaluation':
                        audit_evaluation_reserve(measurement, 4096, 768)
                    return measurement

                results = audit_cases(records, 'native_tokenizer_and_target_masking', token_case, blockers)
            report['case_checks']['native_tokenizer_and_target_masking'] = results
            report['token_measurements'] = [r['result'] if r['status'] == 'pass' else r.get('details')
                                            for r in results]
            # Never materialize a reduced split after a failed case.
            if any(result['status'] != 'pass' for result in results):
                return entries
            for split in ('train', 'evaluation'):
                selected = [r for r in records if r['split'] == split]
                path = output / f'NOT_ADMITTED-{split}-all.jsonl'
                write_jsonl(path, [trainer_row(r) for r in selected])
                entry = {**pin(path), 'path': path.name, 'adapter': 'grounded_messages', 'epochs': 1}
                entries[split].append(entry)
                items, dropped = trainer.build_items(tok, [entry], output, 'qwen3', 3072 if split == 'train' else 4096, 768, False)
                if dropped or len(items) != len(selected):
                    raise ValueError('trainer dropped or changed row count')
            return entries

        entries = stage('native_tokenizer_and_target_masking', tokens)
        if not blockers:
            report['state'] = 'ADMITTED' if args.freeze else 'AUDITED_NOT_FROZEN'
            if args.freeze:
                public = [{'id': r['id'], 'messages': r['messages']} for r in records if r['split'] == 'evaluation']
                metadata = [{'id': r['id'], 'cohort': r['cohort'], 'behavior': r['behavior'], 'family_id': r['family_id'],
                             'case_sha256': case_sha(r), 'secondary_selected': r['id'] in secondary,
                             'source_caveats': design['source_caveats'] if r['cohort'] == 'mois' else []}
                            for r in records if r['split'] == 'evaluation']
                references = [{'id': r['id'], 'reference_answer': r['reference_answer'], 'rubric': r['rubric'],
                               'case_sha256': case_sha(r)} for r in records if r['split'] == 'evaluation']
                for name, value in (('evaluation-inputs.json', public), ('evaluation-metadata.json', metadata),
                                    ('evaluation-references.private.json', references), ('semantic-review-resolution.json', resolution)):
                    write_json(output / name, value)
                train_entries = []
                for entry in entries['train']:
                    old = output / entry['path']
                    destination = output / entry['path'].replace('NOT_ADMITTED-train-', 'train-')
                    old.rename(destination)
                    train_entries.append({**entry, 'path': destination.name})
                manifest = {'schema': 'trid-expanded-training-manifest-v1', 'state': 'ADMITTED',
                            'train': train_entries, 'valid': [], 'rows': 600, 'max_tokens': 3072, 'max_answer_tokens': 768,
                            'input_pins': report['input_pins'], 'source_caveats': design['source_caveats']}
                write_json(output / 'training-manifest.json', manifest)
                frozen = {'schema': 'trid-expanded-frozen-dataset-v1', 'pins': {
                    'design': pin(args.design), 'source_allocation': pin(args.allocation),
                    'evaluation_inputs': pin(output / 'evaluation-inputs.json'),
                    'evaluation_metadata': pin(output / 'evaluation-metadata.json'),
                    'evaluation_references': pin(output / 'evaluation-references.private.json'),
                    'training_manifest': pin(output / 'training-manifest.json'),
                    'semantic_review_resolution': pin(output / 'semantic-review-resolution.json')},
                    'secondary_case_ids': secondary, 'source_caveats': design['source_caveats'],
                    'input_pins': report['input_pins']}
                write_json(output / 'frozen-dataset.json', frozen)
    write_json(output / 'admission-audit.json', report)
    print(canonical({'state': report['state'], 'audit': str(output / 'admission-audit.json'), 'blockers': blockers}))
    return 1 if blockers else 0


if __name__ == '__main__':
    raise SystemExit(main())
