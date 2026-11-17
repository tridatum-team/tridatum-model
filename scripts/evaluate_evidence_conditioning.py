"""Offline public_evidence_v1 diagnostics, distinct from sealed native GPU runs.

The CLI only consumes local, pinned artifacts; it never loads/downloads a model.
The optional Python generation transport accepts an already-loaded HF model. Its
caller remains responsible for existing resource supervision and model admission.
Semantic judgments are independently authored, complete and packet-bound. Format,
citation and completion checks alone never establish evidence-use correctness.
"""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_local_eight import (decode_generation, digest, generation_target,
                                 object_sha, pin as local_pin, public_inputs, require, strict_json)
from fresh_eight_evaluation import (SAMPLING_PROFILE, SAMPLING_PARAMETERS,
                                   case_seed, generation_dispatch)
from trid.service.evidence_conditioning import (PROFILE, packet_from_messages,
                                                validate_evidence_answer)

PROTOCOL = 'trid-evidence-offline-paired-v1'
MANIFEST_SCHEMA = 'trid-evidence-evaluation-manifest-v1'
PACKET_SCHEMA = 'trid-evidence-blind-packet-v1'
MAPPING_SCHEMA = 'trid-evidence-blind-mapping-v1'
JUDGMENT_SCHEMA = 'trid-evidence-judgments-v1'


def load_json(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def artifact_pin(path):
    return local_pin(path)


def pinned_path(root, pin):
    require(isinstance(pin, dict) and isinstance(pin.get('path'), str)
            and isinstance(pin.get('sha256'), str), 'Invalid artifact pin')
    path = Path(root) / pin['path']
    actual = artifact_pin(path)
    require(actual['sha256'] == pin['sha256']
            and ('bytes' not in pin or pin['bytes'] == actual['bytes']), f'Artifact hash mismatch: {path}')
    return path


def indexed(records, expected, label):
    require(isinstance(records, list), f'{label} must be an array')
    result = {}
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get('id'), str),
                f'Invalid {label} ID')
        identifier = record['id']
        require(identifier not in result, f'Duplicate {label} ID: {identifier}')
        require(identifier in expected, f'Unknown {label} ID: {identifier}')
        result[identifier] = record
    require(set(result) == set(expected), f'Missing {label} IDs: {sorted(set(expected) - set(result))}')
    return result


def load_suite(manifest_path, expected_sha=None):
    path = Path(manifest_path).absolute()
    identity = artifact_pin(path)
    if expected_sha is not None:
        require(identity['sha256'] == expected_sha, 'Manifest hash mismatch')
    manifest = load_json(path)
    require(manifest.get('schema') == MANIFEST_SCHEMA and manifest.get('profile') == PROFILE,
            'Unsupported evaluation manifest/profile')
    input_path = pinned_path(path.parent, manifest['inputs'])
    # Reuse the existing strict public field contract, additionally rejecting
    # duplicate JSON keys before its legacy JSON reader is invoked.
    load_json(input_path)
    rows, _ = public_inputs(input_path, manifest['inputs']['sha256'])
    reference_path = pinned_path(path.parent, manifest['references'])
    references = indexed(load_json(reference_path), {row['id'] for row in rows}, 'reference')
    for row in rows:
        reference = references[row['id']]
        require(reference.get('prompt_sha256') == object_sha(row['messages']),
                f"Reference prompt hash mismatch: {row['id']}")
        require(reference.get('row_type', 'evidence') in ('evidence', 'general'), 'Unknown row type')
        require(all(isinstance(reference.get(key), str) and reference[key]
                    for key in ('family_id', 'condition', 'institution', 'parent_id')),
                'Missing reference aggregation identity')
        require('target' in reference and isinstance(reference.get('rubric'), list),
                'Missing private target/rubric')
        if reference.get('row_type', 'evidence') == 'evidence':
            packet_from_messages(row['messages'])
    groups = manifest.get('groups')
    require(isinstance(groups, list), 'Missing relation groups')
    group_ids = set()
    memberships = {identifier: set() for identifier in references}
    for group in groups:
        require(isinstance(group, dict) and isinstance(group.get('id'), str)
                and group['id'] and group['id'] not in group_ids, 'Duplicate/invalid group ID')
        members = group.get('members')
        require(isinstance(members, list) and len(members) >= 2
                and all(isinstance(member, str) and member in references for member in members)
                and len(set(members)) == len(members), 'Unknown/duplicate/missing group members')
        require(group.get('relation') in ('invariant', 'support_removed', 'changed_value_or_condition', 'conflict'),
                'Unknown group relation')
        group_ids.add(group['id'])
        for member in members:
            memberships[member].add(group['id'])
    for identifier, reference in references.items():
        declared = reference.get('relation_groups', [])
        require(isinstance(declared, list) and all(isinstance(item, str) for item in declared)
                and len(declared) == len(set(declared)) and set(declared) == memberships[identifier],
                'Reference/group membership differs')
    return manifest, rows, references, identity


NATIVE_RECEIPT_FIELDS = {
    'rendered_prompt', 'rendered_prompt_sha256', 'prompt_token_ids', 'output_token_ids',
    'dispatch_kwargs', 'generation_profile', 'mode', 'seed', 'template_kwargs',
    'context_limit', 'generation_policy', 'terminal_stop_token_id',
}
POLICY_FIELDS = ('generation_config', 'dispatch_kwargs', 'generation_profile', 'mode',
                 'template_kwargs', 'context_limit')


def validate_generation_receipt(record):
    receipt = record.get('generation_receipt')
    require(receipt in ('hf_native_v1', 'external_unverified'), 'Explicit generation receipt kind required')
    if receipt == 'external_unverified':
        require(not NATIVE_RECEIPT_FIELDS.intersection(record), 'Partial native receipts cannot be external/unverified')
        return
    require(NATIVE_RECEIPT_FIELDS <= set(record), 'Incomplete native generation receipt')
    require('error' in record and isinstance(record.get('raw_output'), str), 'Missing native result/exception receipt')
    require(record['terminal_stop_token_id'] is None or type(record['terminal_stop_token_id']) is int,
            'Invalid native terminal stop token')
    require(type(record.get('context_truncated')) is bool and type(record.get('truncated_output')) is bool,
            'Native receipt requires truncation flags')
    config = record['generation_config']
    budget = config.get('max_new_tokens')
    require(type(budget) is int and budget > 0, 'Invalid native output budget')
    require(type(record['context_limit']) is int and record['context_limit'] > 0, 'Invalid native context limit')
    require(record['mode'] in ('smoke', 'evaluation'), 'Invalid native generation mode')
    require(type(record['seed']) is int and record['seed'] == case_seed(record['mode'], record['id']),
            'Native case seed mismatch')
    dispatch = generation_dispatch(record['generation_profile'])
    require(object_sha(record['dispatch_kwargs']) == object_sha(dispatch)
            and all(object_sha(config.get(key)) == object_sha(value) for key, value in dispatch.items()
                    if key != 'use_model_defaults'),
            'Native dispatch/config mismatch')
    require(object_sha(record['template_kwargs']) == object_sha({'add_generation_prompt': True, 'enable_thinking': False}),
            'Native template policy mismatch')
    policy = record['generation_policy']
    if policy is not None and policy != {}:
        require(isinstance(policy, dict) and policy.get('effective_config_sha256') == object_sha(config)
                and policy.get('generation_mode') == ('sample' if record['generation_profile'] == SAMPLING_PROFILE
                                                       else 'greedy_search'), 'Observed generation policy mismatch')
    if record['error'] is None:
        require(bool(policy), 'Missing observed generation policy')
        require(isinstance(record['rendered_prompt'], str)
                and record['rendered_prompt'].endswith('<think>\n\n</think>\n\n')
                and bool(record['prompt_token_ids']), 'Missing native prompt receipt')
        require(not record['context_truncated']
                and record['input_tokens'] + budget <= record['context_limit'], 'Native context budget mismatch')
    else:
        require(record['finish_reason'] == 'exception', 'Exception must retain exception finish')
    output = record['output_token_ids']
    require(len(output) <= budget, 'Native output exceeds token budget')
    eos = config.get('eos_token_id')
    stops = [] if eos is None else [eos] if type(eos) is int else eos
    require(isinstance(stops, list) and all(type(token) is int and token >= 0 for token in stops),
            'Invalid native EOS configuration')
    require(not any(token in stops for token in output[:-1]), 'Nonterminal native EOS token')
    terminal = output[-1] if output and output[-1] in stops else None
    if record['error'] is None:
        # Same token-policy rules as decode_generation; no tokenizer/model load.
        finish = 'empty' if not output else 'eos' if terminal is not None else 'length'
        require(not output or terminal is not None or len(output) == budget, 'Unexplained native generation termination')
        require(record['finish_reason'] == finish and record['terminal_stop_token_id'] == terminal,
                'Native finish/token receipt mismatch')
        require(record['truncated_output'] == (finish == 'length'), 'Native truncation receipt mismatch')


def validate_paired_generation(records):
    all_records = [record for arm in records.values() for record in arm.values()]
    kinds = {record['generation_receipt'] for record in all_records}
    require(len(kinds) == 1, 'Mixed native/unverified generation receipts')
    native = kinds == {'hf_native_v1'}
    fields = POLICY_FIELDS if native else ('generation_config',)
    baseline = object_sha({key: all_records[0][key] for key in fields})
    require(all(object_sha({key: record[key] for key in fields}) == baseline for record in all_records),
            'Paired/within-arm generation policy mismatch')
    if native:
        first, second = records.values()
        for identifier in first:
            left, right = first[identifier], second[identifier]
            require(left['seed'] == right['seed'], 'Paired case seed mismatch')
            if left['rendered_prompt'] is not None and right['rendered_prompt'] is not None:
                require(left['rendered_prompt'] == right['rendered_prompt'], 'Paired native prompt mismatch')
            if left['prompt_token_ids'] and right['prompt_token_ids']:
                require(left['prompt_token_ids'] == right['prompt_token_ids'], 'Paired native tokenization mismatch')
    return {'receipt_kind': next(iter(kinds)), 'matched_native_policy': native,
            'declared_config_matched': True,
            'observed_responses': sum(bool(record.get('generation_policy')) for record in all_records) if native else 0}


def load_responses(path, rows, expected_sha=None):
    identity = artifact_pin(path)
    if expected_sha is not None:
        require(identity['sha256'] == expected_sha, 'Response artifact hash mismatch')
    text = Path(path).read_text(encoding='utf-8')
    if text.lstrip().startswith('['):
        records = strict_json(text)
    else:
        records = [strict_json(line) for line in text.splitlines() if line.strip()]
    by_id = indexed(records, {row['id'] for row in rows}, 'response')
    for row in rows:
        record = by_id[row['id']]
        require(record.get('prompt_sha256') == object_sha(row['messages']), 'Response prompt hash mismatch')
        require(isinstance(record.get('response'), str), 'Response body must be a string (including empty failures)')
        require(isinstance(record.get('finish_reason'), str), 'Missing finish reason')
        require(record.get('error') is None or isinstance(record['error'], (str, dict)), 'Invalid exception record')
        require(isinstance(record.get('generation_config'), dict), 'Missing actual generation config')
        for key in ('input_tokens', 'output_tokens'):
            require(record.get(key) is None or type(record[key]) is int and record[key] >= 0,
                    'Invalid token count')
        require('input_tokens' in record and 'output_tokens' in record, 'Missing token counts')
        for key in ('context_truncated', 'truncated_output'):
            require(key not in record or type(record[key]) is bool, 'Invalid truncation flag')
        if 'prompt_token_ids' in record:
            require(isinstance(record['prompt_token_ids'], list)
                    and all(type(t) is int and t >= 0 for t in record['prompt_token_ids'])
                    and len(record['prompt_token_ids']) == record['input_tokens'], 'Prompt token count mismatch')
        if 'output_token_ids' in record:
            require(isinstance(record['output_token_ids'], list)
                    and all(type(t) is int and t >= 0 for t in record['output_token_ids'])
                    and len(record['output_token_ids']) == record['output_tokens'], 'Output token count mismatch')
        if 'rendered_prompt' in record and record['rendered_prompt'] is not None:
            require(isinstance(record['rendered_prompt'], str)
                    and hashlib.sha256(record['rendered_prompt'].encode()).hexdigest()
                    == record.get('rendered_prompt_sha256'), 'Rendered prompt hash mismatch')
        validate_generation_receipt(record)
    return by_id, identity


def _blind_payload(rows, references, groups, records, seed):
    rng = random.Random(seed)
    arms = sorted(records)
    rng.shuffle(arms)
    arm_map = {f'arm-{i + 1}': arm for i, arm in enumerate(arms)}
    order = list(rows)
    rng.shuffle(order)
    case_map = {f'case-{i + 1:06d}': row['id'] for i, row in enumerate(order)}
    aliases = {identifier: alias for alias, identifier in case_map.items()}
    response_map = {}
    cases = []
    response_lookup = {}
    for case_alias, row in zip(case_map, order):
        responses = []
        for arm_alias, arm in arm_map.items():
            record = records[arm][row['id']]
            response_id = f'response-{rng.getrandbits(128):032x}'
            response_map[response_id] = {'case_id': row['id'], 'arm_alias': arm_alias}
            response_lookup[(row['id'], arm_alias)] = response_id
            responses.append({'response_id': response_id, 'arm_alias': arm_alias,
                              'response': record['response'],
                              'finish_reason': record['finish_reason'] if record['finish_reason'] in
                              ('stop', 'eos', 'length', 'exception', 'empty') else 'unknown',
                              'exception': record.get('error') is not None,
                              'truncated': bool(record.get('context_truncated') or record.get('truncated_output'))})
        rng.shuffle(responses)
        cases.append({'case_id': case_alias, 'messages': row['messages'], 'responses': responses})
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)
    group_map = {}
    public_groups = []
    for index, group in enumerate(shuffled_groups):
        group_alias = f'group-{index + 1:06d}'
        group_map[group_alias] = group['id']
        public_groups.append({'group_id': group_alias, 'relation': group['relation'],
                              'members': [aliases[identifier] for identifier in group['members']],
                              'arms': [{'arm_alias': alias, 'response_ids': [response_lookup[(identifier, alias)]
                                        for identifier in group['members']]} for alias in arm_map]})
    packet = {'schema': PACKET_SCHEMA, 'protocol': PROTOCOL, 'arm_aliases': list(arm_map),
              'cases': cases, 'groups': public_groups}
    maps = {'arm_map': arm_map, 'case_map': case_map, 'response_map': response_map, 'group_map': group_map}
    return packet, maps


def build_blind_packet(manifest_path, arm_paths, *, seed=None, manifest_sha256=None, arm_sha256=None):
    require(isinstance(arm_paths, dict) and len(arm_paths) == 2
            and all(isinstance(arm, str) and arm for arm in arm_paths), 'Exactly two named arms required')
    manifest, rows, references, manifest_pin = load_suite(manifest_path, manifest_sha256)
    records, pins = {}, {}
    for arm, path in arm_paths.items():
        records[arm], pins[arm] = load_responses(path, rows, None if arm_sha256 is None else arm_sha256[arm])
    generation_authentication = validate_paired_generation(records)
    seed = secrets.token_hex(32) if seed is None else seed
    require(type(seed) in (str, int), 'Blind seed must be text or integer')
    packet, maps = _blind_payload(rows, references, manifest['groups'], records, seed)
    mapping = {'schema': MAPPING_SCHEMA, 'protocol': PROTOCOL, 'manifest': manifest_pin,
               'artifacts': pins, 'seed': seed, 'packet_sha256': object_sha(packet),
               'generation_authentication': generation_authentication, **maps}
    return packet, mapping


def _judgments(judgments, packet, mapping):
    require(isinstance(judgments, dict) and judgments.get('schema') == JUDGMENT_SCHEMA,
            'Unsupported semantic judgment schema')
    require(judgments.get('packet_sha256') == mapping['packet_sha256'], 'Judgment packet hash mismatch')
    case_judgments = {}
    expected = mapping['response_map']
    require(isinstance(judgments.get('cases'), list), 'Missing semantic case judgments')
    for judgment in judgments['cases']:
        require(isinstance(judgment, dict), 'Invalid semantic judgment')
        identifier = judgment.get('response_id')
        require(isinstance(identifier, str) and identifier in expected, 'Unknown semantic response ID')
        require(identifier not in case_judgments, 'Duplicate semantic response ID')
        require(type(judgment.get('pass')) is bool and type(judgment.get('critical_error')) is bool,
                'Semantic judgment requires explicit pass and critical_error booleans')
        require(not (judgment['pass'] and judgment['critical_error']), 'Critical error cannot pass')
        case_judgments[identifier] = judgment
    require(set(case_judgments) == set(expected), 'Incomplete semantic case judgments')
    expected_groups = {(group['group_id'], arm) for group in packet['groups'] for arm in packet['arm_aliases']}
    group_judgments = {}
    require(isinstance(judgments.get('groups'), list), 'Missing semantic group judgments')
    for judgment in judgments['groups']:
        require(isinstance(judgment, dict) and isinstance(judgment.get('group_id'), str)
                and isinstance(judgment.get('arm_alias'), str), 'Invalid semantic group judgment')
        key = (judgment['group_id'], judgment['arm_alias'])
        require(key in expected_groups, 'Unknown semantic group/arm')
        require(key not in group_judgments, 'Duplicate semantic group judgment')
        require(type(judgment.get('pass')) is bool, 'Group judgment requires explicit pass boolean')
        group_judgments[key] = judgment
    require(set(group_judgments) == expected_groups, 'Incomplete semantic group judgments')
    return case_judgments, group_judgments


def structural_checks(record, row, reference):
    completed = (record.get('error') is None and bool(record['response'].strip())
                 and record['finish_reason'] in ('stop', 'eos')
                 and type(record.get('input_tokens')) is int and record['input_tokens'] > 0
                 and type(record.get('output_tokens')) is int and record['output_tokens'] > 0
                 and not record.get('context_truncated', False)
                 and not record.get('truncated_output', False))
    if reference.get('row_type', 'evidence') == 'general':
        structural = bool(record['response'].strip())
        errors = [] if structural else ['empty_response']
        citation_valid = None
    else:
        packet = packet_from_messages(row['messages'])
        result = validate_evidence_answer(record['response'], packet)
        structural, errors = result['valid'], result['errors']
        citation_valid = False
        try:
            body = strict_json(record['response'])
            paragraphs = body.get('paragraphs') if isinstance(body, dict) else None
            known = {item['id'] for item in packet['evidence']}
            citation_valid = (isinstance(paragraphs, list) and 1 <= len(paragraphs) <= 5
                              and all(isinstance(p, dict) and isinstance(p.get('sources'), list)
                                      and (bool(p['sources']) or not known)
                                      and all(isinstance(s, str) and s in known for s in p['sources'])
                                      and len(p['sources']) == len(set(p['sources'])) for p in paragraphs))
        except (ValueError, TypeError):
            pass
    return {'completion_valid': completed, 'structural_valid': bool(structural),
            'citation_valid': citation_valid, 'structural_errors': errors}


def _aggregate(items, semantic):
    result = {'cases': len(items), 'completed': sum(item['completion_valid'] for item in items),
              'structurally_valid': sum(item['structural_valid'] for item in items),
              'citation_applicable': sum(item['citation_valid'] is not None for item in items),
              'citation_valid': sum(item['citation_valid'] is True for item in items)}
    if semantic:
        result.update(semantic_judged=len(items), semantic_pass=sum(item['semantic_pass'] for item in items),
                      critical_errors=sum(item['critical_error'] for item in items),
                      accepted=sum(item['accepted'] for item in items))
    return result


def score_packet(packet, mapping, judgments=None):
    require(isinstance(mapping, dict) and mapping.get('schema') == MAPPING_SCHEMA
            and mapping.get('protocol') == PROTOCOL, 'Unsupported private blind mapping')
    require(object_sha(packet) == mapping.get('packet_sha256'), 'Blind packet hash mismatch')
    manifest_path = pinned_path('.', mapping['manifest'])
    arm_paths = {arm: pinned_path('.', pin) for arm, pin in mapping['artifacts'].items()}
    reconstructed, rebuilt = build_blind_packet(manifest_path, arm_paths, seed=mapping['seed'],
                                               manifest_sha256=mapping['manifest']['sha256'])
    require(reconstructed == packet and rebuilt == mapping, 'Blind mapping/artifact binding mismatch')
    manifest, rows, references, _ = load_suite(manifest_path, mapping['manifest']['sha256'])
    row_map = {row['id']: row for row in rows}
    records = {arm: load_responses(path, rows)[0] for arm, path in arm_paths.items()}
    semantic = judgments is not None
    case_judgments, group_judgments = _judgments(judgments, packet, mapping) if semantic else ({}, {})
    results = []
    for response_id, identity in mapping['response_map'].items():
        identifier, alias = identity['case_id'], identity['arm_alias']
        reference = references[identifier]
        checks = structural_checks(records[mapping['arm_map'][alias]][identifier], row_map[identifier], reference)
        result = {'id': identifier, 'response_id': response_id, 'arm_alias': alias, **checks,
                  **{key: reference[key] for key in ('institution', 'family_id', 'condition', 'parent_id')}}
        if semantic:
            judgment = case_judgments[response_id]
            result.update(semantic_pass=judgment['pass'], critical_error=judgment['critical_error'],
                          accepted=checks['completion_valid'] and checks['structural_valid']
                          and checks['citation_valid'] is not False and judgment['pass'] and not judgment['critical_error'])
        results.append(result)
    arms = {}
    for alias in mapping['arm_map']:
        selected = [item for item in results if item['arm_alias'] == alias]
        summary = _aggregate(selected, semantic)
        summary['by'] = {key: {value: _aggregate([item for item in selected if item[key] == value], semantic)
                               for value in sorted({item[key] for item in selected})}
                         for key in ('institution', 'family_id', 'condition', 'parent_id')}
        summary['relation_groups_expected'] = len(manifest['groups'])
        if semantic:
            summary['relation_groups_judged'] = len(manifest['groups'])
            summary['relation_groups_pass'] = sum(judgment['pass'] for (group, arm), judgment in group_judgments.items() if arm == alias)
        arms[alias] = summary
    group_results = []
    if semantic:
        result_index = {item['response_id']: item for item in results}
        for group in packet['groups']:
            for alias in packet['arm_aliases']:
                relation_pass = group_judgments[(group['group_id'], alias)]['pass']
                members = next(arm['response_ids'] for arm in group['arms'] if arm['arm_alias'] == alias)
                accepted = relation_pass and all(result_index[member]['accepted'] for member in members)
                group_results.append({'group_id': mapping['group_map'][group['group_id']], 'arm_alias': alias,
                                      'relation': group['relation'], 'pass': relation_pass, 'accepted': accepted})
        for alias in arms:
            arms[alias]['relation_groups_accepted'] = sum(group['accepted'] for group in group_results if group['arm_alias'] == alias)
    return {'schema': 'trid-evidence-assessment-v1', 'protocol': PROTOCOL,
            'packet_sha256': mapping['packet_sha256'], 'semantic_evaluated': semantic,
            'manifest': mapping['manifest'], 'suite_scope': manifest.get('scope', 'complete_manifest'),
            'generation_authentication': mapping['generation_authentication'],
            'denominators': {'unique_cases': len(rows), 'arms': len(arms), 'responses': len(results),
                             'relation_groups': len(manifest['groups']),
                             'relation_group_arm_verdicts': len(manifest['groups']) * len(arms)},
            'arms': arms, 'cases': results, 'groups': group_results,
            'limitations': ['Offline protocol; not registered with the sealed cloud evaluator.',
                            'Structural/completion/citation checks do not establish semantic correctness.']}


@contextlib.contextmanager
def observe_policy(model, expected, dispatch):
    """The existing observer's exact-config check, without its fixed768 budget."""
    target = generation_target(model)
    original = target._prepare_generation_config
    sentinel = object()
    previous = target.__dict__.get('_prepare_generation_config', sentinel)
    observed = {}
    expected_sha = object_sha(expected.to_dict())

    def observe(requested, use_model_defaults=None, **kwargs):
        require(use_model_defaults is False
                and all(key in kwargs and kwargs[key] == value
                        for key, value in dispatch.items() if key != 'use_model_defaults'),
                'Explicit generation dispatch controls missing')
        effective, model_kwargs = original(requested, use_model_defaults=use_model_defaults, **kwargs)
        require(object_sha(effective.to_dict()) == expected_sha, 'Actual effective generation config differs')
        observed.update(effective_config_sha256=expected_sha, generation_mode=effective.get_generation_mode().value)
        return effective, model_kwargs

    target._prepare_generation_config = observe
    try:
        yield observed
        require(observed, 'Generation bypassed actual policy observation')
    finally:
        if previous is sentinel:
            delattr(target, '_prepare_generation_config')
        else:
            target._prepare_generation_config = previous


def generate_responses(model, tokenizer, public_rows, *, max_new_tokens=768, context_limit=4096,
                       profile=SAMPLING_PROFILE, mode='evaluation'):
    """Generate once per public row, without downloads, retries, clipping or repair.

    Exceptions are retained as response records, including context overflow. The
    caller supplies an already-admitted model and owns resource supervision. This
    API does not register a GPU job or provide an alternate model-loading route.
    """
    import torch
    from transformers import GenerationConfig
    require(type(max_new_tokens) is int and max_new_tokens > 0, 'Invalid max_new_tokens')
    require(type(context_limit) is int and context_limit > 0, 'Invalid context_limit')
    require(isinstance(public_rows, list) and public_rows, 'Expected nonempty public rows')
    seen = set()
    for row in public_rows:
        require(isinstance(row, dict) and set(row) == {'id', 'messages'}
                and isinstance(row['id'], str) and row['id'] and row['id'] not in seen,
                'Invalid/duplicate public generation ID')
        messages = row['messages']
        require(isinstance(messages, list) and messages
                and all(isinstance(message, dict) and set(message) == {'role', 'content'}
                        and message['role'] in ('system', 'user', 'assistant')
                        and isinstance(message['content'], str) for message in messages)
                and messages[-1]['role'] == 'user', 'Invalid public generation messages')
        seen.add(row['id'])
    require(mode in ('smoke', 'evaluation'), 'Unknown generation scope')
    dispatch = generation_dispatch(profile)
    # Start from neutral HF controls, retaining only model-specific token IDs.
    # Inheriting an arbitrary saved decoder would silently confound this profile.
    requested = GenerationConfig(bos_token_id=model.generation_config.bos_token_id,
                                 eos_token_id=model.generation_config.eos_token_id,
                                 pad_token_id=model.generation_config.pad_token_id)
    requested.update(do_sample=False, num_beams=1, num_beam_groups=1,
                     max_new_tokens=max_new_tokens, min_new_tokens=None, min_length=0,
                     return_dict_in_generate=True, output_scores=False,
                     output_logits=False, num_return_sequences=1, use_cache=True,
                     temperature=1.0, top_p=1.0, top_k=50, typical_p=1.0,
                     repetition_penalty=1.0, no_repeat_ngram_size=0,
                     forced_bos_token_id=None, forced_eos_token_id=None,
                     stop_strings=None, max_time=None)
    if profile == SAMPLING_PROFILE:
        requested.update(**SAMPLING_PARAMETERS)
    if requested.pad_token_id is None:
        requested.pad_token_id = tokenizer.pad_token_id
    target = generation_target(model)
    effective, unused = target._prepare_generation_config(requested, **dispatch)
    expected_mode = 'sample' if profile == SAMPLING_PROFILE else 'greedy_search'
    require(not unused and effective.get_generation_mode().value == expected_mode,
            'Unexpected effective generation policy')
    config = effective.to_dict()
    # Bound against both the explicit evaluation envelope and the actual model.
    model_context = getattr(model.config, 'max_position_embeddings', context_limit)
    limit = min(context_limit, model_context) if type(model_context) is int and model_context > 0 else context_limit
    was_training = model.training
    model.eval()
    records = []
    try:
        for row in public_rows:
            record = {'id': row['id'], 'prompt_sha256': object_sha(row['messages']),
                      'generation_receipt': 'hf_native_v1', 'generation_policy': None, 'terminal_stop_token_id': None,
                      'rendered_prompt': None, 'rendered_prompt_sha256': None,
                      'prompt_token_ids': [], 'output_token_ids': [], 'input_tokens': 0, 'output_tokens': 0,
                      'response': '', 'raw_output': '', 'finish_reason': 'exception', 'error': None,
                      'generation_config': copy.deepcopy(config), 'dispatch_kwargs': dict(dispatch),
                      'generation_profile': profile, 'mode': mode, 'seed': case_seed(mode, row['id']),
                      'template_kwargs': {'add_generation_prompt': True, 'enable_thinking': False},
                      'context_limit': limit, 'context_truncated': False, 'truncated_output': False}
            try:
                prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
                require(isinstance(prompt, str), 'Native chat template did not return text')
                record['rendered_prompt'] = prompt
                record['rendered_prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
                require(prompt.endswith('<think>\n\n</think>\n\n'), 'Native Qwen3 nonthinking prefix differs')
                tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
                require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens),
                        'Invalid native prompt token IDs')
                record.update(prompt_token_ids=tokens, input_tokens=len(tokens))
                require(len(tokens) + max_new_tokens <= limit, 'Input exceeds context envelope; no truncation')
                device = model.get_input_embeddings().weight.device
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                torch.manual_seed(record['seed'])
                if device.type == 'mps':
                    torch.mps.manual_seed(record['seed'])
                with torch.inference_mode(), observe_policy(model, effective, dispatch) as observed:
                    record['generation_policy'] = observed
                    result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            generation_config=effective, **dispatch)
                record['generation_policy'] = dict(observed)
                output = result.sequences[0, len(tokens):].tolist()
                record.update(output_token_ids=output, output_tokens=len(output))
                if output:
                    decoded = decode_generation(tokenizer, output, config['eos_token_id'], max_new_tokens)
                    record.update(raw_output=decoded['raw_output'], response=decoded['response_text'],
                                  finish_reason=decoded['termination_reason'],
                                  truncated_output=decoded['termination_reason'] == 'length',
                                  terminal_stop_token_id=decoded['terminal_stop_token_id'])
                else:
                    record['finish_reason'] = 'empty'
            except Exception as exc:
                record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                record['finish_reason'] = 'exception'
            records.append(record)
    finally:
        model.train(was_training)
    return records


def write_new_json(path, value, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental mutation of historical runs.
    import os
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    blind = sub.add_parser('blind-packet', help='Validate both complete pinned arms and create blinded review artifacts')
    blind.add_argument('--manifest', required=True)
    blind.add_argument('--manifest-sha256', required=True)
    for arm in ('a', 'b'):
        blind.add_argument(f'--arm-{arm}', required=True)
        blind.add_argument(f'--arm-{arm}-sha256', required=True)
        blind.add_argument(f'--arm-{arm}-name', default=arm)
    blind.add_argument('--seed', help='Private reproducibility seed; random by default')
    blind.add_argument('--packet', required=True)
    blind.add_argument('--mapping', required=True)
    score = sub.add_parser('score', help='Structural diagnostics, optionally complete packet-bound semantic judgments')
    score.add_argument('--packet', required=True)
    score.add_argument('--mapping', required=True)
    score.add_argument('--judgments')
    score.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'blind-packet':
        require(args.arm_a_name != args.arm_b_name, 'Arm names must differ')
        require(Path(args.packet).resolve() != Path(args.mapping).resolve(), 'Mapping must be separate from reviewer packet')
        require(not Path(args.packet).exists() and not Path(args.mapping).exists(), 'Blind outputs must be new artifacts')
        packet, mapping = build_blind_packet(args.manifest, {args.arm_a_name: args.arm_a, args.arm_b_name: args.arm_b},
                                             seed=args.seed, manifest_sha256=args.manifest_sha256,
                                             arm_sha256={args.arm_a_name: args.arm_a_sha256, args.arm_b_name: args.arm_b_sha256})
        write_new_json(args.mapping, mapping, private=True)
        write_new_json(args.packet, packet)
    else:
        report = score_packet(load_json(args.packet), load_json(args.mapping),
                              None if args.judgments is None else load_json(args.judgments))
        write_new_json(args.output, report, private=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
