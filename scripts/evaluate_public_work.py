"""Pinned offline public-work diagnostics, not fresh gold or model acceptance.

The private blind packet contains reference rubrics and unchanged observed answers,
but no arm/model labels. External semantic judgments are human/model-authored
claims, not verified native inference. No judgments means no semantic score.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_evidence_conditioning import artifact_pin, load_json, write_new_json
from evaluate_local_eight import object_sha, require, strict_json
from trid.service.public_work import (TASKS, public_row, reference_row, validate_answer,
                                      validate_case, validate_reference)

PROTOCOL = 'trid-public-work-offline-paired-v1'
PACKET_SCHEMA = 'trid-public-work-blind-packet-v1'
MAPPING_SCHEMA = 'trid-public-work-blind-mapping-v1'
JUDGMENT_SCHEMA = 'trid-public-work-judgments-v1'
GENERAL_TASKS = frozenset(('general_dialogue', 'general_arithmetic'))


def pinned_path(root, pin):
    require(isinstance(pin, dict) and isinstance(pin.get('path'), str)
            and type(pin.get('bytes')) is int and pin['bytes'] >= 0
            and isinstance(pin.get('sha256'), str), 'Invalid artifact pin')
    path = Path(root) / pin['path']
    actual = artifact_pin(path)
    require(actual['sha256'] == pin['sha256'] and actual['bytes'] == pin['bytes'],
            f'Artifact hash mismatch: {path}')
    return path


def _index(records, expected, label, *, complete=True):
    require(isinstance(records, list), f'{label} must be an array')
    result = {}
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get('id'), str), f'Invalid {label} ID')
        identifier = record['id']
        require(identifier in expected, f'Unknown {label} ID: {identifier}')
        require(identifier not in result, f'Duplicate {label} ID: {identifier}')
        result[identifier] = record
    require(not complete or set(result) == set(expected), f'Missing {label} IDs')
    return result


def load_suite(manifest_path, expected_sha=None):
    path = Path(manifest_path).resolve()
    identity = artifact_pin(path)
    require(expected_sha is None or identity['sha256'] == expected_sha, 'Manifest hash mismatch')
    manifest = load_json(path)
    require(manifest.get('schema') == 'trid-public-work-suite-v1'
            and manifest.get('profile') == 'public_work_v1' and manifest.get('scope') == 'diagnostic',
            'Unsupported public-work diagnostic suite')
    require(manifest.get('expected_tasks') == list(TASKS), 'Required task inventory mismatch')
    require(all(manifest.get(key) is False for key in
                ('training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen')),
            'Diagnostic suite cannot claim admission or fresh gold')
    source = load_json(pinned_path(path.parent, manifest['source']))
    require(source.get('schema') == 'trid-public-work-source-v1' and source.get('scope') == 'diagnostic'
            and isinstance(source.get('cases'), list), 'Invalid diagnostic source')
    authored = source['cases']
    expected = set()
    for case in authored:
        validate_case(case)
        require(case['id'] not in expected, 'Duplicate authored ID')
        expected.add(case['id'])
    require(expected and manifest.get('case_count') == len(expected), 'Suite case count mismatch')
    rows = load_json(pinned_path(path.parent, manifest['inputs']))
    row_map = _index(rows, expected, 'public input')
    refs = _index(load_json(pinned_path(path.parent, manifest['references'])), expected, 'reference')
    for case in authored:
        row, ref = row_map[case['id']], refs[case['id']]
        validate_reference(ref)
        require(row == public_row(case) and ref == reference_row(case, row),
                'Source/public/reference projection mismatch')
        require(ref['prompt_sha256'] == object_sha(row['messages']), 'Reference prompt hash mismatch')
    return manifest, rows, refs, identity


def _ids(value, expected, label):
    require(isinstance(value, list) and all(isinstance(item, str) for item in value)
            and len(value) == len(set(value)) and set(value) == set(expected), f'{label} mismatch')


def validate_receipt(record, row, runtime):
    require(record.get('generation_receipt') == 'public_work_hf_v1',
            'Only source-bound public_work_hf_v1 receipts are native; external responses are not verified')
    require(record.get('prompt_sha256') == object_sha(row['messages']), 'Response prompt hash mismatch')
    require(all(isinstance(record.get(key), str) for key in ('response', 'raw_output', 'finish_reason')),
            'Malformed response body/finish')
    require('error' in record and (record['error'] is None or isinstance(record['error'], (str, dict))),
            'Missing or malformed exception receipt')
    require(record['error'] is None or record['finish_reason'] == 'exception', 'Exception finish mismatch')
    for flag in ('context_truncated', 'truncated_output'):
        require(type(record.get(flag)) is bool, 'Missing truncation flags')
    for tokens, count in (('prompt_token_ids', 'input_tokens'), ('output_token_ids', 'output_tokens')):
        require(isinstance(record.get(tokens), list)
                and all(type(token) is int and token >= 0 for token in record[tokens])
                and type(record.get(count)) is int and record[count] == len(record[tokens]),
                'Native token count mismatch')
    require(type(record.get('seed')) is int and record.get('mode') == 'evaluation', 'Invalid seed/mode')
    policy = runtime['generation_policy']
    require(type(record.get('context_limit')) is int and record['context_limit'] == policy['context_limit'],
            'Context policy mismatch')
    config = record.get('generation_config')
    require(isinstance(config, dict) and config.get('max_new_tokens') == policy['max_new_tokens'],
            'Output budget policy mismatch')
    require(record['output_tokens'] <= policy['max_new_tokens'], 'Output exceeds requested budget')
    require(isinstance(record.get('template_kwargs'), dict), 'Missing native template kwargs')
    require(record['seed'] == policy.get('seed') and policy.get('seed_policy') == 'constant'
            and record['mode'] == policy.get('mode'), 'Requested seed/mode policy mismatch')
    require(all(config.get(key) == policy.get(key) for key in ('do_sample', 'num_beams')),
            'Requested generation dispatch policy mismatch')
    require(record['template_kwargs'] == {**policy.get('template_kwargs', {}), 'add_generation_prompt': True},
            'Requested template policy mismatch')
    rendered = record.get('rendered_prompt')
    require(rendered is None and record.get('rendered_prompt_sha256') is None and record['error'] is not None
            or isinstance(rendered, str) and hashlib.sha256(rendered.encode()).hexdigest()
            == record.get('rendered_prompt_sha256'), 'Rendered prompt hash mismatch')
    observed = record.get('generation_policy')
    require(isinstance(observed, dict) or observed is None and record['error'] is not None,
            'Missing observed generation policy')
    if record['error'] is None:
        require(bool(observed) and observed.get('effective_config_sha256') == object_sha(config)
                and observed.get('generation_mode') == 'greedy_search',
                'Observed generation config hash/mode mismatch')
        require(record['input_tokens'] > 0 and isinstance(rendered, str), 'Missing native prompt')
        require(record['context_truncated'] or record['input_tokens'] + policy['max_new_tokens']
                <= policy['context_limit'], 'Native context budget mismatch')
        require(record['truncated_output'] == (record['finish_reason'] == 'length'),
                'Native length receipt mismatch')
        eos = config.get('eos_token_id')
        stops = [eos] if type(eos) is int else eos
        require(isinstance(stops, list) and stops and all(type(token) is int and token >= 0 for token in stops),
                'Missing native stop token configuration')
        output = record['output_token_ids']
        if record['finish_reason'] in ('stop', 'eos'):
            require(output and output[-1] in stops, 'Native stop receipt mismatch')
        if record['finish_reason'] == 'length':
            require(len(output) == policy['max_new_tokens'] and output[-1] not in stops,
                    'Native length token budget mismatch')
    elapsed = record.get('wall_seconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'Invalid observed latency')


def load_arm(directory, rows, manifest_path, manifest, completion_sha256=None):
    directory = Path(directory).resolve()
    completion_path = directory / 'completion.json'
    completion_pin = artifact_pin(completion_path)
    require(completion_sha256 is None or completion_pin['sha256'] == completion_sha256,
            'Completion hash mismatch')
    completion = load_json(completion_path)
    require(completion.get('schema') == 'public_work_hf_completion_v1', 'Unsupported completion receipt')
    runtime_path = pinned_path(directory, completion['runtime'])
    response_path = pinned_path(directory, completion['responses'])
    require(runtime_path.resolve() == directory / 'runtime.json'
            and response_path.resolve() == directory / 'responses.jsonl', 'Arm artifact location mismatch')
    runtime = load_json(runtime_path)
    require(runtime.get('schema') == 'public_work_hf_runtime_v1', 'Unsupported native runtime')
    suite_path = pinned_path(directory, runtime['suite_manifest'])
    require(suite_path.resolve() == Path(manifest_path).resolve(), 'Runtime suite identity mismatch')
    for key in ('source', 'inputs'):
        pinned_path(directory, runtime[key])
        require(all(runtime[key][field] == manifest[key][field] for field in ('bytes', 'sha256')),
                f'Runtime {key} pin mismatch')
    model_manifest = load_json(pinned_path(directory, runtime['model_manifest']))
    model = runtime.get('model_identity')
    require(isinstance(model, dict) and all(isinstance(model.get(key), str) and model[key]
            and model[key] == model_manifest.get(key) for key in ('repo_id', 'revision')),
            'Model source identity mismatch')
    journal = response_path.read_bytes()
    lines = journal.splitlines(keepends=True)
    parsed = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            parsed.append(strict_json(line.decode('utf-8')))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            require(index == len(lines) - 1 and not line.endswith(b'\n')
                    and completion.get('complete') is False
                    and completion.get('journal_error') == str(error)
                    and (completion.get('worker_exit_code') not in (None, 0)
                         or bool(completion.get('guard_failure'))),
                    'Malformed journal is not an authenticated interrupted final write')
    records = _index(parsed, {row['id'] for row in rows}, 'response', complete=False)
    expected = {row['id'] for row in rows}
    _ids(completion.get('expected_ids'), expected, 'Completion expected IDs')
    _ids(completion.get('observed_ids'), records, 'Completion observed IDs')
    _ids(completion.get('missing_ids'), expected - records.keys(), 'Completion missing IDs')
    require(type(completion.get('complete')) is bool and 'worker_exit_code' in completion
            and (completion['worker_exit_code'] is None or type(completion['worker_exit_code']) is int)
            and 'guard_failure' in completion
            and (completion['guard_failure'] is None or isinstance(completion['guard_failure'], str)),
            'Malformed termination receipt')
    require(not completion['complete'] or (set(records) == expected and completion['worker_exit_code'] == 0
            and completion['guard_failure'] is None
            and all(record.get('error') is None and record.get('finish_reason') != 'exception'
                    for record in records.values())), 'False complete claim')
    tokenizer = runtime.get('tokenizer_identity')
    unavailable_tokenizer = tokenizer == {'files': [], 'template_sha256': None}
    if unavailable_tokenizer:
        require(not journal and completion['complete'] is False and runtime.get('status') == 'not_started'
                and (completion['worker_exit_code'] not in (None, 0) or bool(completion['guard_failure'])),
                'Unavailable tokenizer requires an authenticated failed zero-response preflight')
    else:
        require(isinstance(tokenizer, dict) and isinstance(tokenizer.get('files'), list)
                and tokenizer['files'] and isinstance(tokenizer.get('template_sha256'), str),
                'Missing tokenizer/template identity')
        model_files = model_manifest.get('files')
        require(isinstance(model_files, list) and all(isinstance(pin, dict)
                and isinstance(pin.get('path'), str) for pin in model_files), 'Invalid model file inventory')
        tokenizer_paths = {}
        for pin in tokenizer['files']:
            path = pinned_path(directory, pin)
            require(path.name not in tokenizer_paths, 'Duplicate tokenizer file identity')
            require(any(Path(source['path']).name == path.name
                        and all(source.get(key) == pin[key] for key in ('bytes', 'sha256'))
                        for source in model_files), 'Tokenizer file not bound to model manifest')
            tokenizer_paths[path.name] = path
        require({'tokenizer.json', 'tokenizer_config.json'} <= tokenizer_paths.keys(),
                'Incomplete official tokenizer inventory')
        if 'chat_template.jinja' in tokenizer_paths:
            template = tokenizer_paths['chat_template.jinja'].read_text(encoding='utf-8')
        else:
            template = load_json(tokenizer_paths['tokenizer_config.json']).get('chat_template')
            if isinstance(template, list):
                template = {item['name']: item['template'] for item in template}
            if isinstance(template, dict):
                template = template.get('default')
        require(isinstance(template, str) and hashlib.sha256(template.encode()).hexdigest()
                == tokenizer['template_sha256'], 'Official template identity mismatch')
    adapter = runtime.get('adapter')
    if adapter is not None:
        require(isinstance(adapter, dict) and adapter.get('model_identity') == model
                and adapter.get('representation') == 'unmerged_peft'
                and isinstance(adapter.get('files'), list), 'Invalid adapter/base identity')
        provenance_path = pinned_path(directory, adapter['provenance'])
        provenance = load_json(provenance_path)
        require(provenance.get('schema') == 'trid-public-evidence-adapter-v2'
                and provenance.get('fresh') is True and provenance.get('merged') is False,
                'Unsupported adapter provenance')
        provenance_root = provenance_path.parent.parent
        training_path = pinned_path(provenance_root, provenance['training'])
        require(pinned_path(directory, adapter['training']).resolve() == training_path.resolve(),
                'Adapter training pin mismatch')
        training = load_json(training_path)
        require(training.get('base_model_id') == model['repo_id']
                and training.get('base_revision') == model['revision'], 'Adapter training base mismatch')
        declared = provenance['adapter']
        require(type(adapter.get('steps')) is int and adapter['steps'] == declared.get('step')
                == provenance.get('observed_steps'), 'Adapter step identity mismatch')
        expected_adapter = {pinned_path(provenance_root, declared[key]).resolve() for key in ('config', 'weights')}
        actual_adapter = [pinned_path(directory, pin).resolve() for pin in adapter['files']]
        require(len(actual_adapter) == 2 and set(actual_adapter) == expected_adapter,
                'Adapter files/provenance mismatch')
    policy = runtime.get('generation_policy')
    require(isinstance(policy, dict) and all(type(policy.get(key)) is int and policy[key] > 0
            for key in ('max_new_tokens', 'context_limit')), 'Missing requested generation policy')
    require(policy.get('profile') == 'public_work_greedy_v1' and policy.get('do_sample') is False
            and policy.get('num_beams') == 1 and policy.get('seed_policy') == 'constant'
            and type(policy.get('seed')) is int and policy.get('mode') == 'evaluation'
            and isinstance(policy.get('template_kwargs'), dict), 'Unsupported requested generation policy')
    for row in rows:
        if row['id'] in records:
            validate_receipt(records[row['id']], row, runtime)
    return records, runtime, completion, completion_pin


def validate_paired_generation(arms):
    left, right = arms.values()
    require(left[1]['generation_policy'] == right[1]['generation_policy'], 'Paired requested policy mismatch')
    same_model = all(left[1]['model_identity'][key] == right[1]['model_identity'][key]
                     for key in ('repo_id', 'revision'))
    # Adapter identity does not license a different tokenizer for the same base.
    def tokenizer_signature(runtime):
        identity = runtime['tokenizer_identity']
        return (identity['template_sha256'], sorted((pin['sha256'], pin['bytes']) for pin in identity['files']))
    tokenizers_available = all(arm[1]['tokenizer_identity']['template_sha256'] is not None for arm in arms.values())
    require(not same_model or not tokenizers_available
            or tokenizer_signature(left[1]) == tokenizer_signature(right[1]),
            'Same-model tokenizer/template identity mismatch')
    for identifier in left[0].keys() & right[0].keys():
        a, b = left[0][identifier], right[0][identifier]
        require(all(a[key] == b[key] for key in ('seed', 'mode', 'context_limit', 'prompt_sha256')),
                'Paired public prompt/seed policy mismatch')
        if same_model and a['error'] is None and b['error'] is None:
            require(all(a[key] == b[key] for key in ('template_kwargs', 'rendered_prompt',
                        'prompt_token_ids', 'generation_config')), 'Same-model native prompt/policy mismatch')
    return {'receipt_kind': 'public_work_hf_v1' if any(arm[0] for arm in arms.values()) else None,
            'common_public_messages': True, 'common_requested_policy': True, 'same_base_model': same_model,
            'cross_tokenizer_comparison': not same_model if tokenizers_available else None,
            'observed_native_response_rows': sum(len(arm[0]) for arm in arms.values()),
            'arms': {name: {'tokenizer_identity_authenticated': arm[1]['tokenizer_identity']['template_sha256'] is not None,
                            'inference_observed': bool(arm[0]),
                            'status': ('failed_before_tokenizer_preflight'
                                       if arm[1]['tokenizer_identity']['template_sha256'] is None
                                       else 'native_receipts_observed' if arm[0] else 'no_inference_observed')}
                     for name, arm in arms.items()},
            'authentication_scope': 'Pinned local artifacts and available native receipts; failed preflight is not verified inference'}


def _eligible(record):
    return record is not None and record['error'] is None and bool(record['response'].strip())


def build_blind_packet(manifest_path, arm_paths, *, seed=None, manifest_sha256=None, arm_sha256=None):
    require(isinstance(arm_paths, dict) and len(arm_paths) == 2
            and all(isinstance(name, str) and name.strip() for name in arm_paths), 'Exactly two named arms required')
    manifest, rows, refs, manifest_pin = load_suite(manifest_path, manifest_sha256)
    arms = {name: load_arm(path, rows, manifest_path, manifest,
                          None if arm_sha256 is None else arm_sha256[name]) for name, path in arm_paths.items()}
    authentication = validate_paired_generation(arms)
    seed = secrets.token_hex(32) if seed is None else seed
    require(type(seed) in (str, int), 'Invalid blind seed')
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    cases, response_map, case_map = [], {}, {}
    for index, row in enumerate(order):
        case_id = f'case-{index + 1:06d}'
        case_map[case_id] = row['id']
        answers = []
        for name in sorted(arms):
            record = arms[name][0].get(row['id'])
            if record is None:
                continue  # Missing output is never invented as a response.
            response_id = f'response-{rng.getrandbits(128):032x}'
            response_map[response_id] = {'case_id': row['id'], 'arm': name}
            answers.append({'response_id': response_id, 'response': record['response'],
                            'raw_output': record['raw_output'], 'finish_reason': record['finish_reason'],
                            'exception': record['error'] is not None,
                            'truncated': record['context_truncated'] or record['truncated_output'],
                            'judgment_required': _eligible(record)})
        rng.shuffle(answers)
        reference = refs[row['id']]
        cases.append({'case_id': case_id, 'messages': row['messages'],
                      'reference': {key: reference[key] for key in
                                    ('target', 'rubric', 'output_contract', 'document_ids')},
                      'responses': answers})
    packet = {'schema': PACKET_SCHEMA, 'protocol': PROTOCOL, 'scope': 'observed_diagnostic_not_fresh_gold',
              'instructions': 'Judge each judgment_required answer against the messages and private rubric; '
                              'explain actual supported/unsupported claims. Labels and format do not establish semantics.',
              'cases': cases}
    mapping = {'schema': MAPPING_SCHEMA, 'protocol': PROTOCOL, 'manifest': manifest_pin,
               'artifacts': {name: arm[3] for name, arm in arms.items()}, 'seed': seed,
               'packet_sha256': object_sha(packet), 'case_map': case_map, 'response_map': response_map,
               'generation_authentication': authentication}
    return packet, mapping


def judgment_schema(packet_sha256, eligible_ids):
    """Machine-readable artifact; intentionally no fabricated judgment rows."""
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object',
            'additionalProperties': False, 'required': ['schema', 'packet_sha256', 'cases'],
            'properties': {'schema': {'const': JUDGMENT_SCHEMA}, 'packet_sha256': {'const': packet_sha256},
                           'cases': {'type': 'array', 'minItems': len(eligible_ids), 'maxItems': len(eligible_ids),
                           'items': {'type': 'object', 'additionalProperties': False,
                           'required': ['response_id', 'semantic_pass', 'critical_error', 'reason'],
                           'properties': {'response_id': {'type': 'string', **({'enum': eligible_ids} if eligible_ids else {})},
                                          'semantic_pass': {'type': 'boolean'},
                                          'critical_error': {'type': 'boolean'},
                                          'reason': {'type': 'string', 'minLength': 12}}}}},
            '$comment': 'Exactly all judgment_required response IDs, once each. Reason must explain answer-specific '
                        'evidence, reasoning or omissions; bare verdict labels are invalid. Critical errors cannot pass.'}


def _judgments(judgments, packet, mapping):
    require(isinstance(judgments, dict) and set(judgments) == {'schema', 'packet_sha256', 'cases'}
            and judgments['schema'] == JUDGMENT_SCHEMA, 'Unsupported semantic judgment schema')
    require(judgments['packet_sha256'] == mapping['packet_sha256'], 'Judgment packet hash mismatch')
    expected = {answer['response_id'] for case in packet['cases'] for answer in case['responses']
                if answer['judgment_required']}
    require(isinstance(judgments['cases'], list), 'Missing semantic judgments')
    result = {}
    for item in judgments['cases']:
        require(isinstance(item, dict) and set(item) == {'response_id', 'semantic_pass', 'critical_error', 'reason'},
                'Malformed semantic judgment')
        identifier = item['response_id']
        require(isinstance(identifier, str) and identifier in expected, 'Unknown/ineligible semantic response ID')
        require(identifier not in result, 'Duplicate semantic response ID')
        require(type(item['semantic_pass']) is bool and type(item['critical_error']) is bool,
                'Semantic judgments require explicit booleans')
        require(not (item['semantic_pass'] and item['critical_error']), 'Critical error cannot pass')
        reason = item['reason']
        require(isinstance(reason, str) and len(reason.strip()) >= 12
                and len(set(reason.strip())) >= 6
                and reason.strip().lower() not in ('semantic pass', 'semantic fail', 'looks correct', 'all correct.'),
                'Meaningful answer-specific semantic reason required')
        result[identifier] = item
    require(set(result) == expected, 'Incomplete semantic judgments')
    return result


def _case_result(record, reference):
    result = {'observed': record is not None, 'missing': record is None,
              'completed': False, 'length': False, 'exception': False, 'truncated': False,
              'format_valid': False, 'citation_valid': False, 'judgment_eligible': _eligible(record),
              'structural_errors': ['missing_response'], 'wall_seconds': None}
    if record is None:
        return result
    checks = validate_answer(record['response'], reference)
    result.update({key: checks[key] for key in ('format_valid', 'citation_valid')})
    result.update(structural_errors=checks['errors'], length=record['finish_reason'] == 'length',
                  exception=record['error'] is not None or record['finish_reason'] == 'exception',
                  truncated=record['context_truncated'] or record['truncated_output'], wall_seconds=record['wall_seconds'])
    result['completed'] = (record['error'] is None and bool(record['response'].strip())
                           and record['finish_reason'] in ('stop', 'eos')
                           and record['input_tokens'] > 0 and record['output_tokens'] > 0
                           and not result['truncated'])
    return result


def _aggregate(items, semantic):
    result = {'expected': len(items)}
    for key in ('observed', 'missing', 'completed', 'length', 'exception', 'truncated',
                'format_valid', 'citation_valid', 'judgment_eligible'):
        result[key] = sum(bool(item[key]) for item in items)
    result['completion_failures'] = len(items) - result['completed']
    result['judgment_excluded'] = len(items) - result['judgment_eligible']
    result['distinct_families'] = len({family for item in items for family in item['family_ids']})
    durations = [item['wall_seconds'] for item in items if item['wall_seconds'] is not None]
    result['latency'] = {'observed_rows': len(durations), 'wall_seconds_sum': sum(durations),
                         'wall_seconds_mean': sum(durations) / len(durations) if durations else None,
                         'ttft_observed': False}
    if semantic:
        result.update(semantic_judged=sum(item['semantic_pass'] is not None for item in items),
                      semantic_pass=sum(item['semantic_pass'] is True for item in items),
                      critical_errors=sum(item['critical_error'] is True for item in items),
                      accepted=sum(item['accepted'] for item in items))
        result['accepted_rate_fixed_denominator'] = result['accepted'] / len(items) if items else None
    return result


def _paired(items, arm_names):
    a, b = arm_names
    left = {item['id']: item for item in items if item['arm'] == a}
    right = {item['id']: item for item in items if item['arm'] == b}
    gains = [key for key in left if not left[key]['accepted'] and right[key]['accepted']]
    regressions = [key for key in left if left[key]['accepted'] and not right[key]['accepted']]
    return {'expected_pairs': len(left), 'both_semantically_judged': sum(
                left[key]['semantic_pass'] is not None and right[key]['semantic_pass'] is not None for key in left),
            'gains': len(gains), 'regressions': len(regressions), 'gain_ids': gains, 'regression_ids': regressions}


def score_packet(packet, mapping, judgments=None):
    require(isinstance(mapping, dict) and mapping.get('schema') == MAPPING_SCHEMA
            and mapping.get('protocol') == PROTOCOL, 'Unsupported private blind mapping')
    require(object_sha(packet) == mapping.get('packet_sha256'), 'Blind packet hash mismatch')
    manifest_path = pinned_path('.', mapping['manifest'])
    arm_paths = {name: pinned_path('.', pin).parent for name, pin in mapping['artifacts'].items()}
    rebuilt_packet, rebuilt_mapping = build_blind_packet(manifest_path, arm_paths, seed=mapping['seed'],
                                  manifest_sha256=mapping['manifest']['sha256'],
                                  arm_sha256={name: pin['sha256'] for name, pin in mapping['artifacts'].items()})
    require(rebuilt_packet == packet and rebuilt_mapping == mapping, 'Blind mapping/artifact binding mismatch')
    manifest, rows, refs, _ = load_suite(manifest_path, mapping['manifest']['sha256'])
    semantic = judgments is not None
    verdicts = _judgments(judgments, packet, mapping) if semantic else {}
    lookup = {(identity['arm'], identity['case_id']): rid for rid, identity in mapping['response_map'].items()}
    results, arms = [], {}
    for name, path in arm_paths.items():
        records, runtime, completion, _ = load_arm(path, rows, manifest_path, manifest)
        selected = []
        for row in rows:
            reference = refs[row['id']]
            result = {'id': row['id'], 'arm': name, 'response_id': lookup.get((name, row['id'])),
                      **{key: reference[key] for key in ('task', 'domain', 'institution', 'family_ids')},
                      'task_group': 'general' if reference['task'] in GENERAL_TASKS else 'work',
                      **_case_result(records.get(row['id']), reference)}
            if semantic:
                verdict = verdicts.get(result['response_id'])
                result.update(semantic_pass=None if verdict is None else verdict['semantic_pass'],
                              critical_error=None if verdict is None else verdict['critical_error'],
                              semantic_reason=None if verdict is None else verdict['reason'])
                result['accepted'] = bool(verdict and verdict['semantic_pass'] and not verdict['critical_error']
                                          and result['completed'] and result['format_valid'] and result['citation_valid'])
            selected.append(result)
        summary = _aggregate(selected, semantic)
        categories = {'task': TASKS, 'task_group': ('general', 'work'),
                      'domain': sorted({item['domain'] for item in selected}),
                      'institution': sorted({item['institution'] for item in selected})}
        summary['by'] = {key: {value: _aggregate([item for item in selected if item[key] == value], semantic)
                              for value in values} for key, values in categories.items()}
        summary['candidate_source'] = {key: runtime.get(key) for key in
                                       ('model_identity', 'model_manifest', 'tokenizer_identity', 'adapter')}
        summary['termination'] = {key: completion[key] for key in
                                   ('complete', 'worker_exit_code', 'guard_failure', 'missing_ids')}
        summary['termination']['journal_error'] = completion.get('journal_error')
        arms[name] = summary
        results.extend(selected)
    report = {'schema': 'trid-public-work-assessment-v1', 'protocol': PROTOCOL,
              'research_only': True, 'quality_accepted': False, 'historically_unseen': False,
              'expert_reviewed': False, 'scope': 'observed_diagnostic_not_fresh_gold',
              'semantic_evaluated': semantic, 'semantic_judgment_provenance': 'external_unverified' if semantic else None,
              'manifest': mapping['manifest'], 'packet_sha256': mapping['packet_sha256'],
              'generation_authentication': mapping['generation_authentication'],
              'denominators': {'unique_cases': len(rows), 'expected_arm_cases': len(rows) * 2,
                               'observed_response_rows': len(mapping['response_map'])},
              'arms': arms, 'cases': results,
              'limitations': ['Observed development diagnostics, not fresh holdout, expert gold, or deployment acceptance.',
                              'Native receipt pinning is not independent execution attestation.',
                              'Format, citation endpoints and completion do not establish semantic correctness.',
                              'Meaningful judgment reasons are required, but their truth and reviewer independence are not certified.',
                              'Missing/unexecuted answers are not judged and remain fixed-denominator failures.']}
    if semantic:
        names = list(arm_paths)
        report['paired'] = {'baseline': names[0], 'candidate': names[1], 'overall': _paired(results, names),
                            'by_task': {task: _paired([item for item in results if item['task'] == task], names)
                                        for task in TASKS},
                            'by_task_group': {group: _paired([item for item in results if item['task_group'] == group], names)
                                              for group in ('general', 'work')}}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    blind = commands.add_parser('blind-packet')
    blind.add_argument('--manifest', required=True)
    blind.add_argument('--manifest-sha256', required=True)
    for arm in ('a', 'b'):
        blind.add_argument(f'--arm-{arm}', required=True, help='Native output directory')
        blind.add_argument(f'--arm-{arm}-sha256', required=True, help='completion.json SHA256')
        blind.add_argument(f'--arm-{arm}-name', default=arm)
    blind.add_argument('--seed')
    blind.add_argument('--output', required=True, help='Exclusive new private packet directory')
    score = commands.add_parser('score')
    score.add_argument('--packet', required=True)
    score.add_argument('--mapping', required=True)
    score.add_argument('--judgments')
    score.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'blind-packet':
        require(args.arm_a_name != args.arm_b_name, 'Arm names must differ')
        require(not Path(args.output).exists(), 'Blind output directory must be new')
        packet, mapping = build_blind_packet(args.manifest, {args.arm_a_name: args.arm_a, args.arm_b_name: args.arm_b},
                            seed=args.seed, manifest_sha256=args.manifest_sha256,
                            arm_sha256={args.arm_a_name: args.arm_a_sha256, args.arm_b_name: args.arm_b_sha256})
        output = Path(args.output)
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        write_new_json(output / 'packet.private.json', packet, private=True)
        write_new_json(output / 'mapping.private.json', mapping, private=True)
        eligible_ids = [answer['response_id'] for case in packet['cases'] for answer in case['responses']
                        if answer['judgment_required']]
        write_new_json(output / 'judgments.schema.json', judgment_schema(mapping['packet_sha256'], eligible_ids))
    else:
        report = score_packet(load_json(args.packet), load_json(args.mapping),
                              None if args.judgments is None else load_json(args.judgments))
        write_new_json(args.output, report, private=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
