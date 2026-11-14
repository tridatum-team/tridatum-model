#!/usr/bin/env python3
"""Offline expanded diagnostic assessment; never selection or deployment qualification.

Commands (all paths belong to --root):
  packet --frozen FILE --frozen-sha256 SHA --primary DIR [--secondary DIR] --out DIR
  shard --packet FILE --packet-sha256 SHA --size 16 --out DIR
  collect --index FILE --index-sha256 SHA --panels PANEL_A PANEL_B --out DIR
  collect --index FILE --index-sha256 SHA --panels PANEL_C --third --out DIR
  adjudication --packet FILE --reports FIRST SECOND --out DIR
  freeze --packet FILE --reports FIRST SECOND [--third FILE --adjudication FILE] --out FILE
  aggregate --judgments FILE --judgments-sha256 SHA --mapping FILE --out FILE

Review JSON: schema trid-expanded-blind-review-v1, packet_sha256, reviewer_id,
independent:true, blinded:true, judgments:{sample_id:{semantic_pass:bool,
format_pass:bool, critical_errors:[specific descriptions], reason:nonempty}}.
Third reports additionally bind adjudication_packet_sha256 and contain ONLY the
flagged IDs. Give reviewers only review-packet.json (or adjudication-packet.json),
never private-mapping.json, runtime files, or other reviewers' votes. Reference
paraphrases are acceptable; JSON validity is not semantic evidence. Freeze writes
resolved anonymous judgments before aggregate opens the arm mapping.

For parallel review, give each reviewer only its shard-NNNN.json. Leaf reports use
the child file SHA and their actual reviewer_id (no adjudication SHA even for third
shards). Panel manifests contain schema trid-expanded-review-panel-v1, panel_id,
index_sha256, reports:[{path,bytes,sha256}]. Paths in pins are repository-relative.
Collect emits panel-1.json/panel-2.json, explicitly marked reviewer_kind:panel with
per-sample actual reviewer IDs and child/report provenance pins. These are accepted
by adjudication/freeze only after recursively authenticating the composition.
For third-review shards, shard adjudication-packet.json and collect one --third
panel; third-panel.json binds both the original and adjudication packet hashes.
Distinct actual reviewer IDs are required per sample across both primary panels
and any third panel. Identity/blinding declarations are auditable provenance, not
external proof of personal identity or independence.

Uncertainty is described with source-family paired summaries, not question-level
confidence intervals. Synthetic general family IDs remain intact. No population
accuracy claim, statistical independence claim, or automatic adoption threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import secrets
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.aws_eval.eight_hf_assessment import (
    bound, digest, object_sha, pin, require, signature, structural_result, validate_report,
)

ARMS = ('base', 'step-000024')
PIN_NAMES = {'design', 'source_allocation', 'evaluation_inputs', 'evaluation_metadata',
             'evaluation_references', 'training_manifest', 'semantic_review_resolution'}
METRICS = ('structural_pass', 'semantic_pass', 'format_pass', 'joint_pass')


def parse_json(text):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def load(path):
    return parse_json(Path(path).read_text(encoding='utf-8'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def indexed(rows, label):
    require(isinstance(rows, list) and all(isinstance(r, dict) and isinstance(r.get('id'), str)
            and r['id'] for r in rows), label + ' rows')
    result = {r['id']: r for r in rows}
    require(len(result) == len(rows), label + ' duplicate IDs')
    return result


def finite_seconds(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def authenticate(path, expected):
    path = Path(path)
    require(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'],
            'artifact hash/size mismatch: ' + path.name)
    return path


def frozen_dataset(root, path, expected_sha):
    require(digest(path) == expected_sha, 'frozen dataset hash mismatch')
    frozen = load(path)
    require(frozen.get('schema') == 'trid-expanded-frozen-dataset-v1'
            and set(frozen.get('pins', {})) == PIN_NAMES, 'frozen dataset pins/schema')
    paths = {name: bound(root, entry) for name, entry in frozen['pins'].items()}
    design = load(paths['design'])
    require(design['primary']['evaluation_questions'] == 200
            and design['primary']['planned_answers'] == 400
            and design['primary']['arms'] == ['base', 'step24']
            and design['primary']['decoder_profile'] == 'unconstrained'
            and design['format_secondary']['questions'] == 40
            and design['format_secondary']['planned_answers'] == 80
            and design['format_secondary']['decoder_profile'] == 'json-schema'
            and design['format_secondary']['scores_combined_with_primary'] is False,
            'registered expanded design differs')
    inputs = indexed(load(paths['evaluation_inputs']), 'inputs')
    metadata = indexed(load(paths['evaluation_metadata']), 'metadata')
    references = indexed(load(paths['evaluation_references']), 'references')
    require(len(inputs) == 200 and set(inputs) == set(metadata) == set(references),
            'full 200-case frozen denominator required')
    require(all(set(row) == {'id', 'messages'} for row in inputs.values()), 'public input fields')
    require(Counter(m['cohort'] for m in metadata.values()) == {'mois': 160, 'general': 40},
            'registered cohort denominator differs')
    families = defaultdict(list)
    for cid, meta in metadata.items():
        require(isinstance(meta.get('family_id'), str) and meta['family_id']
                and isinstance(meta.get('behavior'), str) and meta['behavior'], 'family/behavior missing')
        require(meta.get('case_sha256') == references[cid].get('case_sha256')
                and isinstance(meta.get('case_sha256'), str) and len(meta['case_sha256']) == 64,
                'reference case hash binding differs')
        require(isinstance(references[cid].get('reference_answer'), str)
                and isinstance(references[cid].get('rubric'), dict), 'fixed reference/rubric missing')
        if meta['cohort'] == 'mois':
            families[meta['family_id']].append(meta)
    require(len(families) == 20 and all(len(v) == 8 and len({m['behavior'] for m in v}) == 8
                                     for v in families.values()), '20 families by 8 behaviors required')
    secondary = frozen.get('secondary_case_ids')
    expected_secondary = {cid for cid, m in metadata.items() if m['cohort'] == 'mois'
                          and m['behavior'] in ('multi_item', 'multi_clause_synthesis')}
    require(isinstance(secondary, list) and len(secondary) == len(set(secondary)) == 40
            and set(secondary) == expected_secondary
            and all(m.get('secondary_selected') is (cid in expected_secondary)
                    for cid, m in metadata.items()), 'fixed secondary selection mismatch')
    return frozen, design, inputs, metadata, references


def completed_run(root, directory, cases, design, profile, input_pin=None):
    """Authenticate the evaluator's actual file/row contract, never model weights."""
    directory = Path(directory)
    require(not (directory / 'failure.json').exists(), 'failed local run is not complete')
    names = ['run-manifest.json', 'decoder-profile.json', 'prompt-preflight.json',
             'loaded-runtime.json', 'completion.json', 'base-outputs.jsonl', 'step-000024-outputs.jsonl']
    pins = {name: pin(directory / name, root) for name in names}
    manifest = load(directory / 'run-manifest.json')
    completion = load(directory / 'completion.json')
    decoder = load(directory / 'decoder-profile.json')
    preflight = load(directory / 'prompt-preflight.json')
    loaded = load(directory / 'loaded-runtime.json')
    manifest_sha = pins['run-manifest.json']['sha256']
    require(manifest.get('schema') == 'trid-local-eight-paired-v1' and manifest.get('mode') == 'evaluation'
            and manifest.get('arm_order') == list(ARMS)
            and manifest.get('grading_access') is False and manifest.get('retries') is False
            and manifest.get('resume') is False, 'local run identity/mode differs')
    ids = manifest.get('selected_ids', [])
    require(len(ids) == len(set(ids)) == len(cases) and set(ids) == set(cases), 'selected case denominator')
    require(completion.get('status') == 'complete' and completion.get('mode') == 'evaluation'
            and completion.get('rows_per_arm') == len(cases)
            and completion.get('run_manifest_sha256') == manifest_sha
            and set(completion.get('outputs', {})) == set(ARMS)
            and finite_seconds(completion.get('elapsed_seconds')), 'completion identity/denominator')
    public_path = Path(manifest['input']['path'])
    public_path = public_path if public_path.is_absolute() else Path(root) / public_path
    authenticate(public_path, manifest['input'])
    require(indexed(load(public_path), 'run inputs') == cases, 'run input messages differ')
    if input_pin is not None:
        require(all(manifest['input'][k] == input_pin[k] for k in ('sha256', 'bytes')),
                'primary frozen input pin differs')
    generation = manifest['generation']
    expected_generation = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
        max_new_tokens=768, context_tokens=4096, dtype='bfloat16', backend='transformers-mps')
    require(generation == expected_generation
            and all(generation[k] == design['generation'][k] for k in
                    ('enable_thinking', 'do_sample', 'max_new_tokens', 'context_tokens')),
            'registered generation differs')
    model = manifest['model']
    adapter = manifest['adapter']
    require(model.get('id') == design['model']['repo_id']
            and model.get('revision') == design['model']['revision']
            and model.get('body_hashes_authenticated') is True
            and model.get('representation') == 'hf-safetensors-bfloat16'
            and adapter.get('step') == design['model']['adapter_step'] == 24
            and adapter['files']['adapter_model.safetensors']['sha256'] == design['model']['adapter_sha256'],
            'model/adapter binding differs')
    runtime = manifest['runtime']
    require(runtime.get('device') == 'mps' and runtime.get('dtype') == 'torch.bfloat16'
            and runtime.get('cpu_fallback') is False and runtime.get('cuda_equivalence_claim') is False
            and loaded.get('parameter_dtypes') == ['torch.bfloat16'], 'runtime/device/dtype differs')
    require(manifest.get('decoder_profile') == decoder and decoder.get('profile') == profile
            and decoder.get('semantic_truth_guarantee') is False, 'decoder profile binding differs')
    if profile == 'unconstrained':
        require(set(decoder) == {'profile', 'semantic_truth_guarantee'}, 'unexpected primary decoder settings')
    else:
        from scripts.evaluate_local_eight import mois_schema
        schemas = {cid: mois_schema(case) for cid, case in cases.items()}
        require(decoder.get('schemas') == schemas
                and decoder.get('schema_sha256') == {cid: object_sha(schema) for cid, schema in schemas.items()}
                and decoder.get('package') == 'lm-format-enforcer' and decoder.get('version') == '0.11.3'
                and decoder.get('integration') == 'build_transformers_prefix_allowed_tokens_fn'
                and decoder.get('eos_violation_policy') == 'fatal; never substitute EOS, repair, or retry'
                and decoder.get('production_structural_validator') == 'unchanged',
                'fixed secondary grammar identity/schema differs')
    require(preflight.get('status') == 'passed' and preflight.get('truncation') is False
            and preflight.get('all_selected_preflighted_before_weights') is True, 'prompt preflight missing')
    prompts = indexed(preflight['prompts'], 'preflight')
    require(set(prompts) == set(cases), 'prompt preflight denominator')
    config = loaded['generation_config']
    require(config.get('do_sample') is False and config.get('num_beams') == 1
            and config.get('max_new_tokens') == 768 and config.get('return_dict_in_generate') is True,
            'loaded generation configuration differs')
    policy = loaded.get('generation_policy', {})
    require(policy.get('dispatch_kwargs') == {'use_model_defaults': False, 'do_sample': False, 'num_beams': 1}
            and policy.get('generation_mode') == 'greedy_search'
            and policy.get('effective_config_sha256') == object_sha(config),
            'effective greedy generation policy differs')
    implementation = policy.get('preparation_implementation', {})
    require(isinstance(implementation.get('path'), str) and implementation['path']
            and type(implementation.get('bytes')) is int and implementation['bytes'] > 0
            and isinstance(implementation.get('sha256'), str) and len(implementation['sha256']) == 64
            and all(c in '0123456789abcdef' for c in implementation['sha256']),
            'generation preparation implementation pin missing')
    requested = loaded.get('requested_generation_config', {})
    require(all(requested.get(key) == config[key] for key in
                ('do_sample', 'num_beams', 'max_new_tokens', 'return_dict_in_generate')),
            'requested/effective generation contract differs')
    answers = {}
    for arm in ARMS:
        path = directory / (arm + '-outputs.jsonl')
        authenticate(path, completion['outputs'][arm])
        require(Path(completion['outputs'][arm]['path']).resolve() == path.resolve(),
                'completion output path differs')
        # Parse each line strictly, including duplicate keys and nonfinite values.
        with path.open(encoding='utf-8') as stream:
            lines = [line for line in stream if line.strip()]
        rows = [parse_json(line) for line in lines]
        by_id = indexed(rows, arm)
        require(set(by_id) == set(cases), 'missing/extra answer denominator')
        for cid, row in by_id.items():
            case, prompt = cases[cid], prompts[cid]
            require(row.get('arm') == arm and row.get('run_manifest_sha256') == manifest_sha
                    and row.get('loaded_runtime_sha256') == pins['loaded-runtime.json']['sha256']
                    and row.get('model_id') == model['id'] and row.get('model_revision') == model['revision']
                    and row.get('input_sha256') == manifest['input']['sha256']
                    and row.get('nominated_adapter_weights_sha256') == design['model']['adapter_sha256']
                    and row.get('adapter_weights_sha256') == (None if arm == 'base' else design['model']['adapter_sha256'])
                    and row.get('mode') == 'evaluation' and row.get('decoder_profile') == profile
                    and row.get('decoder_profile_sha256') == pins['decoder-profile.json']['sha256']
                    and row.get('generation') == generation and row.get('generation_config') == config
                    and row.get('generation_policy_sha256') == object_sha(policy)
                    and row.get('effective_generation_mode') == 'greedy_search',
                    'answer arm/runtime/profile/adapter binding differs')
            require(row.get('messages') == prompt.get('messages') == case['messages']
                    and row.get('prompt_sha256') == prompt.get('prompt_sha256') == object_sha(case['messages'])
                    and row.get('prompt_token_ids') == prompt.get('prompt_token_ids')
                    and row.get('rendered_prompt_sha256') == prompt.get('rendered_prompt_sha256'),
                    'answer case/prompt binding differs')
            require(isinstance(row.get('response_text'), str) and row['response_text'].strip()
                    and not row.get('exception') and finite_seconds(row.get('elapsed_seconds')),
                    'missing answer or invalid latency')
            answers[(cid, arm)] = row
    return answers, pins, manifest, loaded


def mapping_body(mapping):
    return {k: v for k, v in mapping.items() if k not in ('schema', 'packet_sha256')}


def make_packet(root, frozen_path, frozen_sha, primary, out, secondary=None):
    frozen, design, cases, metadata, references = frozen_dataset(root, frozen_path, frozen_sha)
    out = Path(out)
    require(not out.exists(), 'packet directory already exists; never rerandomize')
    runs = [('primary', primary, cases, 'unconstrained')]
    if secondary is not None:
        runs.append(('secondary', secondary, {cid: cases[cid] for cid in frozen['secondary_case_ids']}, 'json-schema'))
    mapping = dict(schema='trid-expanded-blind-mapping-v1', frozen_dataset=pin(frozen_path, root),
                   rows={}, runs={}, arms=list(ARMS), primary_case_ids=list(cases),
                   secondary_case_ids=frozen['secondary_case_ids'] if secondary is not None else [])
    packet_rows = []
    for suite, directory, subset, profile in runs:
        answers, pins, manifest, loaded = completed_run(root, directory, subset, design, profile,
            frozen['pins']['evaluation_inputs'] if suite == 'primary' else None)
        mapping['runs'][suite] = {'directory': str(Path(directory).resolve()), 'pins': pins,
                                  'runtime': manifest['runtime'], 'generation': manifest['generation'],
                                  'generation_config': loaded['generation_config'],
                                  'generation_policy': loaded['generation_policy']}
        if suite == 'secondary':
            primary_run = mapping['runs']['primary']
            require(manifest['runtime'] == primary_run['runtime']
                    and manifest['generation'] == primary_run['generation']
                    and loaded['generation_config'] == primary_run['generation_config']
                    and loaded['generation_policy'] == primary_run['generation_policy'],
                    'secondary runtime not paired with primary')
        for (cid, arm), answer in answers.items():
            sid = secrets.token_hex(24)
            reference = references[cid]
            packet_rows.append(dict(sample_id=sid, messages=cases[cid]['messages'],
                response=answer['response_text'], reference_answer=reference['reference_answer'],
                rubric=reference['rubric'], structural=structural_result(cases[cid], answer, manifest['generation'])))
            mapping['rows'][sid] = dict(case_id=cid, case_sha256=reference['case_sha256'],
                cohort=metadata[cid]['cohort'], behavior=metadata[cid]['behavior'],
                family_id=metadata[cid]['family_id'], suite=suite, arm=arm,
                elapsed_seconds=answer['elapsed_seconds'])
    secrets.SystemRandom().shuffle(packet_rows)
    packet = dict(schema='trid-expanded-blind-packet-v1', rows=packet_rows,
        mapping_commitment=object_sha(mapping_body(mapping)),
        instructions='Judge independently and blinded. Assess every material rubric requirement and unsupported claim; '
        'accept semantic paraphrases, never exact-reference match. Score requested format separately; structural flags '
        'are not entailment proof. Record case-specific material critical errors, never formatting-only critical errors. '
        'Do not infer or request model, arm, decoder profile, timing, or other reviewers\' votes.')
    save_new(out / 'review-packet.json', packet)
    mapping['packet_sha256'] = digest(out / 'review-packet.json')
    save_new(out / 'private-mapping.json', mapping)
    return {'packet': str(out / 'review-packet.json'), 'samples': len(packet_rows)}


def validate_blind_packet(packet, *, partial=False):
    require(packet.get('schema') == 'trid-expanded-blind-packet-v1', 'packet schema differs')
    fields = {'schema', 'rows', 'mapping_commitment', 'instructions'}
    if partial:
        fields.add('parent_packet_sha256')
    require(set(packet) == fields
            and all(set(row) == {'sample_id', 'messages', 'response', 'reference_answer', 'rubric', 'structural'}
                    and set(row['structural']) == {'pass', 'reason'}
                    and type(row['structural']['pass']) is bool for row in packet['rows']),
            'blind packet contains undeclared identity/vote fields')
    ids = [r['sample_id'] for r in packet['rows']]
    require(all(isinstance(sid, str) and sid for sid in ids) and len(ids) == len(set(ids))
            and (partial or len(ids) in (400, 480)), 'blind packet denominator')
    return ids


def shard_document(parent, parent_sha, rows):
    return dict(schema='trid-expanded-blind-shard-v1', parent_packet_sha256=parent_sha,
        rows=rows, instructions='Independently review only these samples. Report against this child packet file SHA, '
        'not its parent hash. Use trid-expanded-blind-review-v1 with your own actual reviewer_id, independent:true, '
        'blinded:true and every sample judgment. Judge material meaning and requested format separately; accept '
        'semantic paraphrases. Structural flags are not entailment evidence. Never request arm identities, '
        "timing, decoder profile, or other reviewers' votes.")


def shard(root, packet_path, packet_sha, size, out):
    require(digest(packet_path) == packet_sha, 'shard parent packet hash differs')
    parent = load(packet_path)
    ids = validate_blind_packet(parent, partial='parent_packet_sha256' in parent)
    require(ids and type(size) is int and size > 0, 'nonempty packet and positive shard size required')
    out = Path(out)
    require(not out.exists(), 'shard directory already exists')
    index = dict(schema='trid-expanded-review-shards-v1', parent=pin(packet_path, root), size=size, children=[])
    for start in range(0, len(ids), size):
        subset = parent['rows'][start:start + size]
        path = out / f'shard-{start // size + 1:04d}.json'
        save_new(path, shard_document(parent, packet_sha, subset))
        index['children'].append({'packet': pin(path, root), 'sample_ids': [r['sample_id'] for r in subset]})
    path = out / 'shard-index.json'
    save_new(path, index)
    return {'index': str(path), 'sha256': digest(path), 'samples': len(ids), 'shards': len(index['children'])}


def shard_index(root, path, expected_sha):
    require(digest(path) == expected_sha, 'shard index hash differs')
    index = load(path)
    require(set(index) == {'schema', 'parent', 'size', 'children'}
            and index['schema'] == 'trid-expanded-review-shards-v1'
            and type(index['size']) is int and index['size'] > 0, 'shard index schema/size differs')
    parent_path = bound(root, index['parent'])
    parent = load(parent_path)
    ids = validate_blind_packet(parent, partial='parent_packet_sha256' in parent)
    require(ids and len(index['children']) == math.ceil(len(ids) / index['size']), 'shard coverage denominator')
    children = {}
    for number, entry in enumerate(index['children']):
        require(set(entry) == {'packet', 'sample_ids'}, 'shard index child fields')
        child_path = bound(root, entry['packet'])
        child = load(child_path)
        subset = parent['rows'][number * index['size']:(number + 1) * index['size']]
        require(entry['sample_ids'] == [r['sample_id'] for r in subset]
                and child == shard_document(parent, index['parent']['sha256'], subset),
                'shard content/coverage differs')
        child_sha = entry['packet']['sha256']
        require(child_sha not in children, 'duplicate shard packet')
        children[child_sha] = (entry, child)
    return index, parent, children


def compose_panel(root, index_path, index_sha, panel_path):
    index, parent, children = shard_index(root, index_path, index_sha)
    panel = load(panel_path)
    require(set(panel) == {'schema', 'panel_id', 'index_sha256', 'reports'}
            and panel['schema'] == 'trid-expanded-review-panel-v1'
            and isinstance(panel['panel_id'], str) and panel['panel_id'].strip() == panel['panel_id']
            and panel['panel_id'] and panel['index_sha256'] == index_sha,
            'panel identity/index differs')
    require(isinstance(panel['reports'], list) and len(panel['reports']) == len(children), 'panel report denominator')
    judgments, provenance, seen = {}, {}, set()
    for report_pin in panel['reports']:
        leaf = load(bound(root, report_pin))
        child_sha = leaf.get('packet_sha256')
        require(child_sha in children and child_sha not in seen, 'missing/duplicate/extra child review')
        seen.add(child_sha)
        entry, child = children[child_sha]
        # Leaf reports must name actual reviewers, never recursively composed panels.
        require('composition' not in leaf and 'reviewer_kind' not in leaf and 'panel_id' not in leaf,
                'leaf reviewer must be an actual individual')
        votes = review_document(leaf, child, child_sha, root=root)
        for sid, judgment in votes.items():
            require(sid not in judgments, 'duplicate panel sample')
            judgments[sid] = judgment
            provenance[sid] = dict(reviewer_id=leaf['reviewer_id'], report=report_pin, child_packet=entry['packet'])
    require(seen == set(children) and set(judgments) == {r['sample_id'] for r in parent['rows']},
            'panel sample coverage differs')
    result = dict(schema='trid-expanded-blind-review-v1',
        packet_sha256=parent.get('parent_packet_sha256', index['parent']['sha256']),
        reviewer_id='panel:' + panel['panel_id'], reviewer_kind='panel', panel_id=panel['panel_id'],
        independent=True, blinded=True, judgments=judgments,
        composition=dict(index=pin(index_path, root), panel=pin(panel_path, root), samples=provenance))
    if 'parent_packet_sha256' in parent:
        result['adjudication_packet_sha256'] = index['parent']['sha256']
    return result


def actual_reviewer(report, sid):
    if report.get('reviewer_kind') == 'panel':
        return report['composition']['samples'][sid]['reviewer_id']
    return report['reviewer_id']


def independent_samples(reports, ids):
    for sid in ids:
        require(len({actual_reviewer(report, sid) for report in reports}) == len(reports),
                'actual reviewers must differ for every sample')


def collect(root, index_path, index_sha, panel_paths, out, third=False):
    require(len(panel_paths) == (1 if third else 2), 'two primary panels or one third panel required')
    _, parent, _ = shard_index(root, index_path, index_sha)
    require(('parent_packet_sha256' in parent) is third, 'primary/adjudication panel purpose differs')
    reports = [compose_panel(root, index_path, index_sha, path) for path in panel_paths]
    require(len({r['panel_id'] for r in reports}) == len(reports), 'panel identities must differ')
    independent_samples(reports, [r['sample_id'] for r in parent['rows']])
    out = Path(out)
    require(not out.exists(), 'collected report directory already exists')
    paths = []
    for number, report in enumerate(reports, 1):
        path = out / ('third-panel.json' if third else f'panel-{number}.json')
        save_new(path, report)
        paths.append(pin(path, root))
    return {'reports': paths, 'samples_per_panel': len(parent['rows']), 'actual_reviewer_provenance_preserved': True}


def review_pair(packet_path, reports, root=ROOT):
    require(len(reports) == 2, 'exactly two independent reports required')
    packet = load(packet_path)
    ids = validate_blind_packet(packet)
    documents = [load(p) for p in reports]
    judgments = [review_document(r, packet, digest(packet_path), root=root) for r in documents]
    require(documents[0]['reviewer_id'] != documents[1]['reviewer_id'], 'reviewers must differ')
    independent_samples(documents, ids)
    disagreements = sorted(sid for sid in ids if signature(judgments[0][sid], False) != signature(judgments[1][sid], False))
    return packet, documents, judgments, disagreements


def review_document(report, packet, packet_sha, ids=None, *, root=ROOT):
    require(report.get('schema') == 'trid-expanded-blind-review-v1', 'expanded review schema differs')
    fields = {'schema', 'packet_sha256', 'reviewer_id', 'independent', 'blinded', 'judgments'}
    if ids is not None:
        fields.add('adjudication_packet_sha256')
    if 'composition' in report:
        fields.update(('reviewer_kind', 'panel_id', 'composition'))
        composition = report['composition']
        require(isinstance(composition, dict) and set(composition) == {'index', 'panel', 'samples'},
                'panel composition fields differ')
        index_path = bound(root, composition['index'])
        panel_path = bound(root, composition['panel'])
        expected = compose_panel(root, index_path, composition['index']['sha256'], panel_path)
        require(report == expected, 'composed review/provenance differs')
    require(set(report) == fields, 'review contains undeclared fields')
    require(isinstance(report.get('reviewer_id'), str)
            and report['reviewer_id'] == report['reviewer_id'].strip(), 'reviewer identity whitespace')
    require(not report['reviewer_id'].startswith('panel:') or 'composition' in report,
            'panel identity requires authenticated composition')
    # Reuse the established content-hash-bound complete judgment validator, not old gates/counts.
    return validate_report(dict(report, schema='trid-eight-blind-review-v1'), packet, packet_sha, ids)


def adjudication_document(packet, packet_sha, disagreements):
    document = dict(packet, parent_packet_sha256=packet_sha,
                    rows=[row for row in packet['rows'] if row['sample_id'] in disagreements])
    document['instructions'] += ' Report only these sample IDs against parent_packet_sha256, and include adjudication_packet_sha256.'
    return document


def adjudication(root, packet_path, reports, out):
    packet, _, _, disagreements = review_pair(packet_path, reports, root)
    document = adjudication_document(packet, digest(packet_path), disagreements)
    path = Path(out) / 'adjudication-packet.json'
    save_new(path, document)
    return {'packet': str(path), 'disagreements': len(disagreements), 'third_required': bool(disagreements)}


def freeze(root, packet_path, reports, out, third=None, adjudication_path=None):
    packet, documents, judgments, disagreements = review_pair(packet_path, reports, root)
    third_judgments = {}
    report_pins = [pin(p, root) for p in reports]
    extra = None
    if disagreements:
        require(third is not None and adjudication_path is not None, 'disagreements require blinded third and adjudication packet')
        adjudicated = load(adjudication_path)
        require(adjudicated == adjudication_document(packet, digest(packet_path), disagreements),
                'adjudication content/denominator differs')
        report = load(third)
        require(report.get('reviewer_id') not in {r['reviewer_id'] for r in documents}
                and report.get('adjudication_packet_sha256') == digest(adjudication_path), 'third independence/hash differs')
        third_judgments = review_document(report, packet, digest(packet_path), disagreements, root=root)
        independent_samples([*documents, report], disagreements)
        report_pins.append(pin(third, root))
        extra = pin(adjudication_path, root)
    else:
        require(third is None and adjudication_path is None, 'third forbidden without disagreements')
    result = dict(schema='trid-expanded-frozen-judgments-v1', packet=pin(packet_path, root),
        mapping_commitment=packet['mapping_commitment'], reports=report_pins,
        adjudication=extra, disagreements=disagreements,
        judgments={row['sample_id']: third_judgments.get(row['sample_id'], judgments[0][row['sample_id']])
                   for row in packet['rows']}, identities_joined=False)
    save_new(out, result)
    return {'judgments': str(out), 'sha256': digest(out), 'samples': len(result['judgments'])}


def latency(values):
    ordered = sorted(values)
    return {'n': len(ordered), 'total_seconds': sum(ordered), 'mean_seconds': statistics.mean(ordered),
            'median_seconds': statistics.median(ordered),
            'p95_seconds': ordered[math.ceil(.95 * len(ordered)) - 1]}


def paired_summary(rows):
    by_case = defaultdict(dict)
    for row in rows:
        require(row['arm'] not in by_case[row['case_id']], 'duplicate paired arm')
        by_case[row['case_id']][row['arm']] = row
    require(all(set(pair) == set(ARMS) for pair in by_case.values()), 'incomplete paired summary')
    arms = {}
    for arm in ARMS:
        selected = [pair[arm] for pair in by_case.values()]
        arms[arm] = {metric: sum(r[metric] for r in selected) for metric in METRICS}
        arms[arm]['critical_case_count'] = sum(bool(r['critical_errors']) for r in selected)
        arms[arm]['latency'] = latency([r['elapsed_seconds'] for r in selected])
    paired = {}
    for metric in METRICS:
        gains = sum(not p[ARMS[0]][metric] and p[ARMS[1]][metric] for p in by_case.values())
        losses = sum(p[ARMS[0]][metric] and not p[ARMS[1]][metric] for p in by_case.values())
        paired[metric] = {'gains': gains, 'regressions': losses, 'net_cases': gains - losses,
                          'net_fraction': (gains - losses) / len(by_case)}
    paired['critical_errors'] = {
        'introduced_cases': sum(not p[ARMS[0]]['critical_errors'] and bool(p[ARMS[1]]['critical_errors']) for p in by_case.values()),
        'resolved_cases': sum(bool(p[ARMS[0]]['critical_errors']) and not p[ARMS[1]]['critical_errors'] for p in by_case.values())}
    paired['latency_delta_seconds'] = statistics.mean(
        p[ARMS[1]]['elapsed_seconds'] - p[ARMS[0]]['elapsed_seconds'] for p in by_case.values())
    return {'cases': len(by_case), 'answers': len(rows), 'arms': arms, 'paired': paired}


def suite_summary(rows):
    summary = paired_summary(rows)
    for key in ('cohort', 'behavior', 'family_id'):
        groups = defaultdict(list)
        for row in rows:
            # Keep cohort in family identity even if two authors chose the same label.
            label = row['cohort'] + ':' + row[key] if key == 'family_id' else row[key]
            groups[label].append(row)
        summary['by_' + key] = {label: paired_summary(group) for label, group in sorted(groups.items())}
    family_results = summary['by_family_id']
    summary['cluster_uncertainty'] = {
        'method': 'descriptive source-family paired deltas; no confidence interval or independent-question assumption',
        'clusters': len(family_results), 'synthetic_general_clusters_preserved': True,
        'joint_net_fraction_by_family': {key: val['paired']['joint_pass']['net_fraction'] for key, val in family_results.items()},
        'equal_family_weight_joint_net_fraction': statistics.mean(
            val['paired']['joint_pass']['net_fraction'] for val in family_results.values())}
    return summary


def aggregate(root, judgments_path, judgments_sha, mapping_path, out):
    require(digest(judgments_path) == judgments_sha, 'frozen judgments hash differs')
    frozen = load(judgments_path)
    require(frozen.get('schema') == 'trid-expanded-frozen-judgments-v1'
            and frozen.get('identities_joined') is False, 'anonymous judgment freeze required')
    packet_path = bound(root, frozen['packet'])
    report_paths = [bound(root, entry) for entry in frozen['reports']]
    packet, docs, votes, disagreements = review_pair(packet_path, report_paths[:2], root)
    require(frozen['disagreements'] == disagreements, 'frozen disagreements differ')
    if disagreements:
        require(len(report_paths) == 3 and frozen['adjudication'] is not None, 'frozen third report missing')
        third_path = report_paths[2]
        third = load(third_path)
        adjudicated_path = bound(root, frozen['adjudication'])
        adjudicated = load(adjudicated_path)
        require(third.get('reviewer_id') not in {r['reviewer_id'] for r in docs}
                and third.get('adjudication_packet_sha256') == digest(adjudicated_path)
                and adjudicated == adjudication_document(packet, digest(packet_path), disagreements),
                'frozen third content/independence differs')
        final_votes = review_document(third, packet, digest(packet_path), disagreements, root=root)
        independent_samples([*docs, third], disagreements)
    else:
        require(len(report_paths) == 2 and frozen['adjudication'] is None, 'unexpected frozen third')
        final_votes = {}
    resolved = {row['sample_id']: final_votes.get(row['sample_id'], votes[0][row['sample_id']]) for row in packet['rows']}
    require(frozen['judgments'] == resolved and frozen['mapping_commitment'] == packet['mapping_commitment'],
            'frozen judgment resolution differs')
    # Arm identities are opened only after frozen anonymous judgment reauthentication.
    mapping = load(mapping_path)
    require(mapping.get('schema') == 'trid-expanded-blind-mapping-v1'
            and mapping.get('packet_sha256') == digest(packet_path)
            and object_sha(mapping_body(mapping)) == frozen['mapping_commitment']
            and set(mapping['rows']) == set(resolved), 'private mapping commitment differs')
    dataset_path = bound(root, mapping['frozen_dataset'])
    dataset, design, cases, metadata, references = frozen_dataset(root, dataset_path, mapping['frozen_dataset']['sha256'])
    require(mapping['arms'] == list(ARMS) and mapping['primary_case_ids'] == list(cases)
            and set(mapping['runs']) in ({'primary'}, {'primary', 'secondary'})
            and mapping['secondary_case_ids'] == (dataset['secondary_case_ids'] if 'secondary' in mapping['runs'] else []),
            'mapping suite denominator differs')
    expected = {}
    for suite, run in mapping['runs'].items():
        for entry in run['pins'].values():
            bound(root, entry)
        subset = cases if suite == 'primary' else {cid: cases[cid] for cid in dataset['secondary_case_ids']}
        answers, current_pins, _, _ = completed_run(root, run['directory'], subset, design,
            'unconstrained' if suite == 'primary' else 'json-schema',
            dataset['pins']['evaluation_inputs'] if suite == 'primary' else None)
        require(current_pins == run['pins'], 'run artifact membership changed')
        expected.update({(suite, cid, arm): row for (cid, arm), row in answers.items()})
    joined = []
    seen = set()
    for row in packet['rows']:
        sid = row['sample_id']
        meta = mapping['rows'][sid]
        key = (meta['suite'], meta['case_id'], meta['arm'])
        require(key in expected and key not in seen, 'mapping duplicate/unknown answer')
        seen.add(key)
        answer, cid = expected[key], meta['case_id']
        require(all(meta[k] == metadata[cid][k] for k in ('cohort', 'behavior', 'family_id', 'case_sha256'))
                and meta['elapsed_seconds'] == answer['elapsed_seconds']
                and row['messages'] == cases[cid]['messages'] and row['response'] == answer['response_text']
                and row['reference_answer'] == references[cid]['reference_answer']
                and row['rubric'] == references[cid]['rubric']
                and row['structural'] == structural_result(cases[cid], answer, answer['generation']),
                'packet/reference/output mapping differs')
        judgment = resolved[sid]
        joined.append(dict(meta, **judgment, structural_pass=row['structural']['pass'],
            joint_pass=bool(row['structural']['pass'] and judgment['semantic_pass']
                            and judgment['format_pass'] and not judgment['critical_errors'])))
    require(seen == set(expected), 'full output denominator missing')
    result = dict(schema='trid-expanded-assessment-v1', frozen_judgments=pin(judgments_path, root),
        private_mapping=pin(mapping_path, root), frozen_dataset=mapping['frozen_dataset'],
        primary=suite_summary([r for r in joined if r['suite'] == 'primary']),
        selected_step=None, diagnostic_comparator_step=24, deployment_qualified=False,
        original_no_selection_unchanged=True, broad_accuracy_claim=False,
        source_caveats=dataset.get('source_caveats', design.get('source_caveats', [])),
        independent_primary_reviews=2, adjudicated_samples=len(disagreements),
        interpretation='Source-separated diagnostic only; no automatic qualification or deployment. '
                       'Structure is not semantic correctness; foundation pretraining exposure is unknown.',
        secondary=None)
    if 'secondary' in mapping['runs']:
        secondary_rows = [r for r in joined if r['suite'] == 'secondary']
        result['secondary'] = suite_summary(secondary_rows)
        result['secondary']['combined_with_primary'] = False
        primary_lookup = {(r['case_id'], r['arm']): r for r in joined if r['suite'] == 'primary'}
        benefit = {}
        for arm in ARMS:
            pairs = [(primary_lookup[(r['case_id'], arm)], r) for r in secondary_rows if r['arm'] == arm]
            benefit[arm] = {'cases': len(pairs),
                'format_gains': sum(not a['format_pass'] and b['format_pass'] for a, b in pairs),
                'format_regressions': sum(a['format_pass'] and not b['format_pass'] for a, b in pairs),
                'structural_gains': sum(not a['structural_pass'] and b['structural_pass'] for a, b in pairs),
                'structural_regressions': sum(a['structural_pass'] and not b['structural_pass'] for a, b in pairs),
                'semantic_gains': sum(not a['semantic_pass'] and b['semantic_pass'] for a, b in pairs),
                'semantic_regressions': sum(a['semantic_pass'] and not b['semantic_pass'] for a, b in pairs),
                'introduced_critical_cases': sum(not a['critical_errors'] and bool(b['critical_errors']) for a, b in pairs),
                'latency_delta_seconds': statistics.mean(b['elapsed_seconds'] - a['elapsed_seconds'] for a, b in pairs)}
        result['secondary']['grammar_vs_unconstrained_same_cases'] = benefit
    save_new(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    packet = commands.add_parser('packet')
    packet.add_argument('--frozen', type=Path, required=True)
    packet.add_argument('--frozen-sha256', required=True)
    packet.add_argument('--primary', type=Path, required=True)
    packet.add_argument('--secondary', type=Path)
    packet.add_argument('--out', type=Path, required=True)
    command = commands.add_parser('shard')
    command.add_argument('--packet', type=Path, required=True)
    command.add_argument('--packet-sha256', required=True)
    command.add_argument('--size', type=int, default=16)
    command.add_argument('--out', type=Path, required=True)
    command = commands.add_parser('collect')
    command.add_argument('--index', type=Path, required=True)
    command.add_argument('--index-sha256', required=True)
    command.add_argument('--panels', nargs='+', type=Path, required=True)
    command.add_argument('--third', action='store_true')
    command.add_argument('--out', type=Path, required=True)
    for name in ('adjudication', 'freeze'):
        command = commands.add_parser(name)
        command.add_argument('--packet', type=Path, required=True)
        command.add_argument('--reports', nargs=2, type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        if name == 'freeze':
            command.add_argument('--third', type=Path)
            command.add_argument('--adjudication', type=Path)
    command = commands.add_parser('aggregate')
    command.add_argument('--judgments', type=Path, required=True)
    command.add_argument('--judgments-sha256', required=True)
    command.add_argument('--mapping', type=Path, required=True)
    command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'packet':
        result = make_packet(args.root, args.frozen, args.frozen_sha256, args.primary, args.out, args.secondary)
    elif args.command == 'shard':
        result = shard(args.root, args.packet, args.packet_sha256, args.size, args.out)
    elif args.command == 'collect':
        result = collect(args.root, args.index, args.index_sha256, args.panels, args.out, args.third)
    elif args.command == 'adjudication':
        result = adjudication(args.root, args.packet, args.reports, args.out)
    elif args.command == 'freeze':
        result = freeze(args.root, args.packet, args.reports, args.out, args.third, args.adjudication)
    else:
        result = aggregate(args.root, args.judgments, args.judgments_sha256, args.mapping, args.out)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
