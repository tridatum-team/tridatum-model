#!/usr/bin/env python3
"""Offline expanded diagnostic assessment; never selection or deployment qualification.

Final150 candidates use authenticated local MPS or separately registered CUDA
evaluation-only inference. CUDA training provenance never stands in for inference
receipts: both execution paths require their own source/runtime/export/cleanup.
Historical scores and the immutable paired review/gate protocol are not reinterpreted.
Only primary200 development cases qualify, never secondary or deployment.

Commands (all paths belong to --root):
  packet --frozen FILE --frozen-sha256 SHA --primary DIR [--secondary DIR] --out DIR
  shard --packet FILE --packet-sha256 SHA --size 16 --out DIR
  collect --index FILE --index-sha256 SHA --panels PANEL_A PANEL_B --out DIR
  collect --index FILE --index-sha256 SHA --panels PANEL_C --third --out DIR
  adjudication --packet FILE --reports FIRST SECOND --out DIR
  freeze --packet FILE --reports FIRST SECOND [--third FILE --adjudication FILE] --out FILE
  aggregate --judgments FILE --judgments-sha256 SHA --mapping FILE --out FILE

packet additionally accepts paired --candidate-manifest FILE --candidate-sha256
SHA for the registered final150 development cycle; omission is strict historical
step24 replay. Candidate review uses registered critical-error codes (concrete
evidence belongs in reason), unchanged anonymous protocol instructions, and
complete same-question pairs in each shard. A dispute expands to both samples;
the third report must cover that entire anonymous pair. Candidate aggregation
reauthenticates registration and supervision, never searches checkpoints/profiles
or compares scores against the historical grading policy.

The separate inference-only CUDA sampling condition requires explicit packet
--generation-profile qwen3_nonthinking_sampling and authenticated matching plan,
manifest, runtime, output and per-case seeds. Omission preserves historical greedy
authentication, including archive-only original failure recovery. Aggregate uses
the committed condition without another flag. One fixed seed is not evidence of
stochastic robustness; conditions are never pooled or selected against each other.

Review JSON: schema trid-expanded-blind-review-v1, packet_sha256, reviewer_id,
independent:true, blinded:true, judgments:{sample_id:{semantic_pass:bool,
format_pass:bool, critical_errors:[specific descriptions], reason:nonempty}}.
Third reports additionally bind adjudication_packet_sha256 and contain ONLY the
flagged IDs. Give reviewers only review-packet.json (or adjudication-packet.json),
never private-mapping.json, runtime files, or other reviewers' votes. Reference
paraphrases are acceptable; JSON validity is not semantic evidence. Freeze writes
resolved anonymous judgments before aggregate opens the arm mapping.
Completed string responses, including empty/whitespace answers, remain in the
denominator; structural failures are scored, not rejected as corrupt run records.

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
from scripts.local_eight_candidate import resolve_candidate, public_contract, review_protocol

ARMS = ('base', 'step-000024')
PIN_NAMES = {'design', 'source_allocation', 'evaluation_inputs', 'evaluation_metadata',
             'evaluation_references', 'training_manifest', 'semantic_review_resolution'}
METRICS = ('structural_pass', 'semantic_pass', 'format_pass', 'joint_pass')
ANONYMOUS_PROTOCOL_SHA = 'aa2ea535907290693f2d470cb21cb70d5573be20899da1052fbf3bc9f68abeb8'
GENERATION_PROFILES = ('greedy', 'qwen3_nonthinking_sampling')
PROTOCOL_FIELDS = ('instructions', 'paired_consistency', 'critical_error_codes', 'critical_errors_are_codes')
# Always imported by the evaluator before either code_identity receipt. Keep the
# producer and its local decoding/loading dependencies authenticated, not merely
# whichever unrelated files a receipt elects to list.
CANDIDATE_PRODUCER_SOURCES = frozenset({
    'scripts/evaluate_local_eight.py', 'scripts/local_eight_candidate.py',
    'scripts/train_open_weight_lora.py', 'scripts/aws_eval/grounded_hf_worker.py',
    'scripts/aws_eval/grounded_hf_lifecycle.py', 'scripts/aws_eval/qwen_cuda_smoke.py',
    'scripts/aws_eval/qwen_cpu_smoke.py', 'trid/prompts.py', 'trid/text.py', 'trid/case_prompts.py',
})
# Authenticating a CUDA-trained candidate imports its training contract in the
# local evaluator. These imports do not turn MPS inference into CUDA execution.
CUDA_TRAINED_MPS_SOURCES = CANDIDATE_PRODUCER_SOURCES | {
    'scripts/aws_eval/fresh_eight_experiment.py',
    'scripts/local_mps_training.py',
}


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


def authenticate_producer_sources(root, manifest_sources, runtime_sources, required=CANDIDATE_PRODUCER_SOURCES):
    root = Path(root).resolve()
    for sources in (manifest_sources, runtime_sources):
        require(isinstance(sources, dict) and required <= sources.keys(),
                'candidate mandatory producer source pins missing')
        for name, entry in sources.items():
            relative = Path(name)
            require(bool(relative.parts) and not relative.is_absolute() and relative.as_posix() == name
                    and '..' not in relative.parts and relative.parts[0] in ('scripts', 'trid')
                    and relative.suffix == '.py', 'candidate source key differs from producer identity')
            expected = root / relative
            recorded = Path(entry['path'])
            recorded = recorded if recorded.is_absolute() else root / recorded
            require(expected.is_file() and recorded.is_file()
                    and any(parent.samefile(root) for parent in expected.resolve().parents)
                    and recorded.samefile(expected)
                    and not expected.is_symlink() and expected.stat().st_nlink == 1,
                    'candidate producer source path substitution')
            authenticate(expected, entry)
    require(manifest_sources.keys() <= runtime_sources.keys()
            and all(all(manifest_sources[name][field] == runtime_sources[name][field]
                        for field in ('bytes', 'sha256')) for name in manifest_sources),
            'candidate producer sources changed between runtime receipts')


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


def authenticate_cuda_evaluation(root, directory, contract, manifest, loaded, generation_profile='greedy'):
    """Admit one independently collected CUDA execution, never historical scores."""
    import tempfile
    from scripts import local_eight_candidate as candidate
    from scripts.aws_eval import fresh_eight_evaluation as evaluation
    from scripts.aws_eval import fresh_eight_launch as launch
    from scripts.aws_eval import grounded_hf_lifecycle as lifecycle

    root, directory = Path(root).resolve(), Path(directory).resolve()
    exported, state = directory.parent, directory.parent.parent
    require(directory.name == 'evaluation' and exported.name == 'cuda-evaluation-export',
            'CUDA assessment requires the collected full evaluation namespace')
    plan_path = exported / 'plan.json'
    plan, plan_sha = load(plan_path), digest(plan_path)
    require(evaluation.is_evaluation(plan), 'CUDA evaluation plan schema differs')
    require(evaluation.plan_decoder_profile(plan) == generation_profile,
            'CUDA plan generation profile differs')
    # Metadata recovery is a separate admission, never a native-success alias.
    # Its registered historical validator reauthenticates the preserved failure,
    # original archive, all402 answers, failed-unit boundary and positive cleanup.
    from scripts.aws_eval import fresh_eight_evaluation_recover as recovery
    if (state / recovery.RECEIPT_NAME).exists():
        require(generation_profile == 'greedy', 'historical recovery is greedy only')
        lineage = candidate.cuda_evaluation_lineage(root,
            candidate.bound(root, contract['registration']), contract['registration']['sha256'])
        require(load(exported / 'lineage-binding.json') == lineage
                and manifest.get('evaluation_contract') == lineage['provenance']
                and lineage['adapter_sha256'] == contract['adapter_sha256'],
                'Recovered CUDA original candidate lineage differs')
        return recovery.validate_recovery(root, exported, plan, plan_sha)
    run = state.parent
    authenticate(run / 'plan.json', pin(plan_path, root))
    registration_path, approval_path = run / 'registration.json', run / 'approval.json'
    registration, approval = load(registration_path), load(approval_path)
    registration_sha = digest(registration_path)
    source_archive = candidate.bound(root, dict(path=plan['bundle']['local_path'],
        bytes=plan['bundle']['bytes'], sha256=plan['bundle']['sha256']))
    with tempfile.TemporaryDirectory(prefix='trid-evaluation-evidence-') as temporary:
        sources = Path(temporary).resolve() / 'bundle'
        lifecycle.authenticate_archive(source_archive, plan['bundle'], sources)
        launch.validate_registration(registration, sources, active=False)
        launch.validate_final(registration, registration_sha, sources, state,
                              plan, approval, plan_sha, active=False)
        source_manifest = load(exported / 'source-manifest.json')
        expected_sources = {entry['path']: entry for entry in plan['bundle']['files']
                            if entry['path'] in plan['sources']}
        require(set(expected_sources) == set(plan['sources']) == set(evaluation.REQUIRED_SOURCES),
                'CUDA executable source closure differs')
        for entry in expected_sources.values():
            candidate.bound(sources, entry)
        require(manifest.get('code') == loaded.get('code') == expected_sources,
                'CUDA actual executed source pins differ')
        require(source_manifest == load(candidate.bound(sources, plan['workload']['source_manifest'])),
                'CUDA exported source manifest differs from staged executable closure')
        staged_lineage = load(candidate.bound(sources, plan['workload']['lineage_binding']))
        require(load(exported / 'lineage-binding.json') == staged_lineage,
                'CUDA exported lineage differs from staged lineage')
    lineage = candidate.cuda_evaluation_lineage(root,
        candidate.bound(root, contract['registration']), contract['registration']['sha256'])
    require(staged_lineage == lineage and manifest.get('evaluation_contract') == lineage['provenance']
            and lineage['adapter_sha256'] == contract['adapter_sha256'],
            'CUDA evaluation original candidate lineage differs')
    evaluation.validate_completion(exported, plan)
    supervision = evaluation.validate_supervision(exported, plan)

    completion = load(state / 'completion.json')
    controller = load(state / 'controller-result.json')
    instance = completion.get('instance_id')
    require(completion.get('run_id') == plan['identity']['run_id']
            and completion.get('plan_sha256') == plan_sha
            and isinstance(instance, str) and instance.startswith('i-')
            and 'training_success' not in completion and 'inference_success' not in completion
            and not completion.get('error') and not completion.get('export_error')
            and completion.get('runtime') == load(exported / 'runtime.json'),
            'CUDA terminal runtime/identity differs')
    lifecycle.validate_evaluation_terminal(plan, plan_sha, instance, completion)
    require(supervision['unit'] == completion['stages'][0],
            'CUDA exported supervision differs from independently collected phase')
    export_manifest = load(exported / 'export-manifest.json')
    require(export_manifest.get('run_id') == plan['identity']['run_id']
            and export_manifest.get('plan_sha256') == plan_sha
            and export_manifest.get('status') == {k: v for k, v in completion.items() if k != 'export'},
            'CUDA export terminal identity differs')
    members = export_manifest['files']
    names = [entry['path'] for entry in members]
    require(len(names) == len(set(names)) and set(names) <= set(plan['output_allowlist'])
            and export_manifest.get('total_bytes') == sum(entry['bytes'] for entry in members),
            'CUDA export membership/bytes differ')
    actual = {path.relative_to(exported).as_posix() for path in exported.rglob('*') if path.is_file()}
    require(actual == set(names) | {'export-manifest.json'}, 'CUDA collected export membership differs')
    for entry in members:
        candidate.bound(exported, entry)
    archive = state / 'export.tar'
    archive_pin = completion['export']
    authenticate(archive, archive_pin)
    lifecycle.authenticate_archive(archive, dict(archive_pin,
        unpacked_bytes=lifecycle.output_limit(plan),
        files=members + [dict(candidate.file_pin(exported / 'export-manifest.json'),
                              path='export-manifest.json')]))
    evidence = controller.get('evidence', {})
    require(controller.get('state') == 'complete' and controller.get('evaluation_success') is True
            and controller.get('instance_id') == instance
            and not any(controller.get(key) for key in ('error', 'evidence_error', 'schedule_cleanup_error'))
            and evidence.get('native_pipeline_success') is True
            and evidence.get('evidence_kind') == 'authenticated_archive'
            and evidence.get('export_state') == 'authenticated_terminal'
            and evidence.get('completion') == completion
            and evidence.get('export_sha256') == archive_pin['sha256'],
            'CUDA independently collected terminal success missing')
    volume = lifecycle.load_volume_identity(state, plan, plan_sha)
    cleanup = controller.get('cleanup', {})
    require(volume['instance_id'] == instance and len(volume['root_volume_ids']) == 1
            and cleanup.get('cleanup_state') == 'complete' and cleanup.get('instance_terminated') is True
            and cleanup.get('instance_id') == instance and cleanup.get('remaining_volumes') == []
            and set(cleanup.get('root_volume_ids', [])) == set(volume['root_volume_ids'])
            and set(cleanup.get('recorded_volume_ids', [])) == set(volume['volumes'])
            and set(cleanup.get('absent_volume_ids', [])) == set(volume['volumes']),
            'CUDA owned compute/root-volume cleanup incomplete')
    require(load(state / 'approval-consumed.json') ==
            dict(run_id=plan['identity']['run_id'], plan_sha256=plan_sha),
            'CUDA single consumed approval differs')
    release = load(state / 'release.json')
    require(all(release.get(key) == value for key, value in
                dict(run_id=plan['identity']['run_id'], plan_sha256=plan_sha, instance_id=instance).items()),
            'CUDA runtime release ownership differs')
    require(manifest['runtime']['gpu_uuid'] == loaded['gpu_uuid'] == release['gpu']['uuid']
            and manifest['runtime']['instance_id'] == instance,
            'CUDA observed inference device differs from released instance/GPU')
    for stage in [completion['runtime_unit'], *completion['stages']]:
        lifecycle.verify_gpu_observation(stage['gpu'], plan, release['gpu'])
    prerequisite_cleanup = load(state / 'owned-cleanup-complete.json')
    require(prerequisite_cleanup == dict(registration_sha256=registration_sha,
            cleanup_state='complete', retained_prefix_lifecycle_rules=True, s3_mutations=0),
            'CUDA owned prerequisite cleanup incomplete')
    owned = load(state / 'owned-prerequisites.json')
    require(load(state / 'owned-cleanup-started.json') ==
            dict(registration_sha256=registration_sha, resources=owned['resources']),
            'CUDA prerequisite cleanup authority differs')
    require(load(state / 'registration-consumed.json') ==
            dict(registration_sha256=registration_sha, identity=registration['identity']),
            'CUDA prerequisite registration consumption differs')
    identity = registration['identity']
    deletions = {
        'schedule-group': ('scheduler', 'delete-schedule-group', ['--name', identity['schedule_group']],
                           'ResourceNotFoundException'),
        'profile': ('iam', 'delete-instance-profile', ['--instance-profile-name', identity['instance_profile']],
                    'NoSuchEntity'),
        'security-group': ('ec2', 'delete-security-group',
                           ['--group-id', owned['resources']['security_group']['GroupId']], 'InvalidGroup.NotFound'),
        **{key: ('iam', 'delete-role', ['--role-name', identity[key]], 'NoSuchEntity') for key in launch.ROLE_KEYS}}
    cleanup_paths = []
    for name, (service, operation, arguments, absence_code) in deletions.items():
        paths = [state / ('cleanup-' + name + '-' + suffix + '.json')
                 for suffix in ('intent', 'response', 'readback')]
        intent, response, readback = map(load, paths)
        require(intent == dict(registration_sha256=registration_sha, service=service,
                               operation=operation, args=arguments)
                and response.get('request') == readback.get('request') == intent
                and readback.get('observation') == dict(absent=True, error_code=absence_code),
                'CUDA prerequisite deletion readback missing: ' + name)
        cleanup_paths.extend(paths)
    # Keep immutable cloud/source evidence in the blind mapping commitment. The
    # aggregator reauthenticates this same admission before joining identities.
    evidence_paths = [registration_path, approval_path, source_archive, archive, plan_path, *cleanup_paths,
        exported / 'lineage-binding.json', exported / 'source-manifest.json',
        *[state / name for name in ('completion.json', 'controller-result.json',
            'volume-identity.json', 'release.json', 'approval-consumed.json', 'registration-consumed.json',
            'owned-prerequisites.json', 'owned-cleanup-started.json', 'owned-cleanup-complete.json')]]
    return dict(pins={'cuda-evidence-' + str(i): pin(path, root)
                      for i, path in enumerate(evidence_paths)},
                public_input=candidate.bound(root, manifest['input']))


def completed_run(root, directory, cases, design, profile, input_pin=None, contract=None,
                  generation_profile='greedy'):
    """Authenticate the evaluator's actual file/row contract, never model weights."""
    directory = Path(directory)
    require(generation_profile in GENERATION_PROFILES, 'unknown generation profile')
    sampling = generation_profile == 'qwen3_nonthinking_sampling'
    arms = contract['arms'] if contract else ARMS
    cuda_trained = contract is not None and contract.get('training_runtime') == 'cuda'
    adapter_sha = contract['adapter_sha256'] if contract else design['model']['adapter_sha256']
    names = ['run-manifest.json', 'decoder-profile.json', 'prompt-preflight.json',
             'loaded-runtime.json', 'completion.json', *[arm + '-outputs.jsonl' for arm in arms]]
    pins = {name: pin(directory / name, root) for name in names}
    manifest = load(directory / 'run-manifest.json')
    completion = load(directory / 'completion.json')
    decoder = load(directory / 'decoder-profile.json')
    preflight = load(directory / 'prompt-preflight.json')
    loaded = load(directory / 'loaded-runtime.json')
    manifest_sha = pins['run-manifest.json']['sha256']
    cuda_evaluation = manifest.get('schema') == 'trid-cuda-eight-paired-v1'
    require(not sampling or (cuda_evaluation and contract is not None and cuda_trained
            and profile == 'unconstrained' and len(cases) == 200),
            'sampling requires separate CUDA final150 primary200')
    if sampling:
        require(manifest.get('generation_profile') == loaded.get('generation_profile') == generation_profile,
                'manifest/runtime generation profile binding differs')
    else:
        require('generation_profile' not in manifest and 'generation_profile' not in loaded,
                'historical greedy generation profile differs')
    if not cuda_evaluation:
        require(not (directory / 'failure.json').exists(), 'failed local run is not complete')
    if cuda_evaluation:
        require(contract is not None and cuda_trained and profile == 'unconstrained'
                and len(cases) == 200, 'CUDA evaluation requires final150 primary200')
        cuda_admission = authenticate_cuda_evaluation(
            root, directory, contract, manifest, loaded, generation_profile)
        pins.update(cuda_admission['pins'])
    expected_schema = ('trid-cuda-eight-paired-v1' if cuda_evaluation else
                       'trid-local-eight-paired-v2' if contract else 'trid-local-eight-paired-v1')
    require(manifest.get('schema') == expected_schema
            and manifest.get('mode') == 'evaluation' and manifest.get('arm_order') == list(arms)
            and manifest.get('grading_access') is False and manifest.get('retries') is False
            and manifest.get('resume') is False, 'local run identity/mode differs')
    if contract and not cuda_evaluation:
        require(manifest.get('evaluation_contract') == public_contract(contract)
                and manifest.get('adapter') == contract['adapter'] and profile == 'unconstrained',
                'registered evaluation contract differs')
        model = manifest['model']
        from scripts.evaluate_local_eight import model_identity
        require(model_identity(model['directory'], model['acquisition_manifest']['path']) == model,
                'candidate official base authentication differs')
        required_sources = CUDA_TRAINED_MPS_SOURCES if cuda_trained else CANDIDATE_PRODUCER_SOURCES
        if contract.get('collection_recovery'):
            required_sources = required_sources | {'scripts/aws_eval/fresh_eight_recover.py'}
        authenticate_producer_sources(root, manifest.get('code'), loaded.get('code'), required_sources)
        if cuda_trained:
            from scripts.evaluate_local_eight import accepted_local_supervision
            require(isinstance(manifest.get('local_registration'), dict), 'Local guarded registration missing')
            supervisor = accepted_local_supervision(manifest['local_registration'], directory)
            pins['local-supervision'] = pin(supervisor, root)
    elif not contract:
        require('evaluation_contract' not in manifest, 'historical replay cannot carry a candidate contract')
    ids = manifest.get('selected_ids', [])
    require(len(ids) == len(set(ids)) == len(cases) and set(ids) == set(cases), 'selected case denominator')
    require(completion.get('status') == 'complete' and completion.get('mode') == 'evaluation'
            and completion.get('rows_per_arm') == len(cases)
            and completion.get('run_manifest_sha256') == manifest_sha
            and set(completion.get('outputs', {})) == set(arms)
            and finite_seconds(completion.get('elapsed_seconds')), 'completion identity/denominator')
    if cuda_evaluation:
        public_path = cuda_admission['public_input']
    else:
        public_path = Path(manifest['input']['path'])
        public_path = public_path if public_path.is_absolute() else Path(root) / public_path
    authenticate(public_path, manifest['input'])
    require(indexed(load(public_path), 'run inputs') == cases, 'run input messages differ')
    if input_pin is not None:
        require(all(manifest['input'][k] == input_pin[k] for k in ('sha256', 'bytes')),
                'primary frozen input pin differs')
    generation = manifest['generation']
    expected_generation = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
        max_new_tokens=768, context_tokens=4096, dtype='bfloat16',
        backend='transformers-cuda' if cuda_evaluation else 'transformers-mps')
    design_fields = ('enable_thinking', 'do_sample', 'max_new_tokens', 'context_tokens')
    if sampling:
        from scripts.aws_eval import fresh_eight_evaluation as evaluation
        expected_generation = evaluation.generation_for_profile(generation_profile)
        design_fields = ('enable_thinking', 'max_new_tokens', 'context_tokens')
    require((object_sha(generation) == object_sha(expected_generation) if sampling
             else generation == expected_generation)
            and all(generation[k] == design['generation'][k] for k in design_fields),
            'registered generation differs')
    model = manifest['model']
    adapter = manifest['adapter']
    require(model.get('id') == design['model']['repo_id']
            and model.get('revision') == design['model']['revision']
            and model.get('body_hashes_authenticated') is True
            and model.get('representation') == 'hf-safetensors-bfloat16'
            and adapter.get('step') == (150 if contract else design['model']['adapter_step'])
            and (contract is not None or adapter.get('step') == 24)
            and adapter['files']['adapter_model.safetensors']['sha256'] == adapter_sha,
            'model/adapter binding differs')
    runtime = manifest['runtime']
    require(runtime.get('device') == ('cuda' if cuda_evaluation else 'mps')
            and runtime.get('dtype') == 'torch.bfloat16'
            and runtime.get('cpu_fallback') is False and runtime.get('cuda_equivalence_claim') is False
            and loaded.get('parameter_dtypes') == (['torch.bfloat16', 'torch.float32'] if cuda_trained
                                                   else ['torch.bfloat16']), 'runtime/device/dtype differs')
    if cuda_trained:
        inventory = loaded.get('parameter_inventory', {})
        require(set(inventory) == {'base', 'adapter'}, 'inference parameter inventory missing')
        for group, dtype in (('base', 'torch.bfloat16'), ('adapter', 'torch.float32')):
            values = inventory[group]
            device = 'cuda' if cuda_evaluation else 'mps'
            require(values and all(key in (dtype + '@' + device, dtype + '@' + device + ':0')
                    and all(type(entry.get(field)) is int and entry[field] > 0
                            for field in ('tensors', 'elements', 'bytes'))
                    for key, entry in values.items()), 'inference parameter inventory differs')
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
    policy = loaded.get('generation_policy', {})
    if sampling:
        evaluation.validate_generation_policy(config, policy, generation_profile)
    else:
        require(config.get('do_sample') is False and config.get('num_beams') == 1
                and config.get('max_new_tokens') == 768 and config.get('return_dict_in_generate') is True,
                'loaded generation configuration differs')
        require(policy.get('dispatch_kwargs') == {'use_model_defaults': False, 'do_sample': False, 'num_beams': 1}
                and policy.get('generation_mode') == 'greedy_search'
                and policy.get('effective_config_sha256') == object_sha(config)
                and 'generation_profile' not in policy,
                'effective greedy generation policy differs')
    implementation = policy.get('preparation_implementation', {})
    require(isinstance(implementation.get('path'), str) and implementation['path']
            and type(implementation.get('bytes')) is int and implementation['bytes'] > 0
            and isinstance(implementation.get('sha256'), str) and len(implementation['sha256']) == 64
            and all(c in '0123456789abcdef' for c in implementation['sha256']),
            'generation preparation implementation pin missing')
    requested = loaded.get('requested_generation_config', {})
    requested_keys = ('do_sample', 'num_beams', 'max_new_tokens', 'return_dict_in_generate')
    if sampling:
        requested_keys = tuple(dict.fromkeys((*requested_keys, *evaluation.SAMPLING_PARAMETERS)))
    require(all((key in requested and object_sha(requested[key]) == object_sha(config[key]))
                if sampling else requested.get(key) == config[key] for key in requested_keys),
            'requested/effective generation contract differs')
    answers = {}
    for arm in arms:
        path = directory / (arm + '-outputs.jsonl')
        authenticate(path, completion['outputs'][arm])
        recorded_output = Path(completion['outputs'][arm]['path'])
        if cuda_evaluation:
            require(not recorded_output.is_absolute() and recorded_output.as_posix() == arm + '-outputs.jsonl',
                    'CUDA completion output must be portable within its mode')
            recorded_output = directory / recorded_output
        require(recorded_output.resolve() == path.resolve(),
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
                    and row.get('nominated_adapter_weights_sha256') == adapter_sha
                    and row.get('adapter_weights_sha256') == (None if arm == 'base' else adapter_sha)
                    and row.get('mode') == 'evaluation' and row.get('decoder_profile') == profile
                    and row.get('decoder_profile_sha256') == pins['decoder-profile.json']['sha256']
                    and row.get('generation') == generation and row.get('generation_config') == config
                    and row.get('generation_policy_sha256') == object_sha(policy)
                    and row.get('effective_generation_mode') == ('sample' if sampling else 'greedy_search'),
                    'answer arm/runtime/profile/adapter binding differs')
            if sampling:
                require(row.get('generation_profile') == generation_profile
                        and type(row.get('case_seed')) is int
                        and row['case_seed'] == evaluation.case_seed('evaluation', cid),
                        'answer generation profile/case seed binding differs')
            else:
                require('generation_profile' not in row and 'case_seed' not in row,
                        'historical greedy answer profile/seed differs')
            if contract:
                require(row.get('candidate_manifest_sha256') == contract['registration']['sha256'],
                        'answer registration binding differs')
            require(row.get('messages') == prompt.get('messages') == case['messages']
                    and row.get('prompt_sha256') == prompt.get('prompt_sha256') == object_sha(case['messages'])
                    and row.get('prompt_token_ids') == prompt.get('prompt_token_ids')
                    and row.get('rendered_prompt_sha256') == prompt.get('rendered_prompt_sha256'),
                    'answer case/prompt binding differs')
            require(isinstance(row.get('response_text'), str)
                    and not row.get('exception') and finite_seconds(row.get('elapsed_seconds')),
                    'missing answer or invalid latency')
            answers[(cid, arm)] = row
    return answers, pins, manifest, loaded


def mapping_body(mapping):
    return {k: v for k, v in mapping.items() if k not in ('schema', 'packet_sha256')}


def make_packet(root, frozen_path, frozen_sha, primary, out, secondary=None,
                candidate_manifest=None, candidate_sha=None, generation_profile='greedy'):
    require(bool(candidate_manifest) == bool(candidate_sha), 'candidate manifest and SHA must be paired')
    require(generation_profile in GENERATION_PROFILES, 'unknown generation profile')
    require(generation_profile == 'greedy' or (candidate_manifest and secondary is None),
            'sampling requires registered candidate without secondary search')
    contract = resolve_candidate(root, candidate_manifest, candidate_sha) if candidate_manifest else None
    arms = contract['arms'] if contract else ARMS
    if contract:
        require(secondary is None and frozen_sha == contract['policy_document']['data']['frozen_dataset']['sha256'],
                'candidate requires registered development dataset and no secondary search')
    frozen, design, cases, metadata, references = frozen_dataset(root, frozen_path, frozen_sha)
    out = Path(out)
    require(not out.exists(), 'packet directory already exists; never rerandomize')
    runs = [('primary', primary, cases, 'unconstrained')]
    if secondary is not None:
        runs.append(('secondary', secondary, {cid: cases[cid] for cid in frozen['secondary_case_ids']}, 'json-schema'))
    mapping = dict(schema='trid-expanded-blind-mapping-v2' if contract else 'trid-expanded-blind-mapping-v1',
                   frozen_dataset=pin(frozen_path, root), rows={}, runs={}, arms=list(arms), primary_case_ids=list(cases),
                   secondary_case_ids=frozen['secondary_case_ids'] if secondary is not None else [])
    packet_rows = []
    pair_ids = {}
    if contract:
        mapping['evaluation_contract'] = public_contract(contract)
    for suite, directory, subset, profile in runs:
        answers, pins, manifest, loaded = completed_run(root, directory, subset, design, profile,
            frozen['pins']['evaluation_inputs'] if suite == 'primary' else None, contract=contract,
            generation_profile=generation_profile)
        mapping['runs'][suite] = {'directory': str(Path(directory).resolve()), 'pins': pins,
                                  'runtime': manifest['runtime'], 'generation': manifest['generation'],
                                  'generation_config': loaded['generation_config'],
                                  'generation_policy': loaded['generation_policy']}
        if generation_profile != 'greedy':
            mapping['runs'][suite]['generation_profile'] = generation_profile
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
            if contract:
                packet_rows[-1]['pair_id'] = pair_ids.setdefault((suite, cid), secrets.token_hex(24))
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
    if contract:
        packet['schema'] = 'trid-expanded-blind-packet-v2'
        packet.update(review_protocol(root, contract))
    save_new(out / 'review-packet.json', packet)
    mapping['packet_sha256'] = digest(out / 'review-packet.json')
    save_new(out / 'private-mapping.json', mapping)
    return {'packet': str(out / 'review-packet.json'), 'samples': len(packet_rows)}


def validate_blind_packet(packet, *, partial=False):
    candidate = packet.get('schema') == 'trid-expanded-blind-packet-v2'
    require(packet.get('schema') in ('trid-expanded-blind-packet-v1', 'trid-expanded-blind-packet-v2'), 'packet schema differs')
    fields = {'schema', 'rows', 'mapping_commitment', 'instructions'}
    row_fields = {'sample_id', 'messages', 'response', 'reference_answer', 'rubric', 'structural'}
    if candidate:
        require(object_sha({key: packet.get(key) for key in PROTOCOL_FIELDS}) == ANONYMOUS_PROTOCOL_SHA,
                'prospective anonymous review protocol changed')
        fields.update(('paired_consistency', 'critical_error_codes', 'critical_errors_are_codes'))
        row_fields.add('pair_id')
        require(packet['paired_consistency'] is True and packet['critical_errors_are_codes'] is True
                and isinstance(packet['critical_error_codes'], dict) and packet['critical_error_codes'],
                'registered anonymous review protocol missing')
        pairs = defaultdict(list)
        for row in packet['rows']:
            require(isinstance(row.get('pair_id'), str) and row['pair_id'], 'anonymous pair identity missing')
            pairs[row['pair_id']].append(row)
        require(all(len(pair) == 2 and all(pair[0][key] == pair[1][key]
                    for key in ('messages', 'reference_answer', 'rubric')) for pair in pairs.values()),
                'complete anonymous counterpart context required')
    if partial:
        fields.add('parent_packet_sha256')
    require(set(packet) == fields
            and all(set(row) == row_fields
                    and set(row['structural']) == {'pass', 'reason'}
                    and type(row['structural']['pass']) is bool for row in packet['rows']),
            'blind packet contains undeclared identity/vote fields')
    ids = [r['sample_id'] for r in packet['rows']]
    require(all(isinstance(sid, str) and sid for sid in ids) and len(ids) == len(set(ids))
            and (partial or len(ids) in ((400,) if candidate else (400, 480))), 'blind packet denominator')
    return ids


def shard_document(parent, parent_sha, rows):
    result = dict(schema='trid-expanded-blind-shard-v1', parent_packet_sha256=parent_sha,
                  rows=rows, instructions=parent['instructions'])
    if parent.get('schema') == 'trid-expanded-blind-packet-v2':
        result['schema'] = 'trid-expanded-blind-shard-v2'
        result.update({key: parent[key] for key in
                      ('paired_consistency', 'critical_error_codes', 'critical_errors_are_codes')})
    else:
        result['instructions'] = ('Independently review only these samples. Report against this child packet file SHA, '
            'not its parent hash. Use trid-expanded-blind-review-v1 with your own actual reviewer_id, independent:true, '
            'blinded:true and every sample judgment. Judge material meaning and requested format separately; accept '
            'semantic paraphrases. Structural flags are not entailment evidence. Never request arm identities, '
            "timing, decoder profile, or other reviewers' votes.")
    return result


def shard_subsets(parent, size):
    if parent.get('schema') != 'trid-expanded-blind-packet-v2':
        return [parent['rows'][start:start + size] for start in range(0, len(parent['rows']), size)]
    require(size >= 2, 'paired review shard must hold both responses')
    pairs = {}
    for row in parent['rows']:
        pairs.setdefault(row['pair_id'], []).append(row)
    ordered = [row for pair in pairs.values() for row in pair]
    width = size - size % 2
    return [ordered[start:start + width] for start in range(0, len(ordered), width)]


def shard(root, packet_path, packet_sha, size, out):
    require(digest(packet_path) == packet_sha, 'shard parent packet hash differs')
    parent = load(packet_path)
    ids = validate_blind_packet(parent, partial='parent_packet_sha256' in parent)
    require(ids and type(size) is int and size > 0, 'nonempty packet and positive shard size required')
    out = Path(out)
    require(not out.exists(), 'shard directory already exists')
    index = dict(schema='trid-expanded-review-shards-v1', parent=pin(packet_path, root), size=size, children=[])
    for number, subset in enumerate(shard_subsets(parent, size), 1):
        path = out / f'shard-{number:04d}.json'
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
    subsets = shard_subsets(parent, index['size'])
    require(ids and len(index['children']) == len(subsets), 'shard coverage denominator')
    children = {}
    for number, entry in enumerate(index['children']):
        require(set(entry) == {'packet', 'sample_ids'}, 'shard index child fields')
        child_path = bound(root, entry['packet'])
        child = load(child_path)
        subset = subsets[number]
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
    if packet.get('paired_consistency'):
        disputed_pairs = {row['pair_id'] for row in packet['rows'] if row['sample_id'] in disagreements}
        disagreements = sorted(row['sample_id'] for row in packet['rows'] if row['pair_id'] in disputed_pairs)
    return packet, documents, judgments, disagreements


def review_document(report, packet, packet_sha, ids=None, *, root=ROOT):
    if packet.get('critical_errors_are_codes'):
        require(object_sha({key: packet.get(key) for key in PROTOCOL_FIELDS}) == ANONYMOUS_PROTOCOL_SHA,
                'prospective anonymous review protocol changed')
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
    judgments = validate_report(dict(report, schema='trid-eight-blind-review-v1'), packet, packet_sha, ids)
    if packet.get('critical_errors_are_codes'):
        require(all(set(judgment['critical_errors']) <= set(packet['critical_error_codes'])
                    for judgment in judgments.values()), 'unregistered critical-error code')
    return judgments


def adjudication_document(packet, packet_sha, disagreements):
    document = dict(packet, parent_packet_sha256=packet_sha,
                    rows=[row for row in packet['rows'] if row['sample_id'] in disagreements])
    if not packet.get('paired_consistency'):
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


def paired_summary(rows, arm_names=ARMS):
    by_case = defaultdict(dict)
    for row in rows:
        require(row['arm'] not in by_case[row['case_id']], 'duplicate paired arm')
        by_case[row['case_id']][row['arm']] = row
    require(all(set(pair) == set(arm_names) for pair in by_case.values()), 'incomplete paired summary')
    arms = {}
    for arm in arm_names:
        selected = [pair[arm] for pair in by_case.values()]
        arms[arm] = {metric: sum(r[metric] for r in selected) for metric in METRICS}
        arms[arm]['critical_case_count'] = sum(bool(r['critical_errors']) for r in selected)
        arms[arm]['latency'] = latency([r['elapsed_seconds'] for r in selected])
    paired = {}
    for metric in METRICS:
        gains = sum(not p[arm_names[0]][metric] and p[arm_names[1]][metric] for p in by_case.values())
        losses = sum(p[arm_names[0]][metric] and not p[arm_names[1]][metric] for p in by_case.values())
        paired[metric] = {'gains': gains, 'regressions': losses, 'net_cases': gains - losses,
                          'net_fraction': (gains - losses) / len(by_case)}
    paired['critical_errors'] = {
        'introduced_cases': sum(not p[arm_names[0]]['critical_errors'] and bool(p[arm_names[1]]['critical_errors']) for p in by_case.values()),
        'resolved_cases': sum(bool(p[arm_names[0]]['critical_errors']) and not p[arm_names[1]]['critical_errors'] for p in by_case.values())}
    paired['latency_delta_seconds'] = statistics.mean(
        p[arm_names[1]]['elapsed_seconds'] - p[arm_names[0]]['elapsed_seconds'] for p in by_case.values())
    return {'cases': len(by_case), 'answers': len(rows), 'arms': arms, 'paired': paired}


def suite_summary(rows, arms=ARMS):
    summary = paired_summary(rows, arms)
    for key in ('cohort', 'behavior', 'family_id'):
        groups = defaultdict(list)
        for row in rows:
            # Keep cohort in family identity even if two authors chose the same label.
            label = row['cohort'] + ':' + row[key] if key == 'family_id' else row[key]
            groups[label].append(row)
        summary['by_' + key] = {label: paired_summary(group, arms) for label, group in sorted(groups.items())}
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
    require(mapping.get('schema') == ('trid-expanded-blind-mapping-v2'
            if packet['schema'] == 'trid-expanded-blind-packet-v2' else 'trid-expanded-blind-mapping-v1')
            and mapping.get('packet_sha256') == digest(packet_path)
            and object_sha(mapping_body(mapping)) == frozen['mapping_commitment']
            and set(mapping['rows']) == set(resolved), 'private mapping commitment differs')
    contract = None
    if packet['schema'] == 'trid-expanded-blind-packet-v2':
        registered = mapping['evaluation_contract']['registration']
        contract = resolve_candidate(root, bound(root, registered), registered['sha256'])
        require(mapping['evaluation_contract'] == public_contract(contract)
                and all(packet[key] == value for key, value in review_protocol(root, contract).items())
                and set(mapping['runs']) == {'primary'}
                and mapping['frozen_dataset']['sha256'] == contract['policy_document']['data']['frozen_dataset']['sha256'],
                'prospective candidate/protocol/dataset binding differs')
    else:
        require('evaluation_contract' not in mapping, 'historical mapping cannot nominate candidate')
    arms = contract['arms'] if contract else ARMS
    dataset_path = bound(root, mapping['frozen_dataset'])
    dataset, design, cases, metadata, references = frozen_dataset(root, dataset_path, mapping['frozen_dataset']['sha256'])
    require(mapping['arms'] == list(arms) and mapping['primary_case_ids'] == list(cases)
            and set(mapping['runs']) in ({'primary'}, {'primary', 'secondary'})
            and mapping['secondary_case_ids'] == (dataset['secondary_case_ids'] if 'secondary' in mapping['runs'] else []),
            'mapping suite denominator differs')
    if contract:
        paired_mapping = defaultdict(list)
        for row in packet['rows']:
            paired_mapping[row['pair_id']].append(mapping['rows'][row['sample_id']])
        require(all(len(pair) == 2 and pair[0]['case_id'] == pair[1]['case_id']
                    and pair[0]['suite'] == pair[1]['suite']
                    and {item['arm'] for item in pair} == set(arms) for pair in paired_mapping.values()),
                'anonymous pair mapping differs')
    expected = {}
    for suite, run in mapping['runs'].items():
        for entry in run['pins'].values():
            bound(root, entry)
        subset = cases if suite == 'primary' else {cid: cases[cid] for cid in dataset['secondary_case_ids']}
        answers, current_pins, manifest, loaded = completed_run(root, run['directory'], subset, design,
            'unconstrained' if suite == 'primary' else 'json-schema',
            dataset['pins']['evaluation_inputs'] if suite == 'primary' else None, contract=contract,
            generation_profile=run.get('generation_profile', 'greedy'))
        require(current_pins == run['pins'], 'run artifact membership changed')
        if run.get('generation_profile', 'greedy') != 'greedy':
            require(all(run.get(key) == manifest[key] for key in ('runtime', 'generation'))
                    and all(run.get(key) == loaded[key] for key in ('generation_config', 'generation_policy')),
                    'sampling mapping runtime/generation differs')
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
        primary=suite_summary([r for r in joined if r['suite'] == 'primary'], arms),
        selected_step=None, diagnostic_comparator_step=24, deployment_qualified=False,
        original_no_selection_unchanged=True, broad_accuracy_claim=False,
        source_caveats=dataset.get('source_caveats', design.get('source_caveats', [])),
        independent_primary_reviews=2, adjudicated_samples=len(disagreements),
        interpretation='Source-separated diagnostic only; no automatic qualification or deployment. '
                       'Structure is not semantic correctness; foundation pretraining exposure is unknown.',
        secondary=None)
    if contract:
        result['schema'] = 'trid-expanded-assessment-v2'
        del result['diagnostic_comparator_step']
        result.update(evaluation_contract=public_contract(contract), candidate_step=150,
                      role='development_only_not_unseen_final', historical_scores_unchanged=True)
        primary = result['primary']
        base, candidate = primary['arms']['base'], primary['arms']['candidate']
        mois = primary['by_cohort']['mois']['arms']
        general = primary['by_cohort']['general']['arms']
        checks = dict(semantic_pass=candidate['semantic_pass'] > base['semantic_pass'],
            mois_semantic_pass=mois['candidate']['semantic_pass'] > mois['base']['semantic_pass'],
            joint_pass=candidate['joint_pass'] > base['joint_pass'],
            critical_case_count=candidate['critical_case_count'] <= base['critical_case_count'],
            general_semantic_pass=general['candidate']['semantic_pass'] >= general['base']['semantic_pass'],
            general_critical_case_count=general['candidate']['critical_case_count'] <= general['base']['critical_case_count'])
        result['development_gate'] = dict(policy=contract['policy_document']['evaluation']['development_gate'],
            checks=checks, passed=all(checks.values()), deployment_qualification=False)
        result['interpretation'] = ('Contemporaneous base versus one prospectively fixed final endpoint on development data. '
            'No historical score comparison, checkpoint selection, unseen-final claim, or deployment qualification.')
        generation_profile = mapping['runs']['primary'].get('generation_profile', 'greedy')
        if generation_profile != 'greedy':
            result.update(generation_profile=generation_profile,
                generation_condition='separate_inference_only_fixed_seed_sampling',
                historical_pooling=False, stochastic_robustness_established=False,
                generation=mapping['runs']['primary']['generation'])
            result['interpretation'] += (
                ' Separate nonthinking sampling condition with one fixed paired per-case seed; '
                'no pooling with historical greedy results, decoder tuning, or stochastic robustness claim.')
    if 'secondary' in mapping['runs']:
        secondary_rows = [r for r in joined if r['suite'] == 'secondary']
        result['secondary'] = suite_summary(secondary_rows, arms)
        result['secondary']['combined_with_primary'] = False
        primary_lookup = {(r['case_id'], r['arm']): r for r in joined if r['suite'] == 'primary'}
        benefit = {}
        for arm in arms:
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
    packet.add_argument('--candidate-manifest', type=Path)
    packet.add_argument('--candidate-sha256')
    packet.add_argument('--generation-profile', choices=GENERATION_PROFILES, default='greedy')
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
        result = make_packet(args.root, args.frozen, args.frozen_sha256, args.primary, args.out, args.secondary,
                             args.candidate_manifest, args.candidate_sha256, args.generation_profile)
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
