"""Offline admission/projection and the separate public-evidence GPU contract.

Private annotation and reference artifacts are authenticated only by prepare_projection;
they are never copied into the execution bundle. Research admission is not paid-run
permission or a model-quality verdict.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path, PurePosixPath
import sys

# The supervised isolated interpreter starts with only aws_eval on sys.path.
# This root is derived from the authenticated executable, never caller input.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws_eval.fresh_eight_experiment import pin, require
from scripts.aws_eval.fresh_eight_evaluation import RUNTIME_PACKAGES, ACQUISITION_SHA

SCHEMA = 'trid-public-evidence-gpu-plan-v2'
APPROVAL_SCHEMA = 'trid-public-evidence-gpu-approval-v2'
AUTHORIZATION_SCHEMA = 'trid-public-evidence-gpu-authorization-v2'
RESOURCE_ENVELOPE_SCHEMA = 'trid-public-evidence-resource-envelope-v2'
CLOUD_ADMISSION_SCHEMA = 'trid-public-evidence-cloud-admission-v2'
SOURCE_MANIFEST_SCHEMA = 'trid-public-evidence-source-manifest-v2'
OFFLINE_PROJECTION_SCHEMA = 'trid-public-evidence-offline-projection-v2'
GENERATION_BINDING_SCHEMA = 'trid-public-evidence-generation-binding-v2'
ADAPTER_SCHEMA = 'trid-public-evidence-adapter-v2'
TRAINING_ROWS = 1230
GENERAL_TRAINING_ROWS = 246
EVALUATION_ROWS = 708
PROFILE = 'public_evidence_v1'
COMPLETION_SCOPE = 'public_evidence_training_and_paired_evaluation'
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
RECIPE = dict(rows=TRAINING_ROWS, epochs=1, batch_size=2, micro_batch_size=1,
              rank=8, alpha=16, dropout=0.0, lr=1e-5, seed=14,
              context_tokens=4096, answer_tokens=768, checkpoint_every=0, valid_every=0)
RECIPE['steps'] = RECIPE['rows'] * RECIPE['epochs'] // RECIPE['batch_size']
GENERATION = dict(enable_thinking=False, do_sample=True, temperature=0.7, top_p=0.8,
                  top_k=20, min_p=0.0, num_beams=1, repetition_penalty=1.0,
                  no_repeat_ngram_size=0, seed=14, max_new_tokens=768,
                  context_tokens=4096, dtype='bfloat16', backend='transformers-cuda')
SOURCE_NAMES = {'scripts/aws_eval/public_evidence_experiment.py',
    'scripts/aws_eval/public_evidence_worker.py', 'scripts/aws_eval/grounded_hf_lifecycle.py',
    'scripts/aws_eval/grounded_hf_worker.py', 'scripts/aws_eval/grounded_hf_bootstrap.sh',
    'scripts/aws_eval/fresh_eight_launch.py', 'scripts/train_open_weight_lora.py',
    'scripts/aws_eval/fresh_eight_experiment.py', 'scripts/aws_eval/fresh_eight_evaluation.py',
    'scripts/evaluate_local_eight.py',
    'scripts/aws_eval/qwen_cuda_smoke.py', 'scripts/aws_eval/qwen_cpu_smoke.py',
    'trid/__init__.py', 'trid/service/__init__.py', 'trid/service/types.py',
    'trid/service/mois_graph_generation.py', 'trid/prompts.py', 'trid/text.py', 'trid/case_prompts.py',
    'scripts/evaluate_evidence_conditioning.py', 'trid/service/evidence_conditioning.py'}
GIB = 1024 ** 3
HOST_LIMITS = dict(instance_count=1, reserve_seconds=300, root_gib=120,
    rss_bytes=48 * GIB, allocator_bytes=40 * GIB, gpu_bytes=42 * GIB,
    output_bytes=512 * 1024**2, cpu_cores=6, swap_bytes=0)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def is_public(plan):
    return plan.get('schema') == SCHEMA


def safe_path(root, name):
    relative = PurePosixPath(name)
    require(isinstance(name, str) and str(relative) == name and not relative.is_absolute()
            and '..' not in relative.parts and relative.parts, 'Unsafe relative artifact path')
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root), 'Artifact escapes its root')
    require(all(not parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent),
            'Symlink artifact path forbidden')
    return path


def bound(root, record):
    require(isinstance(record, dict) and set(record) == {'path', 'bytes', 'sha256'}, 'Exact file pin required')
    path = safe_path(root, record['path'])
    require(pin(path, Path(root).resolve()) == record, 'Pinned artifact changed: ' + record['path'])
    return path


def load(root, record):
    return json.loads(bound(root, record).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')


def utc(value):
    require(isinstance(value, str) and value.endswith('Z'), 'Absolute UTC deadline required')
    result = datetime.fromisoformat(value[:-1] + '+00:00')
    require(result.utcoffset().total_seconds() == 0
        and result.isoformat(timespec='seconds').replace('+00:00', 'Z') == value,
        'Whole-second canonical UTC deadline required')
    return result


def validate_limits(plan):
    total = plan['lifetime']['total_seconds']
    require(type(total) is int and 1200 < total <= 28800, 'Explicit bounded wall time required')
    require(plan['lifetime'] == dict(total_seconds=total, termination_grace_seconds=900, automatic_extension=False),
            'Lifetime/grace/extension contract differs')
    require(plan['limits'] == dict(HOST_LIMITS, lifetime_seconds=total - 900)
            and all(type(v) is int for v in plan['limits'].values()), 'Host resource limits differ')
    require(plan['cost']['termination_grace_seconds'] == 900, 'Unpriced termination grace')
    cap = plan['cost']['cap_usd']
    require(type(cap) in (int, float) and Decimal(str(cap)).is_finite() and 0 < cap <= 30,
        'Operator budget must be a finite positive JSON number no greater than USD30')
    utc(plan['absolute_termination_deadline'])


def validate_approval(plan, approval, plan_sha, key_arn):
    expected = dict(schema=APPROVAL_SCHEMA, run_id=plan['identity']['run_id'], plan_sha256=plan_sha,
        instance_limit=1, cap_usd=str(plan['cost']['cap_usd']), workload_sha256=object_sha(plan['workload']),
        authorization_sha256=plan['authorization']['sha256'], limits=plan['limits'],
        ebs_kms_key_arn=key_arn, automatic_retry=False, automatic_extension=False,
        retraining=True, deployment=False, approved=True,
        absolute_termination_deadline=plan['absolute_termination_deadline'])
    require(approval == expected, 'Fresh exact run-bound GPU approval required')


def cost_bound(plan):
    """Price the unchanged envelope; only validated public-work comparators get 256 MiB."""
    validate_limits(plan)
    bundle_bytes_max = 128 * 1024**2
    if plan.get('schema') == 'trid-public-work-gpu-plan-v1':
        from scripts.aws_eval import public_work_experiment
        bundle_bytes_max = public_work_experiment.bundle_bytes_max(plan)
    cost, network = plan['cost'], plan['network']
    cap = Decimal(str(cost['cap_usd']))
    require(cap.is_finite() and cap > 0, 'Explicit positive operator budget required')
    require(type(cost['export_bytes']) is int and 0 < cost['export_bytes'] <= 512 * 1024**2
        and type(cost['controller_retrieval_bytes']) is int
        and cost['export_bytes'] + 121 * 1024**2 <= cost['controller_retrieval_bytes'] <= GIB
        and type(network['egress_byte_cap']) is int
        and cost['export_bytes'] + 16 * 1024**2 <= network['egress_byte_cap'] <= 2 * GIB
        and cost['network_egress_bytes'] == 3 * GIB and plan['bundle']['bytes'] <= bundle_bytes_max
        and plan['bundle']['bytes'] + cost['export_bytes'] + 8 * 1024**2 <= cost['retained_bytes'] <= 2 * GIB
        and cost['retention_days'] == 30
        and all(type(cost[k]) is int and 4608 <= cost[k] <= 10000 for k in ('get_requests', 'put_requests', 'control_requests'))
        and cost['scheduler_invocations'] == 10000 and cost['tax_rate'] == 0.10
        and math.isfinite(cost['contingency_usd']) and cost['contingency_usd'] >= 1,
        'Sealed storage/network/request/tax envelope differs')
    rates = {key: Decimal(str(value)) for key, value in cost['rates'].items()}
    require(all(v.is_finite() and v > 0 for v in rates.values()), 'Invalid observed prices')
    ancillary = Decimal(str(cost['ancillary_bound_usd']))
    require(ancillary.is_finite() and ancillary >= 0, 'Invalid ancillary bound')
    hours = Decimal(plan['lifetime']['total_seconds']) / Decimal(3600)
    usage = hours * (rates['ec2_hour'] + rates['ipv4_hour'])
    usage += Decimal(120) * rates['ebs_gb_month'] * hours / Decimal(28 * 24)
    usage += Decimal(cost['retained_bytes']) / Decimal(10**9) * rates['s3_gb_month'] * Decimal(31) / Decimal(28)
    usage += Decimal(cost['network_egress_bytes']) / Decimal(10**9) * rates['egress_gb']
    usage += cost['get_requests'] * rates['s3_get'] + cost['put_requests'] * rates['s3_put']
    usage += cost['scheduler_invocations'] * rates['scheduler_invocation'] + ancillary
    pretax = usage + Decimal(str(cost['contingency_usd']))
    total = (pretax * (1 + Decimal(str(cost['tax_rate'])))).quantize(Decimal('0.01'), rounding=ROUND_CEILING)
    require(total.is_finite() and total <= cap, 'All-in bound exceeds fresh operator budget')
    return dict(additional_pretax_bound_usd=float(pretax), all_in_bound_usd=str(total),
        current_run_cap_usd=str(cap), tax_rate=cost['tax_rate'], contingency_usd=cost['contingency_usd'],
        invoice=False, spending_hard_cap=False)


def validate_resources(root, plan, *, active=True):
    validate_limits(plan)
    auth = load(root, plan['authorization'])
    require(auth.get('schema') == AUTHORIZATION_SCHEMA and auth.get('approved') is True
        and auth.get('run_id') == plan['identity']['run_id']
        and auth.get('workload_sha256') == object_sha(plan['workload'])
        and auth.get('data_sha256') == object_sha(plan['data'])
        and auth.get('recipe_sha256') == object_sha(plan['recipe'])
        and auth.get('source_manifest_sha256') == plan['workload']['source_manifest']['sha256']
        and auth.get('execution_started') is False and auth.get('additional_gpu_executions') == 1
        and all(auth.get(k) is False for k in ('automatic_retry', 'automatic_extension', 'deployment'))
        and Decimal(str(auth['budget_usd'])) == Decimal(str(plan['cost']['cap_usd'])) > 0
        and auth.get('max_wall_seconds') == plan['lifetime']['total_seconds']
        and auth.get('absolute_deadline_utc') == plan['absolute_termination_deadline'],
        'Fresh unused run/workload/source/data/recipe/budget authorization required')
    if active:
        require(utc(auth['absolute_deadline_utc']) > datetime.now(timezone.utc), 'Authorization deadline expired')
    envelope = load(root, plan['resource_envelope'])
    expected = dict(schema=RESOURCE_ENVELOPE_SCHEMA, identity=plan['identity'],
        authorization=plan['authorization'], instance_type='g6e.2xlarge', root_gib=120,
        root_encrypted=True, root_delete_on_termination=True,
        gpu_execution_lease_seconds_max=plan['limits']['lifetime_seconds'], export_reserve_seconds=300,
        termination_grace_seconds_max=900, total_billable_lifetime_seconds_max=plan['lifetime']['total_seconds'],
        whole_host_egress_bytes_max=2 * GIB, controller_retrieval_bytes_max=GIB,
        priced_egress_bytes=3 * GIB, combined_retained_s3_bytes_max=2 * GIB,
        output_archive_bytes_max=512 * 1024**2, bundle_bytes_max=128 * 1024**2,
        s3_expiration_days=30, multipart_abort_days=1, scheduler_invocations_max=10000,
        billing_hard_cap=False, absolute_termination_deadline=plan['absolute_termination_deadline'],
        automatic_retry=False, automatic_extension=False)
    require(plan['instance_type'] == 'g6e.2xlarge' and all(envelope.get(k) == v for k, v in expected.items())
        and 'aggregate_ledger' not in envelope
        and Decimal(str(envelope['budget_usd'])) == Decimal(str(auth['budget_usd']))
        and Decimal(str(envelope['tax_assumption'])) == Decimal('0.10')
        and Decimal(str(envelope['contingency_before_tax_usd'])) == Decimal(str(plan['cost']['contingency_usd'])),
        'New resource envelope differs from approved execution')
    bound(root, envelope['cost_evidence'])
    return envelope


def validate_plan(plan, base):
    require(is_public(plan), 'Wrong public evidence plan schema')
    validate_workload(plan, {p['path']: p for p in plan['bundle']['files']})
    validate_resources(base, plan)
    return validate_data(base, plan)


def validate_workload(plan, bundled):
    require(is_public(plan) and plan['recipe'] == RECIPE, 'Separate exact 1230/615 recipe required')
    require(set(plan['phase_seconds']) == {'acquire', 'capacity', 'baseline', 'train', 'adapter'}
        and all(type(v) is int and v > 0 for v in plan['phase_seconds'].values())
        and sum(plan['phase_seconds'].values()) + 300 <= plan['limits']['lifetime_seconds'], 'Finite phase allocation differs')
    require(plan['model']['id'] == MODEL_ID and plan['model']['revision'] == MODEL_REVISION
        and plan['model']['trust_remote_code'] is False, 'Official pinned Qwen3-8B required')
    require(plan['runtime']['packages'] == RUNTIME_PACKAGES and plan['runtime']['torch_cuda'] == '12.8',
        'Pinned runtime differs')
    require(set(plan['data']) == {'manifest', 'train', 'admission'}
        and set(plan['workload']) == {'evaluation_inputs', 'source_manifest', 'acquisition_manifest', 'generation'}
        and plan['workload']['generation'] == GENERATION, 'Public workload closure differs')
    require(plan['workload']['acquisition_manifest']['sha256'] == ACQUISITION_SHA, 'Original model metadata required')
    records = [*plan['data'].values(), *(plan['workload'][k] for k in ('evaluation_inputs', 'source_manifest', 'acquisition_manifest')),
               plan['authorization'], plan['resource_envelope'], plan['ancillary_evidence']]
    require(all(bundled.get(p['path']) == p for p in records), 'Unbundled public input/evidence pin')
    require(SOURCE_NAMES == set(plan['sources']) and all(n in bundled and bundled[n]['sha256'] == sha
        for n, sha in plan['sources'].items()), 'Executable source closure differs')
    require(set(plan['output_allowlist']) == output_members(plan)
        and len(plan['output_allowlist']) == len(output_members(plan)), 'Output allowlist differs')
    require(not any(any(token in n.lower() for token in ('private', 'annotations.json', 'final-input', 'source-records'))
        for n in bundled), 'Private references/annotations/authors forbidden in cloud bundle')


def validate_public_rows(rows, count=EVALUATION_ROWS):
    require(isinstance(rows, list) and len(rows) == count, 'Wrong evaluation denominator')
    ids = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'id', 'messages'}
            and isinstance(row['id'], str) and row['id'], 'Public rows must contain only id/messages')
        messages = row['messages']
        require(isinstance(messages, list) and messages, 'Native messages required')
        has_system = isinstance(messages[0], dict) and messages[0].get('role') == 'system'
        for index, message in enumerate(messages):
            role = ('system' if has_system and index == 0 else
                    'user' if (index - int(has_system)) % 2 == 0 else 'assistant')
            require(isinstance(message, dict) and set(message) == {'role', 'content'}
                and message['role'] == role and isinstance(message['content'], str) and message['content'].strip(),
                'Invalid native public conversation framing')
        require(messages[-1]['role'] == 'user', 'Native public conversation must end with user')
        ids.append(row['id'])
    require(len(set(ids)) == len(ids), 'Duplicate public evaluation identity')
    return ids


def validate_admission(admission, train_pin, inputs_pin):
    require(admission.get('schema') == CLOUD_ADMISSION_SCHEMA
        and admission.get('profile') == PROFILE and admission.get('state') == 'ADMITTED_FOR_RESEARCH'
        and admission.get('training_ready') is True and admission.get('semantic_evaluated') is False
        and admission.get('expert_certified') is False
        and admission.get('train') == train_pin and admission.get('evaluation_inputs') == inputs_pin
        and admission.get('recipe') == RECIPE and admission.get('evaluation_rows') == EVALUATION_ROWS
        and admission.get('training_rows') == TRAINING_ROWS
        and admission.get('general_training_rows') == GENERAL_TRAINING_ROWS
        and isinstance(admission.get('reviewers'), list) and len(admission['reviewers']) == 2
        and all(isinstance(r, str) and r.strip() for r in admission['reviewers'])
        and len(set(admission['reviewers'])) == 2
        and admission.get('live_trainer_token_proof') is True, 'Authenticated research-only projection admission required')
    ids = admission.get('training_ids')
    require(isinstance(ids, list) and len(ids) == TRAINING_ROWS
        and all(isinstance(ident, str) and ident.strip() for ident in ids)
        and len(set(ids)) == TRAINING_ROWS, 'Training identity closure differs')
    require(set(admission) == {'schema', 'profile', 'state', 'training_ready', 'semantic_evaluated', 'expert_certified',
        'train', 'evaluation_inputs', 'recipe', 'evaluation_rows', 'training_rows', 'general_training_rows',
        'reviewers', 'live_trainer_token_proof', 'training_ids', 'provenance_sha256'}, 'Cloud receipt contains private/unapproved fields')
    require(set(admission['provenance_sha256']) == {'manifest', 'research_admission', 'candidate_manifest',
        'annotation_packet', 'reports', 'token_proof', 'source_pins'}, 'Admission provenance closure differs')
    hashes = [v for k, v in admission['provenance_sha256'].items() if k != 'reports'] + admission['provenance_sha256']['reports']
    require(len(admission['provenance_sha256']['reports']) == 2
        and all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in hashes),
        'Missing admission provenance hashes')


def validate_registration_bundle(root, reg, files):
    """Authenticate the exact public closure before any bundle archive is built."""
    require(isinstance(files, dict) and len(files) == len(reg['bundle']['files'])
        and files == {p['path']: p for p in reg['bundle']['files']}, 'Registration bundle pins differ')
    acquisition = load(root, reg['workload']['acquisition_manifest'])
    shell = dict(reg, schema=SCHEMA,
        sources={p['path']: p['sha256'] for p in reg['sources']},
        bundle=dict(files=list(files.values())),
        model=dict(id=MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, files=acquisition['files']),
        runtime=dict(packages=RUNTIME_PACKAGES, torch_cuda='12.8', bootstrap=dict(admission=reg['bootstrap_admission'])),
        output_allowlist=sorted(output_members()))
    require(len(shell['sources']) == len(reg['sources']), 'Duplicate executable source pins')
    for record in reg['sources']:
        bound(root, record)
        require(files.get(record['path']) == record, 'Unbundled executable source')
    return validate_data(root, shell)


def validate_data(root, plan):
    bundled = {p['path']: p for p in plan['bundle']['files']}
    require(len(bundled) == len(plan['bundle']['files']), 'Duplicate bundle artifact')
    validate_workload(plan, bundled)
    data, workload = plan['data'], plan['workload']
    manifest = load(root, data['manifest'])
    train = [json.loads(line) for line in bound(root, data['train']).read_text().splitlines()]
    inputs = load(root, workload['evaluation_inputs'])
    ids = validate_public_rows(inputs)
    admission = load(root, data['admission'])
    validate_admission(admission, data['train'], workload['evaluation_inputs'])
    require(set(manifest) == {'id', 'profile', 'rows', 'train', 'valid', 'max_tokens', 'max_answer_tokens'}
        and manifest['profile'] == PROFILE and manifest['rows'] == TRAINING_ROWS and manifest['valid'] == []
        and manifest['max_tokens'] == 4096 and manifest['max_answer_tokens'] == 768
        and len(manifest['train']) == 1, 'Minimal exactly-once training manifest differs')
    entry = manifest['train'][0]
    require(entry == dict(path=PurePosixPath(data['train']['path']).name, bytes=data['train']['bytes'],
        sha256=data['train']['sha256'], adapter='grounded_messages', epochs=1)
        and PurePosixPath(data['train']['path']).parent == PurePosixPath(data['manifest']['path']).parent,
        'Training source file/epoch differs')
    require(len(train) == TRAINING_ROWS, 'Training coverage differs')
    for row in train:
        require(isinstance(row, dict) and set(row) == {'id', 'task', 'split', 'messages', 'answer'}
            and row['split'] == 'train' and isinstance(row['task'], str) and row['task'].strip()
            and isinstance(row.get('answer'), str) and row['answer'].strip(), 'Invalid supervised training row')
        validate_public_rows([{'id': row['id'], 'messages': row['messages']}], 1)
    require([r['id'] for r in train] == admission['training_ids']
        and not set(ids).intersection(admission['training_ids']), 'Training coverage or partition overlap')
    require(sum(r['task'] == 'general_preservation' for r in train) == GENERAL_TRAINING_ROWS,
        'Actual general training exposure count differs')
    acquisition = load(root, workload['acquisition_manifest'])
    require(acquisition['repo_id'] == MODEL_ID and acquisition['revision'] == MODEL_REVISION
        and [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in acquisition['files']] ==
            [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in plan['model']['files']], 'Official model files differ')
    source_manifest = load(root, workload['source_manifest'])
    require(isinstance(source_manifest, dict) and set(source_manifest) == {'schema', 'files'}
        and source_manifest['schema'] == SOURCE_MANIFEST_SCHEMA
        and isinstance(source_manifest['files'], list), 'Exact v2 source manifest required')
    sources = source_manifest['files']
    require(len(sources) == len(plan['sources']) and {p['path'] for p in sources} == set(plan['sources']), 'Source manifest closure differs')
    for record in sources:
        bound(root, record)
        require(record['sha256'] == plan['sources'][record['path']]
            and bundled.get(record['path']) == record, 'Full source seal/bundle record differs')
    resources = load(root, plan['resource_envelope'])
    ancillary = load(root, plan['ancillary_evidence'])
    bootstrap_pin = plan['runtime']['bootstrap']['admission']
    bootstrap = load(root, bootstrap_pin)
    nested = [resources['cost_evidence'], ancillary['bucket_policy_status_evidence'], ancillary['ebs_kms_key_evidence'], *bootstrap['evidence']]
    for record in nested:
        bound(root, record)
    direct = [*data.values(), *(workload[k] for k in ('evaluation_inputs', 'source_manifest', 'acquisition_manifest')),
        plan['authorization'], plan['resource_envelope'], plan['ancillary_evidence'], bootstrap_pin, *nested]
    require({p['path'] for p in plan['bundle']['files']} == set(plan['sources']) | {p['path'] for p in direct},
        'Bundle contains an unapproved side channel')
    return dict(manifest=manifest, rows=train, evaluation_inputs=inputs, sources=sources, admission=admission)


def prepare_projection(admitted_manifest, output, *, root, sources, acquisition_manifest):
    """Authenticate the full LOCAL admission, then write only public execution inputs.

    No authorization, resource observation or cloud-ready plan is fabricated here.
    Caller-supplied source paths are pinned after private admission succeeds.
    """
    from scripts import prepare_evidence_conditioning as preparation
    root, source = Path(root).resolve(), Path(admitted_manifest).resolve()
    output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'New repository-local projection directory required')
    manifest = read_json(source)
    require(manifest.get('schema') == 'trid-evidence-admitted-manifest-v1'
        and manifest.get('profile') == PROFILE and manifest.get('state') == 'ADMITTED_FOR_RESEARCH'
        and manifest.get('training_ready') is True and manifest.get('new_annotations_independently_reviewed') is True,
        'Candidate/unreviewed corpus cannot be projected')
    artifacts = manifest['artifacts']
    required = {'train.jsonl', 'training-metadata.json', 'evaluation-inputs.json',
        'evaluation-references.private.json', 'source-records.private.json', 'derived-matrix.private.json',
        'annotation-review-packet.private.json', 'token-budget.json', 'training-manifest.json',
        'evaluation-manifest.json', 'research-admission.json', 'admission-token-budget.json'}
    require(required <= set(artifacts), 'Full admitted artifact closure required')
    for name, record in artifacts.items():
        require(name == record['path'] and PurePosixPath(name).name == name, 'Admission artifact must be local')
        bound(source.parent, record)
    receipt = load(source.parent, manifest['research_admission'])
    candidate = load(root, manifest['candidate_manifest'])
    require(receipt['prepared_manifest'] == manifest['candidate_manifest']
        and receipt['schema'] == 'trid-evidence-research-admission-v1'
        and receipt['profile'] == PROFILE and receipt['all_annotation_items_passed'] is True
        and receipt['deterministic_derivation_contract'] == 'pass'
        and receipt['expert_certified'] is False, 'Research admission lineage differs')
    require(candidate['schema'] == 'trid-evidence-preparation-manifest-v1'
        and candidate['source_pins'] == manifest['source_pins'] and candidate['annotations'] == manifest['annotations'],
        'Admitted and prepared source pins differ')
    for name, record in candidate['artifacts'].items():
        if name not in ('training-manifest.json', 'evaluation-manifest.json'):
            require(artifacts.get(name) == record, 'Admitted artifact changed after reviewed preparation')
    bound(root, manifest['builder'])
    annotations = load(root, manifest['annotations'])
    original = bound(root, manifest['source_pins']['source_frozen'])
    originals, source_pins, _ = preparation.load_sources(original, root)
    require(source_pins == manifest['source_pins'], 'Current original source admission changed')
    matrix, groups, review_items = preparation.build_candidate(originals, annotations)
    train, metadata = preparation.training_view(matrix)
    require(read_json(source.parent / 'source-records.private.json') == originals
        and read_json(source.parent / 'derived-matrix.private.json') == {'cases': matrix, 'groups': groups}
        and read_json(source.parent / 'training-metadata.json') == metadata, 'Derived source/partition audit changed')
    actual_train = [json.loads(line) for line in bound(source.parent, artifacts['train.jsonl']).read_text().splitlines()]
    inputs = load(source.parent, artifacts['evaluation-inputs.json'])
    development = [m for m in matrix if m['reference']['split'] == 'development']
    require(actual_train == train and inputs == [{'id': m['id'], 'messages': m['messages']} for m in development]
        and load(source.parent, artifacts['evaluation-references.private.json']) == [m['reference'] for m in development],
        'Admitted public/train/reference projections differ')
    validate_public_rows(inputs)
    require(len(train) == TRAINING_ROWS and len({r['id'] for r in train}) == TRAINING_ROWS
        and all(isinstance(r['id'], str) and r['id'].strip() for r in train)
        and not {r['id'] for r in train}.intersection(r['id'] for r in inputs), '1230/708 disjoint denominator required')
    packet = load(source.parent, receipt['annotation_packet'])
    require(packet['items'] == review_items and packet['annotations'] == manifest['annotations']
        and packet['author_identity'] == annotations['author_identity'], 'Reviewed annotations changed')
    reports = [load(source.parent, p) for p in receipt['reports']]
    reviewers = preparation.validate_annotation_reviews(packet, receipt['annotation_packet']['sha256'], reports,
        author_identity=receipt['author_identity'])
    require(reviewers == receipt['reviewers'], 'Review identity binding differs')
    proof = load(source.parent, receipt['live_tokenizer_and_trainer_proof'])
    stored = load(source.parent, receipt['token_budget'])
    preparation.validate_recomputed_token_proof(stored, proof)
    require(proof['context_tokens'] == 4096 and proof['max_answer_tokens'] == 768
        and proof['truncation'] is False and proof['enable_thinking'] is False and proof['failed_ids'] == [],
        'Complete native masking/token budget proof required')
    require({p['id'] for p in proof['measurements']} == {m['id'] for m in matrix} | {r['id'] for r in train},
        'Live proof coverage differs')
    for token_pin in manifest['tokenizer_pins']:
        bound(root, token_pin)
    training = load(source.parent, artifacts['training-manifest.json'])
    evaluation = load(source.parent, artifacts['evaluation-manifest.json'])
    require(training['train'] == [{**artifacts['train.jsonl'], 'adapter': 'grounded_messages', 'epochs': 1}]
        and training['rows'] == TRAINING_ROWS and training['valid'] == []
        and evaluation['inputs'] == artifacts['evaluation-inputs.json']
        and evaluation['references'] == artifacts['evaluation-references.private.json']
        and all(v['research_admission'] == manifest['research_admission'] and v['training_ready'] is True
                for v in (training, evaluation)), 'Final admitted manifest closure differs')
    source_records = [pin(safe_path(root, name), root) for name in sorted(sources)]
    require(SOURCE_NAMES == set(sources) and len(source_records) == len(set(sources)), 'Exact runtime executable source set required')
    acquisition = pin(Path(acquisition_manifest), root)
    require(acquisition['sha256'] == ACQUISITION_SHA, 'Original authoritative model metadata required')
    general = sum(m['row_type'] == 'general' for m in metadata)
    require(general == GENERAL_TRAINING_ROWS
        and sum(r['task'] == 'general_preservation' for r in train) == GENERAL_TRAINING_ROWS,
        'Exact general replay exposure count required')
    train_pin = dict(artifacts['train.jsonl'], path=(output / 'train.jsonl').relative_to(root).as_posix())
    inputs_pin = dict(artifacts['evaluation-inputs.json'],
        path=(output / 'evaluation-inputs.json').relative_to(root).as_posix())
    public_receipt = dict(schema=CLOUD_ADMISSION_SCHEMA, profile=PROFILE,
        state='ADMITTED_FOR_RESEARCH', training_ready=True, semantic_evaluated=False, expert_certified=False,
        train=train_pin, evaluation_inputs=inputs_pin, recipe=RECIPE,
        evaluation_rows=EVALUATION_ROWS, training_rows=TRAINING_ROWS,
        general_training_rows=general, reviewers=reviewers, live_trainer_token_proof=True,
        training_ids=[r['id'] for r in train], provenance_sha256=dict(manifest=pin(source)['sha256'],
            research_admission=manifest['research_admission']['sha256'], candidate_manifest=manifest['candidate_manifest']['sha256'],
            annotation_packet=receipt['annotation_packet']['sha256'], reports=[p['sha256'] for p in receipt['reports']],
            token_proof=receipt['live_tokenizer_and_trainer_proof']['sha256'], source_pins=object_sha(source_pins)))
    validate_admission(public_receipt, train_pin, inputs_pin)
    output.mkdir(parents=True)
    (output / 'train.jsonl').write_bytes(bound(source.parent, artifacts['train.jsonl']).read_bytes())
    (output / 'evaluation-inputs.json').write_bytes(bound(source.parent, artifacts['evaluation-inputs.json']).read_bytes())
    train_pin, inputs_pin = pin(output / 'train.jsonl', root), pin(output / 'evaluation-inputs.json', root)
    minimal = dict(id='public-evidence-gpu-paired1230', profile=PROFILE, rows=TRAINING_ROWS,
        train=[dict(path='train.jsonl', bytes=train_pin['bytes'], sha256=train_pin['sha256'], adapter='grounded_messages', epochs=1)],
        valid=[], max_tokens=4096, max_answer_tokens=768)
    write_json(output / 'training-manifest.json', minimal)
    write_json(output / 'admission.json', public_receipt)
    write_json(output / 'source-manifest.json', dict(schema=SOURCE_MANIFEST_SCHEMA, files=source_records))
    result = dict(schema=OFFLINE_PROJECTION_SCHEMA, cloud_authorized=False,
        recipe=RECIPE, data=dict(manifest=pin(output / 'training-manifest.json', root), train=train_pin,
            admission=pin(output / 'admission.json', root)),
        workload=dict(evaluation_inputs=inputs_pin, source_manifest=pin(output / 'source-manifest.json', root),
            acquisition_manifest=acquisition, generation=GENERATION), sources={p['path']: p['sha256'] for p in source_records})
    write_json(output / 'execution-inputs.json', result)
    return result


def generation_binding(plan, arm, adapter=None):
    require(is_public(plan) and plan['recipe'] == RECIPE, 'Sealed v2 generation recipe required')
    require(arm in ('baseline', 'adapter'), 'Unknown assessment arm')
    require((adapter is None) == (arm == 'baseline'), 'Adapter/base identity mismatch')
    if adapter is not None:
        require(set(adapter) == {'config', 'weights', 'step'} and adapter['step'] == RECIPE['steps'],
            'Only final615 fresh adapter admitted')
    return dict(schema=GENERATION_BINDING_SCHEMA, run_id=plan['identity']['run_id'], arm=arm,
        model=dict(id=MODEL_ID, revision=MODEL_REVISION), evaluation_inputs=plan['workload']['evaluation_inputs'],
        source_manifest=plan['workload']['source_manifest'], recipe_sha256=object_sha(RECIPE), adapter=adapter)


def output_members(plan=None):
    names = {'plan.json', 'runtime.json', 'runtime.log', 'model-acquisition.json', 'source-manifest.json',
        'admission.json', 'evaluation-inputs.json', 'capacity-receipt.json', 'worker-result.json',
        'training/training.json', 'training/provenance.json', 'training/adapter/adapter_config.json',
        'training/adapter/adapter_model.safetensors', 'training/adapter/README.md'}
    names.update(phase + '.log' for phase in ('acquire', 'capacity', 'baseline', 'train', 'adapter'))
    for arm in ('baseline', 'adapter'):
        names.update(arm + '/' + name for name in ('responses.jsonl', 'prompt-preflight.json', 'loaded-runtime.json', 'completion.json'))
    return names


def output_provenance(root, plan, record):
    require(is_public(plan) and plan['recipe'] == RECIPE, 'Sealed v2 output recipe required')
    require(record['plan_sha256'] == pin(Path(root) / 'plan.json')['sha256']
        and record['source_manifest'] == plan['workload']['source_manifest'], 'Worker plan/source provenance differs')
    runtime = load(root, record['runtime'])
    require(record['runtime']['path'] == 'runtime.json' and runtime.get('name') == 'NVIDIA L40S'
        and runtime.get('count') == 1 and runtime.get('bf16') is True
        and runtime.get('packages') == plan['runtime']['packages']
        and runtime.get('cuda') == plan['runtime']['torch_cuda']
        and runtime.get('run_id') == plan['identity']['run_id']
        and runtime.get('plan_sha256') == record['plan_sha256'],
        'Actual CUDA hardware/runtime identity required')


def validate_capacity(root, plan):
    result = read_json(Path(root) / 'capacity-receipt.json')
    require(all(result.get(k) == v for k, v in dict(synthetic=True, discarded=True, microbatches=4,
        context_tokens=RECIPE['context_tokens'], answer_tokens=RECIPE['answer_tokens'],
        admitted_expanded_rows=RECIPE['rows'] * RECIPE['epochs'],
        finite_parameter_update=True, adapter_reload_equal=True, optimizer_updates=2,
        microbatches_per_update=2, optimizer_state_resident_before_second_update=True,
        gradient_checkpointing=True).items()), 'Full resident-optimizer capacity guard failed')
    require(len(result['losses']) == 4 and all(math.isfinite(v) for v in result['losses']), 'Nonfinite/incomplete capacity losses')
    require(result['max_allocated_bytes'] > 0 and result['max_reserved_bytes'] > 0, 'Missing actual CUDA allocation proof')
    output_provenance(root, plan, result)
    validate_inventory(result['parameter_inventory'])
    return result


def validate_inventory(inventory):
    for group, dtype, width in (('base', 'torch.bfloat16', 2), ('adapter', 'torch.float32', 4)):
        rows = inventory[group]
        require(rows and all(key.split('@')[0] == dtype and v['tensors'] > 0 and v['elements'] > 0
            and v['bytes'] == v['elements'] * width for key, v in rows.items()), 'Actual base/adapter precision differs')


def validate_training(root, plan):
    root = Path(root)
    report = read_json(root / 'training/training.json')
    provenance = read_json(root / 'training/provenance.json')
    output_provenance(root, plan, provenance)
    require(provenance['schema'] == ADAPTER_SCHEMA and provenance['run_id'] == plan['identity']['run_id']
        and provenance['fresh'] is True and provenance['merged'] is False
        and provenance['training']['path'] == 'training/training.json' and load(root, provenance['training']) == report,
        'Fresh final adapter provenance differs')
    require(report['status'] == 'complete' and report['steps'] == report['cumulative_steps'] == RECIPE['steps']
        and report['train_rows'] == RECIPE['rows'] * RECIPE['epochs'] and report['valid_rows'] == 0
        and report['dropped_over_max_tokens'] == report['valid_dropped'] == 0
        and report['init_adapter'] is None and report['device'] == 'cuda'
        and report['base_weights'] == dict(representation='hf-safetensors', dtype='torch.bfloat16')
        and report['manifest_sha256'] == plan['data']['manifest']['sha256']
        and report['base_model_id'] == MODEL_ID and report['base_revision'] == MODEL_REVISION,
        'Final615 numerical/data/base acceptance differs')
    for key, expected in dict(batch_size=RECIPE['batch_size'], micro_batch_size=RECIPE['micro_batch_size'],
            lr=RECIPE['lr'], seed=RECIPE['seed'], max_tokens=RECIPE['context_tokens'],
            max_answer_tokens=RECIPE['answer_tokens']).items():
        require(report[key] == expected, 'Actual training recipe differs: ' + key)
    require(all(report['lora'][k] == RECIPE[k] for k in ('rank', 'alpha', 'dropout')), 'Actual LoRA recipe differs')
    admission = read_json(root / 'admission.json')
    validate_admission(admission, plan['data']['train'], plan['workload']['evaluation_inputs'])
    coverage = report['coverage']
    require(len(coverage) == TRAINING_ROWS
        and all(type(r['source_line']) is int for r in coverage)
        and len({(r['source_line'], r['id']) for r in coverage}) == TRAINING_ROWS
        and {r['source_sha256'] for r in coverage} == {plan['data']['train']['sha256']}
        and {r['source_line']: r['id'] for r in coverage} == dict(enumerate(admission['training_ids'], 1)),
        'Exactly-once source line/id coverage failed')
    validate_inventory(provenance['parameter_inventory'])
    masking = provenance['masking']
    require(provenance['fresh_zero_b'] is True and provenance['observed_steps'] == RECIPE['steps']
        and provenance['coverage_exactly_once'] is True and masking['rows'] == TRAINING_ROWS
        and all(masking[k] is True for k in ('all_prompt_tokens_masked', 'all_answer_tokens_supervised', 'eos_supervised'))
        and 0 < masking['max_context_tokens'] <= 4096 and 0 < masking['max_answer_tokens'] <= 768,
        'Observed training initialization/masking/EOS/coverage differs')
    steps = RECIPE['steps']
    require(len(report['history']) == steps, 'Incomplete optimizer history')
    warmup = steps // 20
    def multiplier(step):
        return ((step + 1) / warmup if step < warmup else
                0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))
    for step, row in enumerate(report['history'], 1):
        require(row['step'] == step and math.isfinite(row['train_loss'])
            and math.isclose(row['applied_lr'], RECIPE['lr'] * multiplier(step - 1), rel_tol=1e-12, abs_tol=1e-15)
            and math.isclose(row['next_lr'], RECIPE['lr'] * multiplier(step), rel_tol=1e-12, abs_tol=1e-15), 'Numerical schedule/history differs')
    adapter = provenance['adapter']
    generation_binding(plan, 'adapter', adapter)
    require(adapter['config']['path'] == 'training/adapter/adapter_config.json'
        and adapter['weights']['path'] == 'training/adapter/adapter_model.safetensors', 'Candidate adapter endpoint differs')
    config = load(root, adapter['config']); bound(root, adapter['weights'])
    require(config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0
        and config['peft_type'] == 'LORA' and config['task_type'] == 'CAUSAL_LM'
        and config['bias'] == 'none' and config.get('modules_to_save') is None, 'Unmerged adapter structure differs')
    return report


def validate_outputs(root, plan, arm, *, smoke=False):
    from scripts.evaluate_evidence_conditioning import validate_generation_receipt, case_seed
    require(not smoke, 'Full GPU profile has no hidden smoke/subset assessment')
    root = Path(root)
    cases = read_json(root / 'evaluation-inputs.json')
    ids = validate_public_rows(cases)
    path = root / arm / 'responses.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == EVALUATION_ROWS and [r['id'] for r in rows] == ids, 'All708 outputs required once in admitted order')
    adapter = read_json(root / 'training/provenance.json')['adapter'] if arm == 'adapter' else None
    binding = generation_binding(plan, arm, adapter)
    loaded = read_json(root / arm / 'loaded-runtime.json')
    require(loaded['device'] == 'cuda' and loaded['dtype'] == 'torch.bfloat16'
        and loaded['frozen'] is True and loaded['adapter_merged'] is False and loaded['cpu_fallback'] is False
        and loaded['local_files_only'] is True and loaded['trust_remote_code'] is False
        and loaded['plan_sha256'] == pin(root / 'plan.json')['sha256']
        and loaded['model_files'] == [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in plan['model']['files']]
        and loaded['gpu_uuid'] and loaded['instance_id'] and loaded['adapter'] == adapter
        and loaded['adapter_reload_equal'] is (True if arm == 'adapter' else None),
        'Actual loaded model/adapter/hardware provenance differs')
    runtime = load(root, loaded['runtime'])
    require(runtime['instance_id'] == loaded['instance_id'] and runtime['plan_sha256'] == loaded['plan_sha256'],
        'Loaded runtime identity differs')
    inventory = loaded['parameter_inventory']
    if arm == 'adapter':
        validate_inventory(inventory)
    else:
        require(not inventory['adapter'] and inventory['base']
            and all(k.split('@')[0] == 'torch.bfloat16' and v['bytes'] == v['elements'] * 2
                    for k, v in inventory['base'].items()), 'Baseline has adapter or wrong base precision')
    preflight = read_json(root / arm / 'prompt-preflight.json')
    require(preflight['status'] == 'passed' and preflight['all_cases_preflighted_before_weights'] is True
        and preflight['truncation'] is False and [p['id'] for p in preflight['prompts']] == ids,
        'Full preflight before model loading required')
    for row, case, prompt in zip(rows, cases, preflight['prompts']):
        require(row.get('binding') == binding and row['prompt_sha256'] == object_sha(case['messages'])
            and row['generation_receipt'] == 'hf_native_v1' and row['generation_profile'] == 'qwen3_nonthinking_sampling'
            and row['mode'] == 'evaluation' and row['seed'] == case_seed('evaluation', row['id'])
            and row['context_limit'] == 4096 and row['context_truncated'] is False, 'Native paired generation binding differs')
        validate_generation_receipt(row)
        require(isinstance(row.get('response'), str)
            and (row['error'] is None or isinstance(row['error'], (str, dict)))
            and type(row['input_tokens']) is int and row['input_tokens'] == len(row['prompt_token_ids'])
            and type(row['output_tokens']) is int and row['output_tokens'] == len(row['output_token_ids'])
            and all(type(t) is int and t >= 0 for t in row['prompt_token_ids'] + row['output_token_ids']),
            'Native response/token/exception accounting differs')
        require(all(row['generation_config'].get(k) == v for k, v in GENERATION.items()
            if k in ('do_sample', 'temperature', 'top_p', 'top_k', 'min_p', 'num_beams', 'repetition_penalty', 'no_repeat_ngram_size', 'max_new_tokens')),
            'Actual native sampling configuration differs')
        require(prompt['prompt_sha256'] == row['prompt_sha256'] and prompt['input_tokens'] > 0
            and prompt['input_tokens'] + 768 <= 4096, 'Preflight prompt budget differs')
        if row['rendered_prompt'] is not None:
            require(row['rendered_prompt_sha256'] == hashlib.sha256(row['rendered_prompt'].encode()).hexdigest()
                and row['rendered_prompt_sha256'] == prompt['rendered_prompt_sha256']
                and row['prompt_token_ids'] == prompt['prompt_token_ids'], 'Preflight/native tokenization differs')
    completion = read_json(root / arm / 'completion.json')
    require(completion['status'] == 'complete' and completion['arm'] == arm and completion['rows'] == EVALUATION_ROWS
        and completion['quality_accepted'] is False and completion['outputs'] == pin(path, root)
        and completion['exception_rows'] == sum(r['error'] is not None for r in rows)
        and completion['length_rows'] == sum(r['finish_reason'] == 'length' for r in rows), 'Response completion accounting differs')
    return rows


def validate_terminal(plan, plan_sha, instance_id, completion):
    require(completion['public_evidence_success'] is True and completion['stop_errors'] == []
        and completion['completion_scope'] == COMPLETION_SCOPE and completion['quality_accepted'] is False
        and [s['phase'] for s in completion['stages']] == ['capacity', 'baseline', 'train', 'adapter'],
        'Public evidence execution scope incomplete')
    for phase, stage in [('runtime', completion['runtime_unit']), *[(s['phase'], s) for s in completion['stages']]]:
        require(stage['phase'] == phase and stage['state'] == 'complete'
            and stage['run_id'] == plan['identity']['run_id'] and stage['plan_sha256'] == plan_sha
            and stage['instance_id'] == instance_id and stage['unit'] == plan['identity']['run_id'] + '-' + phase
            and stage['invocation'] and stage['cleanup_verified'] is True
            and stage['stopped_state']['ActiveState'] in ('inactive', 'failed') and stage['gpu'],
            'Owned GPU phase cleanup/provenance differs')


def validate_export(root, plan):
    from scripts.evaluate_evidence_conditioning import validate_paired_generation
    root = Path(root)
    required = output_members(plan) - {n for n in output_members(plan) if n.endswith('.log') or n.endswith('/README.md')}
    require(all((root / name).is_file() and not (root / name).is_symlink() for name in required), 'Successful export missing evidence')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual <= output_members(plan) | {'export-manifest.json'}, 'Forbidden export payload')
    require(read_json(root / 'plan.json') == plan, 'Export plan differs')
    for name, record in [('admission.json', plan['data']['admission']), ('evaluation-inputs.json', plan['workload']['evaluation_inputs']),
                         ('source-manifest.json', plan['workload']['source_manifest'])]:
        require(pin(root / name)['sha256'] == record['sha256'] and (root / name).stat().st_size == record['bytes'], 'Export input seal differs')
    acquisition = read_json(root / 'model-acquisition.json')
    require(acquisition['status'] == 'authenticated' and acquisition['model_id'] == MODEL_ID
        and acquisition['revision'] == MODEL_REVISION and acquisition['files'] ==
        [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in plan['model']['files']], 'Actual model acquisition differs')
    validate_capacity(root, plan); validate_training(root, plan)
    arms = {arm: {r['id']: r for r in validate_outputs(root, plan, arm)} for arm in ('baseline', 'adapter')}
    validate_paired_generation(arms)
    worker = read_json(root / 'worker-result.json')
    require(worker['status'] == 'complete' and worker['run_id'] == plan['identity']['run_id']
        and worker['plan_sha256'] == pin(root / 'plan.json')['sha256'] and worker['assessment_answers'] == 1416
        and worker['quality_accepted'] is False, 'Incomplete worker outcome')
    stages = worker['stages']
    runtime = read_json(root / 'runtime.json')
    require([stage['phase'] for stage in stages] == ['capacity', 'baseline', 'train', 'adapter'],
        'Export lacks a complete supervised phase sequence')
    for stage in stages:
        require(stage['state'] == 'complete' and stage['cleanup_verified'] is True
            and stage['run_id'] == plan['identity']['run_id'] and stage['plan_sha256'] == worker['plan_sha256']
            and stage['instance_id'] == runtime['instance_id'] and stage['invocation']
            and stage['unit'] == plan['identity']['run_id'] + '-' + stage['phase']
            and stage['stopped_state']['ActiveState'] in ('inactive', 'failed') and stage['gpu'],
            'Export cannot claim completion before owned GPU cleanup')
    return dict(status='complete', public_evidence_success=True, training_success=True, evaluation_success=True,
        assessment_answers=1416, completion_scope=COMPLETION_SCOPE, quality_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admitted-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--acquisition-manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_projection(args.admitted_manifest, args.output, root=args.root,
        sources=args.source, acquisition_manifest=args.acquisition_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
