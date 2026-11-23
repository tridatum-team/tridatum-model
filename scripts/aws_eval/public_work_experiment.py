"""Sealed public_work_v1 GPU research contract; S3 contains public inputs only.

Projection is offline admission, never permission to provision compute. Original
suite manifests carry private-file *hashes*, not private source/reference bytes.
"""
import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.aws_eval import public_evidence_experiment as evidence
from scripts.aws_eval.fresh_eight_experiment import pin, require
from scripts.aws_eval.fresh_eight_evaluation import RUNTIME_PACKAGES, ACQUISITION_SHA

SCHEMA = 'trid-public-work-gpu-plan-v1'
APPROVAL_SCHEMA = 'trid-public-work-gpu-approval-v1'
AUTHORIZATION_SCHEMA = 'trid-public-work-gpu-authorization-v1'
RESOURCE_ENVELOPE_SCHEMA = 'trid-public-work-resource-envelope-v1'
CLOUD_ADMISSION_SCHEMA = 'trid-public-work-cloud-admission-v1'
SOURCE_MANIFEST_SCHEMA = 'trid-public-work-source-manifest-v1'
OFFLINE_PROJECTION_SCHEMA = 'trid-public-work-offline-projection-v1'
GENERATION_BINDING_SCHEMA = 'trid-public-work-generation-binding-v1'
ADAPTER_SCHEMA = 'trid-public-work-gpu-adapter-v1'
PROFILE = 'public_work_v1'
COMPLETION_SCOPE = 'public_work_training_and_paired_evaluation'
MODEL_ID = evidence.MODEL_ID
MODEL_REVISION = evidence.MODEL_REVISION
TRAINING_ROWS = 530
EVALUATION_ROWS = 194
ADMITTED_SHA = 'a69e1bac2bbcb80fc4929e01690d8ea2a9ab4f1bc2062fb5c548ddf2200cd2d8'
TRAIN_SHA = '95c44d5211f954411e17bcf89445678b8a8b68007446c2ed1e369cd223ff5af9'
SUITES = {
    'holdout': (170, 'development_holdout', '8e6e6db9fb84a8b12a957175ca220f09a32978dbbaf69c64fce67c9f990c1e2f'),
    'diagnostic': (24, 'diagnostic', 'fbd2132d7ed28d9923e4bda96f4dfa78c8ad97a0b16d39590f89910ef2779130'),
}
SUITE_MANIFEST_SHAS = {
    'holdout': 'c76942dc096c20a6f3d05bb2a5a0587e08a2feb9d04cd711aba73fda6683f1f2',
    'diagnostic': '9f3b979e5d82b0f65db16ee5ee70bd4ead79899e4f35574d2e8a938b97a6de5b',
}
RECIPE = dict(rows=530, epochs=2, steps=530, batch_size=2, micro_batch_size=1,
    lr=5e-5, rank=16, alpha=32, dropout=0.0, context_tokens=4096,
    answer_tokens=768, seed=14, checkpoint_every=0, valid_every=0)
GENERATION = dict(profile='public_work_greedy_v1', do_sample=False, num_beams=1,
    max_new_tokens=512, context_limit=4096, seed=0, seed_policy='constant',
    mode='evaluation', template_kwargs={'enable_thinking': False})
SOURCE_NAMES = evidence.SOURCE_NAMES | {
    'scripts/aws_eval/public_work_experiment.py', 'scripts/aws_eval/public_work_worker.py',
    'scripts/evaluate_public_model.py', 'scripts/evaluate_public_work.py',
    'scripts/prepare_public_work.py', 'scripts/qwen_linux_guard.py', 'trid/service/public_work.py'}
HOST_LIMITS = evidence.HOST_LIMITS
GIB = evidence.GIB
canonical = evidence.canonical
object_sha = evidence.object_sha
safe_path = evidence.safe_path
bound = evidence.bound
load = evidence.load
read_json = evidence.read_json
write_json = evidence.write_json
utc = evidence.utc
validate_limits = evidence.validate_limits
cost_bound = evidence.cost_bound
validate_public_rows = evidence.validate_public_rows
validate_inventory = evidence.validate_inventory


def is_public(plan):
    return plan.get('schema') == SCHEMA


def validate_approval(plan, approval, plan_sha, key_arn):
    expected = dict(schema=APPROVAL_SCHEMA, run_id=plan['identity']['run_id'], plan_sha256=plan_sha,
        instance_limit=1, cap_usd=str(plan['cost']['cap_usd']), workload_sha256=object_sha(plan['workload']),
        authorization_sha256=plan['authorization']['sha256'], limits=plan['limits'],
        ebs_kms_key_arn=key_arn, automatic_retry=False, automatic_extension=False,
        retraining=True, deployment=False, approved=True,
        absolute_termination_deadline=plan['absolute_termination_deadline'])
    require(approval == expected, 'Fresh exact public-work approval required')


def validate_resources(root, plan, *, active=True):
    validate_limits(plan)
    auth = load(root, plan['authorization'])
    require(auth.get('schema') == AUTHORIZATION_SCHEMA and auth.get('approved') is True
        and auth.get('run_id') == plan['identity']['run_id']
        and auth.get('workload_sha256') == object_sha(plan['workload'])
        and auth.get('data_sha256') == object_sha(plan['data'])
        and auth.get('recipe_sha256') == object_sha(RECIPE)
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
        'Public-work resource envelope differs')
    bound(root, envelope['cost_evidence'])
    return envelope


def workload_pins(workload):
    return [value for key, value in workload.items() if key != 'generation']


def validate_workload(plan, bundled):
    require(is_public(plan) and plan['recipe'] == RECIPE, 'Separate exact 530-row two-epoch recipe required')
    require(set(plan['phase_seconds']) == {'acquire', 'baseline', 'train', 'adapter'}
        and all(type(v) is int and v > 0 for v in plan['phase_seconds'].values())
        and sum(plan['phase_seconds'].values()) + 300 <= plan['limits']['lifetime_seconds'], 'Finite phase allocation differs')
    require(plan['model']['id'] == MODEL_ID and plan['model']['revision'] == MODEL_REVISION
        and plan['model']['trust_remote_code'] is False, 'Official pinned Qwen3-8B required')
    require(plan['runtime']['packages'] == RUNTIME_PACKAGES and plan['runtime']['torch_cuda'] == '12.8', 'Pinned runtime differs')
    require(set(plan['data']) == {'manifest', 'train', 'admission', 'admitted_manifest'}
        and set(plan['workload']) == {'evaluation_inputs', 'source_manifest', 'acquisition_manifest', 'generation',
            'holdout_inputs', 'holdout_manifest', 'diagnostic_inputs', 'diagnostic_manifest'}
        and plan['workload']['generation'] == GENERATION, 'Public-work workload closure differs')
    require(plan['data']['admitted_manifest']['sha256'] == ADMITTED_SHA
        and plan['data']['train']['sha256'] == TRAIN_SHA, 'Original admitted corpus changed')
    require(all(plan['workload'][name + '_manifest']['sha256'] == sha
        for name, sha in SUITE_MANIFEST_SHAS.items()), 'Original public suite manifest changed')
    require(plan['workload']['acquisition_manifest']['sha256'] == ACQUISITION_SHA, 'Original model metadata required')
    records = [*plan['data'].values(), *workload_pins(plan['workload']),
        plan['authorization'], plan['resource_envelope'], plan['ancillary_evidence']]
    require(all(bundled.get(p['path']) == p for p in records), 'Unbundled public input/evidence pin')
    require(SOURCE_NAMES == set(plan['sources']) and all(n in bundled and bundled[n]['sha256'] == sha
        for n, sha in plan['sources'].items()), 'Executable source closure differs')
    require(set(plan['output_allowlist']) == output_members(plan)
        and len(plan['output_allowlist']) == len(output_members(plan)), 'Output allowlist differs')
    require(not any(any(token in n.lower() for token in ('private', 'references.', 'source-records', 'final-input'))
        for n in bundled), 'Private references/sources/targets forbidden in bundle')


def validate_admitted(manifest, train_record):
    require(manifest.get('schema') == 'trid-public-work-training-manifest-v1'
        and manifest.get('profile') == PROFILE and manifest.get('state') == 'ADMITTED_FOR_RESEARCH'
        and manifest.get('training_ready') is True and manifest.get('rows') == TRAINING_ROWS
        and manifest.get('valid') == [] and manifest.get('semantic_evaluated') is False
        and manifest.get('expert_certified') is False, 'Admitted research-only 530-row manifest required')
    require(manifest.get('train') == [dict(path='train.jsonl', bytes=train_record['bytes'],
        sha256=train_record['sha256'], adapter='grounded_messages', epochs=1, split='train')], 'Admitted training file differs')
    reviews = manifest.get('reviews')
    require(isinstance(reviews, list) and len(reviews) == 2
        and len({r.get('sha256') for r in reviews}) == 2
        and all(set(r) == {'path', 'bytes', 'sha256'} and isinstance(r['bytes'], int) and r['bytes'] > 0
            and isinstance(r['path'], str) and len(r['sha256']) == 64
            and all(c in '0123456789abcdef' for c in r['sha256']) for r in reviews), 'Two distinct review pins required')


def validate_suite(manifest, rows, name, inputs_pin):
    count, scope, sha = SUITES[name]
    require(set(manifest) == {'case_count', 'expected_tasks', 'expert_reviewed', 'historically_unseen',
        'inputs', 'profile', 'quality_accepted', 'references', 'schema', 'scope', 'source', 'training_ready'},
        'Private/unapproved suite metadata fields forbidden')
    require(manifest.get('schema') == 'trid-public-work-suite-v1' and manifest.get('profile') == PROFILE
        and manifest.get('case_count') == count and manifest.get('scope') == scope
        and manifest.get('training_ready') is False and inputs_pin['sha256'] == sha
        and manifest['inputs'] == dict(inputs_pin, path='inputs.json'), 'Pinned public suite differs')
    return validate_public_rows(rows, count)


def minimal_manifest(train_pin):
    return dict(id='public-work-gpu-530x2', profile=PROFILE, rows=530,
        train=[dict(path='train.jsonl', bytes=train_pin['bytes'], sha256=train_pin['sha256'],
                    adapter='grounded_messages', epochs=2)], valid=[], max_tokens=4096, max_answer_tokens=768)


def validate_admission(admission, data, workload, rows, admitted):
    expected = dict(schema=CLOUD_ADMISSION_SCHEMA, profile=PROFILE, state='ADMITTED_FOR_RESEARCH',
        training_ready=True, semantic_evaluated=False, expert_certified=False,
        admitted_manifest=data['admitted_manifest'], train=data['train'], recipe=RECIPE,
        evaluation_inputs=workload['evaluation_inputs'], training_rows=530, evaluation_rows=194,
        training_ids=[r['id'] for r in rows], review_sha256=[r['sha256'] for r in admitted['reviews']])
    require(admission == expected, 'Public admission/provenance differs')


def validate_registration_bundle(root, reg, files):
    require(isinstance(files, dict) and len(files) == len(reg['bundle']['files'])
        and files == {p['path']: p for p in reg['bundle']['files']}, 'Registration bundle pins differ')
    acquisition = load(root, reg['workload']['acquisition_manifest'])
    shell = dict(reg, schema=SCHEMA, sources={p['path']: p['sha256'] for p in reg['sources']},
        bundle=dict(files=list(files.values())),
        model=dict(id=MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, files=acquisition['files']),
        runtime=dict(packages=RUNTIME_PACKAGES, torch_cuda='12.8', bootstrap=dict(admission=reg['bootstrap_admission'])),
        output_allowlist=sorted(output_members()))
    require(len(shell['sources']) == len(reg['sources']), 'Duplicate source pins')
    return validate_data(root, shell)


def validate_data(root, plan):
    bundled = {p['path']: p for p in plan['bundle']['files']}
    require(len(bundled) == len(plan['bundle']['files']), 'Duplicate bundle artifact')
    validate_workload(plan, bundled)
    data, workload = plan['data'], plan['workload']
    admitted = load(root, data['admitted_manifest'])
    validate_admitted(admitted, data['train'])
    manifest = load(root, data['manifest'])
    require(manifest == minimal_manifest(data['train'])
        and PurePosixPath(data['train']['path']).parent == PurePosixPath(data['manifest']['path']).parent,
        'Two-epoch training projection differs')
    rows = [json.loads(line) for line in bound(root, data['train']).read_text().splitlines()]
    require(len(rows) == 530, 'Wrong training row count')
    for row in rows:
        require(set(row) == {'id', 'task', 'split', 'messages', 'answer'} and row['split'] == 'train'
            and isinstance(row['task'], str) and row['task'].strip()
            and isinstance(row['answer'], str) and row['answer'].strip(), 'Invalid supervised training row')
        validate_public_rows([{'id': row['id'], 'messages': row['messages']}], 1)
    require(len({r['id'] for r in rows}) == 530, 'Duplicate training IDs')
    combined, suites = [], {}
    for name in SUITES:
        suite = load(root, workload[name + '_manifest'])
        inputs = load(root, workload[name + '_inputs'])
        validate_suite(suite, inputs, name, workload[name + '_inputs'])
        combined.extend(inputs)
        suites[name] = dict(inputs=workload[name + '_inputs'], suite_manifest=workload[name + '_manifest'], source=suite['source'])
    inputs = load(root, workload['evaluation_inputs'])
    require(inputs == combined, 'Combined public evaluation differs')
    ids = validate_public_rows(inputs, 194)
    require(not set(ids).intersection(r['id'] for r in rows), 'Training/evaluation overlap')
    admission = load(root, data['admission'])
    validate_admission(admission, data, workload, rows, admitted)
    acquisition = load(root, workload['acquisition_manifest'])
    require(acquisition['repo_id'] == MODEL_ID and acquisition['revision'] == MODEL_REVISION
        and [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in acquisition['files']] ==
            [{k: p[k] for k in ('path', 'bytes', 'sha256')} for p in plan['model']['files']], 'Official model files differ')
    source_manifest = load(root, workload['source_manifest'])
    require(set(source_manifest) == {'schema', 'files'} and source_manifest['schema'] == SOURCE_MANIFEST_SCHEMA,
        'Exact source manifest required')
    sources = source_manifest['files']
    require(len(sources) == len(plan['sources']) and {p['path'] for p in sources} == SOURCE_NAMES, 'Source closure differs')
    for record in sources:
        bound(root, record)
        require(record['sha256'] == plan['sources'][record['path']] and bundled.get(record['path']) == record,
            'Full source seal/bundle record differs')
    resources, ancillary = load(root, plan['resource_envelope']), load(root, plan['ancillary_evidence'])
    bootstrap_pin = plan['runtime']['bootstrap']['admission']
    bootstrap = load(root, bootstrap_pin)
    nested = [resources['cost_evidence'], ancillary['bucket_policy_status_evidence'], ancillary['ebs_kms_key_evidence'], *bootstrap['evidence']]
    for record in nested:
        bound(root, record)
    direct = [*data.values(), *workload_pins(workload), plan['authorization'], plan['resource_envelope'],
        plan['ancillary_evidence'], bootstrap_pin, *nested]
    require(set(bundled) == SOURCE_NAMES | {p['path'] for p in direct}, 'Bundle contains an unapproved side channel')
    return dict(manifest=manifest, rows=rows, evaluation_inputs=inputs, suites=suites, sources=sources, admission=admission)


def validate_plan(plan, base):
    require(is_public(plan), 'Wrong public-work plan schema')
    validate_resources(base, plan)
    return validate_data(base, plan)


def prepare_projection(admitted_manifest, output, *, root, sources, acquisition_manifest,
                       holdout_manifest, diagnostic_manifest):
    root, source, output = Path(root).resolve(), Path(admitted_manifest).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'New repository-local projection required')
    require(pin(source)['sha256'] == ADMITTED_SHA, 'Original admitted manifest changed')
    admitted = read_json(source)
    train_record = {k: admitted['train'][0][k] for k in ('path', 'bytes', 'sha256')}
    validate_admitted(admitted, train_record)
    train_file = bound(source.parent, train_record)
    rows = [json.loads(line) for line in train_file.read_text().splitlines()]
    require(len(rows) == TRAINING_ROWS and len({r['id'] for r in rows}) == TRAINING_ROWS, 'Wrong training coverage')
    for record in admitted['reviews']:
        review = Path(record['path'])
        require(review.resolve().is_relative_to(root) and not review.is_symlink()
            and pin(review)['sha256'] == record['sha256'] and review.stat().st_size == record['bytes'], 'Review pin changed')
    public = {}
    combined = []
    for name, location in [('holdout', holdout_manifest), ('diagnostic', diagnostic_manifest)]:
        location = Path(location).resolve()
        require(pin(location)['sha256'] == SUITE_MANIFEST_SHAS[name], 'Original public suite manifest changed')
        manifest = read_json(location)
        input_path = bound(location.parent, manifest['inputs'])
        inputs = read_json(input_path)
        validate_suite(manifest, inputs, name, manifest['inputs'])
        public[name] = (location, input_path)
        combined.extend(inputs)
    ids = validate_public_rows(combined, EVALUATION_ROWS)
    require(not set(ids).intersection(r['id'] for r in rows), 'Training/evaluation overlap')
    require(set(sources) == SOURCE_NAMES and len(sources) == len(SOURCE_NAMES), 'Exact executable source set required')
    source_records = [pin(safe_path(root, name), root) for name in sorted(sources)]
    acquisition = pin(Path(acquisition_manifest).resolve(), root)
    require(acquisition['sha256'] == ACQUISITION_SHA, 'Original model metadata required')
    output.mkdir(parents=True)
    (output / 'admitted-manifest.json').write_bytes(source.read_bytes())
    (output / 'train.jsonl').write_bytes(train_file.read_bytes())
    train_pin = pin(output / 'train.jsonl', root)
    write_json(output / 'training-manifest.json', minimal_manifest(train_pin))
    write_json(output / 'evaluation-inputs.json', combined)
    write_json(output / 'source-manifest.json', dict(schema=SOURCE_MANIFEST_SCHEMA, files=source_records))
    workload = dict(evaluation_inputs=pin(output / 'evaluation-inputs.json', root),
        source_manifest=pin(output / 'source-manifest.json', root), acquisition_manifest=acquisition, generation=GENERATION)
    for name, (manifest_path, inputs_path) in public.items():
        directory = output / name
        directory.mkdir()
        (directory / 'manifest.json').write_bytes(manifest_path.read_bytes())
        (directory / 'inputs.json').write_bytes(inputs_path.read_bytes())
        workload[name + '_manifest'] = pin(directory / 'manifest.json', root)
        workload[name + '_inputs'] = pin(directory / 'inputs.json', root)
    data = dict(manifest=pin(output / 'training-manifest.json', root), train=train_pin,
        admitted_manifest=pin(output / 'admitted-manifest.json', root))
    admission = dict(schema=CLOUD_ADMISSION_SCHEMA, profile=PROFILE, state='ADMITTED_FOR_RESEARCH',
        training_ready=True, semantic_evaluated=False, expert_certified=False,
        admitted_manifest=data['admitted_manifest'], train=train_pin, recipe=RECIPE,
        evaluation_inputs=workload['evaluation_inputs'], training_rows=530, evaluation_rows=194,
        training_ids=[r['id'] for r in rows], review_sha256=[r['sha256'] for r in admitted['reviews']])
    write_json(output / 'admission.json', admission)
    data['admission'] = pin(output / 'admission.json', root)
    result = dict(schema=OFFLINE_PROJECTION_SCHEMA, cloud_authorized=False, recipe=RECIPE, data=data,
        workload=workload, sources={p['path']: p['sha256'] for p in source_records})
    write_json(output / 'execution-inputs.json', result)
    return result


def generation_binding(plan, arm, adapter=None):
    require(is_public(plan) and plan['recipe'] == RECIPE and arm in ('baseline', 'adapter'), 'Sealed generation recipe required')
    require((adapter is None) == (arm == 'baseline'), 'Adapter/base identity mismatch')
    if adapter is not None:
        require(set(adapter) == {'config', 'weights', 'step'} and adapter['step'] == 530, 'Only final530 adapter admitted')
    return dict(schema=GENERATION_BINDING_SCHEMA, run_id=plan['identity']['run_id'], arm=arm,
        model=dict(id=MODEL_ID, revision=MODEL_REVISION), evaluation_inputs=plan['workload']['evaluation_inputs'],
        source_manifest=plan['workload']['source_manifest'], recipe_sha256=object_sha(RECIPE), adapter=adapter)


def output_members(plan=None):
    names = {'plan.json', 'runtime.json', 'runtime.log', 'model-acquisition.json', 'source-manifest.json',
        'admission.json', 'evaluation-inputs.json', 'worker-result.json', 'training/training.json',
        'training/provenance.json', 'training/adapter/adapter_config.json',
        'training/adapter/adapter_model.safetensors', 'training/adapter/README.md'}
    names.update(phase + '.log' for phase in ('acquire', 'baseline', 'train', 'adapter'))
    for name in SUITES:
        names.update('suites/' + name + '/' + filename for filename in ('manifest.json', 'inputs.json'))
    names.update('tokenizer/' + name for name in ('tokenizer.json', 'tokenizer_config.json',
        'chat_template.jinja', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json'))
    for arm in ('baseline', 'adapter'):
        names.update(arm + '/' + name for name in ('prompt-preflight.json', 'loaded-runtime.json', 'completion.json'))
        for suite in SUITES:
            names.update(arm + '/' + suite + '/' + name for name in ('responses.jsonl', 'runtime.json', 'completion.json'))
    return names


def output_provenance(root, plan, record):
    require(is_public(plan) and plan['recipe'] == RECIPE, 'Sealed output recipe required')
    require(record['plan_sha256'] == pin(Path(root) / 'plan.json')['sha256']
        and record['source_manifest'] == plan['workload']['source_manifest'], 'Worker plan/source provenance differs')
    runtime = load(root, record['runtime'])
    require(record['runtime']['path'] == 'runtime.json' and runtime.get('name') == 'NVIDIA L40S'
        and runtime.get('count') == 1 and runtime.get('bf16') is True
        and runtime.get('packages') == plan['runtime']['packages'] and runtime.get('cuda') == plan['runtime']['torch_cuda']
        and runtime.get('run_id') == plan['identity']['run_id'] and runtime.get('plan_sha256') == record['plan_sha256'],
        'Actual CUDA hardware/runtime identity required')


def validate_training(root, plan):
    from scripts.aws_eval.public_work_worker import validate_training as validator
    return validator(root, plan)


def validate_outputs(root, plan, arm, *, smoke=False):
    from scripts.aws_eval.public_work_worker import validate_outputs as validator
    return validator(root, plan, arm, smoke=smoke)


def validate_terminal(plan, plan_sha, instance_id, completion):
    require(completion['public_work_success'] is True and completion['stop_errors'] == []
        and completion['completion_scope'] == COMPLETION_SCOPE and completion['quality_accepted'] is False
        and [s['phase'] for s in completion['stages']] == ['baseline', 'train', 'adapter'], 'Public-work execution incomplete')
    for phase, stage in [('runtime', completion['runtime_unit']), *[(s['phase'], s) for s in completion['stages']]]:
        require(stage['phase'] == phase and stage['state'] == 'complete'
            and stage['run_id'] == plan['identity']['run_id'] and stage['plan_sha256'] == plan_sha
            and stage['instance_id'] == instance_id and stage['unit'] == plan['identity']['run_id'] + '-' + phase
            and stage['invocation'] and stage['cleanup_verified'] is True
            and stage['stopped_state']['ActiveState'] in ('inactive', 'failed') and stage['gpu'], 'Owned GPU cleanup differs')


def validate_export(root, plan):
    root = Path(root)
    optional = {n for n in output_members() if n.endswith('.log') or n.endswith('/README.md') or n.startswith('tokenizer/')}
    required = output_members() - optional | {'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json'}
    require(all((root / n).is_file() and not (root / n).is_symlink() for n in required), 'Successful export missing evidence')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} <= output_members() | {'export-manifest.json'},
        'Forbidden export payload')
    require(read_json(root / 'plan.json') == plan, 'Export plan differs')
    for name, record in [('admission.json', plan['data']['admission']), ('evaluation-inputs.json', plan['workload']['evaluation_inputs']),
                         ('source-manifest.json', plan['workload']['source_manifest'])]:
        require(pin(root / name)['sha256'] == record['sha256'] and (root / name).stat().st_size == record['bytes'], 'Export input seal differs')
    acquisition = read_json(root / 'model-acquisition.json')
    require(acquisition['status'] == 'authenticated' and acquisition['repo_id'] == MODEL_ID
        and acquisition['revision'] == MODEL_REVISION and acquisition['files'] ==
        [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in plan['model']['files']], 'Actual model acquisition differs')
    validate_training(root, plan)
    for arm in ('baseline', 'adapter'):
        validate_outputs(root, plan, arm)
    worker = read_json(root / 'worker-result.json')
    require(worker['status'] == 'complete' and worker['run_id'] == plan['identity']['run_id']
        and worker['plan_sha256'] == pin(root / 'plan.json')['sha256'] and worker['assessment_answers'] == 388
        and worker['quality_accepted'] is False, 'Incomplete worker outcome')
    runtime = read_json(root / 'runtime.json')
    require([s['phase'] for s in worker['stages']] == ['baseline', 'train', 'adapter'], 'Wrong supervised phase order')
    for stage in worker['stages']:
        require(stage['state'] == 'complete' and stage['cleanup_verified'] is True
            and stage['run_id'] == plan['identity']['run_id'] and stage['plan_sha256'] == worker['plan_sha256']
            and stage['instance_id'] == runtime['instance_id'] and stage['invocation']
            and stage['unit'] == plan['identity']['run_id'] + '-' + stage['phase']
            and stage['stopped_state']['ActiveState'] in ('inactive', 'failed') and stage['gpu'], 'Owned export cleanup differs')
    return dict(status='complete', public_work_success=True, training_success=True, evaluation_success=True,
        assessment_answers=388, completion_scope=COMPLETION_SCOPE, quality_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('admitted-manifest', 'output', 'root', 'acquisition-manifest', 'holdout-manifest', 'diagnostic-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', action='append', required=True)
    args = parser.parse_args(argv)
    result = prepare_projection(args.admitted_manifest, args.output, root=args.root, sources=args.source,
        acquisition_manifest=args.acquisition_manifest, holdout_manifest=args.holdout_manifest,
        diagnostic_manifest=args.diagnostic_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
