"""Permanent fresh-eight preparation; no historical run imports or invented admission.

Commands (ROOT is the repository, STATE a new run-owned directory):
  python scripts/aws_eval/fresh_eight_launch.py prepare --registration REG \
    --registration-sha256 SHA --root ROOT --state STATE
  python scripts/aws_eval/fresh_eight_launch.py prepare-prerequisites \
    --registration REG --registration-sha256 SHA --root ROOT --state STATE --execute
  python scripts/aws_eval/fresh_eight_launch.py validate --registration REG \
    --registration-sha256 SHA --root ROOT --state STATE --plan PLAN \
    --plan-sha256 PSHA --approval APPROVAL --approval-sha256 ASHA
Replace validate with stage, launch, reconcile, or owned-prerequisite-cleanup and
add --execute for live actions. Cleanup before final plan is also supported, but
only when no dispatch evidence or live owned compute exists. Every mutation has
an exclusive durable intent, successful response and readback; an ambiguous
mutation is NEVER automatically retried. A durably accepted schedule-group
deletion alone may resume bounded read-only absence reconciliation, preserving
its original intent/response; other partial cleanup requires parent recovery.

Registration JSON schema trid-fresh-eight-prerequisites-v2 requires:
  approved:true, automatic_retry:false, identity:(exact lifecycle plan identity),
  controller_arn, authorization:{path,bytes,sha256}, aggregate_ledger:(same pin),
  resource_envelope:(same pin), sources:[{path,bytes,sha256}],
  iam:{instance_role:{arn,trust,inline},scheduler_role:{arn,trust,inline}},
  network:{vpc_id,subnet_id,availability_zone,security_group_name,
           security_group_description}, bucket_lifecycle_before:(full GET response),
  retention_days:(integer 1..30), lifecycle_rule_ids:{code_prefix,output_prefix},
  bundle:{local_path,files:[{path,bytes,sha256}]}.
Names must contain this registration's run_id. Sources must include this module
and grounded_hf_lifecycle.py; all pins are root-relative regular files without
symlink components. Parent provides exact policy documents, prices, admissions,
final plan and approval; this helper never synthesizes them. Offline prepare
emits bundle-prepared.json; live preparation emits owned-prerequisites.json.
Final plan must pin observations in that receipt and the prepared bundle.
Authorization, aggregate ledger and resource envelope must also enter the final
bundle. Executable bundle closure is checked by lifecycle validators.
Registration SHA is a parent trust anchor, not a cryptographic user signature.

The aggregate ledger schema is trid-fresh-eight-aggregate-budget-v2, with exactly:
  schema, run_id, authorization:(Pin), previous_run_id,
  previous_authorization_consumed:true, prior_conditional_all_in_bound_usd:"5.89",
  current_run_cap_usd:"4.11", aggregate_budget_usd:"10.00",
  additional_gpu_executions:1, automatic_retry:false, automatic_extension:false.
The new grant is SHA-bound; the consumed historical grant cannot authorize this
execution. Resource envelope schema is trid-fresh-eight-training-resource-envelope-v2;
it binds authorization and aggregate_ledger Pins, work1800/export300/grace900/
total2700, encrypted120GiB delete-on-termination root, and unchanged finite
network/storage/request/tax allowances. cost.cap_usd is numeric 4.11.
Approval schema is trid-fresh-eight-training-approval-v2 with exactly:
  schema, run_id, plan_sha256, instance_limit:1, aggregate_budget_usd:"10.00",
  prior_conditional_all_in_bound_usd:"5.89", current_run_cap_usd:"4.11",
  aggregate_ledger_sha256, authorization_sha256, recipe, workload_sha256,
  limits, ebs_kms_key_arn, automatic_retry:false, automatic_extension:false,
  deployment:false, approved:true.
Success means authenticated CUDA capacity/training/export plus positive compute
cleanup only: training_success, never quality acceptance or CUDA evaluation.

Public-evidence training/evaluation uses the same commands with the distinct
trid-public-evidence-prerequisites-v2 registration and
trid-public-evidence-gpu-plan-v2 plan. It requires a new run-bound authorization,
approved cost/resource envelope, future absolute UTC deadline and current
prerequisite observations. No historical authorization or fixed-dollar receipt
is reused. Public projection preparation is available offline through
public_evidence_experiment.py; private references never enter the bundle.

Public-work research uses trid-public-work-prerequisites-v1 and the separate
trid-public-work-gpu-plan-v1 contract, never an evidence-profile fallback.
After launch/reconcile proves compute cleanup, retrieve --execute can explicitly
recover an interrupted controller export into state/retrieved-export/. It needs
the same registration/plan hashes and exact consumed approval, not a new launch
approval. This one-shot retrieval preserves prior retrieval intents and files.
After launch or retrieve authenticates a local export, s3-purge --execute deletes
only code/<run_id>/ and outputs/<run_id>/. It requires the same registration/plan
hashes, journals each deletion with absence readback, and preserves all bucket
lifecycle rules. Metadata-only completion cannot authorize purge. Existing
mutation intents forbid replay.

Sampling evaluation uses the same commands with a fresh registration/run/state:
  schema:trid-cuda-evaluation-prerequisites-v1,
  decoder_profile:qwen3_nonthinking_sampling,
  absolute_termination_deadline:2026-11-16T23:00:00Z.
Bind the separate overnight-gpu-authorization.json (never authorization.json
from the consumed greedy run), and the sampling resource-envelope/approval
schemas. The operator USD20 ceiling is not a new user-selected dollar cap.
The envelope is work13500 (including export300), grace900, total14400; launch
requires strictly more than14460 seconds before the absolute deadline, checked
again immediately before dispatch. Preparation/validation after this prospective
window fails closed; historical evaluation audit uses active=False only.
Sampling reconciliation at/after the immutable compute cutoff has a separate
300-second exact-owned termination/absence recovery window. Its receipt marks
post_deadline_recovery; it is not evidence of termination within the priced
four-hour assumption. Only positively verified instance/root-volume cleanup
opens a separate900-second schedule-cleanup/evidence window on the same client,
without resetting any request or retrieval-byte counters or renewing compute.

Permissions: controller needs STS GetCallerIdentity; IAM Get/List role/profile,
CreateRole/PutRolePolicy/CreateInstanceProfile/AddRoleToInstanceProfile and exact
inverse cleanup actions; EC2 Describe*, Create/DeleteSecurityGroup; Scheduler
Get/List/Tag-at-create/Create/DeleteScheduleGroup; existing-bucket S3
Get/PutLifecycleConfiguration and HeadBucket. Stage additionally needs exact
code-prefix ListBucket/PutObject/GetObject; launch/reconcile require existing
lifecycle permissions (RunInstances, PassRole, schedule/termination and evidence
operations). No CreateBucket/DeleteBucket permissions are needed. Only the
explicit public-work s3-purge additionally needs exact-prefix DeleteObject,
GetBucketVersioning, ListBucketMultipartUploads, GetObject (HEAD), ListBucket,
and GetLifecycleConfiguration readbacks. Retrieval needs exact output-prefix
GetObject plus the existing compute-cleanup read/termination-schedule permissions.
All creation tags are the registered exact ownership tags. No managed IAM policy
is attached. IAM trust and permissions are validated against lifecycle rules.

S3 lifecycle GET/PUT has no compare-and-swap. We compare the entire authorized
snapshot, reread immediately before PUT and compare readback. Observed drift
fails closed; an undetectable concurrent write remains an unavoidable race.
Only two exact fresh-prefix expiration/multipart rules are appended, preserving
all historical rules and other configuration. Rules remain after cleanup, with
<=30-day expiration and a <=31-day storage billing envelope in the final plan.
"""
import argparse
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time

if __package__:
    from . import grounded_hf_lifecycle as lifecycle
else:
    import grounded_hf_lifecycle as lifecycle

require = lifecycle.require
SCHEMA = 'trid-fresh-eight-prerequisites-v2'
EVALUATION_SCHEMA = 'trid-cuda-evaluation-prerequisites-v1'
PUBLIC_EVIDENCE_SCHEMA = 'trid-public-evidence-prerequisites-v2'
PUBLIC_WORK_SCHEMA = 'trid-public-work-prerequisites-v1'
HELPERS = {'scripts/aws_eval/fresh_eight_launch.py', 'scripts/aws_eval/grounded_hf_lifecycle.py'}
ROLE_KEYS = ('instance_role', 'scheduler_role')


def safe_path(root, name):
    lifecycle.relative(name)
    root = Path(root).absolute()
    path = root / name
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink path rejected: ' + name)
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes root')
    return path


def pin_bytes(root, pin):
    path = safe_path(root, pin['path'])
    require(path.is_file() and path.stat().st_size == pin['bytes'], 'Pinned file size differs: ' + pin['path'])
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Pinned bytes differ: ' + pin['path'])
    return raw


def bound_json(root, pin):
    return json.loads(pin_bytes(root, pin))


def record(path, value):
    """Exclusive and fsynced, including directory entry, before any mutation."""
    path = Path(path)
    raw = lifecycle.canonical(value) + b'\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return value


def load(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing regular receipt: ' + str(path))
    return json.loads(path.read_bytes())


def tags(identity):
    return [{'Key': key, 'Value': value} for key, value in sorted(identity['tags'].items())]


def verify_tags(rows, identity):
    require(len(rows) == len(identity['tags']) and {r['Key']: r['Value'] for r in rows} == identity['tags'],
            'Exact fresh resource tags differ')


def evaluation_registration(reg):
    return reg.get('schema') == EVALUATION_SCHEMA


def public_evidence_registration(reg):
    return reg.get('schema') == PUBLIC_EVIDENCE_SCHEMA


def public_work_registration(reg):
    return reg.get('schema') == PUBLIC_WORK_SCHEMA


def public_contract_registration(reg):
    return public_evidence_registration(reg) or public_work_registration(reg)


def registration_plan_schema(reg):
    require(reg.get('schema') in (SCHEMA, EVALUATION_SCHEMA, PUBLIC_EVIDENCE_SCHEMA, PUBLIC_WORK_SCHEMA),
            'Unknown prerequisite registration schema; generic fallback forbidden')
    if public_work_registration(reg):
        return lifecycle.PUBLIC_WORK_SCHEMA
    if public_evidence_registration(reg):
        return lifecycle.PUBLIC_EVIDENCE_SCHEMA
    return lifecycle.EVALUATION_SCHEMA if evaluation_registration(reg) else lifecycle.FRESH_SCHEMA


def validate_lineage(root, pin):
    """Only the controller opens the original full recovery evidence closure."""
    import sys
    scripts = str(Path(__file__).resolve().parents[1])
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import local_eight_candidate
    lineage = bound_json(root, pin)
    candidate = lineage['candidate']
    actual = local_eight_candidate.cuda_evaluation_lineage(
        Path(root), safe_path(root, candidate['path']), candidate['sha256'])
    require(lineage == actual, 'Staged CUDA evaluation lineage differs from authenticated final150')


def validate_registration(reg, root, *, active=True, launch_window=True):
    fields = {'schema', 'approved', 'automatic_retry', 'identity', 'controller_arn', 'authorization',
              'resource_envelope', 'aggregate_ledger', 'sources', 'iam', 'network', 'bucket_lifecycle_before',
              'retention_days', 'lifecycle_rule_ids', 'bundle'}
    evaluation = evaluation_registration(reg)
    public = public_contract_registration(reg)
    require(active or evaluation or public, 'Historical prerequisite admission requires a bounded profile')
    if public:
        fields = (fields - {'aggregate_ledger'}) | {
            'workload', 'recipe', 'limits', 'lifetime', 'cost', 'absolute_termination_deadline',
            'data', 'phase_seconds', 'ancillary_evidence', 'bootstrap_admission'}
        shell = dict(reg, schema=registration_plan_schema(reg))
        lifecycle.absolute_deadline(shell)
        if active and launch_window:
            lifecycle.validate_launch_window(shell, time.time())
    if evaluation:
        fields = (fields - {'aggregate_ledger'}) | {'lineage_binding'}
        if reg.get('decoder_profile') == lifecycle.SAMPLING_PROFILE:
            fields |= {'decoder_profile', 'absolute_termination_deadline'}
            shell = dict(schema=lifecycle.EVALUATION_SCHEMA,
                         workload={'decoder_profile': reg['decoder_profile']},
                         absolute_termination_deadline=reg['absolute_termination_deadline'])
            lifecycle.absolute_deadline(shell)
            if active and launch_window:
                lifecycle.validate_launch_window(shell, time.time())
    require(set(reg) == fields and reg['schema'] in (SCHEMA, EVALUATION_SCHEMA, PUBLIC_EVIDENCE_SCHEMA, PUBLIC_WORK_SCHEMA) and reg['approved'] is True
            and reg['automatic_retry'] is False, 'Exact parent prerequisite registration required')
    ident = reg['identity']
    run = ident['run_id']
    require(re.fullmatch('[a-z][a-z0-9-]{8,49}', run) and ident['account'] == '148761639313'
            and ident['region'] == 'ap-northeast-2' and ident['bucket'] == 'trid-training-148761639313',
            'Wrong fresh account/region/existing bucket')
    require(ident['tags'] == {'trid:eval': ident['eval_tag'], 'trid:experiment': ident['experiment'], 'trid:run': run},
            'Exact fresh ownership required')
    require(reg['controller_arn'].startswith('arn:aws:iam::' + ident['account'] + ':')
            or reg['controller_arn'].startswith('arn:aws:sts::' + ident['account'] + ':'), 'Wrong controller account')
    names = [ident[k] for k in (*ROLE_KEYS, 'instance_profile', 'schedule_group', 'schedule_name')]
    names.append(reg['network']['security_group_name'])
    require(len(set(names)) == len(names) and all(run in n and re.fullmatch('[A-Za-z0-9_-]{1,64}', n) for n in names),
            'Distinct exact fresh resource names required')
    for key in ('code_prefix', 'output_prefix'):
        lifecycle.relative(ident[key].rstrip('/'))
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*/', ident[key])
                and ident[key].endswith('/' + run + '/'), 'Exact fresh prefix required')
    require(not ident['code_prefix'].startswith(ident['output_prefix'])
            and not ident['output_prefix'].startswith(ident['code_prefix']), 'Overlapping fresh prefixes')
    require(type(reg['retention_days']) is int and 1 <= reg['retention_days'] <= 30, 'Retention exceeds bound')
    require(set(reg['network']) == {'vpc_id', 'subnet_id', 'availability_zone', 'security_group_name',
                                    'security_group_description'}, 'Exact parent network required')
    pin_bytes(root, reg['authorization'])
    if evaluation:
        pin_bytes(root, reg['lineage_binding'])
        if active:
            validate_lineage(root, reg['lineage_binding'])
    elif not public:
        pin_bytes(root, reg['aggregate_ledger'])
    envelope = bound_json(root, reg['resource_envelope'])
    pin_bytes(root, envelope['cost_evidence'])
    resource_plan = dict(reg, schema=registration_plan_schema(reg), instance_type='g6e.2xlarge')
    if public:
        lifecycle.public_contract_module(resource_plan).validate_resources(root, resource_plan, active=active and launch_window)
    else:
        validator = lifecycle.validate_evaluation_resources if evaluation else lifecycle.validate_fresh_resources
        if evaluation and reg.get('decoder_profile') == lifecycle.SAMPLING_PROFILE:
            resource_plan['workload'] = {'decoder_profile': reg['decoder_profile']}
        validator(root, resource_plan)
    require(envelope['billing_hard_cap'] is False, 'Resource budget is conditional, not a billing hard cap')
    require(reg['retention_days'] == envelope['s3_expiration_days']
            and envelope['multipart_abort_days'] == 1, 'Envelope retention differs')
    source_pins = lifecycle.pins(reg['sources'], 32 * 1024**2)
    require(HELPERS <= set(source_pins), 'Preparation helper source seals required')
    for pin in source_pins.values():
        pin_bytes(root, pin)
    for name, executing in (('scripts/aws_eval/fresh_eight_launch.py', __file__),
                            ('scripts/aws_eval/grounded_hf_lifecycle.py', lifecycle.__file__)):
        require(not active or lifecycle.digest(Path(executing)) == source_pins[name]['sha256'],
                'Executing helper differs from parent source seal: ' + name)
    require(set(reg['iam']) == set(ROLE_KEYS), 'Only two fresh roles allowed')
    for key in ROLE_KEYS:
        policy = reg['iam'][key]
        require(set(policy) == {'arn', 'trust', 'inline'} and policy['arn'] ==
                f"arn:aws:iam::{ident['account']}:role/{ident[key]}" and policy['inline'], 'Exact fresh role ARN/policies required')
        require(all(re.fullmatch(r'[\w+=,.@-]{1,128}', n) for n in policy['inline']), 'Invalid inline policy name')
    shell = dict(schema=registration_plan_schema(reg), identity=ident, iam=reg['iam'])
    lifecycle.validate_worker_policy(shell)
    lifecycle.verify_scheduler_trust(reg['iam']['scheduler_role'], ident)
    require(reg['iam']['instance_role']['trust'] == {'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}, 'EC2-only trust required')
    statements = [s for p in reg['iam']['scheduler_role']['inline'].values() for s in p['Statement']]
    require(statements == [dict(Effect='Allow', Action='ec2:TerminateInstances',
        Resource=f"arn:aws:ec2:{ident['region']}:{ident['account']}:instance/*",
        Condition={'StringEquals': {f'ec2:ResourceTag/{k}': v for k, v in ident['tags'].items()}})],
        'Scheduler policy must require every fresh ownership tag')
    lifecycle_merge(reg)
    require(set(reg['bundle']) == {'local_path', 'files'}, 'Explicit bundle files only')
    safe_path(root, reg['bundle']['local_path'])
    lifecycle.pins(reg['bundle']['files'], 256 * 1024**2)
    if public:
        files = {row['path']: row for row in reg['bundle']['files']}
        require(all(files.get(reg[key]['path']) == reg[key] for key in ('authorization', 'resource_envelope'))
                and all(files.get(name) == pin for name, pin in source_pins.items()),
                'Public evidence bundle must bind fresh authorization and current source closure')
        require(not any(name.endswith(('.tar', '.tar.gz', '.zip')) for name in files),
                'Historical archives must not enter the public evidence bundle')
        lifecycle.public_contract_module(resource_plan).validate_registration_bundle(root, reg, files)
    if evaluation:
        files = {row['path']: row for row in reg['bundle']['files']}
        require(all(files.get(reg[key]['path']) == reg[key]
                    for key in ('authorization', 'resource_envelope', 'lineage_binding'))
                and all(files.get(name) == pin for name, pin in source_pins.items()),
                'CUDA evaluation bundle must bind authorization, lineage and current source closure')
        require(not any(name.endswith(('.tar', '.tar.gz', '.zip')) for name in files),
                'Historical recovery archives must not be uploaded')
    return reg


def lifecycle_merge(reg):
    before = reg['bucket_lifecycle_before']
    require(isinstance(before, dict) and set(before) <= {'Rules', 'TransitionDefaultMinimumObjectSize'}
            and isinstance(before.get('Rules'), list), 'Full authorized lifecycle snapshot required')
    ids = reg['lifecycle_rule_ids']
    require(set(ids) == {'code_prefix', 'output_prefix'} and len(set(ids.values())) == 2
            and all(isinstance(v, str) and reg['identity']['run_id'] in v and len(v) <= 255 for v in ids.values()),
            'Two exact fresh lifecycle rule IDs required')
    require(not set(ids.values()) & {rule.get('ID') for rule in before['Rules']}, 'Lifecycle rule ID collision')
    after = copy.deepcopy(before)
    for key, rule_id in ids.items():
        after['Rules'].append(dict(ID=rule_id, Status='Enabled', Filter={'Prefix': reg['identity'][key]},
            Expiration={'Days': reg['retention_days']}, AbortIncompleteMultipartUpload={'DaysAfterInitiation': 1}))
    shell = dict(schema=lifecycle.FRESH_SCHEMA, identity=reg['identity'],
                 cost=dict(bucket_lifecycle_configuration=after, retention_days=reg['retention_days']))
    lifecycle.validate_lifecycle_configuration(shell, after)
    return after


def prepare_bundle(reg, root, state):
    """No recursive collection, symlinks, credentials, implicit pins or wall clock."""
    files = lifecycle.pins(reg['bundle']['files'], 256 * 1024**2)
    if public_contract_registration(reg):
        shell = dict(reg, schema=registration_plan_schema(reg))
        lifecycle.public_contract_module(shell).validate_registration_bundle(root, reg, files)
    target = safe_path(root, reg['bundle']['local_path'])
    require(reg['bundle']['local_path'] not in files, 'Bundle cannot contain itself')
    for name in files:
        require(not any(x in name.lower() for x in ('.env', 'credentials', '.pem', '.aws/', '.ssh/',
                    'id_rsa', 'id_ed25519', 'secret', 'token-cache')), 'Credential-like member denied')
        raw = pin_bytes(root, files[name])
        require(not re.search(rb'(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', raw),
                'Credential material denied: ' + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for name, pin in sorted(files.items()):
                raw = pin_bytes(root, pin)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(raw), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                archive.addfile(info, io.BytesIO(raw))
        stream.flush()
        os.fsync(stream.fileno())
    bundle = dict(reg['bundle'], bytes=target.stat().st_size, sha256=lifecycle.digest(target),
                  unpacked_bytes=sum(p['bytes'] for p in files.values()))
    envelope = bound_json(root, reg['resource_envelope'])
    cap = 128 * 1024**2 if evaluation_registration(reg) or public_contract_registration(reg) else 32 * 1024**2
    if public_work_registration(reg):
        cap = lifecycle.public_contract_module(shell).bundle_bytes_max(shell)
    require(bundle['bytes'] <= envelope['bundle_bytes_max'] <= cap, 'Bundle too large')
    lifecycle.authenticate_archive(target, bundle)
    return record(Path(state) / 'bundle-prepared.json', bundle)


def absent(client, service, operation, code, *args):
    try:
        client(service, operation, *args)
    except lifecycle.AWSCallError as exc:
        require(exc.code == code, 'Absence unresolved: ' + str(exc))
        return {'absent': True, 'error_code': code}
    raise RuntimeError('Resource already exists: ' + operation)


def mutate(client, state, registration_sha, name, service, operation, args, observe):
    """Existing intent forbids replay even when response/readback was lost."""
    request = dict(registration_sha256=registration_sha, service=service, operation=operation, args=list(args))
    record(Path(state) / (name + '-intent.json'), request)
    response = client(service, operation, *args)
    record(Path(state) / (name + '-response.json'), dict(request=request, response=response))
    observation = observe(response)
    record(Path(state) / (name + '-readback.json'), dict(request=request, observation=observation))
    return observation


def delete_schedule_group(client, reg, registration_sha, state, expected, *,
                          timeout=60, clock=time.monotonic, pause=time.sleep):
    """Await asynchronous absence; accepted deletion is never sent twice."""
    state = Path(state)
    name = 'cleanup-schedule-group'
    request = dict(registration_sha256=registration_sha, service='scheduler',
                   operation='delete-schedule-group', args=['--name', reg['identity']['schedule_group']])
    intent_path = state / (name + '-intent.json')
    response_path = state / (name + '-response.json')
    readback_path = state / (name + '-readback.json')

    def observe(response):
        deadline = clock() + timeout
        require(0 < timeout <= 60, 'Group absence wait exceeds bound')
        for attempt in range(31):
            try:
                group = client('scheduler', 'get-schedule-group', *request['args'])
            except lifecycle.AWSCallError as exc:
                require(exc.code == 'ResourceNotFoundException', 'Group absence unresolved: ' + str(exc))
                return dict(absent=True, error_code=exc.code)
            # Names/ARNs can be reused. Preserve the original creation identity
            # and all fields except the deletion's mutable state/timestamp.
            stable = lambda value: {k: v for k, v in value.items()
                                    if k not in ('State', 'LastModificationDate', 'ResponseMetadata')}
            require(stable(group) == stable(expected) and group.get('State') in ('ACTIVE', 'DELETING')
                    and group.get('LastModificationDate'), 'Pending group replaced/modified')
            remaining = deadline - clock()
            require(remaining > 0 and attempt < 30, 'Accepted group deletion still pending; read-only reconciliation required')
            pause(min(2, remaining))

    if intent_path.exists():
        require(load(intent_path) == request and response_path.exists(),
                'Ambiguous group deletion; never repeat mutation')
        accepted = load(response_path)
        require(set(accepted) == {'request', 'response'} and accepted['request'] == request
                and isinstance(accepted['response'], dict)
                and set(accepted['response']) <= {'ResponseMetadata'},
                'Accepted group deletion provenance differs')
        if readback_path.exists():
            require(load(readback_path) == dict(request=request,
                observation=dict(absent=True, error_code='ResourceNotFoundException')),
                'Group deletion readback differs')
        observation = observe(accepted['response'])
        if not readback_path.exists():
            record(readback_path, dict(request=request, observation=observation))
        return observation
    require(not response_path.exists() and not readback_path.exists(), 'Orphan group deletion receipt')
    require(group_snapshot(client, reg) == expected, 'Group drift before deletion')
    return mutate(client, state, registration_sha, name, 'scheduler', 'delete-schedule-group',
                  request['args'], observe)


def caller_matches(client, reg):
    caller = client('sts', 'get-caller-identity')
    require(caller['Account'] == reg['identity']['account'] and caller['Arn'] == reg['controller_arn'],
            'Parent controller identity changed')
    return caller


def role_snapshot(client, reg, key, expected_inline=None):
    name = reg['identity'][key]
    role = client('iam', 'get-role', '--role-name', name)['Role']
    verify_tags(client('iam', 'list-role-tags', '--role-name', name)['Tags'], reg['identity'])
    require(role['Arn'] == reg['iam'][key]['arn'] and role['Path'] == '/' and role.get('RoleId')
            and role.get('CreateDate') and not role.get('PermissionsBoundary'), 'Role identity differs')
    inventory = lifecycle.role_inventory(client, name)
    expected = dict(reg['iam'][key], inline=reg['iam'][key]['inline'] if expected_inline is None else expected_inline)
    require(inventory == expected, 'Role trust/policy drift')
    return dict(identity={k: role[k] for k in ('Arn', 'RoleId', 'RoleName', 'Path', 'CreateDate')}, inventory=inventory,
                tags=tags(reg['identity']))


def profile_snapshot(client, reg, attached=True):
    identity = reg['identity']
    profile = client('iam', 'get-instance-profile', '--instance-profile-name', identity['instance_profile'])['InstanceProfile']
    verify_tags(client('iam', 'list-instance-profile-tags', '--instance-profile-name', identity['instance_profile'])['Tags'], identity)
    require(profile['Arn'] == f"arn:aws:iam::{identity['account']}:instance-profile/{identity['instance_profile']}"
            and profile['Path'] == '/' and profile.get('InstanceProfileId') and profile.get('CreateDate'), 'Profile identity differs')
    require([r['Arn'] for r in profile['Roles']] == ([reg['iam']['instance_role']['arn']] if attached else []),
            'Profile membership differs')
    return {k: profile[k] for k in ('Arn', 'InstanceProfileId', 'InstanceProfileName', 'Path', 'CreateDate', 'Roles')} | {'Tags': tags(identity)}


def group_snapshot(client, reg):
    identity = reg['identity']
    group = client('scheduler', 'get-schedule-group', '--name', identity['schedule_group'])
    arn = f"arn:aws:scheduler:{identity['region']}:{identity['account']}:schedule-group/{identity['schedule_group']}"
    require(group['Arn'] == arn and group['Name'] == identity['schedule_group'] and group['State'] == 'ACTIVE'
            and group.get('CreationDate') and group.get('LastModificationDate'), 'Group identity differs')
    verify_tags(client('scheduler', 'list-tags-for-resource', '--resource-arn', arn)['Tags'], identity)
    return {k: v for k, v in group.items() if k != 'ResponseMetadata'}


def sg_snapshot(client, reg, sgid):
    rows = client('ec2', 'describe-security-groups', '--group-ids', sgid)['SecurityGroups']
    require(len(rows) == 1, 'Missing exact SG')
    row = rows[0]
    network = reg['network']
    require(row['GroupId'] == sgid and row['OwnerId'] == reg['identity']['account']
            and row['GroupName'] == network['security_group_name'] and row['VpcId'] == network['vpc_id']
            and row['Description'] == network['security_group_description'] and row['IpPermissions'] == [], 'SG identity/ingress differs')
    verify_tags(row['Tags'], reg['identity'])
    return row


def lifecycle_get(client, reg):
    return client('s3api', 'get-bucket-lifecycle-configuration', '--bucket', reg['identity']['bucket'],
                  '--expected-bucket-owner', reg['identity']['account'])


def append_lifecycle_configuration(reg, registration_sha, state, client):
    identity, after = reg['identity'], lifecycle_merge(reg)
    require(lifecycle_get(client, reg) == reg['bucket_lifecycle_before'], 'Immediate prewrite lifecycle drift')
    args = ['--bucket', identity['bucket'], '--expected-bucket-owner', identity['account'],
            '--lifecycle-configuration', json.dumps({'Rules': after['Rules']})]
    if 'TransitionDefaultMinimumObjectSize' in after:
        args += ['--transition-default-minimum-object-size', after['TransitionDefaultMinimumObjectSize']]
    observed = mutate(client, state, registration_sha, 'append-lifecycle', 's3api',
        'put-bucket-lifecycle-configuration', args, lambda response: lifecycle_get(client, reg))
    require(observed == after, 'Lifecycle readback drift; no automatic rollback')
    return observed


def prepare_prerequisites(reg, registration_sha, state, client):
    identity, network = reg['identity'], reg['network']
    caller_matches(client, reg)
    # Complete absence preflight before creating anything. No error is absence
    # except the exact service not-found error; permission/timeouts fail closed.
    for key in ROLE_KEYS:
        absent(client, 'iam', 'get-role', 'NoSuchEntity', '--role-name', identity[key])
    absent(client, 'iam', 'get-instance-profile', 'NoSuchEntity', '--instance-profile-name', identity['instance_profile'])
    absent(client, 'scheduler', 'get-schedule-group', 'ResourceNotFoundException', '--name', identity['schedule_group'])
    groups = client('ec2', 'describe-security-groups', '--filters',
        'Name=vpc-id,Values=' + network['vpc_id'], 'Name=group-name,Values=' + network['security_group_name'])
    require(groups['SecurityGroups'] == [], 'Fresh SG name already exists')
    subnets = client('ec2', 'describe-subnets', '--subnet-ids', network['subnet_id'])['Subnets']
    require(len(subnets) == 1 and subnets[0]['VpcId'] == network['vpc_id']
            and subnets[0]['AvailabilityZone'] == network['availability_zone'] and subnets[0]['State'] == 'available', 'Network differs')
    client('s3api', 'head-bucket', '--bucket', identity['bucket'], '--expected-bucket-owner', identity['account'])
    require(lifecycle_get(client, reg) == reg['bucket_lifecycle_before'], 'Authorized bucket lifecycle snapshot drift')
    record(Path(state) / 'prerequisite-preflight.json', dict(registration_sha256=registration_sha, absence=True,
                                                          subnet=subnets[0], bucket_lifecycle=reg['bucket_lifecycle_before']))
    mutate_ = lambda name, service, op, args, observe: mutate(client, state, registration_sha, name, service, op, args, observe)
    observed = {}
    for key in ROLE_KEYS:
        name = identity[key]
        absent(client, 'iam', 'get-role', 'NoSuchEntity', '--role-name', name)
        empty = mutate_('create-' + key, 'iam', 'create-role', ['--role-name', name, '--path', '/',
            '--assume-role-policy-document', json.dumps(reg['iam'][key]['trust']), '--tags', json.dumps(tags(identity))],
            lambda response, key=key: role_snapshot(client, reg, key, {}))
        created = load(Path(state) / ('create-' + key + '-response.json'))['response']['Role']
        require(all(created.get(k) == v for k, v in empty['identity'].items()), 'Role create/readback identity mismatch')
        installed = {}
        for index, (policy_name, policy) in enumerate(reg['iam'][key]['inline'].items()):
            current = role_snapshot(client, reg, key, installed)
            require(current['identity'] == empty['identity'], 'Role replaced before policy mutation')
            installed[policy_name] = policy
            observed[key] = mutate_('policy-' + key + '-' + str(index), 'iam', 'put-role-policy',
                ['--role-name', name, '--policy-name', policy_name, '--policy-document', json.dumps(policy)],
                lambda response, key=key: role_snapshot(client, reg, key, installed))
    absent(client, 'iam', 'get-instance-profile', 'NoSuchEntity', '--instance-profile-name', identity['instance_profile'])
    empty_profile = mutate_('create-instance_profile', 'iam', 'create-instance-profile',
        ['--instance-profile-name', identity['instance_profile'], '--path', '/', '--tags', json.dumps(tags(identity))],
        lambda response: profile_snapshot(client, reg, False))
    created = load(Path(state) / 'create-instance_profile-response.json')['response']['InstanceProfile']
    require(all(created.get(k) == empty_profile[k] for k in ('Arn', 'InstanceProfileId', 'CreateDate')), 'Profile create identity mismatch')
    require(profile_snapshot(client, reg, False) == empty_profile, 'Profile changed before attachment')
    observed['instance_profile'] = mutate_('attach-instance_profile', 'iam', 'add-role-to-instance-profile',
        ['--instance-profile-name', identity['instance_profile'], '--role-name', identity['instance_role']],
        lambda response: profile_snapshot(client, reg))
    absent(client, 'scheduler', 'get-schedule-group', 'ResourceNotFoundException', '--name', identity['schedule_group'])
    observed['schedule_group'] = mutate_('create-schedule_group', 'scheduler', 'create-schedule-group',
        ['--name', identity['schedule_group'], '--tags', json.dumps(tags(identity))], lambda response: group_snapshot(client, reg))
    require(load(Path(state) / 'create-schedule_group-response.json')['response']['ScheduleGroupArn'] == observed['schedule_group']['Arn'],
            'Group creation identity mismatch')
    require(client('ec2', 'describe-security-groups', '--filters',
        'Name=vpc-id,Values=' + network['vpc_id'], 'Name=group-name,Values=' + network['security_group_name'])['SecurityGroups'] == [],
        'Fresh SG appeared before creation')
    observed['security_group'] = mutate_('create-security_group', 'ec2', 'create-security-group',
        ['--group-name', network['security_group_name'], '--description', network['security_group_description'],
         '--vpc-id', network['vpc_id'], '--tag-specifications', json.dumps([dict(ResourceType='security-group', Tags=tags(identity))])],
        lambda response: sg_snapshot(client, reg, response['GroupId']))
    observed['bucket_lifecycle_configuration'] = append_lifecycle_configuration(reg, registration_sha, state, client)
    return record(Path(state) / 'owned-prerequisites.json', dict(schema='trid-fresh-eight-owned-prerequisites-v2',
        registration_sha256=registration_sha, identity=identity, resources=observed))


def validate_final(reg, registration_sha, root, state, plan, approval, plan_sha, *, active=True):
    evaluation = evaluation_registration(reg)
    public = public_contract_registration(reg)
    require(active or evaluation or public, 'Historical final-plan admission requires a bounded profile')
    require(plan.get('schema') == registration_plan_schema(reg)
            and plan['identity'] == reg['identity'] and plan['iam'] == reg['iam']
            and plan['controller_arn'] == reg['controller_arn'] and plan['authorization'] == reg['authorization']
            and plan['resource_envelope'] == reg['resource_envelope'],
            'Final plan not bound to parent registration')
    if public:
        require(all(plan[key] == reg[key] for key in
                    ('workload', 'recipe', 'limits', 'lifetime', 'cost', 'absolute_termination_deadline',
                     'data', 'phase_seconds', 'ancillary_evidence'))
                and plan['runtime']['bootstrap']['admission'] == reg['bootstrap_admission'],
                'Final public evidence workload, budget or deadline differs from registration')
    if evaluation:
        require(plan['workload']['lineage_binding'] == reg['lineage_binding'],
                'Final CUDA lineage differs from parent registration')
        require(lifecycle.sampling_mode(plan) == (reg.get('decoder_profile') == lifecycle.SAMPLING_PROFILE),
                'Final decoder profile differs from parent registration')
        if lifecycle.sampling_mode(plan):
            require(plan['absolute_termination_deadline'] == reg['absolute_termination_deadline'],
                    'Final absolute deadline differs from parent registration')
        if active:
            validate_lineage(root, reg['lineage_binding'])
    elif not public:
        require(plan['aggregate_ledger'] == reg['aggregate_ledger'], 'Final aggregate ledger differs')
    envelope = bound_json(root, reg['resource_envelope'])
    require(plan['bundle']['bytes'] <= envelope['bundle_bytes_max']
            and plan['limits']['output_bytes'] <= envelope['output_archive_bytes_max']
            and plan['network']['egress_byte_cap'] <= envelope['whole_host_egress_bytes_max']
            and plan['cost']['controller_retrieval_bytes'] <= envelope['controller_retrieval_bytes_max']
            and plan['cost']['export_bytes'] <= envelope['output_archive_bytes_max']
            and plan['cost']['network_egress_bytes'] <= envelope['priced_egress_bytes']
            and plan['cost']['tax_rate'] == float(envelope['tax_assumption'])
            and plan['cost']['contingency_usd'] == float(envelope['contingency_before_tax_usd']),
            'Final plan exceeds parent resource/cost envelope')
    require(plan['bundle'] == load(Path(state) / 'bundle-prepared.json'), 'Prepared bundle differs from final plan')
    receipt = load(Path(state) / 'owned-prerequisites.json')
    require(receipt['registration_sha256'] == registration_sha and receipt['identity'] == reg['identity'], 'Ownership registration mismatch')
    resources = receipt['resources']
    for key in ('vpc_id', 'subnet_id', 'availability_zone'):
        require(plan['network'][key] == reg['network'][key], 'Final network differs')
    require(plan['network']['security_group_observation'] == resources['security_group']
            and plan['network']['security_group_id'] == resources['security_group']['GroupId']
            and plan['cost']['bucket_lifecycle_configuration'] == resources['bucket_lifecycle_configuration'] == lifecycle_merge(reg)
            and plan['cost']['retention_days'] == reg['retention_days'], 'Final resource observations differ')
    for key in ROLE_KEYS:
        require(resources[key]['inventory'] == plan['iam'][key], 'Final IAM observations differ')
    result = lifecycle.validate_plan(plan, approval, plan_sha,
                                    now=None if active else lifecycle.timestamp(plan['checked_at']))
    if active:
        lifecycle.authenticate_archive(safe_path(root, plan['bundle']['local_path']), plan['bundle'])
    for name, sha in plan['sources'].items():
        path = safe_path(root, name)
        require(path.is_file() and lifecycle.digest(path) == sha, 'Final source pin differs: ' + name)
    lifecycle.validate_data(root, plan)
    lifecycle.validate_cost_admission(root, plan)
    lifecycle.validate_bootstrap_admission(root, plan)
    return result


def stage(reg, registration_sha, root, state, plan, plan_sha, plan_path, client):
    caller_matches(client, reg)
    lifecycle.live_preflight(plan, client)
    for name, path, sha in (('plan.json', Path(plan_path), plan_sha),
                            ('bundle.tar', safe_path(root, plan['bundle']['local_path']), plan['bundle']['sha256'])):
        require(lifecycle.digest(path) == sha, 'Staging bytes changed')
        key = reg['identity']['code_prefix'] + name
        listing = client('s3api', 'list-objects-v2', '--bucket', reg['identity']['bucket'], '--prefix', key, '--max-keys', '1')
        require(not listing.get('Contents') and not listing.get('IsTruncated'), 'Stage key already exists or ambiguous')
        observation = mutate(client, state, registration_sha, 'stage-' + name.replace('.', '-'), 's3api', 'put-object',
            ['--bucket', reg['identity']['bucket'], '--key', key, '--body', str(path), '--server-side-encryption', 'AES256',
             '--metadata', json.dumps({'sha256': sha}), '--if-none-match', '*'],
            lambda response: client('s3api', 'head-object', '--bucket', reg['identity']['bucket'], '--key', key))
        require(observation['ContentLength'] == path.stat().st_size and observation['Metadata'].get('sha256') == sha,
                'Staging readback differs')
    return {'staged': True, 'plan_sha256': plan_sha}


def reconcile(plan, plan_sha, state, client):
    state = Path(state)
    require(load(state / 'approval-consumed.json') == dict(run_id=plan['identity']['run_id'], plan_sha256=plan_sha),
            'Exact consumed approval required')
    evidence = lifecycle.load_volume_identity(state, plan, plan_sha)
    result = lifecycle.reconcile_compute(client, plan, plan_sha, state, evidence['instance_id'],
                                         evidence=evidence, evidence_path=state / 'volume-identity.json')
    lifecycle.save(state / 'reconcile-result.json', result)
    if result['cleanup_state'] == 'complete':
        lifecycle.cleanup_schedule(client, plan['identity'], state, result, plan=plan, plan_sha=plan_sha)
    return result


def verify_compute_cleanup(reg, state, client, plan=None, plan_sha=None):
    state, ident = Path(state), reg['identity']
    dispatched = any(os.path.lexists(state / name) for name in
        ('dispatch-authorization.json', 'launch-dispatched.json', 'volume-identity.json', 'approval-consumed.json',
         'remote-approval-intent.json', 'schedule-intent.json', 'release-intent.json', 'release.json'))
    if dispatched:
        require(plan is not None, 'Paid/ambiguous dispatch requires final plan and exact compute cleanup proof')
        evidence = lifecycle.load_volume_identity(state, plan, plan_sha)
        path = state / 'reconcile-result.json'
        result = load(path if path.exists() else state / 'controller-result.json')
        result = result.get('cleanup', result)
        require(result['cleanup_state'] == 'complete' and result['instance_terminated'] is True
                and result['instance_id'] == evidence['instance_id'] and evidence['root_volume_ids']
                and set(result['root_volume_ids']) == set(evidence['root_volume_ids'])
                and set(result['recorded_volume_ids']) == set(evidence['volumes'])
                and set(result['absent_volume_ids']) == set(evidence['volumes']) and not result['remaining_volumes'],
                'Exact compute/root-volume cleanup proof required')
        try:
            response = client('ec2', 'describe-instances', '--instance-ids', evidence['instance_id'])
        except lifecycle.AWSCallError as exc:
            require(exc.code == 'InvalidInstanceID.NotFound', 'Instance absence unresolved')
        else:
            rows = [r for group in response['Reservations'] for r in group['Instances']]
            require(len(rows) == 1 and lifecycle.owned(rows[0], ident, evidence['instance_id']) == evidence['instance_id']
                    and rows[0]['State']['Name'] == 'terminated', 'Exact compute still live')
        for volume_id in evidence['volumes']:
            absent(client, 'ec2', 'describe-volumes', 'InvalidVolume.NotFound', '--volume-ids', volume_id)
        lifecycle.cleanup_schedule(client, ident, state, result, plan=plan, plan_sha=plan_sha)
    else:
        lifecycle.require_no_dispatch_evidence(state)
        response = client('ec2', 'describe-instances', '--filters', 'Name=client-token,Values=' + ident['run_id'])
        require(not any(g['Instances'] for g in response['Reservations']), 'Unrecorded dispatch exists')
    rows = lifecycle.discover(client, ident)
    require(all(dispatched and row['InstanceId'] == evidence['instance_id'] and row['State']['Name'] == 'terminated'
                for row in rows), 'Unexpected owned compute remains')
    require(client('ec2', 'describe-volumes', '--filters', *lifecycle.filters(ident))['Volumes'] == [], 'Owned volumes remain')
    absent(client, 'scheduler', 'get-schedule', 'ResourceNotFoundException', '--name', ident['schedule_name'], '--group-name', ident['schedule_group'])
    try:
        schedules = client('scheduler', 'list-schedules', '--group-name', ident['schedule_group'])['Schedules']
    except lifecycle.AWSCallError as exc:
        require(exc.code == 'ResourceNotFoundException', 'Schedule inventory absence unresolved')
        absent(client, 'scheduler', 'get-schedule-group', 'ResourceNotFoundException', '--name', ident['schedule_group'])
    else:
        require(schedules == [], 'Scheduler deletion must precede prerequisite cleanup')


def verify_profile_unused(client, profile_arn):
    regions = client('ec2', 'describe-regions', '--all-regions')['Regions']
    require(regions and len({r['RegionName'] for r in regions}) == len(regions)
            and all(r['OptInStatus'] in ('opt-in-not-required', 'opted-in', 'not-opted-in') for r in regions),
            'Incomplete enabled-region inventory')
    for region in regions:
        if region['OptInStatus'] == 'not-opted-in':
            continue
        rows = client('ec2', 'describe-iam-instance-profile-associations', '--filters',
                      'Name=state,Values=associating,associated,disassociating', region=region['RegionName'])['IamInstanceProfileAssociations']
        require(not any(r.get('IamInstanceProfile', {}).get('Arn') == profile_arn for r in rows), 'Profile still associated')
        instances = client('ec2', 'describe-instances', '--filters', 'Name=iam-instance-profile.arn,Values=' + profile_arn,
                           region=region['RegionName'])['Reservations']
        require(not any(r['State']['Name'] != 'terminated' for g in instances for r in g['Instances']), 'Profile referenced by compute')


def verify_sg_unused(client, sgid):
    require(client('ec2', 'describe-network-interfaces', '--filters', 'Name=group-id,Values=' + sgid)['NetworkInterfaces'] == [],
            'SG interfaces remain')
    require(client('ec2', 'describe-security-group-references', '--group-id', sgid)['SecurityGroupReferenceSet'] == [], 'SG cross-VPC references')
    for direction in ('ip-permission', 'egress.ip-permission'):
        require(client('ec2', 'describe-security-groups', '--filters', 'Name=' + direction + '.group-id,Values=' + sgid)['SecurityGroups'] == [],
                'SG referenced by unrelated rules')


def journal_observation(state, name, registration_sha):
    """Return only successful, read-back mutations; never adopt unknown writes."""
    intent_path = Path(state) / (name + '-intent.json')
    if not intent_path.exists():
        return None
    intent = load(intent_path)
    require(intent['registration_sha256'] == registration_sha, 'Journal registration mismatch')
    response_path, readback_path = [Path(state) / (name + suffix) for suffix in ('-response.json', '-readback.json')]
    if not response_path.exists() or not readback_path.exists():
        raise RuntimeError('Ambiguous mutation preserved: ' + name)
    response, readback = load(response_path), load(readback_path)
    require(response['request'] == readback['request'] == intent, 'Mutation provenance differs')
    return readback['observation']


def partial_prerequisite_cleanup(reg, registration_sha, state, client, plan=None, plan_sha=None):
    """Clean proven successful subset after preparation failure; preserve unknowns."""
    state, identity = Path(state), reg['identity']
    verify_compute_cleanup(reg, state, client, plan, plan_sha)
    known, unresolved, deleted = {}, {}, []
    for key in (*ROLE_KEYS, 'instance_profile', 'schedule_group', 'security_group'):
        name = 'create-' + key
        try:
            observed = journal_observation(state, name, registration_sha)
            if observed is None:
                continue
            created = load(state / (name + '-response.json'))['response']
            if key in ROLE_KEYS:
                require(all(created['Role'].get(k) == v for k, v in observed['identity'].items()),
                        'Role creation identity differs')
                for index in range(len(reg['iam'][key]['inline'])):
                    latest = journal_observation(state, 'policy-' + key + '-' + str(index), registration_sha)
                    if latest is not None:
                        require(latest['identity'] == observed['identity'], 'Role policy readback identity differs')
                        observed = latest
            elif key == 'instance_profile':
                require(all(created['InstanceProfile'].get(k) == observed[k]
                            for k in ('Arn', 'InstanceProfileId', 'CreateDate')), 'Profile creation identity differs')
                latest = journal_observation(state, 'attach-instance_profile', registration_sha)
                if latest is not None:
                    require(all(latest[k] == observed[k] for k in observed if k != 'Roles'), 'Profile attach identity differs')
                    observed = latest
            elif key == 'schedule_group':
                require(created['ScheduleGroupArn'] == observed['Arn'], 'Group creation identity differs')
            else:
                require(created['GroupId'] == observed['GroupId'], 'SG creation identity differs')
            known[key] = observed
        except RuntimeError as exc:
            unresolved[key] = str(exc)
    record(state / 'owned-cleanup-started.json', dict(registration_sha256=registration_sha, resources=known,
                                                    preserved_unresolved=unresolved, partial_preparation=True))
    mutate_ = lambda name, service, op, args, observe: mutate(client, state, registration_sha, 'cleanup-' + name, service, op, args, observe)
    # One failure does not justify touching the failed resource, but independent
    # provably-created resources remain reachable. Never retry a failed mutation.
    for key in ('schedule_group', 'instance_profile', *ROLE_KEYS, 'security_group'):
        if key not in known:
            continue
        expected = known[key]
        try:
            if key == 'schedule_group':
                delete_schedule_group(client, reg, registration_sha, state, expected)
            elif key == 'instance_profile':
                attached = bool(expected['Roles'])
                require(profile_snapshot(client, reg, attached) == expected, 'Profile drift')
                verify_profile_unused(client, expected['Arn'])
                if attached:
                    require('instance_role' in known and role_snapshot(client, reg, 'instance_role',
                        known['instance_role']['inventory']['inline']) == known['instance_role'], 'Unproven attached role')
                    empty = mutate_('profile-detach', 'iam', 'remove-role-from-instance-profile',
                        ['--instance-profile-name', identity[key], '--role-name', identity['instance_role']],
                        lambda response: profile_snapshot(client, reg, False))
                    require(all(empty[k] == expected[k] for k in empty if k != 'Roles'), 'Profile replaced')
                mutate_('profile', 'iam', 'delete-instance-profile', ['--instance-profile-name', identity[key]],
                    lambda response: absent(client, 'iam', 'get-instance-profile', 'NoSuchEntity', '--instance-profile-name', identity[key]))
            elif key in ROLE_KEYS:
                policies = dict(expected['inventory']['inline'])
                require(role_snapshot(client, reg, key, policies) == expected, 'Role drift')
                require(client('iam', 'list-instance-profiles-for-role', '--role-name', identity[key])['InstanceProfiles'] == [],
                        'Role referenced; preserve dependency')
                for index, name in enumerate(list(policies)):
                    del policies[name]
                    current = mutate_('policy-' + key + '-' + str(index), 'iam', 'delete-role-policy',
                        ['--role-name', identity[key], '--policy-name', name],
                        lambda response, key=key: role_snapshot(client, reg, key, policies))
                    require(current['identity'] == expected['identity'], 'Role replaced')
                mutate_(key, 'iam', 'delete-role', ['--role-name', identity[key]],
                    lambda response, key=key: absent(client, 'iam', 'get-role', 'NoSuchEntity', '--role-name', identity[key]))
            else:
                sgid = expected['GroupId']
                require(sg_snapshot(client, reg, sgid) == expected, 'SG drift')
                verify_sg_unused(client, sgid)
                mutate_('security-group', 'ec2', 'delete-security-group', ['--group-id', sgid],
                    lambda response: absent(client, 'ec2', 'describe-security-groups', 'InvalidGroup.NotFound', '--group-ids', sgid))
            deleted.append(key)
        except (RuntimeError, OSError, TimeoutError) as exc:
            unresolved[key] = str(exc)
    return record(state / 'owned-cleanup-complete.json', dict(registration_sha256=registration_sha,
        cleanup_state='unresolved' if unresolved else 'complete', deleted_resources=deleted,
        preserved_unresolved=unresolved, retained_prefix_lifecycle_rules=True, s3_mutations=0))


def owned_prerequisite_cleanup(reg, registration_sha, state, client, plan=None, plan_sha=None):
    state, identity = Path(state), reg['identity']
    caller_matches(client, reg)
    if not (state / 'owned-prerequisites.json').exists():
        return partial_prerequisite_cleanup(reg, registration_sha, state, client, plan, plan_sha)
    receipt = load(state / 'owned-prerequisites.json')
    require(receipt['registration_sha256'] == registration_sha and receipt['identity'] == identity, 'Ownership receipt mismatch')
    expected = receipt['resources']
    # Bind successful creation and readback, including immutable IAM IDs/dates.
    for key in (*ROLE_KEYS, 'instance_profile', 'schedule_group', 'security_group'):
        created = load(state / ('create-' + key + '-response.json'))
        intent = load(state / ('create-' + key + '-intent.json'))
        readback = load(state / ('create-' + key + '-readback.json'))
        require(created['request'] == intent == readback['request'] and intent['registration_sha256'] == registration_sha,
                'Creation provenance differs')
        if key in ROLE_KEYS:
            value = created['response']['Role']
            require(all(value.get(k) == v for k, v in expected[key]['identity'].items()), 'Role immutable creation ID differs')
        elif key == 'instance_profile':
            value = created['response']['InstanceProfile']
            require(all(value.get(k) == expected[key][k] for k in ('Arn', 'InstanceProfileId', 'CreateDate')), 'Profile immutable creation ID differs')
        elif key == 'schedule_group':
            require(created['response']['ScheduleGroupArn'] == expected[key]['Arn']
                    and readback['observation'] == expected[key], 'Group creation/readback differs')
        else:
            require(created['response']['GroupId'] == expected[key]['GroupId']
                    and readback['observation'] == expected[key], 'SG creation/readback differs')
    started_path = state / 'owned-cleanup-started.json'
    resuming_group = started_path.exists()
    if resuming_group:
        require(load(started_path) == dict(registration_sha256=registration_sha, resources=expected),
                'Original cleanup authority differs')
        require({path.name for path in state.glob('cleanup-*-intent.json')}
                == {'cleanup-schedule-group-intent.json'},
                'Partial prerequisite cleanup requires explicit parent recovery')
        # A DELETING group may already reject ListSchedules. Reconcile only its
        # accepted deletion before the normal compute/dependency observations.
        delete_schedule_group(client, reg, registration_sha, state, expected['schedule_group'])
    verify_compute_cleanup(reg, state, client, plan, plan_sha)
    for key in ROLE_KEYS:
        require(role_snapshot(client, reg, key) == expected[key], 'Live role replaced/modified')
        profiles = client('iam', 'list-instance-profiles-for-role', '--role-name', identity[key])['InstanceProfiles']
        require([p['Arn'] for p in profiles] == ([expected['instance_profile']['Arn']] if key == 'instance_role' else []),
                'Unexpected role/profile reference')
    require(profile_snapshot(client, reg) == expected['instance_profile'], 'Live profile replaced/modified')
    if not resuming_group:
        require(group_snapshot(client, reg) == expected['schedule_group'], 'Live group replaced/modified')
    sgid = expected['security_group']['GroupId']
    require(sg_snapshot(client, reg, sgid) == expected['security_group'], 'Live SG replaced/modified')
    verify_profile_unused(client, expected['instance_profile']['Arn'])
    verify_sg_unused(client, sgid)
    if not resuming_group:
        record(started_path, dict(registration_sha256=registration_sha, resources=expected))
    mutate_ = lambda name, service, op, args, observe: mutate(client, state, registration_sha, 'cleanup-' + name, service, op, args, observe)
    if not resuming_group:
        delete_schedule_group(client, reg, registration_sha, state, expected['schedule_group'])
    verify_profile_unused(client, expected['instance_profile']['Arn'])
    require(profile_snapshot(client, reg) == expected['instance_profile'], 'Profile drift before detach')
    empty = mutate_('profile-detach', 'iam', 'remove-role-from-instance-profile',
        ['--instance-profile-name', identity['instance_profile'], '--role-name', identity['instance_role']],
        lambda response: profile_snapshot(client, reg, False))
    require(all(empty[k] == expected['instance_profile'][k] for k in empty if k != 'Roles'), 'Profile identity changed during detach')
    mutate_('profile', 'iam', 'delete-instance-profile', ['--instance-profile-name', identity['instance_profile']],
        lambda response: absent(client, 'iam', 'get-instance-profile', 'NoSuchEntity', '--instance-profile-name', identity['instance_profile']))
    for key in ROLE_KEYS:
        require(role_snapshot(client, reg, key) == expected[key], 'Role drift before deletion')
        require(client('iam', 'list-instance-profiles-for-role', '--role-name', identity[key])['InstanceProfiles'] == [], 'Role still referenced')
        policies = dict(reg['iam'][key]['inline'])
        for index, name in enumerate(list(policies)):
            del policies[name]
            observed = mutate_('policy-' + key + '-' + str(index), 'iam', 'delete-role-policy',
                ['--role-name', identity[key], '--policy-name', name],
                lambda response, key=key: role_snapshot(client, reg, key, policies))
            require(observed['identity'] == expected[key]['identity'], 'Role replaced during cleanup')
        mutate_(key, 'iam', 'delete-role', ['--role-name', identity[key]],
            lambda response, key=key: absent(client, 'iam', 'get-role', 'NoSuchEntity', '--role-name', identity[key]))
    require(sg_snapshot(client, reg, sgid) == expected['security_group'], 'SG drift before delete')
    verify_sg_unused(client, sgid)
    mutate_('security-group', 'ec2', 'delete-security-group', ['--group-id', sgid],
            lambda response: absent(client, 'ec2', 'describe-security-groups', 'InvalidGroup.NotFound', '--group-ids', sgid))
    return record(state / 'owned-cleanup-complete.json', dict(registration_sha256=registration_sha,
        cleanup_state='complete', retained_prefix_lifecycle_rules=True, s3_mutations=0))


def retrieve_run_export(reg, registration_sha, state, client, plan, plan_sha):
    """Recover export once without overwriting interrupted launch evidence."""
    state, identity = Path(state), reg['identity']
    require(public_work_registration(reg) and lifecycle.public_work_mode(plan)
            and plan['identity'] == identity and lifecycle.SHA.fullmatch(plan_sha or ''),
            'Retrieval requires the distinct bound public-work contract')
    require(load(state / 'approval-consumed.json') ==
            dict(run_id=identity['run_id'], plan_sha256=plan_sha),
            'Retrieval requires exact consumed approval')
    caller_matches(client, reg)
    verify_compute_cleanup(reg, state, client, plan, plan_sha)
    evidence = lifecycle.load_volume_identity(state, plan, plan_sha)
    release = load(state / 'release.json')
    require(release.get('run_id') == identity['run_id']
            and release.get('plan_sha256') == plan_sha
            and release.get('instance_id') == evidence['instance_id'],
            'Retrieval release ownership differs')
    destination = state / 'retrieved-export'
    destination.mkdir()
    record(destination / 'release.json', release)
    record(state / 'export-retrieve-intent.json',
        dict(schema='trid-public-work-export-retrieval-v1', registration_sha256=registration_sha,
             run_id=identity['run_id'], plan_sha256=plan_sha, instance_id=evidence['instance_id'],
             destination=destination.name))
    result = lifecycle.collect_export(client, plan, plan_sha, destination, evidence['instance_id'])
    return record(state / 'export-retrieved.json', result)


def purge_run_s3(reg, registration_sha, state, client, plan, plan_sha):
    """Explicit post-export purge; never alter retention or retry a mutation."""
    state, identity = Path(state), reg['identity']
    require(public_work_registration(reg) and lifecycle.public_work_mode(plan)
            and plan['identity'] == identity and lifecycle.SHA.fullmatch(plan_sha or ''),
            'S3 purge requires the distinct bound public-work contract')
    prefixes = [identity['code_prefix'], identity['output_prefix']]
    require(prefixes == ['code/' + identity['run_id'] + '/', 'outputs/' + identity['run_id'] + '/'],
            'S3 purge permits only exact run code/outputs prefixes')
    caller_matches(client, reg)
    export_state = state / 'retrieved-export'
    require(not export_state.is_symlink(), 'Symlink export state forbidden')
    if not (export_state / 'local-export-authenticated.json').exists():
        export_state = state
    authentication = load(export_state / 'local-export-authenticated.json')
    completion = load(export_state / 'completion.json')
    evidence = lifecycle.load_volume_identity(state, plan, plan_sha)
    require(authentication == dict(schema='trid-public-work-local-export-authentication-v1',
                run_id=identity['run_id'], plan_sha256=plan_sha, instance_id=evidence['instance_id'],
                completion_sha256=lifecycle.digest(export_state / 'completion.json'),
                export=completion.get('export'), export_state=authentication.get('export_state'))
            and authentication['export_state'] in ('authenticated_terminal', 'authenticated_partial')
            and completion.get('run_id') == identity['run_id']
            and completion.get('plan_sha256') == plan_sha
            and completion.get('instance_id') == evidence['instance_id'],
            'Local authenticated export identity differs; metadata-only completion cannot authorize purge')
    pin = authentication['export']
    require(isinstance(pin, dict) and set(pin) == {'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= lifecycle.export_limit(plan)
            and isinstance(pin['sha256'], str) and lifecycle.SHA.fullmatch(pin['sha256']),
            'Local authenticated export pin differs')
    archive_path = export_state / 'export.tar'
    require(archive_path.is_file() and not archive_path.is_symlink()
            and archive_path.stat().st_size == pin['bytes']
            and lifecycle.digest(archive_path) == pin['sha256'],
            'Local authenticated export is missing or changed')
    verify_compute_cleanup(reg, state, client, plan, plan_sha)
    bucket = identity['bucket']
    require(not client('s3api', 'get-bucket-versioning', '--bucket', bucket).get('Status'),
            'Versioned/suspended bucket purge is not admitted')
    before = client('s3api', 'get-bucket-lifecycle-configuration', '--bucket', bucket)
    lifecycle.validate_lifecycle_configuration(plan, before)

    def inventory(prefix):
        listing = client('s3api', 'list-objects-v2', '--bucket', bucket,
                         '--prefix', prefix, '--max-keys', '1000')
        uploads = client('s3api', 'list-multipart-uploads', '--bucket', bucket,
                         '--prefix', prefix, '--max-uploads', '1')
        require(not listing.get('IsTruncated') and not uploads.get('IsTruncated')
                and not uploads.get('Uploads'), 'Incomplete or multipart run inventory; preserve objects')
        rows = listing.get('Contents', [])
        require(isinstance(rows, list) and all(isinstance(row.get('Key'), str)
                and row['Key'].startswith(prefix) for row in rows)
                and len({row['Key'] for row in rows}) == len(rows), 'S3 prefix inventory differs')
        return rows

    inventories = {prefix: inventory(prefix) for prefix in prefixes}
    record(state / 's3-purge-authorized.json',
        dict(schema='trid-public-work-s3-purge-authorization-v1',
             registration_sha256=registration_sha, run_id=identity['run_id'], plan_sha256=plan_sha,
             instance_id=evidence['instance_id'], bucket=bucket, prefixes=prefixes,
             local_export_authentication=authentication, inventory=inventories,
             lifecycle_before=before, compute_cleanup='complete'))

    def object_absent(key):
        try:
            client('s3api', 'head-object', '--bucket', bucket, '--key', key)
        except lifecycle.AWSCallError as exc:
            require(exc.code in ('404', 'NoSuchKey', 'NotFound'), 'Object absence unresolved: ' + key)
            return dict(key=key, absent=True, error_code=exc.code)
        raise RuntimeError('Object remains after purge: ' + key)

    keys = sorted(row['Key'] for rows in inventories.values() for row in rows)
    for index, key in enumerate(keys):
        mutate(client, state, registration_sha, 's3-purge-object-' + str(index),
               's3api', 'delete-object', ['--bucket', bucket, '--key', key],
               lambda response, key=key: object_absent(key))
    absence = {prefix: inventory(prefix) for prefix in prefixes}
    require(all(not rows for rows in absence.values()), 'Run objects remain after purge')
    after = client('s3api', 'get-bucket-lifecycle-configuration', '--bucket', bucket)
    require(after == before, 'Bucket lifecycle rules changed during purge')
    return record(state / 's3-purge-complete.json',
        dict(schema='trid-public-work-s3-purge-receipt-v1', registration_sha256=registration_sha,
             run_id=identity['run_id'], plan_sha256=plan_sha, bucket=bucket, prefixes=prefixes,
             export=pin, deleted_keys=keys, s3_mutations=len(keys), absence_readback=absence,
             cleanup_state='complete', retained_prefix_lifecycle_rules=True,
             lifecycle_sha256=lifecycle.object_sha(after)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('prepare', 'validate', 'prepare-prerequisites', 'stage', 'launch', 'reconcile', 'retrieve', 'owned-prerequisite-cleanup', 's3-purge'))
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--execute', action='store_true', help='Explicit parent permission for this live action; never inferred')
    args = parser.parse_args()
    live = args.action not in ('prepare', 'validate')
    require(not live or args.execute, 'Live actions require explicit --execute')
    require(not any(p.is_symlink() for p in (args.state, *args.state.parents)), 'Symlink state forbidden')
    args.state.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.state / 'fresh-controller.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg = lifecycle.read_bound(args.registration, args.registration_sha256)
        validate_registration(reg, args.root,
                              launch_window=args.action not in ('reconcile', 'retrieve', 'owned-prerequisite-cleanup', 's3-purge'))
        binding = dict(registration_sha256=args.registration_sha256, identity=reg['identity'])
        binding_path = args.state / 'registration-consumed.json'
        if binding_path.exists():
            require(load(binding_path) == binding, 'State belongs to another registration')
        else:
            record(binding_path, binding)
        plan = lifecycle.read_bound(args.plan, args.plan_sha256) if args.plan else None
        if plan is not None:
            require(plan['identity'] == reg['identity']
                    and plan.get('schema') == registration_plan_schema(reg),
                    'Wrong final plan identity')
        needs_final = args.action in ('validate', 'stage', 'launch')
        if needs_final:
            require(plan is not None and args.approval and args.approval_sha256, 'Parent-bound plan and approval required')
            approval = lifecycle.read_bound(args.approval, args.approval_sha256)
            validated = validate_final(reg, args.registration_sha256, args.root, args.state, plan, approval, args.plan_sha256)
        client = lifecycle.Client(reg['identity'], deadline=time.time() + 1200, request_limit=4096) if live else None
        if args.action == 'prepare':
            result = prepare_bundle(reg, args.root, args.state)
        elif args.action == 'validate':
            result = validated
        elif args.action == 'prepare-prerequisites':
            result = prepare_prerequisites(reg, args.registration_sha256, args.state, client)
        elif args.action == 'stage':
            result = stage(reg, args.registration_sha256, args.root, args.state, plan, args.plan_sha256, args.plan, client)
        elif args.action == 'launch':
            client.deadline = time.time() + lifecycle.lifetime_seconds(plan) + plan['cost']['termination_grace_seconds']
            if lifecycle.sampling_mode(plan) or lifecycle.public_contract_mode(plan):
                client.deadline = min(client.deadline, lifecycle.absolute_deadline(plan))
            client.request_limit = plan['cost']['control_requests']
            result = lifecycle.launch(plan, approval, args.plan_sha256, args.root, args.state, client)
        elif args.action == 'reconcile':
            require(plan is not None, 'Final plan required for reconcile')
            caller_matches(client, reg)
            result = reconcile(plan, args.plan_sha256, args.state, client)
        elif args.action == 'retrieve':
            require(plan is not None, 'Final plan required for retrieval')
            result = retrieve_run_export(reg, args.registration_sha256, args.state, client, plan, args.plan_sha256)
        elif args.action == 's3-purge':
            require(plan is not None, 'Final plan required for S3 purge')
            result = purge_run_s3(reg, args.registration_sha256, args.state, client, plan, args.plan_sha256)
        else:
            result = owned_prerequisite_cleanup(reg, args.registration_sha256, args.state, client, plan, args.plan_sha256)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('cleanup_state') == 'unresolved' or (args.action == 'launch' and not result.get(lifecycle.success_field(plan))):
            raise SystemExit(1)
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    main()
