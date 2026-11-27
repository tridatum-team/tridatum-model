"""Read-only AWS observation collection and explicitly authorized public-work inputs.

Run from the repository root, after the current offline projection is complete::

  export AWS_PROFILE
  python3 scripts/aws_eval/public_work_operator_inputs.py observe \
    --run-dir "$RUN" --reference-run-dir "$REFERENCE" \
    --projection "$RUN/projection/execution-inputs.json" \
    --run-id "$RUN_ID" --profile "$AWS_PROFILE" --allow-read-only-aws

Review every response/error in $RUN/operator-observations.json and its pinned
observations, including global/regional ancillary services. Then, within the
original one-hour observation window::

  python3 scripts/aws_eval/public_work_operator_inputs.py authorize \
    --run-dir "$RUN" --reference-run-dir "$REFERENCE" \
    --projection "$RUN/projection/execution-inputs.json" \
    --run-id "$RUN_ID" --profile "$AWS_PROFILE" \
    --observations-sha256 "$REVIEWED_OBSERVATIONS_SHA256" \
    --absolute-deadline-utc "$FRESH_DEADLINE_UTC" \
    --approve-fresh-one-run --approve-ancillary-retention-network-cost-review

Both invocations are recorded verbatim. observe prints the observation SHA.
If an observation failed for a reason other than a recognized absence, authorize
also requires --error-review FILE. FILE is a JSON object keyed by observation
name, each value {observation_sha256, reason, evidence:[Pin,...]}. Evidence must
be fresh successful read-only observation receipts with query, exit_code,
response, started_at, completed_at; these explain why an API error does not
leave an unpriced service. AccessDenied is NEVER interpreted as absence.
The operator, not this program, performs that independent review; flags record
that judgment and the fresh user grant, not permission inferred from history.

Exact read-only argv inventory is written BEFORE calls to readonly-calls.json.
It reuses only requests from the reference provisional-plan's pinned observation
inventory, substituting fresh run-owned names/tags/prefixes and CLI profile.
This preserves the first-run regional/global inventory (EC2/network/quota,
S3/S3Control, IAM absence, Scheduler absence, KMS, Pricing, CloudTrail, Backup,
Config, GuardDuty, Inspector, Macie, Access Analyzer, IPAM, SSM/QuickSetup,
Organizations), plus an explicit STS GetCallerIdentity. Every request is checked
against a read-only operation prefix; no shell, SDK mutation, automatic retry,
resource creation, launch, or upload exists here. AWS_MAX_ATTEMPTS=1 is enforced.
No request/response from the old run is relabeled as a fresh observation.

The reference supplies fixed same-AMI dependency pins, infrastructure constraints,
policies and finite resource/cost limits only. Current projection data, recipe,
dynamic suites, optional prior comparator and source pins replace old ones.
This generator deliberately retains the reference subnet/AZ, eight-hour envelope,
phase limits and <=$30 cap; a new infrastructure
or resource envelope needs a separately reviewed contract. Price rates are always
recomputed from the freshly returned products, never copied from history.
Only the exactly pinned original prior adapter permits a 256 MiB input bundle;
other workloads retain 128 MiB, and output/resource/time/cost ceilings are unchanged.

After authorize, next-commands.json contains exact prepare (offline),
prepare-prerequisites (MUTATES AWS), and finalize (offline) argv. Finalize takes
--run-dir --state --registration-sha256. Subsequent fresh_eight_launch stage,
launch, reconcile and owned-prerequisite-cleanup/s3-purge MUTATE AWS and remain
separate explicitly executed operator actions. No plan/approval is launchable
until real prepare/prerequisite readbacks and final approval are bound.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.aws_eval import fresh_eight_launch as fresh
from scripts.aws_eval import grounded_hf_lifecycle as lifecycle
from scripts.aws_eval import public_work_experiment as public

require = fresh.require


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def resolved(path):
    path = str(Path(path).resolve())
    return Path(unicodedata.normalize('NFD', path) if sys.platform == 'darwin' else path)


def pin(root, path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular nonsymlink file required')
    raw = path.read_bytes()
    return dict(path=path.relative_to(root).as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def save(root, path, value):
    fresh.record(path, value)
    return pin(root, path)


def substitute(value, replacements):
    if isinstance(value, str):
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [substitute(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, replacements) for key, item in value.items()}
    return value


def readonly(query):
    require(isinstance(query, list) and len(query) >= 3 and query[0] == 'aws'
            and all(isinstance(arg, str) for arg in query)
            and (query[2].startswith(('get-', 'describe-', 'list-'))
                 or query[1:3] == ['inspector2', 'batch-get-account-status']),
            'Only explicit read-only AWS requests admitted')
    require((query[1] != 'sts' or query[2] == 'get-caller-identity')
            and query[2] not in ('get-authorization-token', 'get-login-password',
                                'get-credentials', 'get-role-credentials'),
            'Credential issuance is not an observation')
    require(not any('file://' in arg or 'fileb://' in arg for arg in query),
            'Indirect CLI inputs forbidden')
    require(not any(arg.split('=', 1)[0] in ('--endpoint-url', '--no-sign-request')
                    for arg in query), 'Only authenticated standard AWS endpoints admitted')
    return query


def complete_response(response):
    require(isinstance(response, dict), 'AWS observation must be a JSON object')
    continuation = {'nexttoken', 'nextmarker', 'nextcontinuationtoken', 'nextpagetoken', 'istruncated'}
    require(not any(value for key, value in response.items()
                    if key.replace('_', '').lower() in continuation),
            'Incomplete AWS observation; pagination tokens cannot be admitted')
    return response


def context(args):
    root = resolved(Path(__file__).parents[2])
    run, reference = resolved(args.run_dir), resolved(args.reference_run_dir)
    require(run.is_relative_to(root) and reference.is_relative_to(root) and run != reference,
            'Distinct repository-local run and reference directories required')
    require(re.fullmatch('[a-z][a-z0-9-]{8,49}', args.run_id), 'Invalid fresh run ID')
    reg = fresh.load(reference / 'registration.json')
    old = fresh.load(reference / 'plan.json')
    require(reg['schema'] == fresh.PUBLIC_WORK_SCHEMA and old['schema'] == public.SCHEMA,
            'Public-work reference contract required')
    require(args.run_id != reg['identity']['run_id'], 'Historical run authorization is not reusable')
    identity = copy.deepcopy(reg['identity'])
    require(identity['account'] == '148761639313' and identity['region'] == 'ap-northeast-2'
            and identity['bucket'] == 'trid-training-148761639313',
            'Only the fixed first-run account/region/bucket is admitted')
    replacements = {identity['run_id']: args.run_id, identity['eval_tag']: args.run_id}
    identity = substitute(identity, replacements)
    projection_path = resolved(args.projection)
    require(projection_path.is_relative_to(run), 'Current projection must belong to this run')
    projection = fresh.load(projection_path)
    require(projection['schema'] == public.OFFLINE_PROJECTION_SCHEMA
            and projection['cloud_authorized'] is False, 'Offline public projection required')
    public.validate_recipe(projection['recipe'])
    require(set(projection['sources']) == public.SOURCE_NAMES, 'Source whitelist differs')
    sources = fresh.bound_json(root, projection['workload']['source_manifest'])['files']
    require({p['path']: p['sha256'] for p in sources} == projection['sources'],
            'Projection source seal differs')
    for item in sources:
        fresh.pin_bytes(root, item)
    provisional = fresh.load(reference / 'provisional-plan.json')
    cloud = fresh.bound_json(root, provisional['cloud_evidence'])
    return root, run, reg, old, identity, replacements, projection, cloud


def observe(args, ctx):
    root, run, reg, old, identity, replacements, projection, cloud = ctx
    require(args.allow_read_only_aws, 'Explicit --allow-read-only-aws required')
    require(not (run / 'operator-observations.json').exists()
            and not (run / 'observations').exists(), 'New observation directory required; never replay')
    run.mkdir(parents=True, exist_ok=True)
    directory = run / 'observations'
    directory.mkdir()
    calls = {}
    for name, record in cloud['observations'].items():
        previous = fresh.bound_json(root, record)
        query = substitute(previous.get('query', previous.get('command')), replacements)
        readonly(query)
        require(query.count('--profile') <= 1, 'Ambiguous CLI profile')
        if '--profile' in query:
            query[query.index('--profile') + 1] = args.profile
        else:
            query += ['--profile', args.profile]
        calls[name] = query
    calls['caller_identity'] = ['aws', 'sts', 'get-caller-identity', '--profile', args.profile,
                               '--region', identity['region'], '--output', 'json', '--no-cli-pager',
                               '--cli-connect-timeout', '5', '--cli-read-timeout', '30']
    inventory = save(root, run / 'readonly-calls.json', dict(cloud_mutations=False, calls=calls,
        invocation=[sys.executable, *sys.argv], reference_plan=pin(root, args.reference_run_dir.resolve() / 'plan.json')))
    observations = {}
    environment = dict(os.environ, AWS_PROFILE=args.profile, AWS_MAX_ATTEMPTS='1',
                       AWS_RETRY_MODE='standard', AWS_PAGER='',
                       AWS_IGNORE_CONFIGURED_ENDPOINT_URLS='true')
    for name, query in calls.items():
        started = now()
        result = subprocess.run(readonly(query), env=environment, capture_output=True, text=True,
                                timeout=180, check=False)
        completed = now()
        error = lifecycle.AWSCallError(query[1], query[2], result.stderr).code if result.returncode else None
        # Optional-field-only AWS responses (for example unset S3 acceleration)
        # have empty successful CLI output; match the lifecycle client's {} shape.
        response = (json.loads(result.stdout) if result.stdout.strip() else {}) if result.returncode == 0 else None
        observations[name] = save(root, directory / (name + '.json'), dict(query=query,
            started_at=started, completed_at=completed, exit_code=result.returncode,
            error_code=error, response=response, stdout=result.stdout, stderr=result.stderr,
            cloud_mutations=False, environment={'AWS_MAX_ATTEMPTS': '1', 'AWS_PROFILE': args.profile}))
        if result.returncode == 0:
            complete_response(response)
    result = save(root, run / 'operator-observations.json', dict(
        schema='trid-public-work-operator-observations-v1', run_id=identity['run_id'],
        identity=identity, projection=pin(root, args.projection.resolve()), inventory=inventory,
        observations=observations, cloud_mutations=False, old_observations_reused=False))
    print(lifecycle.canonical(result).decode())


def observations(args, root, run, identity):
    record = pin(root, run / 'operator-observations.json')
    require(record['sha256'] == args.observations_sha256, 'Explicit reviewed observation SHA required')
    ledger = fresh.bound_json(root, record)
    require(ledger['identity'] == identity and ledger['projection'] == pin(root, args.projection.resolve())
            and ledger['old_observations_reused'] is False, 'Observation run/projection differs')
    rows = {key: fresh.bound_json(root, item) for key, item in ledger['observations'].items()}
    for row in rows.values():
        readonly(row['query'])
        if row['exit_code'] == 0:
            require(row.get('error_code') is None, 'Successful observation contains an error')
            complete_response(row['response'])
        require(lifecycle.timestamp(row['started_at']) <= lifecycle.timestamp(row['completed_at'])
                <= datetime.now(timezone.utc).timestamp(), 'Invalid observation timing')
    earliest = min(row['started_at'] for row in rows.values())
    latest = max(row['completed_at'] for row in rows.values())
    require(0 <= datetime.now(timezone.utc).timestamp() - lifecycle.timestamp(earliest) < 3600,
            'Original observation window expired; never relabel timestamps')
    review = fresh.load(args.error_review) if args.error_review else {}
    absence = {
        'bucket_policy': 'NoSuchBucketPolicy', 'bucket_policy_status': 'NoSuchBucketPolicy',
        'bucket_replication': 'ReplicationConfigurationNotFoundError',
        'bucket_object_lock': 'ObjectLockConfigurationNotFoundError',
        'bucket_metadata_configuration': 'MetadataConfigurationNotFound',
        'instance_role_absent': 'NoSuchEntity', 'scheduler_role_absent': 'NoSuchEntity',
        'instance_profile_absent': 'NoSuchEntity', 'schedule_absent': 'ResourceNotFoundException',
        'schedule_group_absent': 'ResourceNotFoundException',
        'organization': 'AWSOrganizationsNotInUseException',
    }
    reviewed = set()
    for name, row in rows.items():
        if row['exit_code'] == 0:
            continue
        error = lifecycle.AWSCallError(row['query'][1], row['query'][2], row['stderr']).code
        require(error == row['error_code'], 'Error code does not match actual stderr')
        if name in absence and error == absence[name]:
            continue
        resolution = review.get(name, {})
        require(resolution.get('observation_sha256') == ledger['observations'][name]['sha256']
                and isinstance(resolution.get('reason'), str) and resolution['reason'].strip()
                and resolution.get('evidence'), 'Unresolved read-only observation failure: ' + name)
        for evidence in resolution['evidence']:
            actual = fresh.bound_json(root, evidence)
            readonly(actual['query'])
            require(actual['exit_code'] == 0 and actual.get('response') is not None
                    and lifecycle.timestamp(earliest) <= lifecycle.timestamp(actual['started_at'])
                    <= lifecycle.timestamp(actual['completed_at']) <= datetime.now(timezone.utc).timestamp(),
                    'Error resolution requires genuine fresh successful observations')
        reviewed.add(name)
    require(set(review) == reviewed, 'Error review must match unresolved failures exactly')
    return record, ledger, rows, earliest, latest


def authorize(args, ctx):
    root, run, reference_reg, reference_plan, identity, replacements, projection, reference_cloud = ctx
    require(args.approve_fresh_one_run and args.approve_ancillary_retention_network_cost_review,
            'Fresh one-run authorization and independent complete cost review flags required')
    require(args.absolute_deadline_utc and args.observations_sha256,
            'Explicit fresh deadline and reviewed observation SHA required')
    require(not any((run / name).exists() for name in ('authorization.json', 'registration.json',
            'resource-envelope.json', 'cost-admission.json', 'provisional-plan.json')),
            'Operator inputs already exist; never overwrite or replay a grant')
    observation_pin, ledger, rows, earliest, checked = observations(args, root, run, identity)

    def response(name):
        row = rows[name]
        require(row['exit_code'] == 0 and row.get('error_code') is None,
                'Successful current AWS observation required: ' + name)
        return row['response']

    historical_regions = fresh.bound_json(root, reference_cloud['observations']['enabled_regions'])
    covered = {row['RegionName'] for row in historical_regions['response']['Regions']}
    current_regions = response('enabled_regions')['Regions']
    require(covered and {row['RegionName'] for row in current_regions} == covered
            and len(current_regions) == len(covered)
            and all(row['OptInStatus'] in ('opt-in-not-required', 'opted-in') for row in current_regions),
            'Enabled regions differ from audited request coverage; a new inventory is required')

    caller = response('caller_identity')
    require(caller['Account'] == identity['account'], 'Wrong observed AWS account')
    for name in ('instance_role_absent', 'scheduler_role_absent', 'instance_profile_absent',
                 'schedule_absent', 'schedule_group_absent'):
        require(rows[name]['exit_code'] != 0 and rows[name]['error_code'] in
                ('NoSuchEntity', 'ResourceNotFoundException'), 'Fresh resource already exists: ' + name)
    require(not response('new_security_group_absent')['SecurityGroups'], 'Fresh SG already exists')
    for prefix in ('code_prefix', 'output_prefix'):
        require(not response(prefix + '_empty').get('Contents')
                and not response(prefix + '_multipart').get('Uploads'), 'Fresh S3 prefix is occupied')
    plan = {key: copy.deepcopy(reference_plan[key]) for key in (
        'schema', 'instance_type', 'architecture', 'vcpus', 'host_mib', 'disk_os_reserve_bytes',
        'model', 'runtime', 'limits', 'lifetime', 'phase_seconds')}
    plan.update(identity=identity, controller_arn=caller['Arn'], checked_at=checked,
                absolute_termination_deadline=args.absolute_deadline_utc)
    plan.update({key: projection[key] for key in ('data', 'workload', 'recipe', 'sources')})
    plan['output_allowlist'] = sorted(public.output_members(plan))
    plan['cost'] = {key: copy.deepcopy(value) for key, value in reference_plan['cost'].items()
                    if key not in ('rates', 'bucket_lifecycle_configuration')}
    require(0 < plan['cost']['cap_usd'] <= 30, 'Maximum USD30 cap required')
    public.validate_limits(plan)
    lifecycle.absolute_deadline(plan)
    lifecycle.validate_launch_window(plan, datetime.now(timezone.utc).timestamp())
    images = response('ami')['Images']
    require(len(images) == 1, 'Single selected AMI observation required')
    image = images[0]
    require(image['State'] == 'available' and image['RootDeviceType'] == 'ebs'
            and not image.get('BillingProducts') and not image.get('ProductCodes'), 'Unpriced/unavailable AMI')
    lifecycle.validate_ami_mappings(image, reference_plan['ami'])
    plan['ami'] = {key: image[key] for key in ('ImageId', 'OwnerId', 'Name', 'Architecture', 'RootDeviceName')}
    plan['ami'].update({key: reference_plan['ami'][key] for key in ('SnapshotId', 'root_volume_gib')})
    require(plan['ami'] == reference_plan['ami'], 'Only unchanged same-AMI bootstrap evidence may be reused')
    types = response('instance_types')['InstanceTypes']
    require(len(types) == 1 and types[0]['VCpuInfo']['DefaultVCpus'] == 8
            and types[0]['MemoryInfo']['SizeInMiB'] == 65536, 'Fixed host shape changed')
    plan['gpu_type_observation'] = types[0]['GpuInfo']['Gpus']
    lifecycle.verify_gpu_catalog(plan, types[0]['GpuInfo'])
    network = reference_reg['network']
    subnets = [s for s in response('subnets')['Subnets'] if s['SubnetId'] == network['subnet_id']]
    require(len(subnets) == 1 and subnets[0]['VpcId'] == network['vpc_id']
            and subnets[0]['AvailabilityZone'] == network['availability_zone']
            and subnets[0]['State'] == 'available', 'Selected subnet differs')
    routes = response('route_tables')['RouteTables']
    explicit = [r for r in routes if any(a.get('SubnetId') == network['subnet_id'] for a in r['Associations'])]
    effective = explicit or [r for r in routes if r['VpcId'] == network['vpc_id']
                             and any(a.get('Main') for a in r['Associations'])]
    require(len(effective) == 1, 'Exact effective route table required')
    route = effective[0]
    gateways = [r['GatewayId'] for r in route['Routes'] if r.get('DestinationCidrBlock') == '0.0.0.0/0'
                and r.get('GatewayId', '').startswith('igw-') and r['State'] == 'active']
    require(len(gateways) == 1 and not any(r.get('NatGatewayId') for r in route['Routes']), 'Direct IGW only')
    plan['network'] = {key: copy.deepcopy(reference_plan['network'][key]) for key in
                       ('vpc_id', 'subnet_id', 'availability_zone', 'download_hosts', 'egress_byte_cap')}
    plan['network'].update(route_table_id=route['RouteTableId'], route_table_observation=route,
                           internet_gateway_id=gateways[0])
    plan['ebs_kms_key_arn'] = response('ebs_managed_key')['KeyMetadata']['Arn']
    policy = rows['bucket_policy_status']
    plan['bucket_policy_status'] = (lifecycle.private_policy_status(response('bucket_policy_status'))
        if policy['exit_code'] == 0 else dict(state='absent', error_code='NoSuchBucketPolicy'))
    require(policy['exit_code'] == 0 or policy['error_code'] == 'NoSuchBucketPolicy', 'Unknown policy status')
    require(response('bucket_location')['LocationConstraint'] == identity['region']
            and not response('bucket_versioning').get('Status'), 'Bucket region/versioning differs')
    lifecycle.validate_bucket_encryption(response('bucket_encryption')['ServerSideEncryptionConfiguration'])
    plan['ancillary_checks'] = {}
    for name, historical in reference_plan['ancillary_checks'].items():
        check = {key: copy.deepcopy(historical[key]) for key in ('service', 'operation', 'args')}
        row = rows[name]
        if row['exit_code'] == 0:
            check['response_sha256'] = lifecycle.object_sha(row['response'])
        else:
            require(name in ('bucket_policy', 'bucket_replication') and row['error_code'] ==
                    {'bucket_policy': 'NoSuchBucketPolicy', 'bucket_replication': 'ReplicationConfigurationNotFoundError'}[name],
                    'Ancillary failure cannot be admitted as absence')
            check['expected_absence'] = row['error_code']
        plan['ancillary_checks'][name] = check
    plan['price_queries'], plan['cost']['rates'] = {}, {}
    for name, historical in reference_plan['price_queries'].items():
        query = {key: copy.deepcopy(historical[key]) for key in ('service', 'unit', 'filters')}
        products = [json.loads(value) for value in response('price_' + name)['PriceList']]
        query['products_sha256'] = lifecycle.on_demand_price_products_sha256(products)
        prices = [float(d['pricePerUnit']['USD']) for p in products
                  for term in p.get('terms', {}).get('OnDemand', {}).values()
                  for d in term['priceDimensions'].values() if d['unit'] == query['unit']]
        require(prices and all(math.isfinite(v) and v >= 0 for v in prices), 'Missing finite observed price')
        plan['price_queries'][name], plan['cost']['rates'][name] = query, max(prices)
    reg = dict(schema=reference_reg['schema'], approved=True, automatic_retry=False,
        identity=identity, controller_arn=caller['Arn'], iam=substitute(reference_reg['iam'], replacements),
        network=substitute(reference_reg['network'], replacements),
        bucket_lifecycle_before=response('bucket_lifecycle'), retention_days=30,
        lifecycle_rule_ids={key: identity['run_id'] + '-' + key for key in ('code_prefix', 'output_prefix')},
        bootstrap_admission=reference_reg['bootstrap_admission'])
    plan['iam'] = reg['iam']
    plan['cost']['bucket_lifecycle_configuration'] = fresh.lifecycle_merge(reg)
    lifecycle.validate_lifecycle_configuration(plan, plan['cost']['bucket_lifecycle_configuration'])
    # Price the sealed maximum before the bundle exists; finalization replaces
    # this planning bound with the actual authenticated prepare receipt.
    bundle_bytes_upper_bound = public.bundle_bytes_max(plan)
    bound_cost = public.cost_bound(dict(plan, bundle={'bytes': bundle_bytes_upper_bound}))
    approval_intent = save(root, run / 'approval-intent.json', dict(
        schema='trid-public-work-operator-intent-v1', approved=True, run_id=identity['run_id'],
        observations=observation_pin, projection=pin(root, args.projection.resolve()),
        invocation=[sys.executable, *sys.argv], approved_at=now(),
        independent_ancillary_retention_network_review=True,
        authorization_evidence=pin(root, resolved(args.authorization_evidence))
            if getattr(args, 'authorization_evidence', None) else None,
        error_review=pin(root, args.error_review.resolve()) if args.error_review else None))
    plan['authorization'] = save(root, run / 'authorization.json', dict(
        schema=public.AUTHORIZATION_SCHEMA, run_id=identity['run_id'], approved=True,
        workload_sha256=public.object_sha(plan['workload']), data_sha256=public.object_sha(plan['data']),
        recipe_sha256=public.object_sha(plan['recipe']), source_manifest_sha256=plan['workload']['source_manifest']['sha256'],
        execution_started=False, additional_gpu_executions=1, automatic_retry=False,
        automatic_extension=False, deployment=False, budget_usd=plan['cost']['cap_usd'],
        max_wall_seconds=plan['lifetime']['total_seconds'], absolute_deadline_utc=args.absolute_deadline_utc,
        approval_intent=approval_intent))
    cost_evidence = save(root, run / 'cost-evidence.json', dict(
        schema='trid-public-work-current-cost-evidence-v1', observations=observation_pin,
        checked_at=checked, rates=plan['cost']['rates'], conditional_bound=bound_cost,
        bundle_bytes_upper_bound=bundle_bytes_upper_bound,
        independent_review=approval_intent, billing_hard_cap=False))
    envelope = fresh.bound_json(root, reference_reg['resource_envelope'])
    envelope.update(identity=identity, authorization=plan['authorization'], cost_evidence=cost_evidence,
                    absolute_termination_deadline=args.absolute_deadline_utc,
                    bundle_bytes_max=bundle_bytes_upper_bound)
    plan['resource_envelope'] = save(root, run / 'resource-envelope.json', envelope)
    admission = dict(schema='trid-grounded-hf-cost-admission-v1', status='admitted',
        run_id=identity['run_id'], account=identity['account'], region=identity['region'],
        ancillary_bound_usd=plan['cost']['ancillary_bound_usd'],
        ancillary_checks_sha256=lifecycle.object_sha(plan['ancillary_checks']),
        pricing_products_sha256={key: q['products_sha256'] for key, q in plan['price_queries'].items()},
        bucket_lifecycle_configuration_sha256=lifecycle.object_sha(plan['cost']['bucket_lifecycle_configuration']),
        bucket_policy_status_sha256=lifecycle.object_sha(plan['bucket_policy_status']),
        ebs_kms_key_arn=plan['ebs_kms_key_arn'], no_unpriced_services=True,
        retention_closure_reviewed=True, network_request_closure_reviewed=True, checked_at=checked,
        bucket_policy_status_evidence=ledger['observations']['bucket_policy_status'],
        ebs_kms_key_evidence=ledger['observations']['ebs_managed_key'])
    plan['ancillary_evidence'] = save(root, run / 'cost-admission.json', admission)
    for key in ('authorization', 'resource_envelope', 'workload', 'recipe', 'limits', 'lifetime',
                'cost', 'absolute_termination_deadline', 'data', 'phase_seconds', 'ancillary_evidence'):
        reg[key] = plan[key]
    reg['sources'] = fresh.bound_json(root, plan['workload']['source_manifest'])['files']
    bootstrap = fresh.bound_json(root, reg['bootstrap_admission'])
    records = [*reg['sources'], *plan['data'].values(), *public.workload_pins(plan['workload']),
        plan['authorization'], plan['resource_envelope'], plan['ancillary_evidence'],
        reg['bootstrap_admission'], cost_evidence, admission['bucket_policy_status_evidence'],
        admission['ebs_kms_key_evidence'], *bootstrap['evidence']]
    files = {item['path']: item for item in records}
    for item in files.values():
        fresh.pin_bytes(root, item)
    reg['bundle'] = dict(local_path=(run / 'bundle.tar').relative_to(root).as_posix(),
                         files=[files[key] for key in sorted(files)])
    fresh.validate_registration(reg, root)
    plan['bundle'] = {'files': reg['bundle']['files']}
    lifecycle.validate_cost_admission(root, plan)
    lifecycle.validate_bootstrap_admission(root, plan)
    del plan['bundle']
    registration = save(root, run / 'registration.json', reg)
    cloud_pin = save(root, run / 'cloud-evidence.json', dict(
        schema='trid-public-work-current-cloud-evidence-v1', run_id=identity['run_id'],
        observations=ledger['observations'], checked_at=checked, old_observations_reused=False,
        operator_review=approval_intent, observation_inventory=ledger['inventory'],
        bootstrap_historical_same_ami_only=reg['bootstrap_admission']))
    expiry = (datetime.fromisoformat(earliest.replace('Z', '+00:00')) + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    provisional = save(root, run / 'provisional-plan.json', dict(
        schema='trid-public-work-provisional-plan-v1', launchable=False, registration=registration,
        cloud_evidence=cloud_pin, earliest_observed_at=earliest, observations_expire_at=expiry,
        pending=['exact prepare bundle readback', 'exact prepare-prerequisites owned readbacks',
                 'final plan/approval assembly and validation'], plan_fields=plan))
    common = ['--registration', registration['path'], '--registration-sha256', registration['sha256'],
              '--root', '.', '--state', (run / 'state').relative_to(root).as_posix()]
    commands = [dict(action=action, cloud_mutation=live,
        argv=['python3', 'scripts/aws_eval/fresh_eight_launch.py', action, *common, *(['--execute'] if live else [])])
        for action, live in [('prepare', False), ('prepare-prerequisites', True)]]
    commands.append(dict(action='finalize', cloud_mutation=False,
        argv=['python3', 'scripts/aws_eval/public_work_finalize_plan.py', '--run-dir',
              run.relative_to(root).as_posix(), '--state', (run / 'state').relative_to(root).as_posix(),
              '--registration-sha256', registration['sha256'], '--approve-fresh-plan']))
    save(root, run / 'next-commands.json', dict(cwd=str(root), commands=commands,
        environment={'AWS_PROFILE': args.profile},
        observations_expire_at=expiry, registration_sha256=registration['sha256'],
        later_mutating_actions=['stage', 'launch', 'reconcile', 'owned-prerequisite-cleanup', 's3-purge']))
    print(lifecycle.canonical(dict(registration=registration, provisional=provisional,
                                  cloud_mutations=False)).decode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('observe', 'authorize'))
    for name in ('run-dir', 'reference-run-dir', 'projection'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--allow-read-only-aws', action='store_true')
    parser.add_argument('--observations-sha256')
    parser.add_argument('--absolute-deadline-utc')
    parser.add_argument('--approve-fresh-one-run', action='store_true')
    parser.add_argument('--approve-ancillary-retention-network-cost-review', action='store_true')
    parser.add_argument('--error-review', type=Path)
    parser.add_argument('--authorization-evidence', type=Path,
                        help='Optional pinned local record of the fresh owner grant')
    args = parser.parse_args(argv)
    for name in ('run_dir', 'reference_run_dir', 'projection', 'error_review', 'authorization_evidence'):
        if getattr(args, name) is not None:
            setattr(args, name, resolved(getattr(args, name)))
    ctx = context(args)
    if args.action == 'observe':
        observe(args, ctx)
    else:
        authorize(args, ctx)


if __name__ == '__main__':
    main()
