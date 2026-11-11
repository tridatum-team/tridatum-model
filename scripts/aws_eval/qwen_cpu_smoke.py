"""One approved externally traced Linux run; no model/launch retries or fallbacks.

Public commands: preflight, launch, status, terminate. Remote-stage is private to
verified cloud-init. Launch/scheduler are NOT atomic. A verified independent
schedule and matching release object are required before model admission.
Preflight/launch require --max-seconds 7200 --approved-usd 10 and additional USD3.
Prior bound: USD6.860792723718095; authenticated prior Linux caches only, no regeneration.
AWS CLI uses one attempt and pip disables retries. Bootstrap alone permits one
additional DNF package-download retry within its 300-second outer timeout;
this transport exception never permits package/version substitutions.
RunInstances CLI JSON carries raw UTF-8 user data; the CLI's botocore handler
performs the sole API base64 encoding. The 16KiB limit applies to those raw bytes.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time

ACCOUNT = '148761639313'
REGION = 'ap-northeast-2'
BUCKET = 'trid-training-148761639313'
PREFIX = 'eval/aws-eval-006'
EXPERIMENT = 'qwen-specialization-013'
VPC = 'vpc-0bef0e7b66432d802'
PROFILE = 'trid-aws-eval-006-instance'
SCHEDULER = 'trid-aws-eval-006-scheduler'
ROOT = Path('/Users/eomgyuhyeon/사업/trid')
TYPE = 'm7g.2xlarge'
MAX_OUTPUT = 512 * 1024 ** 2
RUN_ID = 'qwen013-traced-20261109203337'
ORIGIN_RUN_ID = 'qwen013-packed-20261109070127'
RUN_PATH = f'runs/{EXPERIMENT}/{RUN_ID}'
PLAN = f'{RUN_PATH}/execution-plan.json'
ALLOWLIST = f'{RUN_PATH}/input-allowlist.json'
PROVENANCE = f'{RUN_PATH}/provenance.json'
APPROVAL = f'{RUN_PATH}/approval.json'
PRIOR_BOUND_USD = 6.860792723718095
CUMULATIVE_CAP_USD = 10.0
ADDITIONAL_CAP_USD = 3.0
MAX_SECONDS = 7200
END = None
RUNTIME_VERSIONS = {'torch': '2.9.1+cpu', 'transformers': '4.57.6', 'peft': '0.17.1'}
PACKAGES = {'torch': '2.9.1', 'transformers': '4.57.6', 'peft': '0.17.1',
            'numpy': '2.5.4', 'tokenizers': '0.22.2', 'safetensors': '0.8.0',
            'gguf': '0.19.0', 'accelerate': '1.15.0', 'huggingface_hub': '0.36.2',
            'psutil': '7.2.2', 'jinja2': '3.1.6', 'sympy': '1.14.0'}
ENGINE_PINS = {
    'scripts/train_qwen_preferences.py': '08ee99939ae43d5d3bde6368f92e5ffdd279f088124c74dbecec4ec36f8a382c',
    'scripts/train_open_weight_lora.py': '6fb8f928fb4d448a719ea5bc116a8fed5033bc175311446bdd700507406fe753',
    'scripts/q8_gguf_base.py': '8c599d3a72372c0e102200ee17dec9f2e489a6ff0dd91163c11435d6c5ffd38f'}
REQUIRED_SOURCES = set(ENGINE_PINS) | {
    'scripts/qwen_linux_guard.py', 'scripts/aws_eval/qwen_cpu_smoke.py',
    'scripts/aws_eval/qwen_cpu_bootstrap.sh', 'tests/test_qwen_linux_guard.py',
    'tests/test_qwen_cpu_smoke.py', 'tests/test_train_open_weight_lora.py',
    'tests/test_train_qwen_preferences.py',
    'scripts/trace_qwen_preferences.py', 'tests/test_trace_qwen_preferences.py',
    'scripts/aws_eval/qwen_cpu_pipeline.py', 'tests/test_qwen_cpu_pipeline.py',
    'scripts/aws_eval/qwen_cpu_native.py', 'scripts/aws_eval/qwen_cpu_native_harness.py',
    'tests/test_qwen_cpu_native.py', 'scripts/aws_eval/qwen_native_setup.sh',
    'tests/fixtures/qwen_native/run_native_development.py'}
COMPATIBILITY_TESTS = (
    'test_train_open_weight_lora.PackedHeadTests',
    'test_train_qwen_preferences.CacheBindingTests',
    'test_train_qwen_preferences.PairLikelihoodTests',
    'test_train_qwen_preferences.CombinedStepTests')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def exclusive_intent(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def aws(service, operation, *args, region=REGION):
    seconds = 90 if END is None else min(90, END - time.time())
    require(seconds > 0, 'Controller deadline exhausted')
    env = {**os.environ, 'AWS_PAGER': '', 'AWS_MAX_ATTEMPTS': '1',
           'AWS_RETRY_MODE': 'standard'}
    result = subprocess.run(['aws', service, operation, *args, '--region', region,
                             '--output', 'json', '--no-cli-pager',
                             '--cli-connect-timeout', '10', '--cli-read-timeout', '30'],
                            env=env, capture_output=True, text=True, timeout=seconds)
    require(result.returncode == 0, f'{service} {operation} failed: {result.stderr[:1000]}')
    return json.loads(result.stdout) if result.stdout.strip() else {}


def identity(allow_root):
    # CloudShell console credentials are ephemeral; never read/export credential values.
    require(os.environ.get('AWS_EXECUTION_ENV') == 'CloudShell', 'Run from signed-in console CloudShell only')
    require(not os.environ.get('AWS_PROFILE'), 'Do not switch the operator profile')
    require(not os.environ.get('AWS_ACCESS_KEY_ID') or bool(os.environ.get('AWS_SESSION_TOKEN')),
            'Long-term access keys are prohibited')
    me = aws('sts', 'get-caller-identity')
    require(me['Account'] == ACCOUNT, 'Wrong AWS account')
    root = me['Arn'] == f'arn:aws:iam::{ACCOUNT}:root'
    require(not root or allow_root, 'Console root requires --allow-console-root')
    require(root or me['Arn'].startswith(f'arn:aws:sts::{ACCOUNT}:assumed-role/'),
            'Only temporary console-root or assumed-role callers permitted')
    return me


def tags(run_id):
    return [{'Key': 'trid:eval', 'Value': 'aws-eval-006'},
            {'Key': 'trid:experiment', 'Value': EXPERIMENT},
            {'Key': 'trid:run', 'Value': run_id}]


def instances(run_id):
    data = aws('ec2', 'describe-instances', '--filters',
               'Name=tag:trid:eval,Values=aws-eval-006',
               f'Name=tag:trid:experiment,Values={EXPERIMENT}',
               f'Name=tag:trid:run,Values={run_id}')
    return [item for row in data['Reservations'] for item in row['Instances']]



def owned_identity(row, run, expected=None):
    actual = {item['Key']: item['Value'] for item in row.get('Tags', [])}
    require(all(actual.get(item['Key']) == item['Value'] for item in tags(run))
            and row.get('ClientToken') == run
            and (expected is None or row.get('InstanceId') == expected),
            'Instance ownership mismatch')


def simulate(name, action, resources, context=None):
    args = ['--policy-source-arn', f'arn:aws:iam::{ACCOUNT}:role/{name}',
            '--action-names', action, '--resource-arns', *resources]
    if context:
        args += ['--context-entries', json.dumps(context)]
    result = aws('iam', 'simulate-principal-policy', *args)
    require(not result.get('IsTruncated'), 'Incomplete policy simulation')
    decisions = []
    for evaluation in result['EvaluationResults']:
        require(evaluation.get('EvalActionName') == action
                and evaluation.get('EvalDecision') == 'allowed'
                and not evaluation.get('MissingContextValues'),
                f'Policy simulation rejected {name}/{action}')
        specific = evaluation.get('ResourceSpecificResults')
        if specific:
            for resource in specific:
                require(not resource.get('MissingContextValues'), 'Missing resource policy context')
                decisions.append((resource.get('EvalResourceName'), resource.get('EvalResourceDecision')))
        else:
            decisions.append((evaluation.get('EvalResourceName'), evaluation.get('EvalDecision')))
    require(len(decisions) == len(resources)
            and {resource for resource, _ in decisions} == set(resources)
            and all(decision == 'allowed' for _, decision in decisions),
            f'Policy simulation did not allow every exact resource for {name}/{action}')
    return result


def reconcile_cleanup(run, expected, deadline, *, discover=None, call=None, clock=None, pause=None):
    """Observe exact ownership, termination and no EBS; empty discovery is unknown."""
    discover = instances if discover is None else discover
    call = aws if call is None else call
    clock = time.time if clock is None else clock
    pause = time.sleep if pause is None else pause
    result = {'cleanup_state': 'cleanup_unresolved',
              'launch_outcome': 'instance_found' if expected else 'launch_outcome_unknown',
              'instance_id': expected, 'observations': 0, 'errors': [],
              'termination_attempts': 0, 'remaining_owned_volumes': None}
    while clock() < deadline:
        try:
            rows = discover(run)
            result['observations'] += 1
            result['last_instances'] = rows
            if rows:
                require(len(rows) == 1, 'Reconciliation instance count mismatch')
                row = rows[0]
                owned_identity(row, run, expected)
                expected = row['InstanceId']
                result.update(instance_id=expected, launch_outcome='instance_found')
                if row['State']['Name'] == 'terminated':
                    volumes = call('ec2', 'describe-volumes', '--filters',
                        f'Name=tag:trid:run,Values={run}',
                        'Name=tag:trid:eval,Values=aws-eval-006',
                        f'Name=tag:trid:experiment,Values={EXPERIMENT}')['Volumes']
                    result['remaining_owned_volumes'] = volumes
                    if not volumes:
                        result['cleanup_state'] = 'complete'
                        return result
                elif row['State']['Name'] != 'shutting-down':
                    result['termination_attempts'] += 1
                    call('ec2', 'terminate-instances', '--instance-ids', expected)
        except Exception as error:
            result['errors'].append(f'{type(error).__name__}: {error}')
        remaining = deadline - clock()
        if remaining > 0:
            pause(min(5, remaining))
    return result


def budget_bound(ec2, ebs, ip, transfer, storage, put):
    require(all(rate > 0 for rate in (ec2, ebs, ip, transfer, storage, put))
            and ec2 <= .5 and ebs <= .2 and ip <= .02
            and transfer <= 1 and storage <= .1 and put <= .0001,
            'Observed price outside approved conservative envelope')
    # Full two-hour lifecycle PLUS five-minute cleanup for compute, IP and EBS.
    # 30-day input/output storage, bounded output egress, generous request reserve.
    hours = (MAX_SECONDS + 300) / 3600
    additional = (hours * (ec2 + ip) + 64 * ebs * hours / (28 * 24)
                  + MAX_OUTPUT / 10**9 * transfer
                  + (2 * 1024**3 + MAX_OUTPUT) / 10**9 * storage + 1.0)
    total = PRIOR_BOUND_USD + additional
    require(additional <= ADDITIONAL_CAP_USD and total <= CUMULATIVE_CAP_USD,
            'Conservative additional/cumulative all-in bound exceeds approval')
    return {'bound_usd': total, 'prior_bound_usd': PRIOR_BOUND_USD,
            'additional_bound_usd': additional, 'compute_hours': hours,
            'compute_hourly': ec2, 'ebs_gb_month': ebs,
            'ebs_gb': 64, 'ebs_hours_conservative': hours,
            'public_ipv4_hourly': ip, 'output_cap_bytes': MAX_OUTPUT,
            'cleanup_allowance_seconds': 300, 'retention_assumption_days': 30,
            'input_cap_bytes': 2 * 1024**3, 'request_control_reserve_usd': 1.0,
            'transfer_gb_usd': transfer, 'storage_gb_month_usd': storage,
            'put_request_usd': put}

def role_record(name, service):
    role = aws('iam', 'get-role', '--role-name', name)['Role']
    require(role['Arn'] == f'arn:aws:iam::{ACCOUNT}:role/{name}', 'Unexpected role ARN/path')
    statements = role['AssumeRolePolicyDocument']['Statement']
    require(any(s.get('Effect') == 'Allow' and service in (
        s.get('Principal', {}).get('Service', []) if isinstance(s.get('Principal', {}).get('Service'), list)
        else [s.get('Principal', {}).get('Service')]) for s in statements), 'Service trust absent')
    policies = {}
    for policy in aws('iam', 'list-role-policies', '--role-name', name)['PolicyNames']:
        policies[policy] = aws('iam', 'get-role-policy', '--role-name', name, '--policy-name', policy)['PolicyDocument']
    attached = []
    for item in aws('iam', 'list-attached-role-policies', '--role-name', name)['AttachedPolicies']:
        info = aws('iam', 'get-policy', '--policy-arn', item['PolicyArn'])['Policy']
        version = aws('iam', 'get-policy-version', '--policy-arn', item['PolicyArn'],
                      '--version-id', info['DefaultVersionId'])['PolicyVersion']
        attached.append({'arn': item['PolicyArn'], 'document': version['Document']})
    require(policies, 'Existing inline policy absent')
    return {'role': role, 'inline': policies, 'attached': attached}


def verify_role_scopes(record):
    instance_statements = [
        {'Sid': 'ReadOnlyTheRunCode', 'Effect': 'Allow', 'Action': 's3:GetObject',
         'Resource': f'arn:aws:s3:::{BUCKET}/{PREFIX}/code/*'},
        {'Sid': 'WriteOnlyRunResults', 'Effect': 'Allow', 'Action': 's3:PutObject',
         'Resource': f'arn:aws:s3:::{BUCKET}/{PREFIX}/runs/*'},
        {'Sid': 'SyncNeedsListingOfTheRunsPrefix', 'Effect': 'Allow', 'Action': 's3:ListBucket',
         'Resource': f'arn:aws:s3:::{BUCKET}',
         'Condition': {'StringLike': {'s3:prefix': [f'{PREFIX}/runs/*']}}}]
    scheduler_statements = [{'Sid': 'TerminateOnlyEvalInstances', 'Effect': 'Allow',
        'Action': 'ec2:TerminateInstances', 'Resource': f'arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*',
        'Condition': {'StringEquals': {'aws:ResourceTag/trid:eval': 'aws-eval-006'}}}]
    for key, policy, statements, service, attached in (
        ('instance_role', 'instance-role-policy', instance_statements, 'ec2.amazonaws.com',
         ['arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore']),
        ('scheduler_role', 'scheduler-role-policy', scheduler_statements, 'scheduler.amazonaws.com', [])):
        row = record[key]
        trust = {'Effect': 'Allow', 'Principal': {'Service': service}, 'Action': 'sts:AssumeRole'}
        if key == 'scheduler_role':
            trust['Condition'] = {'StringEquals': {
                'aws:SourceArn': f'arn:aws:scheduler:{REGION}:{ACCOUNT}:schedule-group/default',
                'aws:SourceAccount': ACCOUNT}}
        require(row['role']['AssumeRolePolicyDocument'] ==
                {'Version': '2012-10-17', 'Statement': [trust]}
                and not row['role'].get('PermissionsBoundary'), 'Existing role trust/boundary changed')
        require(row['inline'] == {policy: {'Version': '2012-10-17', 'Statement': statements}}
                and sorted(p['arn'] for p in row['attached']) == attached,
                'Existing exact S3/termination policy scope changed')


def price(service, filters, unit):
    data = aws('pricing', 'get-products', '--service-code', service, '--filters',
               json.dumps([{'Type': 'TERM_MATCH', 'Field': k, 'Value': v} for k, v in filters.items()]),
               region='us-east-1')
    values = []
    products = []
    for text in data['PriceList']:
        item = json.loads(text)
        products.append(item)
        for term in item.get('terms', {}).get('OnDemand', {}).values():
            for dim in term['priceDimensions'].values():
                if dim['unit'] == unit:
                    values.append(float(dim['pricePerUnit']['USD']))
    require(values and max(values) > 0, f'No positive {service} {unit} price')
    return max(values), products


def preflight(a):
    require(a.approved_usd == CUMULATIVE_CAP_USD and a.max_seconds == MAX_SECONDS,
            'Approval must be exactly cumulative USD10 / additional USD3 / 7200 seconds')
    result = {'identity': identity(a.allow_console_root), 'region': REGION}
    info = aws('ec2', 'describe-instance-types', '--instance-types', TYPE)['InstanceTypes'][0]
    require(info['VCpuInfo']['DefaultVCpus'] == 8 and info['MemoryInfo']['SizeInMiB'] == 32768
            and info['ProcessorInfo']['SupportedArchitectures'] == ['arm64'], 'Unexpected instance hardware')
    result['type'] = info
    quota = aws('service-quotas', 'get-service-quota', '--service-code', 'ec2', '--quota-code', 'L-1216C47A')['Quota']['Value']
    running = aws('ec2', 'describe-instances', '--filters', 'Name=instance-state-name,Values=pending,running')
    standard = [i for r in running['Reservations'] for i in r['Instances']
                if i['InstanceType'][0] in 'acdhimrtz' and i.get('InstanceLifecycle') != 'spot']
    used = 0
    for kind in sorted({i['InstanceType'] for i in standard}):
        n = aws('ec2', 'describe-instance-types', '--instance-types', kind)['InstanceTypes'][0]['VCpuInfo']['DefaultVCpus']
        used += n * sum(i['InstanceType'] == kind for i in standard)
    require(used + 8 <= quota, 'Insufficient standard On-Demand vCPU quota')
    result['quota'] = {'limit': quota, 'used_vcpu_conservative': used}
    subnet = aws('ec2', 'describe-subnets', '--subnet-ids', a.subnet_id)['Subnets'][0]
    require(subnet['VpcId'] == VPC and subnet['State'] == 'available' and subnet['AvailableIpAddressCount'] > 0,
            'Subnet/VPC unavailable')
    offering = aws('ec2', 'describe-instance-type-offerings', '--location-type', 'availability-zone', '--filters',
                   f'Name=instance-type,Values={TYPE}', f'Name=location,Values={subnet["AvailabilityZone"]}')
    require(len(offering['InstanceTypeOfferings']) == 1, 'Chosen AZ does not offer m7g.2xlarge; no fallback')
    routes = aws('ec2', 'describe-route-tables', '--filters', f'Name=vpc-id,Values={VPC}')['RouteTables']
    specific = [r for r in routes if any(x.get('SubnetId') == a.subnet_id for x in r['Associations'])]
    route = specific or [r for r in routes if any(x.get('Main') for x in r['Associations'])]
    require(len(route) == 1 and any(x.get('DestinationCidrBlock') == '0.0.0.0/0'
            and x.get('GatewayId', '').startswith('igw-') and x.get('State') == 'active'
            for x in route[0]['Routes']), 'Require direct IGW egress, no NAT expense')
    groups = aws('ec2', 'describe-security-groups', '--filters', f'Name=vpc-id,Values={VPC}',
                 'Name=group-name,Values=trid-aws-eval-006-no-ingress')['SecurityGroups']
    require(len(groups) == 1 and groups[0]['IpPermissions'] == [], 'Security group must have ZERO ingress')
    require(any(p.get('IpProtocol') == '-1' and any(r.get('CidrIp') == '0.0.0.0/0'
                for r in p.get('IpRanges', [])) for p in groups[0]['IpPermissionsEgress']), 'No outbound package/S3 connectivity')
    result.update(subnet=subnet, security_group=groups[0], route_table=route[0])
    profile = aws('iam', 'get-instance-profile', '--instance-profile-name', PROFILE)['InstanceProfile']
    require(profile['Arn'] == f'arn:aws:iam::{ACCOUNT}:instance-profile/{PROFILE}'
            and [x['RoleName'] for x in profile['Roles']] == [PROFILE], 'Unexpected profile role')
    result['instance_profile'] = profile
    result['instance_role'] = role_record(PROFILE, 'ec2.amazonaws.com')
    result['scheduler_role'] = role_record(SCHEDULER, 'scheduler.amazonaws.com')
    verify_role_scopes(result)
    bucket_region = aws('s3api', 'get-bucket-location', '--bucket', BUCKET)['LocationConstraint']
    require(bucket_region == REGION, 'Input/output bucket must be in Seoul')
    public = aws('s3api', 'get-public-access-block', '--bucket', BUCKET)['PublicAccessBlockConfiguration']
    require(all(public.get(k) is True for k in (
        'BlockPublicAcls', 'IgnorePublicAcls', 'BlockPublicPolicy', 'RestrictPublicBuckets')),
        'Private bucket public-access boundary absent')
    result['bucket'] = {'name': BUCKET, 'region': bucket_region, 'public_access_block': public}
    ami_id = aws('ssm', 'get-parameter', '--name',
                 '/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64')['Parameter']['Value']
    ami = aws('ec2', 'describe-images', '--image-ids', ami_id, '--owners', 'amazon')['Images']
    require(len(ami) == 1 and ami[0]['Architecture'] == 'arm64' and ami[0]['State'] == 'available'
            and ami[0]['RootDeviceType'] == 'ebs' and ami[0]['Name'].startswith('al2023-ami-'), 'Invalid AL2023 ARM AMI')
    require(ami[0].get('VirtualizationType') == 'hvm' and not ami[0].get('Platform')
            and len(ami[0]['BlockDeviceMappings']) == 1
            and ami[0]['BlockDeviceMappings'][0]['DeviceName'] == ami[0]['RootDeviceName'],
            'Unexpected AMI platform or extra chargeable devices')
    root_ebs = ami[0]['BlockDeviceMappings'][0].get('Ebs', {})
    require(type(root_ebs.get('VolumeSize')) is int and 0 < root_ebs['VolumeSize'] <= 64,
            'AMI root snapshot exceeds approved 64GiB volume')
    result['ami'] = ami[0]
    ec2, ec2_evidence = price('AmazonEC2', {'regionCode': REGION, 'instanceType': TYPE,
        'operatingSystem': 'Linux', 'tenancy': 'Shared', 'preInstalledSw': 'NA', 'capacitystatus': 'Used'}, 'Hrs')
    ebs, ebs_evidence = price('AmazonEC2', {'regionCode': REGION, 'volumeApiName': 'gp3',
                                        'productFamily': 'Storage'}, 'GB-Mo')
    ip, ip_evidence = price('AmazonVPC', {'regionCode': REGION,
                                        'usagetype': 'APN2-PublicIPv4:InUseAddress'}, 'Hrs')
    transfer, transfer_evidence = price('AWSDataTransfer', {
        'fromLocation': 'Asia Pacific (Seoul)', 'toLocation': 'External',
        'transferType': 'AWS Outbound'}, 'GB')
    storage, storage_evidence = price('AmazonS3', {'regionCode': REGION,
        'productFamily': 'Storage', 'volumeType': 'Standard'}, 'GB-Mo')
    put, put_evidence = price('AmazonS3', {'regionCode': REGION, 'group': 'S3-API-Tier1'}, 'Requests')
    result['cost'] = budget_bound(ec2, ebs, ip, transfer, storage, put)
    result['cost']['price_products'] = (ec2_evidence + ebs_evidence + ip_evidence
                                       + transfer_evidence + storage_evidence + put_evidence)
    result['checked_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    return result


def launch(a):
    global END
    require(a.run_id == RUN_ID, 'Only the exact approved packed run ID is authorized')
    require(re.fullmatch(r'[0-9a-f]{64}', a.bundle_sha256), 'Invalid bundle SHA256')
    expected = f's3://{BUCKET}/{PREFIX}/code/{a.run_id}/inputs.tar'
    require(a.bundle_s3_uri == expected, 'Bundle must belong to exact run input prefix')
    require(a.provenance == PROVENANCE,
            'Only the designated evidence provenance member is accepted')
    state = Path(a.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    receipt = state / f'{a.run_id}.json'
    # Both approval and run are durable one-shot identities, even before preflight.
    exclusive_intent(state / f'{RUN_ID}-approval.intent.json',
                     {'run_id': a.run_id, 'approval': APPROVAL, 'client_token': a.run_id,
                      'bundle_sha256': a.bundle_sha256, 'state': 'approval_consumed'})
    exclusive_intent(receipt, {'run_id': a.run_id, 'state': 'preflight_started'})
    record = preflight(a)
    require(not instances(a.run_id), 'Run ID already exists; never relaunch')
    obj = aws('s3api', 'head-object', '--bucket', BUCKET, '--key', f'{PREFIX}/code/{a.run_id}/inputs.tar')
    require(0 < obj['ContentLength'] <= 2 * 1024 ** 3, 'Bundle exceeds 2GiB input bound')
    require(obj.get('ServerSideEncryption') in ('AES256', 'aws:kms'), 'Input object must be encrypted')
    # Policy simulation supplements live policy evidence; never edits IAM.
    scheduler_arn = f'arn:aws:iam::{ACCOUNT}:role/{SCHEDULER}'
    record['simulations'] = [
        simulate(PROFILE, 's3:GetObject', [
            f'arn:aws:s3:::{BUCKET}/{PREFIX}/code/{a.run_id}/inputs.tar',
            f'arn:aws:s3:::{BUCKET}/{PREFIX}/code/{a.run_id}/release.json']),
        simulate(PROFILE, 's3:PutObject', [
            f'arn:aws:s3:::{BUCKET}/{PREFIX}/runs/{a.run_id}/results.tar',
            f'arn:aws:s3:::{BUCKET}/{PREFIX}/runs/{a.run_id}/completion.json']),
        simulate(SCHEDULER, 'ec2:TerminateInstances', [
            f'arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*'],
            [{'ContextKeyName': 'aws:ResourceTag/trid:eval',
              'ContextKeyValues': ['aws-eval-006'], 'ContextKeyType': 'string'}])]
    now = dt.datetime.now(dt.timezone.utc)
    deadline = now + dt.timedelta(minutes=119)
    # Scheduler minute precision may execute up to 59s after this timestamp.
    scheduled = deadline.replace(second=0, microsecond=0)
    config = {'run_id': a.run_id, 'bundle_s3_uri': expected, 'bundle_sha256': a.bundle_sha256,
              'request_started': now.isoformat(), 'schedule_at': scheduled.isoformat(),
              'deadline_unix': int((now + dt.timedelta(seconds=MAX_SECONDS)).timestamp()),
              'worker_deadline_unix': int((now + dt.timedelta(seconds=MAX_SECONDS - 300)).timestamp()),
              'plan': PLAN, 'allowlist': ALLOWLIST, 'approval': APPROVAL,
              'provenance': a.provenance}
    body = Path(__file__).with_name('qwen_cpu_bootstrap.sh').read_text()
    user_data = ('#!/bin/bash\nif [ "${1:-}" != worker ]; then\n'
                 'shutdown -h +115 "Qwen013 hard lifetime" || { poweroff -f; exit 1; }\nfi\n')
    user_data += 'QWEN_CONFIG=' + shlex.quote(base64.b64encode(json.dumps(config).encode()).decode()) + '\n' + body
    require(len(user_data.encode()) <= 16384, 'EC2 user data exceeds 16KiB')
    request = {'ImageId': record['ami']['ImageId'], 'InstanceType': TYPE, 'MinCount': 1, 'MaxCount': 1,
               'ClientToken': a.run_id, 'IamInstanceProfile': {'Name': PROFILE},
               'InstanceInitiatedShutdownBehavior': 'terminate',
               'MetadataOptions': {'HttpTokens': 'required', 'HttpEndpoint': 'enabled', 'HttpPutResponseHopLimit': 1},
               'NetworkInterfaces': [{'DeviceIndex': 0, 'SubnetId': a.subnet_id,
                   'Groups': [record['security_group']['GroupId']], 'AssociatePublicIpAddress': True,
                   'DeleteOnTermination': True}],
               'BlockDeviceMappings': [{'DeviceName': record['ami']['RootDeviceName'], 'Ebs': {
                   'VolumeSize': 64, 'VolumeType': 'gp3', 'Encrypted': True, 'DeleteOnTermination': True}}],
               'TagSpecifications': [{'ResourceType': kind, 'Tags': tags(a.run_id)} for kind in ('instance', 'volume')],
               # RunInstances' botocore base64_encode_user_data handler also runs
               # for --cli-input-json. Pre-encoding here would double-encode the
               # script and make EC2 apply its raw-byte limit to base64 text.
               # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html#user-data-api-cli
               'UserData': user_data}
    record.update(config=config, request=request, state='launch_intent_saved')
    save(receipt, record)
    instance_id = None
    END = config['deadline_unix']
    try:
        launched = aws('ec2', 'run-instances', '--cli-input-json', json.dumps(request))
        require(len(launched['Instances']) == 1, 'Unexpected instance count')
        instance_id = launched['Instances'][0]['InstanceId']
        owned_identity(launched['Instances'][0], a.run_id, instance_id)
        record.update(instance_id=instance_id, state='launched_without_schedule')
        save(receipt, record)
        target = {'Arn': 'arn:aws:scheduler:::aws-sdk:ec2:terminateInstances', 'RoleArn': scheduler_arn,
                  'Input': json.dumps({'InstanceIds': [instance_id]}),
                  'RetryPolicy': {'MaximumRetryAttempts': 0, 'MaximumEventAgeInSeconds': 60}}
        expression = 'at(' + scheduled.strftime('%Y-%m-%dT%H:%M:%S') + ')'
        aws('scheduler', 'create-schedule', '--name', a.run_id, '--group-name', 'default',
            '--schedule-expression', expression, '--schedule-expression-timezone', 'UTC',
            '--flexible-time-window', '{"Mode":"OFF"}', '--state', 'ENABLED',
            '--action-after-completion', 'DELETE', '--target', json.dumps(target), '--client-token', a.run_id)
        check = aws('scheduler', 'get-schedule', '--name', a.run_id, '--group-name', 'default')
        require(check['State'] == 'ENABLED' and check['ActionAfterCompletion'] == 'DELETE'
                and check['ScheduleExpression'] == expression and check['ScheduleExpressionTimezone'] == 'UTC'
                and check['FlexibleTimeWindow'] == {'Mode': 'OFF'}
                and check['Target']['Arn'] == target['Arn'] and check['Target']['RoleArn'] == scheduler_arn
                and json.loads(check['Target']['Input']) == {'InstanceIds': [instance_id]}
                and check['Target']['RetryPolicy'] == target['RetryPolicy'], 'Scheduler readback mismatch')
        require(time.time() + 600 < config['deadline_unix'], 'Launch/setup took too long for safe admission')
        record.update(schedule=check, state='schedule_verified')
        save(receipt, record)
        release = {**config, 'instance_id': instance_id, 'account': ACCOUNT, 'region': REGION,
                   'schedule_name': a.run_id, 'schedule_verified': True}
        release_file = state / f'{a.run_id}-release.json'
        save(release_file, release)
        aws('s3api', 'put-object', '--bucket', BUCKET, '--key', f'{PREFIX}/code/{a.run_id}/release.json',
            '--body', str(release_file), '--server-side-encryption', 'AES256')
        record['state'] = 'released_fresh_linux_cache_and_two_update_smoke_only'
        save(receipt, record)
        # Successful launch is not cleanup evidence. Remain the owning observer.
        while time.time() < config['deadline_unix'] - 300:
            rows = instances(a.run_id)
            if rows:
                require(len(rows) == 1, 'Unexpected instance count while observing')
                owned_identity(rows[0], a.run_id, instance_id)
                if rows[0]['State']['Name'] == 'terminated':
                    break
            time.sleep(10)
        END = config['deadline_unix'] + 300
        record['cleanup'] = reconcile_cleanup(a.run_id, instance_id, END)
        record['cleanup_state'] = record['cleanup']['cleanup_state']
        record['state'] = ('workflow_terminated_cleanup_complete' if record['cleanup_state'] == 'complete'
                           else 'cleanup_unresolved')
        save(receipt, record)
        require(record['cleanup_state'] == 'complete', 'Owned instance/EBS cleanup remains unresolved')
        try:
            aws('scheduler', 'delete-schedule', '--name', a.run_id, '--group-name', 'default',
                '--client-token', a.run_id)
            record['schedule_cleanup'] = 'deleted_after_observed_termination'
        except RuntimeError as error:
            if 'ResourceNotFoundException' not in str(error):
                raise
            record['schedule_cleanup'] = 'already_absent_after_observed_termination'
        save(receipt, record)
        return {'state': record['state'], 'instance_id': instance_id, 'receipt': str(receipt),
                'schedule': a.run_id, 'cost': record['cost'], 'cleanup': record['cleanup']}
    except BaseException as error:
        record.update(state='failed', error=f'{type(error).__name__}: {error}')
        if record.get('cleanup_state') != 'complete':
            END = min(time.time() + 180, config['deadline_unix'] + 300)
            try:
                record['cleanup'] = reconcile_cleanup(a.run_id, instance_id, END)
                record['cleanup_state'] = record['cleanup']['cleanup_state']
                record['launch_outcome'] = record['cleanup']['launch_outcome']
            except BaseException as cleanup:
                record.update(cleanup_state='cleanup_unresolved',
                              cleanup_error=f'{type(cleanup).__name__}: {cleanup}')
        if record.get('cleanup_state') != 'complete':
            record['state'] = 'cleanup_unresolved'
        save(receipt, record)
        raise
    finally:
        END = None


def frozen_child(plan):
    child = list(plan['argv'][plan['argv'].index('--') + 1:])
    require(child[0] == 'scripts/train_qwen_preferences.py' and len(child) % 2 == 1,
            'Unexpected frozen trainer command')
    require('--ids' not in child and child.count('--out') == 1, 'Unexpected frozen partition/output')
    for option, value in (('--max-steps', '2'), ('--mode', 'train'), ('--batch-pairs', '4'),
                          ('--checkpoint-every', '2'), ('--valid-every', '2')):
        require(child.count(option) == 1 and child[child.index(option) + 1] == value,
                'Only frozen two-update smoke is authorized')
    for option, key in (('--cache', 'path'), ('--cache-sha256', 'sha256')):
        require([child[i + 1] for i in range(1, len(child), 2) if child[i] == option]
                == [row[key] for row in plan['verified_cache_shards']], 'Frozen cache arguments differ')
    return child


def cache_audit(plan):
    """Read immutable accepted metadata/partitions without changing any source scores."""
    rows = plan['verified_cache_shards']
    require(len(rows) == 6, 'Expected six original cache shards')
    old = []
    for row in rows:
        require(digest(row['path']) == row['sha256'], 'Frozen audit cache hash mismatch')
        old.append(json.loads(Path(row['path']).read_text()))
    require([len(s['ids']) for s in old] == [16] * 5 + [11], 'Original partition differs')
    ids = [pid for shard in old for pid in shard['ids']]
    require(len(ids) == len(set(ids)) == 91, 'Original partition must cover 91 unique pairs')
    keys = ('kind', 'manifest_sha256', 'reference', 'base', 'tokenizer', 'family',
            'max_tokens', 'objective', 'replay', 'base_weights', 'init_adapter',
            'admitted_pairs', 'seed', 'device', 'dtype', 'manifest', 'manifest_id')
    expected = {key: old[0][key] for key in keys}
    require(expected['admitted_pairs'] == 91 and expected['replay']['rows_admitted'] == 288,
            'Frozen pair/replay population differs')
    require(all(all(shard[key] == value for key, value in expected.items()) for shard in old),
            'Original immutable audit bindings disagree')
    digests = {pid: shard['scores'][pid]['pair_sha256'] for shard in old for pid in shard['ids']}
    return [shard['ids'] for shard in old], expected, digests


def validate_linux_cache(directory):
    """Nonroot, no model load: frozen verifier over authenticated prior Linux bytes."""
    require(sys.platform == 'linux' and platform.machine() == 'aarch64'
            and sys.version_info[:2] == (3, 12), 'Actual Linux scoring runtime mismatch')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import train_qwen_preferences as trainer
    import torch
    import transformers
    import peft
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)
    require(versions == RUNTIME_VERSIONS and torch.version.cuda is None
            and not torch.cuda.is_available(), 'Actual scoring runtime mismatch')
    plan = json.loads((ROOT / PLAN).read_text())
    reuse = verify_cache_reuse(plan)
    partitions, expected, digests = cache_audit(plan)
    expected = {**expected, 'engine': trainer.engine_identity(versions, 'cpu', torch.float32),
                'versions': versions}
    paths = [Path(row['path']) for row in plan['verified_cache_shards']]
    require(directory == paths[0].parent, 'Cache validator directory differs from approved origin')
    shards = [json.loads(path.read_text()) for path in paths]
    require([s['ids'] for s in shards] == partitions, 'Prior Linux shard partition differs')
    require(all(s.get('engine') == expected['engine'] and s.get('versions') == versions for s in shards),
            'Prior Linux cache current runtime/engine binding mismatch')
    scores = trainer.verify_cache_shards(shards, expected, digests)
    require(len(scores) == len(digests), 'Prior Linux cache did not validate every required pair')
    print(json.dumps({'validated_pairs': len(scores), 'replay_rows': 288,
                      'engine': expected['engine'], **reuse,
                      'shards': [{'path': str(p), 'sha256': digest(p)} for p in paths]}), flush=True)


def verify_cache_reuse(plan, files=None):
    """Authenticate the exact retained Linux cache set; never copy, relabel or regenerate."""
    reuse = plan['cache_reuse']
    require(reuse.get('origin_run_id') == ORIGIN_RUN_ID
            and reuse.get('recompute_on_failure') is False, 'Prior Linux cache reuse policy mismatch')
    origin = ROOT / 'runs' / EXPERIMENT / ORIGIN_RUN_ID
    acceptance_pin = plan['cache_acceptance']
    recovery_pin = reuse['recovery_verification']
    require(acceptance_pin['path'] == str(origin / 'remote-output/linux-cache-acceptance.json')
            and recovery_pin['path'] == str(origin / 'output-recovery-verification.json'),
            'Prior Linux cache provenance path mismatch')
    rows = plan['verified_cache_shards']
    directory = origin / 'remote-output/reference-cache-linux'
    paths = [directory / f'shard-{n:03d}.json' for n in range(1, 7)]
    require([row['path'] for row in rows] == [str(path) for path in paths],
            'Prior Linux cache exact six source paths mismatch')
    require(directory.is_dir() and not directory.is_symlink()
            and set(directory.iterdir()) == set(paths), 'Prior Linux cache extra/missing shards')
    for row in [acceptance_pin, recovery_pin, *rows]:
        path = Path(row['path'])
        require(path.is_file() and not path.is_symlink() and digest(path) == row['sha256'],
                'Prior Linux cache/provenance hash mismatch')
        if files is not None:
            name = path.relative_to(ROOT).as_posix()
            require(name in files and files[name]['sha256'] == row['sha256'],
                    'Prior Linux cache/provenance not authenticated in bundle')
    recovery = json.loads(Path(recovery_pin['path']).read_text())
    require(recovery.get('run_id') == ORIGIN_RUN_ID
            and recovery.get('all_retained_files_hash_verified') is True
            and recovery.get('export_complete') is False
            and isinstance(recovery.get('export_errors'), list) and recovery['export_errors'],
            'Prior Linux recovery identity/incomplete-export evidence mismatch')
    acceptance = json.loads(Path(acceptance_pin['path']).read_text())
    require(acceptance.get('accepted') is True and acceptance.get('pairs') == 91
            and acceptance.get('replay_rows') == 288 and acceptance.get('cache_recomputed') is True
            and acceptance.get('old_scores_reused') is False and acceptance.get('shards') == rows,
            'Prior Linux acceptance/shard binding mismatch')
    cache_audit(plan)
    return {'cache_recomputed': False, 'old_scores_reused': True, 'origin_run_id': ORIGIN_RUN_ID,
            'origin_acceptance': acceptance_pin, 'origin_recovery_verification': recovery_pin,
            'origin_export_complete': False, 'origin_export_errors': recovery['export_errors'],
            'recompute_on_failure': False}


def verify_approval(approval):
    expected = {'run_id': RUN_ID, 'instance_limit': 1, 'region': REGION,
        'instance_type': TYPE, 'max_seconds': MAX_SECONDS, 'additional_cap_usd': ADDITIONAL_CAP_USD,
        'cumulative_cap_usd': CUMULATIVE_CAP_USD, 'prior_conservative_bound_usd': PRIOR_BOUND_USD,
        'automatic_model_or_launch_retry': False, 'instance_or_az_fallback': False,
        'fresh_pricing_and_owned_cleanup_required': True}
    require(all(approval.get(k) == v for k, v in expected.items()), 'Exact one-shot approval mismatch')


def verify_sources(plan, files):
    pins = plan['source_pins']
    require(REQUIRED_SOURCES <= set(pins), 'Required source/test authentication missing')
    require(all(pins.get(path) == sha for path, sha in ENGINE_PINS.items()),
            'Accepted packed engine identity changed')
    for path, sha in pins.items():
        require(path in files and files[path]['sha256'] == sha
                and digest(ROOT / path) == sha, f'Frozen source mismatch: {path}')


def verify_allowlist(allowlist, files):
    rows = allowlist['files']
    names = {row['path'] for row in rows}
    require(allowlist.get('required_root') == str(ROOT) and allowlist.get('run_id') == RUN_ID
            and len(names) == len(rows) == allowlist.get('file_count')
            and allowlist.get('total_bytes') == sum(row['bytes'] for row in rows)
            and 0 < allowlist['total_bytes'] <= 2 * 1024**3
            and set(files) == names | {ALLOWLIST} and ALLOWLIST not in names,
            'Exact current-run allowlist mismatch')
    require(all(files.get(row['path']) == row for row in rows), 'Input allowlist binding changed')


def compatibility_selectors(plan):
    selectors = plan['compatibility_tests']
    require(isinstance(selectors, list) and len(selectors) == len(set(selectors))
            and set(COMPATIBILITY_TESTS[:2]) <= set(selectors) <= set(COMPATIBILITY_TESTS),
            'Only bounded packed-head/cache compatibility selectors are authorized')
    return selectors


def run_model_smoke(output, argv, unit, outcome, py, guard):
    """Dispatch the canonical smoke exactly once through the guard and external tracer."""
    from qwen_cpu_pipeline import traced_training_argv
    name = 'model-smoke'
    receipt = output / f'{name}-supervisor.json'
    trace_path = output / 'model-smoke-phases.jsonl'
    dispatched = traced_training_argv(py, argv, trace_path)
    entry = {'name': name, 'mode': 'train', 'state': 'dispatch_requested',
             'canonical_argv': argv, 'dispatched_argv': dispatched, 'trace_path': str(trace_path),
             'trainer_started_observed': False, 'model_load_observed': False}
    require(outcome['training_children_requested'] == 0, 'Smoke training already requested; no retry')
    outcome['invocations'].append(entry)
    outcome['training_children_requested'] += 1
    outcome['state'] = name
    save(output / 'smoke-outcome.json', outcome)
    rc = None
    result = {}
    try:
        rc = unit(name, [py, '-B', guard, '--receipt', str(receipt), '--seconds', '180',
                         '--', *dispatched], 180)
    finally:
        if receipt.is_file():
            result = json.loads(receipt.read_text())
        loaded = False
        log_path = output / f'{name}.log'
        if log_path.is_file():
            with log_path.open() as log:
                for line in log:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and 'weight_loading' in event:
                        loaded = True
        entry.update(state='returned' if rc is not None else 'interrupted',
                     returncode=rc, trainer_started_observed=result.get('startup_released') is True,
                     model_load_observed=loaded, guard_receipt_present=bool(result))
        outcome['model_invocations'] = sum(x['trainer_started_observed'] for x in outcome['invocations'])
        outcome['model_loads_observed'] = sum(x['model_load_observed'] for x in outcome['invocations'])
        save(output / 'smoke-outcome.json', outcome)
    require(rc == 0 and result.get('startup_released') is True
            and result.get('stop_reason') == 'child_exit' and result.get('child_exit_code') == 0
            and result.get('reaped') is True and result.get('cleanup_state') == 'completed'
            and result.get('cleanup_errors') == [] and result.get('live_members_before_reap') == []
            and result.get('first_stop_reason') is None and result.get('seconds') == 180
            and result.get('argv') == dispatched
            and 0 <= result.get('elapsed_seconds', -1) <= 180
            and 0 < result.get('rss_peak_bytes', 0) <= 6 * 1024 ** 3,
            f'{name} failed/stopped or receipt incomplete; retained evidence, no retry')
    samples = result.get('samples', [])
    require(samples and all(s['cgroup']['memory_max_bytes'] == 6 * 1024 ** 3
            and s['cgroup']['swap_max_bytes'] == s['cgroup']['swap_current_bytes'] == 0
            and s['rss_bytes'] <= 6 * 1024 ** 3
            and all(s['cgroup']['events'].get(k, 0) == 0 for k in ('max', 'oom', 'oom_kill', 'oom_group_kill'))
            and 'memory_psi' in s['host'] for s in samples), 'Invalid native resource evidence')
    require(loaded, f'{name} completed without observed real model loading')


def run_unit(config, output, name, command, seconds, constrained=True, allow_loopback=False):
    """Own the exclusive dispatch receipt, separately from pipeline plan metadata."""
    run_id = config['run_id']
    # Total unit allowance includes systemd's five-second termination grace.
    # Return before remote-stage TERM at W-305; its five-second grace is not work time.
    require(time.time() + seconds + 15 + 305 < config['worker_deadline_unix'],
            'Insufficient worker lifetime for child, unit return, wrapper TERM grace, and export reserve')
    full = ['systemd-run', '--quiet', '--wait', '--collect', '--pipe',
            '--unit', f'{run_id}-{name}', '--uid', 'qwen-smoke',
            '--property', f'WorkingDirectory={ROOT}', '--property', 'KillMode=control-group',
            '--property', f'RuntimeMaxSec={max(1, seconds - 5)}', '--property', 'TimeoutStopSec=5',
            '--property', 'BindsTo=qwen013-bootstrap.service',
            '--property', 'After=qwen013-bootstrap.service',
            '--property', 'NoNewPrivileges=yes', '--property', 'ProtectSystem=strict',
            '--property', f'ReadWritePaths={output} /tmp', '--property', 'ProtectHome=yes',
            '--property', 'PrivateDevices=yes', '--property', 'RestrictSUIDSGID=yes',
            '--property', 'IPAddressDeny=any', '--property', 'UMask=0077',
            '--setenv', 'PYTHONDONTWRITEBYTECODE=1', '--setenv', 'CUDA_VISIBLE_DEVICES=',
            '--setenv', 'HF_HUB_OFFLINE=1', '--setenv', 'TRANSFORMERS_OFFLINE=1',
            '--setenv', 'TOKENIZERS_PARALLELISM=false', '--setenv', 'OMP_NUM_THREADS=8',
            '--setenv', 'MKL_NUM_THREADS=8', '--setenv', f'HF_HOME={output}/hf-cache',
            '--setenv', 'AWS_EC2_METADATA_DISABLED=true']
    if constrained:
        full += ['--property', 'MemoryMax=6G', '--property', 'MemorySwapMax=0']
    if allow_loopback:
        require(name == 'native-probe-train' or name.startswith('native-case-'),
                'Loopback exception is only for native probe/case HTTP')
        full += ['--property', 'IPAddressAllow=localhost']
    command_record = {'argv': full + ['--'] + command, 'state': 'dispatch_requested'}
    path = output / f'{name}-command.json'
    require(not path.exists(), 'Unit already dispatched; no retries')
    exclusive_intent(path, command_record)
    try:
        with open(output / f'{name}.log', 'xb') as log:
            result = subprocess.run([*full, '--', *command], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=seconds + 15, env={'PATH': os.environ['PATH'], 'LANG': 'C.UTF-8'})
    except BaseException as error:
        # Only this exact owned unit; never enumerate/kill unrelated services.
        stopped = subprocess.run(['systemctl', 'stop', f'{run_id}-{name}.service'],
                                 capture_output=True, timeout=10)
        command_record.update(state='orchestrator_error', error=type(error).__name__,
                              cleanup_returncode=stopped.returncode)
        save(path, command_record)
        raise
    command_record.update(state='finished', returncode=result.returncode)
    save(path, command_record)
    return result.returncode


def remote_stage():
    """Root orchestrator; all code/tests/model children use dedicated nonroot units."""
    import pwd
    import urllib.request
    config = json.loads(Path('/var/lib/qwen013/config.json').read_text())
    run_id = config['run_id']
    require(sys.platform == 'linux' and os.geteuid() == 0, 'Remote Linux root orchestrator required')
    require(Path.cwd() == ROOT, 'Lexical project root mismatch')
    require(run_id == RUN_ID and config.get('plan') == PLAN
            and config.get('allowlist') == ALLOWLIST and config.get('approval') == APPROVAL
            and config.get('provenance') == PROVENANCE, 'Invalid exact run metadata identity')
    require(config.get('worker_deadline_unix') == config['deadline_unix'] - 300,
            'Worker deadline must precede controller termination by 300 seconds')
    output = ROOT / RUN_PATH / 'remote-output'
    require(not output.exists(), 'Output must be fresh')
    output.mkdir(mode=0o755)
    user = pwd.getpwnam('qwen-smoke')
    os.chown(output, user.pw_uid, user.pw_gid)
    outcome = {'state': 'pre_model_checks', 'cache_children_requested': 0,
               'training_children_requested': 0, 'model_invocations': 0,
               'model_loads_observed': 0, 'invocations': [],
               'full_training_started': False, 'native_generation_started': False}
    save(output / 'smoke-outcome.json', outcome)
    py = '/opt/qwen013-venv/bin/python'
    guard = str(ROOT / 'scripts/qwen_linux_guard.py')
    bundle = json.loads(Path('/var/lib/qwen013/verified-manifest.json').read_text())
    files = {r['path']: r for r in bundle['files']}
    plan = json.loads((ROOT / PLAN).read_text())
    verify_sources(plan, files)
    for key in ('manifest', 'cache_acceptance'):
        row = plan[key]
        require(digest(row['path']) == row['sha256'], f'Frozen {key} mismatch')
    verify_allowlist(json.loads((ROOT / ALLOWLIST).read_text()), files)
    approval = json.loads((ROOT / APPROVAL).read_text())
    verify_approval(approval)
    reuse = verify_cache_reuse(plan, files)
    provenance = ROOT / config['provenance']
    require(provenance.relative_to(ROOT).as_posix() in files, 'Provenance not verified')
    evidence = json.loads(provenance.read_text())
    require(isinstance(evidence, dict) and evidence.get('required_root') == str(ROOT)
            and evidence.get('run_id') == run_id, 'Provenance run/root mismatch')
    save(output / 'provenance.json', evidence)
    save(output / 'input-verification.json', {'bundle_sha256': config['bundle_sha256'],
         'files': bundle['files'], 'source_pins': plan['source_pins'],
         'guard_sha256': digest(guard), 'old_macos_guard_not_used': True,
         'cache_recomputed_at_input_verification': False,
         'cache_policy': 'authenticated prior Linux cache reuse; no regeneration fallback',
         'cache_reuse': reuse,
         'pressure_value_fabricated': False})

    def unit(name, command, seconds, constrained=True, allow_loopback=False):
        return run_unit(config, output, name, command, seconds,
                        constrained=constrained, allow_loopback=allow_loopback)

    def probe(name, code, seconds, expected, constrained=True):
        receipt = output / f'{name}-receipt.json'
        rc = unit(name, [py, '-B', guard, '--receipt', str(receipt), '--seconds', str(seconds),
                         '--', py, '-I', '-B', '-c', code], seconds + 10, constrained)
        result = json.loads(receipt.read_text())
        require(result['stop_reason'] == expected and result['cleanup_state'] == 'completed'
                and not result['cleanup_errors'] and not result['live_members_before_reap'],
                f'Guard probe {name} failed')
        if constrained:
            require(result['startup_released'] and result['reaped'], f'Probe {name} did not run/reap')
        else:
            require(not result['startup_released'] and 'child_pid' not in result and rc != 0,
                    'Unconstrained child admitted')
        if expected == 'child_exit':
            require(rc == 0 and result['child_exit_code'] == 0, 'Success probe failed')
        if expected == 'deadline':
            require(rc != 0 and result['elapsed_seconds'] <= seconds + 5, 'Deadline not enforced')
        return result

    require(unit('guard-unittest', [py, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                                   '-p', 'test_qwen_linux_guard.py', '-v'], 180) == 0,
            'Guard unittest failed; no model')
    marker = output / 'outside-child-must-not-exist'
    probe('outside', f'from pathlib import Path; Path({str(marker)!r}).touch()', 15, 'guard_error', False)
    require(not marker.exists(), 'Unconstrained child executed')
    probe('inside', 'print("real Linux constrained child")', 15, 'child_exit')
    sleeper = ('import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
               'pid=os.fork(); print(pid,flush=True); time.sleep(120)')
    probe('deadline', sleeper, 8, 'deadline')
    descendant = ('import os,signal,time; pid=os.fork(); '
                  'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                  'os._exit(0) if pid else time.sleep(120)')
    probe('descendants', descendant, 15, 'leader_exited_with_live_descendants')
    environment_code = (
        'import importlib.metadata as m,json,os,platform,sys,torch,transformers,peft; '
        f'pins={PACKAGES!r}; '
        'versions={p:m.version(p) for p in pins}; '
        'assert versions==pins,(versions,pins); '
        'assert sys.version_info[:2]==(3,12); '
        f'assert os.getuid()==os.geteuid()=={user.pw_uid} and os.getuid()!=0; '
        'assert platform.machine()==\"aarch64\"; '
        'assert torch.version.cuda is None and not torch.cuda.is_available(); '
        'runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__); '
        f'assert runtime=={RUNTIME_VERSIONS!r}; '
        'print(json.dumps({\"versions\":versions,\"python\":sys.version,'
        '\"architecture\":platform.machine(),\"torch_runtime_version\":torch.__version__,'
        '\"runtime_versions\":runtime,\"cpu_only\":True}))')
    require(unit('environment', [py, '-I', '-B', '-c', environment_code], 60) == 0,
            'Pinned environment or CPU-only check failed')
    selectors = compatibility_selectors(plan)
    compatibility_code = (
        'import sys,unittest; sys.path.insert(0,"tests"); '
        f'suite=unittest.defaultTestLoader.loadTestsFromNames({selectors!r}); '
        'result=unittest.TextTestRunner(verbosity=2).run(suite); '
        'sys.exit(0 if result.testsRun>0 and result.wasSuccessful() and not result.skipped else 1)')
    require(unit('packed-compatibility', [py, '-B', '-c', compatibility_code], 180) == 0,
            'Packed-head/cache compatibility regression failed; no actual weights')
    child = frozen_child(plan)
    # Release is a separate launcher write after get-schedule readback, NOT bundled authority.
    token_request = urllib.request.Request('http://169.254.169.254/latest/api/token', method='PUT',
                                          headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
    with urllib.request.urlopen(token_request, timeout=5) as response:
        token = response.read().decode()
    request = urllib.request.Request('http://169.254.169.254/latest/dynamic/instance-identity/document',
                                     headers={'X-aws-ec2-metadata-token': token})
    with urllib.request.urlopen(request, timeout=5) as response:
        machine = json.load(response)
    require(machine['accountId'] == ACCOUNT and machine['region'] == REGION
            and machine['instanceType'] == TYPE, 'Remote instance identity mismatch')
    release_path = Path('/var/lib/qwen013/release.json')
    # Exactly one bounded fetch; setup has given launcher ample time. Missing release fails closed.
    aws('s3api', 'get-object', '--bucket', BUCKET, '--key',
        f'{PREFIX}/code/{run_id}/release.json', str(release_path))
    release = json.loads(release_path.read_text())
    require(all(release.get(k) == v for k, v in config.items())
            and release.get('instance_id') == machine['instanceId']
            and release.get('account') == ACCOUNT and release.get('region') == REGION
            and release.get('schedule_name') == run_id and release.get('schedule_verified') is True
            and time.time() + 1500 < config['worker_deadline_unix'], 'Missing/mismatched/late scheduler release')
    save(output / 'release-verification.json', release)

    cache_dir = Path(plan['verified_cache_shards'][0]['path']).parent
    # Read-only nonroot validation binds all retained scores to this actual runtime.
    require(unit('cache-validation', [py, '-B', str(Path(__file__).resolve()),
                 'validate-linux-cache', str(cache_dir)], 120) == 0,
            'Prior Linux cache validation failed; no model dispatched and no regeneration')
    cache_shards = plan['verified_cache_shards']
    save(output / 'linux-cache-acceptance.json', {'accepted': True, 'pairs': 91, 'replay_rows': 288,
         **reuse, 'shards': cache_shards, 'validation_log': 'cache-validation.log'})
    # Only output changes. The trainer independently re-encodes current data and
    # checks all cache bindings before model load, retaining the unchanged 1e-3 gate.
    train = list(child)
    train[train.index('--out') + 1] = str(output / 'train')
    from qwen_cpu_pipeline import traced_training_argv
    trace_path = output / 'model-smoke-phases.jsonl'
    save(output / 'smoke-command.json', {'original_argv': plan['argv'], 'child_argv': train,
         'dispatched_argv': traced_training_argv(py, train, trace_path), 'trace_path': str(trace_path),
         'child_changes': ['--out fresh remote subtree'],
         'guard_seconds': 180, 'maximum_optimizer_updates': 2,
         'initial_agreement': 'original trainer, unchanged 1e-3 tolerance'})
    run_model_smoke(output, train, unit, outcome, py, guard)
    training = json.loads((output / 'train/training.json').read_text())
    require(training['status'] == 'complete' and training['steps'] == 2
            and training['device'] == 'cpu' and training['dtype'] == 'torch.float32'
            and training['consumption']['pairs_consumed_through_step'] == plan['expected_pairs']
            and training['consumption']['replay_rows_consumed_through_step'] == plan['expected_replay_rows']
            and training['initial_check']['tolerance'] == 1e-3
            and training['initial_check']['initial_cache_max_abs_error'] <= 1e-3,
            'Original trainer output did not satisfy frozen smoke acceptance')
    save(output / 'smoke-acceptance.json', {'accepted': True, 'updates': 2,
         'training_sha256': digest(output / 'train/training.json'),
         'initial_check': training['initial_check'], 'consumption': training['consumption']})
    outcome['state'] = 'accepted_two_update_smoke'
    save(output / 'smoke-outcome.json', outcome)
    from qwen_cpu_pipeline import continue_after_smoke
    continue_after_smoke(config, output, plan, train, cache_shards, unit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'launch', 'status', 'terminate'):
        p = sub.add_parser(name)
        p.add_argument('--allow-console-root', action='store_true')
        if name in ('preflight', 'launch'):
            p.add_argument('--subnet-id', required=True)
            p.add_argument('--approved-usd', type=float, required=True)
            p.add_argument('--max-seconds', type=int, required=True)
        if name != 'preflight':
            p.add_argument('--run-id', required=True)
        if name == 'launch':
            p.add_argument('--bundle-s3-uri', required=True)
            p.add_argument('--bundle-sha256', required=True)
            p.add_argument('--state-dir', default='qwen013-cloud-state')
            p.add_argument('--provenance', default=PROVENANCE)
    a = parser.parse_args()
    if a.command == 'preflight':
        result = preflight(a)
    elif a.command == 'launch':
        result = launch(a)
    else:
        identity(a.allow_console_root)
        require(a.run_id == RUN_ID, 'Only the exact approved packed run ID is authorized')
        owned = instances(a.run_id)
        result = {'instances': owned}
        if a.command == 'terminate':
            result['cleanup'] = reconcile_cleanup(a.run_id, None, time.time() + 180)
        else:
            result['outputs'] = aws('s3api', 'list-objects-v2', '--bucket', BUCKET,
                                     '--prefix', f'{PREFIX}/runs/{a.run_id}/')
            try:
                result['schedule'] = aws('scheduler', 'get-schedule', '--name', a.run_id, '--group-name', 'default')
            except RuntimeError as error:
                if 'ResourceNotFoundException' not in str(error):
                    raise
                result['schedule'] = {'state': 'absent_or_auto_deleted'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['remote-stage']:
        remote_stage()
    elif len(sys.argv) == 3 and sys.argv[1] == 'validate-linux-cache':
        validate_linux_cache(Path(sys.argv[2]))
    else:
        main()
