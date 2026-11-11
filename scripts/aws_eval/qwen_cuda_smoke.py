"""One explicitly approved CUDA run; no launch/model retries or runtime fallback.

Admission, approval and input archive are SHA-bound. AMI, driver, disk and all
price inputs come from authenticated admission, rechecked before RunInstances.
The old CPU controller is imported only for identity-independent pure helpers.
"""
import argparse
import base64
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET

if __package__:
    from . import qwen_cpu_smoke as CPU
else:
    import qwen_cpu_smoke as CPU

ACCOUNT, REGION, BUCKET, PREFIX = CPU.ACCOUNT, CPU.REGION, CPU.BUCKET, CPU.PREFIX
ROOT, EXPERIMENT, VPC = CPU.ROOT, CPU.EXPERIMENT, CPU.VPC
PROFILE, SCHEDULER = CPU.PROFILE, CPU.SCHEDULER
RUN_ID = 'qwen013-cuda-20261111015752'
TYPE = 'g6.xlarge'
SUBNET = 'subnet-025ee7d30ecf0228c'
AZ = 'ap-northeast-2a'
RUN_PATH = f'runs/{EXPERIMENT}/{RUN_ID}'
PLAN, ALLOWLIST, APPROVAL, PROVENANCE = [f'{RUN_PATH}/{name}.json' for name in
                                      ('execution-plan', 'input-allowlist', 'approval', 'provenance')]
MAX_SECONDS, MAX_INPUT, MAX_OUTPUT = 7200, 2 * 1024**3, 512 * 1024**2
PRIOR_BOUND_USD, ADDITIONAL_CAP_USD, CUMULATIVE_CAP_USD = 21.772065718278096, 4.0, 26.0
TRAINER = 'scripts/train_qwen_preferences_cuda.py'
BOOTSTRAP = 'scripts/aws_eval/qwen_cuda_bootstrap.sh'
GENERATED_SMOKE = f'{RUN_PATH}/generated-cuda-smoke.py'
PACKAGES = {**CPU.PACKAGES, 'torch': '2.9.1+cu128'}
REQUIRED_SOURCES = CPU.REQUIRED_SOURCES | {
    TRAINER, BOOTSTRAP, GENERATED_SMOKE, 'scripts/aws_eval/qwen_cuda_smoke.py',
    'tests/test_train_qwen_preferences_cuda.py', 'tests/test_qwen_cuda_smoke.py'}
require, digest, save, exclusive_intent = CPU.require, CPU.digest, CPU.save, CPU.exclusive_intent


def read_bound(path, sha):
    require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'Invalid SHA256')
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and digest(path) == sha,
            f'Authenticated input mismatch: {path}')
    return json.loads(path.read_text())


def validate_admission(a, *, now=None):
    require(a.get('schema') == 'trid-cuda-admission-v1' and a.get('run_id') == RUN_ID,
            'Admission identity mismatch')
    checked = dt.datetime.fromisoformat(a['checked_at'])
    require(checked.tzinfo is not None, 'Admission timestamp needs timezone')
    age = (time.time() if now is None else now) - checked.timestamp()
    require(0 <= age <= 3600, 'Admission must be fresh within one hour')
    ami = a['ami']
    require(re.fullmatch('ami-[0-9a-f]{8,17}', ami['ImageId'])
            and re.fullmatch('[0-9]{12}', ami['OwnerId'])
            and isinstance(ami['Name'], str) and 'Ubuntu' in ami['Name']
            and '24.04' in ami['Name']
            and re.fullmatch('snap-[0-9a-f]{8,17}', ami['SnapshotId'])
            and ami['RootDeviceName'].startswith('/dev/'), 'Invalid selected NVIDIA Ubuntu AMI')
    require(type(ami['root_volume_gib']) is int and type(a['root_volume_gib']) is int
            and 1 <= ami['root_volume_gib'] <= a['root_volume_gib'] <= 256,
            'Root volume must cover authenticated AMI root mapping')
    runtime = a['runtime']
    require(runtime['packages'] == PACKAGES
            and runtime['torch_index_url'] == 'https://download.pytorch.org/whl/cu128'
            and runtime['torch_runtime_version'] == '2.9.1+cu128'
            and runtime['cuda_version'] == '12.8'
            and runtime['minimum_driver_version'] == '570.26'
            and runtime['driver_compatibility_source'] ==
                'https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-driver'
            and runtime['gpu_model'] == 'L4' and runtime['gpu_count'] == 1
            and runtime['gpu_min_total_bytes'] == 6 * 1024**3
            and runtime['gpu_max_total_bytes'] == 24 * 1024**3
            and runtime['torch_wheel_url'] ==
                'https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
            and runtime['torch_wheel_sha256'] ==
                '7cb4018f4ce68b61fd3ef87dc1c4ca520731c7b5b200e360ad47b612d7844063',
            'Pinned CUDA package/compatibility admission mismatch')
    require(set(a['prices']) == {'ec2', 'ebs', 'ip', 'transfer', 'storage', 'put'},
            'All six live price inputs required')
    budget_bound(a['prices'], a['root_volume_gib'])
    return a


def verify_approval(approval, admission_sha256):
    expected = dict(run_id=RUN_ID, instance_limit=1, region=REGION, instance_type=TYPE,
        subnet_id=SUBNET, availability_zone=AZ, max_seconds=MAX_SECONDS,
        additional_cap_usd=ADDITIONAL_CAP_USD, cumulative_cap_usd=CUMULATIVE_CAP_USD,
        prior_conservative_bound_usd=PRIOR_BOUND_USD, automatic_model_or_launch_retry=False,
        instance_or_az_fallback=False, fresh_pricing_and_owned_cleanup_required=True,
        admission_sha256=admission_sha256)
    require(all(approval.get(k) == v for k, v in expected.items()), 'Exact CUDA approval mismatch')


def budget_bound(prices, root_gib):
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in prices.values()), 'Prices must be observed positive finite USD rates')
    hours = (MAX_SECONDS + 300) / 3600
    additional = (hours * (prices['ec2'] + prices['ip'])
                  + root_gib * prices['ebs'] * hours / (28 * 24)
                  + MAX_OUTPUT / 10**9 * prices['transfer']
                  + (MAX_INPUT + MAX_OUTPUT) / 10**9 * prices['storage'] + 1.0)
    total = PRIOR_BOUND_USD + additional
    require(prices['put'] * 10000 <= 1.0, 'Request reserve insufficient')
    require(additional <= ADDITIONAL_CAP_USD and total <= CUMULATIVE_CAP_USD,
            'Conservative additional/cumulative budget exceeds approval')
    return dict(bound_usd=total, prior_bound_usd=PRIOR_BOUND_USD,
                additional_bound_usd=additional, compute_hours=hours, ebs_gb=root_gib,
                prices=prices, input_cap_bytes=MAX_INPUT, output_cap_bytes=MAX_OUTPUT,
                cleanup_allowance_seconds=300, retention_assumption_days=30,
                request_control_reserve_usd=1.0, invoice=False)


class Client:
    """Deadline is local to this controller, never CPU.END or another module global."""
    def __init__(self, deadline=None):
        self.deadline = deadline

    def __call__(self, service, operation, *args, region=REGION):
        seconds = 90 if self.deadline is None else min(90, self.deadline - time.time())
        require(seconds > 0, 'Controller deadline exhausted')
        result = subprocess.run(['aws', service, operation, *args, '--region', region,
            '--output', 'json', '--no-cli-pager', '--cli-connect-timeout', '10',
            '--cli-read-timeout', '30'], capture_output=True, text=True, timeout=seconds,
            env={**os.environ, 'AWS_PAGER': '', 'AWS_MAX_ATTEMPTS': '1', 'AWS_RETRY_MODE': 'standard'})
        require(result.returncode == 0, f'{service} {operation}: {result.stderr[:1000]}')
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def identity(self, allow_root):
        require(os.environ.get('AWS_EXECUTION_ENV') == 'CloudShell'
                and not os.environ.get('AWS_PROFILE')
                and (not os.environ.get('AWS_ACCESS_KEY_ID') or os.environ.get('AWS_SESSION_TOKEN')),
                'Temporary signed-in CloudShell credentials required')
        me = self('sts', 'get-caller-identity')
        require(me['Account'] == ACCOUNT and (me['Arn'].startswith(f'arn:aws:sts::{ACCOUNT}:assumed-role/')
                or allow_root and me['Arn'] == f'arn:aws:iam::{ACCOUNT}:root'), 'Caller not approved')
        return me

    def instances(self, run):
        data = self('ec2', 'describe-instances', '--filters',
                    'Name=tag:trid:eval,Values=aws-eval-006',
                    f'Name=tag:trid:experiment,Values={EXPERIMENT}', f'Name=tag:trid:run,Values={run}')
        return [item for row in data['Reservations'] for item in row['Instances']]

    def role(self, name):
        role = self('iam', 'get-role', '--role-name', name)['Role']
        require(role['Arn'] == f'arn:aws:iam::{ACCOUNT}:role/{name}', 'Unexpected role ARN')
        inline = {}
        listed = self('iam', 'list-role-policies', '--role-name', name)
        require(not listed.get('IsTruncated'), 'Incomplete role policy evidence')
        for policy in listed['PolicyNames']:
            inline[policy] = self('iam', 'get-role-policy', '--role-name', name,
                                 '--policy-name', policy)['PolicyDocument']
        attached = []
        listed = self('iam', 'list-attached-role-policies', '--role-name', name)
        require(not listed.get('IsTruncated'), 'Incomplete attached policy evidence')
        for item in listed['AttachedPolicies']:
            info = self('iam', 'get-policy', '--policy-arn', item['PolicyArn'])['Policy']
            version = self('iam', 'get-policy-version', '--policy-arn', item['PolicyArn'],
                           '--version-id', info['DefaultVersionId'])['PolicyVersion']
            attached.append({'arn': item['PolicyArn'], 'document': version['Document']})
        return dict(role=role, inline=inline, attached=attached)

    def simulate(self, name, action, resources, context=None):
        args = ['--policy-source-arn', f'arn:aws:iam::{ACCOUNT}:role/{name}',
                '--action-names', action, '--resource-arns', *resources]
        if context:
            args += ['--context-entries', json.dumps(context)]
        result = self('iam', 'simulate-principal-policy', *args)
        require(not result.get('IsTruncated'), 'Incomplete policy simulation')
        decisions = []
        for row in result['EvaluationResults']:
            require(row.get('EvalActionName') == action and row.get('EvalDecision') == 'allowed'
                    and not row.get('MissingContextValues'), 'Policy simulation rejected action')
            for specific in row.get('ResourceSpecificResults') or [row]:
                require(not specific.get('MissingContextValues'), 'Missing policy resource context')
                decisions.append((specific.get('EvalResourceName'),
                                  specific.get('EvalResourceDecision', specific.get('EvalDecision'))))
        require(len(decisions) == len(resources) and {r for r, _ in decisions} == set(resources)
                and all(d == 'allowed' for _, d in decisions), 'Policy did not allow exact resources')
        return result

    def price(self, service, filters, unit):
        data = self('pricing', 'get-products', '--service-code', service, '--filters',
                    json.dumps([dict(Type='TERM_MATCH', Field=k, Value=v) for k, v in filters.items()]),
                    region='us-east-1')
        require(not data.get('NextToken'), 'Incomplete price evidence')
        products = [json.loads(s) for s in data['PriceList']]
        values = [float(d['pricePerUnit']['USD']) for p in products
                  for t in p.get('terms', {}).get('OnDemand', {}).values()
                  for d in t['priceDimensions'].values() if d['unit'] == unit]
        require(values and all(math.isfinite(v) and v >= 0 for v in values)
                and max(values) > 0, 'Missing finite positive price evidence')
        return max(values), products


def conservative_gpu_quota(client):
    """Read-only contract inventory, including stopped G/VT GPUs and all markets."""
    quota = client('service-quotas', 'get-service-quota', '--service-code', 'ec2',
                   '--quota-code', 'L-DB2E81BA')['Quota']['Value']
    states = ('pending', 'running', 'stopping', 'stopped')
    inventory = client('ec2', 'describe-instances', '--filters',
                       'Name=instance-state-name,Values=' + ','.join(states))
    instances = [item for reservation in inventory['Reservations'] for item in reservation['Instances']
                 if item['InstanceType'].split('.')[0].startswith(('g', 'vt'))]
    require(all(item['State']['Name'] in states for item in instances),
            'GPU quota inventory contains unexpected states')
    sizes = {}
    for kind in sorted({item['InstanceType'] for item in instances}):
        rows = client('ec2', 'describe-instance-types', '--instance-types', kind)['InstanceTypes']
        require(len(rows) == 1 and rows[0]['InstanceType'] == kind
                and type(rows[0]['VCpuInfo']['DefaultVCpus']) is int
                and rows[0]['VCpuInfo']['DefaultVCpus'] > 0, 'Invalid GPU quota type evidence')
        sizes[kind] = rows[0]['VCpuInfo']['DefaultVCpus']
    used = sum(sizes[item['InstanceType']] for item in instances)
    require(used + 4 <= quota, 'Insufficient conservative G/VT quota including stopped GPUs')
    return dict(limit=quota, used_vcpu_conservative=used, quota_code='L-DB2E81BA',
                counted_states=list(states), all_markets_counted=True,
                instances=[dict(instance_id=item['InstanceId'], instance_type=item['InstanceType'],
                                state=item['State']['Name'], vcpus=sizes[item['InstanceType']])
                           for item in instances])


def validate_ami_mappings(ami, pin):
    """Only the pinned root EBS is chargeable; distinct instance-store aliases are allowed."""
    mappings = ami.get('BlockDeviceMappings')
    require(ami.get('RootDeviceName') == pin['RootDeviceName']
            and isinstance(mappings, list) and mappings, 'Missing/mismatched AMI root mappings')
    devices, virtuals, roots = set(), set(), []
    for mapping in mappings:
        require(isinstance(mapping, dict), 'Malformed AMI block-device mapping')
        device = mapping.get('DeviceName')
        require(isinstance(device, str) and re.fullmatch(r'/dev/[A-Za-z0-9][A-Za-z0-9._-]*', device),
                'Malformed AMI device name')
        # EC2 sd/xvd aliases and partition spellings must not describe competing disks.
        disk = re.fullmatch(r'/dev/(?:sd|xvd)([a-z]+)[0-9]*', device)
        identity = ('linux-disk', disk.group(1)) if disk else ('device', device)
        require(identity not in devices, 'Duplicate/conflicting AMI devices')
        devices.add(identity)
        if set(mapping) == {'DeviceName', 'Ebs'}:
            ebs = mapping['Ebs']
            require(device == pin['RootDeviceName'] and isinstance(ebs, dict)
                    and ebs.get('SnapshotId') == pin['SnapshotId']
                    and type(ebs.get('VolumeSize')) is int
                    and ebs['VolumeSize'] == pin['root_volume_gib'],
                    'Additional EBS or changed pinned AMI root mapping')
            roots.append(mapping)
        else:
            require(set(mapping) == {'DeviceName', 'VirtualName'}
                    and device != pin['RootDeviceName']
                    and isinstance(mapping['VirtualName'], str)
                    and re.fullmatch(r'ephemeral(?:0|[1-9][0-9]*)', mapping['VirtualName']),
                    'Only well-formed instance-store descriptors may accompany root EBS')
            require(mapping['VirtualName'] not in virtuals, 'Duplicate AMI instance-store descriptor')
            virtuals.add(mapping['VirtualName'])
    require(len(roots) == 1, 'Exactly one pinned root EBS mapping required')
    return roots[0]


def preflight(args, client=None):
    c = client or Client()
    admission = validate_admission(read_bound(args.admission, args.admission_sha256))
    approval = read_bound(args.approval, args.approval_sha256)
    verify_approval(approval, args.admission_sha256)
    require(args.max_seconds == MAX_SECONDS and args.approved_usd == CUMULATIVE_CAP_USD,
            'Explicit 7200 seconds / cumulative USD26 required')
    r = dict(identity=c.identity(args.allow_console_root), admission=admission,
             admission_sha256=args.admission_sha256, approval_sha256=args.approval_sha256)
    rows = c('ec2', 'describe-instance-types', '--instance-types', TYPE)['InstanceTypes']
    require(len(rows) == 1, 'Missing exact instance hardware')
    info = rows[0]
    gpu = info.get('GpuInfo', {}).get('Gpus', [])
    require(info['VCpuInfo']['DefaultVCpus'] == 4 and info['MemoryInfo']['SizeInMiB'] == 16384
            and info['ProcessorInfo']['SupportedArchitectures'] == ['x86_64']
            and len(gpu) == 1 and gpu[0]['Name'] == 'L4' and gpu[0]['Count'] == 1,
            'Unexpected g6.xlarge hardware')
    r['type'] = info
    r['quota'] = conservative_gpu_quota(c)
    subnet = c('ec2', 'describe-subnets', '--subnet-ids', SUBNET)['Subnets']
    require(len(subnet) == 1 and subnet[0]['VpcId'] == VPC and subnet[0]['State'] == 'available'
            and subnet[0]['AvailabilityZone'] == AZ and subnet[0]['AvailableIpAddressCount'] > 0,
            'Exact approved subnet unavailable')
    offer = c('ec2', 'describe-instance-type-offerings', '--location-type', 'availability-zone',
              '--filters', f'Name=instance-type,Values={TYPE}', f'Name=location,Values={AZ}')
    require(len(offer['InstanceTypeOfferings']) == 1, 'Exact AZ unavailable; no fallback')
    routes = c('ec2', 'describe-route-tables', '--filters', f'Name=vpc-id,Values={VPC}')['RouteTables']
    chosen = [r for r in routes if any(a.get('SubnetId') == SUBNET for a in r['Associations'])]
    chosen = chosen or [r for r in routes if any(a.get('Main') for a in r['Associations'])]
    require(len(chosen) == 1 and any(r.get('DestinationCidrBlock') == '0.0.0.0/0'
            and r.get('GatewayId', '').startswith('igw-') and r.get('State') == 'active'
            for r in chosen[0]['Routes']), 'Direct IGW required; no NAT expense')
    groups = c('ec2', 'describe-security-groups', '--filters', f'Name=vpc-id,Values={VPC}',
               'Name=group-name,Values=trid-aws-eval-006-no-ingress')['SecurityGroups']
    require(len(groups) == 1 and groups[0]['IpPermissions'] == []
            and any(p.get('IpProtocol') == '-1' and any(x.get('CidrIp') == '0.0.0.0/0'
                    for x in p.get('IpRanges', [])) for p in groups[0]['IpPermissionsEgress']),
            'Zero-ingress outbound-only group required')
    r.update(subnet=subnet[0], route_table=chosen[0], security_group=groups[0])
    profile = c('iam', 'get-instance-profile', '--instance-profile-name', PROFILE)['InstanceProfile']
    require(profile['Arn'] == f'arn:aws:iam::{ACCOUNT}:instance-profile/{PROFILE}'
            and [x['RoleName'] for x in profile['Roles']] == [PROFILE], 'Unexpected profile')
    r.update(instance_profile=profile, instance_role=c.role(PROFILE), scheduler_role=c.role(SCHEDULER))
    CPU.verify_role_scopes(r)
    require(c('s3api', 'get-bucket-location', '--bucket', BUCKET)['LocationConstraint'] == REGION,
            'Bucket must be in Seoul')
    public = c('s3api', 'get-public-access-block', '--bucket', BUCKET)['PublicAccessBlockConfiguration']
    require(all(public.get(k) is True for k in ('BlockPublicAcls', 'IgnorePublicAcls',
            'BlockPublicPolicy', 'RestrictPublicBuckets')), 'Private bucket boundary missing')
    images = c('ec2', 'describe-images', '--image-ids', admission['ami']['ImageId'],
               '--owners', 'amazon')['Images']
    require(len(images) == 1, 'Selected image not official Amazon-owned')
    ami, pin = images[0], admission['ami']
    require(all(ami.get(k) == pin[k] for k in ('ImageId', 'OwnerId', 'Name', 'RootDeviceName'))
            and ami['Architecture'] == 'x86_64' and ami['State'] == 'available'
            and ami['RootDeviceType'] == 'ebs' and ami['VirtualizationType'] == 'hvm'
            and not ami.get('Platform') and not ami.get('ProductCodes'),
            'Selected AMI identity/platform changed')
    validate_ami_mappings(ami, pin)
    r['ami'] = ami
    queries = {
        'ec2': ('AmazonEC2', dict(regionCode=REGION, instanceType=TYPE, operatingSystem='Linux',
                tenancy='Shared', preInstalledSw='NA', capacitystatus='Used'), 'Hrs'),
        'ebs': ('AmazonEC2', dict(regionCode=REGION, volumeApiName='gp3', productFamily='Storage'), 'GB-Mo'),
        'ip': ('AmazonVPC', dict(regionCode=REGION, usagetype='APN2-PublicIPv4:InUseAddress'), 'Hrs'),
        'transfer': ('AWSDataTransfer', dict(fromLocation='Asia Pacific (Seoul)',
                     toLocation='External', transferType='AWS Outbound'), 'GB'),
        'storage': ('AmazonS3', dict(regionCode=REGION, productFamily='Storage', volumeType='Standard'), 'GB-Mo'),
        'put': ('AmazonS3', dict(regionCode=REGION, group='S3-API-Tier1'), 'Requests')}
    rates, evidence = {}, {}
    for key, query in queries.items():
        rates[key], evidence[key] = c.price(*query)
    require(rates == admission['prices'], 'Live rates differ from authenticated admission')
    r['cost'] = budget_bound(rates, admission['root_volume_gib'])
    r['price_products'] = evidence
    r['checked_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    return r


def verify_schedule(check, target, expression):
    require(check.get('State') == 'ENABLED' and check.get('ActionAfterCompletion') == 'DELETE'
            and check.get('ScheduleExpression') == expression
            and check.get('ScheduleExpressionTimezone') == 'UTC'
            and check.get('FlexibleTimeWindow') == {'Mode': 'OFF'}
            and all(check.get('Target', {}).get(k) == target[k] for k in ('Arn', 'RoleArn', 'RetryPolicy'))
            and json.loads(check['Target']['Input']) == json.loads(target['Input']),
            'Independent scheduler readback mismatch')


def authenticate_archive(path, archive_sha, approval_sha, admission_sha):
    """Read-only prelaunch authentication: no extraction or model/data execution."""
    require(Path(path).stat().st_size <= MAX_INPUT and digest(path) == archive_sha,
            'Whole input archive authentication failed')
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)) <= 129, 'Duplicate/excess archive members')
        for member in members:
            name = Path(member.name)
            require(member.isfile() and not member.issym() and not member.islnk()
                    and not member.pax_headers and not member.sparse and not name.is_absolute()
                    and '..' not in name.parts and str(name) == member.name
                    and 'final24' not in member.name.lower(), 'Unsafe/noncanonical/held-out archive member')
        def document(name):
            member = archive.getmember(name)
            require(member.size <= 2 * 1024**2, 'Oversize control document')
            with archive.extractfile(member) as stream:
                return json.load(stream)
        manifest = document('remote-bundle-manifest.json')
        rows = manifest['files']
        files = {r['path']: r for r in rows}
        require(manifest.get('required_root') == str(ROOT) and manifest.get('run_id') == RUN_ID
                and len(files) == len(rows) <= 128
                and set(names) == set(files) | {'remote-bundle-manifest.json'}
                and REQUIRED_SOURCES | {PLAN, ALLOWLIST, APPROVAL, PROVENANCE} <= set(files)
                and sum(r['bytes'] for r in rows) <= MAX_INPUT, 'Exact archive manifest mismatch')
        allow = document(ALLOWLIST)
        allowed = {r['path']: r for r in allow['files']}
        require(allow.get('run_id') == RUN_ID and allow.get('required_root') == str(ROOT)
                and len(allowed) == len(allow['files']) == allow.get('file_count')
                and allow.get('total_bytes') == sum(r['bytes'] for r in allow['files'])
                and ALLOWLIST not in allowed and set(files) == set(allowed) | {ALLOWLIST}
                and all(files[k] == v for k, v in allowed.items()), 'Archive allowlist mismatch')
        for name, row in files.items():
            member = archive.getmember(name)
            require(member.size == row['bytes'], 'Archive member size mismatch')
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == row['sha256'], 'Archive member SHA mismatch: ' + name)
            if name.startswith('runs/open-weight-base/'):
                require(name in {'runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc/' + asset
                    for asset in ('config.json', 'generation_config.json', 'tokenizer.json',
                                  'tokenizer_config.json', 'vocab.json', 'merges.txt')},
                    'Unapproved HF base/weight asset')
        plan = document(PLAN)
        require(REQUIRED_SOURCES <= set(plan['source_pins'])
                and all(files[p]['sha256'] == sha for p, sha in plan['source_pins'].items())
                and all(plan['source_pins'].get(p) == sha for p, sha in CPU.ENGINE_PINS.items()),
                'Archive source pins differ')
        require(files[APPROVAL]['sha256'] == approval_sha, 'Archive approval differs')
        verify_approval(document(APPROVAL), admission_sha)
        pin = plan['admission']
        require(pin['sha256'] == admission_sha and files[pin['path']]['sha256'] == admission_sha,
                'Archive admission differs')
        validate_admission(document(pin['path']))
        provenance = document(PROVENANCE)
        require(provenance.get('run_id') == RUN_ID and provenance.get('required_root') == str(ROOT),
                'Archive provenance differs')
        for pin in [plan['manifest'], *plan['verified_cache_shards']]:
            name = Path(pin['path']).relative_to(ROOT).as_posix()
            require(files[name]['sha256'] == pin['sha256'], 'Frozen audit input differs')
        CPU.frozen_child(plan)
        CPU.compatibility_selectors(plan)
        require(type(plan['cuda_cache_seconds']) is int and plan['cuda_cache_seconds'] == 180
                and plan['native']['build']['architecture'] == 'x86_64', 'CUDA plan runtime differs')
        return dict(archive_sha256=archive_sha, member_count=len(members),
                    source_pins=plan['source_pins'], all_members_authenticated=True)


def staged_source_paths():
    """Exact launcher stage map; no remote-only pipeline/model sources run locally."""
    return {
        'scripts/aws_eval/qwen_cuda_smoke.py': Path(__file__).absolute(),
        'scripts/aws_eval/qwen_cpu_smoke.py': Path(CPU.__file__).absolute(),
        BOOTSTRAP: Path(__file__).absolute().with_name('qwen_cuda_bootstrap.sh'),
    }


def authenticate_staged_sources(source_pins):
    records, body = {}, None
    for name, path in staged_source_paths().items():
        require(path.is_file() and not path.is_symlink(), 'Unsafe staged launcher source: ' + name)
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        require(source_pins.get(name) == sha, 'Staged launcher source differs from approved archive: ' + name)
        records[name] = dict(staged_path=str(path), sha256=sha)
        if name == BOOTSTRAP:
            # Use these exact authenticated bytes; never reread a different body.
            body = data.decode('utf-8')
    require(body is not None, 'Authenticated bootstrap is missing')
    return body, records


def launch(args, client=None):
    c = client or Client()
    require(args.run_id == RUN_ID and re.fullmatch('[0-9a-f]{64}', args.bundle_sha256),
            'Exact one-shot run/archive identity required')
    uri = f's3://{BUCKET}/{PREFIX}/code/{RUN_ID}/inputs.tar'
    require(args.bundle_s3_uri == uri, 'Wrong run input object')
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    receipt = state / f'{RUN_ID}.json'
    exclusive_intent(state / f'{RUN_ID}-approval.intent.json', dict(run_id=RUN_ID,
        admission_sha256=args.admission_sha256, approval_sha256=args.approval_sha256,
        bundle_sha256=args.bundle_sha256, client_token=RUN_ID, state='approval_consumed'))
    exclusive_intent(receipt, dict(run_id=RUN_ID, state='preflight_started'))
    record = preflight(args, c)
    require(not c.instances(RUN_ID), 'Run already exists; no second launch')
    obj = c('s3api', 'head-object', '--bucket', BUCKET, '--key', f'{PREFIX}/code/{RUN_ID}/inputs.tar')
    require(0 < obj['ContentLength'] <= MAX_INPUT
            and obj.get('ServerSideEncryption') in ('AES256', 'aws:kms'), 'Input object size/encryption mismatch')
    free = shutil.disk_usage('/tmp').free
    require(free >= obj['ContentLength'] + 256 * 1024**2, 'Insufficient owned /tmp archive scratch')
    # CloudShell HOME has a 1GiB quota; only small durable receipts belong there.
    # The exclusive temporary directory is ours and is removed on either outcome.
    with tempfile.TemporaryDirectory(prefix=f'{RUN_ID}-inputs-', dir='/tmp') as scratch:
        archive = Path(scratch) / 'inputs.tar'
        c('s3api', 'get-object', '--bucket', BUCKET, '--key',
          f'{PREFIX}/code/{RUN_ID}/inputs.tar', str(archive))
        record['archive_verification'] = authenticate_archive(
            archive, args.bundle_sha256, args.approval_sha256, args.admission_sha256)
        record['archive_verification']['scratch_free_bytes_before_download'] = free
    record['simulations'] = [
        c.simulate(PROFILE, 's3:GetObject', [f'arn:aws:s3:::{BUCKET}/{PREFIX}/code/{RUN_ID}/{n}'
                   for n in ('inputs.tar', 'release.json')]),
        c.simulate(PROFILE, 's3:PutObject', [f'arn:aws:s3:::{BUCKET}/{PREFIX}/runs/{RUN_ID}/{n}'
                   for n in ('results.tar', 'completion.json')]),
        c.simulate(SCHEDULER, 'ec2:TerminateInstances', [f'arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*'],
                   [dict(ContextKeyName='aws:ResourceTag/trid:eval', ContextKeyValues=['aws-eval-006'],
                         ContextKeyType='string')])]
    now = dt.datetime.now(dt.timezone.utc)
    scheduled = (now + dt.timedelta(minutes=119)).replace(second=0, microsecond=0)
    config = dict(run_id=RUN_ID, bundle_s3_uri=uri, bundle_sha256=args.bundle_sha256,
        request_started=now.isoformat(), schedule_at=scheduled.isoformat(),
        deadline_unix=int(now.timestamp()) + MAX_SECONDS,
        worker_deadline_unix=int(now.timestamp()) + MAX_SECONDS - 300,
        plan=PLAN, allowlist=ALLOWLIST, approval=APPROVAL, provenance=PROVENANCE,
        admission_sha256=args.admission_sha256, approval_sha256=args.approval_sha256,
        admission=record['admission'])
    body, record['staged_source_verification'] = authenticate_staged_sources(
        record['archive_verification']['source_pins'])
    user_data = ('#!/bin/bash\nif [ "${1:-}" != worker ]; then\n'
                 'shutdown -h +115 "Qwen013 CUDA hard lifetime" || { poweroff -f; exit 1; }\nfi\n'
                 + 'QWEN_CONFIG=' + shlex.quote(base64.b64encode(
                     gzip.compress(json.dumps(config, separators=(',', ':')).encode(), mtime=0)).decode())
                 + '\n' + body)
    require(len(user_data.encode()) <= 16384, 'Raw UTF-8 user data exceeds 16KiB')
    request = dict(ImageId=record['ami']['ImageId'], InstanceType=TYPE, MinCount=1, MaxCount=1,
        ClientToken=RUN_ID, IamInstanceProfile={'Name': PROFILE}, InstanceInitiatedShutdownBehavior='terminate',
        MetadataOptions=dict(HttpTokens='required', HttpEndpoint='enabled', HttpPutResponseHopLimit=1),
        NetworkInterfaces=[dict(DeviceIndex=0, SubnetId=SUBNET, Groups=[record['security_group']['GroupId']],
                                AssociatePublicIpAddress=True, DeleteOnTermination=True)],
        BlockDeviceMappings=[dict(DeviceName=record['ami']['RootDeviceName'], Ebs=dict(
            VolumeSize=record['admission']['root_volume_gib'], VolumeType='gp3', Encrypted=True,
            DeleteOnTermination=True))],
        TagSpecifications=[dict(ResourceType=k, Tags=CPU.tags(RUN_ID)) for k in ('instance', 'volume')],
        UserData=user_data)
    record.update(config=config, request=request, state='launch_intent_saved')
    save(receipt, record)
    instance_id = None
    c.deadline = config['deadline_unix']
    try:
        response = c('ec2', 'run-instances', '--cli-input-json', json.dumps(request))
        require(len(response['Instances']) == 1, 'Unexpected instance count')
        instance_id = response['Instances'][0]['InstanceId']
        CPU.owned_identity(response['Instances'][0], RUN_ID, instance_id)
        record.update(instance_id=instance_id, state='launched_without_schedule')
        save(receipt, record)
        target = dict(Arn='arn:aws:scheduler:::aws-sdk:ec2:terminateInstances',
            RoleArn=f'arn:aws:iam::{ACCOUNT}:role/{SCHEDULER}', Input=json.dumps({'InstanceIds': [instance_id]}),
            RetryPolicy=dict(MaximumRetryAttempts=0, MaximumEventAgeInSeconds=60))
        expression = 'at(' + scheduled.strftime('%Y-%m-%dT%H:%M:%S') + ')'
        c('scheduler', 'create-schedule', '--name', RUN_ID, '--group-name', 'default',
          '--schedule-expression', expression, '--schedule-expression-timezone', 'UTC',
          '--flexible-time-window', '{"Mode":"OFF"}', '--state', 'ENABLED',
          '--action-after-completion', 'DELETE', '--target', json.dumps(target), '--client-token', RUN_ID)
        check = c('scheduler', 'get-schedule', '--name', RUN_ID, '--group-name', 'default')
        verify_schedule(check, target, expression)
        require(time.time() + 600 < config['deadline_unix'], 'Late scheduler admission')
        record.update(schedule=check, state='schedule_verified')
        save(receipt, record)
        release = dict(**config, instance_id=instance_id, account=ACCOUNT, region=REGION,
                       schedule_name=RUN_ID, schedule_verified=True)
        release_path = state / f'{RUN_ID}-release.json'
        save(release_path, release)
        c('s3api', 'put-object', '--bucket', BUCKET, '--key', f'{PREFIX}/code/{RUN_ID}/release.json',
          '--body', str(release_path), '--server-side-encryption', 'AES256')
        record['state'] = 'released_fresh_cuda_cache_then_one_smoke'
        save(receipt, record)
        while time.time() < config['deadline_unix'] - 300:
            rows = c.instances(RUN_ID)
            if rows:
                require(len(rows) == 1, 'Unexpected owned instance count')
                CPU.owned_identity(rows[0], RUN_ID, instance_id)
                if rows[0]['State']['Name'] == 'terminated':
                    break
            time.sleep(10)
        c.deadline = config['deadline_unix'] + 300
        record['cleanup'] = CPU.reconcile_cleanup(RUN_ID, instance_id, c.deadline,
                                                  discover=c.instances, call=c)
        record['cleanup_state'] = record['cleanup']['cleanup_state']
        require(record['cleanup_state'] == 'complete', 'Instance/volume cleanup unresolved')
        record['state'] = 'workflow_terminated_cleanup_complete'
        try:
            c('scheduler', 'delete-schedule', '--name', RUN_ID, '--group-name', 'default', '--client-token', RUN_ID)
            record['schedule_cleanup'] = 'deleted_after_observed_termination'
        except RuntimeError as error:
            if 'ResourceNotFoundException' not in str(error):
                raise
            record['schedule_cleanup'] = 'already_absent_after_observed_termination'
        save(receipt, record)
        return record
    except BaseException as error:
        record.update(state='failed', error=f'{type(error).__name__}: {error}')
        if record.get('cleanup_state') != 'complete':
            c.deadline = min(time.time() + 180, config['deadline_unix'] + 300)
            record['cleanup'] = CPU.reconcile_cleanup(RUN_ID, instance_id, c.deadline,
                                                      discover=c.instances, call=c)
            record['cleanup_state'] = record['cleanup']['cleanup_state']
            record['launch_outcome'] = record['cleanup']['launch_outcome']
        if record.get('cleanup_state') != 'complete':
            record['state'] = 'cleanup_unresolved'
        save(receipt, record)
        raise


def verify_inputs(config, files):
    require(config['run_id'] == RUN_ID and config['plan'] == PLAN and config['allowlist'] == ALLOWLIST
            and config['approval'] == APPROVAL and config['provenance'] == PROVENANCE,
            'Remote metadata identity mismatch')
    plan = json.loads((ROOT / PLAN).read_text())
    pins = plan['source_pins']
    require(REQUIRED_SOURCES <= set(pins) and all(pins.get(p) == h for p, h in CPU.ENGINE_PINS.items()),
            'Required CUDA/frozen source pins missing')
    for path, sha in pins.items():
        require(path in files and files[path]['sha256'] == sha and digest(ROOT / path) == sha,
                'Source authentication mismatch: ' + path)
    approval = read_bound(ROOT / APPROVAL, config['approval_sha256'])
    verify_approval(approval, config['admission_sha256'])
    admission = plan['admission']
    require(admission['sha256'] == config['admission_sha256']
            and admission['path'] in files and files[admission['path']]['sha256'] == admission['sha256']
            and read_bound(ROOT / admission['path'], admission['sha256']) == config['admission'],
            'Plan/live admission binding mismatch')
    allowlist = json.loads((ROOT / ALLOWLIST).read_text())
    rows = allowlist['files']
    names = {r['path'] for r in rows}
    require(allowlist.get('required_root') == str(ROOT) and allowlist.get('run_id') == RUN_ID
            and len(names) == len(rows) == allowlist.get('file_count')
            and allowlist.get('total_bytes') == sum(r['bytes'] for r in rows)
            and 0 < allowlist['total_bytes'] <= MAX_INPUT and set(files) == names | {ALLOWLIST}
            and ALLOWLIST not in names and all(files.get(r['path']) == r for r in rows),
            'Exact allowlist mismatch')
    for pin in [plan['manifest'], *plan['verified_cache_shards']]:
        path = Path(pin['path'])
        name = path.relative_to(ROOT).as_posix()
        require(name in files and files[name]['sha256'] == pin['sha256']
                and digest(path) == pin['sha256'], 'Frozen manifest/audit cache not authenticated')
    provenance = json.loads((ROOT / PROVENANCE).read_text())
    require(provenance.get('run_id') == RUN_ID and provenance.get('required_root') == str(ROOT),
            'Provenance identity mismatch')
    CPU.frozen_child(plan)
    CPU.compatibility_selectors(plan)
    return plan


def gpu_minor_from_xml(text, expected_uuid):
    """Parse NVIDIA's full-query XML, not an assumed selective-query field."""
    require(isinstance(expected_uuid, str) and re.fullmatch(
        r'GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}', expected_uuid),
        'Invalid expected GPU UUID')
    require(isinstance(text, str) and 0 < len(text) <= 1024**2
            and '<!ENTITY' not in text, 'Invalid/oversize NVIDIA XML')
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise RuntimeError(f'Malformed NVIDIA XML: {error}') from error
    require(root.tag == 'nvidia_smi_log', 'Unexpected NVIDIA XML root')

    def scalar(parent, name):
        elements = parent.findall(name)
        require(len(elements) == 1 and len(elements[0]) == 0 and elements[0].text is not None,
                'Missing/ambiguous NVIDIA XML field: ' + name)
        return elements[0].text.strip()

    gpus = root.findall('gpu')
    require(scalar(root, 'attached_gpus') == '1' and len(gpus) == len(root.findall('.//gpu')) == 1,
            'NVIDIA XML must report exactly one attached GPU')
    require(scalar(gpus[0], 'uuid') == expected_uuid, 'NVIDIA XML GPU UUID mismatch')
    minor = scalar(gpus[0], 'minor_number')
    # Linux dev_t has20 minor bits; no GPU-index-to-minor assumption is made.
    require(re.fullmatch(r'0|[1-9][0-9]{0,6}', minor) and int(minor) < 2**20,
            'Invalid/unavailable NVIDIA device minor')
    return int(minor)


def bounded_diagnostic(value, limit=1000):
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value[:limit].decode('utf-8', errors='replace')
    return str(value)[:limit]


def gpu_device_paths(uuid):
    """Resolve exact UUID/minor through documented -q/-x; no alternate discovery."""
    # NVIDIA documents XML queries and Minor Number -> /dev/nvidia[minor number]:
    # https://docs.nvidia.com/deploy/nvidia-smi/index.html
    command = ['nvidia-smi', '-q', '-x']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('nvidia-smi -q -x exceeded5 seconds; stderr='
                           + repr(bounded_diagnostic(error.stderr))) from error
    except OSError as error:
        raise RuntimeError('Cannot execute nvidia-smi -q -x: ' + bounded_diagnostic(error)) from error
    diagnostic = ('stderr=' + repr(bounded_diagnostic(result.stderr))
                  + '; stdout=' + repr(bounded_diagnostic(result.stdout, 500)))
    require(result.returncode == 0, f'nvidia-smi -q -x failed rc={result.returncode}; {diagnostic}')
    try:
        minor = gpu_minor_from_xml(result.stdout, uuid)
    except RuntimeError as error:
        raise RuntimeError(f'nvidia-smi -q -x identity rejected: {error}; {diagnostic}') from error
    paths = [f'/dev/nvidia{minor}', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']
    require(all(not Path(p).is_symlink() and stat.S_ISCHR(Path(p).stat().st_mode) for p in paths),
            'Required NVIDIA character device missing')
    return paths


def sandbox_argv(name, command, seconds, output, gpu_uuid, devices, *, constrained=True, allow_loopback=False):
    require(re.fullmatch('[a-z0-9-]+', name) and 5 < seconds <= 1200, 'Invalid unit name/deadline')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', gpu_uuid), 'Explicit GPU UUID required')
    require(len(devices) == 4 and re.fullmatch('/dev/nvidia[0-9]+', devices[0])
            and devices[1:] == ['/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'],
            'Exact NVIDIA device allowlist required')
    argv = ['systemd-run', '--quiet', '--wait', '--collect', '--pipe', '--unit', f'{RUN_ID}-{name}',
            '--uid', 'qwen-smoke']
    properties = [f'WorkingDirectory={ROOT}', 'KillMode=control-group', f'RuntimeMaxSec={seconds - 5}',
        'TimeoutStopSec=5', 'BindsTo=qwen013-bootstrap.service', 'After=qwen013-bootstrap.service',
        'NoNewPrivileges=yes', 'ProtectSystem=strict', f'ReadWritePaths={output} /tmp', 'ProtectHome=yes',
        'PrivateDevices=no', 'DevicePolicy=closed', 'RestrictSUIDSGID=yes', 'IPAddressDeny=any', 'UMask=0077']
    properties += [f'DeviceAllow={p} rw' for p in devices]
    if constrained:
        properties += ['MemoryMax=6G', 'MemorySwapMax=0']
    if allow_loopback:
        require(name == 'native-probe-train' or name.startswith('native-case-'), 'Invalid loopback exception')
        properties += ['IPAddressAllow=localhost']
    for value in properties:
        argv += ['--property', value]
    for value in ('PYTHONDONTWRITEBYTECODE=1', f'CUDA_VISIBLE_DEVICES={gpu_uuid}', 'HF_HUB_OFFLINE=1',
                  'TRANSFORMERS_OFFLINE=1', 'TOKENIZERS_PARALLELISM=false', 'OMP_NUM_THREADS=4',
                  'MKL_NUM_THREADS=4', f'HF_HOME={output}/hf-cache', 'AWS_EC2_METADATA_DISABLED=true'):
        argv += ['--setenv', value]
    return [*argv, '--', *command]


def cuda_child(plan, output, runtime_path, runtime_sha, shards):
    original = CPU.frozen_child(plan)
    child = [TRAINER]
    for index in range(1, len(original), 2):
        option, value = original[index:index + 2]
        if option not in ('--cache', '--cache-sha256'):
            child += [option, str(output / 'train') if option == '--out' else value]
    child += ['--cuda-runtime', str(runtime_path), '--cuda-runtime-sha256', runtime_sha]
    for shard in shards:
        child += ['--cache', shard['path'], '--cache-sha256', shard['sha256']]
    return child


def cache_child(base, destination, ids):
    require(len(ids) in (11, 16) and len(ids) == len(set(ids)), 'Invalid cache partition')
    child = [base[0]]
    for offset in range(1, len(base), 2):
        option, value = base[offset:offset + 2]
        if option not in ('--max-steps', '--checkpoint-every', '--cache', '--cache-sha256'):
            child += [option, value]
    child[child.index('--mode') + 1] = 'cache'
    child[child.index('--out') + 1] = str(destination)
    return child + ['--ids', *ids]


def validate_runtime(runtime, admission, uuid):
    expected = admission['runtime']
    require(runtime.get('schema') == 'trid-cuda-runtime-v1' and runtime.get('trainer_entry') == TRAINER
            and runtime.get('device') == 'cuda' and runtime.get('dtype') == 'torch.float32'
            and runtime.get('architecture') == 'x86_64' and runtime.get('gpu_uuid') == uuid
            and runtime.get('versions') == dict(torch=expected['torch_runtime_version'],
                    transformers=PACKAGES['transformers'], peft=PACKAGES['peft']), 'CUDA runtime mismatch')
    gpu = runtime['gpu']
    require(re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,2}', gpu['driver_version']), 'Malformed actual driver')
    version = tuple(int(v) for v in gpu['driver_version'].split('.'))
    floor = tuple(int(v) for v in expected['minimum_driver_version'].split('.'))
    require(gpu['uuid'] == uuid and version >= floor
            and gpu['name'] in ('NVIDIA L4', 'L4')
            and expected['gpu_min_total_bytes'] <= gpu['total_bytes'] <= expected['gpu_max_total_bytes'],
            'Actual GPU/driver violates authenticated capability constraints')
    engine = runtime['engine']
    require(engine['cuda_runtime'] == expected['cuda_version'] and engine['gpu'] == gpu
            and engine['math'] == dict(cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
                                      float32_matmul_precision='highest')
            and engine['code']['train_qwen_preferences_cuda.py'] == digest(ROOT / TRAINER),
            'CUDA engine/math/source identity mismatch')
    for path, sha in CPU.ENGINE_PINS.items():
        require(engine['code'][Path(path).name] == sha, 'Frozen engine changed')


def validate_cache(directory, plan, runtime):
    """Nonroot no-model validation, preserving old partitions but never old scores."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    import train_qwen_preferences as trainer
    partitions, expected, digests = CPU.cache_audit(plan)
    paths = [directory / f'shard-{i:03d}.json' for i in range(1, 7)]
    require(set(directory.iterdir()) == set(paths), 'Fresh CUDA cache member set mismatch')
    shards = [json.loads(p.read_text()) for p in paths]
    expected.update(device='cuda', dtype='torch.float32', versions=runtime['versions'], engine=runtime['engine'])
    require([s['ids'] for s in shards] == partitions
            and all(s.get('cuda_runtime') == runtime for s in shards), 'CUDA cache partition/runtime mismatch')
    scores = trainer.verify_cache_shards(shards, expected, digests)
    require(len(scores) == 91, 'CUDA cache must cover all91 pairs')
    return [dict(path=str(p), sha256=digest(p)) for p in paths]


def run_unit(config, output, uuid, devices, name, command, seconds,
             constrained=True, allow_loopback=False):
    """Own the exclusive dispatch receipt, separately from pipeline plan metadata."""
    require(time.time() + seconds + 15 + 305 < config['worker_deadline_unix'],
            'Insufficient worker lifetime including cleanup/export reserve')
    argv = sandbox_argv(name, command, seconds, output, uuid, devices,
                        constrained=constrained, allow_loopback=allow_loopback)
    record = dict(argv=argv, state='dispatch_requested')
    path = output / f'{name}-command.json'
    require(not path.exists(), 'Unit already dispatched; no retries')
    exclusive_intent(path, record)
    try:
        with (output / f'{name}.log').open('xb') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=seconds + 15,
                                    env={'PATH': os.environ['PATH'], 'LANG': 'C.UTF-8'})
    except BaseException as error:
        stop = subprocess.run(['systemctl', 'stop', f'{RUN_ID}-{name}.service'], capture_output=True, timeout=10)
        record.update(state='orchestrator_error', error=type(error).__name__, cleanup_returncode=stop.returncode)
        save(path, record)
        raise
    record.update(state='finished', returncode=result.returncode)
    save(path, record)
    return result.returncode


def remote_stage():
    import pwd
    import urllib.request
    require(sys.platform == 'linux' and platform.machine() == 'x86_64' and os.geteuid() == 0
            and Path.cwd() == ROOT, 'Remote exact Linux x86_64 root required')
    config = json.loads(Path('/var/lib/qwen013/config.json').read_text())
    require(config['worker_deadline_unix'] == config['deadline_unix'] - 300, 'Invalid worker deadline')
    files = {r['path']: r for r in json.loads(Path('/var/lib/qwen013/verified-manifest.json').read_text())['files']}
    plan = verify_inputs(config, files)
    output = ROOT / RUN_PATH / 'remote-output'
    require(not output.exists(), 'Fresh CUDA output required')
    output.mkdir(mode=0o755)
    user = pwd.getpwnam('qwen-smoke')
    os.chown(output, user.pw_uid, user.pw_gid)
    py, guard = '/opt/qwen013-venv/bin/python', str(ROOT / 'scripts/qwen_linux_guard.py')
    c = Client(config['worker_deadline_unix'])
    token_request = urllib.request.Request('http://169.254.169.254/latest/api/token', method='PUT',
        headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
    with urllib.request.urlopen(token_request, timeout=5) as response:
        token = response.read().decode()
    request = urllib.request.Request('http://169.254.169.254/latest/dynamic/instance-identity/document',
        headers={'X-aws-ec2-metadata-token': token})
    with urllib.request.urlopen(request, timeout=5) as response:
        machine = json.load(response)
    require(machine['accountId'] == ACCOUNT and machine['region'] == REGION
            and machine['instanceType'] == TYPE and machine['availabilityZone'] == AZ
            and machine['imageId'] == config['admission']['ami']['ImageId'], 'Remote instance identity mismatch')
    release_path = Path('/var/lib/qwen013/release.json')
    c('s3api', 'get-object', '--bucket', BUCKET, '--key', f'{PREFIX}/code/{RUN_ID}/release.json', str(release_path))
    release = json.loads(release_path.read_text())
    require(all(release.get(k) == v for k, v in config.items()) and release.get('instance_id') == machine['instanceId']
            and release.get('account') == ACCOUNT and release.get('region') == REGION
            and release.get('schedule_name') == RUN_ID and release.get('schedule_verified') is True
            and time.time() + 1500 < config['worker_deadline_unix'], 'Missing/mismatched/late scheduler release')
    save(output / 'release-verification.json', release)
    save(output / 'input-verification.json', dict(bundle_sha256=config['bundle_sha256'], files=list(files.values()),
         source_pins=plan['source_pins'], cache_recomputed=True, old_scores_reused=False))
    save(output / 'provenance.json', json.loads((ROOT / PROVENANCE).read_text()))
    discovered = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5)
    uuid = discovered.stdout.strip()
    require(discovered.returncode == 0 and re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid),
            f'Exactly one GPU required; rc={discovered.returncode}; '
            f'stderr={bounded_diagnostic(discovered.stderr)!r}; '
            f'stdout={bounded_diagnostic(discovered.stdout, 500)!r}')
    subprocess.run(['nvidia-modprobe', '-u', '-c', '0'], check=True, timeout=10)
    devices = gpu_device_paths(uuid)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from qwen_linux_guard import gpu_metrics
    observed = gpu_metrics(uuid)
    require(not observed['compute_processes'] and observed['used_bytes'] < 6 * 1024**3,
            'Dedicated clean GPU admission required')
    save(output / 'gpu-admission.json', dict(metrics=observed, devices=devices))

    def unit(name, command, seconds, constrained=True, allow_loopback=False):
        return run_unit(config, output, uuid, devices, name, command, seconds,
                        constrained=constrained, allow_loopback=allow_loopback)

    from qwen_cpu_pipeline import validate_guard, validate_training, continue_after_smoke

    def guarded(name, command, seconds):
        receipt = output / f'{name}-supervisor.json'
        rc = unit(name, [py, '-B', guard, '--gpu-uuid', uuid, '--receipt', str(receipt),
                         '--seconds', str(seconds), '--', *command], seconds)
        require(rc == 0, f'{name} failed; no retry')
        result = json.loads(receipt.read_text())
        validate_guard(result, seconds, command, gpu_uuid=uuid)
        require(result['gpu']['identity'] == {k: observed[k] for k in
                ('uuid', 'name', 'pci_bus_id', 'total_bytes', 'driver_version')},
                'Guard identity changed since root GPU admission')

    require(unit('guard-unittest', [py, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                                   '-p', 'test_qwen_linux_guard.py', '-v'], 180) == 0, 'Guard tests failed')
    def probe(name, code, seconds, expected, constrained=True):
        receipt = output / f'{name}-receipt.json'
        command = [py, '-B', guard, '--gpu-uuid', uuid, '--receipt', str(receipt), '--seconds', str(seconds),
                   '--', py, '-I', '-B', '-c', code]
        rc = unit(name, command, seconds + 10, constrained)
        result = json.loads(receipt.read_text())
        require(result.get('stop_reason') == expected and result.get('cleanup_state') == 'completed'
                and not result.get('cleanup_errors') and not result.get('live_members_before_reap'),
                'Guard probe failed: ' + name)
        require((result.get('startup_released') is True and result.get('reaped') is True) if constrained
                else (result.get('startup_released') is False and 'child_pid' not in result and rc != 0),
                'Guard probe startup/reap mismatch')
        if expected == 'child_exit':
            require(rc == 0 and result['child_exit_code'] == 0, 'Successful probe failed')
        else:
            require(rc != 0, 'Rejected/deadline probe returned success')
    marker = output / 'outside-child-must-not-exist'
    probe('outside', f'from pathlib import Path; Path({str(marker)!r}).touch()', 15, 'guard_error', False)
    require(not marker.exists(), 'Unconstrained child executed')
    probe('inside', 'print("constrained CUDA guard child")', 15, 'child_exit')
    probe('deadline', 'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
          'pid=os.fork(); print(pid,flush=True); time.sleep(120)', 8, 'deadline')
    probe('descendants', 'import os,signal,time; pid=os.fork(); '
          'signal.signal(signal.SIGTERM,signal.SIG_IGN); os._exit(0) if pid else time.sleep(120)',
          15, 'leader_exited_with_live_descendants')
    environment = ('import importlib.metadata as m,os,sys; '
        f'assert {{k:m.version(k) for k in {PACKAGES!r}}}=={PACKAGES!r}; '
        f'assert os.getuid()==os.geteuid()=={user.pw_uid} and os.getuid()!=0; '
        'assert sys.version_info[:2]==(3,12)')
    guarded('environment', [py, '-I', '-B', '-c', environment], 60)
    selectors = CPU.compatibility_selectors(plan)
    compatibility = ('import sys,unittest;sys.path.insert(0,"tests");'
        f'suite=unittest.defaultTestLoader.loadTestsFromNames({selectors!r});'
        'r=unittest.TextTestRunner(verbosity=2).run(suite);'
        'sys.exit(0 if r.testsRun>0 and r.wasSuccessful() and not r.skipped else 1)')
    guarded('packed-compatibility', [py, '-B', '-c', compatibility], 180)
    runtime_path = output / 'cuda-runtime.json'
    guarded('cuda-inspect', [py, '-B', TRAINER, '--inspect-runtime', str(runtime_path)], 60)
    runtime = json.loads(runtime_path.read_text())
    validate_runtime(runtime, config['admission'], uuid)
    os.chown(runtime_path, 0, 0)
    runtime_path.chmod(0o444)
    runtime_sha = digest(runtime_path)
    config.update(cuda_runtime_path=str(runtime_path), cuda_runtime_sha256=runtime_sha)
    generated_output = output / 'generated-cuda-smoke.json'
    guarded('cuda-generated-smoke', [py, '-B', GENERATED_SMOKE, '--output', str(generated_output),
             '--runtime', str(runtime_path), '--runtime-sha256', runtime_sha], 120)
    generated = json.loads(generated_output.read_text())
    require(generated.get('schema') == 'trid-generated-cuda-smoke-v1'
            and generated.get('status') == 'pass' and generated.get('runtime_authenticated') is True
            and generated.get('completed_adamw_updates') == 2,
            'Generated CUDA correctness smoke did not pass')
    partitions, _, _ = CPU.cache_audit(plan)
    cache_dir = output / 'reference-cache-cuda'
    cache_dir.mkdir(mode=0o755)
    os.chown(cache_dir, user.pw_uid, user.pw_gid)
    seconds = plan['cuda_cache_seconds']
    require(type(seconds) is int and seconds == 180, 'Exact180-second cache partition deadline required')
    outcome = dict(state='fresh_cuda_cache', cache_children_requested=0, training_children_requested=0,
                   automatic_retries=0, old_scores_reused=False, full_training_started=False)
    save(output / 'smoke-outcome.json', outcome)
    base = cuda_child(plan, output, runtime_path, runtime_sha, [])
    for index, ids in enumerate(partitions, 1):
        child = cache_child(base, cache_dir / f'shard-{index:03d}.json', ids)
        outcome['cache_children_requested'] += 1
        save(output / 'smoke-outcome.json', outcome)
        guarded(f'cuda-cache-{index:03d}', [py, '-B', *child], seconds)
        shard_path = Path(child[child.index('--out') + 1])
        require(shard_path.is_file() and not shard_path.is_symlink(), 'Missing CUDA cache shard')
        os.chown(shard_path, 0, 0)
        shard_path.chmod(0o444)
    os.chown(cache_dir, 0, 0)
    cache_dir.chmod(0o555)
    guarded('cuda-cache-validation', [py, '-B', str(Path(__file__).resolve()), 'validate-cuda-cache',
             str(cache_dir), str(runtime_path), runtime_sha], 120)
    cache_shards = [dict(path=str(cache_dir / f'shard-{i:03d}.json'),
                        sha256=digest(cache_dir / f'shard-{i:03d}.json')) for i in range(1, 7)]
    save(output / 'cuda-cache-acceptance.json', dict(accepted=True, pairs=91, replay_rows=288,
        cache_recomputed=True, old_scores_reused=False, partitions=partitions, shards=cache_shards,
        cuda_runtime_sha256=runtime_sha, engine=runtime['engine']))
    train = cuda_child(plan, output, runtime_path, runtime_sha, cache_shards)
    outcome.update(state='model-smoke', training_children_requested=1)
    save(output / 'smoke-outcome.json', outcome)
    save(output / 'smoke-command.json', dict(original_argv=plan['argv'], child_argv=train,
        dispatched_argv=[py, '-B', *train], guard_seconds=180, maximum_optimizer_updates=2,
        initial_agreement_tolerance=1e-3, cuda_runtime_sha256=runtime_sha))
    guarded('model-smoke', [py, '-B', *train], 180)
    training = json.loads((output / 'train/training.json').read_text())
    validate_training(training, 2, cache_shards, ROOT, runtime=runtime)
    require(training['consumption']['pairs_consumed_through_step'] == plan['expected_pairs']
            and training['consumption']['replay_rows_consumed_through_step'] == plan['expected_replay_rows'],
            'Smoke expected population mismatch')
    save(output / 'smoke-acceptance.json', dict(accepted=True, updates=2,
        training_sha256=digest(output / 'train/training.json'), initial_check=training['initial_check'],
        consumption=training['consumption'], cuda_runtime_sha256=runtime_sha))
    outcome['state'] = 'accepted_two_update_smoke'
    save(output / 'smoke-outcome.json', outcome)
    # Native setup is root-only compilation, after smoke acceptance; inference stays -ngl0.
    remaining = min(1195, int(config['worker_deadline_unix'] - time.time() - 310))
    require(remaining > 0, 'No native setup time before cleanup reserve')
    subprocess.run(['/bin/bash', str(ROOT / 'scripts/aws_eval/qwen_native_setup.sh'), 'setup', str(ROOT)],
        check=True, timeout=remaining, env={**os.environ, 'QWEN_NATIVE_ARCHITECTURE': 'x86_64'})
    continue_after_smoke(config, output, plan, train, cache_shards, unit, runtime=runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'launch', 'status', 'terminate'):
        p = sub.add_parser(name)
        p.add_argument('--allow-console-root', action='store_true')
        if name in ('preflight', 'launch'):
            for key in ('admission', 'admission-sha256', 'approval', 'approval-sha256'):
                p.add_argument('--' + key, required=True)
            p.add_argument('--max-seconds', type=int, required=True)
            p.add_argument('--approved-usd', type=float, required=True)
        if name != 'preflight':
            p.add_argument('--run-id', required=True)
        if name == 'launch':
            p.add_argument('--bundle-s3-uri', required=True)
            p.add_argument('--bundle-sha256', required=True)
            p.add_argument('--state-dir', required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        result = preflight(args)
    elif args.command == 'launch':
        result = launch(args)
    else:
        c = Client(time.time() + 180)
        c.identity(args.allow_console_root)
        require(args.run_id == RUN_ID, 'Only exact CUDA run permitted')
        rows = c.instances(RUN_ID)
        for row in rows:
            CPU.owned_identity(row, RUN_ID)
        result = {'instances': rows}
        if args.command == 'terminate':
            result['cleanup'] = CPU.reconcile_cleanup(RUN_ID, None, c.deadline, discover=c.instances, call=c)
        else:
            result['outputs'] = c('s3api', 'list-objects-v2', '--bucket', BUCKET, '--prefix', f'{PREFIX}/runs/{RUN_ID}/')
            try:
                result['schedule'] = c('scheduler', 'get-schedule', '--name', RUN_ID, '--group-name', 'default')
            except RuntimeError as error:
                if 'ResourceNotFoundException' not in str(error):
                    raise
                result['schedule'] = {'state': 'absent_or_auto_deleted'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['remote-stage']:
        remote_stage()
    elif len(sys.argv) == 5 and sys.argv[1] == 'validate-cuda-cache':
        runtime = read_bound(sys.argv[3], sys.argv[4])
        print(json.dumps(validate_cache(Path(sys.argv[2]), json.loads((ROOT / PLAN).read_text()), runtime)))
    else:
        main()
