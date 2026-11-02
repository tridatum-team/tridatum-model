"""AWS-EVAL-006 operator: one short-lived EC2 GPU instance per run, which builds llama.cpp at the pinned commit with
CUDA, fetches the two pinned models and tokenizers, runs the blind P/Q packages and terminates itself.

Subcommands (AWS CLI v2 underneath; region ap-northeast-2, account 148761639313 only):
  preflight            read-only: identity, quota, offering, price, AMI, network, IAM/SG state, B's cost gate
  bootstrap-identity   ONE TIME, account owner, only with the user's approval (--user-approved): instance role +
                       profile, scheduler role, no-ingress security group, login-only operator and C-runner users
                       (deploy/aws-eval-006/*.json). Every other subcommand refuses the root user for paid work.
  launch --mode M      M = smoke (public 2-turn check of both packages) or full (C's private set, outputs
                       encrypted to C's certificate); --dry-run validates every parameter and permission at no cost.
                       A paid launch requires B's AWS-EVAL-006-COST report for this account and region.
                       After the instance starts, a one-time EventBridge Scheduler schedule terminates it at the
                       limit + 10 min, independently of the instance OS and of any Mac; if that schedule cannot be
                       created the instance is terminated at once.
  watchdog-test        two t3.micro instances that never shut themselves down from a job: one ends by its OS timer
                       (3 min; scheduler backstop at 15 min), the other only by the scheduler (6 min); cents
  watchdog-status STAMP  read-only: each watchdog instance's state, which mechanism ended it, schedule deleted or not
  status RUN_ID        instance state, phase object, results listing (never downloads outputs)
  terminate RUN_ID     terminate the run's instance(s) now
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT, REGION = '148761639313', 'ap-northeast-2'
BUCKET, PREFIX = f'trid-training-{ACCOUNT}', 'eval/aws-eval-006'
INSTANCE_TYPE, CUDA_ARCH = 'g6.2xlarge', '89'  # 8 vCPU / 32 GiB / NVIDIA L4 24 GB (sm_89); G/VT quota is 8 vCPU
AMI = 'ami-032ead1ff8d54d3f5'  # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261027, pinned
LLAMA_COMMIT = '8ca4060b7e30e76baaa5670ee1ec1b760b7f8552'
VOLUME_GB = 100
TAG = 'aws-eval-006'
INSTANCE_ROLE = 'trid-aws-eval-006-instance'
SCHEDULER_ROLE = 'trid-aws-eval-006-scheduler'
SECURITY_GROUP = 'trid-aws-eval-006-no-ingress'
VPC = 'vpc-0bef0e7b66432d802'  # default VPC; subnets in AZs a, c, d offer g6.2xlarge
SUBNETS = ['subnet-025ee7d30ecf0228c', 'subnet-07a376ff8d92d3525', 'subnet-0811f3a7d5b1eb737']
COST_REPORT = ROOT / 'reports/coordination/AWS-EVAL-006-COST.md'
# The remote run receives only the id+turns derivative of C's private set (80 rows, 250 turns, no gold fields);
# C's sealed original (af46eec1…, with expectations) never leaves C (AWS-EVAL-006-PRECOMMIT §3, C-TO-A §1).
PRIVATE_INPUT_SHA = '4134c4bb66146637be00ff15a4fff78c332cda5138423ea50a646fe49265836c'
C_CERT = ROOT / 'reports/coordination/AWS-EVAL-006/c-checkpoint-cert.pem'
POLICIES = ROOT / 'deploy/aws-eval-006'
OPERATOR_USER, C_RUNNER_USER = 'trid-aws-eval-006-operator', 'trid-aws-eval-006-c-runner'
MAX_MINUTES = {'smoke': 60, 'full': 150}  # OS timer per mode; --max-minutes may only lower it
HARD_STOP_GRACE_MINUTES = 10  # the scheduler terminates at the OS limit + 10 min
GPU_RESERVED_LIMIT_MINUTES = 230  # (60 + 10) + (150 + 10): every GPU launch of this approval, retries included
LEDGER = ROOT / 'reports/coordination/AWS-EVAL-006-GPU-LEDGER.json'
WAIT_INPUT_MINUTES = 30  # full mode: how long the instance waits for C to deliver the private set
HF = 'https://huggingface.co'
GEMMA, QWEN = ('676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee',
               '061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a')
GEMMA_TOK = f'{HF}/google/gemma-4-E4B-it-qat-q4_0-unquantized/resolve/476025a01dbf99361c062bbeca3d6a76bb4c4566'
QWEN_TOK = f'{HF}/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
ASSETS = [  # url, path under /opt/trid, sha256 — official sources at pinned revisions
    (f'{HF}/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/gemma-4-E4B_q4_0-it.gguf',
     f'models/{GEMMA}.gguf', GEMMA),
    (f'{HF}/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf',
     f'models/{QWEN}.gguf', QWEN),
    (f'{GEMMA_TOK}/tokenizer.json', f'tokenizers/{GEMMA}/tokenizer.json', 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'),
    (f'{GEMMA_TOK}/tokenizer_config.json', f'tokenizers/{GEMMA}/tokenizer_config.json', '3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999'),
    (f'{GEMMA_TOK}/chat_template.jinja', f'tokenizers/{GEMMA}/chat_template.jinja', '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'),
    (f'{GEMMA_TOK}/config.json', f'tokenizers/{GEMMA}/config.json', 'f2db6d6e24cb4b695308897d82c8424fae946e95272a27de68b6e908c77b0f95'),
    (f'{QWEN_TOK}/tokenizer.json', f'tokenizers/{QWEN}/tokenizer.json', 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'),
    (f'{QWEN_TOK}/tokenizer_config.json', f'tokenizers/{QWEN}/tokenizer_config.json', 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'),
    (f'{QWEN_TOK}/vocab.json', f'tokenizers/{QWEN}/vocab.json', 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
    (f'{QWEN_TOK}/merges.txt', f'tokenizers/{QWEN}/merges.txt', '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'),
    (f'{QWEN_TOK}/config.json', f'tokenizers/{QWEN}/config.json', '1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'),
]
CODE_FILES = ['scripts/base_dialogue_runner.py', 'scripts/chat_blind_runner.py',
              'reports/coordination/CHAT-BLIND-005/shared-settings.json',
              'reports/coordination/CHAT-CAPABILITY-006/chat-system-candidate.txt',
              'reports/coordination/CHAT-EVAL-004/public-20.jsonl',
              'runs/chat-blind-005/packages/P/package.json', 'runs/chat-blind-005/packages/Q/package.json']


def aws(*args, check=True, parse=True):
    r = subprocess.run(['aws', *args, '--region', REGION, '--output', 'json'], capture_output=True, text=True,
                       env={**os.environ, 'AWS_PAGER': ''})
    if check and r.returncode:
        raise SystemExit(f'aws {" ".join(args[:2])} failed: {r.stderr.strip()[:500]}')
    if not parse:
        return r
    return json.loads(r.stdout) if r.stdout.strip() else {}


def identity():
    me = aws('sts', 'get-caller-identity')
    if me['Account'] != ACCOUNT:
        raise SystemExit(f'wrong account {me["Account"]}')
    return me['Arn']


def cost_gate():
    """B's cost report must exist and name this account and region before anything paid starts."""
    if not COST_REPORT.exists():
        return 'missing ' + str(COST_REPORT.relative_to(ROOT))
    text = COST_REPORT.read_text()
    missing = [v for v in (ACCOUNT, REGION) if v not in text]
    return f'cost report does not name {missing}' if missing else None


def security_group_id():
    groups = aws('ec2', 'describe-security-groups', '--filters', f'Name=group-name,Values={SECURITY_GROUP}',
                 f'Name=vpc-id,Values={VPC}')['SecurityGroups']
    return groups[0]['GroupId'] if groups else None


def preflight(_):
    out = {'identity': identity(), 'region': REGION}
    out['g_vt_vcpu_quota'] = aws('service-quotas', 'get-service-quota', '--service-code', 'ec2', '--quota-code',
                                 'L-DB2E81BA')['Quota']['Value']
    running = aws('ec2', 'describe-instances', '--filters', 'Name=instance-state-name,Values=pending,running',
                  '--query', 'Reservations[].Instances[].InstanceType')
    out['running_instance_types'] = running
    out['offered_azs'] = sorted(o['Location'] for o in aws(
        'ec2', 'describe-instance-type-offerings', '--location-type', 'availability-zone', '--filters',
        f'Name=instance-type,Values={INSTANCE_TYPE}')['InstanceTypeOfferings'])
    image = aws('ec2', 'describe-images', '--image-ids', AMI)['Images'][0]
    out['ami'] = {'id': AMI, 'name': image['Name'], 'state': image['State']}
    out['instance_profile'] = bool(aws('iam', 'get-instance-profile', '--instance-profile-name', INSTANCE_ROLE,
                                       check=False, parse=False).returncode == 0)
    out['scheduler_role'] = bool(aws('iam', 'get-role', '--role-name', SCHEDULER_ROLE, check=False, parse=False).returncode == 0)
    out['security_group'] = security_group_id()
    out['cost_gate'] = cost_gate() or 'pass'
    print(json.dumps(out, ensure_ascii=False, indent=1))


def require_non_root(arn):
    """Paid launches and IAM writes never run as the root user (MACMINI-DEPLOY-002: root is monitor-only)."""
    if arn.endswith(':root'):
        raise SystemExit('refused: caller is the root user; use the trid-aws-eval-006-operator login (aws login '
                         '--profile trid-aws-eval-006-operator) created by the user-approved one-time bootstrap')


def policy(name):
    return (POLICIES / f'{name}.json').read_text()


def bootstrap_identity(a):
    """One time, by the account owner after the user approves it (the only step that needs IAM admin rights):
    instance role + profile, scheduler role, no-ingress security group and two login-only IAM users without access
    keys — the operator (A: launch/terminate/schedule tagged eval instances) and the C runner (deliver the set over
    SSM to tagged instances, read encrypted checkpoints). Their console passwords and MFA are set by the user in
    the console; the CLI then uses `aws login`. Policies come from deploy/aws-eval-006/*.json (validated by IAM Access
    Analyzer and simulated, see AWS-EVAL-006-INFRA)."""
    if not a.user_approved:
        raise SystemExit('refused: the one-time identity bootstrap needs the user\'s explicit approval (--user-approved)')
    identity()
    tags = ['--tags', f'Key=trid:eval,Value={TAG}']
    for role, trust, inline, description in (
            (INSTANCE_ROLE, 'instance-role-trust', 'instance-role-policy', 'AWS-EVAL-006 GPU eval instance: SSM + one S3 prefix'),
            (SCHEDULER_ROLE, 'scheduler-role-trust', 'scheduler-role-policy',
             'AWS-EVAL-006 hard stop: terminate only instances tagged trid:eval=aws-eval-006')):
        if aws('iam', 'get-role', '--role-name', role, check=False, parse=False).returncode:
            aws('iam', 'create-role', '--role-name', role, '--assume-role-policy-document', policy(trust),
                '--description', description, *tags)
        aws('iam', 'put-role-policy', '--role-name', role, '--policy-name', inline, '--policy-document', policy(inline))
    aws('iam', 'attach-role-policy', '--role-name', INSTANCE_ROLE, '--policy-arn',
        'arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore')
    if aws('iam', 'get-instance-profile', '--instance-profile-name', INSTANCE_ROLE, check=False, parse=False).returncode:
        aws('iam', 'create-instance-profile', '--instance-profile-name', INSTANCE_ROLE, *tags)
        aws('iam', 'add-role-to-instance-profile', '--instance-profile-name', INSTANCE_ROLE, '--role-name', INSTANCE_ROLE)
    # Inline user policies are capped at 2,048 bytes in total (the operator policy is ~4.3 KB), so each user gets one
    # customer-managed policy of the same name (6,144-byte cap), kept in sync with deploy/aws-eval-006/*.json.
    for user, document in ((OPERATOR_USER, 'operator-policy'), (C_RUNNER_USER, 'c-runner-policy')):
        if aws('iam', 'get-user', '--user-name', user, check=False, parse=False).returncode:
            aws('iam', 'create-user', '--user-name', user, *tags)
        arn = f'arn:aws:iam::{ACCOUNT}:policy/trid/{user}'
        if aws('iam', 'get-policy', '--policy-arn', arn, check=False, parse=False).returncode:
            aws('iam', 'create-policy', '--policy-name', user, '--path', '/trid/', '--policy-document', policy(document),
                '--description', f'AWS-EVAL-006 {document}', *tags)
        else:
            for v in aws('iam', 'list-policy-versions', '--policy-arn', arn)['Versions']:
                if not v['IsDefaultVersion']:
                    aws('iam', 'delete-policy-version', '--policy-arn', arn, '--version-id', v['VersionId'])
            aws('iam', 'create-policy-version', '--policy-arn', arn, '--policy-document', policy(document), '--set-as-default')
        aws('iam', 'attach-user-policy', '--user-name', user, '--policy-arn', arn)
        # `aws login` with console credentials needs this AWS managed policy (OAuth2 token for the CLI); no access keys
        aws('iam', 'attach-user-policy', '--user-name', user, '--policy-arn',
            'arn:aws:iam::aws:policy/SignInLocalDevelopmentAccess')
    if not security_group_id():
        group = aws('ec2', 'create-security-group', '--group-name', SECURITY_GROUP, '--vpc-id', VPC, '--description',
                    'AWS-EVAL-006: no inbound rules (SSM only)', '--tag-specifications',
                    f'ResourceType=security-group,Tags=[{{Key=trid:eval,Value={TAG}}}]')
        assert not aws('ec2', 'describe-security-groups', '--group-ids', group['GroupId'])['SecurityGroups'][0]['IpPermissions']
    print(json.dumps({'instance_role': INSTANCE_ROLE, 'scheduler_role': SCHEDULER_ROLE, 'operator_user': OPERATOR_USER,
                      'c_runner_user': C_RUNNER_USER, 'security_group': security_group_id(),
                      'next': 'user sets console password + MFA for both users; then aws login --profile <user>'}, indent=1))


def code_archive(mode):
    """Deterministic tar.gz of the runner, shared settings, contract, public set, package settings and asset list."""
    buf = io.BytesIO()
    members = [(f, (ROOT / f).read_bytes()) for f in CODE_FILES]
    members.append(('assets.tsv', ''.join(f'{u}\t{p}\t{s}\n' for u, p, s in ASSETS).encode()))
    members.append(('c-cert.pem', C_CERT.read_bytes()))  # checkpoints of every mode are encrypted to C's certificate
    with tarfile.open(fileobj=buf, mode='w:gz', format=tarfile.PAX_FORMAT) as tar:
        for name, data in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o600
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def user_data(mode, run_id, code_key, code_sha, max_minutes):
    values = {'RUN_ID': run_id, 'MODE': mode, 'MAX_MINUTES': max_minutes, 'BUCKET': BUCKET, 'PREFIX': PREFIX,
              'CODE_KEY': code_key, 'CODE_SHA': code_sha, 'LLAMA_COMMIT': LLAMA_COMMIT, 'CUDA_ARCH': CUDA_ARCH,
              'INPUT_SHA': PRIVATE_INPUT_SHA, 'WAIT_INPUT_MINUTES': WAIT_INPUT_MINUTES,
              'CERT_SHA': hashlib.sha256(C_CERT.read_bytes()).hexdigest()}
    head = '#!/bin/bash\n' + ''.join(f"{k}='{v}'\n" for k, v in values.items())
    return head + (Path(__file__).parent / 'bootstrap.sh').read_text()


def run_instances(spec, dry_run):
    args = ['ec2', 'run-instances', '--cli-input-json', json.dumps(spec)]
    if dry_run:
        r = aws(*args, '--dry-run', check=False, parse=False)
        if 'DryRunOperation' not in r.stderr:
            raise SystemExit(f'dry run failed: {r.stderr.strip()[:600]}')
        return None
    return aws(*args)['Instances'][0]


def instance_spec(run_id, image, instance_type, subnet, user_data_text, volume_gb):
    tags = [{'Key': 'Name', 'Value': run_id}, {'Key': 'trid:eval', 'Value': TAG}, {'Key': 'trid:run-id', 'Value': run_id}]
    return {
        'ImageId': image, 'InstanceType': instance_type, 'MinCount': 1, 'MaxCount': 1,
        'IamInstanceProfile': {'Name': INSTANCE_ROLE},
        'NetworkInterfaces': [{'DeviceIndex': 0, 'SubnetId': subnet, 'Groups': [security_group_id()],
                               'AssociatePublicIpAddress': True, 'DeleteOnTermination': True}],
        'MetadataOptions': {'HttpTokens': 'required', 'HttpPutResponseHopLimit': 1, 'HttpEndpoint': 'enabled'},
        'BlockDeviceMappings': [{'DeviceName': '/dev/sda1' if image == AMI else '/dev/xvda',
                                 'Ebs': {'VolumeSize': volume_gb, 'VolumeType': 'gp3', 'Encrypted': True,
                                         'DeleteOnTermination': True}}],
        'InstanceInitiatedShutdownBehavior': 'terminate',
        'TagSpecifications': [{'ResourceType': 'instance', 'Tags': tags}, {'ResourceType': 'volume', 'Tags': tags}],
        'UserData': user_data_text}


def hard_stop(instance_id, at_utc):
    """One-time EventBridge Scheduler schedule that terminates the instance; independent of the instance and the Macs."""
    target = {'Arn': 'arn:aws:scheduler:::aws-sdk:ec2:terminateInstances',
              'RoleArn': f'arn:aws:iam::{ACCOUNT}:role/{SCHEDULER_ROLE}',
              'Input': json.dumps({'InstanceIds': [instance_id]})}
    r = aws('scheduler', 'create-schedule', '--name', f'trid-aws-eval-006-{instance_id}', '--schedule-expression',
            f'at({at_utc:%Y-%m-%dT%H:%M:%S})', '--schedule-expression-timezone', 'UTC', '--flexible-time-window',
            'Mode=OFF', '--action-after-completion', 'DELETE', '--target', json.dumps(target), check=False, parse=False)
    if r.returncode:
        aws('ec2', 'terminate-instances', '--instance-ids', instance_id)
        raise SystemExit(f'hard-stop schedule failed, instance {instance_id} terminated: {r.stderr.strip()[:400]}')


def launch(a):
    arn = identity()
    gate = cost_gate()
    if not a.dry_run:
        require_non_root(arn)
        if gate:
            raise SystemExit(f'paid launch refused: {gate}')
    if not C_CERT.exists():
        raise SystemExit('needs C certificate ' + str(C_CERT.relative_to(ROOT)))
    cap = MAX_MINUTES[a.mode]
    if a.max_minutes is not None and not 1 <= a.max_minutes <= cap:
        raise SystemExit(f'--max-minutes must be an integer in 1..{cap} for {a.mode}')
    max_minutes = a.max_minutes or cap
    reserve = max_minutes + HARD_STOP_GRACE_MINUTES  # the scheduler's hard stop is the true upper bound
    used = gpu_minutes_reserved()
    if used + reserve > GPU_RESERVED_LIMIT_MINUTES:
        raise SystemExit(f'refused: {used} GPU minutes already reserved + {reserve} > {GPU_RESERVED_LIMIT_MINUTES} '
                         '(this approval); a relaunch or retry needs a new approval')
    run_id = f'aws-eval-006-{a.mode}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}'
    archive = code_archive(a.mode)
    code_sha = hashlib.sha256(archive).hexdigest()
    code_key = f'{PREFIX}/code/{run_id}/code.tar.gz'
    text = user_data(a.mode, run_id, code_key, code_sha, max_minutes)
    if len(text.encode()) > 16384:
        raise SystemExit('user data over 16 KB')
    record = {'run_id': run_id, 'mode': a.mode, 'max_minutes': max_minutes, 'instance_type': INSTANCE_TYPE, 'ami': AMI,
              'code_key': code_key, 'code_sha256': code_sha, 'llama_commit': LLAMA_COMMIT, 'dry_run': a.dry_run}
    for subnet in SUBNETS:
        spec = instance_spec(run_id, AMI, INSTANCE_TYPE, subnet, text, VOLUME_GB)
        if a.dry_run:
            run_instances(spec, True)
            record['dry_run_result'] = f'DryRunOperation (would succeed) in {subnet}'
            break
        Path('/tmp/aws-eval-006-code.tar.gz').write_bytes(archive)
        aws('s3', 'cp', '--only-show-errors', '--sse', 'AES256', '/tmp/aws-eval-006-code.tar.gz', f's3://{BUCKET}/{code_key}',
            parse=False)
        entry = ledger_add({'attempt': f'{run_id}@{subnet}', 'run_id': run_id, 'mode': a.mode, 'max_minutes': max_minutes,
                            'reserved_minutes': reserve, 'reserved_utc': f'{dt.datetime.now(dt.timezone.utc):%FT%TZ}',
                            'state': 'reserved'})
        r = aws('ec2', 'run-instances', '--cli-input-json', json.dumps(spec), check=False, parse=False)
        if r.returncode:
            # no instance exists, so nothing runs or bills: the reservation is released (recorded, not deleted)
            ledger_update(entry, state=f'not launched: {r.stderr.strip()[:120]}', reserved_minutes=0)
            if 'InsufficientInstanceCapacity' in r.stderr:
                continue
            raise SystemExit(f'run-instances failed: {r.stderr.strip()[:600]}')
        instance = json.loads(r.stdout)['Instances'][0]
        launched = dt.datetime.now(dt.timezone.utc)
        stop_at = launched + dt.timedelta(minutes=reserve)
        # Between run-instances returning and the schedule existing (seconds), the instance's own OS timer is the only
        # limit; if this process dies in that gap the run still ends by the OS timer, but not by the scheduler. Not
        # an atomic guarantee.
        ledger_update(entry, state='launched', instance_id=instance['InstanceId'], launched_utc=f'{launched:%FT%TZ}',
                      hard_stop_utc=f'{stop_at:%FT%TZ}')
        hard_stop(instance['InstanceId'], stop_at)
        ledger_update(entry, state='launched; hard stop scheduled')
        record.update(instance_id=instance['InstanceId'], subnet=subnet, launched_utc=f'{launched:%FT%TZ}',
                      hard_stop_utc=f'{stop_at:%FT%TZ}', gpu_minutes_reserved_total=gpu_minutes_reserved())
        break
    else:
        raise SystemExit('no capacity in any offered AZ')
    print(json.dumps(record, indent=1))


def ledger_entries():
    return json.loads(LEDGER.read_text())['entries'] if LEDGER.exists() else []


def gpu_minutes_reserved():
    return sum(e['reserved_minutes'] for e in ledger_entries())


def ledger_add(entry):
    entries = ledger_entries() + [entry]
    LEDGER.write_text(json.dumps({'limit_minutes': GPU_RESERVED_LIMIT_MINUTES, 'entries': entries}, indent=1) + '\n')
    return entry['attempt']


def ledger_update(attempt, **fields):
    entries = ledger_entries()
    for e in entries:
        if e['attempt'] == attempt:
            e.update(fields)
    LEDGER.write_text(json.dumps({'limit_minutes': GPU_RESERVED_LIMIT_MINUTES, 'entries': entries}, indent=1) + '\n')


WATCHDOG_OS_MINUTES, WATCHDOG_SCHEDULER_MINUTES, WATCHDOG_BACKSTOP_MINUTES = 3, 6, 15


def watchdog_test(a):
    """Two t3.micro instances (Amazon Linux 2023): A ends by its OS timer (no job, no trap), B has no OS timer and
    ends only by the scheduler. Both tagged and encrypted like the real runs. A also gets a later scheduler backstop
    (MODEL-RESUME-A-028), so no instance of the test is left without an independent limit if its OS timer fails; which
    mechanism ended each instance is read from its state reason (watchdog-status)."""
    arn = identity()
    gate = cost_gate()
    if not a.dry_run:
        require_non_root(arn)
        if gate:
            raise SystemExit(f'paid launch refused: {gate}')
    image = aws('ssm', 'get-parameter', '--name', '/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64',
                '--query', 'Parameter.Value')
    stamp = f'{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}'
    cases = {'os-timer': (f'#!/bin/bash\nshutdown -h +{WATCHDOG_OS_MINUTES} "watchdog test"\n', WATCHDOG_BACKSTOP_MINUTES),
             'scheduler': ('#!/bin/bash\n# no OS timer: only the EventBridge schedule may end this instance\n',
                           WATCHDOG_SCHEDULER_MINUTES)}
    out = {'stamp': stamp}
    for name, (script, stop_minutes) in cases.items():
        spec = instance_spec(f'aws-eval-006-watchdog-{name}-{stamp}', image, 't3.micro', SUBNETS[0], script, 8)
        if a.dry_run:
            run_instances(spec, True)
            out[name] = 'DryRunOperation (would succeed)'
            continue
        instance = run_instances(spec, False)
        launched = dt.datetime.now(dt.timezone.utc)
        hard_stop(instance['InstanceId'], launched + dt.timedelta(minutes=stop_minutes))
        out[name] = {'instance_id': instance['InstanceId'], 'launched_utc': f'{launched:%FT%TZ}',
                     'schedule': f'trid-aws-eval-006-{instance["InstanceId"]}',
                     'schedule_utc': f'{launched + dt.timedelta(minutes=stop_minutes):%FT%TZ}'}
    print(json.dumps(out, indent=1))


def watchdog_status(a):
    """Read-only: each watchdog instance's state, state reason (Client.InstanceInitiatedShutdown = its OS timer;
    Client.UserInitiatedShutdown = a TerminateInstances call, here the scheduler's) and whether its one-time schedule
    still exists (it deletes itself after running)."""
    out = {}
    for name in ('os-timer', 'scheduler'):
        rows = aws('ec2', 'describe-instances', '--filters', f'Name=tag:trid:run-id,Values=aws-eval-006-watchdog-{name}-{a.stamp}',
                   '--query', 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime,StateReason.Code,StateTransitionReason]')
        cases = []
        for iid, state, launched, reason_code, reason in rows:
            sched = aws('scheduler', 'get-schedule', '--name', f'trid-aws-eval-006-{iid}', check=False, parse=False)
            cases.append({'instance_id': iid, 'state': state, 'launched': launched, 'state_reason_code': reason_code,
                          'state_transition_reason': reason,
                          'schedule': 'exists' if sched.returncode == 0 else
                          'deleted (not found)' if 'ResourceNotFound' in sched.stderr else sched.stderr.strip()[:160]})
        out[name] = cases
    print(json.dumps(out, indent=1))


def instances_of(run_id):
    return aws('ec2', 'describe-instances', '--filters', f'Name=tag:trid:run-id,Values={run_id}', '--query',
               'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]')


def status(a):
    out = {'instances': instances_of(a.run_id)}
    phase = aws('s3', 'cp', f's3://{BUCKET}/{PREFIX}/runs/{a.run_id}/phase.json', '-', check=False, parse=False)
    out['phase'] = json.loads(phase.stdout) if phase.returncode == 0 and phase.stdout.strip() else None
    listing = aws('s3api', 'list-objects-v2', '--bucket', BUCKET, '--prefix', f'{PREFIX}/runs/{a.run_id}/', check=False)
    out['objects'] = [(o['Key'].split(f'{a.run_id}/', 1)[1], o['Size']) for o in listing.get('Contents', [])]
    print(json.dumps(out, ensure_ascii=False, indent=1))


def terminate(a):
    ids = [i[0] for i in instances_of(a.run_id) if i[1] not in ('terminated', 'shutting-down')]
    if ids:
        aws('ec2', 'terminate-instances', '--instance-ids', *ids)
    print(json.dumps({'terminated': ids}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('preflight').set_defaults(fn=preflight)
    s = sub.add_parser('bootstrap-identity')
    s.add_argument('--user-approved', action='store_true')
    s.set_defaults(fn=bootstrap_identity)
    s = sub.add_parser('launch')
    s.add_argument('--mode', choices=('smoke', 'full'), required=True)
    s.add_argument('--max-minutes', type=int)
    s.add_argument('--dry-run', action='store_true')
    s.set_defaults(fn=launch)
    s = sub.add_parser('watchdog-test')
    s.add_argument('--dry-run', action='store_true')
    s.set_defaults(fn=watchdog_test)
    for name, fn in (('status', status), ('terminate', terminate)):
        s = sub.add_parser(name)
        s.add_argument('run_id')
        s.set_defaults(fn=fn)
    s = sub.add_parser('watchdog-status')
    s.add_argument('stamp', help='the stamp watchdog-test printed (YYYYMMDDTHHMMSSZ)')
    s.set_defaults(fn=watchdog_status)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
