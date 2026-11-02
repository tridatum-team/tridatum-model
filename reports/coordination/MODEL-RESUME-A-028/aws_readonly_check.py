"""MODEL-RESUME-A-028: read-only AWS state check for AWS-EVAL-006 with the operator profile only — nothing is created,
started, changed or deleted (Describe/Get/List calls; EC2 DryRun is run separately by aws_eval_006.py --dry-run).
Calls the operator policy does not allow are recorded as AccessDenied, not retried with another identity.
usage: AWS_PROFILE=trid-aws-eval-006-operator python aws_readonly_check.py <out.json>"""
import datetime as dt
import json
import os
import subprocess
import sys

REGION, ACCOUNT, VPC = 'ap-northeast-2', '148761639313', 'vpc-0bef0e7b66432d802'
SUBNETS = ['subnet-025ee7d30ecf0228c', 'subnet-07a376ff8d92d3525', 'subnet-0811f3a7d5b1eb737']
if os.environ.get('AWS_PROFILE') != 'trid-aws-eval-006-operator':
    raise SystemExit('refused: run with AWS_PROFILE=trid-aws-eval-006-operator (the default profile is root)')


def aws(*args, region=REGION):
    r = subprocess.run(['aws', *args, '--region', region, '--output', 'json'], capture_output=True, text=True,
                       env={**os.environ, 'AWS_PAGER': ''})
    if r.returncode:
        err = r.stderr.strip()
        kind = next((k for k in ('AccessDenied', 'UnauthorizedOperation', 'ResourceNotFound', 'NoSuchEntity')
                     if k in err), 'error')
        return {'_error': kind, '_message': err[:240]}
    return json.loads(r.stdout) if r.stdout.strip() else {}


out = {'checked_utc': f'{dt.datetime.now(dt.timezone.utc):%FT%TZ}', 'identity': aws('sts', 'get-caller-identity')}
inst = aws('ec2', 'describe-instances', '--filters', 'Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down')
out['instances_not_terminated'] = [
    {'id': i['InstanceId'], 'type': i['InstanceType'], 'state': i['State']['Name'],
     'tags': {t['Key']: t['Value'] for t in i.get('Tags', []) if t['Key'] in ('Name', 'trid:eval', 'trid:run-id')}}
    for r in inst.get('Reservations', []) for i in r['Instances']] if '_error' not in inst else inst
vols = aws('ec2', 'describe-volumes')
out['volumes'] = [{'id': v['VolumeId'], 'size_gb': v['Size'], 'state': v['State'], 'encrypted': v['Encrypted'],
                   'attached_to': [a['InstanceId'] for a in v.get('Attachments', [])]}
                  for v in vols.get('Volumes', [])] if '_error' not in vols else vols
out['eval_tagged_instances_any_state'] = aws('ec2', 'describe-instances', '--filters', 'Name=tag:trid:eval,Values=aws-eval-006',
                                             '--query', 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]')
sg = aws('ec2', 'describe-security-groups', '--filters', 'Name=group-name,Values=trid-aws-eval-006-no-ingress', f'Name=vpc-id,Values={VPC}')
out['security_group'] = [{'id': g['GroupId'], 'inbound_rules': len(g['IpPermissions']),
                          'outbound': [(p.get('IpProtocol'), [x['CidrIp'] for x in p.get('IpRanges', [])]) for p in g['IpPermissionsEgress']]}
                         for g in sg.get('SecurityGroups', [])] if '_error' not in sg else sg
sn = aws('ec2', 'describe-subnets', '--subnet-ids', *SUBNETS)
out['subnets'] = [{'id': s['SubnetId'], 'az': s['AvailabilityZone'], 'free_ips': s['AvailableIpAddressCount'],
                   'map_public_ip': s['MapPublicIpOnLaunch'], 'vpc': s['VpcId']} for s in sn.get('Subnets', [])] if '_error' not in sn else sn
out['quotas'] = {code: (q.get('Quota', {}).get('Value') if '_error' not in q else q) for code, q in (
    ('L-DB2E81BA running on-demand G and VT vCPU', aws('service-quotas', 'get-service-quota', '--service-code', 'ec2', '--quota-code', 'L-DB2E81BA')),
    ('L-1216C47A running on-demand standard vCPU', aws('service-quotas', 'get-service-quota', '--service-code', 'ec2', '--quota-code', 'L-1216C47A')))}
for role in ('trid-aws-eval-006-instance', 'trid-aws-eval-006-scheduler'):
    r = aws('iam', 'get-role', '--role-name', role)
    out[f'role {role}'] = {'trust': r['Role']['AssumeRolePolicyDocument'], 'max_session_s': r['Role'].get('MaxSessionDuration')} \
        if '_error' not in r else r
ip = aws('iam', 'get-instance-profile', '--instance-profile-name', 'trid-aws-eval-006-instance')
out['instance_profile'] = [r['RoleName'] for r in ip['InstanceProfile']['Roles']] if '_error' not in ip else ip
# Scheduler: no list permission is granted to the operator; a Get on a name that does not exist shows whether Get is
# allowed under the name prefix without creating anything.
out['scheduler_get_probe'] = aws('scheduler', 'get-schedule', '--name', 'trid-aws-eval-006-readonly-probe')
out['scheduler_list'] = aws('scheduler', 'list-schedules', '--name-prefix', 'trid-aws-eval-006')
wd = aws('ssm', 'get-parameter', '--name', '/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64',
         '--query', 'Parameter.Value')
out['watchdog_image'] = wd
if isinstance(wd, str):
    img = aws('ec2', 'describe-images', '--image-ids', wd)
    out['watchdog_image_detail'] = [{'name': i['Name'], 'state': i['State'], 'owner': i['OwnerId'], 'root': i['RootDeviceName']}
                                    for i in img.get('Images', [])] if '_error' not in img else img
s3 = aws('s3api', 'list-objects-v2', '--bucket', f'trid-training-{ACCOUNT}', '--prefix', 'eval/aws-eval-006/')
out['s3_eval_prefix_objects'] = [(o['Key'], o['Size']) for o in s3.get('Contents', [])] if '_error' not in s3 else s3


def price(instance_type):
    f = [f'Type=TERM_MATCH,Field={k},Value={v}' for k, v in (
        ('instanceType', instance_type), ('regionCode', REGION), ('operatingSystem', 'Linux'), ('tenancy', 'Shared'),
        ('preInstalledSw', 'NA'), ('capacitystatus', 'Used'))]
    p = aws('pricing', 'get-products', '--service-code', 'AmazonEC2', '--filters', *f, region='us-east-1')
    if '_error' in p:
        return p
    usd = []
    for item in p.get('PriceList', []):
        item = json.loads(item)
        for term in item['terms'].get('OnDemand', {}).values():
            for d in term['priceDimensions'].values():
                usd.append(float(d['pricePerUnit']['USD']))
    return sorted(set(usd))


out['on_demand_usd_per_hour'] = {t: price(t) for t in ('g6.2xlarge', 't3.micro')}
open(sys.argv[1], 'w').write(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print('written', sys.argv[1])
