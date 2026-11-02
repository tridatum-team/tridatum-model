"""Throwaway (MODEL-RESUME-A-028): per explicit profile, STS identity and the credential expiry, never printing keys.
Refuses to run if AWS_* credential variables are set in the environment (they would override --profile)."""
import datetime as dt
import json
import os
import subprocess

leaks = [k for k in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_PROFILE', 'AWS_DEFAULT_PROFILE')
         if os.environ.get(k)]
print('credential env vars set:', leaks or 'none')
if any(k.startswith('AWS_ACCESS') or k.startswith('AWS_SECRET') or k == 'AWS_SESSION_TOKEN' for k in leaks):
    raise SystemExit('refused: environment credentials would override the profile')
now = dt.datetime.now(dt.timezone.utc)
for profile in ('trid-aws-eval-006-operator', 'trid-aws-eval-006-c-runner'):
    env = {**os.environ, 'AWS_PAGER': ''}
    me = subprocess.run(['aws', 'sts', 'get-caller-identity', '--profile', profile, '--region', 'ap-northeast-2',
                         '--output', 'json'], capture_output=True, text=True, env=env)
    arn = json.loads(me.stdout)['Arn'] if me.returncode == 0 else f'ERROR {me.stderr.strip()[:160]}'
    cred = subprocess.run(['aws', 'configure', 'export-credentials', '--profile', profile, '--format', 'process'],
                          capture_output=True, text=True, env=env)
    expiry = None
    if cred.returncode == 0:
        exp = json.loads(cred.stdout).get('Expiration')
        if exp:
            t = dt.datetime.fromisoformat(exp.replace('Z', '+00:00'))
            expiry = {'expiration_utc': f'{t:%FT%TZ}', 'minutes_left': round((t - now).total_seconds() / 60, 1)}
    else:
        expiry = f'ERROR {cred.stderr.strip()[:160]}'
    print(json.dumps({'profile': profile, 'arn': arn, 'credentials': expiry, 'checked_utc': f'{now:%FT%TZ}'}))
