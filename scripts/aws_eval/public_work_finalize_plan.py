"""Assemble a fresh public-work final plan from preparation readbacks, offline.

No AWS client, subprocess, network, cloud mutation, refresh or retry. Existing
final files are never replaced; original observation timestamps are preserved.
Dynamic suite and optional prior-comparator outputs must match the sealed workload;
finalization never repairs a stale provisional output allowlist.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.aws_eval import fresh_eight_launch as fresh
from scripts.aws_eval import grounded_hf_lifecycle as lifecycle
from scripts.aws_eval import public_work_experiment as public


def normalized(path):
    return Path(unicodedata.normalize('NFD', str(Path(path).resolve())))


def finalize(run_dir, state, registration_sha256, *, root):
    root, run, state = normalized(root), normalized(run_dir), normalized(state)
    fresh.require(run.is_relative_to(root / 'runs') and state.is_relative_to(run),
                  'Repository-local run and run-owned state required')
    fresh.require(not any((run / name).exists() for name in
                          ('plan.json', 'approval.json', 'launch-package-pins.json')),
                  'Final artifacts already exist; do not overwrite or replay')
    provisional = fresh.load(run / 'provisional-plan.json')
    reg_pin = provisional['registration']
    fresh.require(reg_pin['sha256'] == registration_sha256,
                  'Explicit registration trust anchor differs')
    reg = fresh.bound_json(root, reg_pin)
    fresh.require(reg['schema'] == 'trid-public-work-prerequisites-v1',
                  'Public-work registration required')
    fresh.validate_registration(reg, root)
    expires = datetime.fromisoformat(provisional['observations_expire_at'].replace('Z', '+00:00'))
    fresh.require(datetime.now(timezone.utc) < expires,
                  'Original observations expired; obtain new evidence/admission, never relabel timestamps')
    cloud = fresh.bound_json(root, provisional['cloud_evidence'])
    for observation in cloud['observations'].values():
        fresh.pin_bytes(root, observation)
    owned = fresh.load(state / 'owned-prerequisites.json')
    bundle = fresh.load(state / 'bundle-prepared.json')
    fresh.require(owned['registration_sha256'] == reg_pin['sha256']
                  and owned['identity'] == reg['identity'], 'Owned readback belongs to another registration')
    resources = owned['resources']
    fresh.require(resources['bucket_lifecycle_configuration'] == fresh.lifecycle_merge(reg),
                  'Actual lifecycle readback differs from the approved complete merge')
    for key in fresh.ROLE_KEYS:
        fresh.require(resources[key]['inventory'] == reg['iam'][key],
                      'Actual role readback differs from registration: ' + key)
    plan = copy.deepcopy(provisional['plan_fields'])
    expected_outputs = public.output_members(plan)
    fresh.require(set(plan['output_allowlist']) == expected_outputs
                  and len(plan['output_allowlist']) == len(expected_outputs),
                  'Provisional output allowlist differs from sealed workload')
    plan['bundle'] = bundle
    plan['network'].update(security_group_id=resources['security_group']['GroupId'],
                           security_group_observation=resources['security_group'])
    plan['cost']['bucket_lifecycle_configuration'] = resources['bucket_lifecycle_configuration']
    plan_sha = hashlib.sha256(lifecycle.canonical(plan) + b'\n').hexdigest()
    approval = dict(schema=public.APPROVAL_SCHEMA,
        run_id=reg['identity']['run_id'], plan_sha256=plan_sha, instance_limit=1,
        cap_usd=str(plan['cost']['cap_usd']), workload_sha256=lifecycle.object_sha(plan['workload']),
        authorization_sha256=plan['authorization']['sha256'], limits=plan['limits'],
        ebs_kms_key_arn=plan['ebs_kms_key_arn'], automatic_retry=False,
        automatic_extension=False, retraining=True, deployment=False, approved=True,
        absolute_termination_deadline=plan['absolute_termination_deadline'])
    fresh.validate_final(reg, reg_pin['sha256'], root, state, plan, approval, plan_sha)
    fresh.record(run / 'plan.json', plan)
    fresh.record(run / 'approval.json', approval)
    result = dict(registration=reg_pin, plan=public.pin(run / 'plan.json', root),
        approval=public.pin(run / 'approval.json', root),
        owned_prerequisites=public.pin(state / 'owned-prerequisites.json', root),
        prepared_bundle=public.pin(state / 'bundle-prepared.json', root),
        cloud_evidence=provisional['cloud_evidence'], assembler=public.pin(normalized(__file__), root),
        cloud_mutations=0, automatic_retry=False)
    fresh.record(run / 'launch-package-pins.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--approve-fresh-plan', action='store_true', required=True,
                        help='Record operator approval of this freshly assembled plan only')
    args = parser.parse_args(argv)
    result = finalize(args.run_dir, args.state, args.registration_sha256,
                      root=Path(__file__).resolve().parents[2])
    print(lifecycle.canonical(result).decode())


if __name__ == '__main__':
    main()
