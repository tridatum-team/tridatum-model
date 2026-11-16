"""Offline, parent-registered metadata recovery of one immutable CUDA execution.

No AWS client, generation, retry or native-success rewrite exists here. Register
with ``registration-spec --root ROOT --state STATE --source-archive TAR`` after
archiving the two SOURCES below; review and save that JSON, then invoke
``finalize --root ROOT --registration REG``. ``validate`` replays authentication,
not inference. All original failure evidence remains in the canonical export.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tarfile
import tempfile
import types
import uuid

SCHEMA = 'trid-cuda-evaluation-metadata-recovery-v1'
REGISTRATION_SCHEMA = SCHEMA + '-registration'
RUN = 'runs/mois-eight-cuda-evaluation-20261116T134815Z'
RUN_ID = 'mois-eight-cuda-eval-20261116-134815'
INSTANCE = 'i-0acc7c0208d7121d4'
VOLUME = 'vol-02404094c7979dcd4'
SOURCES = ('scripts/aws_eval/fresh_eight_evaluation_recover.py',
           'scripts/aws_eval/fresh_eight_evaluation.py')
ANCHORS = {
    'plan.json': (53209, '933dd5019c1abf2e4842dcc3495c6c77c1035ac060a606a62740ea64e5e8b932'),
    'registration.json': (18083, 'f4b32206c3f7980c76fc33eaa23063caafc525e01d7d579a63d6f04e3c636954'),
    'approval.json': (935, '96d04aa57737318f2b85a71439fab0548933ec714ed23d7765f3f1fde593218d'),
    'cloud-state/export.tar': (8161280, 'af75b62050e505458748ff99bf5c933062a5d64d552fbfb04f8ad42ca18d3224'),
}
FAILURE = dict(automatic_retry=False, completed_rows_retained=True,
    elapsed_seconds=5787.645240746,
    message="[Errno 13] Permission denied: '/opt/trid-" + RUN_ID + "/output/model-acquisition.json'",
    quality_accepted=False, status='failed', type='PermissionError')
PREFIX = 'RuntimeError: Native stage failed: evaluation: '
REG_NAME = 'evaluation-recovery-registration.json'
RECEIPT_NAME = 'evaluation-recovery-receipt.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def relative(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and not p.is_absolute() and '..' not in p.parts
            and p.as_posix() == name and name != '.', 'Unsafe portable path')
    return name


def pin(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file()
            and path.stat().st_nlink == 1, 'Missing or unsafe evidence file')
    name = relative(path.relative_to(root).as_posix())
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=name, bytes=path.stat().st_size, sha256=sha)


def bound(root, record):
    require(set(record) == {'path', 'bytes', 'sha256'} and type(record['bytes']) is int
            and record['bytes'] >= 0, 'Invalid evidence pin')
    path = Path(root) / relative(record['path'])
    require(pin(path, root) == record, 'Pinned evidence bytes differ: ' + record['path'])
    return path


def byte_pin(name, raw):
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def archive_files(path, records=None):
    """Authenticate every member before permitting any extraction; never extractall."""
    expected = None
    if records is not None:
        expected = {relative(p['path']): p for p in records}
        require(len(expected) == len(records), 'Duplicate registered archive member')
    result = {}
    total = 0
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            name = relative(member.name)
            require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.sparse and name not in result
                    and set(member.pax_headers) <= {'mtime'}
                    and ('mtime' not in member.pax_headers or math.isfinite(float(member.pax_headers['mtime']))),
                    'Unsafe archive member')
            total += member.size
            require(0 <= member.size <= 128 * 1024**2 and total <= 512 * 1024**2
                    and len(result) < 4096, 'Archive resource envelope exceeded')
            raw = archive.extractfile(member).read()
            require(len(raw) == member.size, 'Truncated archive member')
            if expected is not None:
                require(name in expected and byte_pin(name, raw) == expected[name], 'Archive member bytes differ')
            result[name] = raw
    if expected is not None:
        require(set(result) == set(expected), 'Archive member set differs')
    return result


def materialize(folder, members):
    folder = Path(folder)
    require(not folder.exists() and not folder.is_symlink(), 'Extraction destination already exists')
    require(not any(p.is_symlink() for p in folder.parents), 'Unsafe extraction parent')
    folder.mkdir(parents=True)
    for name, raw in members.items():
        path = folder / relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)


def assert_export(folder, members):
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), 'Canonical export missing')
    actual = set()
    for path in folder.rglob('*'):
        require(not path.is_symlink(), 'Export symlink forbidden')
        if path.is_dir():
            continue
        name = path.relative_to(folder).as_posix()
        require(name in members and pin(path, folder) == byte_pin(name, members[name]), 'Canonical export bytes differ')
        actual.add(name)
    require(actual == set(members), 'Canonical export membership differs')


@contextmanager
def modules(folder):
    """Isolated historical package: no imports from the mutable checkout."""
    name = '_trid_recovery_' + uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(Path(folder) / 'scripts/aws_eval')]
    sys.modules[name] = package
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield lambda module: importlib.import_module(name + '.' + module)
    finally:
        sys.dont_write_bytecode = old
        for key in list(sys.modules):
            if key == name or key.startswith(name + '.'):
                del sys.modules[key]


def validate_failure(failure):
    require(canonical(failure) == canonical(FAILURE), 'Only the exact post-generation PermissionError is recoverable')


def parse_failed_unit(completion, plan, release):
    error = completion.get('error')
    require(isinstance(error, str) and error.startswith(PREFIX), 'Unknown native failure prefix')
    try:
        unit = ast.literal_eval(error[len(PREFIX):])
    except (ValueError, SyntaxError) as exc:
        raise ValueError('Malformed captured failed unit') from exc
    keys = {'Result', 'ExecMainStatus', 'ControlGroup', 'MemoryMax', 'MemorySwapMax', 'DevicePolicy',
            'Environment', 'User', 'Group', 'LoadState', 'ActiveState', 'SubState', 'InvocationID'}
    require(isinstance(unit, dict) and set(unit) == keys and all(isinstance(v, str) for v in unit.values()),
            'Captured failed unit shape differs')
    expected = dict(Result='exit-code', ExecMainStatus='1', ControlGroup='',
        MemoryMax=str(plan['limits']['rss_bytes']), MemorySwapMax='0', DevicePolicy='closed',
        User='995', Group='985', LoadState='loaded', ActiveState='failed', SubState='failed')
    require(all(unit[k] == v for k, v in expected.items()), 'Failed unit status/limits/user/reap differs')
    gpu = release['gpu']['uuid']
    base = '/opt/trid-' + RUN_ID + '/scratch/'
    env = dict(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', AWS_EC2_METADATA_DISABLED='true', PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6',
        HF_HOME=base + 'cache', TMPDIR=base + 'tmp', TRID_OWNED_RUN=RUN_ID)
    parts = shlex.split(unit['Environment'])
    require(len(parts) == len(env) and set(parts) == {k + '=' + v for k, v in env.items()},
            'Failed unit offline environment/GPU differs')
    require(re.fullmatch('[0-9a-f]{32}', unit['InvocationID']) is not None
            and unit['InvocationID'] != completion['runtime_unit']['invocation'], 'Failed unit invocation missing/reused')
    attempt = completion['phase_attempt']
    require(set(attempt) == {'phase', 'unit', 'work_deadline', 'started_at', 'registered_seconds',
                            'effective_seconds', 'phase_deadline'}
            and attempt['phase'] == 'evaluation' and attempt['unit'] == RUN_ID + '-evaluation', 'Phase attempt differs')
    require(all(type(attempt[k]) in (int, float) and math.isfinite(attempt[k]) for k in
                ('work_deadline', 'started_at', 'phase_deadline', 'registered_seconds', 'effective_seconds')),
            'Nonfinite phase lease')
    seconds = attempt['effective_seconds']
    require(type(seconds) is int and type(attempt['registered_seconds']) is int
            and attempt['registered_seconds'] == plan['limits']['lifetime_seconds'] - plan['limits']['reserve_seconds'] == 18600
            and FAILURE['elapsed_seconds'] < seconds <= 18600
            and release['launch_time'] <= attempt['started_at'] < attempt['phase_deadline'] <= attempt['work_deadline']
            and attempt['work_deadline'] == release['deadline'] - plan['limits']['reserve_seconds']
            and seconds <= attempt['phase_deadline'] - attempt['started_at'] <= seconds + 30,
            'Evaluation timeout/phase allocation differs')
    return unit


def authenticate_factoring(original, recovery):
    """Prove the registered evidence validator retains every historical content check."""
    old_tree, new_tree = ast.parse(original), ast.parse(recovery)
    old = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    require([ast.dump(n) for n in old_tree.body if not isinstance(n, ast.FunctionDef)]
            == [ast.dump(n) for n in new_tree.body if not isinstance(n, ast.FunctionDef)],
            'Recovery profile constants/imports differ from historical execution')
    def strip_doc(node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    for name, node in old.items():
        if name in ('validate_mode', 'validate_completion'):
            target = name + '_artifacts'
            node.name = target
            if name == 'validate_mode':
                gate = ast.parse("require(not (folder / 'failure.json').exists(), 'Failure evidence forbids completion')").body[0]
                matches = [i for i, row in enumerate(node.body) if ast.dump(row) == ast.dump(gate)]
                require(len(matches) == 1, 'Historical failure gate differs')
                del node.body[matches[0]]
            else:
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == 'validate_mode':
                        child.id = 'validate_mode_artifacts'
        else:
            target = name
        require(target in new, 'Recovery validator removed historical contract')
        strip_doc(node)
        strip_doc(new[target])
        require(ast.dump(node) == ast.dump(new[target]), 'Recovery content validator differs: ' + name)


def source_bound_failure(bundle):
    worker = bundle['scripts/aws_eval/grounded_hf_worker.py'].decode()
    tree = ast.parse(worker)
    stage = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'supervise_phase')
    source = ast.get_source_segment(worker, stage)
    require("'Native stage failed: ' + phase + ': ' + str(state)" in source
            and "state['MemoryMax'] == str(LIMITS['rss_bytes'])" in source
            and "used <= LIMITS['gpu_bytes']" in source
            and 'finally:' in source and 'stopped = stop_unit(' in source,
            'Historical failed-unit capture/supervision boundary differs')
    require("report['error'] = f'{type(exc).__name__}: {exc}'" in worker, 'Historical typed error capture differs')


def lineage_pins(root, lineage):
    candidate = bound(root, lineage['candidate'])
    require(load(candidate) == dict(schema='trid-fresh-eight-cuda-training-candidate-v2',
        plan=lineage['provenance']['plan'], candidate_id=lineage['provenance']['candidate_id'],
        arm='candidate', step=150, completion_scope='cuda_capacity_train_export_only',
        quality_accepted=False, cloud_evaluation=False), 'Original candidate commitment differs')
    found = {}
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'bytes', 'sha256'}:
                base = candidate.parent if value['path'] == 'plan.json' or value['path'].startswith('train/') else root
                p = bound(base, value)
                found[str(p)] = pin(p, root)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(lineage)
    return sorted(found.values(), key=lambda p: p['path'])


def registration_spec(root, state, recovery_source_archive):
    root, state = Path(root).resolve(), Path(state).resolve()
    require(state == root / RUN / 'cloud-state', 'Recovery run namespace differs')
    anchors = {}
    for name, (size, sha) in ANCHORS.items():
        record = dict(path=RUN + '/' + name, bytes=size, sha256=sha)
        bound(root, record)
        anchors[name] = record
    plan = load(bound(root, anchors['plan.json']))
    bundle = dict(path=plan['bundle']['local_path'], bytes=plan['bundle']['bytes'], sha256=plan['bundle']['sha256'])
    bound(root, bundle)
    sources = [pin(root / name, root) for name in SOURCES]
    source_archive = pin(recovery_source_archive, root)
    archive_files(bound(root, source_archive), sources)
    originals = [pin(p, root) for p in sorted(state.glob('*.json')) if not p.name.startswith('evaluation-recovery-')]
    lineage = load(bound(root, plan['workload']['lineage_binding']))
    members = archive_files(bound(root, anchors['cloud-state/export.tar']))
    failure_pin = byte_pin('evaluation/failure.json', members['evaluation/failure.json'])
    return dict(schema=REGISTRATION_SCHEMA, approved=True, metadata_only=True, additional_gpu_executions=0,
        aws_requests=0, native_success=False, quality_accepted=False, deployment=False,
        state=state.relative_to(root).as_posix(), anchors=anchors, source_bundle=bundle,
        original_failure=failure_pin, originals=originals, lineage=lineage_pins(root, lineage),
        recovery_sources=sources, recovery_source_archive=source_archive,
        instance_id=INSTANCE, root_volume_id=VOLUME)


def validate_registration(root, registration_path, *, active=False):
    root, registration_path = Path(root).resolve(), Path(registration_path).absolute()
    r = load(registration_path)
    require(set(r) == {'schema', 'approved', 'metadata_only', 'additional_gpu_executions', 'aws_requests',
        'native_success', 'quality_accepted', 'deployment', 'state', 'anchors', 'source_bundle',
        'original_failure', 'originals', 'lineage', 'recovery_sources', 'recovery_source_archive',
        'instance_id', 'root_volume_id'}, 'Recovery registration fields differ')
    require(r['schema'] == REGISTRATION_SCHEMA and r['approved'] is True and r['metadata_only'] is True
            and type(r['additional_gpu_executions']) is int and r['additional_gpu_executions'] == 0
            and type(r['aws_requests']) is int and r['aws_requests'] == 0
            and all(r[k] is False for k in ('native_success', 'quality_accepted', 'deployment'))
            and r['state'] == RUN + '/cloud-state' and r['instance_id'] == INSTANCE and r['root_volume_id'] == VOLUME,
            'Not the approved metadata-only recovery')
    state = root / r['state']
    require(registration_path.parent == state and registration_path.name == REG_NAME, 'Recovery registration endpoint differs')
    pin(registration_path, root)
    require(r['anchors'] == {n: dict(path=RUN + '/' + n, bytes=b, sha256=s) for n, (b, s) in ANCHORS.items()},
            'Immutable original run anchors differ')
    for record in [*r['anchors'].values(), r['source_bundle'], *r['originals'], *r['lineage']]:
        bound(root, record)
    originals = {p['path']: p for p in r['originals']}
    require(len(originals) == len(r['originals']) and all(Path(n).parent == Path(r['state']) for n in originals),
            'Original evidence inventory differs')
    required = {'completion.json', 'controller-result.json', 'volume-identity.json', 'release.json',
        'approval-consumed.json', 'registration-consumed.json', 'owned-prerequisites.json',
        'owned-cleanup-started.json', 'owned-cleanup-complete.json', 'bundle-prepared.json',
        *('cleanup-' + n + '-' + s + '.json' for n in
          ('schedule-group', 'profile', 'security-group', 'instance_role', 'scheduler_role')
          for s in ('intent', 'response', 'readback'))}
    require({r['state'] + '/' + n for n in required} <= set(originals), 'Required original proof not registered')
    require({p['path'] for p in r['recovery_sources']} == set(SOURCES) and len(r['recovery_sources']) == 2,
            'Recovery executable closure differs')
    code = archive_files(bound(root, r['recovery_source_archive']), r['recovery_sources'])
    if active:
        require(os.path.samefile(__file__, root / SOURCES[0]), 'Executing recovery module is not registered file')
        for p in r['recovery_sources']:
            bound(root, p)
    return r, state, code


def validate_cleanup(root, state, registration, plan, completion, controller, lifecycle, launch):
    reg_sha = ANCHORS['registration.json'][1]
    volume = lifecycle.load_volume_identity(state, plan, ANCHORS['plan.json'][1])
    cleanup = controller['cleanup']
    require(volume['instance_id'] == INSTANCE and volume['root_volume_ids'] == [VOLUME]
            and set(volume['volumes']) == {VOLUME} and cleanup.get('cleanup_state') == 'complete'
            and cleanup.get('instance_terminated') is True and cleanup.get('instance_id') == INSTANCE
            and cleanup.get('errors') == [] and cleanup.get('remaining_volumes') == []
            and all(cleanup.get(k) == [VOLUME] for k in ('root_volume_ids', 'recorded_volume_ids', 'absent_volume_ids')),
            'Exact owned instance/root positive cleanup missing')
    owned = load(state / 'owned-prerequisites.json')
    require(load(state / 'owned-cleanup-complete.json') == dict(registration_sha256=reg_sha,
        cleanup_state='complete', retained_prefix_lifecycle_rules=True, s3_mutations=0)
        and load(state / 'owned-cleanup-started.json') == dict(registration_sha256=reg_sha, resources=owned['resources'])
        and load(state / 'registration-consumed.json') == dict(registration_sha256=reg_sha, identity=registration['identity']),
        'Owned prerequisite cleanup authority differs')
    identity = registration['identity']
    deletions = {
        'schedule-group': ('scheduler', 'delete-schedule-group', ['--name', identity['schedule_group']], 'ResourceNotFoundException'),
        'profile': ('iam', 'delete-instance-profile', ['--instance-profile-name', identity['instance_profile']], 'NoSuchEntity'),
        'security-group': ('ec2', 'delete-security-group', ['--group-id', owned['resources']['security_group']['GroupId']], 'InvalidGroup.NotFound'),
        **{k: ('iam', 'delete-role', ['--role-name', identity[k]], 'NoSuchEntity') for k in launch.ROLE_KEYS}}
    for name, (service, operation, args, code) in deletions.items():
        intent, response, readback = [load(state / ('cleanup-' + name + '-' + s + '.json')) for s in ('intent', 'response', 'readback')]
        require(intent == dict(registration_sha256=reg_sha, service=service, operation=operation, args=args)
                and response.get('request') == readback.get('request') == intent
                and readback.get('observation') == dict(absent=True, error_code=code), 'Prerequisite absence proof differs: ' + name)
    return cleanup


def authenticate(root, r, state, code):
    """Recompute content, failure, source, ownership and cleanup; never trust a summary."""
    plan = load(bound(root, r['anchors']['plan.json']))
    registration = load(bound(root, r['anchors']['registration.json']))
    approval = load(bound(root, r['anchors']['approval.json']))
    plan_sha = r['anchors']['plan.json']['sha256']
    require(r['source_bundle'] == dict(path=plan['bundle']['local_path'], bytes=plan['bundle']['bytes'],
        sha256=plan['bundle']['sha256']), 'Original source bundle differs')
    bundle = archive_files(bound(root, r['source_bundle']), plan['bundle']['files'])
    source_bound_failure(bundle)
    authenticate_factoring(bundle[SOURCES[1]], code[SOURCES[1]])
    members = archive_files(bound(root, r['anchors']['cloud-state/export.tar']))
    with tempfile.TemporaryDirectory(prefix='trid-cuda-recovery-') as tmp:
        tmp = Path(tmp).resolve()
        materialize(tmp / 'original', bundle)
        materialize(tmp / 'recovery', code)
        materialize(tmp / 'output', members)
        output = tmp / 'output'
        with modules(tmp / 'original') as original, modules(tmp / 'recovery') as recovery:
            historical = original('fresh_eight_evaluation')
            lifecycle = original('grounded_hf_lifecycle')
            launch = original('fresh_eight_launch')
            profile = recovery('fresh_eight_evaluation')
            launch.validate_registration(registration, tmp / 'original', active=False)
            launch.validate_final(registration, r['anchors']['registration.json']['sha256'], tmp / 'original',
                                  state, plan, approval, plan_sha, active=False)
            data = historical.validate_data(tmp / 'original', plan)
            require(lineage_pins(root, data['lineage']) == r['lineage'], 'Original candidate lineage proofs differ')
            completion = load(state / 'completion.json')
            controller = load(state / 'controller-result.json')
            export = load(output / 'export-manifest.json')
            require(export['run_id'] == RUN_ID and export['plan_sha256'] == plan_sha
                    and export['status'] == {k: v for k, v in completion.items() if k != 'export'}
                    and completion['export'] == {k: r['anchors']['cloud-state/export.tar'][k] for k in ('bytes', 'sha256')},
                    'Original archive terminal binding differs')
            records = export['files']
            require(len({p['path'] for p in records}) == len(records)
                    and {p['path'] for p in records} <= set(plan['output_allowlist'])
                    and set(members) == {p['path'] for p in records} | {'export-manifest.json'}
                    and export['total_bytes'] == sum(p['bytes'] for p in records)
                    and export['export_cap_bytes'] == export['total_output_cap_bytes'] == plan['limits']['output_bytes'],
                    'Original export member/limit contract differs')
            for record in records:
                bound(output, record)
            require(byte_pin('evaluation/failure.json', members['evaluation/failure.json']) == r['original_failure'],
                    'Original failure bytes differ')
            validate_failure(load(output / 'evaluation/failure.json'))
            require('smoke/failure.json' not in members and 'evaluation/supervisor-result.json' not in members,
                    'Unexpected failure or invented supervisor acceptance')
            require(completion.get('evaluation_success') is False and completion.get('quality_accepted') is False
                    and completion.get('completion_scope') == historical.COMPLETION_SCOPE
                    and completion.get('run_id') == RUN_ID and completion.get('instance_id') == INSTANCE
                    and completion.get('plan_sha256') == plan_sha and completion.get('stages') == []
                    and completion.get('stop_errors') == [] and not completion.get('export_error')
                    and not any(k in completion for k in ('training_success', 'inference_success'))
                    and completion['runtime'] == load(output / 'runtime.json'), 'Original native-failed status changed')
            evidence = controller.get('evidence', {})
            require(controller.get('state') == 'native_failed' and controller.get('evaluation_success') is False
                    and controller.get('quality_accepted') is False and controller.get('instance_id') == INSTANCE
                    and not any(controller.get(k) for k in ('error', 'evidence_error', 'schedule_cleanup_error'))
                    and evidence.get('native_pipeline_success') is False and evidence.get('evidence_kind') == 'authenticated_archive'
                    and evidence.get('export_state') == 'authenticated_partial' and evidence.get('completion') == completion
                    and evidence.get('export_sha256') == completion['export']['sha256'], 'Original failed controller evidence changed')
            release = load(state / 'release.json')
            require(all(release[k] == v for k, v in dict(run_id=RUN_ID, plan_sha256=plan_sha, instance_id=INSTANCE).items())
                    and release['launch_time'] == controller['launch_time'] and release['deadline'] == controller['deadline']
                    and release['deadline'] - release['launch_time'] == plan['limits']['lifetime_seconds']
                    and release.get('schedule_verified') is True
                    and load(state / 'approval-consumed.json') == dict(run_id=RUN_ID, plan_sha256=plan_sha),
                    'Original one-shot approval/release differs')
            unit = parse_failed_unit(completion, plan, release)
            runtime = completion['runtime_unit']
            stopped = runtime['stopped_state']
            require(runtime['phase'] == 'runtime' and runtime['state'] == 'complete'
                    and runtime['unit'] == RUN_ID + '-runtime' and runtime['run_id'] == RUN_ID
                    and runtime['plan_sha256'] == plan_sha and runtime['instance_id'] == INSTANCE
                    and runtime['cleanup_verified'] is True and re.fullmatch('[0-9a-f]{32}', runtime['invocation'])
                    and stopped.get('ControlGroup') == '' and stopped.get('Result') == 'success'
                    and stopped.get('ExecMainStatus') == '0' and stopped.get('ActiveState') == 'inactive'
                    and stopped.get('LoadState') == 'not-found', 'Original runtime failure/reap differs')
            lifecycle.verify_gpu_observation(runtime['gpu'], plan, release['gpu'])
            cleanup = validate_cleanup(root, state, registration, plan, completion, controller, lifecycle, launch)
            # Failure is present. Content-only validation does not assert process success.
            worker = profile.validate_completion_artifacts(output, plan)
            manifest = load(output / 'evaluation/run-manifest.json')
            require(manifest['selected_ids'] == [p['id'] for p in data['evaluation_inputs']]
                    and manifest['runtime']['gpu_uuid'] == release['gpu']['uuid']
                    and manifest['runtime']['instance_id'] == INSTANCE, 'Public denominator or observed GPU differs')
            proof = dict(native_state='native_failed', native_success=False, metadata_only=True,
                generation_calls=worker['generation_calls'], rows_per_arm=worker['rows_per_arm'],
                smoke_rows_per_arm=worker['smoke_rows_per_arm'], failed_unit=unit, phase_attempt=completion['phase_attempt'],
                runtime_unit=runtime, cleanup=cleanup, original_failure=r['original_failure'],
                worker_result=byte_pin('evaluation/evaluation-worker.json', members['evaluation/evaluation-worker.json']),
                members=[byte_pin(n, raw) for n, raw in sorted(members.items())])
    return proof, members, plan


def receipt_value(root, registration_path, r, proof):
    return dict(schema=SCHEMA, state='metadata_recovered_native_failed', registration=pin(registration_path, root),
        original_archive=r['anchors']['cloud-state/export.tar'], original_failure=r['original_failure'],
        source_bundle=r['source_bundle'], executed_recovery_sources=r['recovery_source_archive'],
        metadata_only=True, additional_gpu_executions=0, aws_requests=0, native_success=False,
        native_pipeline_success=False, complete_answers_authenticated=True,
        original_post_generation_failure=dict(FAILURE), quality_accepted=False, deployment=False, validation=proof)


def finalize(root, registration_path):
    root = Path(root).resolve()
    r, state, code = validate_registration(root, registration_path, active=True)
    require(not (state / RECEIPT_NAME).exists(), 'Recovery already finalized')
    proof, members, _ = authenticate(root, r, state, code)
    output = state / 'cuda-evaluation-export'
    materialize(output, members)
    receipt = receipt_value(root, registration_path, r, proof)
    with (state / RECEIPT_NAME).open('xb') as stream:
        stream.write(canonical(receipt) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def _validate_receipt(root, output, plan, plan_sha):
    root, output = Path(root).resolve(), Path(output).absolute()
    state = root / RUN / 'cloud-state'
    require(output == state / 'cuda-evaluation-export', 'Recovery consumption endpoint differs')
    receipt = load(state / RECEIPT_NAME)
    registration_path = bound(root, receipt['registration'])
    r, _, code = validate_registration(root, registration_path)
    proof, members, original_plan = authenticate(root, r, state, code)
    require(original_plan == plan and plan_sha == r['anchors']['plan.json']['sha256'], 'Recovery consumer plan differs')
    require(receipt == receipt_value(root, registration_path, r, proof), 'Recovery receipt does not match recomputed evidence')
    assert_export(output, members)
    paths = [registration_path, state / RECEIPT_NAME]
    records = [*r['anchors'].values(), r['source_bundle'], r['recovery_source_archive'], *r['originals'], *r['lineage']]
    return dict(pins={**{'cuda-recovery-' + str(i): p for i, p in enumerate(records)},
                      **{'cuda-recovery-receipt-' + str(i): pin(p, root) for i, p in enumerate(paths)}},
                public_input=bound(root, plan['workload']['evaluation_inputs']))


def validate_recovery(root, output, plan, plan_sha):
    """Dispatch to the registered historical recovery implementation, not HEAD."""
    root = Path(root).resolve()
    receipt = load(Path(output).parent / RECEIPT_NAME)
    registration_path = bound(root, receipt['registration'])
    r, _, code = validate_registration(root, registration_path)
    with tempfile.TemporaryDirectory(prefix='trid-recovery-validator-') as tmp:
        folder = Path(tmp).resolve() / 'source'
        materialize(folder, code)
        with modules(folder) as imported:
            return imported('fresh_eight_evaluation_recover')._validate_receipt(root, output, plan, plan_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('registration-spec', 'finalize', 'validate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--source-archive', type=Path)
    parser.add_argument('--registration', type=Path)
    args = parser.parse_args(argv)
    if args.operation == 'registration-spec':
        require(args.state is not None and args.source_archive is not None, 'State and source archive required')
        value = registration_spec(args.root, args.state, args.source_archive)
    elif args.operation == 'finalize':
        require(args.registration is not None, 'Registration required')
        value = finalize(args.root, args.registration)
    else:
        state = args.root / RUN / 'cloud-state'
        plan = load(state.parent / 'plan.json')
        value = validate_recovery(args.root, state / 'cuda-evaluation-export', plan, ANCHORS['plan.json'][1])
        value['public_input'] = str(value['public_input'])
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
