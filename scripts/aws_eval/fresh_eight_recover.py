"""Explicit, parent-registered collection recovery; never launches or mutates AWS resources.

The failed controller and its partial archive remain immutable. A separate receipt
accepts only fully authenticated training evidence and fresh positive cleanup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time

if __package__:
    from . import grounded_hf_lifecycle as lifecycle
    from . import fresh_eight_experiment as fresh
    from . import qwen_cuda_smoke as smoke
else:
    import grounded_hf_lifecycle as lifecycle
    import fresh_eight_experiment as fresh
    import qwen_cuda_smoke as smoke

require = lifecycle.require
SCHEMA = 'trid-fresh-eight-collection-recovery-v1'
REGISTRATION_SCHEMA = 'trid-fresh-eight-collection-recovery-registration-v1'
AUTHORIZATION_SCHEMA = 'trid-fresh-eight-collection-recovery-authorization-v1'
PRIOR_BYTES = 512 * 1024**2
PRIOR_REQUESTS = 4096
MAX_REQUESTS = 128
SOURCES = ('scripts/aws_eval/fresh_eight_recover.py',
           'scripts/aws_eval/grounded_hf_lifecycle.py',
           'scripts/aws_eval/fresh_eight_experiment.py',
           'scripts/aws_eval/qwen_cuda_smoke.py',
           'scripts/aws_eval/qwen_cpu_smoke.py')
ORIGINALS = ('completion.json', 'controller-result.json', 'release.json',
             'volume-identity.json', 'export-retrieval-intent.json',
             'source-bundle.tar', 'export.tar')


def pin(root, path):
    root, path = Path(root).resolve(), Path(path)
    require(all(not parent.is_symlink() for parent in (path, *path.parents)),
            'Recovery evidence path contains a symlink')
    require(not path.is_symlink() and path.is_file() and path.stat().st_nlink == 1,
            'Recovery evidence must be an unlinked regular file')
    return dict(path=path.resolve().relative_to(root).as_posix(),
                bytes=path.stat().st_size, sha256=lifecycle.digest(path))


def bound(root, record):
    path = Path(root) / lifecycle.relative(record['path'])
    require(pin(root, path) == record, 'Recovery evidence pin differs')
    return path


def load(path):
    return json.loads(Path(path).read_bytes())


def create_once(path, value):
    """Durable intent before any request; an interrupted attempt is never refunded."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def registration_spec(root, state, authorization_path, deadline):
    """Offline parent helper, not an authorization or an AWS operation."""
    root, state = Path(root).resolve(), Path(state).resolve()
    plan_path = state.parent / 'plan.json'
    plan, completion = load(plan_path), load(state / 'completion.json')
    return dict(schema=REGISTRATION_SCHEMA, run_id=plan['identity']['run_id'],
        collection_only=True, additional_gpu_executions=0,
        authorization=pin(root, authorization_path), plan=pin(root, plan_path),
        originals={name: pin(root, state / name) for name in ORIGINALS},
        sources=[pin(root, root / name) for name in SOURCES],
        bucket=plan['identity']['bucket'], key=plan['identity']['output_prefix'] + 'export.tar',
        export=completion['export'], instance_id=completion['instance_id'],
        root_volume_ids=load(state / 'volume-identity.json')['root_volume_ids'],
        registered_at=time.time(), deadline=deadline,
        prior_retrieval_bytes=PRIOR_BYTES, prior_requests=PRIOR_REQUESTS,
        new_retrieval_byte_limit=PRIOR_BYTES, new_request_limit=MAX_REQUESTS,
        cumulative_retrieval_byte_limit=1024**3,
        cumulative_request_limit=plan['cost']['get_requests'])


def validate_context(root, authorization, registration):
    """Bind registration to the parent's pre-existing evidence, not fresh pins."""
    require(isinstance(authorization.get('context'), dict),
            'Original collection authorization context missing')
    context = load(bound(root, authorization['context']))
    require(context.get('schema') == 'trid-existing-training-collection-authorization-v1'
            and context.get('plan') == registration['plan']
            and context.get('completion') == registration['originals']['completion.json']
            and context.get('original_controller') == registration['originals']['controller-result.json']
            and context.get('original_partial_archive') == registration['originals']['export.tar']
            and context.get('expected_export') == registration['export']
            and context.get('instance_id') == registration['instance_id']
            and registration['root_volume_ids'] == [context.get('root_volume_id')]
            and context.get('controller_retrieval_total_bytes_max') == 1024**3
            and context.get('prior_activity_conservative_debit_bytes') == PRIOR_BYTES
            and context.get('new_payload_bytes_max') == PRIOR_BYTES
            and context.get('automatic_gpu_retry') is False
            and context.get('preserve_original_failure') is True
            and context.get('budget_change_usd') == 0,
            'Parent original evidence or collection-only allowance differs')
    for key in ('plan', 'completion', 'original_controller', 'original_partial_archive'):
        bound(root, context[key])
    return context


def validate_executing_module(root, name, module_path):
    """Require the registered physical file, including on Unicode-normalizing FS."""
    expected, actual = Path(root) / name, Path(module_path)
    require(all(not path.is_symlink() for path in (actual, *actual.parents))
            and actual.is_file() and actual.stat().st_nlink == 1
            and os.path.samefile(expected, actual),
            f'Executing recovery module differs from registered checkout: '
            f'expected={str(expected)!r}; actual={str(actual)!r}')


def authenticate_recovery_sources(root, archive_pin, sources):
    """Read sealed historical code only; PAX timestamps never control extraction."""
    path = bound(root, archive_pin)
    expected = lifecycle.pins(sources, sum(row['bytes'] for row in sources))
    require(set(expected) == set(SOURCES), 'Historical recovery source closure differs')
    seen = set()
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            name = lifecycle.relative(member.name)
            require(member.isfile() and not member.sparse and name not in seen
                    and name in expected and member.size == expected[name]['bytes']
                    and set(member.pax_headers) <= {'mtime'}
                    and ('mtime' not in member.pax_headers
                         or lifecycle.math.isfinite(float(member.pax_headers['mtime']))),
                    'Unsafe or unregistered historical recovery source member')
            seen.add(name)
            sha = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    sha.update(block)
            require(sha.hexdigest() == expected[name]['sha256'], 'Historical recovery source bytes differ')
    require(seen == set(expected), 'Historical recovery source member missing')


def validate_registration(root, registration_path, *, active=False, source_archive=None):
    root, registration_path = Path(root).resolve(), Path(registration_path).resolve()
    registration = load(registration_path)
    state = registration_path.parent
    fields = set(registration_spec(root, state, bound(root, registration['authorization']),
                                   registration['deadline']))
    require(set(registration) == fields and registration['schema'] == REGISTRATION_SCHEMA
            and registration['collection_only'] is True
            and registration['additional_gpu_executions'] == 0,
            'Collection-only registration differs')
    authorization = load(bound(root, registration['authorization']))
    require(authorization['schema'] == AUTHORIZATION_SCHEMA
            and authorization['run_id'] == registration['run_id']
            and authorization['collection_only'] is True
            and authorization['additional_gpu_executions'] == 0
            and authorization['prior_retrieval_bytes'] == PRIOR_BYTES
            and authorization['prior_requests'] == PRIOR_REQUESTS,
            'Explicit collection-only parent authorization missing')
    validate_context(root, authorization, registration)
    if source_archive is None:
        require(registration['sources'] == [pin(root, root / name) for name in SOURCES],
                'Current recovery executable closure differs')
    else:
        require(not active and {row['path'] for row in registration['sources']} == set(SOURCES),
                'Historical source evidence cannot execute collection')
        authenticate_recovery_sources(root, source_archive, registration['sources'])
    if active:
        for name, module_path in ((SOURCES[0], __file__), (SOURCES[1], lifecycle.__file__),
                                  (SOURCES[2], fresh.__file__), (SOURCES[3], smoke.__file__),
                                  (SOURCES[4], smoke.CPU.__file__)):
            validate_executing_module(root, name, module_path)
    require(registration['plan'] == pin(root, state.parent / 'plan.json')
            and registration['originals'] == {name: pin(root, state / name) for name in ORIGINALS},
            'Original failure evidence changed')
    plan = load(bound(root, registration['plan']))
    plan_sha = registration['plan']['sha256']
    require(fresh.is_fresh(plan) and plan['identity']['run_id'] == registration['run_id'],
            'Recovery requires the original fresh training plan')
    completion, controller = load(state / 'completion.json'), load(state / 'controller-result.json')
    instance = registration['instance_id']
    require(controller.get('training_success') is False and controller.get('instance_id') == instance
            and controller.get('dispatch_attempted') is True and controller.get('evidence_error')
            and controller.get('state') == 'released'
            and not any(controller.get(key) for key in ('error', 'schedule_cleanup_error'))
            and controller.get('cleanup', {}).get('cleanup_state') == 'complete'
            and controller['cleanup'].get('instance_terminated') is True
            and controller['cleanup'].get('instance_id') == instance
            and not controller['cleanup'].get('remaining_volumes')
            and completion.get('training_success') is True,
            'Recovery requires an original collection failure after successful training')
    require(completion['run_id'] == registration['run_id'] and completion['plan_sha256'] == plan_sha
            and completion['instance_id'] == instance,
            'Original completion ownership differs')
    lifecycle.validate_fresh_terminal(plan, plan_sha, instance, completion)
    require(registration['export'] == completion['export']
            and set(registration['export']) == {'bytes', 'sha256'}
            and type(registration['export']['bytes']) is int
            and 0 < registration['export']['bytes'] <= PRIOR_BYTES
            and lifecycle.SHA.fullmatch(registration['export']['sha256'])
            and registration['bucket'] == plan['identity']['bucket']
            and registration['key'] == plan['identity']['output_prefix'] + 'export.tar',
            'Exact original export identity differs')
    volume = load(state / 'volume-identity.json')
    require(volume['instance_id'] == instance and volume['run_id'] == registration['run_id']
            and volume['plan_sha256'] == plan_sha and len(volume['root_volume_ids']) == 1
            and registration['root_volume_ids'] == volume['root_volume_ids']
            and re.fullmatch(r'i-[0-9a-f]+', instance)
            and re.fullmatch(r'vol-[0-9a-f]+', volume['root_volume_ids'][0]),
            'Exact original instance/root-volume identity differs')
    intent = load(state / 'export-retrieval-intent.json')
    require(intent == dict(run_id=registration['run_id'], plan_sha256=plan_sha,
                          instance_id=instance, byte_limit=1024**3)
            and plan['cost']['controller_retrieval_bytes'] == 1024**3
            and registration['prior_retrieval_bytes'] == PRIOR_BYTES
            and registration['new_retrieval_byte_limit'] == PRIOR_BYTES
            and registration['cumulative_retrieval_byte_limit'] == 1024**3
            and registration['prior_requests'] == PRIOR_REQUESTS
            and registration['new_request_limit'] == MAX_REQUESTS
            and registration['cumulative_request_limit'] == plan['cost']['get_requests']
            and PRIOR_REQUESTS + MAX_REQUESTS <= registration['cumulative_request_limit'],
            'Cumulative retrieval/request envelope differs')
    require(0 < registration['deadline'] - registration['registered_at'] <= 3600,
            'Recovery must have a finite one-hour maximum deadline')
    if active:
        require(registration['registered_at'] <= time.time() < registration['deadline'],
                'Recovery registration is expired or future dated')
    with tempfile.TemporaryDirectory() as temporary:
        historical = Path(temporary) / 'historical'
        lifecycle.authenticate_archive(state / 'source-bundle.tar', plan['bundle'], historical)
        fresh.validate_data(historical, plan)
    return registration, plan


class RecoveryClient(lifecycle.Client):
    """Read-only allowlist with write-ahead, failed-attempt-inclusive accounting."""
    def __init__(self, plan, registration, directory):
        super().__init__(plan['identity'], deadline=registration['deadline'],
                         request_limit=registration['new_request_limit'])
        self.registration = registration
        self.directory = Path(directory)
        self.directory.mkdir()
        self.reserved_bytes = 0
        self.reservations = []

    def __call__(self, service, operation, *args, **kwargs):
        r = self.registration
        lifecycle.remaining(self.deadline)
        require(not kwargs, 'Recovery client endpoint/region overrides forbidden')
        if service == 's3api' and operation in ('head-object', 'get-object'):
            require(args[:4] == ('--bucket', r['bucket'], '--key', r['key']),
                    'Recovery cannot retrieve a different object')
            requested = 1024**2
            if operation == 'get-object':
                match = re.fullmatch(r'bytes=([0-9]+)-([0-9]+)', args[args.index('--range') + 1])
                require(match is not None, 'Recovery range is missing')
                start, end = map(int, match.groups())
                requested = end - start + 1
                require(0 <= start <= end < r['export']['bytes'] and requested <= 8 * 1024**2
                        and '--if-match' in args, 'Recovery range/object conditional exceeds bounds')
        elif service == 'ec2' and operation == 'describe-instances':
            require(args == ('--instance-ids', r['instance_id']), 'Recovery instance lookup differs')
            requested = 1024**2
        elif service == 'ec2' and operation == 'describe-volumes':
            require(args == ('--volume-ids', r['root_volume_ids'][0]), 'Recovery volume lookup differs')
            requested = 1024**2
        else:
            raise ValueError('Recovery permits only exact-object/instance/volume read operations')
        require(len(self.reservations) < r['new_request_limit']
                and self.reserved_bytes + requested <= r['new_retrieval_byte_limit'],
                'Recovery cumulative allowance exhausted')
        reservation = dict(sequence=len(self.reservations) + 1, service=service, operation=operation,
                           arguments=list(args), reserved_bytes=requested, reserved_at=time.time())
        create_once(self.directory / f'{reservation["sequence"]:04d}.json', reservation)
        self.reservations.append(reservation)
        self.reserved_bytes += requested
        return super().__call__(service, operation, *args)


def observe_cleanup(client, registration):
    """No tag-search inference: require positive exact-ID terminal/absence proof."""
    observations = []
    for operation, option, identifier, absent in (
        ('describe-instances', '--instance-ids', registration['instance_id'], 'InvalidInstanceID.NotFound'),
        ('describe-volumes', '--volume-ids', registration['root_volume_ids'][0], 'InvalidVolume.NotFound')):
        started = time.time()
        try:
            response = client('ec2', operation, option, identifier)
        except lifecycle.AWSCallError as error:
            require(error.code == absent, 'Cleanup lookup failed without positive absence')
            response = dict(error_code=absent)
        observations.append(dict(operation=operation, identifier=identifier,
                                 started_at=started, observed_at=time.time(), response=response))
    validate_cleanup(observations, registration)
    return observations


def validate_cleanup(observations, registration):
    require(isinstance(observations, list) and len(observations) == 2, 'Fresh cleanup proof missing')
    for row, operation, identifier in zip(observations, ('describe-instances', 'describe-volumes'),
            (registration['instance_id'], registration['root_volume_ids'][0])):
        require(row['operation'] == operation and row['identifier'] == identifier
                and registration['registered_at'] <= row['started_at'] <= row['observed_at'] < registration['deadline'],
                'Fresh exact cleanup observation differs')
        response = row['response']
        if operation == 'describe-instances':
            if response == {'error_code': 'InvalidInstanceID.NotFound'}:
                continue
            instances = [i for reservation in response['Reservations'] for i in reservation['Instances']]
            require(len(instances) == 1 and instances[0]['InstanceId'] == identifier
                    and instances[0]['State']['Name'] == 'terminated', 'Exact instance is not terminated')
        else:
            require(response == {'error_code': 'InvalidVolume.NotFound'},
                    'Exact root volume has not been positively observed absent')


def validate_export(root, state, plan, registration, *, extract=False):
    path, output = state / 'recovered-export.tar', state / 'recovered-fresh-export'
    export = registration['export']
    require(pin(root, path)['bytes'] == export['bytes']
            and lifecycle.digest(path) == export['sha256'], 'Recovered archive bytes differ')
    completion = load(state / 'completion.json')
    with tarfile.open(path, 'r:') as archive:
        member = archive.getmember('export-manifest.json')
        require(member.isfile() and member.size <= 8 * 1024**2, 'Unsafe export manifest')
        with archive.extractfile(member) as stream:
            raw = stream.read()
    manifest = json.loads(raw)
    require(manifest['run_id'] == registration['run_id']
            and manifest['plan_sha256'] == registration['plan']['sha256']
            and manifest['status'] == {k: v for k, v in completion.items() if k != 'export'}
            and {r['path'] for r in manifest['files']} <= set(plan['output_allowlist']),
            'Recovered export membership/status differs')
    rows = manifest['files'] + [dict(path='export-manifest.json', bytes=len(raw),
                                    sha256=hashlib.sha256(raw).hexdigest())]
    lifecycle.authenticate_archive(path, dict(**export, files=rows,
        unpacked_bytes=lifecycle.output_limit(plan)), output if extract else None)
    require(not any(p.is_symlink() for p in output.rglob('*')),
            'Recovered extracted tree contains a symlink')
    require({p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
            == {r['path'] for r in rows}, 'Recovered extracted membership differs')
    for record in rows:
        require(pin(output, fresh.bound(output, record)) == record,
                'Recovered extracted file differs')
    require((output / 'plan.json').read_bytes() == (state.parent / 'plan.json').read_bytes(),
            'Export plan differs from sealed original plan')
    accepted = fresh.validate_completion(output, plan)
    release = load(state / 'release.json')
    require(release['run_id'] == registration['run_id']
            and release['instance_id'] == registration['instance_id']
            and release['plan_sha256'] == registration['plan']['sha256'], 'GPU release ownership differs')
    for stage in [completion['runtime_unit'], *completion['stages']]:
        lifecycle.verify_gpu_observation(stage['gpu'], plan, release['gpu'])
    return accepted


def validate_ledger(root, state, registration, records, object_identity=None):
    directory = state / 'collection-recovery-requests'
    actual = sorted(directory.glob('*.json'))
    require(records == [pin(root, path) for path in actual] and 0 < len(records) <= MAX_REQUESTS,
            'Recovery request ledger is incomplete')
    rows = [load(bound(root, record)) for record in records]
    require([row['sequence'] for row in rows] == list(range(1, len(rows) + 1)),
            'Recovery request ledger sequence differs')
    total = 0
    expected = [('s3api', 'head-object', 1024**2)]
    for start in range(0, registration['export']['bytes'], 8 * 1024**2):
        expected.append(('s3api', 'get-object',
                         min(8 * 1024**2, registration['export']['bytes'] - start)))
    expected.extend([('ec2', 'describe-instances', 1024**2),
                     ('ec2', 'describe-volumes', 1024**2)])
    require(len(rows) == len(expected), 'Recovery request sequence is incomplete')
    offset = 0
    for row, (service, operation, size) in zip(rows, expected):
        require(set(row) == {'sequence', 'service', 'operation', 'arguments', 'reserved_bytes', 'reserved_at'}
                and registration['registered_at'] <= row['reserved_at'] < registration['deadline']
                and type(row['reserved_bytes']) is int and row['reserved_bytes'] == size
                and (row['service'], row['operation']) == (service, operation),
                'Recovery request reservation differs')
        args = row['arguments']
        if service == 's3api':
            prefix = ['--bucket', registration['bucket'], '--key', registration['key']]
            if operation == 'head-object':
                require(args == prefix, 'Recovery HEAD request differs')
            else:
                require(object_identity is not None and object_identity.get('ETag'),
                        'Recovery object identity missing')
                version = (['--version-id', object_identity['VersionId']]
                           if object_identity.get('VersionId') else [])
                require(args[:-1] == prefix + ['--range', f'bytes={offset}-{offset + size - 1}',
                        '--if-match', object_identity['ETag']] + version,
                        'Recovery range/object request differs')
                offset += size
        else:
            expected_args = (['--instance-ids', registration['instance_id']]
                             if operation == 'describe-instances'
                             else ['--volume-ids', registration['root_volume_ids'][0]])
            require(args == expected_args, 'Recovery cleanup request differs')
        total += row['reserved_bytes']
    require(total <= PRIOR_BYTES and PRIOR_BYTES + total <= 1024**3
            and PRIOR_REQUESTS + len(rows) <= registration['cumulative_request_limit'],
            'Aggregate original plus recovery allowance exceeded')
    return dict(prior_retrieval_bytes=PRIOR_BYTES, new_reserved_bytes=total,
                cumulative_reserved_bytes=PRIOR_BYTES + total, prior_requests=PRIOR_REQUESTS,
                new_reserved_requests=len(rows), cumulative_reserved_requests=PRIOR_REQUESTS + len(rows))


def validate_collection_recovery(root, output, plan, plan_sha):
    root, output = Path(root).resolve(), Path(output).resolve()
    state = output.parent
    require(output.name == 'recovered-fresh-export', 'Recovery export location differs')
    if (state / 'collection-finalization.json').exists():
        return validate_collection_finalization(root, output, plan, plan_sha)
    receipt_path = state / 'collection-recovery.json'
    receipt = load(receipt_path)
    registration_path = bound(root, receipt['registration'])
    require(os.path.samefile(registration_path.parent, state),
            'Recovery registration belongs to a different state directory')
    require(set(receipt) == {'schema', 'state', 'collection_only', 'additional_gpu_executions',
            'quality_accepted', 'cloud_evaluation', 'registration', 'originals', 'sources',
            'archive', 'candidate', 'object_identity', 'validation', 'cleanup', 'requests',
            'accounting', 'accepted_at'}
            and not (state / 'collection-recovery-failure.json').exists(),
            'Recovery receipt has failure or unsupported fields')
    registration, original_plan = validate_registration(root, registration_path)
    require(original_plan == plan and registration['plan']['sha256'] == plan_sha,
            'Recovery consumer plan differs')
    require(receipt['schema'] == SCHEMA and receipt['state'] == 'accepted'
            and receipt['collection_only'] is True and receipt['quality_accepted'] is False
            and receipt['cloud_evaluation'] is False and receipt['additional_gpu_executions'] == 0
            and receipt['registration'] == pin(root, registration_path)
            and receipt['originals'] == registration['originals']
            and receipt['sources'] == registration['sources']
            and receipt['archive'] == pin(root, state / 'recovered-export.tar')
            and receipt['candidate'] == pin(root, output / 'candidate.json')
            and registration['registered_at'] <= receipt['accepted_at'] < registration['deadline'],
            'Accepted recovery receipt binding differs')
    validate_cleanup(receipt['cleanup'], registration)
    require(all(row['observed_at'] <= receipt['accepted_at'] for row in receipt['cleanup'])
            and receipt['object_identity']['ContentLength'] == registration['export']['bytes']
            and set(receipt['object_identity']) <= {'ContentLength', 'ETag', 'VersionId'},
            'Recovery object or cleanup acceptance timing differs')
    require(receipt['accounting'] == validate_ledger(root, state, registration, receipt['requests'],
                                                    receipt['object_identity']),
            'Recovery accounting receipt differs')
    require(receipt['validation'] == validate_export(root, state, plan, registration),
            'Recovery full export validation differs')
    return dict(collection_recovery=pin(root, receipt_path),
                recovery_registration=pin(root, registration_path),
                controller_result=registration['originals']['controller-result.json'],
                remote_completion=registration['originals']['completion.json'],
                original_bundle=registration['originals']['source-bundle.tar'],
                recovered_archive=receipt['archive'])


def recover(root, registration_path):
    root, registration_path = Path(root).resolve(), Path(registration_path).resolve()
    registration, plan = validate_registration(root, registration_path, active=True)
    state = registration_path.parent
    create_once(state / 'collection-recovery-started.json',
                dict(registration=pin(root, registration_path), started_at=time.time()))
    try:
        client = RecoveryClient(plan, registration, state / 'collection-recovery-requests')
        object_identity = lifecycle.download_export_ranges(client, plan['identity'],
            registration['export'], state / 'recovered-export.tar')
        validation = validate_export(root, state, plan, registration, extract=True)
        cleanup = observe_cleanup(client, registration)
        requests = [pin(root, path) for path in sorted(client.directory.glob('*.json'))]
        accounting = validate_ledger(root, state, registration, requests, object_identity)
        lifecycle.remaining(registration['deadline'])
        # Recheck the registered evidence and executable closure immediately before acceptance.
        validate_registration(root, registration_path, active=True)
        receipt = dict(schema=SCHEMA, state='accepted', collection_only=True,
            additional_gpu_executions=0, quality_accepted=False, cloud_evaluation=False,
            registration=pin(root, registration_path), originals=registration['originals'],
            sources=registration['sources'], archive=pin(root, state / 'recovered-export.tar'),
            candidate=pin(root, state / 'recovered-fresh-export/candidate.json'),
            object_identity=object_identity, validation=validation, cleanup=cleanup,
            requests=requests, accounting=accounting)
        accepted_at = time.time()
        require(registration['registered_at'] <= accepted_at < registration['deadline'],
                'Recovery deadline exhausted before acceptance')
        receipt['accepted_at'] = accepted_at
        create_once(state / 'collection-recovery.json', receipt)
        return receipt
    except Exception as error:
        create_once(state / 'collection-recovery-failure.json',
                    dict(state='failed', error=f'{type(error).__name__}: {error}', failed_at=time.time(),
                         registration=pin(root, registration_path), additional_gpu_executions=0))
        raise


FINALIZATION_SCHEMA = 'trid-fresh-eight-collection-finalization-v1'
FINALIZATION_REGISTRATION_SCHEMA = 'trid-fresh-eight-collection-finalization-registration-v1'


def prior_object_identity(root, request_pins, export):
    rows = [load(bound(root, record)) for record in request_pins]
    gets = [row for row in rows if row['operation'] == 'get-object']
    require(gets, 'Predecessor has no export retrieval reservations')
    args = gets[0]['arguments']
    identity = dict(ETag=args[args.index('--if-match') + 1], ContentLength=export['bytes'])
    if '--version-id' in args:
        identity['VersionId'] = args[args.index('--version-id') + 1]
    return identity


def finalization_spec(root, state, prior_registration_path, source_archive_path, authorization_path, deadline):
    """Parent registers metadata-only finalization of one already retrieved archive."""
    root, state = Path(root).resolve(), Path(state).resolve()
    prior = load(prior_registration_path)
    requests = [pin(root, path) for path in sorted((state / 'collection-recovery-requests').glob('*.json'))]
    identity = prior_object_identity(root, requests, prior['export'])
    accounting = validate_ledger(root, state, prior, requests, identity)
    return dict(schema=FINALIZATION_REGISTRATION_SCHEMA, collection_only=True,
        additional_gpu_executions=0, registered_at=time.time(), deadline=deadline,
        authorization=pin(root, authorization_path), account=load(bound(root, prior['plan']))['identity']['account'],
        prior_registration=pin(root, prior_registration_path),
        prior_started=pin(root, state / 'collection-recovery-started.json'),
        prior_failure=pin(root, state / 'collection-recovery-failure.json'),
        prior_requests=requests, prior_accounting=accounting, object_identity=identity,
        source_archive=pin(root, source_archive_path), sources=[pin(root, root / name) for name in SOURCES],
        archive=pin(root, state / 'recovered-export.tar'),
        candidate=pin(root, state / 'recovered-fresh-export/candidate.json'),
        plan=prior['plan'], originals=prior['originals'],
        instance_id=prior['instance_id'], root_volume_ids=prior['root_volume_ids'],
        new_request_limit=3, new_retrieval_byte_limit=3 * 1024**2)


def validate_prior_cleanup(cleanup, registration):
    require(cleanup.get('cleanup_state') == 'complete'
            and cleanup.get('instance_id') == registration['instance_id']
            and cleanup.get('instance_terminated') is True
            and cleanup.get('root_volume_ids') == registration['root_volume_ids']
            and cleanup.get('absent_volume_ids') == registration['root_volume_ids']
            and cleanup.get('recorded_volume_ids') == registration['root_volume_ids']
            and cleanup.get('remaining_volumes') == [],
            'Historical exact instance/root positive cleanup proof missing')


def validate_finalization_registration(root, registration_path, *, active=False):
    root, registration_path = Path(root).resolve(), Path(registration_path).resolve()
    r, state = load(registration_path), registration_path.parent
    require(r.get('schema') == FINALIZATION_REGISTRATION_SCHEMA
            and r.get('collection_only') is True and r.get('additional_gpu_executions') == 0,
            'Explicit metadata-only finalization registration missing')
    prior_path = bound(root, r['prior_registration'])
    require(os.path.samefile(prior_path.parent, state), 'Finalization predecessor state differs')
    prior, plan = validate_registration(root, prior_path, source_archive=r['source_archive'])
    authorization = load(bound(root, r['authorization']))
    require(authorization.get('schema') == 'trid-fresh-eight-collection-finalization-authorization-v1'
            and authorization.get('collection_only') is True and authorization.get('additional_gpu_executions') == 0
            and all(authorization.get(key) == r[key] for key in
                    ('prior_registration', 'prior_failure', 'source_archive', 'archive'))
            and authorization.get('context') == load(bound(root, prior['authorization']))['context']
            and authorization.get('max_metadata_requests') == 3
            and authorization.get('max_metadata_bytes') == 3 * 1024**2,
            'Parent finalization permission differs')
    bound(root, authorization['context'])
    if not active:
        custody_path = state / 'collection-finalization-source-archive.json'
        if custody_path.exists():
            pin(root, custody_path)
            custody = load(custody_path)
            require(set(custody) == {'schema', 'finalization_registration', 'source_archive'}
                    and custody['schema'] == 'trid-finalization-source-archive-v1'
                    and custody['finalization_registration'] == pin(root, registration_path),
                    'Historical finalization source custody differs')
            authenticate_recovery_sources(root, custody['source_archive'], r['sources'])
    expected = finalization_spec(root, state, prior_path, bound(root, r['source_archive']),
                                 bound(root, r['authorization']), r['deadline'])
    expected['registered_at'] = r['registered_at']
    if not active and custody_path.exists():
        expected['sources'] = r['sources']
    require(r == expected, 'Finalization registration or predecessor evidence differs')
    started, failure = load(bound(root, r['prior_started'])), load(bound(root, r['prior_failure']))
    require(started['registration'] == r['prior_registration']
            and failure['registration'] == r['prior_registration']
            and failure['state'] == 'failed' and failure['additional_gpu_executions'] == 0
            and failure.get('error')
            and prior['registered_at'] <= started['started_at'] <= failure['failed_at']
            and failure['failed_at'] <= r['registered_at']
            and not (state / 'collection-recovery.json').exists(),
            'Finalization requires one preserved failed collection attempt')
    require(r['archive']['bytes'] == prior['export']['bytes']
            and r['archive']['sha256'] == prior['export']['sha256']
            and r['new_request_limit'] == 3 and r['new_retrieval_byte_limit'] == 3 * 1024**2
            and r['prior_accounting']['new_reserved_bytes'] + r['new_retrieval_byte_limit'] <= PRIOR_BYTES
            and r['prior_accounting']['cumulative_reserved_requests'] + 3 <= prior['cumulative_request_limit']
            and 0 < r['deadline'] - r['registered_at'] <= 3600,
            'Finalization would exceed original cumulative allowance or deadline')
    prior_cleanup = load(bound(root, r['originals']['controller-result.json']))['cleanup']
    validate_prior_cleanup(prior_cleanup, r)
    if active:
        require(r['registered_at'] <= time.time() < r['deadline'], 'Finalization deadline expired')
        for name, module_path in ((SOURCES[0], __file__), (SOURCES[1], lifecycle.__file__),
                                  (SOURCES[2], fresh.__file__), (SOURCES[3], smoke.__file__),
                                  (SOURCES[4], smoke.CPU.__file__)):
            validate_executing_module(root, name, module_path)
    return r, prior, plan, prior_cleanup


class FinalizationClient(lifecycle.Client):
    """Account identity and two exact EC2 reads; no S3 transport exists here."""
    def __init__(self, plan, registration, directory):
        super().__init__(plan['identity'], deadline=registration['deadline'], request_limit=3)
        self.registration, self.directory = registration, Path(directory)
        self.directory.mkdir()
        self.reservations = []

    def __call__(self, service, operation, *args, **kwargs):
        r = self.registration
        lifecycle.remaining(r['deadline'])
        allowed = [('sts', 'get-caller-identity', ()),
                   ('ec2', 'describe-instances', ('--instance-ids', r['instance_id'])),
                   ('ec2', 'describe-volumes', ('--volume-ids', r['root_volume_ids'][0]))]
        index = len(self.reservations)
        require(not kwargs and index < 3 and (service, operation, args) == allowed[index],
                'Finalization permits only registered STS and exact EC2 reads')
        reserved = (index + 1) * 1024**2
        require(reserved <= r['new_retrieval_byte_limit']
                and r['prior_accounting']['new_reserved_bytes'] + reserved <= PRIOR_BYTES,
                'Finalization aggregate byte allowance exhausted')
        row = dict(sequence=index + 1, service=service, operation=operation,
                   arguments=list(args), reserved_bytes=1024**2, reserved_at=time.time())
        create_once(self.directory / f'{index + 1:04d}.json', row)
        self.reservations.append(row)
        return super().__call__(service, operation, *args)


def validate_finalization_cleanup(observations, registration, prior_cleanup):
    validate_prior_cleanup(prior_cleanup, registration)
    require(isinstance(observations, list) and len(observations) == 2, 'Finalization cleanup proof missing')
    for row, operation, identifier in zip(observations, ('describe-instances', 'describe-volumes'),
            (registration['instance_id'], registration['root_volume_ids'][0])):
        require(row['operation'] == operation and row['identifier'] == identifier
                and registration['registered_at'] <= row['started_at'] <= row['observed_at'] < registration['deadline'],
                'Fresh finalization observation identity/time differs')
        response = row['response']
        require(isinstance(response, dict), 'Fresh finalization response is missing')
        if 'error_code' in response:
            expected_code = ('InvalidInstanceID.NotFound' if operation == 'describe-instances'
                             else 'InvalidVolume.NotFound')
            require(response['error_code'] == expected_code and set(response) <= {'error_code', 'error'},
                    'Fresh exact cleanup lookup failed without typed absence')
            continue
        require(not any(key in response for key in ('error', 'error_type')),
                'Fresh exact cleanup lookup failed')
        response_key = 'Reservations' if operation == 'describe-instances' else 'Volumes'
        require(set(response) <= {response_key, 'ResponseMetadata'},
                'Fresh exact cleanup response contains unrelated or incomplete fields')
        if operation == 'describe-instances':
            reservations = response.get('Reservations')
            require(isinstance(reservations, list), 'Exact instance response is incomplete')
            if reservations == []:
                continue
            instances = [instance for reservation in reservations for instance in reservation['Instances']]
            require(len(instances) == 1 and instances[0]['InstanceId'] == identifier
                    and instances[0]['State']['Name'] == 'terminated',
                    'Fresh exact instance response is not terminated')
        else:
            require(isinstance(response.get('Volumes'), list) and response['Volumes'] == [],
                    'Fresh exact root-volume response is not absent')


def validate_finalization_caller(row, registration):
    require(row['operation'] == 'get-caller-identity'
            and row['identifier'] == registration['account']
            and registration['registered_at'] <= row['started_at'] <= row['observed_at'] < registration['deadline']
            and isinstance(row['response'], dict)
            and row['response'].get('Account') == registration['account'],
            'Fresh finalization account identity differs')


def observe_finalization_cleanup(client, registration, prior_cleanup):
    started = time.time()
    response = None
    try:
        response = client('sts', 'get-caller-identity')
    except Exception as error:
        response = dict(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        caller = dict(operation='get-caller-identity', identifier=registration['account'],
                      started_at=started, observed_at=time.time(), response=response)
        create_once(client.directory.parent / 'collection-finalization-observation-0.json', caller)
    validate_finalization_caller(caller, registration)
    rows = []
    for index, (operation, option, identifier) in enumerate((
            ('describe-instances', '--instance-ids', registration['instance_id']),
            ('describe-volumes', '--volume-ids', registration['root_volume_ids'][0])), 1):
        started = time.time()
        response = None
        try:
            response = client('ec2', operation, option, identifier)
        except lifecycle.AWSCallError as error:
            response = dict(error_code=error.code, error=error.stderr)
            if error.code not in ('InvalidInstanceID.NotFound', 'InvalidVolume.NotFound'):
                raise
        except Exception as error:
            response = dict(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row = dict(operation=operation, identifier=identifier, started_at=started,
                       observed_at=time.time(), response=response)
            create_once(client.directory.parent / f'collection-finalization-observation-{index}.json', row)
            rows.append(row)
    validate_finalization_cleanup(rows, registration, prior_cleanup)
    return rows


def finalization_accounting(root, state, registration, request_pins):
    paths = sorted((state / 'collection-finalization-requests').glob('*.json'))
    require(request_pins == [pin(root, path) for path in paths] and len(paths) == 3,
            'Finalization request ledger differs')
    for index, (record, service, operation, args) in enumerate(zip(request_pins,
            ('sts', 'ec2', 'ec2'), ('get-caller-identity', 'describe-instances', 'describe-volumes'),
            ([], ['--instance-ids', registration['instance_id']],
             ['--volume-ids', registration['root_volume_ids'][0]])), 1):
        row = load(bound(root, record))
        require(set(row) == {'sequence', 'service', 'operation', 'arguments', 'reserved_bytes', 'reserved_at'}
                and row['sequence'] == index and row['service'] == service
                and row['operation'] == operation and row['arguments'] == args
                and row['reserved_bytes'] == 1024**2
                and registration['registered_at'] <= row['reserved_at'] < registration['deadline'],
                'Finalization reservation exceeds read-only allowance')
    prior = registration['prior_accounting']
    return dict(prior_reserved_bytes=prior['cumulative_reserved_bytes'], new_reserved_bytes=3 * 1024**2,
                cumulative_reserved_bytes=prior['cumulative_reserved_bytes'] + 3 * 1024**2,
                prior_reserved_requests=prior['cumulative_reserved_requests'], new_reserved_requests=3,
                cumulative_reserved_requests=prior['cumulative_reserved_requests'] + 3)


def finalize(root, registration_path):
    root, registration_path = Path(root).resolve(), Path(registration_path).resolve()
    r, prior, plan, prior_cleanup = validate_finalization_registration(root, registration_path, active=True)
    state = registration_path.parent
    create_once(state / 'collection-finalization-started.json',
                dict(registration=pin(root, registration_path), started_at=time.time()))
    try:
        validation = validate_export(root, state, plan, prior, extract=False)
        client = FinalizationClient(plan, r, state / 'collection-finalization-requests')
        cleanup = observe_finalization_cleanup(client, r, prior_cleanup)
        requests = [pin(root, path) for path in sorted(client.directory.glob('*.json'))]
        accounting = finalization_accounting(root, state, r, requests)
        validate_finalization_registration(root, registration_path, active=True)
        receipt = dict(schema=FINALIZATION_SCHEMA, state='accepted', collection_only=True,
            additional_gpu_executions=0, quality_accepted=False, cloud_evaluation=False,
            registration=pin(root, registration_path), prior_registration=r['prior_registration'],
            prior_failure=r['prior_failure'], source_archive=r['source_archive'],
            archive=pin(root, state / 'recovered-export.tar'),
            candidate=pin(root, state / 'recovered-fresh-export/candidate.json'),
            validation=validation, cleanup=cleanup,
            observations=[pin(root, state / f'collection-finalization-observation-{i}.json') for i in (0, 1, 2)],
            requests=requests, accounting=accounting)
        accepted_at = time.time()
        require(r['registered_at'] <= accepted_at < r['deadline'], 'Finalization deadline expired before acceptance')
        receipt['accepted_at'] = accepted_at
        create_once(state / 'collection-finalization.json', receipt)
        return receipt
    except Exception as error:
        create_once(state / 'collection-finalization-failure.json',
                    dict(state='failed', error=f'{type(error).__name__}: {error}', failed_at=time.time(),
                         registration=pin(root, registration_path), additional_gpu_executions=0))
        raise


def validate_collection_finalization(root, output, plan, plan_sha):
    state = output.parent
    receipt_path = state / 'collection-finalization.json'
    receipt = load(receipt_path)
    require(set(receipt) == {'schema', 'state', 'collection_only', 'additional_gpu_executions',
            'quality_accepted', 'cloud_evaluation', 'registration', 'prior_registration', 'prior_failure',
            'source_archive', 'archive', 'candidate', 'validation', 'cleanup', 'observations',
            'requests', 'accounting', 'accepted_at'}
            and not (state / 'collection-finalization-failure.json').exists(),
            'Finalization receipt has unsupported fields or its own failure')
    registration_path = bound(root, receipt['registration'])
    require(os.path.samefile(registration_path.parent, state), 'Finalization receipt state differs')
    r, prior, original_plan, prior_cleanup = validate_finalization_registration(root, registration_path)
    require(original_plan == plan and r['plan']['sha256'] == plan_sha
            and receipt['schema'] == FINALIZATION_SCHEMA and receipt['state'] == 'accepted'
            and receipt['collection_only'] is True and receipt['additional_gpu_executions'] == 0
            and receipt['quality_accepted'] is False and receipt['cloud_evaluation'] is False
            and all(receipt[key] == r[key] for key in
                    ('prior_registration', 'prior_failure', 'source_archive', 'archive', 'candidate'))
            and r['registered_at'] <= receipt['accepted_at'] < r['deadline'],
            'Accepted finalization receipt binding differs')
    require(receipt['observations'] == [pin(root, state / f'collection-finalization-observation-{i}.json') for i in (0, 1, 2)]
            and receipt['cleanup'] == [load(bound(root, p)) for p in receipt['observations'][1:]],
            'Finalization durable observations differ')
    caller = load(bound(root, receipt['observations'][0]))
    validate_finalization_caller(caller, r)
    require(caller['observed_at'] <= receipt['accepted_at'], 'Finalization caller observation occurred after acceptance')
    validate_finalization_cleanup(receipt['cleanup'], r, prior_cleanup)
    require(all(row['observed_at'] <= receipt['accepted_at'] for row in receipt['cleanup'])
            and receipt['accounting'] == finalization_accounting(root, state, r, receipt['requests'])
            and receipt['validation'] == validate_export(root, state, plan, prior, extract=False),
            'Finalization timing, accounting, or full export validation differs')
    return dict(collection_recovery=pin(root, receipt_path), collection_finalization=pin(root, receipt_path),
                recovery_registration=pin(root, registration_path), predecessor_registration=r['prior_registration'],
                collection_failure=r['prior_failure'], executed_recovery_sources=r['source_archive'],
                controller_result=prior['originals']['controller-result.json'],
                remote_completion=prior['originals']['completion.json'],
                original_bundle=prior['originals']['source-bundle.tar'], recovered_archive=r['archive'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    require(lifecycle.SHA.fullmatch(args.registration_sha256)
            and lifecycle.digest(args.registration) == args.registration_sha256,
            'Parent registration digest differs')
    if load(args.registration).get('schema') == FINALIZATION_REGISTRATION_SCHEMA:
        receipt = finalize(args.root, args.registration)
    else:
        receipt = recover(args.root, args.registration)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
