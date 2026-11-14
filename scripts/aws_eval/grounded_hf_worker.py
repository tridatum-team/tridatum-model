"""Authenticated root supervisor and separate offline, nonroot native HF stages."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import resource
import runpy
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

# -I excludes the script directory. Only this already-authenticated directory is
# added; no current directory, user site, remote implementation, or identity hack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grounded_hf_lifecycle import (LIMITS, MAX_EXPORT, Client, authenticate_archive,
    canonical, digest, exclusive_intent, object_sha, pins, read_bound, relative,
    remaining, require, save, timestamp, validate_data, validate_cost_admission, validate_bootstrap_admission,
    verify_termination_ticket, verify_release, verify_gpu_observation, gpu_minor_from_xml,
    inference_mode, followup_mode, eight_mode as is_eight)


def eight_module(plan):
    if is_eight(plan):
        import eight_hf_worker
        return eight_hf_worker
    return None


def ordered_module(plan):
    if inference_mode(plan):
        import ordered_hf_comparison
        return ordered_hf_comparison
    return None


class DownloadRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts, budget):
        self.hosts, self.budget = set(hosts), budget

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.hosts)
        self.budget[0] -= 1
        require(self.budget[0] >= 0, 'Download request budget exhausted')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_url(url, hosts):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname in hosts and parsed.port in (None, 443)
            and not parsed.username and not parsed.password, 'Unapproved HTTPS download origin')


def download(row, directory, hosts, deadline, budget):
    """One finite acquisition; no retry, resume, fallback, or hub cache aliases."""
    name = relative(row['path'])
    check_url(row['url'], hosts)
    target = Path(directory) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    budget[0] -= 1
    require(budget[0] >= 0, 'Download request budget exhausted')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), DownloadRedirects(hosts, budget))
    h, size = hashlib.sha256(), 0
    with opener.open(row['url'], timeout=min(30, remaining(deadline))) as response, target.open('xb') as out:
        require(response.status == 200, 'Non-full artifact response')
        length = response.headers.get('Content-Length')
        require(length is None or int(length) == row['bytes'], 'Download declared size differs')
        while True:
            remaining(deadline)
            block = response.read(min(1024 * 1024, row['bytes'] - size + 1))
            if not block:
                break
            size += len(block)
            require(size <= row['bytes'], 'Download overflow')
            out.write(block); h.update(block)
        out.flush(); os.fsync(out.fileno())
    require(size == row['bytes'] and h.hexdigest() == row['sha256'], 'Downloaded artifact authentication failed')
    target.chmod(0o444)
    return target


def command(argv, deadline, *, timeout=90, **kwargs):
    result = subprocess.run(argv, timeout=min(timeout, remaining(deadline)), check=False,
                            capture_output=True, text=True, **kwargs)
    require(result.returncode == 0, 'Command failed: ' + argv[0] + ': ' + result.stderr[:1000])
    return result.stdout


def identity_document(deadline):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = 'http://169.254.169.254/latest/'
    req = urllib.request.Request(endpoint + 'api/token', method='PUT',
                                 headers={'X-aws-ec2-metadata-token-ttl-seconds': '300'})
    with opener.open(req, timeout=min(5, remaining(deadline))) as response:
        token = response.read(4096).decode()
    req = urllib.request.Request(endpoint + 'dynamic/instance-identity/document',
                                 headers={'X-aws-ec2-metadata-token': token})
    with opener.open(req, timeout=min(5, remaining(deadline))) as response:
        return json.loads(response.read(16384))


def gpu_observation(deadline, *, with_usage=False):
    fields = 'uuid,name,memory.total,driver_version' + (',memory.used' if with_usage else '')
    text = command(['nvidia-smi', '--query-gpu=' + fields,
                    '--format=csv,noheader,nounits'], deadline, timeout=5)
    rows = [r.strip().split(', ') for r in text.strip().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == (5 if with_usage else 4), 'Exactly one NVIDIA GPU required')
    uuid, name, mib, driver = rows[0][:4]
    require(mib.isdigit(), 'Physical NVML memory total is unavailable')
    xml = command(['nvidia-smi', '-q', '-x'], deadline, timeout=5)
    minor = gpu_minor_from_xml(xml, uuid)
    paths = [f'/dev/nvidia{minor}', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']
    require(all(not Path(p).is_symlink() and stat.S_ISCHR(Path(p).stat().st_mode) for p in paths),
            'Exact NVIDIA character devices unavailable')
    observed = dict(uuid=uuid, minor=minor, name=name, total_mib=int(mib), driver=driver)
    if with_usage:
        require(rows[0][4].isdigit(), 'Physical NVML memory usage is unavailable')
        return observed, int(rows[0][4]) * 1024**2
    return observed


def stage_argv(plan, base, phase, seconds, gpu, uid, gid):
    ordered = ordered_module(plan)
    eight = eight_module(plan)
    allowed = (eight.phase_kind(phase) is not None if eight else
        ((phase == 'runtime' or ordered.phase_kind(plan, phase) is not None) if ordered else
         phase in ('runtime', 'capacity', 'baseline', 'train', 'adapter')))
    require(allowed and 0 < seconds <= 12600, 'Invalid phase/deadline')
    base = Path(base)
    unit = plan['identity']['run_id'] + '-' + phase
    properties = ['Type=exec', 'RemainAfterExit=yes', 'KillMode=control-group', 'SendSIGKILL=yes', 'TimeoutStopSec=10',
        f'RuntimeMaxSec={seconds}', 'MemoryMax=48G', 'MemorySwapMax=0', 'CPUQuota=600%',
        'TasksMax=128', 'OOMPolicy=kill', 'NoNewPrivileges=yes', 'ProtectSystem=strict',
        'ProtectHome=yes', 'PrivateNetwork=yes', 'RestrictAddressFamilies=AF_UNIX',
        'DevicePolicy=closed', 'PrivateDevices=no', 'RestrictSUIDSGID=yes', 'UMask=0077',
        'LimitCORE=0', f'LimitFSIZE={LIMITS["output_bytes"]}', 'LimitNOFILE=256',
        f'WorkingDirectory={base / "code"}', f'ReadWritePaths={base / "output"}',
        'InaccessiblePaths=/root /home /run/credentials /var/lib/cloud',
        f'StandardOutput=append:{base / "output" / (phase + ".log")}',
        f'StandardError=append:{base / "output" / (phase + ".log")}']
    selector = ordered is not None and ordered.phase_kind(plan, phase) == 'select'
    if ordered:
        hidden = [base / 'model'] if selector else [
            base / 'code' / plan['workload']['development_references']['path']]
        properties.append('InaccessiblePaths=' + ' '.join(str(path) for path in hidden))
    if eight and phase not in ('final_base', 'final_adapter'):
        properties.append(f'InaccessiblePaths=-{base / "final-inputs.json"}')
    if not selector:
        properties += [f'DeviceAllow={path} rw' for path in
                       [f'/dev/nvidia{gpu["minor"]}', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']]
    argv = ['systemd-run', '--quiet', '--unit', unit, '--uid', str(uid), '--gid', str(gid)]
    for prop in properties:
        argv += ['--property', prop]
    for value in [f'CUDA_VISIBLE_DEVICES={"" if selector else gpu["uuid"]}', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                  'HF_DATASETS_OFFLINE=1', 'AWS_EC2_METADATA_DISABLED=true', 'PYTHONDONTWRITEBYTECODE=1',
                  'TOKENIZERS_PARALLELISM=false', 'OMP_NUM_THREADS=6', 'MKL_NUM_THREADS=6',
                  f'HF_HOME={base / "output" / "cache"}', f'TMPDIR={base / "output" / "tmp"}',
                  f'TRID_OWNED_RUN={plan["identity"]["run_id"]}']:
        argv += ['--setenv', value]
    return [*argv, '--', str(base / 'venv/bin/python'), '-I', '-B',
            str(base / 'code/scripts/aws_eval/grounded_hf_worker.py'), phase,
            '--plan', str(base / 'plan.json'), '--plan-sha256', digest(base / 'plan.json'), '--base', str(base)]


def unit_state(unit, deadline):
    result = subprocess.run(['systemctl', 'show', unit,
        '--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,ControlGroup,Environment,InvocationID,MemoryMax,MemorySwapMax,DevicePolicy,User,Group'],
        capture_output=True, text=True, timeout=min(10, remaining(deadline)))
    value = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    require(result.returncode == 0 or value.get('LoadState') == 'not-found', 'Cannot observe owned unit state')
    return value


def assert_owned_unit(state, run):
    require(f'TRID_OWNED_RUN={run}' in state.get('Environment', '').split()
            and state.get('InvocationID'), 'Refusing to act on unowned systemd unit')


def cgroup_empty(state):
    group = state.get('ControlGroup')
    if not group:
        return state.get('ActiveState') in ('inactive', 'failed') or state.get('SubState') == 'exited'
    require(group.startswith('/') and '..' not in Path(group).parts, 'Unsafe cgroup path')
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    if not root.exists():
        return state.get('ActiveState') in ('inactive', 'failed') or state.get('SubState') == 'exited'
    return all(not path.read_text().strip() for path in root.rglob('cgroup.procs'))


def stop_unit(unit, run, deadline):
    state = unit_state(unit, deadline)
    assert_owned_unit(state, run)
    group = state.get('ControlGroup')
    command(['systemctl', 'stop', unit], deadline)
    stopped = unit_state(unit, deadline)
    if stopped.get('LoadState') != 'not-found':
        assert_owned_unit(stopped, run)
    require(stopped.get('LoadState') == 'not-found' or stopped.get('ActiveState') in ('inactive', 'failed'),
            'Owned systemd unit did not stop')
    require(cgroup_empty(dict(ControlGroup=group, ActiveState='inactive')),
            'Owned process group did not stop and reap')
    return stopped


def supervise_phase(plan, base, phase, gpu, uid, gid, deadline, owned_units, *, seconds=None):
    """External GPU/cgroup guard remains root-owned, outside the native child."""
    ordered = ordered_module(plan)
    run = plan['identity']['run_id']
    unit = run + '-' + phase
    old = unit_state(unit, deadline)
    require(old['LoadState'] == 'not-found', 'Stage unit already exists; no reuse')
    if phase == 'runtime':
        allocation = 60
    elif seconds is not None:
        allocation = seconds
    else:
        allocations = plan['workload']['phase_seconds'] if ordered else plan['phase_seconds']
        allocation = allocations[phase]
    seconds = min(allocation, int(remaining(deadline)) - 15)
    require(seconds > 0, 'Insufficient stop reserve')
    verify_gpu_observation(gpu_observation(deadline), plan, gpu)
    # Record ownership before dispatch: a lost systemd-run response is ambiguous.
    exclusive_intent(Path(base) / (phase + '-dispatch.json'), dict(unit=unit, plan_sha256=digest(Path(base) / 'plan.json')))
    owned_units.append(unit)
    command(stage_argv(plan, base, phase, seconds, gpu, uid, gid), deadline)
    phase_deadline = min(deadline, time.time() + seconds)
    try:
        while True:
            remaining(phase_deadline)
            state = unit_state(unit, deadline)
            assert_owned_unit(state, run)
            require(state['MemoryMax'] == str(LIMITS['rss_bytes']) and state['MemorySwapMax'] == '0'
                    and state['DevicePolicy'] == 'closed' and state['User'] == str(uid)
                    and state['Group'] == str(gid), 'Effective native cgroup/user boundary differs')
            if ordered and ordered.phase_kind(plan, phase) == 'inference':
                ordered.check_question_deadline(Path(base) / 'output' / (phase + '-receipt.json'))
            observed, used = gpu_observation(phase_deadline, with_usage=True)
            verify_gpu_observation(observed, plan, gpu)
            require(used <= LIMITS['gpu_bytes'], 'External 42GiB GPU cap exceeded')
            if state['ActiveState'] in ('inactive', 'failed') or state.get('SubState') == 'exited':
                require(cgroup_empty(state) and state['Result'] == 'success' and state['ExecMainStatus'] == '0',
                        'Native stage failed: ' + phase + ': ' + str(state))
                receipt = dict(phase=phase, state='complete', unit=unit, invocation=state['InvocationID'], gpu=observed)
                if followup_mode(plan) or is_eight(plan):
                    receipt.update(run_id=run, plan_sha256=digest(Path(base) / 'plan.json'),
                                   instance_id=json.loads((Path(base) / 'release.json').read_text())['instance_id'])
                return receipt
            time.sleep(0.5)
    finally:
        stop_unit(unit, run, min(deadline + 15, time.time() + 30))
        owned_units.remove(unit)


def export_inventory(root, maximum):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Unsafe output root')
    rows, total = [], 0
    for path in sorted(root.rglob('*')):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), 'Output symlink rejected')
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Output nonregular/hardlink rejected')
        name = relative(path.relative_to(root).as_posix())
        require(not name.startswith(('model/', 'wheels/')) and not name.endswith(('.bin', '.pt', '.pth')),
                'Unapproved base/pickle export')
        total += info.st_size
        require(total <= maximum and len(rows) < 4096, 'Output envelope overflow; no silent truncation')
        rows.append(dict(path=name, bytes=info.st_size, sha256=digest(path)))
    return dict(files=rows, total_bytes=total)


def export_outputs(base, plan, report, deadline):
    base = Path(base)
    inventory = export_inventory(base / 'output', LIMITS['output_bytes'])
    # Exact output membership is registered, not just an extension allowlist.
    permitted = set(plan['output_allowlist'])
    actual = {r['path'] for r in inventory['files']}
    require(actual <= permitted, 'Unregistered output member: ' + str(sorted(actual - permitted)))
    manifest = dict(run_id=plan['identity']['run_id'], plan_sha256=digest(base / 'plan.json'),
                    status=report, **inventory, export_cap_bytes=MAX_EXPORT,
                    total_output_cap_bytes=LIMITS['output_bytes'],
                    excluded=['public base weights', 'discarded synthetic capacity adapter'])
    payload = canonical(manifest)
    # Include archive headers/padding and terminal blocks in the same 2GiB cap.
    predicted = 10240 + sum(512 + ((r['bytes'] + 511) // 512) * 512 for r in inventory['files'])
    predicted += 512 + ((len(payload) + 511) // 512) * 512
    require(predicted <= MAX_EXPORT, 'Export archive overhead exceeds envelope')
    import io
    target = base / 'export.tar'
    with tarfile.open(target, 'x:', format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo('export-manifest.json'); info.size = len(payload); info.mode = 0o444
        archive.addfile(info, io.BytesIO(payload))
        for row in inventory['files']:
            remaining(deadline)
            path = base / 'output' / row['path']
            require(digest(path) == row['sha256'], 'Output changed after processes stopped')
            info = tarfile.TarInfo(row['path']); info.size = row['bytes']; info.mode = 0o444
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    require(target.stat().st_size <= MAX_EXPORT, 'Archive overflow')
    return dict(path=target, bytes=target.stat().st_size, sha256=digest(target), manifest=manifest)


def install_network_cap(plan, deadline, base):
    # An actual packet-byte gate, not a billing alert. Fail if quota extension is
    # unavailable. All host outbound traffic shares the conservative allowance.
    base = Path(base)
    owner = base / 'bootstrap-network-owner'
    info = owner.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0
            and owner.read_text() == plan['identity']['run_id'] + '\n' + digest(base / 'plan.json') + '\n',
            'Bootstrap network quota ownership differs')
    command(['nft', 'list', 'table', 'inet', 'trid_bootstrap'], deadline)
    chain = 'TRID_' + hashlib.sha256(plan['identity']['run_id'].encode()).hexdigest()[:12]
    command(['iptables', '-N', chain], deadline)
    command(['iptables', '-A', chain, '-m', 'quota', '--quota',
             str(plan['network']['egress_byte_cap']), '-j', 'RETURN'], deadline)
    command(['iptables', '-A', chain, '-j', 'REJECT'], deadline)
    command(['iptables', '-I', 'OUTPUT', '1', '!', '-d', '127.0.0.0/8', '-j', chain], deadline)
    # IPv6 must not bypass the registered IPv4/egress model.
    command(['ip6tables', '-P', 'OUTPUT', 'DROP'], deadline)
    # Both full-family gates must be active before releasing the smaller
    # bootstrap quota. Any earlier failure leaves that quota in force.
    command(['nft', 'delete', 'table', 'inet', 'trid_bootstrap'], deadline)


def extract_bootstrap_member(archive_path, pin, target):
    """Extract only the authenticated helper, never install a .deb or run maintainer scripts."""
    require(Path(archive_path).stat().st_size <= pin['max_unpacked_bytes'], 'Bootstrap tar exceeds bound')
    seen, selected, total = set(), None, 0
    with tarfile.open(archive_path, 'r:') as archive:
        for member in archive:
            name = member.name[2:] if member.name.startswith('./') else member.name
            if name in ('', '.') and member.isdir():
                continue
            relative(name)
            require(name not in seen and len(seen) < 256 and not member.pax_headers and not member.sparse
                    and (member.isfile() or member.isdir()), 'Unsafe bootstrap archive member')
            seen.add(name); total += member.size
            require(total <= pin['max_unpacked_bytes'], 'Bootstrap expanded-byte overflow')
            if name == pin['member']['path']:
                require(member.isfile() and member.size == pin['member']['bytes'], 'Bootstrap helper size differs')
                with archive.extractfile(member) as stream:
                    require(hashlib.file_digest(stream, 'sha256').hexdigest() == pin['member']['sha256'],
                            'Bootstrap helper hash differs')
                selected = member
        require(selected is not None, 'Pinned bootstrap helper missing')
        with archive.extractfile(selected) as source, Path(target).open('xb') as sink:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                sink.write(block)
    Path(target).chmod(0o555)
    return Path(target)


def observe_prerequisites(admission, deadline):
    """Convert declared path/capability expectations into actual host observations."""
    observed = {}
    for name, expectation in admission['expected_commands'].items():
        found = shutil.which(name)
        require(found == expectation['path'] and Path(found).is_absolute(),
                'Expected base-command path differs: ' + name)
        resolved = Path(found).resolve()
        checksum = digest(resolved)
        require('sha256' not in expectation or checksum == expectation['sha256'],
                'Expected base-command hash differs: ' + name)
        stdout = command([found, *expectation['probe_args']], deadline, timeout=10)
        require(len(stdout.encode()) <= 65536, 'Base-command capability receipt exceeds bound')
        observed[name] = dict(found_path=found, resolved_path=str(resolved), sha256=checksum,
                              probe_args=expectation['probe_args'], stdout=stdout)
    command([sys.executable, '-I', '-B', '-c', 'import venv,ensurepip'], deadline, timeout=10)
    venv_version = command(['dpkg-query', '-W', '-f=${Version}', 'python3.12-venv'], deadline, timeout=10).strip()
    require(venv_version == admission['venv_package_version'], 'Preinstalled same-AMI venv package changed')
    aws_version = observed['aws']['stdout'].split()[0]
    require(aws_version == 'aws-cli/' + admission['aws_cli_version'], 'Preinstalled same-AMI AWS CLI changed')
    python_version = platform.python_version()
    require(python_version == admission['python_version']
            and observed['python3']['stdout'].strip() == 'Python ' + python_version,
            'Expected base Python interpreter differs')
    return dict(prerequisites_verified=True, commands=observed, python_version=python_version,
                venv_package_version=venv_version, aws_cli_version=aws_version.removeprefix('aws-cli/'))


def prepare_bootstrap_dependencies(plan, base, admission, deadline, budget, units):
    base = Path(base)
    prerequisites = observe_prerequisites(admission, deadline)
    directory = base / 'bootstrap-artifacts'; directory.mkdir(mode=0o700)
    hosts = set(plan['runtime']['download_hosts']) | set(plan['network']['download_hosts'])
    deb = plan['runtime']['bootstrap']['nvidia_modprobe']
    package = download(deb, directory, hosts, deadline, budget)
    fields = command(['dpkg-deb', '--field', str(package), 'Package', 'Version', 'Architecture'],
                     deadline, timeout=10)
    actual = dict(line.split(': ', 1) for line in fields.splitlines() if ': ' in line)
    require(actual == {'Package': deb['package'], 'Version': deb['version'], 'Architecture': deb['architecture']},
            'Pinned historical NVIDIA package identity differs')
    unit = plan['identity']['run_id'] + '-bootstrap-deb'
    require(unit_state(unit, deadline)['LoadState'] == 'not-found', 'Bootstrap unit already exists')
    exclusive_intent(base / 'bootstrap-deb-dispatch.json', dict(unit=unit, sha256=deb['sha256']))
    archive_path = directory / 'data.tar'
    archive_path.touch(mode=0o600, exist_ok=False)
    units.append(unit)
    argv = ['systemd-run', '--quiet', '--unit', unit,
            '--setenv', 'TRID_OWNED_RUN=' + plan['identity']['run_id']]
    for prop in ('Type=exec', 'RemainAfterExit=yes', 'KillMode=control-group', 'SendSIGKILL=yes',
                 'RuntimeMaxSec=30', 'TimeoutStopSec=5', 'MemoryMax=256M', 'MemorySwapMax=0',
                 'CPUQuota=100%', 'PrivateNetwork=yes', 'NoNewPrivileges=yes',
                 f'LimitFSIZE={deb["max_unpacked_bytes"]}', f'StandardOutput=file:{archive_path}'):
        argv += ['--property', prop]
    # systemd owns/reaps the decompressor and all children even if the CLI
    # response is lost. No dpkg installation/apt resolver is invoked.
    command([*argv, '--', 'dpkg-deb', '--fsys-tarfile', str(package)], deadline, timeout=10)
    unpack_deadline = min(deadline, time.time() + 35)
    while True:
        remaining(unpack_deadline)
        state = unit_state(unit, deadline)
        assert_owned_unit(state, plan['identity']['run_id'])
        if state.get('SubState') == 'exited' or state.get('ActiveState') == 'failed':
            break
        time.sleep(0.1)
    require(state['Result'] == 'success' and state['ExecMainStatus'] == '0', 'Pinned deb decompression failed')
    stop_unit(unit, plan['identity']['run_id'], deadline)
    units.remove(unit)
    helper = extract_bootstrap_member(archive_path, deb, directory / 'nvidia-modprobe')
    archive_path.unlink()
    command([str(helper), '-u'], deadline, timeout=15)
    return prerequisites


def install_pinned_runtime(plan, base, deadline):
    base = Path(base)
    command([sys.executable, '-I', '-B', '-m', 'venv', str(base / 'venv')], deadline)
    prefix = [str(base / 'venv/bin/python'), '-I', '-B', '-m', 'pip']
    command([*prefix, 'install', '--no-index', '--no-deps', '--no-cache-dir', '--disable-pip-version-check',
             *[str(base / 'wheels' / row['path']) for row in plan['runtime']['wheels']]], deadline,
            env={'PATH': '/usr/bin:/bin', 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})


def bootstrap(plan, plan_sha, base):
    require(os.geteuid() == 0 and platform.machine() == 'x86_64', 'Root x86_64 supervisor required')
    ordered = ordered_module(plan)
    eight = eight_module(plan)
    base = Path(base)
    require(base == Path('/opt/trid-' + plan['identity']['run_id']) and not base.is_symlink(), 'Owned bootstrap root differs')
    exclusive_intent(base / 'bootstrap-dispatch.json', dict(plan_sha256=plan_sha))
    initial_deadline = time.time() + 180
    identity = identity_document(initial_deadline)
    ident = plan['identity']
    require(identity['accountId'] == ident['account'] and identity['region'] == ident['region']
            and identity['instanceType'] == plan['instance_type'] and identity['imageId'] == plan['ami']['ImageId']
            and identity['availabilityZone'] == plan['network']['availability_zone'], 'IMDS instance identity mismatch')
    # Pending time is only an early, conservative local timer; release replaces
    # it with the controller's observed EC2 LaunchTime deadline.
    deadline = timestamp(identity['pendingTime']) + 14400
    require(time.time() < deadline - 1800, 'Bootstrap already outside workload window')
    shutdown_at = dt.datetime.fromtimestamp(deadline - 60).strftime('%H:%M')
    command(['/sbin/shutdown', '-c'], initial_deadline)
    command(['/sbin/shutdown', '-h', shutdown_at], initial_deadline)
    success_key = 'experimental_success' if eight else ('inference_success' if ordered else 'training_success')
    report = {success_key: False}
    report.update(stages=[], run_id=ident['run_id'],
                  instance_id=identity['instanceId'], plan_sha256=plan_sha)
    units, export, client = [], None, Client(ident, deadline=deadline - 30, request_limit=plan['cost']['get_requests'] + 8)
    try:
        require(platform.python_version() == plan['runtime']['python_version'], 'Bootstrap Python pin differs')
        authenticate_archive(base / 'bundle.tar', plan['bundle'])
        validate_data(base / 'code', plan)
        validate_cost_admission(base / 'code', plan)
        bootstrap_admission = validate_bootstrap_admission(base / 'code', plan)
        install_network_cap(plan, initial_deadline, base)
        ticket = None
        if eight:
            path = eight.await_object(client, ident['bucket'], ident['code_prefix'] + 'termination.json',
                base / 'termination.json', min(deadline - 1800, timestamp(identity['pendingTime']) + 600),
                1024**2, max_polls=120)
            ticket = json.loads(path.read_text())
        else:
            for attempt in range(120):
                require(time.time() < min(deadline - 1800, timestamp(identity['pendingTime']) + 600),
                        'Independent termination ticket deadline exhausted')
                try:
                    client('s3api', 'get-object', '--bucket', ident['bucket'], '--key', ident['code_prefix'] + 'termination.json',
                           '--range', 'bytes=0-1048575', str(base / 'termination.json'))
                    ticket = json.loads((base / 'termination.json').read_text())
                    break
                except Exception:
                    time.sleep(5)
        require(ticket is not None, 'No independently verified termination ticket; no bootstrap downloads')
        deadline = verify_termination_ticket(ticket, plan, plan_sha, identity['instanceId'], time.time())
        client.deadline = deadline - 30
        acquire_phase = 'acquire14' if followup_mode(plan) else ('acquire8' if ordered else 'acquire')
        allocations = plan['workload']['phase_seconds'] if ordered else plan['phase_seconds']
        acquisition_deadline = min(deadline - 1800, time.time() + allocations[acquire_phase])
        budget = [plan['cost']['download_requests']]
        prerequisites = prepare_bootstrap_dependencies(plan, base, bootstrap_admission, acquisition_deadline, budget, units)
        gpu = gpu_observation(acquisition_deadline)
        verify_gpu_observation(gpu, plan)
        ready = dict(run_id=ident['run_id'], plan_sha256=plan_sha, instance_id=identity['instanceId'],
                     gpu=gpu, prerequisites=prerequisites)
        save(base / 'ready.json', ready)
        require((base / 'ready.json').stat().st_size <= 1024**2, 'Readiness receipt exceeds reserved envelope')
        client('s3api', 'put-object', '--bucket', ident['bucket'], '--key', ident['output_prefix'] + 'ready.json',
               '--body', str(base / 'ready.json'), '--if-none-match', '*', '--server-side-encryption', 'AES256')
        release = None
        if eight:
            path = eight.await_object(client, ident['bucket'], ident['code_prefix'] + 'release.json',
                base / 'release.json', min(deadline - 1800, timestamp(identity['pendingTime']) + 900),
                1024**2, max_polls=180)
            release = json.loads(path.read_text())
        else:
            for attempt in range(120):
                require(time.time() < min(deadline - 1800, timestamp(identity['pendingTime']) + 900),
                        'Release wait deadline exhausted')
                try:
                    client('s3api', 'get-object', '--bucket', ident['bucket'], '--key', ident['code_prefix'] + 'release.json',
                           '--range', 'bytes=0-1048575', str(base / 'release.json'))
                    release = json.loads((base / 'release.json').read_text())
                    break
                except Exception:
                    time.sleep(5)
        require(release is not None, 'Independent verified termination release missing')
        work_deadline = verify_release(release, plan, plan_sha, identity['instanceId'], time.time())
        verify_gpu_observation(gpu, plan, release['gpu'])
        verify_gpu_observation(gpu_observation(work_deadline), plan, release['gpu'])
        deadline = release['deadline']
        client.deadline = deadline - 30
        command(['/sbin/shutdown', '-c'], deadline)
        command(['/sbin/shutdown', '-h', dt.datetime.fromtimestamp(deadline - 60).strftime('%H:%M')], deadline)
        exclusive_intent(base / 'model-acquisition-dispatch.json', dict(release_sha256=digest(base / 'release.json')))
        acquisition_deadline = min(work_deadline, acquisition_deadline)
        (base / 'model').mkdir(mode=0o755); (base / 'wheels').mkdir(mode=0o755)
        hosts = set(plan['network']['download_hosts']) | set(plan['runtime']['download_hosts'])
        for row in plan['runtime']['wheels']:
            download(row, base / 'wheels', hosts, acquisition_deadline, budget)
        install_pinned_runtime(plan, base, acquisition_deadline)
        user = 'trid-' + hashlib.sha256(ident['run_id'].encode()).hexdigest()[:12]
        command(['useradd', '--system', '--no-create-home', '--shell', '/usr/sbin/nologin', user], acquisition_deadline)
        account = pwd.getpwnam(user)
        require(account.pw_uid > 0 and account.pw_gid > 0, 'Dedicated native identity must not be root')
        base.chmod(0o755); (base / 'plan.json').chmod(0o444); (base / 'release.json').chmod(0o444)
        # New runtime imports and CUDA initialization belong only to the
        # nonroot sandbox. Make authenticated files readable before its probe.
        for tree in (base / 'code', base / 'venv', base / 'model'):
            tree.chmod(0o755)
            for path in tree.rglob('*'):
                if path.is_symlink():
                    require(tree.name == 'venv', 'Unexpected authenticated tree symlink')
                    continue
                if path.is_dir():
                    path.chmod(0o755)
                else:
                    path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        (base / 'output').mkdir(mode=0o700)
        command(['mount', '-t', 'tmpfs', '-o', 'size=2G,nr_inodes=4096,nosuid,nodev,noexec,mode=0700',
                 'trid-output', str(base / 'output')], acquisition_deadline)
        os.chown(base / 'output', account.pw_uid, account.pw_gid)
        for name in ('tmp', 'cache'):
            (base / 'output' / name).mkdir()
            os.chown(base / 'output' / name, account.pw_uid, account.pw_gid)
        report['runtime_unit'] = supervise_phase(plan, base, 'runtime', gpu, account.pw_uid, account.pw_gid,
                                                 acquisition_deadline, units)
        report['runtime'] = json.loads((base / 'output/runtime.json').read_text())
        if ordered:
            ordered.run_ordered(sys.modules[__name__], plan, base, report, client, gpu, account,
                                work_deadline, acquisition_deadline, budget, hosts, units)
        elif eight:
            eight.run_experiment(sys.modules[__name__], plan, base, report, client, gpu, account,
                                 work_deadline, acquisition_deadline, budget, hosts, units)
        else:
            for row in plan['model']['files']:
                download(row, base / 'model', hosts, acquisition_deadline, budget)
            validate_model(base / 'model', plan)
            for path in (base / 'model').iterdir():
                path.chmod(0o444)
            for phase in ('capacity', 'baseline', 'train', 'adapter'):
                report['stages'].append(supervise_phase(plan, base, phase, gpu, account.pw_uid, account.pw_gid, work_deadline, units))
                if phase == 'capacity':
                    # Never inherit the synthetic adapter, optimizer, or RNG process.
                    shutil.rmtree(base / 'output' / 'capacity')
                    receipt_path = base / 'output/capacity-receipt.json'
                    receipt = json.loads(receipt_path.read_text())
                    require(not (base / 'output/capacity').exists(), 'Synthetic capacity artifacts remain')
                    receipt['discarded'] = True
                    save(receipt_path, receipt)
        report[success_key] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['stop_errors'] = []
        for unit in units:
            try:
                stop_unit(unit, ident['run_id'], min(deadline - 60, time.time() + 30))
            except Exception as exc:
                report['stop_errors'].append(str(exc))
        if report['stop_errors']:
            report[success_key] = False
        try:
            require(not report['stop_errors'], 'Export forbidden while an owned process remains unresolved')
            if (base / 'output/capacity').exists():
                require(not (base / 'output/capacity').is_symlink(), 'Synthetic scratch ownership differs')
                shutil.rmtree(base / 'output/capacity')
            if (base / 'output').exists():
                # These directories are scratch only; unexpected content is not
                # silently excluded. Every surviving file must be registered.
                export = export_outputs(base, plan, report, deadline - 60)
                client('s3api', 'put-object', '--bucket', ident['bucket'], '--key', ident['output_prefix'] + 'export.tar',
                       '--body', str(export['path']), '--if-none-match', '*', '--server-side-encryption', 'AES256',
                       '--metadata', json.dumps({'sha256': export['sha256'], 'run': ident['run_id']}))
                report['export'] = {k: export[k] for k in ('bytes', 'sha256')}
        except Exception as exc:
            report['export_error'] = str(exc)
        try:
            save(base / 'completion.json', report)
            require((base / 'completion.json').stat().st_size <= 1024**2, 'Completion receipt exceeds reserved envelope')
            client('s3api', 'put-object', '--bucket', ident['bucket'], '--key', ident['output_prefix'] + 'completion.json',
                   '--body', str(base / 'completion.json'), '--if-none-match', '*', '--server-side-encryption', 'AES256')
        finally:
            # Instance-initiated shutdown was admitted as terminate. Parent must
            # still observe TERMINATED + owned EBS absence; this is not proof.
            subprocess.run(['/sbin/shutdown', '-h', 'now'], timeout=10, check=False)
    return report


def validate_model(directory, plan):
    directory = Path(directory)
    expected = pins(plan['model']['files'], plan['model']['max_bytes'])
    require({p.name for p in directory.iterdir()} == set(expected), 'Unexpected model directory membership')
    for name, row in expected.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
                and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Model pin differs')
    config = json.loads((directory / 'config.json').read_text())
    require(config.get('model_type') == 'qwen3' and not config.get('auto_map'), 'Remote/custom model implementation rejected')
    tokenizer = json.loads((directory / 'tokenizer_config.json').read_text())
    require(not tokenizer.get('auto_map'), 'Remote/custom tokenizer implementation rejected')
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    require(shards == {n for n in expected if n.endswith('.safetensors')}, 'Incomplete or extraneous HF weight shards')


def native_runtime(plan, base):
    require(os.geteuid() != 0 and not any(k.startswith('AWS_') and k != 'AWS_EC2_METADATA_DISABLED' for k in os.environ),
            'Native process must be nonroot without cloud credentials')
    require(platform.machine() == 'x86_64' and platform.python_version() == plan['runtime']['python_version'],
            'Native Python/architecture differs')
    release = json.loads((Path(base) / 'release.json').read_text())
    require(os.environ['CUDA_VISIBLE_DEVICES'] == release['gpu']['uuid'], 'Exact GPU UUID allowlist differs')
    observed = gpu_observation(min(release['deadline'] - 1800, time.time() + 15))
    verify_gpu_observation(observed, plan, release['gpu'])
    import torch
    require(torch.__version__ == plan['runtime']['packages']['torch'] and torch.version.cuda == plan['runtime']['torch_cuda']
            and torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            'Actual pinned CUDA/BF16 runtime unavailable')
    gpu = torch.cuda.get_device_properties(0)
    require(gpu.total_memory >= LIMITS['gpu_bytes'] and gpu.name == plan['runtime']['gpu_name'], 'Native GPU shape differs')
    # Torch's allocatable total is a different quantity from NVML's physical
    # total. Use it only to express the unchanged absolute 40GiB allocator cap.
    torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes'] / gpu.total_memory, 0)
    torch.set_num_threads(6)
    torch.manual_seed(14 if ordered_module(plan) else plan['recipe']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return torch


def runtime_stage(plan, base):
    import importlib.metadata
    actual = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}
    expected = {k.lower().replace('_', '-'): v for k, v in plan['runtime']['packages'].items()}
    require(actual == expected, 'Installed package inventory differs (including transitive/bootstrap distributions)')
    torch = native_runtime(plan, base)
    receipt = dict(packages=actual, torch=torch.__version__, cuda=torch.version.cuda,
        count=torch.cuda.device_count(), bf16=torch.cuda.is_bf16_supported(), name=torch.cuda.get_device_name(0))
    if followup_mode(plan) or is_eight(plan):
        receipt.update(run_id=plan['identity']['run_id'], plan_sha256=digest(Path(base) / 'plan.json'),
                       instance_id=json.loads((Path(base) / 'release.json').read_text())['instance_id'])
    save(Path(base) / 'output/runtime.json', receipt)


def trainer_module(base):
    # Same actual training implementation and answer-only chunked loss as the
    # final CLI, without running its main until the registered train stage.
    path = Path(base) / 'code/scripts/train_open_weight_lora.py'
    return runpy.run_path(str(path), run_name='grounded_capacity_trainer')


def synthetic_capacity_item(tok, trainer, folder, context_tokens, answer_tokens):
    """Exercise the real grounded schema/template/EOS path, never raw token IDs."""
    require(0 < answer_tokens < context_tokens and tok.eos_token_id is not None,
            'Invalid synthetic context/answer envelope')

    def exact_text(seed, target, measure):
        padding = 0
        for _ in range(16):
            text = seed + ' x' * padding
            length = measure(text)
            if length == target:
                return text
            padding += target - length
            require(0 <= padding <= 4 * context_tokens, 'Synthetic text cannot fit token envelope')
        raise RuntimeError('Synthetic text did not reach exact token envelope')

    system = dict(role='system', content='This is fictional synthetic capacity data, not a real source or evaluation.')
    answer = exact_text('Fictional synthetic output:', answer_tokens,
        lambda text: len(tok(text + tok.eos_token, add_special_tokens=False)['input_ids']))

    def prompt_length(text):
        prompt = tok.apply_chat_template([system, dict(role='user', content=text)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return len(tok(prompt, add_special_tokens=False)['input_ids'])

    user = exact_text('A fictional blue square appears in this invented document.', context_tokens - answer_tokens,
                      prompt_length)
    row = dict(id='synthetic-capacity-only', task='synthetic-capacity', split='train',
               messages=[system, dict(role='user', content=user)], answer=answer)
    path = Path(folder) / 'synthetic.jsonl'
    with path.open('x') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    entry = dict(path=path.name, sha256=digest(path), adapter='grounded_messages', split='train', epochs=1)
    items, dropped = trainer['build_items'](tok, [entry], Path(folder), 'qwen3', context_tokens, answer_tokens, False)
    require(dropped == 0 and len(items) == 1, 'Synthetic grounded schema did not produce exactly one item')
    item = items[0]
    require(len(item['ids']) == context_tokens and len(item['labels']) == context_tokens
            and item['labels'][:context_tokens - answer_tokens] == [-100] * (context_tokens - answer_tokens)
            and item['labels'][-answer_tokens:] == item['ids'][-answer_tokens:]
            and item['labels'][-1] == tok.eos_token_id,
            'Synthetic grounded context/answer/EOS boundary differs')
    return item


def capacity_stage(plan, base, torch):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    trainer = trainer_module(base)
    directory = Path(base) / 'model'
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    recipe = plan['recipe']
    manifest_path = Path(base) / 'code' / plan['data']['manifest']['path']
    manifest = json.loads(manifest_path.read_text())
    items, dropped = trainer['build_items'](tok, manifest['train'], manifest_path.parent,
        'qwen3', recipe['context_tokens'], recipe['answer_tokens'], True)
    require(dropped == 0 and len(items) == recipe['rows'] * recipe['epochs'],
            'Actual native-template tokenization differs from approved finite data')
    tokenized_rows = len(items)
    del items
    cases = validate_data(Path(base) / 'code', plan)
    if is_eight(plan):
        cases = cases['development_inputs']
        valid, valid_dropped = trainer['build_items'](tok, manifest['valid'], manifest_path.parent,
            'qwen3', recipe['context_tokens'], recipe['answer_tokens'], False)
        require(valid_dropped == 0 and len(valid) == 8, 'Eight development tokenization differs')
        del valid
    for case in cases:
        prompt = tok.apply_chat_template(case['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_tokens = len(tok.encode(prompt, add_special_tokens=False))
        if is_eight(plan):
            eight_module(plan).validate_prompt_budget(plan, prompt_tokens)
        else:
            require(prompt_tokens + recipe['answer_tokens'] <= recipe['context_tokens'],
                    'Blind native-template token budget exceeds approval')
    folder = Path(base) / 'output/capacity'
    folder.mkdir(mode=0o700)
    item = synthetic_capacity_item(tok, trainer, folder, recipe['context_tokens'], recipe['answer_tokens'])
    ids = item['ids']
    model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.bfloat16,
                                                local_files_only=True, trust_remote_code=False).to('cuda')
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'],
        lora_dropout=recipe['dropout'], target_modules=trainer['TARGETS'], task_type='CAUSAL_LM'))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=recipe['lr'], weight_decay=0.0)
    # Only fictional grounded-schema data reaches this optimizer. Both approved
    # microbatches exercise the full context/answer and optimizer allocation.
    adapter_b = [p for name, p in model.named_parameters() if p.requires_grad and '.lora_B.' in name]
    require(adapter_b and all(int(torch.count_nonzero(p)) == 0 for p in adapter_b),
            'Fresh synthetic LoRA must start with zero B matrices')
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(recipe['batch_size']):
        loss = trainer['batch_loss'](model, [item], 'cuda', tok.pad_token_id or tok.eos_token_id, 'qwen3') / recipe['batch_size']
        require(bool(torch.isfinite(loss)), 'Synthetic capacity loss is not finite')
        loss.backward(); losses.append(float(loss.detach()))
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    require(bool(torch.isfinite(grad_norm)) and float(grad_norm) > 0, 'Synthetic gradient is zero or nonfinite')
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in parameters)
            and any(int(torch.count_nonzero(p)) > 0 for p in adapter_b),
            'Synthetic optimizer did not make a finite parameter update')
    model.save_pretrained(folder / 'adapter', safe_serialization=True)
    expected = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    del loss, optimizer, parameters, adapter_b, model
    import gc
    gc.collect(); torch.cuda.empty_cache()
    fresh = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.bfloat16,
                                                local_files_only=True, trust_remote_code=False).to('cuda')
    loaded = PeftModel.from_pretrained(fresh, folder / 'adapter', is_trainable=False, local_files_only=True)
    actual = dict(loaded.named_parameters())
    require(all(k in actual and torch.equal(v, actual[k].detach().cpu()) for k, v in expected.items()),
            'Capacity adapter save/reload mismatch')
    loaded.eval()
    with torch.no_grad():
        output = loaded(input_ids=torch.tensor([ids[:32]], device='cuda'), use_cache=False)
        require(bool(torch.isfinite(output.logits).all()), 'Reloaded capacity forward is nonfinite')
    save(Path(base) / 'output/capacity-receipt.json', dict(synthetic=True, discarded=False,
        microbatches=recipe['batch_size'], context_tokens=len(ids), answer_tokens=recipe['answer_tokens'], losses=losses,
        finite_parameter_update=True, adapter_reload_equal=True, synthetic_row_sha256=digest(folder / 'synthetic.jsonl'),
        admitted_expanded_rows=tokenized_rows,
        max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved()))


def train_argv(plan, base):
    r = plan['recipe']; base = Path(base)
    argv = [str(base / 'code/scripts/train_open_weight_lora.py'), '--model', str(base / 'model'),
        '--model-id', plan['model']['id'], '--model-revision', plan['model']['revision'], '--family', 'qwen3',
        '--manifest', str(base / 'code' / plan['data']['manifest']['path']), '--out', str(base / 'output/training'),
        '--steps', str(r['steps']), '--batch-size', str(r['batch_size']), '--micro-batch-size', str(r['micro_batch_size']),
        '--lr', str(r['lr']), '--rank', str(r['rank']), '--alpha', str(r['alpha']), '--dropout', str(r['dropout']),
        '--max-tokens', str(r['context_tokens']), '--max-answer-tokens', str(r['answer_tokens']),
        '--seed', str(r['seed']), '--device', 'cuda']
    if is_eight(plan):
        argv += ['--checkpoint-every', '12', '--valid-every', '12']
    return argv


def train_stage(plan, base, torch):
    argv = train_argv(plan, base)
    sys.argv = argv
    runpy.run_path(argv[0], run_name='__main__')
    report = json.loads((Path(base) / 'output/training/training.json').read_text())
    require(report['status'] == 'complete' and report['steps'] == plan['recipe']['steps']
            and report['train_rows'] == plan['recipe']['rows'] * plan['recipe']['epochs']
            and report['dropped_over_max_tokens'] == 0 and report['init_adapter'] is None
            and report['base_weights'] == {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'},
            'Final native training receipt differs from finite fresh-base contract')


def decode_generation(tokenizer, output_token_ids, eos_token_id, max_new_tokens):
    """Keep full protocol diagnostics; remove only a terminal configured stop from the response body."""
    require(type(max_new_tokens) is int and max_new_tokens > 0
            and isinstance(output_token_ids, list) and 0 < len(output_token_ids) <= max_new_tokens
            and all(type(token) is int and token >= 0 for token in output_token_ids),
            'Generated token sequence exceeds or violates the registered envelope')
    if eos_token_id is None:
        stops = ()
    elif type(eos_token_id) is int:
        stops = (eos_token_id,)
    else:
        require(isinstance(eos_token_id, list), 'Actual generation EOS configuration must be an integer, list, or null')
        stops = eos_token_id
    require(all(type(token) is int and token >= 0 for token in stops), 'Invalid actual generation EOS token ID')
    terminal = output_token_ids[-1] if output_token_ids[-1] in stops else None
    reason = 'eos' if terminal is not None else 'length'
    require(reason == 'eos' or len(output_token_ids) == max_new_tokens, 'Unexplained generation termination')
    raw = tokenizer.decode(output_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    response = (tokenizer.decode(output_token_ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if terminal is not None else raw)
    return dict(raw_output=raw, response_text=response, termination_reason=reason,
                terminal_stop_token_id=terminal, generated_tokens=len(output_token_ids))


def evaluation_stage(plan, base, torch, adapter):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = Path(base)
    cases = validate_data(base / 'code', plan)
    tok = AutoTokenizer.from_pretrained(base / 'model', local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base / 'model', torch_dtype=torch.bfloat16,
                                                local_files_only=True, trust_remote_code=False).to('cuda')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, base / 'output/training/adapter', is_trainable=False, local_files_only=True)
    model.eval()
    target = base / 'output' / ('adapter-outputs.jsonl' if adapter else 'baseline-outputs.jsonl')
    with target.open('x') as stream:
        for case in cases:
            prompt = tok.apply_chat_template(case['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
            width = inputs['input_ids'].shape[1]
            require(width + plan['recipe']['answer_tokens'] <= plan['recipe']['context_tokens'],
                    'Blind prompt exceeds registered context; no truncation')
            with torch.inference_mode():
                result = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=plan['recipe']['answer_tokens'],
                    use_cache=True, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                    return_dict_in_generate=True)
            ids = result.sequences[0, width:].tolist()
            decoded = decode_generation(tok, ids, model.generation_config.eos_token_id, plan['recipe']['answer_tokens'])
            row = dict(id=case['id'], kind=case['kind'], family=case['family'], cluster_id=plan['evaluation']['cluster_id'],
                       prompt_sha256=object_sha(case['messages']),
                       rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                       prompt_token_ids=inputs['input_ids'][0].tolist(), output_token_ids=ids,
                       adapter=adapter, **decoded)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    plan = read_bound(args.plan, args.plan_sha256)
    if args.phase == 'bootstrap':
        bootstrap(plan, args.plan_sha256, args.base)
        return
    require(plan['limits'] == LIMITS, 'Native fixed resource policy differs')
    if args.phase == 'runtime':
        runtime_stage(plan, args.base)
        return
    ordered = ordered_module(plan)
    if ordered:
        ordered.child_stage(sys.modules[__name__], plan, args.base, args.phase)
        return
    eight = eight_module(plan)
    if eight:
        eight.child_stage(sys.modules[__name__], plan, args.base, args.phase)
        return
    require(args.phase in ('capacity', 'baseline', 'train', 'adapter'), 'Unknown training stage')
    validate_model(args.base / 'model', plan)
    torch = native_runtime(plan, args.base)
    if args.phase == 'capacity':
        capacity_stage(plan, args.base, torch)
    elif args.phase == 'train':
        train_stage(plan, args.base, torch)
    else:
        evaluation_stage(plan, args.base, torch, args.phase == 'adapter')


if __name__ == '__main__':
    main()
