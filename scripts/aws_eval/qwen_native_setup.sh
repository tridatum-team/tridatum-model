#!/bin/bash
# Authenticated root-only native setup/export; never executes training/model code.
set -Eeuo pipefail
umask 077
exec python3.12 - "$@" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time

BASE = pathlib.Path('/var/lib/qwen013')
RUNTIME = pathlib.Path('/opt/qwen013-native')
COMMIT = '8ca4060b7e30e76baaa5670ee1ec1b760b7f8552'
CONVERTER_SHA = 'ddb143ac34022b0646e51d9c35e9cfa3ec3d0a36091975ebc88f553dd18ca350'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(root, name):
    p = pathlib.PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
            'Noncanonical relative file: ' + name)
    target = root / name
    require(all(not part.is_symlink() for part in (target, *target.parents)), 'Symlink in file path')
    s = target.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Not an exclusive regular file')
    return target


def export(root, rc):
    config = json.loads((BASE / 'config.json').read_text())
    out = root / 'runs/qwen-specialization-013' / config['run_id'] / 'remote-output'
    candidates = [(BASE / name, name) for name in ('bootstrap.log', 'owned-unit-cleanup.json',
                  'native-build.json', 'native-build.log', 'native-converter-install.json')]
    excluded, errors, included, skipped = [], [], [], []
    manifest_path = out / 'export-manifest.json'
    try:
        manifest_path = safe_file(out, 'export-manifest.json')
        require(manifest_path.stat().st_size <= 4 * 1024**2, 'Oversize output manifest')
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get('schema_version') == 1 and manifest.get('complete') is True,
                'Incomplete export manifest')
        files, excluded = manifest['files'], manifest['excluded']
        names = [row['path'] for row in files + excluded]
        require(len(names) == len(set(names)) <= 4096 and 'export-manifest.json' not in names,
                'Duplicate/reserved manifest path')
        actual = set()
        for path in out.rglob('*'):
            require(not path.is_symlink(), 'Symlink in output tree')
            if not path.is_dir():
                actual.add(str(path.relative_to(out)))
        require(actual == set(names) | {'export-manifest.json'}, 'Output manifest membership mismatch')
        for row in files + excluded:
            path = safe_file(out, row['path'])
            require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                    'Output manifest integrity mismatch: ' + row['path'])
        require(all(isinstance(row.get('reason'), str) and row['reason'] for row in excluded),
                'Missing duplicate exclusion reason')
        require(sum(row['bytes'] for row in files) == manifest['retained_bytes'], 'Retained total mismatch')
        candidates += [(out / row['path'], 'output/' + row['path']) for row in files]
        candidates.append((manifest_path, 'output/export-manifest.json'))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        errors.append('Manifest unavailable/invalid; incomplete evidence export: ' + str(exc))
        excluded = []
        if out.is_dir() and not out.is_symlink():
            candidates += [(p, 'output/' + str(p.relative_to(out))) for p in sorted(out.rglob('*'))]
    used = 0
    candidates.sort(key=lambda x: (x[0].suffix not in ('.json', '.log'), x[1]))
    with tarfile.open(BASE / 'results.tar', 'w', format=tarfile.USTAR_FORMAT) as archive:
        for path, name in candidates:
            try:
                s = path.lstat()
                if not stat.S_ISREG(s.st_mode):
                    continue
                require(s.st_nlink == 1 and all(not p.is_symlink() for p in path.parents),
                        'Unsafe export file')
                # Never silently trim a retained candidate or evidence log.
                if used + s.st_size > 500 * 1024**2:
                    skipped.append({'path': name, 'bytes': s.st_size, 'sha256': digest(path),
                                    'reason': 'payload_cap; export incomplete'})
                    errors.append('Required export file exceeds remaining payload: ' + name)
                    continue
                item = tarfile.TarInfo(name)
                item.size, item.mode = s.st_size, 0o600
                with path.open('rb') as stream:
                    archive.addfile(item, stream)
                used += s.st_size
                included.append({'path': name, 'bytes': s.st_size, 'sha256': digest(path)})
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(name + ': ' + str(exc))
    require((BASE / 'results.tar').stat().st_size <= 511 * 1024**2, 'Archive exceeds envelope')
    result = {'run_id': config['run_id'], 'exit_code': rc, 'included': included, 'excluded': excluded,
              'skipped': skipped, 'errors': errors, 'export_complete': not errors,
              'archived_payload_bytes': used, 'completion_scope': 'bootstrap_exit_and_export_only',
              'no_model_retry': True}
    (BASE / 'completion.json').write_text(json.dumps(result, indent=2))
    require((BASE / 'completion.json').stat().st_size <= 1024**2, 'Completion exceeds envelope')
    (BASE / 'output-uri').write_text('s3://trid-training-148761639313/eval/aws-eval-006/runs/' + config['run_id'])


def setup(root):
    started = time.time()
    receipt = {'schema_version': 1, 'completed': False, 'source_commit': COMMIT,
               'started_unix': started, 'automatic_retries': 0}
    log_path = BASE / 'native-build.log'
    receipt_path = BASE / 'native-build.json'
    log_path.touch(exist_ok=False)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    receipt_path.chmod(0o644)
    try:
        require(os.geteuid() == 0 and platform.system() == 'Linux',
                'Native setup requires Linux root')
        config = json.loads((BASE / 'config.json').read_text())
        plan = json.loads(safe_file(root, config['plan']).read_text())
        build = plan['native']['build']
        architecture = build.get('architecture', 'aarch64')
        require(architecture in ('aarch64', 'x86_64') and platform.machine() == architecture
                and os.environ.get('QWEN_NATIVE_ARCHITECTURE', architecture) == architecture,
                'Native setup architecture differs from authenticated plan')
        receipt['architecture'] = architecture
        seconds = build['setup_seconds']
        require(type(seconds) is int and 1 <= seconds <= 1200, 'Invalid finite setup deadline')
        deadline = min(started + seconds, config['worker_deadline_unix'] - 305)
        require(deadline > started, 'No setup time before cleanup reserve')
        require(build['source_commit'] == COMMIT and build['server_build'] == 'b1-8ca4060',
                'Native source/build identity mismatch')
        archive = safe_file(root, build['source_archive'])
        require(re.fullmatch('[0-9a-f]{64}', build['source_archive_sha256']) is not None
                and digest(archive) == build['source_archive_sha256'], 'Source archive SHA mismatch')
        verified = json.loads((BASE / 'verified-manifest.json').read_text())
        rows = {row['path']: row for row in verified['files']}
        require(rows[build['source_archive']]['sha256'] == build['source_archive_sha256'],
                'Source archive not authenticated by input allowlist')
        entry = build['source_entry']
        require(entry == 'llama.cpp-' + COMMIT, 'Source entry mismatch')
        receipt.update(source_archive={'path': str(archive), 'sha256': digest(archive)}, source_entry=entry)
        require(not RUNTIME.exists(), 'Native runtime must be fresh; no retry')
        RUNTIME.mkdir(mode=0o755)
        source = RUNTIME / 'source'
        source.mkdir(mode=0o755)
        # Validate the complete tar before writing. No tar extraction APIs or links.
        with tarfile.open(archive, 'r:*') as tar:
            members = tar.getmembers()
            names = [m.name.rstrip('/') if m.isdir() else m.name for m in members]
            require(len(names) == len(set(names)) <= 30000, 'Duplicate/excess source members')
            require(sum(m.size for m in members) <= 512 * 1024**2, 'Source expansion cap')
            for m, name in zip(members, names):
                p = pathlib.PurePosixPath(name)
                require(str(p) == name and p.parts[0] == entry and '..' not in p.parts
                        and not p.is_absolute() and (m.isfile() or m.isdir()) and not m.sparse
                        and not any(k not in ('mtime', 'atime', 'ctime', 'comment') for k in m.pax_headers),
                        'Unsafe source archive member: ' + name)
                require('.git' not in p.parts, 'Git metadata prohibited')
            for m, name in zip(members, names):
                require(time.time() < deadline, 'Setup extraction deadline')
                relative = pathlib.PurePosixPath(name).relative_to(entry)
                target = source / relative
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o755)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                    with target.open('xb') as output, tar.extractfile(m) as stream:
                        shutil.copyfileobj(stream, output, 1024 * 1024)
                    target.chmod(0o644)
        converter = source / 'convert_lora_to_gguf.py'
        require(digest(converter) == CONVERTER_SHA, 'Pinned converter entry SHA mismatch')
        env = {'PATH': '/usr/bin:/bin', 'HOME': '/root', 'LANG': 'C.UTF-8',
               'PYTHONNOUSERSITE': '1', 'PIP_CONFIG_FILE': '/dev/null', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}

        def run(argv, cap, offline=True):
            remaining = int(deadline - time.time()) - 5
            require(remaining > 0, 'Native setup deadline exhausted')
            command = ['timeout', '-k', '5', str(min(cap, remaining))]
            if offline:
                command += ['unshare', '--net', '--']
            command += argv
            with log_path.open('a') as log:
                log.write(json.dumps({'argv': argv, 'network_disabled': offline}) + '\n')
                log.flush()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
            require(result.returncode == 0, 'Native setup command failed: ' + argv[0])

        def versions(python):
            target = BASE / ('native-training-versions.json' if 'qwen013-venv' in python else 'native-converter-versions.json')
            run([python, '-I', '-c', 'import importlib.metadata,json,pathlib,sys; '
                 'pathlib.Path(sys.argv[1]).write_text(json.dumps({d.metadata["Name"].lower():d.version '
                 'for d in importlib.metadata.distributions()},sort_keys=True))', str(target)], 10)
            return json.loads(target.read_text())

        training_python = '/opt/qwen013-venv/bin/python'
        before = versions(training_python)
        receipt['training_versions_before'] = before
        converter_python = str(RUNTIME / 'converter-venv/bin/python')
        run(['python3.12', '-m', 'venv', str(RUNTIME / 'converter-venv')], 30)
        # Isolated CPU wheel environment; no training environment mutation, source builds or retries.
        torch_pin = 'torch==2.11.0+cpu' if architecture == 'x86_64' else 'torch==2.11.0'
        pins = [torch_pin, 'numpy==2.2.6', 'transformers==4.57.6', 'sentencepiece==0.2.1',
                'gguf==0.19.0', 'protobuf==4.25.8']
        run([converter_python, '-m', 'pip', 'install', '--retries', '0', '--timeout', '30',
             '--only-binary=:all:', '--disable-pip-version-check', '--no-cache-dir',
             '--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
             '--report', str(BASE / 'native-converter-install.json'), *pins], 420, offline=False)
        installed = versions(converter_python)
        require(not any(k.startswith(('nvidia-', 'cuda-')) for k in installed), 'Non-CPU converter dependency')
        require(all(installed.get(pin.split('==')[0], '').split('+')[0] == pin.split('==')[1].split('+')[0]
                    for pin in pins), 'Converter direct dependency pin mismatch')
        torch_identity = BASE / 'native-converter-torch.json'
        run([converter_python, '-I', '-c', 'import json,pathlib,sys,torch; '
             'pathlib.Path(sys.argv[1]).write_text(json.dumps({"version":torch.__version__,'
             '"cuda":torch.version.cuda}))', str(torch_identity)], 15)
        actual_torch = json.loads(torch_identity.read_text())
        require(actual_torch['cuda'] is None, 'Converter Torch is not CPU-only')
        if architecture == 'x86_64':
            require(installed.get('torch') == actual_torch['version'] == '2.11.0+cpu',
                    'Explicit CPU converter wheel identity mismatch')
        receipt['converter'] = {'path': str(converter), 'sha256': CONVERTER_SHA,
                                'python': converter_python, 'versions': installed, 'pins': pins,
                                'torch_version': actual_torch['version'], 'torch_cuda': actual_torch['cuda']}
        flags = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=/usr/bin/gcc',
                 '-DCMAKE_CXX_COMPILER=/usr/bin/g++', '-DBUILD_SHARED_LIBS=ON',
                 '-DLLAMA_BUILD_NUMBER=1', '-DLLAMA_BUILD_COMMIT=8ca4060',
                 '-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE', '-DGIT_EXECUTABLE=/bin/false', '-DGIT_EXE=/bin/false',
                 '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF',
                 '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_UI=OFF',
                 '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_CONNECT=OFF', '-DLLAMA_OPENSSL=OFF',
                 '-DLLAMA_LLGUIDANCE=OFF', '-DGGML_CPU=ON', '-DGGML_NATIVE=OFF',
                 '-DGGML_CPU_REPACK=OFF',
                 '-DGGML_CPU_KLEIDIAI=OFF', '-DGGML_BACKEND_DL=OFF', '-DGGML_CCACHE=OFF',
                 '-DGGML_OPENMP_FETCH=OFF', '-DGGML_BUILD_TESTS=OFF', '-DGGML_BUILD_EXAMPLES=OFF',
                 '-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
        if architecture == 'aarch64':
            flags += ['-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod']
        flags += ['-DGGML_' + backend + '=OFF' for backend in ('CUDA', 'HIP', 'MUSA', 'METAL',
                  'VULKAN', 'WEBGPU', 'SYCL', 'OPENCL', 'OPENVINO', 'CANN', 'BLAS', 'ACCELERATE',
                  'RPC', 'HEXAGON', 'ZENDNN', 'ZDNN', 'VIRTGPU', 'VIRTGPU_BACKEND', 'ET', 'ET_SYSEMU')]
        receipt['cmake_flags'] = flags
        build_dir = RUNTIME / 'build'
        run(['cmake', '-S', str(source), '-B', str(build_dir), *flags], 90)
        run(['cmake', '--build', str(build_dir), '--target', 'llama-server', '--parallel', '4'], 600)
        server_dir = root / 'runs/tools/llama.cpp-8ca4060'
        require(not server_dir.exists(), 'Native server installation must be fresh')
        server_dir.mkdir(parents=True, mode=0o755)
        server_dir.chmod(0o755)
        server_dir.parent.chmod(0o755)
        binary = server_dir / 'llama-server'
        shutil.copyfile(build_dir / 'bin/llama-server', binary)
        binary.chmod(0o755)
        for library in (build_dir / 'bin').glob('*.so*'):
            destination = server_dir / library.name
            if library.is_symlink():
                link = os.readlink(library)
                require(pathlib.PurePosixPath(link).name == link, 'Nonlocal built library link')
                destination.symlink_to(link)
            else:
                shutil.copyfile(library, destination)
                destination.chmod(0o644)
        env['LD_LIBRARY_PATH'] = str(server_dir)
        # Metadata only: never open a model, start HTTP, or infer quality here.
        version_file = BASE / 'native-server-version.txt'
        run(['/bin/sh', '-c', '"$1" --version > "$2" 2>&1', 'version', str(binary), str(version_file)], 10)
        version = version_file.read_text()
        require(re.search(r'build 1, commit 8ca4060\)', version) is not None
                and 'Linux' in version and architecture in version, 'Actual native version mismatch')
        libs_file = BASE / 'native-server-libraries.txt'
        run(['/bin/sh', '-c', 'ldd "$1" > "$2"', 'libraries', str(binary), str(libs_file)], 10)
        libraries = []
        for line in libs_file.read_text().splitlines():
            require('not found' not in line, 'Unresolved shared library')
            matches = re.findall(r'(?:=>\s+|^\s*)(/\S+)', line)
            for name in matches:
                path = pathlib.Path(name).resolve(strict=True)
                libraries.append({'path': str(path), 'sha256': digest(path)})
        require(libraries, 'No actual runtime shared library evidence')
        cache = (build_dir / 'CMakeCache.txt').read_text()
        receipt['compiler'] = {'c': '/usr/bin/gcc', 'cxx': '/usr/bin/g++',
                               'version_metadata': version, 'cmake_cache_sha256': digest(build_dir / 'CMakeCache.txt')}
        require('CMAKE_BUILD_TYPE:STRING=Release' in cache, 'Release build not established')
        receipt['server'] = {'path': str(binary), 'sha256': digest(binary), 'version': version,
                             'build': 'b1-8ca4060', 'shared_libraries': libraries,
                             'library_path': str(server_dir)}
        after = versions(training_python)
        receipt['training_versions_after'] = after
        require(before == after, 'Training environment changed during native setup')
        # Root controls all runtime content; unprivileged model children may read/execute only.
        for directory, dirs, files in os.walk(RUNTIME):
            pathlib.Path(directory).chmod(0o755)
            for name in files:
                path = pathlib.Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        receipt['completed'] = True
    except BaseException as exc:
        receipt['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        receipt['finished_unix'] = time.time()
        receipt['log'] = {'path': str(log_path), 'sha256': digest(log_path)}
        receipt_path.write_text(json.dumps(receipt, indent=2))
        receipt_path.chmod(0o644)


require(len(sys.argv) in (3, 4), 'Usage: qwen_native_setup.sh setup|export ROOT [exit-code]')
require(os.geteuid() == 0, 'Root required')
root = pathlib.Path(sys.argv[2])
if sys.argv[1] == 'setup' and len(sys.argv) == 3:
    setup(root)
elif sys.argv[1] == 'export' and len(sys.argv) == 4:
    export(root, int(sys.argv[3]))
else:
    raise RuntimeError('Invalid native setup operation')
PY
