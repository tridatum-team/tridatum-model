"""Portable, unfused MLX bundles; no server and no implicit downloads.

Requires preinstalled CPython 3.12 and pip on macOS arm64. First print
`dependency-specs`, then explicitly download those exact wheels with pip on the
source machine. `pack --wheelhouse DIR` never downloads anything. On the target:
First independently authenticate the manifest SHA256 and launcher file pin using
a trusted verifier; then follow this order (integrity-only uses no site packages):
python3.12 -I -S BUNDLE/runtime/mlx_offline.py verify BUNDLE \
    --manifest-sha256 TRUSTED_SHA256 --integrity-only
python3.12 -I -m pip install --no-index --find-links BUNDLE/wheelhouse \
    --require-hashes -r BUNDLE/requirements.lock
python3.12 -I BUNDLE/runtime/mlx_offline.py verify BUNDLE --manifest-sha256 TRUSTED_SHA256
python3.12 -I BUNDLE/runtime/mlx_offline.py run BUNDLE --manifest-sha256 TRUSTED_SHA256 \
    --messages messages.json --output result.json
Use a fresh virtual environment. Transfer the manifest SHA256 through a trusted
channel. BEFORE installation or executing transferred Python, independently
authenticate manifest.json and the launcher's file pin using a trusted verifier.
The transferred launcher cannot authenticate its own potentially replaced code.
"""

# sys is built in. Refuse unsafe startup before importing any shadowable module.
import sys
if __name__ == '__main__' and not sys.flags.isolated:
    sys.stderr.write('Refusing non-isolated startup: invoke Python with -I.\n')
    raise SystemExit(2)

import argparse
import email
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import zipfile

SCHEMA = 'trid-offline-mlx-bundle-v1'
RUNTIME_FILES = ('mlx_offline.py', 'mlx_inference.py', 'diagnose_mlx_inference.py',
                 'mlx_grounded.py', 'mlx_grounded_guard.py')
FORBIDDEN_SUFFIXES = {'.bin', '.pt', '.pth', '.pkl', '.pickle', '.py', '.pyc', '.so', '.dylib'}
FORBIDDEN_PACKAGES = {'torch', 'torchvision', 'torchaudio'}
LIMITS = {'rss_stop_bytes': 6 * 1024**3, 'mlx_stop_bytes': 6 * 1024**3,
          'swap_growth_stop_bytes': 256 * 1024**2, 'pressure_required': 1,
          'question_seconds': 180, 'inference_seconds': 1800}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def exclusive_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json_bytes(value))


def file_pin(path, relative=None):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'path': relative if relative is not None else str(Path(path).resolve()),
            'sha256': digest.hexdigest(), 'bytes': size}


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid manifest relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Unsafe manifest path: {value!r}')
    if ':' in value:
        raise ValueError(f'Unsafe manifest path: {value!r}')
    return path


def regular_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Expected real directory: {root}')
    result = []
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f'Symlink or special file is forbidden: {path}')
        result.extend(Path(current) / name for name in files)
    return sorted(result)


def canonical_name(name):
    from packaging.utils import canonicalize_name
    return canonicalize_name(name)


def dependency_specs():
    """Resolve installed mandatory dependencies, including requested extras, without imports."""
    from packaging.requirements import Requirement
    # The receipt records safetensors even when MLX itself reads that format.
    pending = [(name, frozenset()) for name in ('mlx-lm', 'packaging', 'safetensors')]
    seen = set()
    versions = {}
    while pending:
        name, extras = pending.pop()
        name = canonical_name(name)
        if name in FORBIDDEN_PACKAGES:
            raise ValueError(f'Runtime closure unexpectedly requires forbidden package {name}')
        state = (name, extras)
        if state in seen:
            continue
        seen.add(state)
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for item in distribution.requires or []:
            requirement = Requirement(item)
            if requirement.marker and not any(
                    requirement.marker.evaluate({'extra': extra}) for extra in ('', *extras)):
                continue
            installed = importlib.metadata.version(requirement.name)
            if requirement.specifier and not requirement.specifier.contains(installed):
                raise ValueError(f'Installed dependency does not satisfy {requirement}: {installed}')
            pending.append((requirement.name, frozenset(requirement.extras)))
    return [{'name': name, 'version': version} for name, version in sorted(versions.items())]


def require_target(target=None):
    expected = {'system': 'Darwin', 'machine': 'arm64', 'python': '>=3.12,<3.13',
                'implementation': 'CPython'}
    if target is not None and target != expected:
        raise ValueError('Unsupported target metadata; native MLX requires macOS arm64 CPython 3.12')
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
            or platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 12)):
        raise ValueError('Native MLX bundle requires macOS arm64 CPython 3.12; Linux/x86 are unsupported')
    return expected


def wheel_inventory(wheelhouse, packages):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    expected = {entry['name']: entry['version'] for entry in packages}
    if len(expected) != len(packages) or FORBIDDEN_PACKAGES.intersection(expected):
        raise ValueError('Duplicate or forbidden runtime package')
    compatible = set(sys_tags())
    found = {}
    rows = []
    for path in regular_files(wheelhouse):
        if path.parent != Path(wheelhouse) or path.suffix != '.whl':
            raise ValueError(f'Wheelhouse may contain only wheel files: {path.name}')
        name, version, _, tags = parse_wheel_filename(path.name)
        name = str(name)
        if expected.get(name) != str(version) or name in found:
            raise ValueError(f'Unexpected, duplicate or wrong-version wheel: {path.name}')
        if not compatible.intersection(tags):
            raise ValueError(f'Wheel incompatible with target interpreter/platform: {path.name}')
        with zipfile.ZipFile(path) as archive:
            metadata_names = [item for item in archive.namelist() if item.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError(f'Invalid wheel metadata: {path.name}')
            metadata = email.message_from_bytes(archive.read(metadata_names[0]))
            if canonical_name(metadata['Name']) != name or metadata['Version'] != str(version):
                raise ValueError(f'Wheel metadata/filename mismatch: {path.name}')
        found[name] = path.name
        rows.append({'name': name, 'version': str(version), 'file': path.name,
                     'sha256': file_pin(path)['sha256']})
    if set(found) != set(expected):
        raise ValueError(f'Wheelhouse missing runtime dependencies: {sorted(set(expected) - set(found))}')
    return sorted(rows, key=lambda row: row['name'])


def reject_remote_config(value):
    if isinstance(value, dict):
        if value.get('auto_map') or value.get('trust_remote_code') or value.get('model_file'):
            raise ValueError('Custom/remote model code is forbidden')
        for child in value.values():
            reject_remote_config(child)
    elif isinstance(value, list):
        for child in value:
            reject_remote_config(child)


def validate_model_files(root, adapter=False):
    paths = regular_files(root)
    for path in paths:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f'Unsafe model/adapter file: {path.name}')
        if path.suffix == '.json':
            reject_remote_config(json.loads(path.read_bytes()))
    required = 'adapter_config.json' if adapter else 'config.json'
    if not (Path(root) / required).is_file():
        raise ValueError(f'Missing {required}')
    if not any(path.suffix == '.safetensors' for path in paths):
        raise ValueError('Only safetensors model/adapter weights are supported')
    if adapter and not (Path(root) / 'adapters.safetensors').is_file():
        raise ValueError('Missing adapters.safetensors')
    return paths


def copy_tree(source, destination):
    """Dereference source snapshot file links, never create links in the artifact."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f'Missing source directory: {source}')
    destination.mkdir()
    for current, directories, files in os.walk(source, followlinks=False):
        relative = Path(current).relative_to(source)
        for name in directories:
            child = Path(current) / name
            if child.is_symlink():
                raise ValueError(f'Source directory symlink is forbidden: {child}')
            (destination / relative / name).mkdir()
        for name in files:
            child = Path(current) / name
            if not child.is_file():
                raise ValueError(f'Source special file is forbidden: {child}')
            with child.open('rb') as reader, (destination / relative / name).open('xb') as writer:
                shutil.copyfileobj(reader, writer, 1024**2)


def runtime_module(runtime, filename):
    sys.dont_write_bytecode = True
    name = '_offline_' + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, Path(runtime) / filename)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load bundled runtime: {filename}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tree_pins(root):
    return [file_pin(path, path.relative_to(root).as_posix()) for path in regular_files(root)]


def binding(model_files):
    return hashlib.sha256(json_bytes(model_files)).hexdigest()


def pack(model, profile, wheelhouse, output, source_repo, source_revision, adapter=None,
         license_file=None, license_source=None, source_card=None):
    target = require_target()
    if not source_repo or not source_revision:
        raise ValueError('Source repository and immutable revision metadata are required')
    if license_source is not None and license_file is None:
        raise ValueError('--license-source requires --license-file')
    runtime_source = Path(__file__).resolve().parent
    inference = runtime_module(runtime_source, 'mlx_inference.py')
    selected = inference.validate_profile(profile)
    packages = dependency_specs()
    wheels = wheel_inventory(Path(wheelhouse), packages)
    destination = Path(output).absolute()
    for source in (model, adapter, wheelhouse):
        if source is not None and destination.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError('Bundle output must not be inside a source directory')
    destination.mkdir(parents=False, exist_ok=False)
    copy_tree(model, destination / 'model')
    validate_model_files(destination / 'model')
    config = inference.local_model_config(destination / 'model')
    if selected['context_tokens'] > config.get('text_config', config)['max_position_embeddings']:
        raise ValueError('Selected profile exceeds actual model context')
    if adapter is not None:
        copy_tree(adapter, destination / 'adapter')
        validate_model_files(destination / 'adapter', adapter=True)
    copy_tree(wheelhouse, destination / 'wheelhouse')
    runtime = destination / 'runtime'
    runtime.mkdir()
    for name in RUNTIME_FILES:
        with (runtime_source / name).open('rb') as reader, (runtime / name).open('xb') as writer:
            shutil.copyfileobj(reader, writer)
    requirements = ''.join(f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}\n"
                           for row in wheels)
    with (destination / 'requirements.lock').open('x', encoding='utf-8') as stream:
        stream.write(requirements)
    exclusive_json(destination / 'profile.json', selected)
    provenance_files = {}
    if license_file is not None or source_card is not None:
        (destination / 'licenses').mkdir()
        for kind, source, name in (('license', license_file, 'LICENSE'),
                                   ('source_card', source_card, 'SOURCE_CARD')):
            if source is not None:
                if not Path(source).is_file() or Path(source).stat().st_size == 0:
                    raise ValueError(f'Nonempty {kind} file required')
                target_file = destination / 'licenses' / name
                with Path(source).open('rb') as reader, target_file.open('xb') as writer:
                    shutil.copyfileobj(reader, writer)
                provenance_files[kind] = {
                    'file': file_pin(target_file, f'licenses/{name}'),
                    'source': license_source if kind == 'license' and license_source else source_repo}
    model_files = tree_pins(destination / 'model')
    manifest = {
        'schema': SCHEMA, 'target': target,
        'source': {'repository': source_repo, 'revision': source_revision},
        'license_provenance': provenance_files,
        'model': {'path': 'model', 'files': model_files, 'binding_sha256': binding(model_files)},
        'adapter': (None if adapter is None else {
            'path': 'adapter', 'files': tree_pins(destination / 'adapter'),
            'base_binding_sha256': binding(model_files),
            'tensor_coverage': 'exact keys and shapes required by native load_local before inference'}),
        'runtime': {'path': 'runtime', 'files': list(RUNTIME_FILES), 'packages': packages,
                    'source_python': platform.python_version(), 'source_platform': platform.platform()},
        'profile': selected, 'penalty_scope': 'generated_only',
        'weights': {'format': 'safetensors', 'fused': False, 'requantized': False},
        'wheelhouse': {'path': 'wheelhouse', 'wheels': wheels},
        'installation': {
            'prerequisites': ['macOS arm64', 'CPython 3.12', 'pip', 'fresh virtual environment'],
            'argv': ['python3.12', '-I', '-m', 'pip', 'install', '--no-index', '--find-links',
                     '{bundle}/wheelhouse', '--require-hashes', '-r', '{bundle}/requirements.lock'],
            'verify_argv': ['python3.12', '-I', '{bundle}/runtime/mlx_offline.py', 'verify', '{bundle}',
                            '--manifest-sha256', '{trusted_manifest_sha256}'],
            'integrity_argv': ['python3.12', '-I', '-S', '{bundle}/runtime/mlx_offline.py',
                               'verify', '{bundle}', '--manifest-sha256',
                               '{trusted_manifest_sha256}', '--integrity-only'],
            'run_argv': ['python3.12', '-I', '{bundle}/runtime/mlx_offline.py', 'run', '{bundle}',
                         '--manifest-sha256', '{trusted_manifest_sha256}',
                         '--messages', '{messages_json}', '--output', '{exclusive_output_json}'],
            'required_order': ['external_manifest_and_launcher_authentication',
                               'integrity_argv', 'argv', 'verify_argv', 'run_argv'],
            'integrity_only_scope': 'Authenticates every asset, file coverage and profile using '
                                    'standard library only; does not qualify target, installed '
                                    'dependencies, wheel compatibility or model generation.',
            'bootstrap_trust': 'Before install or executing transferred Python, independently authenticate '
                               'manifest SHA256 and launcher file pin using a trusted verifier.',
            'network_required': False},
        'qualification_contract': {
            'performed_by_pack': False,
            'required_smoke': [
                'Authenticate external manifest SHA and launcher pin, then run isolated --integrity-only.',
                'Install requirements.lock using bundled wheels into a disposable CPython 3.12 venv.',
                'Relocate the bundle outside the source checkout; source repo/model paths must be unavailable.',
                'Run bundled verify, then run with real messages and network attempts denied.',
                'Retain exact output, guard receipt, worker receipt and install log; incomplete is not a pass.',
            ],
            'required_negative_cases': ['changed file', 'missing file', 'extra file', 'symlink',
                                        'path traversal', 'wrong target', 'missing wheel',
                                        'adapter/base mismatch', 'partial adapter tensors',
                                        'over-context input', 'incomplete thinking']},
        'safety': {'limits': LIMITS, 'remote_code': False, 'network': False,
                   'checksum_authentication': False, 'external_manifest_sha256_required': True},
        'files': tree_pins(destination),
    }
    exclusive_json(destination / 'manifest.json', manifest)
    return {'bundle': str(destination), 'manifest': file_pin(destination / 'manifest.json')}


def verify(bundle, manifest_sha256, check_environment=True):
    root = Path(bundle).absolute()
    paths = regular_files(root)
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or any(character not in '0123456789abcdef' for character in manifest_sha256)):
        raise ValueError('An externally trusted lowercase manifest SHA256 is required')
    raw_manifest = (root / 'manifest.json').read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != manifest_sha256:
        raise ValueError('Manifest SHA256 mismatch against external trust anchor')
    manifest = json.loads(raw_manifest)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported offline bundle schema')
    if check_environment:
        require_target(manifest.get('target'))
    expected = {}
    for item in manifest['files']:
        relative = str(safe_relative(item['path']))
        if relative in expected or relative == 'manifest.json':
            raise ValueError(f'Duplicate/self-referential manifest file: {relative}')
        if (type(item.get('bytes')) is not int or item['bytes'] < 0
                or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64):
            raise ValueError(f'Invalid file pin: {relative}')
        expected[relative] = item
    actual = {path.relative_to(root).as_posix() for path in paths} - {'manifest.json'}
    if actual != set(expected):
        raise ValueError(f'Bundle file coverage mismatch: missing={sorted(set(expected)-actual)}, '
                         f'extra={sorted(actual-set(expected))}')
    for relative, pin in expected.items():
        if file_pin(root / relative, relative) != pin:
            raise ValueError(f'Bundle file changed: {relative}')
    if manifest['model']['path'] != 'model':
        raise ValueError('Invalid model root')
    validate_model_files(root / 'model')
    model_files = tree_pins(root / 'model')
    if manifest['model']['files'] != model_files or manifest['model']['binding_sha256'] != binding(model_files):
        raise ValueError('Base model binding mismatch')
    adapter = manifest['adapter']
    if adapter is not None:
        if adapter['path'] != 'adapter' or adapter['base_binding_sha256'] != binding(model_files):
            raise ValueError('Adapter/base binding mismatch')
        validate_model_files(root / 'adapter', adapter=True)
        if adapter['files'] != tree_pins(root / 'adapter'):
            raise ValueError('Adapter file coverage mismatch')
    elif any(relative.startswith('adapter/') for relative in expected):
        raise ValueError('Unbound adapter files')
    if (manifest['runtime']['path'] != 'runtime' or manifest['runtime']['files'] != list(RUNTIME_FILES)
            or set(path.name for path in regular_files(root / 'runtime')) != set(RUNTIME_FILES)):
        raise ValueError('Bundled runtime coverage mismatch')
    inference = runtime_module(root / 'runtime', 'mlx_inference.py')
    profile = inference.validate_profile(manifest['profile'])
    config = inference.local_model_config(root / 'model')
    if profile['context_tokens'] > config.get('text_config', config)['max_position_embeddings']:
        raise ValueError('Selected profile exceeds actual model context')
    if profile != json.loads((root / 'profile.json').read_bytes()) or manifest['penalty_scope'] != 'generated_only':
        raise ValueError('Selected profile/penalty semantics mismatch')
    if manifest['weights'] != {'format': 'safetensors', 'fused': False, 'requantized': False}:
        raise ValueError('Unsupported model transformation')
    if manifest['safety']['limits'] != LIMITS or manifest['safety']['network'] or manifest['safety']['remote_code']:
        raise ValueError('Unsafe runtime policy')
    if manifest['wheelhouse']['path'] != 'wheelhouse':
        raise ValueError('Invalid wheelhouse root')
    if check_environment:
        wheels = wheel_inventory(root / 'wheelhouse', manifest['runtime']['packages'])
        if wheels != manifest['wheelhouse']['wheels']:
            raise ValueError('Wheelhouse metadata mismatch')
        requirements = ''.join(f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}\n" for row in wheels)
        if (root / 'requirements.lock').read_text() != requirements:
            raise ValueError('Offline requirements lock mismatch')
    return manifest


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty JSON list')
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] not in ('system', 'user', 'assistant')
                or not isinstance(message['content'], str) or not message['content'].strip()):
            raise ValueError('Each message requires only role and string content')
    if messages[-1]['role'] != 'user':
        raise ValueError('Generation must follow a user message')
    return messages


def run(bundle, messages, output, manifest_sha256):
    # Set before importing any bundled inference/runtime module or native library.
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
                       'USE_TORCH': '0', 'USE_TF': '0'})
    sys.dont_write_bytecode = True
    manifest = verify(bundle, manifest_sha256)
    for package in manifest['runtime']['packages']:
        installed = importlib.metadata.version(package['name'])
        if installed != package['version']:
            raise ValueError(f"Runtime version mismatch: {package['name']} requires {package['version']}, found {installed}")
    messages = validate_messages(messages)
    root = Path(bundle).absolute()
    destination = Path(output).absolute()
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError('Run output must be outside the immutable bundle')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Output already exists: {destination}')
    work = destination.with_name(destination.name + '.run')
    work.mkdir(exist_ok=False)
    cases_path = work / 'cases.json'
    exclusive_json(cases_path, [{'id': 'offline-input', 'messages': messages}])
    runtime = root / 'runtime'
    plan = {'schema': 'trid-offline-mlx-inference-v1', 'run_id': work.name,
            'model': {'path': str(root / 'model'), 'files': manifest['model']['files']},
            'adapter': (None if manifest['adapter'] is None else {
                'path': str(root / 'adapter'), 'files': manifest['adapter']['files']}),
            'cases': file_pin(cases_path), 'profiles': [manifest['profile']],
            'limits': dict(LIMITS), 'output_dir': str(work / 'worker'),
            'source_pins': [file_pin(runtime / name) for name in RUNTIME_FILES], 'parity': False}
    plan_path = work / 'plan.json'
    exclusive_json(plan_path, plan)
    guard = runtime_module(runtime, 'mlx_grounded_guard.py')
    record = guard.supervise(str(plan_path), 'inference', str(work / 'guard-receipt.json'),
                             str(work / 'guard.log'),
                             [sys.executable, '-I', '-B', str(runtime / 'mlx_offline.py'),
                              '_worker', str(root), '--manifest-sha256', manifest_sha256,
                              '--plan', str(plan_path)])
    rows_path = work / 'worker' / 'inference-outputs.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line] if rows_path.exists() else []
    result = {'schema': 'trid-offline-mlx-run-v1', 'guard_success': record['success'],
              'guard_stop_reason': record['stop_reason'], 'records': rows,
              'manifest_sha256': file_pin(root / 'manifest.json')['sha256'],
              'profile': manifest['profile'], 'penalty_scope': 'generated_only',
              'evidence_directory': str(work),
              'answer_complete': (record['success'] and len(rows) == 1
                                  and rows[0].get('termination_reason') == 'eos'
                                  and rows[0].get('framing_status') == 'complete'
                                  and rows[0].get('reasoning_complete') is True)}
    exclusive_json(destination, result)
    return result


def guarded_worker(bundle, manifest_sha256, plan):
    """Isolated child: admit exact sources without adding an import search path."""
    if not sys.flags.isolated:
        raise ValueError('Guarded worker requires isolated Python (-I)')
    verify(bundle, manifest_sha256)
    runtime = Path(bundle).absolute() / 'runtime'
    # Canonical names satisfy sibling imports, but no unverified runtime/cwd
    # directory is added to sys.path where it could shadow standard libraries.
    for filename in ('mlx_inference.py', 'mlx_grounded.py', 'mlx_grounded_guard.py'):
        sys.modules[Path(filename).stem] = runtime_module(runtime, filename)
    diagnostic = runtime_module(runtime, 'diagnose_mlx_inference.py')
    return diagnostic.main(['--plan', plan])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    dependencies = sub.add_parser('dependency-specs', help='Print exact installed runtime closure; never downloads')
    dependencies.add_argument('--json', action='store_true')
    packing = sub.add_parser('pack', help='Copy a complete unfused bundle into an exclusive new directory')
    for name in ('model', 'profile', 'wheelhouse', 'output', 'source-repo', 'source-revision'):
        packing.add_argument('--' + name, required=True)
    packing.add_argument('--adapter')
    packing.add_argument('--license-file', help='Copy full license into hashed licenses/LICENSE')
    packing.add_argument('--license-source', help='Official license provenance URL; defaults to source-repo')
    packing.add_argument('--source-card', help='Copy source model card into hashed licenses/SOURCE_CARD')
    checking = sub.add_parser('verify', help='Verify integrity and macOS arm64 CPython 3.12 compatibility')
    checking.add_argument('bundle')
    checking.add_argument('--manifest-sha256', required=True, help='SHA256 received through an external trusted channel')
    checking.add_argument('--integrity-only', action='store_true',
                          help='Stdlib-only pre-install integrity; skips target/wheel/runtime qualification')
    running = sub.add_parser('run', help='Run one messages JSON list under the bundled process-group watchdog')
    running.add_argument('bundle')
    running.add_argument('--messages', required=True)
    running.add_argument('--output', required=True)
    running.add_argument('--manifest-sha256', required=True, help='SHA256 received through an external trusted channel')
    worker = sub.add_parser('_worker', help='Internal isolated process-group worker; requires guard startup gate')
    worker.add_argument('bundle')
    worker.add_argument('--manifest-sha256', required=True)
    worker.add_argument('--plan', required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['--verify']:
        arguments[0] = 'verify'
    args = parser.parse_args(arguments)
    try:
        if args.command == '_worker':
            return guarded_worker(args.bundle, args.manifest_sha256, args.plan)
        if args.command == 'dependency-specs':
            require_target()
            packages = dependency_specs()
            print(json.dumps(packages) if args.json else '\n'.join(f"{p['name']}=={p['version']}" for p in packages))
            return 0
        if args.command == 'pack':
            result = pack(args.model, json.loads(Path(args.profile).read_bytes()), args.wheelhouse,
                          args.output, args.source_repo, args.source_revision, args.adapter,
                          args.license_file, args.license_source, args.source_card)
        elif args.command == 'verify':
            manifest = verify(args.bundle, args.manifest_sha256, check_environment=not args.integrity_only)
            result = {'verified': True, 'external_manifest_sha256_matched': True, 'schema': manifest['schema'],
                      'verification_mode': 'integrity_only' if args.integrity_only else 'integrity_and_target',
                      'target_and_wheel_compatibility_checked': not args.integrity_only,
                      'model_runtime_qualified': False,
                      'manifest_sha256': file_pin(Path(args.bundle) / 'manifest.json')['sha256']}
        else:
            result = run(args.bundle, json.loads(Path(args.messages).read_bytes()), args.output,
                         args.manifest_sha256)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if args.command != 'run' or result['answer_complete'] else 1
    except Exception as error:
        print(json.dumps({'success': False, 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
