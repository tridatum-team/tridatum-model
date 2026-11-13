"""Artifact tests use inert files and startup-only subprocesses; no models/downloads.

These checks do not qualify a wheel installation or native generation. The parent
must separately install the exact wheelhouse into a disposable CPython 3.12
venv, relocate the bundle outside the repo, remove source paths from visibility,
and exercise the guarded CLI with real weights and network denial.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import mlx_offline as offline
from scripts import mlx_inference as inference


TARGET = {'system': 'Darwin', 'machine': 'arm64', 'python': '>=3.12,<3.13',
          'implementation': 'CPython'}
PACKAGES = [{'name': 'offline-fixture', 'version': '1.0'}]


def make_wheel(directory, version='1.0', tags='py3-none-any'):
    path = directory / f'offline_fixture-{version}-{tags}.whl'
    prefix = f'offline_fixture-{version}.dist-info'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(prefix + '/METADATA',
                         f'Metadata-Version: 2.1\nName: offline-fixture\nVersion: {version}\n')
        archive.writestr(prefix + '/WHEEL', f'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tags}\n')
        archive.writestr(prefix + '/RECORD', '')
    return path


def make_model(directory):
    directory.mkdir()
    offline.exclusive_json(directory / 'config.json',
                           {'model_type': 'qwen3', 'max_position_embeddings': 8192})
    offline.exclusive_json(directory / 'tokenizer_config.json', {})
    offline.exclusive_json(directory / 'tokenizer.json', {})
    (directory / 'model.safetensors').write_bytes(struct.pack('<Q', 2) + b'{}')


class OfflineArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'source-model'
        make_model(self.model)
        self.wheels = self.root / 'source-wheels'
        self.wheels.mkdir()
        make_wheel(self.wheels)
        self.bundle = self.root / 'bundle'
        self.profile = inference.historical_profile()

    def pack(self, adapter=None, **provenance):
        # Only machine identity / installed dependency discovery are replaced;
        # copying, wheel inspection, hashing and all bundle admission are real.
        with patch.object(offline, 'require_target', return_value=TARGET), \
                patch.object(offline, 'dependency_specs', return_value=PACKAGES):
            return offline.pack(self.model, self.profile, self.wheels, self.bundle,
                                'https://example.invalid/model', 'a' * 40, adapter, **provenance)

    def manifest(self):
        return json.loads((self.bundle / 'manifest.json').read_bytes())

    def replace_manifest(self, manifest):
        (self.bundle / 'manifest.json').write_bytes(offline.json_bytes(manifest))

    def test_real_copy_relocates_without_original_model_or_wheels(self):
        self.pack()
        expected = (self.model / 'model.safetensors').read_bytes()
        shutil.rmtree(self.model)
        shutil.rmtree(self.wheels)
        moved = self.root / 'elsewhere'
        self.bundle.rename(moved)
        manifest = offline.verify(moved, offline.file_pin(moved / 'manifest.json')['sha256'], check_environment=False)
        self.assertEqual((moved / 'model/model.safetensors').read_bytes(), expected)
        self.assertFalse(any(path.is_symlink() for path in moved.rglob('*')))
        self.assertEqual(manifest['profile'], self.profile)
        self.assertFalse(manifest['weights']['fused'])
        self.assertFalse(manifest['weights']['requantized'])
        self.assertFalse((moved / 'runtime/__pycache__').exists())
        self.assertIn('--no-index', manifest['installation']['argv'])
        self.assertIn('--require-hashes', manifest['installation']['argv'])

    def test_pack_is_exclusive(self):
        self.pack()
        with self.assertRaises(FileExistsError):
            self.pack()

    def test_source_snapshot_links_are_materialized(self):
        original = self.model / 'model.safetensors'
        blob = self.root / 'snapshot-blob'
        original.rename(blob)
        original.symlink_to(blob)
        self.pack()
        blob.unlink()
        self.assertFalse((self.bundle / 'model/model.safetensors').is_symlink())
        offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_changed_missing_and_extra_files_fail(self):
        self.pack()
        path = self.bundle / 'model/model.safetensors'
        original = path.read_bytes()
        path.write_bytes(original + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'changed'):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing='):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)
        path.write_bytes(original)
        (self.bundle / 'unlisted').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'extra='):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_manifest_path_traversal_is_rejected(self):
        self.pack()
        original = self.manifest()
        for unsafe in ('../outside', '/absolute', 'a/../../outside', 'a\\outside',
                       'a//outside', './outside', 'C:/outside'):
            with self.subTest(unsafe=unsafe):
                manifest = json.loads(json.dumps(original))
                manifest['files'][0]['path'] = unsafe
                self.replace_manifest(manifest)
                with self.assertRaisesRegex(ValueError, 'path'):
                    offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_manifest_duplicate_and_symlink_are_rejected(self):
        self.pack()
        manifest = self.manifest()
        manifest['files'].append(dict(manifest['files'][0]))
        self.replace_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)
        manifest['files'].pop()
        self.replace_manifest(manifest)
        path = self.bundle / 'profile.json'
        path.unlink()
        path.symlink_to(self.model / 'config.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_directory_symlinks_are_rejected_before_traversal(self):
        self.pack()
        (self.bundle / 'linked-directory').symlink_to(self.model, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_base_adapter_binding_is_enforced(self):
        adapter = self.root / 'source-adapter'
        adapter.mkdir()
        offline.exclusive_json(adapter / 'adapter_config.json',
                               {'fine_tune_type': 'lora', 'num_layers': 1, 'lora_parameters': {}})
        (adapter / 'adapters.safetensors').write_bytes(struct.pack('<Q', 2) + b'{}')
        self.pack(adapter)
        manifest = self.manifest()
        self.assertEqual(manifest['adapter']['base_binding_sha256'],
                         manifest['model']['binding_sha256'])
        manifest['adapter']['base_binding_sha256'] = '0' * 64
        self.replace_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Adapter/base binding'):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_unsafe_pickle_and_remote_code_are_rejected(self):
        for filename, contents in (('pytorch_model.bin', b'not safe'),
                                   ('config.json', b'{"model_type":"qwen3","auto_map":{"AutoModel":"evil.Code"}}')):
            with self.subTest(filename=filename):
                path = self.model / filename
                previous = path.read_bytes() if path.exists() else None
                path.write_bytes(contents)
                with self.assertRaisesRegex(ValueError, 'Unsafe|remote'):
                    offline.validate_model_files(self.model)
                if previous is None:
                    path.unlink()
                else:
                    path.write_bytes(previous)

    def test_incomplete_wheelhouse_is_not_qualified(self):
        with self.assertRaisesRegex(ValueError, 'missing runtime'):
            offline.wheel_inventory(self.wheels, PACKAGES + [{'name': 'absent', 'version': '1.0'}])

    def test_wrong_version_extra_and_incompatible_wheels_fail(self):
        with self.assertRaisesRegex(ValueError, 'wrong-version'):
            offline.wheel_inventory(self.wheels, [{'name': 'offline-fixture', 'version': '2.0'}])
        (self.wheels / 'notes.txt').write_text('not a wheel')
        with self.assertRaisesRegex(ValueError, 'only wheel'):
            offline.wheel_inventory(self.wheels, PACKAGES)
        (self.wheels / 'notes.txt').unlink()
        next(self.wheels.glob('*.whl')).unlink()
        make_wheel(self.wheels, tags='cp39-cp39-win_amd64')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            offline.wheel_inventory(self.wheels, PACKAGES)

    def test_target_mismatch_is_explicit(self):
        with patch.object(offline.platform, 'system', return_value='Linux'):
            with self.assertRaisesRegex(ValueError, 'macOS arm64'):
                offline.require_target()

    def test_output_cannot_be_nested_in_source(self):
        self.bundle = self.model / 'recursive'
        with self.assertRaisesRegex(ValueError, 'inside a source'):
            self.pack()
        self.assertFalse(self.bundle.exists())

    def test_profile_manifest_disagreement_is_rejected(self):
        self.pack()
        manifest = self.manifest()
        manifest['profile']['temperature'] = 0.7
        self.replace_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'profile/penalty'):
            offline.verify(self.bundle, offline.file_pin(self.bundle / 'manifest.json')['sha256'], check_environment=False)

    def test_joint_manifest_and_runtime_tampering_fails_external_anchor(self):
        receipt = self.pack()
        trusted = receipt['manifest']['sha256']
        manifest = self.manifest()
        runtime_file = self.bundle / 'runtime/mlx_inference.py'
        runtime_file.write_text("raise AssertionError('untrusted code executed')\n")
        replacement = offline.file_pin(runtime_file, 'runtime/mlx_inference.py')
        manifest['files'] = [replacement if pin['path'] == replacement['path'] else pin
                             for pin in manifest['files']]
        self.replace_manifest(manifest)
        with patch.object(offline, 'runtime_module', side_effect=AssertionError('import before trust')):
            with self.assertRaisesRegex(ValueError, 'external trust anchor'):
                offline.verify(self.bundle, trusted, check_environment=False)

    def test_external_anchor_required_before_runtime_import(self):
        self.pack()
        with patch.object(offline, 'runtime_module', side_effect=AssertionError('import before trust')):
            for anchor in (None, '', 'not-a-digest', '0' * 64):
                with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                    offline.verify(self.bundle, anchor, check_environment=False)

    def test_license_and_source_card_are_copied_and_hashed_separately(self):
        license_file = self.root / 'source-license'
        source_card = self.root / 'source-card'
        license_file.write_text('Test license fixture; not a legal Apache license.\n')
        source_card.write_text('Test model card fixture.\n')
        before = offline.tree_pins(self.model)
        receipt = self.pack(license_file=license_file, source_card=source_card,
                            license_source='https://example.invalid/official/resolve/' + 'b' * 40 + '/LICENSE')
        manifest = offline.verify(self.bundle, receipt['manifest']['sha256'], check_environment=False)
        self.assertEqual(offline.tree_pins(self.model), before)
        self.assertEqual((self.bundle / 'licenses/LICENSE').read_bytes(), license_file.read_bytes())
        self.assertEqual((self.bundle / 'licenses/SOURCE_CARD').read_bytes(), source_card.read_bytes())
        self.assertEqual(manifest['license_provenance']['license']['source'],
                         'https://example.invalid/official/resolve/' + 'b' * 40 + '/LICENSE')
        self.assertNotIn('revision', manifest['license_provenance']['license'])
        self.assertNotIn('revision', manifest['license_provenance']['source_card'])
        self.assertEqual(manifest['source']['revision'], 'a' * 40)
        self.assertEqual(manifest['license_provenance']['source_card']['source'],
                         manifest['source']['repository'])
        self.assertIn('licenses/LICENSE', {pin['path'] for pin in manifest['files']})
        self.assertIn('licenses/SOURCE_CARD', {pin['path'] for pin in manifest['files']})

    def test_launcher_rejects_unisolated_before_shadow_imports(self):
        self.pack()
        runtime = self.bundle / 'runtime'
        sentinel = self.root / 'shadow-executed'
        (runtime / 'email.py').write_text(
            f"open({str(sentinel)!r}, 'w').write('executed')\nraise RuntimeError('shadow')\n")
        result = subprocess.run(
            [sys.executable, str(runtime / 'mlx_offline.py'), '--help'],
            cwd=runtime, capture_output=True, text=True, timeout=20,
            env={**os.environ, 'PYTHONPATH': str(runtime)})
        self.assertEqual(result.returncode, 2)
        self.assertIn('invoke Python with -I', result.stderr)
        self.assertFalse(sentinel.exists())

    def test_isolated_launcher_ignores_shadow_then_rejects_extra_file(self):
        receipt = self.pack()
        runtime = self.bundle / 'runtime'
        sentinel = self.root / 'shadow-executed'
        (runtime / 'email.py').write_text(
            f"open({str(sentinel)!r}, 'w').write('executed')\nraise RuntimeError('shadow')\n")
        environment = {**os.environ, 'PYTHONPATH': str(runtime)}
        help_result = subprocess.run(
            [sys.executable, '-I', str(runtime / 'mlx_offline.py'), '--help'],
            cwd=runtime, env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertFalse(sentinel.exists())
        # Skip platform admission only to exercise this real isolated-process
        # integrity boundary on non-macOS test hosts as well.
        admission = (
            'import importlib.util,sys; '
            'spec=importlib.util.spec_from_file_location("trusted_launcher",sys.argv[1]); '
            'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
            'module.verify(sys.argv[2],sys.argv[3],check_environment=False)'
        )
        checked = subprocess.run(
            [sys.executable, '-I', '-c', admission, str(runtime / 'mlx_offline.py'),
             str(self.bundle), receipt['manifest']['sha256']],
            cwd=runtime, env=environment, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn('Bundle file coverage mismatch', checked.stderr)
        self.assertFalse(sentinel.exists())

    def test_preinstall_integrity_succeeds_without_site_packages(self):
        receipt = self.pack()
        launcher = self.bundle / 'runtime/mlx_offline.py'
        command = [sys.executable, '-I', '-S', str(launcher), 'verify', str(self.bundle),
                   '--manifest-sha256', receipt['manifest']['sha256']]
        result = subprocess.run(command + ['--integrity-only'], cwd=self.root,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertTrue(status['verified'])
        self.assertEqual(status['verification_mode'], 'integrity_only')
        self.assertFalse(status['target_and_wheel_compatibility_checked'])
        self.assertFalse(status['model_runtime_qualified'])
        # Full verification must still enforce target/wheel requirements. On
        # a compatible target -S deliberately makes packaging unavailable.
        full = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(full.returncode, 0)
        self.assertFalse(json.loads(full.stderr)['success'])

    def test_preinstall_integrity_rejects_changed_lock_or_wheel(self):
        receipt = self.pack()
        command = [sys.executable, '-I', '-S', str(self.bundle / 'runtime/mlx_offline.py'),
                   'verify', str(self.bundle), '--manifest-sha256',
                   receipt['manifest']['sha256'], '--integrity-only']
        for path in (self.bundle / 'requirements.lock', next((self.bundle / 'wheelhouse').glob('*.whl'))):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b'changed-after-transfer')
                    result = subprocess.run(command, cwd=self.root, capture_output=True,
                                            text=True, timeout=20)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Bundle file changed', result.stderr)
                finally:
                    path.write_bytes(original)


class OfflineAdmissionTests(unittest.TestCase):
    def test_invalid_messages_fail_without_native_import(self):
        for messages in ([], {}, [{'role': 'user', 'content': ''}],
                         [{'role': 'assistant', 'content': 'not a new request'}],
                         [{'role': 'user', 'content': [], 'images': []}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                offline.validate_messages(messages)

    def test_network_connections_are_denied_not_merely_hinted_by_environment(self):
        # Keep all mutations scoped; no socket connects to an actual address.
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=False))
            for name in ('create_connection', 'getaddrinfo', 'gethostbyname',
                         'gethostbyname_ex', 'gethostbyaddr'):
                stack.enter_context(patch.object(socket, name))
            for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
                if hasattr(socket.socket, name):
                    stack.enter_context(patch.object(socket.socket, name))
            inference.configure_offline()
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')
            self.assertEqual(os.environ['USE_TORCH'], '0')
            with self.assertRaisesRegex(RuntimeError, 'Network access forbidden'):
                socket.create_connection(('example.invalid', 443))
            with self.assertRaisesRegex(RuntimeError, 'Network access forbidden'):
                socket.getaddrinfo('example.invalid', 443)
            with socket.socket() as sock:
                with self.assertRaisesRegex(RuntimeError, 'Network access forbidden'):
                    sock.connect(('127.0.0.1', 9))
                with self.assertRaisesRegex(RuntimeError, 'Network access forbidden'):
                    sock.connect_ex(('127.0.0.1', 9))
                with self.assertRaisesRegex(RuntimeError, 'Network access forbidden'):
                    sock.sendto(b'blocked', ('127.0.0.1', 9))

    def test_remote_model_identifier_fails_before_native_loader(self):
        with self.assertRaisesRegex(RuntimeError, 'absolute local model'):
            inference.load_local('organization/model')

    def test_adapter_requires_complete_keys_and_matching_shapes(self):
        # Only MLX's tree traversal boundary is replaced; coverage validation is
        # the actual bundled implementation and cannot accept a partial adapter.
        utils = ModuleType('mlx.utils')
        utils.tree_flatten = lambda parameters: parameters
        model = SimpleNamespace(trainable_parameters=lambda: [
            ('layers.0.q.lora_a', SimpleNamespace(shape=(4, 2))),
            ('layers.0.q.lora_b', SimpleNamespace(shape=(2, 4))),
        ])
        good = {'layers.0.q.lora_a': SimpleNamespace(shape=(4, 2)),
                'layers.0.q.lora_b': SimpleNamespace(shape=(2, 4))}
        with patch.dict(sys.modules, {'mlx.utils': utils}):
            inference.validate_adapter_tensors(model, good)
            for invalid in ({}, {'layers.0.q.lora_a': good['layers.0.q.lora_a']},
                            {**good, 'unexpected': SimpleNamespace(shape=(1,))}):
                with self.subTest(keys=list(invalid)):
                    with self.assertRaisesRegex(RuntimeError, 'exact coverage'):
                        inference.validate_adapter_tensors(model, invalid)
            with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
                inference.validate_adapter_tensors(
                    model, {**good, 'layers.0.q.lora_a': SimpleNamespace(shape=(2, 4))})


if __name__ == '__main__':
    unittest.main()
