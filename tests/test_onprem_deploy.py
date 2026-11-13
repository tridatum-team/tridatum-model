"""Content checks for deploy/onprem (images are built on a Docker machine)."""
import ast
import json
import re
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / 'deploy' / 'onprem'
MODEL_SHA = '061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a'
TOK_SHA = '73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d'


def read(name):
    return (DEPLOY / name).read_text(encoding='utf-8')


def copied_trid_files():
    """Files under trid/ that Dockerfile.app copies into the image."""
    files = []
    for line in re.findall(r'^COPY\s+(.+?)(?<!\\)\n(?!\s)', read('Dockerfile.app') + '\n', re.M | re.S):
        parts = line.replace('\\\n', ' ').split()
        sources = [p for p in parts[:-1] if p.startswith('trid/')]
        for src in sources:
            path = ROOT / src
            if src.endswith('/'):
                files.extend(p for p in path.rglob('*.py'))
            else:
                files.append(path)
    return files


@unittest.skipIf(yaml is None, 'PyYAML not installed')
class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.compose = yaml.safe_load(read('compose.yaml'))

    def test_compose_llm_and_app_only_on_internal_network(self):
        services = self.compose['services']
        self.assertTrue(self.compose['networks']['internal']['internal'])
        for name in ('llm', 'app'):
            self.assertEqual(list(services[name]['networks']), ['internal'], name)
        self.assertEqual(sorted(services['tls']['networks']), ['edge', 'internal'])

    def test_only_tls_publishes_port_443(self):
        services = self.compose['services']
        for name, service in services.items():
            if name == 'tls':
                self.assertEqual(service['ports'], ['443:443'])
            else:
                self.assertNotIn('ports', service, name)

    def test_restart_and_volumes(self):
        services = self.compose['services']
        for service in services.values():
            self.assertEqual(service['restart'], 'unless-stopped')
        self.assertIn('mem_limit', services['app'])
        self.assertIn('./data:/data', services['app']['volumes'])
        self.assertIn('./certs:/certs:ro', services['tls']['volumes'])

    def test_images_pinned_by_version(self):
        self.assertRegex(self.compose['services']['tls']['image'], r'^caddy:\d+\.\d+\.\d+$')


class DockerfileTests(unittest.TestCase):
    def test_dockerfile_llm_disables_native_and_pins_commit(self):
        text = read('Dockerfile.llm')
        for flag in ('-DGGML_NATIVE=OFF', '-DGGML_AVX2=ON', '-DGGML_CUDA=OFF',
                     '-DGGML_METAL=OFF', '-DLLAMA_CURL=OFF'):
            self.assertIn(flag, text)
        self.assertIn('8ca4060b7e30e76baaa5670ee1ec1b760b7f8552', text)
        self.assertIn('git rev-parse HEAD', text)
        self.assertIn('--no-webui', text)
        self.assertIn('-c 4096', text)

    def test_dockerfile_llm_checks_gguf_sha256(self):
        text = read('Dockerfile.llm')
        self.assertIn(MODEL_SHA, text)
        self.assertIn('sha256sum -c', text)

    def test_app_image_checks_tokenizer_and_isolated_imports(self):
        text = read('Dockerfile.app')
        self.assertIn(TOK_SHA, text)
        self.assertIn('/opt/trid/tokenizer.json', text)
        self.assertIn('python -I -c "import trid.pdf_extract_worker, trid.onprem.server, trid.onprem.__main__"', text)
        self.assertIn('python -I -c "import torch"', text)
        self.assertIn('--no-index --no-deps --require-hashes', text)
        self.assertIn('"python", "-I", "-m", "trid.onprem"', text)
        self.assertIn('trid.onprem.__main__', text)
        self.assertNotIn('WORKDIR /data', text)
        self.assertRegex(text, r'(?m)^FROM python:3\.12\.\d+-slim-bookworm$')
        self.assertRegex(text, r'(?m)^USER 10001')
        self.assertIn('"serve", "--config", "/data/onprem.json"', text)

    def test_app_image_copies_no_torch_modules(self):
        files = copied_trid_files()
        names = {p.name for p in files}
        for needed in ('pdf_documents.py', 'pdf_isolated.py', 'pdf_extract_worker.py',
                       'document_case.py', 'llama_client.py', 'prompts.py',
                       'case_prompts.py', 'server.py', '__main__.py'):
            self.assertIn(needed, names)
        self.assertNotIn('short_reader.py', names)
        copied_modules = set()
        for path in files:
            rel = path.relative_to(ROOT).with_suffix('')
            parts = list(rel.parts)
            if parts[-1] == '__init__':
                parts.pop()
            copied_modules.add('.'.join(parts))
        for path in files:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or '']
                for module in modules:
                    root = module.split('.')[0]
                    self.assertNotIn(root, ('torch', 'numpy', 'safetensors'), f'{path.name}: {module}')
            # Module-level imports of the trid package must stay inside the copy.
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or '').startswith('trid'):
                    self.assertIn(node.module, copied_modules, f'{path.name}: {node.module}')


class RequirementsTests(unittest.TestCase):
    def test_app_requirements_two_pins_with_hashes(self):
        lines = [l for l in read('requirements-app.txt').splitlines() if l.strip() and not l.startswith('#')]
        self.assertEqual(len(lines), 2)
        pins = sorted(re.match(r'^(\S+==\S+) --hash=sha256:[0-9a-f]{64}$', l).group(1) for l in lines)
        self.assertEqual(pins, ['pypdf==6.21.2', 'tokenizers==0.23.3'])


class CaddyTests(unittest.TestCase):
    def test_caddyfile_settings(self):
        text = read('Caddyfile')
        for expected in ('admin off', 'auto_https off', '{$TRID_HOSTNAME}',
                         '/certs/cert.pem /certs/key.pem', 'reverse_proxy app:8080',
                         'header_up X-Real-IP {remote_host}', 'header_up -X-Forwarded-For',
                         'max_size {$TRID_MAX_BODY:3MB}', 'ocsp_stapling off', 'protocols h1 h2'):
            self.assertIn(expected, text)


class BundleTests(unittest.TestCase):
    def test_example_config_points_at_internal_llm(self):
        config = json.loads(read('onprem.example.json'))
        self.assertEqual(config['llm_base'], 'http://llm:8080')
        self.assertEqual(config['tokenizer_path'], '/opt/trid/tokenizer.json')

    def test_build_bundle_and_licenses(self):
        script = read('build_bundle.sh')
        for expected in (MODEL_SHA, TOK_SHA, 'docker save', 'SHA256SUMS', 'IMAGES.txt',
                         'RepoDigests', 'docker pull', 'base $base', '--no-deps --require-hashes', 'make_self_signed.sh', 'onprem.example.json'):
            self.assertIn(expected, script)
        licenses = read('LICENSES.md')
        for name in ('Qwen3', 'llama.cpp', 'Caddy', 'tokenizers', 'pypdf', 'PSF'):
            self.assertIn(name, licenses)

    def test_self_signed_script(self):
        text = read('make_self_signed.sh')
        self.assertIn('-days 825', text)
        self.assertIn('chmod 600', text)
        self.assertNotIn('chmod 644', text)
        self.assertIn('subjectAltName', text)


if __name__ == '__main__':
    unittest.main()
