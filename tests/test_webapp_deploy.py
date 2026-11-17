"""Content checks for deploy/onprem_web (images are built on a Docker machine); ports test_onprem_deploy.py."""
import hashlib
import json
import re
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / 'deploy' / 'onprem_web'
OLD = ROOT / 'deploy' / 'onprem'
TOK_SHA = '73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d'
HTMX = ROOT / 'onprem_web' / 'static' / 'vendor' / 'htmx.min.js'


def read(name, base=DEPLOY):
    return (base / name).read_text(encoding='utf-8')


def requirement_lines():
    return [l for l in read('requirements-app.txt').splitlines() if l.strip() and not l.startswith('#')]


class RequirementsTests(unittest.TestCase):
    def test_requirements_all_hashed_and_no_torch_numpy(self):
        pins = []
        for line in requirement_lines():
            m = re.fullmatch(r'(\S+)==(\S+) --hash=sha256:[0-9a-f]{64}', line)
            self.assertIsNotNone(m, line)
            pins.append(m.group(1).lower())
        self.assertEqual(sorted(pins), ['asgiref', 'django', 'pypdf', 'sqlparse', 'tokenizers', 'waitress'])
        text = read('requirements-app.txt').lower()
        for banned in ('torch', 'numpy', 'safetensors'):
            self.assertNotIn(banned, text)
        old = {l.split(' ')[0] for l in read('requirements-app.txt', OLD).splitlines() if '==' in l}
        new = {l.split(' ')[0] for l in requirement_lines()}
        self.assertTrue(old <= new, 'tokenizers/pypdf pins must match the old version')

    def test_licenses_list_every_requirement(self):
        licenses = read('LICENSES.md')
        for line in requirement_lines():
            name = line.split('==')[0]
            self.assertRegex(licenses, rf'(?im)^\| {re.escape(name)}\b', name)
        for name, lic in (('Django', 'BSD-3-Clause'), ('asgiref', 'BSD-3-Clause'), ('sqlparse', 'BSD-3-Clause'),
                          ('waitress', 'ZPL-2.1'), ('htmx', '0BSD')):
            self.assertRegex(licenses, rf'(?m)^\| {name}\b.*\| {re.escape(lic)} \|$', name)
        self.assertIn(hashlib.sha256(HTMX.read_bytes()).hexdigest(), licenses)
        self.assertIn('2.0.4', licenses)
        for row in re.findall(r'(?m)^\| (Qwen3|llama\.cpp|Caddy|Pretendard|Python)\b', read('LICENSES.md', OLD)):
            self.assertIn(f'| {row}', licenses)


@unittest.skipIf(yaml is None, 'PyYAML not installed')
class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.compose = yaml.safe_load(read('compose.yaml'))

    def test_compose_only_tls_publishes_and_app_read_only_cap_drop(self):
        services = self.compose['services']
        self.assertTrue(self.compose['networks']['internal']['internal'])
        for name, service in services.items():
            if name == 'tls':
                self.assertEqual(service['ports'], ['443:443'])
            else:
                self.assertNotIn('ports', service, name)
                self.assertEqual(list(service['networks']), ['internal'], name)
            self.assertTrue(service['read_only'], name)
            self.assertEqual(service['cap_drop'], ['ALL'], name)
            self.assertEqual(service['restart'], 'unless-stopped', name)
            self.assertEqual(service['logging'], {'driver': 'json-file',
                                                  'options': {'max-size': '10m', 'max-file': '3'}}, name)
        self.assertIn('./data:/data', services['app']['volumes'])
        self.assertIn('mem_limit', services['app'])
        self.assertIn('/tmp', services['app']['tmpfs'])
        self.assertIn('TRID_MAX_BODY', services['tls']['environment'])
        self.assertIn('../onprem/Caddyfile:/etc/caddy/Caddyfile:ro', services['tls']['volumes'])
        self.assertEqual(services['tls']['image'], yaml.safe_load(read('compose.yaml', OLD))['services']['tls']['image'])

    def test_compose_project_and_image_names_differ_from_old(self):
        old = yaml.safe_load(read('compose.yaml', OLD))
        self.assertEqual(self.compose['name'], 'trid-onprem-web')
        self.assertNotEqual(self.compose['name'], old['name'])
        self.assertEqual(self.compose['services']['app']['image'], 'trid-onprem-web-app:latest')
        self.assertNotEqual(self.compose['services']['app']['image'], old['services']['app']['image'])
        self.assertEqual(self.compose['services']['llm']['image'], 'trid-onprem-llm:latest')


class DockerfileTests(unittest.TestCase):
    def test_dockerfile_runs_isolated_as_10001_and_checks_tokenizer(self):
        text = read('Dockerfile.app')
        self.assertRegex(text, r'(?m)^FROM python:3\.12\.\d+-slim-bookworm$')
        self.assertIn('COPY deploy/onprem_web/wheels /tmp/wheels', text)
        self.assertIn('COPY deploy/onprem_web/requirements-app.txt /tmp/requirements-app.txt', text)
        self.assertIn('--no-index --no-deps --require-hashes', text)
        self.assertIn('COPY onprem_web/', text)
        self.assertIn('COPY trid/onprem/', text)
        self.assertIn('COPY trid/onprem_ui/fonts/', text)
        self.assertIn(TOK_SHA, text)
        self.assertIn('sha256sum -c', text)
        self.assertIn('python -I -c "import torch"', text)
        self.assertIn('python -I -c "import numpy"', text)
        self.assertRegex(text, r'python -I -c "import [^"]*\bonprem_web\.__main__"')
        self.assertRegex(text, r'(?m)^USER 10001:10001$')
        self.assertRegex(text, r'(?m)^WORKDIR /$')
        self.assertTrue(text.rstrip().endswith(
            'CMD ["python", "-I", "-m", "onprem_web", "serve", "--config", "/data/onprem.json"]'))

    def test_dockerfile_copies_every_onprem_web_import(self):
        """Every trid module that onprem_web imports at module level is copied into the image."""
        text = read('Dockerfile.app') + read('Dockerfile.app', OLD)
        for path in (ROOT / 'onprem_web').rglob('*.py'):
            for module in re.findall(r'(?m)^from (trid(?:\.\w+)*) import', path.read_text(encoding='utf-8')):
                rel = module.replace('.', '/')
                self.assertTrue(f'{rel}.py' in text or f'{rel}/' in text or rel.rsplit('/', 1)[0] + '/' in text,
                                f'{path.name}: {module}')

    def test_dockerignore_allowlist_includes_onprem_web(self):
        lines = read('.dockerignore', ROOT).splitlines()
        for entry in ('!onprem_web/', '!deploy/onprem_web/wheels/', '!deploy/onprem_web/requirements-app.txt'):
            self.assertIn(entry, lines)
        old_entries = ['!trid/__init__.py', '!trid/onprem/', '!trid/onprem_ui/', '!deploy/onprem/wheels/',
                       '!deploy/onprem/requirements-app.txt', '!runs/reader-factorial-01-cpt-v1-v2-v4/tokenizer.json',
                       '!runs/open-weight-base/Qwen3-1.7B-Q8_0.gguf']
        for entry in old_entries:
            self.assertIn(entry, lines)

    def test_example_config_has_old_keys(self):
        self.assertEqual(set(json.loads(read('onprem.example.json'))),
                         set(json.loads(read('onprem.example.json', OLD))))


class GuideTests(unittest.TestCase):
    def test_guide_mentions_single_version_per_server_and_create_admin(self):
        text = (ROOT / 'docs' / 'ONPREM_WEB.md').read_text(encoding='utf-8')
        for needle in ('한 서버에는 한 판만 설치', 'python -I -m onprem_web create-admin', '--password-stdin',
                       'data/web.db', 'web.db-wal', 'certs/cert.pem', '마이그레이션',
                       "unittest discover -s tests -p 'test_webapp_*.py'", 'release_check.py'):
            self.assertIn(needle, text)

    def test_bundle_script_uses_new_paths(self):
        text = read('build_bundle.sh')
        self.assertIn('deploy/onprem_web/wheels', text)
        self.assertIn('-r deploy/onprem_web/requirements-app.txt', text)
        self.assertIn('-f deploy/onprem_web/Dockerfile.app -t trid-onprem-web-app:latest', text)
        self.assertIn('docs/ONPREM_WEB.md', text)
        self.assertIn('--require-hashes', text)
