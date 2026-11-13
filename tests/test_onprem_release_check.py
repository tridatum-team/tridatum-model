"""release_check.py against a fake in-process server, plus release material checks."""
import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / 'deploy' / 'onprem'
SECRET_ANSWER = 'SECRET-ANSWER-TEXT-zqx'


def load_module():
    spec = importlib.util.spec_from_file_location('release_check', DEPLOY / 'release_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_fake(login_status='ok', fail_question=None, model_failure_question=None):
    state = {'calls': [], 'asked': 0}
    base_holder = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, status, value, mime='application/json', cookie=False):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            if cookie:
                self.send_header('Set-Cookie', '__Host-trid=abc123; Path=/; HttpOnly; SameSite=Strict; Secure')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state['calls'].append(('GET', self.path))
            if self.headers.get('Cookie') != '__Host-trid=abc123':
                self.send(401, {'error': 'x'}); return
            page = ('<meta name="csrf-token" content="tok-1"><meta name="must-change" content="'
                    + ('1' if login_status == 'must_change' else '0') + '">')
            self.send(200, page.encode(), 'text/html')

        def do_POST(self):
            size = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(size)
            state['calls'].append(('POST', self.path))
            if self.headers.get('Origin') != base_holder['base']:
                self.send(403, {'error': 'origin'}); return
            if self.path == '/api/login':
                data = json.loads(body)
                if data['password'] != 'pw-correct-1':
                    self.send(401, {'error': '아이디 또는 비밀번호가 올바르지 않습니다.'}); return
                self.send(200, {'status': login_status}, cookie=True); return
            if self.headers.get('X-Trid-Token') != 'tok-1':
                self.send(403, {'error': 'csrf'}); return
            if self.path == '/api/load':
                state['pdf'] = body
                state['load_headers'] = dict(self.headers)
                self.send(200, {'document_id': 'd1', 'name': 'x.pdf', 'pages': ['a', 'b', 'c'], 'warnings': []}); return
            if self.path == '/api/ask':
                n = state['asked'] = state['asked'] + 1
                if fail_question == n:
                    self.send(502, {'error': '모델 서버에 연결할 수 없습니다. 관리자에게 문의해 주세요.'}); return
                if model_failure_question == n:
                    self.send(200, {'status': 'model_failure', 'answer': SECRET_ANSWER, 'candidates': []}); return
                cands = [{'page': 2, 'text': 't', 'prediction': {'source_character_span': [0, 1]}},
                         {'page': 3, 'text': 't', 'prediction': {'source_character_span': [0, 0]}}]
                self.send(200, {'status': 'quoted', 'answer': SECRET_ANSWER, 'candidates': cands,
                                'commentary': SECRET_ANSWER if n == 1 else None}); return
            if self.path == '/api/logout':
                self.send(200, {'ok': True}); return
            self.send(404, {'error': 'x'})

    server = HTTPServer(('127.0.0.1', 0), Handler)
    base_holder['base'] = f'http://127.0.0.1:{server.server_address[1]}'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, base_holder['base'], state


class ReleaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.pdf = tmp / 'doc.pdf'
        self.pdf.write_bytes(b'%PDF-fake')
        self.questions = tmp / 'q.json'
        self.questions.write_text(json.dumps([
            {'id': 'q1', 'kind': 'answerable', 'question': '질문 하나?'},
            {'id': 'q2', 'kind': 'numeric', 'question': '질문 둘?'}], ensure_ascii=False), encoding='utf-8')
        self.out = tmp / 'report.json'
        self.mod = load_module()

    def run_check(self, base, password='pw-correct-1'):
        argv = ['--base', base, '--user', 'admin', '--pdf', str(self.pdf),
                '--questions', str(self.questions), '--out', str(self.out)]
        with mock.patch.object(self.mod.getpass, 'getpass', return_value=password):
            return self.mod.main(argv)

    def start(self, **kw):
        server, base, state = start_fake(**kw)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return base, state

    def test_release_check_writes_one_row_per_question(self):
        base, state = self.start()
        self.assertEqual(self.run_check(base), 0)
        report = json.loads(self.out.read_text(encoding='utf-8'))
        self.assertEqual([r['id'] for r in report['questions']], ['q1', 'q2'])
        first = report['questions'][0]
        self.assertEqual(first['status'], 'quoted')
        self.assertEqual(first['quotes'], 1)
        self.assertEqual(first['pages'], [2])
        self.assertTrue(first['has_commentary'])
        self.assertFalse(report['questions'][1]['has_commentary'])
        self.assertIsInstance(first['seconds'], float)
        self.assertEqual(report['upload']['pages'], 3)
        self.assertEqual(state['pdf'], b'%PDF-fake')
        self.assertEqual(state['calls'][-1], ('POST', '/api/logout'))

    def test_file_name_header_is_not_sent(self):
        base, state = self.start()
        self.assertEqual(self.run_check(base), 0)
        self.assertNotIn('X-File-Name', state['load_headers'])

    def test_default_port_is_dropped_from_origin(self):
        for base, origin in (('https://host.example:443', 'https://host.example'),
                             ('http://host.example:80', 'http://host.example'),
                             ('https://host.example:8443', 'https://host.example:8443'),
                             ('https://host.example', 'https://host.example')):
            self.assertEqual(self.mod.Client(base, False, None).origin, origin)

    def test_report_never_contains_answer_or_question_text(self):
        base, _ = self.start()
        self.run_check(base)
        text = self.out.read_text(encoding='utf-8')
        self.assertNotIn(SECRET_ANSWER, text)
        self.assertNotIn('질문 하나', text)

    def test_must_change_fails_with_message_and_no_report(self):
        base, state = self.start(login_status='must_change')
        with mock.patch('sys.stderr') as err:
            self.assertEqual(self.run_check(base), 1)
        written = ''.join(c.args[0] for c in err.write.call_args_list)
        self.assertIn('먼저 웹에서 비밀번호를 변경하세요', written)
        self.assertFalse(self.out.exists())
        self.assertNotIn(('POST', '/api/load'), state['calls'])
        self.assertEqual(state['calls'][-1], ('POST', '/api/logout'))

    def test_wrong_password_fails(self):
        base, _ = self.start()
        self.assertEqual(self.run_check(base, password='nope'), 1)
        self.assertFalse(self.out.exists())

    def test_failed_question_is_recorded_and_run_continues(self):
        base, state = self.start(fail_question=1)
        self.assertEqual(self.run_check(base), 1)
        report = json.loads(self.out.read_text(encoding='utf-8'))
        self.assertEqual(report['questions'][0]['status'], 'http_502')
        self.assertEqual(report['questions'][1]['status'], 'quoted')
        self.assertEqual(state['calls'][-1], ('POST', '/api/logout'))

    def test_model_failure_counts_as_error(self):
        base, _ = self.start(model_failure_question=2)
        self.assertEqual(self.run_check(base), 1)
        report = json.loads(self.out.read_text(encoding='utf-8'))
        self.assertEqual(report['questions'][1]['status'], 'model_failure')
        self.assertEqual(report['summary']['errors'], 1)
        self.assertEqual(report['summary']['status_counts'], {'model_failure': 1, 'quoted': 1})

    def test_password_is_not_a_command_line_option(self):
        text = (DEPLOY / 'release_check.py').read_text(encoding='utf-8')
        self.assertNotIn('--password', text)
        self.assertIn('getpass', text)


class ReleaseMaterialTests(unittest.TestCase):
    def test_release_document_is_synthetic_demo_material(self):
        text = (DEPLOY / 'release-document.txt').read_text(encoding='utf-8')
        self.assertTrue(text.startswith('가상 자료 · 실제 기관 규정 아님'))

    def test_release_questions_shape(self):
        items = json.loads((DEPLOY / 'release-questions.json').read_text(encoding='utf-8'))
        self.assertTrue(6 <= len(items) <= 8)
        self.assertEqual({i['kind'] for i in items}, {'answerable', 'not_in_document', 'numeric'})
        self.assertEqual(len({i['id'] for i in items}), len(items))

    def test_build_bundle_requires_onprem_guide_and_ships_release_material(self):
        script = (DEPLOY / 'build_bundle.sh').read_text(encoding='utf-8')
        self.assertIn('docs/ONPREM.md', script)
        self.assertNotIn('warning: docs/ONPREM.md', script)
        self.assertNotIn('if [ -f docs/ONPREM.md ]', script)
        for name in ('release_check.py', 'release-questions.json', 'release-document.txt'):
            self.assertIn(name, script)
        self.assertTrue((ROOT / 'docs' / 'ONPREM.md').is_file())


if __name__ == '__main__':
    unittest.main()
