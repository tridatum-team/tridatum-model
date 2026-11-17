"""The Django build answers like the stdlib build, and the shared release check runs against it unmodified.

Both servers run in this process over real HTTP with the same fake answer and parse_pdf, so a
difference here is a difference in the new code (fix the new version, never the old one).
"""

import contextlib
import http.client
import io
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import webapp_support as S
from test_onprem_release_check import SECRET_ANSWER, load_module

from django.contrib.sessions.models import Session
from django.core.wsgi import get_wsgi_application
from waitress import create_server

from onprem_web.accounts import services as accounts
from onprem_web.accounts.models import User
from onprem_web.audit import services as audit
from trid.document_case import CONTEXT_OVERFLOW
from trid.onprem.auth import Auth
from trid.onprem.config import Config
from trid.onprem.server import make_server
from trid.onprem.store import Store
from trid.pdf_documents import Chunk, Document, DocumentError

HOST, ORIGIN, PW = S.HOST, S.ORIGIN, S.PW
USER, ADMIN = 'parity-user', 'parity-admin'
PDF = b'%PDF-1.4 parity'
PAGE_TEXT = 'PARITY-PAGE-TEXT-qzv 본문 한 쪽'
DOWN, OVERFLOW = 'MODEL-DOWN?', 'OVERFLOW?'


def fake_parse(body, name, tok, limits=None):
    if not body.startswith(b'%PDF'):
        raise DocumentError('PDF를 읽지 못했습니다.')
    pages = [PAGE_TEXT, '둘째 쪽 내용', '셋째 쪽 내용']
    return Document(name, pages, [Chunk(i + 1, 0, len(p), p) for i, p in enumerate(pages)], [])


def fake_answer(document, question):
    if question == DOWN:
        raise OSError('connection refused')
    if question == OVERFLOW:
        raise DocumentError(CONTEXT_OVERFLOW)
    cands = [{'page': 2, 'text': '둘째 쪽 내용', 'source_label': '2쪽',
              'prediction': {'source_character_span': [0, 2]}},
             {'page': 3, 'text': '셋째 쪽 내용', 'prediction': {'source_character_span': [0, 0]}}]
    return {'status': 'quoted', 'answer': SECRET_ANSWER, 'candidates': cands, 'commentary': '해설',
            'context_limits': {'table_rows': 2}, 'checked': 3, 'raw_output': 'RAW-OUTPUT',
            'checked_chunks': 3, 'total_chunks': 3, 'model': 'fake'}


class Http:
    """A browser-like client: real HTTP to 127.0.0.1 with Host/Origin of the configured hostname."""

    def __init__(self, port):
        self.port = port
        self.cookie = ''
        self.token = ''

    def request(self, method, path, body=None, origin=ORIGIN, token=True, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        h = {'Host': HOST, 'X-Real-IP': '10.0.0.9', **(headers or {})}
        if method == 'POST' and origin:
            h['Origin'] = origin
        if self.cookie:
            h['Cookie'] = self.cookie
        if token and self.token:
            h['X-Trid-Token'] = self.token
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            h['Content-Type'] = 'application/json'
        if method == 'POST':
            body = body if body is not None else b''
            h['Content-Length'] = str(len(body))
        try:
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            data = r.read()
            return r.status, r, data
        finally:
            conn.close()

    def json(self, method, path, body=None, **kw):
        status, r, data = self.request(method, path, body, **kw)
        try:
            value = json.loads(data)
        except ValueError:
            value = data
        return status, value, r

    def login(self, username=USER, password=PW):
        status, value, r = self.json('POST', '/api/login', {'username': username, 'password': password})
        if status == 200:
            self.cookie = '; '.join(h.split(';', 1)[0] for h in r.msg.get_all('Set-Cookie') or [])
            page = self.request('GET', '/')[2].decode('utf-8')
            self.token = re.search(r'<meta name="csrf-token" content="([^"]*)"', page).group(1)
        return status, value

    def load(self):
        return self.json('POST', '/api/load', PDF, headers={'Content-Type': 'application/pdf'})[:2]

    def ask(self, question, document_id):
        return self.json('POST', '/api/ask', {'question': question, 'document_id': document_id})[:2]


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches = [mock.patch('trid.onprem.server.parse_pdf', side_effect=fake_parse),
                       mock.patch('onprem_web.workspace.services.parse_pdf', side_effect=fake_parse)]
        for p in cls.patches:
            p.start()
        # Old build
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = Store(Path(cls.tmp.name) / 'onprem.db')
        for name, role in ((USER, 'user'), (ADMIN, 'admin')):
            cls.store.create_user(name, PW, role, must_change=False)
        cls.old = make_server(Config(hostname=HOST, port=0), cls.store, Auth(cls.store), object(),
                              answer=fake_answer)
        threading.Thread(target=cls.old.serve_forever, daemon=True).start()
        # New build: committed rows (other threads read them), removed in tearDownClass
        for name, role in ((USER, 'user'), (ADMIN, 'admin')):
            accounts.create_user(name, PW, role, must_change=False)
        cls.new = create_server(get_wsgi_application(), host='127.0.0.1', port=0, threads=4,
                                ident=None, clear_untrusted_proxy_headers=True)
        cls.stop_new = staticmethod(S.start_waitress(cls.new))

    @classmethod
    def tearDownClass(cls):
        cls.old.shutdown()
        cls.old.server_close()
        cls.store.close()
        cls.tmp.cleanup()
        cls.stop_new()
        for p in cls.patches:
            p.stop()
        Session.objects.all().delete()
        User.objects.filter(username__in=(USER, ADMIN)).delete()
        audit.purge_before(float('inf'))  # leave no committed audit rows for later test modules

    def setUp(self):
        S.make_runtime(answer=fake_answer)  # fresh registry and per-minute counters for each test

    def clients(self):
        return Http(self.old.server_address[1]), Http(self.new.effective_port)

    def both(self, action):
        """Run `action(client)` on old and new; returns (old_result, new_result)."""
        return tuple(action(c) for c in self.clients())

    def test_same_fake_answer_same_json(self):
        def flow(c):
            self.assertEqual(c.login()[0], 200)
            status, loaded = c.load()
            self.assertEqual(status, 200, loaded)
            document_id = loaded.pop('document_id')
            return loaded, c.ask('질문입니다?', document_id)
        (old_loaded, old_ask), (new_loaded, new_ask) = self.both(flow)
        self.assertEqual(new_loaded, old_loaded)
        self.assertEqual(new_ask, old_ask)
        self.assertEqual(old_ask[0], 200)
        for key in ('checked', 'raw_output'):
            self.assertNotIn(key, old_ask[1])
            self.assertNotIn(key, new_ask[1])

    def test_same_errors_same_json(self):
        def flow(c):
            out = {'bad_login': c.login(password='wrong password 9')}
            self.assertEqual(c.login()[0], 200)
            document_id = c.load()[1]['document_id']
            out['too_long'] = c.ask('가' * 2001, document_id)
            out['wrong_id'] = c.ask('질문?', 'not-the-id')
            out['model_down'] = c.ask(DOWN, document_id)
            out['overflow'] = c.ask(OVERFLOW, document_id)
            out['no_token'] = c.json('POST', '/api/ask', {'question': 'q', 'document_id': document_id},
                                     token=False)[:2]
            out['bad_origin'] = c.json('POST', '/api/ask', {'question': 'q', 'document_id': document_id},
                                       origin='https://evil.example')[:2]
            return out
        old, new = self.both(flow)
        for case in old:
            self.assertEqual(new[case], old[case], case)
        self.assertEqual(old['model_down'][0], 502)
        self.assertEqual(old['bad_login'][0], 401)

    def test_release_check_runs_unmodified_against_new_version(self):
        mod = load_module()
        port = self.new.effective_port

        class Loopback(http.client.HTTPConnection):
            """Stands in for HTTPSConnection: plain HTTP to the in-process server, Host of the hostname."""

            def __init__(self, host, port_, timeout=None, context=None):
                super().__init__('127.0.0.1', port, timeout=timeout)

            def request(self, method, url, body=None, headers=None, **kw):
                super().request(method, url, body=body, headers={'Host': HOST, **(headers or {})}, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdf, questions, report = tmp / 'doc.pdf', tmp / 'q.json', tmp / 'report.json'
            pdf.write_bytes(PDF)
            questions.write_text(json.dumps([
                {'id': 'q1', 'kind': 'answerable', 'question': '질문 하나?'},
                {'id': 'q2', 'kind': 'numeric', 'question': '질문 둘?'}], ensure_ascii=False), encoding='utf-8')
            argv = ['--base', ORIGIN, '--user', ADMIN, '--pdf', str(pdf), '--questions', str(questions),
                    '--out', str(report)]
            out = io.StringIO()
            with mock.patch.object(mod.http.client, 'HTTPSConnection', Loopback), \
                    mock.patch.object(mod.getpass, 'getpass', return_value=PW), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = mod.main(argv)
            text = out.getvalue()
            self.assertEqual(code, 0, text)
            self.assertNotIn(SECRET_ANSWER, text)
            self.assertNotIn(PAGE_TEXT, text)
            result = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual([r['id'] for r in result['questions']], ['q1', 'q2'])
            self.assertEqual(result['questions'][0]['status'], 'quoted')
            self.assertNotIn(SECRET_ANSWER, report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
