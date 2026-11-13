import contextlib
import http.client
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from trid.onprem.auth import Auth
from trid.onprem.config import Config, load_config
from trid.onprem.model_card import MODEL_CARD
from trid.onprem.server import make_server
from trid.onprem.store import DETAIL_KEYS, Store
from trid.pdf_documents import Chunk, Document, DocumentError

HOST = 'test.example'
ORIGIN = 'https://' + HOST
PW = 'correct horse 1'
STATUS_TEXT = '실험 서비스 · 원본 모델(미특화) · 품질 기준 미달'
SECRET_NAME = 'secret-file-name-xyz.pdf'
SECRET_QUESTION = '비밀 질문 문자열 qzx 는 무엇인가요?'
DOWN = '모델 서버에 연결할 수 없습니다. 관리자에게 문의해 주세요.'


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def fake_doc(name='doc.pdf'):
    return Document(name, ['본문 한 쪽'], [Chunk(1, 0, 5, '본문 한 쪽')], [])


def ok_answer(document, question):
    return {'status': 'quoted', 'answer': 'a', 'checked': 1, 'raw_output': 'raw', 'candidates': []}


class Client:
    def __init__(self, server, ip=None):
        self.port = server.server_address[1]
        self.sid = None
        self.csrf = None
        self.ip = ip

    def request(self, method, path, body=None, headers=None, origin=ORIGIN, token=True, cookie=True,
                raw_headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        h = dict(headers or {})
        if origin and method == 'POST':
            h['Origin'] = origin
        if cookie and self.sid:
            h['Cookie'] = f'__Host-trid={self.sid}'
        if token and self.csrf:
            h['X-Trid-Token'] = self.csrf
        if self.ip:
            h['X-Real-IP'] = self.ip
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
            h.setdefault('Content-Type', 'application/json')
        conn.putrequest(method, path)
        for k, v in h.items():
            conn.putheader(k, v)
        for k, v in (raw_headers or {}).items():
            conn.putheader(k, v)
        if body is not None and not (raw_headers and 'Content-Length' in raw_headers):
            conn.putheader('Content-Length', str(len(body)))
        elif body is None and method == 'POST' and not raw_headers:
            conn.putheader('Content-Length', '0')
        conn.endheaders(body)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp, data

    def json(self, *args, **kwargs):
        resp, data = self.request(*args, **kwargs)
        return resp.status, (json.loads(data) if data[:1] == b'{' else data), resp

    def login(self, username='alice', password=PW):
        status, value, resp = self.json('POST', '/api/login', {'username': username, 'password': password})
        cookie = resp.getheader('Set-Cookie')
        if cookie:
            self.sid = cookie.split(';')[0].split('=', 1)[1]
            page = self.request('GET', '/')[1].decode()
            marker = 'name="csrf-token" content="'
            if marker in page:
                self.csrf = page.split(marker)[1].split('"')[0]
        return status, value

    def post(self, path, body=None, **kwargs):
        return self.json('POST', path, body if body is not None else {}, **kwargs)

    def upload(self, name=SECRET_NAME):
        with mock.patch('trid.onprem.server.parse_pdf', return_value=fake_doc(name)):
            return self.json('POST', '/api/load', b'%PDF-1.4 fake',
                             headers={'X-File-Name': name})


class ServerTests(unittest.TestCase):
    answer = staticmethod(ok_answer)
    config_overrides = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mono = Clock(0.0)
        self.wall = Clock(1_700_000_000.0)
        self.db = Path(self.tmp.name) / 'app.db'
        self.store = Store(self.db, clock=self.wall)
        self.addCleanup(self.store.close)
        self.auth = Auth(self.store, clock=self.mono, wall=self.wall)
        self.store.create_user('alice', PW, 'user', must_change=False)
        self.store.create_user('bob', PW, 'user', must_change=False)
        self.config = Config(hostname=HOST, port=0, **self.config_overrides)
        calls = self.calls = []

        def answer(document, question):
            calls.append(question)
            return type(self).answer(document, question)

        self.server = make_server(self.config, self.store, self.auth, object(), answer=answer)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def client(self, name='alice', ip=None):
        c = Client(self.server, ip)
        self.assertEqual(c.login(name)[0], 200)
        return c

    def events(self, event=None):
        return self.store.query_events(event=event)

    def test_unauthenticated_get_serves_login_and_api_returns_401(self):
        c = Client(self.server)
        resp, data = c.request('GET', '/')
        self.assertEqual(resp.status, 200)
        self.assertIn('action="/api/login"', data.decode())
        self.assertNotIn('csrf-token', data.decode())
        self.assertIn("script-src 'self'", resp.getheader('Content-Security-Policy'))
        self.assertEqual(resp.getheader('X-Frame-Options'), 'DENY')
        self.assertEqual(c.post('/api/ask', {'question': 'q', 'document_id': 'x'})[0], 401)
        self.assertEqual(c.post('/api/load', {})[0], 401)
        status, value, _ = c.json('GET', '/healthz')
        self.assertEqual((status, value), (200, {'ok': True}))

    def test_login_sets_host_cookie_and_wrong_origin_rejected(self):
        c = Client(self.server)
        body = {'username': 'alice', 'password': PW}
        self.assertEqual(c.json('POST', '/api/login', body, origin='https://evil.example')[0], 403)
        self.assertEqual(c.json('POST', '/api/login', body, origin=None)[0], 403)
        status, value, resp = c.json('POST', '/api/login', body)
        self.assertEqual((status, value), (200, {'status': 'ok'}))
        cookie = resp.getheader('Set-Cookie')
        self.assertTrue(cookie.startswith('__Host-trid='))
        for part in ('Path=/', 'Secure', 'HttpOnly', 'SameSite=Strict'):
            self.assertIn(part, cookie)
        self.assertNotIn('Domain', cookie)

    def test_invalid_and_locked_login_are_identical(self):
        c = Client(self.server)
        responses = []
        for _ in range(6):
            status, value, _ = c.json('POST', '/api/login', {'username': 'alice', 'password': 'wrong-password'})
            responses.append((status, value))
        status, value, _ = c.json('POST', '/api/login', {'username': 'nobody', 'password': 'wrong-password'})
        responses.append((status, value))
        self.assertEqual(len(set(map(repr, responses))), 1)
        self.assertEqual(responses[0], (401, {'error': '아이디 또는 비밀번호가 올바르지 않습니다.'}))
        self.assertTrue(self.events('login_locked'))

    def test_csrf_required_on_post(self):
        c = self.client()
        self.assertEqual(c.post('/api/clear', token=False)[0], 403)
        saved, c.csrf = c.csrf, 'wrong-token'
        self.assertEqual(c.post('/api/clear')[0], 403)
        c.csrf = saved
        self.assertEqual(c.post('/api/clear', origin='https://evil.example')[0], 403)
        self.assertEqual(c.post('/api/clear')[0], 200)

    def test_must_change_blocks_load_until_password_changed(self):
        self.store.create_user('carol', PW, 'user', must_change=True)
        c = Client(self.server)
        status, value = c.login('carol')
        self.assertEqual((status, value), (200, {'status': 'must_change'}))
        status, value, _ = c.upload()
        self.assertEqual(status, 403)
        self.assertTrue(value['must_change'])
        self.assertEqual(c.post('/api/ask', {'question': 'q', 'document_id': 'x'})[0], 403)
        status, value, _ = c.post('/api/password', {'current': PW, 'new': 'a brand new pw 22'})
        self.assertEqual(status, 200, value)
        self.assertEqual(c.upload()[0], 200)

    def test_password_errors_are_400(self):
        c = self.client()
        status, value, _ = c.post('/api/password', {'current': 'nope-nope-nope', 'new': 'a brand new pw 22'})
        self.assertEqual(status, 400)
        status, value, _ = c.post('/api/password', {'current': PW, 'new': 'short'})
        self.assertEqual(status, 400)
        self.assertEqual(c.post('/api/password', {'current': PW})[0], 400)

    def test_upload_then_ask_logs_hash_pages_seconds_status_only(self):
        c = self.client(ip='203.0.113.7')
        status, value, _ = c.upload()
        self.assertEqual(status, 200)
        self.assertEqual(value['name'], SECRET_NAME)
        self.assertEqual(value['pages'], ['본문 한 쪽'])
        status, value, _ = c.post('/api/ask', {'question': SECRET_QUESTION, 'document_id': value['document_id']})
        self.assertEqual(status, 200)
        self.assertEqual(value['status'], 'quoted')
        self.assertNotIn('checked', value)
        self.assertNotIn('raw_output', value)
        self.assertEqual(self.calls, [SECRET_QUESTION])
        self.assertEqual(c.post('/api/clear')[0], 200)
        events = self.store.query_events()
        for event in events:
            keys = set(event['detail']) if isinstance(event.get('detail'), dict) else set(json.loads(event['detail']))
            self.assertLessEqual(keys, DETAIL_KEYS)
        upload = self.events('pdf_upload')[0]
        detail = upload['detail'] if isinstance(upload['detail'], dict) else json.loads(upload['detail'])
        self.assertEqual(len(detail['sha256']), 64)
        self.assertEqual(detail['pages'], 1)
        self.assertEqual(detail['status'], 'ok')
        self.assertEqual(upload['ip'], '203.0.113.7')
        ask = self.events('ask')[0]
        detail = ask['detail'] if isinstance(ask['detail'], dict) else json.loads(ask['detail'])
        self.assertEqual(detail['status'], 'quoted')
        self.assertIn('seconds', detail)
        self.assertTrue(self.events('clear'))
        self.store.close()
        raw = self.db.read_bytes()
        for wal in Path(self.tmp.name).glob('app.db-*'):
            raw += wal.read_bytes()
        for secret in (SECRET_NAME, SECRET_QUESTION, '본문 한 쪽'):
            self.assertNotIn(secret.encode(), raw)

    def test_document_is_per_session(self):
        a, b = self.client('alice'), self.client('bob')
        document_id = a.upload()[1]['document_id']
        self.assertEqual(b.post('/api/ask', {'question': 'q', 'document_id': document_id})[0], 409)
        self.assertEqual(a.post('/api/ask', {'question': 'q', 'document_id': document_id})[0], 200)

    def test_expired_session_mid_ask_returns_401_and_drops_document(self):
        c = self.client()
        document_id = c.upload()[1]['document_id']
        session = self.auth.session(c.sid)
        self.assertIsNotNone(session.document)
        self.mono.t += 1801
        self.wall.t += 1801
        status, value, _ = c.post('/api/ask', {'question': 'q', 'document_id': document_id})
        self.assertEqual(status, 401)
        self.assertIsNone(session.document)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.events('session_expired'))

    def test_model_down_releases_lock_and_logs_kind_only(self):
        c = self.client()
        document_id = c.upload()[1]['document_id']
        outcomes = [ConnectionRefusedError(SECRET_QUESTION), None]

        def answer(document, question):
            outcome = outcomes.pop(0)
            if outcome:
                raise outcome
            return {'status': 'not_found'}

        server = make_server(self.config, self.store, self.auth, object(), answer=answer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        c.port = server.server_address[1]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status, value, _ = c.post('/api/ask', {'question': SECRET_QUESTION, 'document_id': document_id})
            self.assertEqual((status, value), (502, {'error': DOWN}))
            self.assertEqual(c.post('/api/ask', {'question': SECRET_QUESTION, 'document_id': document_id})[0], 200)
        self.assertNotIn('qzx', err.getvalue())
        self.assertFalse(server.generation_lock.locked())
        asks = self.events('ask')
        errors = [json.loads(e['detail']) if isinstance(e['detail'], str) else e['detail'] for e in asks]
        self.assertIn('model_unavailable', [d.get('error') for d in errors])
        self.assertNotIn(SECRET_QUESTION.encode(), self.db.read_bytes())

    def test_error_mapping_for_overflow_bad_request_and_unexpected(self):
        from trid.document_case import CONTEXT_OVERFLOW
        c = self.client()
        document_id = c.upload()[1]['document_id']
        failures = iter([DocumentError(CONTEXT_OVERFLOW), DocumentError('질문을 입력해 주세요.'),
                         RuntimeError(SECRET_QUESTION)])

        def answer(document, question):
            raise next(failures)

        server = make_server(self.config, self.store, self.auth, object(), answer=answer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        c.port = server.server_address[1]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            s1, v1, _ = c.post('/api/ask', {'question': 'q', 'document_id': document_id})
            s2, v2, _ = c.post('/api/ask', {'question': 'q', 'document_id': document_id})
            s3, v3, _ = c.post('/api/ask', {'question': 'q', 'document_id': document_id})
        self.assertEqual((s1, v1), (422, {'error': CONTEXT_OVERFLOW}))
        self.assertEqual((s2, v2), (400, {'error': '질문을 입력해 주세요.'}))
        self.assertEqual((s3, v3), (500, {'error': '처리 중 오류가 발생했습니다.'}))
        self.assertEqual(err.getvalue().strip(), 'RuntimeError')
        kinds = [json.loads(e['detail']).get('error') if isinstance(e['detail'], str) else e['detail'].get('error')
                 for e in self.events('ask')]
        self.assertEqual(sorted(k for k in kinds if k), ['bad_request', 'context_overflow'])

    def test_client_disconnect_during_failing_ask_prints_no_traceback(self):
        c = self.client()
        document_id = c.upload()[1]['document_id']
        started, release = threading.Event(), threading.Event()

        def answer(document, question):
            started.set()
            release.wait(5)
            raise RuntimeError(SECRET_QUESTION)

        server = make_server(self.config, self.store, self.auth, object(), answer=answer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        body = json.dumps({'question': SECRET_QUESTION, 'document_id': document_id}, ensure_ascii=False).encode()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            import socket
            import struct
            sock = socket.create_connection(('127.0.0.1', server.server_address[1]), timeout=10)
            head = '\r\n'.join([
                'POST /api/ask HTTP/1.1', 'Host: x', f'Origin: {ORIGIN}', f'Cookie: __Host-trid={c.sid}',
                f'X-Trid-Token: {c.csrf}', f'Content-Length: {len(body)}', '', '']).encode()
            sock.sendall(head + body)
            self.assertTrue(started.wait(5))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            sock.close()
            release.set()
            deadline = time.monotonic() + 5
            while server.generation_lock.locked() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.3)
        self.assertNotIn('qzx', err.getvalue())
        self.assertNotIn('Traceback', err.getvalue())

    def test_generation_lock_busy_returns_503_after_lock_wait(self):
        c = self.client()
        document_id = c.upload()[1]['document_id']
        self.server.generation_lock.acquire()
        try:
            started = time.monotonic()
            status, value, _ = c.post('/api/ask', {'question': 'q', 'document_id': document_id})
            waited = time.monotonic() - started
        finally:
            self.server.generation_lock.release()
        self.assertEqual(status, 503)
        self.assertEqual(value, {'error': '다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.'})
        self.assertGreaterEqual(waited, 1.5)
        self.assertEqual(self.calls, [])
        self.assertEqual(c.post('/api/ask', {'question': 'q', 'document_id': document_id})[0], 200)

    def test_rate_limit_21st_request_in_minute_is_429(self):
        c = self.client()
        statuses = [c.post('/api/clear')[0] for _ in range(21)]
        self.assertEqual(statuses[:20], [200] * 20)
        self.assertEqual(statuses[20], 429)

    def test_oversize_body_413(self):
        c = self.client()
        big = b'x' * (self.config.pdf.max_bytes + 1)

        def rejected(client, path, body):
            # The server answers 413 from the headers and closes without reading the body, so a client
            # still sending a large body can get a connection reset before it reads the response.
            try:
                return client.request('POST', path, body)[0].status == 413
            except (ConnectionError, http.client.RemoteDisconnected):
                return True

        self.assertTrue(rejected(c, '/api/load', big))
        self.assertTrue(rejected(c, '/api/ask', b'x' * (16 * 1024 + 1)))
        self.assertTrue(rejected(Client(self.server), '/api/login', b'x' * (16 * 1024 + 1)))

    def test_content_length_required_and_chunked_rejected(self):
        c = self.client()
        conn = http.client.HTTPConnection('127.0.0.1', c.port, timeout=10)
        conn.putrequest('POST', '/api/clear')
        for k, v in {'Origin': ORIGIN, 'Cookie': f'__Host-trid={c.sid}', 'X-Trid-Token': c.csrf}.items():
            conn.putheader(k, v)
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 411)
        resp, _ = c.request('POST', '/api/clear', b'0\r\n\r\n', raw_headers={'Transfer-Encoding': 'chunked'})
        self.assertEqual(resp.status, 400)

    def test_index_page_shows_model_status_text(self):
        c = self.client()
        resp, data = c.request('GET', '/')
        page = data.decode()
        self.assertIn(STATUS_TEXT, page)
        self.assertIn(f'<meta name="csrf-token" content="{c.csrf}">', page)
        self.assertNotIn('<script>', page)
        self.assertEqual(MODEL_CARD['status'], STATUS_TEXT)
        self.assertIn('Apache-2.0', page)
        login_page = Client(self.server).request('GET', '/')[1].decode()
        self.assertIn(STATUS_TEXT, login_page)

    def test_logout_clears_session_and_document(self):
        c = self.client()
        c.upload()
        session = self.auth.session(c.sid)
        self.assertEqual(c.post('/api/logout')[0], 200)
        self.assertIsNone(session.document)
        self.assertEqual(c.post('/api/clear')[0], 401)
        self.assertTrue(self.events('logout'))

    def test_bad_client_ip_header_falls_back_to_peer(self):
        c = Client(self.server, ip='not-an-ip')
        self.assertEqual(c.login()[0], 200)
        self.assertEqual(self.events('login_ok')[0]['ip'], '127.0.0.1')

    def test_upload_failure_logged_without_file_name(self):
        c = self.client()
        with mock.patch('trid.onprem.server.parse_pdf', side_effect=DocumentError('PDF 파일 형식이 아닙니다.')):
            status, value, _ = c.json('POST', '/api/load', b'junk', headers={'X-File-Name': SECRET_NAME})
        self.assertEqual((status, value), (400, {'error': 'PDF 파일 형식이 아닙니다.'}))
        upload = self.events('pdf_upload')[0]
        self.assertEqual(upload['result'], 'error')
        self.assertNotIn(SECRET_NAME.encode(), self.db.read_bytes())


class InsecureDevTests(unittest.TestCase):
    def test_dev_mode_uses_plain_cookie_and_loopback_origin(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / 'app.db')
        self.addCleanup(store.close)
        store.create_user('alice', PW, 'user', must_change=False)
        server = make_server(Config(hostname=HOST, port=0), store, Auth(store), object(),
                             answer=ok_answer, insecure_dev=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        c = Client(server)
        body = {'username': 'alice', 'password': PW}
        self.assertEqual(c.json('POST', '/api/login', body)[0], 403)
        status, _, resp = c.json('POST', '/api/login', body,
                                 origin=f'http://127.0.0.1:{server.server_address[1]}')
        self.assertEqual(status, 200)
        cookie = resp.getheader('Set-Cookie')
        self.assertTrue(cookie.startswith('trid-dev='))
        self.assertNotIn('Secure', cookie)
        self.assertNotIn('__Host-', cookie)


class ConfigTests(unittest.TestCase):
    def test_load_config_defaults_and_unknown_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.json'
            path.write_text(json.dumps({'hostname': 'h.example', 'port': 9000, 'pdf': {'max_pages': 30}}),
                            encoding='utf-8')
            config = load_config(path)
            self.assertEqual((config.hostname, config.port, config.pdf.max_pages), ('h.example', 9000, 30))
            self.assertEqual((config.slots, config.per_minute, config.context, config.family),
                             (8, 20, 4096, 'qwen3'))
            self.assertEqual(config.pdf.max_bytes, 2 * 1024 * 1024)
            path.write_text(json.dumps({'hostname': 'h', 'bogus': 1}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(json.dumps({'hostname': 'h', 'pdf': {'bogus': 1}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(json.dumps({'port': 1}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == '__main__':
    unittest.main()
