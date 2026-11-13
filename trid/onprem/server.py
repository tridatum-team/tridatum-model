"""Staff-facing HTTP server of the on-prem build (login session instead of a proxy secret).

Nothing here writes file names, questions, answers or PDF text anywhere: usage events carry only
the fields the store allows, and stderr gets exception class names only.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import ipaddress
import json
import re
import secrets
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from trid.document_case import CONTEXT_OVERFLOW, answer_document_case, fit_own_passages
from trid.llama_client import complete, count_prompt_tokens
from trid.onprem.admin import POST_ROUTES as ADMIN_ROUTES, AdminApi
from trid.onprem.model_card import MODEL_CARD
from trid.pdf_documents import DocumentError
from trid.pdf_isolated import parse_pdf

UI_DIR = Path(__file__).resolve().parent.parent / 'onprem_ui'
SMALL_BODY = 16 * 1024
STATIC = {
    '/app.js': 'text/javascript; charset=utf-8',
    '/login.js': 'text/javascript; charset=utf-8',
    '/admin.js': 'text/javascript; charset=utf-8',
    '/common.css': 'text/css; charset=utf-8',
}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; "
       "object-src 'none'; base-uri 'none'; form-action 'self'")
BUSY_ASK = '다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.'
BAD_LOGIN = '아이디 또는 비밀번호가 올바르지 않습니다.'
MODEL_DOWN = '모델 서버에 연결할 수 없습니다. 관리자에게 문의해 주세요.'
GENERIC_ERROR = '처리 중 오류가 발생했습니다.'
MUST_CHANGE = '비밀번호를 먼저 변경해 주세요.'
ADMIN_ONLY = '관리자 권한이 필요합니다.'
CSRF_FAIL = '세션이 만료됐습니다. 페이지를 새로고침해 주세요.'
_STATUS_RE = re.compile(r'[A-Za-z0-9._:-]{1,128}')


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, slots=8):
        self.slots = threading.BoundedSemaphore(slots)
        super().__init__(*args)

    def handle_error(self, request, client_address):
        # Never print a traceback: its chained exceptions can carry questions or file names.
        print(sys.exc_info()[0].__name__, file=sys.stderr)

    def process_request(self, request, address):
        if not self.slots.acquire(False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def _default_answer(config):
    def answer(document, question):
        return answer_document_case(
            document, question, config.llm_base, config.family, config.context,
            max_tokens=config.max_tokens, complete=complete,
            fit_context=lambda s, b, base, c, m: fit_own_passages(
                s, b, lambda msgs: count_prompt_tokens(base, msgs), c, m))
    return answer


def make_server(config, store, auth, tok, answer=None, insecure_dev: bool = False,
                host: str | None = None):
    answer = answer or _default_answer(config)
    cookie_name = 'trid-dev' if insecure_dev else '__Host-trid'
    gen_lock = threading.Lock()
    rate_lock = threading.Lock()
    origin_holder = {'value': 'https://' + config.hostname}
    admin_api = AdminApi(store, auth)

    def page(name):
        return (UI_DIR / name).read_text(encoding='utf-8')

    status_text = html.escape(MODEL_CARD['status'])
    card_text = html.escape(
        f"{MODEL_CARD['label']} · {MODEL_CARD['version']} · {MODEL_CARD['status']}\n"
        f"기반: {MODEL_CARD['base_model']} · 라이선스: {MODEL_CARD['license']}\n"
        f"실측: {MODEL_CARD['evaluation']}\n한계: {MODEL_CARD['limits']}")

    class Handler(BaseHTTPRequestHandler):
        server_version = 'trid'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass

        # -- helpers ---------------------------------------------------------
        def respond(self, status, value, mime='application/json; charset=utf-8', cookie=None, clear_cookie=False, extra=None):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            for key, val in {
                'Content-Type': mime, 'Content-Length': str(len(body)), 'Cache-Control': 'no-store',
                'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                'Referrer-Policy': 'no-referrer', 'Connection': 'close',
                'Content-Security-Policy': CSP,
            }.items():
                self.send_header(key, val)
            for key, val in (extra or {}).items():
                self.send_header(key, val)
            attrs = '; Path=/; HttpOnly; SameSite=Strict' + ('' if insecure_dev else '; Secure')
            if cookie:
                self.send_header('Set-Cookie', f'{cookie_name}={cookie}{attrs}')
            elif clear_cookie:
                self.send_header('Set-Cookie', f'{cookie_name}={attrs}; Max-Age=0')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def client_ip(self):
            for candidate in (self.headers.get('X-Real-IP', '').strip(), self.client_address[0]):
                try:
                    return str(ipaddress.ip_address(candidate))
                except ValueError:
                    continue
            return None

        def sid(self):
            jar = SimpleCookie()
            try:
                jar.load(self.headers.get('Cookie', ''))
                return jar[cookie_name].value if cookie_name in jar else ''
            except Exception:
                return ''

        def read_body(self, limit):
            """Return the body, or None after answering with an error."""
            if self.headers.get('Transfer-Encoding'):
                self.respond(400, {'error': '허용되지 않은 요청입니다.'}); return None
            raw = self.headers.get('Content-Length')
            if raw is None:
                self.respond(411, {'error': '요청 길이가 필요합니다.'}); return None
            try:
                size = int(raw)
            except ValueError:
                size = -1
            if size < 0:
                self.respond(400, {'error': '요청 길이가 올바르지 않습니다.'}); return None
            if size > limit:
                self.respond(413, {'error': '입력이 너무 큽니다.'}); return None
            body = self.rfile.read(size)
            if len(body) != size:
                self.respond(400, {'error': '입력 전송이 완료되지 않았습니다.'}); return None
            return body

        def json_body(self, body):
            try:
                value = json.loads(body)
            except (ValueError, RecursionError):
                value = None
            return value if isinstance(value, dict) else None

        # -- GET -------------------------------------------------------------
        def do_GET(self):
            try:
                self.get()
            except Exception as e:
                print(type(e).__name__, file=sys.stderr)
                try:
                    self.respond(500, {'error': GENERIC_ERROR})
                except OSError:
                    pass

        def get(self):
            auth.prune()
            path = urlsplit(self.path).path
            if path == '/healthz':
                self.respond(200, {'ok': True}); return
            if path in STATIC:
                try:
                    data = (UI_DIR / path[1:]).read_bytes()
                except OSError:
                    self.respond(404, {'error': '없는 페이지입니다.'}); return
                self.respond(200, data, STATIC[path]); return
            if path == '/admin':
                self.admin_page(); return
            if path == '/api/admin/events.csv':
                self.admin_csv(); return
            if path != '/':
                self.respond(404, {'error': '없는 페이지입니다.'}); return
            session = auth.session(self.sid())
            if session is None:
                text = page('login.html').replace('MODEL_STATUS_PLACEHOLDER', status_text)
            else:
                text = (page('index.html')
                        .replace('TOKEN_PLACEHOLDER', html.escape(session.csrf, quote=True))
                        .replace('MUST_CHANGE_PLACEHOLDER', '1' if session.must_change else '0')
                        .replace('ROLE_PLACEHOLDER', html.escape(session.role, quote=True))
                        .replace('USERNAME_PLACEHOLDER', html.escape(session.username, quote=True))
                        .replace('MODEL_STATUS_PLACEHOLDER', status_text)
                        .replace('MODEL_CARD_PLACEHOLDER', card_text))
            self.respond(200, text.encode(), 'text/html; charset=utf-8')

        def admin_page(self):
            session = auth.session(self.sid())
            if session is None:
                self.respond(200, page('login.html').replace('MODEL_STATUS_PLACEHOLDER', status_text).encode(),
                             'text/html; charset=utf-8'); return
            if session.role != 'admin':
                self.respond(403, {'error': ADMIN_ONLY}); return
            if session.must_change:
                self.respond(403, {'error': MUST_CHANGE, 'must_change': True}); return
            text = page('admin.html').replace('TOKEN_PLACEHOLDER', html.escape(session.csrf, quote=True))
            self.respond(200, text.encode(), 'text/html; charset=utf-8')

        def admin_csv(self):
            session = auth.session(self.sid())
            if session is None:
                self.respond(401, {'error': '로그인이 필요합니다.'}, clear_cookie=True); return
            query = urlsplit(self.path).query
            token = parse_qs(query).get('token', [''])[-1]
            if not token.isascii() or not hmac.compare_digest(token, session.csrf):
                self.respond(403, {'error': CSRF_FAIL}); return
            if session.role != 'admin':
                self.respond(403, {'error': ADMIN_ONLY}); return
            if session.must_change:
                self.respond(403, {'error': MUST_CHANGE, 'must_change': True}); return
            if not self.rate_ok(session):
                return
            status, value, extra = admin_api.csv(session, query, self.client_ip())
            if status != 200:
                self.respond(status, value); return
            self.respond(200, value, 'text/csv; charset=utf-8', extra=extra)

        def rate_ok(self, session):
            with rate_lock:
                now = time.monotonic()
                session.calls = [t for t in session.calls if now - t < 60]
                if len(session.calls) >= config.per_minute:
                    self.respond(429, {'error': '요청이 많습니다. 1분 뒤 다시 시도해 주세요.'}); return False
                session.calls.append(now)
            return True

        # -- POST ------------------------------------------------------------
        def do_POST(self):
            try:
                self.post()
            except Exception as e:
                print(type(e).__name__, file=sys.stderr)
                try:
                    self.respond(500, {'error': GENERIC_ERROR})
                except OSError:
                    pass

        def post(self):
            auth.prune()
            path = urlsplit(self.path).path
            if self.headers.get('Origin') != origin_holder['value']:
                self.respond(403, {'error': '허용되지 않은 요청입니다.'}); return
            ip = self.client_ip()
            if path == '/api/login':
                self.login(ip); return
            routes = {'/api/logout', '/api/password', '/api/load', '/api/ask', '/api/clear'} | ADMIN_ROUTES
            sid = self.sid()
            session = auth.session(sid)
            if session is None:
                self.respond(401, {'error': '로그인이 필요합니다.'}, clear_cookie=True); return
            token = self.headers.get('X-Trid-Token', '')
            if not token.isascii() or not hmac.compare_digest(token, session.csrf):
                self.respond(403, {'error': CSRF_FAIL}); return
            if session.must_change and path not in ('/api/password', '/api/logout'):
                self.respond(403, {'error': MUST_CHANGE, 'must_change': True}); return
            if path not in routes:
                self.respond(404, {'error': '없는 페이지입니다.'}); return
            if path in ADMIN_ROUTES and session.role != 'admin':
                self.respond(403, {'error': ADMIN_ONLY}); return
            if not self.rate_ok(session):
                return
            if path == '/api/logout':
                if self.read_body(SMALL_BODY) is None:
                    return
                auth.logout(sid, ip)
                self.respond(200, {'ok': True}, clear_cookie=True); return
            body = self.read_body(config.pdf.max_bytes if path == '/api/load' else SMALL_BODY)
            if body is None:
                return
            if path in ADMIN_ROUTES:
                request = self.json_body(body)
                if request is None:
                    self.respond(400, {'error': '요청 형식이 올바르지 않습니다.'}); return
                status, value = admin_api.post(path, session, request, ip)
                self.respond(status, value); return
            if path == '/api/clear':
                session.document = None
                session.document_id = None
                store.log('clear', session.username, ip, 'ok')
                self.respond(200, {'cleared': True}); return
            if path == '/api/password':
                self.password(session, body, ip); return
            if path == '/api/load':
                self.load(session, body, ip); return
            self.ask(session, body, ip)

        def login(self, ip):
            body = self.read_body(SMALL_BODY)
            if body is None:
                return
            request = self.json_body(body)
            if request is None or not isinstance(request.get('username'), str) \
                    or not isinstance(request.get('password'), str):
                self.respond(400, {'error': '아이디와 비밀번호를 입력해 주세요.'}); return
            status, session = auth.login(request['username'], request['password'], ip)
            if status in ('ok', 'must_change'):
                self.respond(200, {'status': status}, cookie=session.sid); return
            if status == 'busy':
                self.respond(503, {'error': '현재 접속 인원이 많습니다. 잠시 후 다시 시도해 주세요.'}); return
            self.respond(401, {'error': BAD_LOGIN})

        def password(self, session, body, ip):
            request = self.json_body(body)
            if request is None or not isinstance(request.get('current'), str) \
                    or not isinstance(request.get('new'), str):
                self.respond(400, {'error': '현재 비밀번호와 새 비밀번호를 입력해 주세요.'}); return
            try:
                auth.change_password(session, request['current'], request['new'], ip)
            except PermissionError:
                self.respond(401, {'error': '로그인이 필요합니다.'}, clear_cookie=True); return
            except ValueError as e:
                self.respond(400, {'error': str(e)}); return
            self.respond(200, {'ok': True})

        def load(self, session, body, ip):
            digest = hashlib.sha256(body).hexdigest()
            if not gen_lock.acquire(timeout=config.lock_wait):
                self.respond(503, {'error': BUSY_ASK}); return
            try:
                name = unquote(self.headers.get('X-File-Name', 'document.pdf'))[:200] or 'document.pdf'
                try:
                    doc = parse_pdf(body, name, tok, limits=config.pdf)
                except DocumentError as e:
                    store.log('pdf_upload', session.username, ip, 'error', sha256=digest, status='error')
                    self.respond(400, {'error': str(e)}); return
                except Exception as e:
                    print(type(e).__name__, file=sys.stderr)
                    store.log('pdf_upload', session.username, ip, 'error', sha256=digest, status='error')
                    self.respond(500, {'error': GENERIC_ERROR}); return
                document_id = secrets.token_urlsafe(24)
                session.document = doc
                session.document_id = document_id
                store.log('pdf_upload', session.username, ip, 'ok',
                          sha256=digest, pages=len(doc.pages), status='ok')
                self.respond(200, {'document_id': document_id, 'name': doc.name,
                                   'pages': doc.pages, 'warnings': doc.warnings})
            finally:
                gen_lock.release()

        def ask(self, session, body, ip):
            request = self.json_body(body)
            question = request.get('question') if request else None
            if (not isinstance(question, str) or len(question) > config.max_question
                    or not isinstance(request.get('document_id'), str)):
                self.respond(400, {'error': f'질문은 {config.max_question:,}자 이하로 입력해 주세요.'}); return
            document = session.document
            current = session.document_id
            if document is None or current is None or not hmac.compare_digest(
                    request['document_id'].encode(), current.encode()):
                self.respond(409, {'error': '이 브라우저에서 문서를 먼저 넣어 주세요.'}); return
            started = time.monotonic()

            def log(result, **detail):
                store.log('ask', session.username, ip, result,
                          seconds=round(time.monotonic() - started, 2), **detail)

            if not gen_lock.acquire(timeout=config.lock_wait):
                log('error', status='error', error='busy')
                self.respond(503, {'error': BUSY_ASK}); return
            try:
                result = answer(document, question)
            except DocumentError as e:
                overflow = str(e) == CONTEXT_OVERFLOW
                log('error', status='error', error='context_overflow' if overflow else 'bad_request')
                self.respond(422 if overflow else 400, {'error': str(e)}); return
            except OSError:
                log('error', status='error', error='model_unavailable')
                self.respond(502, {'error': MODEL_DOWN}); return
            except Exception as e:
                print(type(e).__name__, file=sys.stderr)
                log('error', status='error')
                self.respond(500, {'error': GENERIC_ERROR}); return
            finally:
                gen_lock.release()
            status = result.get('status')
            log('ok', status=status if isinstance(status, str) and _STATUS_RE.fullmatch(status) else 'unknown')
            self.respond(200, {k: v for k, v in result.items() if k not in ('checked', 'raw_output')})

    if host is None:
        host = '127.0.0.1'  # the serve command passes 0.0.0.0 explicitly inside the container
    server = BoundedServer((host, config.port), Handler, slots=config.slots)
    server.generation_lock = gen_lock
    if insecure_dev:
        origin_holder['value'] = f'http://127.0.0.1:{server.server_address[1]}'
    return server
