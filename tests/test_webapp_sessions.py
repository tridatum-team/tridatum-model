import json
import re
import unittest

import webapp_support as S  # noqa: F401  (boots Django and migrates)

from django.contrib.sessions.models import Session
from django.test import TestCase

from onprem_web.accounts import services
from onprem_web.audit.models import Event
from trid.onprem.server import BAD_LOGIN, CSRF_FAIL, MUST_CHANGE

PW = S.PW
NEW_PW = 'a brand new pw 22'
ALLOWED_KEYS = {'_auth_user_id', '_auth_user_backend', '_auth_user_hash', '_csrftoken', 'pv', 'had_document'}
_TOKEN = re.compile(r'<meta name="csrf-token" content="([^"]*)"')


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Browser:
    """A Django test client that logs in and keeps the CSRF token like the page script does."""

    def __init__(self):
        self.c = S.client()
        self.token = None

    def post(self, path, body=None, token=True, **extra):
        if token and self.token:
            extra['HTTP_X_TRID_TOKEN'] = self.token
        return self.c.post(path, json.dumps(body if body is not None else {}),
                           content_type='application/json', secure=True, **extra)

    def get(self, path='/'):
        return self.c.get(path, secure=True)

    def login(self, username='alice', password=PW):
        r = self.post('/api/login', {'username': username, 'password': password})
        if r.status_code == 200:
            m = _TOKEN.search(self.get('/').content.decode())
            self.token = m.group(1) if m else None
        return r

    @property
    def key(self):
        return self.c.cookies['__Host-trid'].value


def events(event):
    return [(e.username, e.result) for e in Event.objects.filter(event=event).order_by('id')]


class SessionTests(TestCase):
    def setUp(self):
        self.clock = Clock()
        self.rt = S.make_runtime(clock=self.clock)
        services.create_user('alice', PW, 'user', must_change=False)
        services.create_user('bob', PW, 'user', must_change=False)

    def logged_in(self, name='alice'):
        b = Browser()
        self.assertEqual(b.login(name).status_code, 200)
        return b

    def test_login_sets_single_host_cookie(self):
        b = Browser()
        r = b.post('/api/login', {'username': 'alice', 'password': PW})
        self.assertEqual((r.status_code, r.json()), (200, {'status': 'ok'}))
        self.assertEqual(list(r.cookies), ['__Host-trid'])
        self.assertEqual(len(r.cookies.output().splitlines()), 1)
        morsel = r.cookies['__Host-trid']
        self.assertEqual(morsel['path'], '/')
        self.assertEqual(morsel['domain'], '')
        self.assertTrue(morsel['secure'])
        self.assertTrue(morsel['httponly'])
        self.assertEqual(morsel['samesite'], 'Strict')
        self.assertIn(morsel.value, self.rt.registry.keys())
        # bad credentials and missing fields
        self.assertEqual(Browser().post('/api/login', {'username': 'alice', 'password': 'wrong-password'}).json(),
                         {'error': BAD_LOGIN})
        r = Browser().post('/api/login', {'username': 'alice'})
        self.assertEqual((r.status_code, r.json()), (400, {'error': '아이디와 비밀번호를 입력해 주세요.'}))

    def test_bad_origin_rejected_even_on_login(self):
        b = Browser()
        body = {'username': 'alice', 'password': PW}
        r = b.post('/api/login', body, HTTP_ORIGIN='https://evil.example')
        self.assertEqual((r.status_code, r.json()), (403, {'error': '허용되지 않은 요청입니다.'}))
        self.assertNotIn('__Host-trid', r.cookies)
        self.assertEqual(b.login().status_code, 200)
        r = b.post('/api/logout', HTTP_ORIGIN='https://evil.example')
        self.assertEqual(r.status_code, 403)
        self.assertEqual(len(self.rt.registry.keys()), 1)

    def test_post_without_token_is_csrf_fail(self):
        b = self.logged_in()
        r = b.post('/api/logout', token=False)
        self.assertEqual((r.status_code, r.json()), (403, {'error': CSRF_FAIL}))
        r = b.post('/api/logout', token=False, HTTP_X_TRID_TOKEN='wrong-token')
        self.assertEqual(r.status_code, 403)
        key = b.key
        r = b.post('/api/logout')
        self.assertEqual((r.status_code, r.json()), (200, {'ok': True}))
        self.assertNotIn(key, self.rt.registry.keys())
        self.assertFalse(Session.objects.filter(session_key=key).exists())
        self.assertEqual(events('logout'), [('alice', 'ok')])
        r = b.post('/api/logout')
        self.assertEqual((r.status_code, r.json()), (401, {'error': '로그인이 필요합니다.'}))

    def test_unknown_session_is_401_and_clears_cookie(self):
        b = Browser()
        b.c.cookies['__Host-trid'] = 'x' * 32
        r = b.post('/api/password', {'current': PW, 'new': NEW_PW})
        self.assertEqual((r.status_code, r.json()), (401, {'error': '로그인이 필요합니다.'}))
        self.assertEqual(r.cookies['__Host-trid']['max-age'], 0)
        self.assertNotIn('csrf-token', b.get('/').content.decode())

    def test_idle_30_minutes_expires_and_logs(self):
        b = self.logged_in()
        key = b.key
        entry = self.rt.registry.get(key)
        entry.document, entry.document_id = object(), 'doc-1'
        self.clock.t += 1799
        self.assertEqual(b.post('/api/password', {}).status_code, 400)  # still alive; touches
        self.clock.t += 1800
        r = b.post('/api/password', {})
        self.assertEqual((r.status_code, r.json()), (401, {'error': '로그인이 필요합니다.'}))
        self.assertEqual((entry.document, entry.document_id), (None, None))
        self.assertEqual(self.rt.registry.keys(), set())
        self.assertFalse(Session.objects.filter(session_key=key).exists())
        self.assertEqual(events('session_expired'), [('alice', 'expired')])

    def test_session_cap_32_returns_503(self):
        browsers = [self.logged_in() for _ in range(32)]
        self.assertEqual(len(self.rt.registry.keys()), 32)
        b = Browser()
        r = b.login()
        self.assertEqual((r.status_code, r.json()),
                         (503, {'error': '현재 접속 인원이 많습니다. 잠시 후 다시 시도해 주세요.'}))
        self.assertNotIn('__Host-trid', b.c.cookies)
        self.assertEqual(Session.objects.count(), 32)
        self.assertEqual(events('login_failed'), [('alice', 'busy')])
        # idle sessions are pruned to make room, and the pruned ones cannot come back
        self.clock.t += 1800
        self.assertEqual(Browser().login().status_code, 200)
        self.assertEqual(len(self.rt.registry.keys()), 1)
        self.assertEqual(Session.objects.count(), 1)
        self.assertEqual(browsers[0].post('/api/password', {}).status_code, 401)

    def test_restart_readmits_db_session(self):
        b = self.logged_in()
        S.make_runtime(clock=self.clock)  # fresh registry, same DB: the process restarted
        self.assertEqual(b.post('/api/password', {}).status_code, 400)
        self.assertEqual(S.runtime.get().registry.keys(), {b.key})

    def test_per_minute_20_returns_429(self):
        b = self.logged_in()
        statuses = [b.post('/api/password', {}).status_code for _ in range(21)]
        self.assertEqual(statuses[:20], [400] * 20)
        self.assertEqual(statuses[20], 429)
        self.assertEqual(b.post('/api/password', {}).json(),
                         {'error': '요청이 많습니다. 1분 뒤 다시 시도해 주세요.'})
        self.clock.t += 60
        self.assertEqual(b.post('/api/password', {}).status_code, 400)

    def test_must_change_gate_blocks_load(self):
        services.create_user('carol', PW, 'user', must_change=True)
        b = Browser()
        r = b.login('carol')
        self.assertEqual((r.status_code, r.json()), (200, {'status': 'must_change'}))
        r = b.post('/api/load')
        self.assertEqual((r.status_code, r.json()), (403, {'error': MUST_CHANGE, 'must_change': True}))
        self.assertEqual(b.post('/api/ask', {'question': 'q', 'document_id': 'x'}).status_code, 403)
        self.assertIn('<meta name="must-change" content="1">', b.get('/').content.decode())
        r = b.post('/api/password', {'current': PW, 'new': NEW_PW})
        self.assertEqual((r.status_code, r.json()), (200, {'ok': True}))
        self.assertNotIn('must-change', b.get('/').content.decode())
        self.assertNotEqual(b.post('/api/load').status_code, 403)

    def test_password_errors_match_old_server(self):
        b = self.logged_in()
        r = b.post('/api/password', {'current': PW})
        self.assertEqual((r.status_code, r.json()), (400, {'error': '현재 비밀번호와 새 비밀번호를 입력해 주세요.'}))
        r = b.post('/api/password', {'current': 'nope-nope-nope', 'new': NEW_PW})
        self.assertEqual((r.status_code, r.json()), (400, {'error': '현재 비밀번호가 맞지 않습니다.'}))
        r = b.post('/api/password', {'current': PW, 'new': PW})
        self.assertEqual(r.json(), {'error': '새 비밀번호는 현재 비밀번호와 달라야 합니다.'})

    def test_lock_from_wrong_current_password_ends_all_sessions(self):
        b, other = self.logged_in(), self.logged_in()
        for _ in range(5):
            self.assertEqual(b.post('/api/password', {'current': 'nope-nope-nope', 'new': NEW_PW}).status_code,
                             400)
        self.assertEqual(self.rt.registry.keys(), set())
        self.assertEqual(Session.objects.count(), 0)
        self.assertEqual(other.post('/api/password', {}).status_code, 401)

    def test_password_change_ends_other_sessions_and_their_documents(self):
        b, other, bob = self.logged_in(), self.logged_in(), self.logged_in('bob')
        old_key, other_key = b.key, other.key
        mine, theirs = self.rt.registry.get(old_key), self.rt.registry.get(other_key)
        mine.document = theirs.document = object()
        r = b.post('/api/password', {'current': PW, 'new': NEW_PW})
        self.assertEqual((r.status_code, r.json()), (200, {'ok': True}))
        self.assertNotEqual(b.key, old_key)
        self.assertIs(self.rt.registry.get(b.key), mine)
        self.assertIsNotNone(mine.document)
        self.assertNotIn(other_key, self.rt.registry.keys())
        self.assertIsNone(theirs.document)
        self.assertEqual(self.rt.registry.keys(), {b.key, bob.key})
        self.assertFalse(Session.objects.filter(session_key=other_key).exists())
        self.assertEqual(other.post('/api/password', {}).status_code, 401)
        self.assertEqual(b.post('/api/password', {}).status_code, 400)  # same token still works
        self.assertEqual(bob.post('/api/password', {}).status_code, 400)
        self.assertEqual(Browser().login('alice', NEW_PW).status_code, 200)

    def test_admin_disable_ends_session(self):
        root = services.create_user('root', PW, 'admin', must_change=False)
        b = self.logged_in('bob')
        key = b.key
        services.admin_update(root, 'bob', disabled=True, ip=None)
        r = b.post('/api/password', {})
        self.assertEqual((r.status_code, r.json()), (401, {'error': '로그인이 필요합니다.'}))
        self.assertNotIn(key, self.rt.registry.keys())
        self.assertEqual(events('session_expired'), [('bob', 'revoked')])
        self.assertEqual(Browser().login('bob').status_code, 401)

    def test_session_store_holds_no_content(self):
        services.create_user('carol', PW, 'user', must_change=True)
        browsers = [self.logged_in('alice'), self.logged_in('carol')]
        for b in browsers:
            b.get('/')
            b.post('/api/password', {})
        self.assertEqual(Session.objects.count(), 2)
        for row in Session.objects.all():
            keys = set(row.get_decoded())
            self.assertLessEqual(keys, ALLOWED_KEYS)
            self.assertIn('_auth_user_id', keys)

    def test_x_real_ip_parsed_or_none(self):
        b = Browser()
        b.post('/api/login', {'username': 'alice', 'password': PW}, HTTP_X_REAL_IP='not-an-ip')
        Browser().login('bob')
        self.assertEqual([e.ip for e in Event.objects.filter(event='login_ok').order_by('id')],
                         [None, '10.0.0.5'])


if __name__ == '__main__':
    unittest.main()
