"""Korean admin screens: /admin, /ui/admin/users, /ui/admin/events and the CSV export.

Ports the scenarios of test_onprem_admin.py to the Django build.
"""

import json
import re
from urllib.parse import urlencode

import webapp_support as S  # noqa: F401  (boots Django and migrates)
from test_webapp_ui import UiBrowser
from test_webapp_workspace import PDF, fake_parse

from unittest import mock

from django.test import TestCase
from django.urls import get_resolver

from onprem_web.accounts import services as accounts
from onprem_web.accounts.models import User
from onprem_web.audit import services as audit
from onprem_web.audit.models import Event
from trid.onprem.admin import NOT_FOUND_USER, PAGE_SIZE

PW = S.PW
NEW_PW = 'another horse 22'
ADMIN_ONLY = '관리자 권한이 필요합니다.'
LAST_ADMIN = '마지막 관리자는 잠그거나 권한을 바꿀 수 없습니다.'
SHA = 'ab' * 32


def details(event):
    return [json.loads(e.detail) for e in Event.objects.filter(event=event).order_by('id')]


def body_rows(html: str) -> int:
    tbody = re.search(r'<tbody id="events">(.*?)</tbody>', html, re.S).group(1)
    return len(re.findall(r'<tr[\s>]', tbody))


class AdminBrowser(UiBrowser):
    def form(self, path, data, token=True, **extra):
        """POST as htmx does (see UiBrowser.form); `extra` adds headers such as HX-Prompt."""
        if token:
            extra.update(HTTP_X_TRID_TOKEN=self.token, HTTP_HX_REQUEST='true')
        if not data:
            extra['CONTENT_LENGTH'] = '0'
        return self.c.post(path, urlencode(data), content_type='application/x-www-form-urlencoded',
                           secure=True, **extra)

    def csv(self, data, token=True):
        """The CSV form: a plain browser form POST with the token in csrfmiddlewaretoken."""
        fields = dict(data)
        if token:
            fields['csrfmiddlewaretoken'] = self.token
        return self.c.post('/api/admin/events.csv', urlencode(fields),
                           content_type='application/x-www-form-urlencoded', secure=True)


class AdminTests(TestCase):
    @classmethod
    def setUpTestData(cls):  # once per class: each scrypt hash costs about 0.7 s
        accounts.create_user('alice', PW, 'user', must_change=False)
        accounts.create_user('root', PW, 'admin', must_change=False)
        accounts.create_user('root2', PW, 'admin', must_change=False)

    def setUp(self):
        self.rt = S.make_runtime(lock_wait=0.1)
        patcher = mock.patch('onprem_web.workspace.services.parse_pdf', side_effect=fake_parse)
        patcher.start()
        self.addCleanup(patcher.stop)

    def browser(self, name='root'):
        b = AdminBrowser()
        self.assertEqual(b.login(name).status_code, 200)
        return b

    # -- access ----------------------------------------------------------------

    def test_staff_cannot_open_admin_or_csv(self):
        alice = self.browser('alice')
        r = alice.get('/admin')
        self.assertEqual((r.status_code, r.json()), (403, {'error': ADMIN_ONLY}))
        for path, data in (('/ui/admin/users', {'username': 'x1', 'password': NEW_PW, 'role': 'user'}),
                           ('/ui/admin/users/alice', {'disabled': '1'})):
            r = alice.form(path, data)
            self.assertEqual((r.status_code, r.json()), (403, {'error': ADMIN_ONLY}), path)
        r = alice.ui_get('/ui/admin/events')
        self.assertEqual((r.status_code, r.json()), (403, {'error': ADMIN_ONLY}))
        r = alice.csv({})
        self.assertEqual((r.status_code, r.json()), (403, {'error': ADMIN_ONLY}))
        self.assertFalse(User.objects.filter(username='x1').exists())
        self.assertTrue(User.objects.get(username='alice').is_active)
        self.assertEqual(details('admin_export'), [])

        anon = AdminBrowser()
        r = anon.get('/admin')
        self.assertEqual(r.status_code, 200)
        self.assertIn('action="/api/login"', r.content.decode())
        self.assertEqual(anon.c.post('/api/admin/events.csv', '', content_type='application/x-www-form-urlencoded',
                                     secure=True).status_code, 401)
        self.assertEqual(anon.ui_get('/ui/admin/events').status_code, 401)

    def test_admin_page_matches_old_layout(self):
        b = self.browser()
        r = b.get('/admin')
        self.assertEqual(r.status_code, 200)
        page = r.content.decode()
        for needle in ('<title>TRID 관리자</title>', 'id="users-title"', 'id="create-form"', 'id="events-title"',
                       'id="filter-form"', 'id="csv-form"', 'id="users-message"', 'id="events-message"',
                       '새 계정과 초기화된 계정은 첫 로그인 때 비밀번호를 바꿔야 합니다.',
                       '<option value="clear">자료 지우기</option>', '<td>alice</td>', '<td>직원</td>',
                       'href="/">직원 화면으로</a>'):
            self.assertIn(needle, page)
        found = re.findall(r'<script[^>]*\ssrc="([^"]+)"', page) + re.findall(r'<link[^>]*href="([^"]+)"', page)
        self.assertIn('/static/admin.js', found)
        for src in found:
            self.assertEqual(b.c.get(src, secure=True).status_code, 200, src)
        self.assertRegex(page, r'<meta name="csrf-token" content="[A-Za-z0-9]{64}">')
        self.assertRegex(page, r'name="csrfmiddlewaretoken" value="[A-Za-z0-9]{64}"')
        self.assertNotIn('scrypt', page)
        self.assertNotIn('pbkdf2', page)

    # -- accounts ----------------------------------------------------------------

    def test_create_logs_target_and_action(self):
        b = self.browser()
        r = b.form('/ui/admin/users', {'username': 'Carol', 'password': 'temporary pass 1', 'role': 'user'})
        self.assertEqual(r.status_code, 200, r.content)
        html = r.content.decode()
        self.assertIn('carol 계정을 만들었습니다. 첫 로그인 때 비밀번호를 바꿉니다.', html)
        self.assertIn('<td>carol</td>', html)
        self.assertTrue(User.objects.get(username='carol').must_change)
        self.assertEqual(details('admin_create_user'), [{'target': 'carol', 'action': 'create'}])
        for data, message in (
                ({'username': 'carol', 'password': 'temporary pass 1', 'role': 'user'}, '이미 있는 계정입니다.'),
                ({'username': 'dave', 'password': 'temporary pass 1', 'role': 'root'}, '알 수 없는 역할입니다.'),
                ({'username': 'dave', 'role': 'user'}, 'password 값이 올바르지 않습니다.')):
            r = b.form('/ui/admin/users', data)
            self.assertEqual((r.status_code, r.json()), (400, {'error': message}), data)
        self.assertEqual(len(details('admin_create_user')), 1)

    def test_update_actions(self):
        b = self.browser()
        r = b.form('/ui/admin/users/alice', {'disabled': '1'})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertIn('alice 계정을 잠갔습니다.', r.content.decode())
        self.assertFalse(User.objects.get(username='alice').is_active)
        r = b.form('/ui/admin/users/alice', {'disabled': '0', 'unlock': '1'})
        self.assertIn('alice 계정의 잠금을 풀었습니다.', r.content.decode())
        self.assertTrue(User.objects.get(username='alice').is_active)
        r = b.form('/ui/admin/users/alice', {'role': 'admin'})
        self.assertIn('alice의 역할을 바꿨습니다.', r.content.decode())
        self.assertEqual(User.objects.get(username='alice').role, 'admin')
        r = b.form('/ui/admin/users/alice', {'reset_password': NEW_PW})
        self.assertIn('alice의 비밀번호를 초기화했습니다.', r.content.decode())
        self.assertTrue(User.objects.get(username='alice').must_change)
        self.assertEqual(details('admin_reset_password'), [{'target': 'alice', 'action': 'reset'}])
        self.assertEqual([d['action'] for d in details('admin_update_user')], ['disable', 'enable', 'unlock', 'role'])  # old unlock: enable + unlock

        r = b.form('/ui/admin/users/nobody', {'disabled': '1'})
        self.assertEqual((r.status_code, r.json()), (404, {'error': NOT_FOUND_USER}))
        r = b.form('/ui/admin/users/alice', {'role': 'root'})
        self.assertEqual((r.status_code, r.json()), (400, {'error': '알 수 없는 역할입니다.'}))
        r = b.form('/ui/admin/users/alice', {'disabled': 'yes'})
        self.assertEqual((r.status_code, r.json()), (400, {'error': '요청 값이 올바르지 않습니다.'}))
        r = b.form('/ui/admin/users/alice', {'reset_password': 'short'})
        self.assertEqual(r.status_code, 400)

    def test_last_admin_message_shown(self):
        b = self.browser()
        self.assertEqual(b.form('/ui/admin/users/root2', {'disabled': '1'}).status_code, 200)
        r = b.form('/ui/admin/users/root', {'role': 'user'})
        self.assertEqual((r.status_code, r.json()), (400, {'error': LAST_ADMIN}))
        self.assertEqual(User.objects.get(username='root').role, 'admin')

    def test_user_rows_show_old_status_text_and_actions(self):
        accounts.create_user('carol', PW, 'user', must_change=True)
        b = self.browser()
        b.form('/ui/admin/users/alice', {'disabled': '1'})
        html = b.get('/admin').content.decode()
        self.assertIn('<td>잠금(관리자)</td>', html)
        self.assertIn('<td>사용 가능 · 비밀번호 변경 필요</td>', html)
        self.assertIn('관리자로 변경', html)
        self.assertIn('직원로 변경', html)  # the old admin.js text: `${ROLE_LABELS[next]}로 변경`

    # -- usage log ---------------------------------------------------------------

    def test_events_filter_and_paging(self):
        for _ in range(PAGE_SIZE + 5):
            audit.log('clear', 'alice', None, 'ok')
        b = self.browser()
        r = b.ui_get('/ui/admin/events')
        self.assertEqual(r.status_code, 200)
        html = r.content.decode()
        self.assertEqual(body_rows(html), PAGE_SIZE)
        self.assertIn(f'1–{PAGE_SIZE}번째', html)
        r = b.ui_get(f'/ui/admin/events?username=alice&event=clear&offset={PAGE_SIZE}')
        html = r.content.decode()
        self.assertEqual(body_rows(html), 5)
        self.assertIn(f'{PAGE_SIZE + 1}–{PAGE_SIZE + 5}번째', html)
        r = b.ui_get('/ui/admin/events?username=bob')
        self.assertIn('기록이 없습니다.', r.content.decode())
        r = b.ui_get('/ui/admin/events?since=4102444800')
        self.assertEqual(body_rows(r.content.decode()), 0)
        for query, message in (('event=nope', '알 수 없는 이벤트입니다.'), ('since=x', 'since 값이 올바르지 않습니다.'),
                               ('until=inf', 'until 값이 올바르지 않습니다.'),
                               ('offset=-1', 'offset 값이 올바르지 않습니다.')):
            r = b.ui_get('/ui/admin/events?' + query)
            self.assertEqual((r.status_code, r.json()), (400, {'error': message}), query)

    def test_event_detail_cell_shortens_sha256(self):
        audit.log('pdf_upload', 'alice', None, 'ok', sha256=SHA, pages=3)
        html = self.browser().ui_get('/ui/admin/events?event=pdf_upload').content.decode()
        self.assertIn(f'title="pages: 3 · sha256: {SHA}"', html)
        self.assertIn(f'pages: 3 · sha256: {SHA[:12]}…</td>', html)
        self.assertIn(f'data-ts="', html)
        self.assertIn('<td class="nowrap">PDF 올리기</td>', html)

    def test_csv_requires_post_with_token(self):
        audit.log('clear', 'alice', None, 'ok')
        b = self.browser()
        self.assertEqual(b.c.get('/api/admin/events.csv', secure=True).status_code, 405)
        self.assertEqual(b.csv({'event': 'clear'}, token=False).status_code, 403)
        r = b.csv({'event': 'clear'})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertTrue(r.content.startswith(b'\xef\xbb\xbf'))
        self.assertEqual(r['Content-Type'], 'text/csv; charset=utf-8')
        self.assertRegex(r['Content-Disposition'], r'^attachment; filename="trid-usage-\d{8}\.csv"$')
        self.assertIn('clear', r.content.decode('utf-8-sig'))
        self.assertEqual(details('admin_export'), [{'count': 1}])
        r = b.csv({'since': 'abc'})
        self.assertEqual((r.status_code, r.json()), (400, {'error': 'since 값이 올바르지 않습니다.'}))
        self.assertEqual(len(details('admin_export')), 1)

    def test_admin_cannot_see_staff_documents(self):
        alice = self.browser('alice')
        r = alice.c.post('/api/load', PDF, content_type='application/pdf', secure=True,
                         HTTP_X_TRID_TOKEN=alice.token)
        self.assertEqual(r.status_code, 200)
        b = self.browser()
        for r in (b.get('/admin'), b.ui_get('/ui/admin/events')):
            self.assertEqual(r.status_code, 200)
            self.assertNotIn('본문 한 쪽', r.content.decode())

    def test_no_edit_or_delete_route_for_events(self):
        patterns = [str(p.pattern) for p in get_resolver().url_patterns]
        for p in patterns:
            if 'events' in p:
                self.assertNotRegex(p, r'delete|update|purge', p)
        b = self.browser()
        for path in ('/api/admin/events/delete', '/ui/admin/events/delete', '/ui/admin/events/update'):
            self.assertEqual(b.form(path, {}).status_code, 404, path)


class LabelTests(TestCase):
    def test_labels_copied_from_old_admin_js(self):
        from pathlib import Path
        from onprem_web.staffadmin import views
        text = (Path(__file__).resolve().parent.parent / 'trid' / 'onprem_ui' / 'admin.js').read_text(encoding='utf-8')
        for name, table in (('EVENT_LABELS', views.EVENT_LABELS), ('ROLE_LABELS', views.ROLE_LABELS)):
            body = re.search(r'const %s = \{(.*?)\};' % name, text, re.S).group(1)
            self.assertEqual(dict(re.findall(r"(\w+): '([^']*)'", body)), table, name)
