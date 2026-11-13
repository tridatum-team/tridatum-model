import json
import unittest

from tests import test_onprem_server as base
from tests.test_onprem_server import PW, Client

LAST_ADMIN = '마지막 관리자는 잠그거나 권한을 바꿀 수 없습니다.'
NEW_PW = 'another pass 22'


class AdminTests(unittest.TestCase):
    # Reuse the server fixture without inheriting (and re-running) its tests.
    config_overrides = {}
    answer = staticmethod(base.ServerTests.answer)
    client = base.ServerTests.client
    events = base.ServerTests.events

    def setUp(self):
        base.ServerTests.setUp(self)
        self.store.create_user('root', PW, 'admin', must_change=False)
        self.store.create_user('root2', PW, 'admin', must_change=False)

    def rows(self, c, path, body=None):
        resp, data = c.request('POST', path, body or {})
        return resp.status, (json.loads(data) if resp.status == 200 else data)

    def admin(self, name='root'):
        return self.client(name)

    def test_non_admin_gets_403_on_every_admin_route(self):
        c = self.client('alice')
        for path in ('/api/admin/users/list', '/api/admin/users/create', '/api/admin/users/update',
                     '/api/admin/events'):
            self.assertEqual(c.post(path, {})[0], 403, path)
        self.assertEqual(c.request('GET', '/admin')[0].status, 403)
        self.assertEqual(c.request('GET', f'/api/admin/events.csv?token={c.csrf}')[0].status, 403)
        anon = Client(self.server)
        self.assertEqual(anon.post('/api/admin/users/list')[0], 401)
        resp, data = anon.request('GET', '/admin')
        self.assertIn('action="/api/login"', data.decode())
        self.assertEqual(anon.request('GET', '/api/admin/events.csv?token=x')[0].status, 401)

    def test_admin_page_served_to_admin(self):
        c = self.admin()
        resp, data = c.request('GET', '/admin')
        self.assertEqual(resp.status, 200)
        self.assertIn(c.csrf, data.decode())

    def test_admin_creates_user_who_must_change_password(self):
        c = self.admin()
        status, value, _ = c.post('/api/admin/users/create',
                                  {'username': ' Carol ', 'role': 'user', 'password': 'temporary pass 1'})
        self.assertEqual(status, 200, value)
        self.assertTrue(self.store.get_user('carol').must_change)
        carol = Client(self.server)
        self.assertEqual(carol.login('carol', 'temporary pass 1')[1], {'status': 'must_change'})
        self.assertEqual(carol.post('/api/ask', {'question': 'q', 'document_id': 'x'})[0], 403)
        ev = self.events('admin_create_user')
        self.assertEqual(ev[0]['detail'], {'target': 'carol', 'action': 'create'})
        self.assertEqual(c.post('/api/admin/users/create',
                                {'username': 'dave', 'role': 'user', 'password': 'short'})[0], 400)
        self.assertEqual(c.post('/api/admin/users/create',
                                {'username': 'carol', 'role': 'user', 'password': 'temporary pass 1'})[0], 400)
        self.assertEqual(c.post('/api/admin/users/create', {'username': 'x'})[0], 400)

    def test_admin_reset_password_ends_target_sessions(self):
        c = self.admin()
        alice = self.client('alice')
        self.assertEqual(alice.post('/api/clear')[0], 200)
        status, value, _ = c.post('/api/admin/users/update',
                                  {'username': 'alice', 'reset_password': NEW_PW})
        self.assertEqual(status, 200, value)
        self.assertEqual(alice.post('/api/clear')[0], 401)
        self.assertTrue(self.store.get_user('alice').must_change)
        self.assertEqual(Client(self.server).login('alice', NEW_PW)[1], {'status': 'must_change'})
        self.assertTrue(self.events('admin_reset_password'))

    def test_admin_unlock_clears_lock(self):
        c = self.admin()
        for _ in range(5):
            Client(self.server).login('alice', 'wrong password!')
        listed = self.rows(c, '/api/admin/users/list')[1]
        self.assertTrue(next(u for u in listed if u['username'] == 'alice')['locked'])
        self.assertEqual(Client(self.server).login('alice', PW)[0], 401)
        self.assertEqual(c.post('/api/admin/users/update', {'username': 'alice', 'unlock': True})[0], 200)
        listed = self.rows(c, '/api/admin/users/list')[1]
        self.assertFalse(next(u for u in listed if u['username'] == 'alice')['locked'])
        self.assertEqual(Client(self.server).login('alice', PW)[0], 200)
        self.assertIn({'target': 'alice', 'action': 'unlock'},
                      [e['detail'] for e in self.events('admin_update_user')])

    def test_update_unknown_user_404_and_bad_fields_400(self):
        c = self.admin()
        self.assertEqual(c.post('/api/admin/users/update', {'username': 'nobody', 'disabled': True})[0], 404)
        self.assertEqual(c.post('/api/admin/users/update', {'username': 'alice', 'role': 'root'})[0], 400)
        self.assertEqual(c.post('/api/admin/users/update', {'username': 'alice', 'disabled': 'yes'})[0], 400)
        self.assertEqual(c.post('/api/admin/users/update', {'role': 'user'})[0], 400)

    def test_last_admin_demotion_rejected_over_http(self):
        c = self.admin()
        self.assertEqual(c.post('/api/admin/users/update', {'username': 'root2', 'disabled': True})[0], 200)
        status, value, _ = c.post('/api/admin/users/update', {'username': 'root', 'role': 'user'})
        self.assertEqual((status, value['error']), (400, LAST_ADMIN))
        self.assertEqual(self.store.get_user('root').role, 'admin')

    def test_users_list_never_contains_hash(self):
        c = self.admin()
        status, value = self.rows(c, '/api/admin/users/list')
        self.assertEqual(status, 200)
        self.assertEqual(set(value[0]), {'username', 'role', 'disabled', 'locked', 'must_change'})
        raw = c.request('POST', '/api/admin/users/list', {})[1].decode()
        self.assertNotIn('scrypt', raw)
        for u in self.store.list_users():
            self.assertNotIn(u.pw_hash, raw)

    def test_events_query_limits_and_filters(self):
        c = self.admin()
        for _ in range(205):
            self.store.log('clear', 'alice', None, 'ok')
        status, rows = self.rows(c, '/api/admin/events')
        self.assertEqual((status, len(rows)), (200, 200))
        self.assertEqual(len(self.rows(c, '/api/admin/events',
                                    {'username': 'alice', 'event': 'clear', 'offset': 200})[1]), 5)
        self.assertEqual(c.post('/api/admin/events', {'event': 'nope'})[0], 400)
        self.assertEqual(c.post('/api/admin/events', {'since': 'x'})[0], 400)
        self.assertEqual(c.post('/api/admin/events', {'offset': -1})[0], 400)

    def test_events_csv_has_bom_and_logs_export(self):
        c = self.admin()
        self.store.log('clear', 'alice', None, 'ok')
        resp, data = c.request('GET', f'/api/admin/events.csv?token={c.csrf}&event=clear')
        self.assertEqual(resp.status, 200)
        self.assertTrue(data.startswith(b'\xef\xbb\xbf'))
        self.assertEqual(resp.getheader('Content-Type'), 'text/csv; charset=utf-8')
        self.assertRegex(resp.getheader('Content-Disposition'),
                         r'^attachment; filename="trid-usage-\d{8}\.csv"$')
        self.assertIn('clear', data.decode('utf-8-sig'))
        ev = self.events('admin_export')
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]['detail'], {'count': 1})
        self.assertEqual(c.request('GET', '/api/admin/events.csv?token=wrong')[0].status, 403)
        self.assertEqual(c.request('GET', '/api/admin/events.csv')[0].status, 403)
        self.assertEqual(c.request('GET', f'/api/admin/events.csv?token={c.csrf}&since=abc')[0].status, 400)
        self.assertEqual(len(self.events('admin_export')), 1)

    def test_no_route_deletes_or_edits_events(self):
        c = self.admin()
        for path in ('/api/admin/events/delete', '/api/admin/events/update', '/api/admin/events/purge'):
            self.assertEqual(c.post(path, {})[0], 404, path)
        for method in ('DELETE', 'PUT'):
            try:
                resp, _ = c.request(method, '/api/admin/events')
                self.assertGreaterEqual(resp.status, 400)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()
