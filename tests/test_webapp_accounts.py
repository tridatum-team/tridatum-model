import json
import re
import unittest
from unittest import mock

import webapp_support as S  # noqa: F401  (boots Django and migrates)

from django.contrib.auth.hashers import ScryptPasswordHasher
from django.test import TestCase

from onprem_web.accounts import services
from onprem_web.accounts.models import User
from onprem_web.audit.models import Event
from trid.onprem.auth import _LAST_ADMIN

PW = S.PW
T0 = 1_700_000_000.0
_SAFE = re.compile(r'[A-Za-z0-9._:-]+')


def events(event=None):
    qs = Event.objects.order_by('id')
    if event is not None:
        qs = qs.filter(event=event)
    return [(e.username, e.ip, e.event, e.result, json.loads(e.detail)) for e in qs]


class AccountTests(TestCase):
    def setUp(self):
        self.alice = services.create_user('alice', PW, 'user', must_change=False)

    def fresh(self, name='alice'):
        return User.objects.get(username=name)

    def test_five_failures_lock_for_15_minutes(self):
        for _ in range(5):
            self.assertIsNone(services.attempt_login('alice', 'wrong-password', '10.0.0.1', now=T0))
        self.assertEqual(self.fresh().locked_until, T0 + 900)
        self.assertIsNone(services.attempt_login('alice', PW, '10.0.0.1', now=T0))
        self.assertEqual(len(events('login_locked')), 1)
        # a locked attempt does not extend the lock
        self.assertIsNone(services.attempt_login('alice', 'wrong-password', None, now=T0 + 10))
        self.assertEqual(self.fresh().locked_until, T0 + 900)
        self.assertIsNone(services.attempt_login('alice', PW, '10.0.0.1', now=T0 + 500))
        user = services.attempt_login('alice', PW, '10.0.0.1', now=T0 + 900)
        self.assertEqual(user.username, 'alice')
        self.assertEqual((self.fresh().failed, self.fresh().locked_until), (0, 0.0))
        self.assertEqual(events('login_ok'), [('alice', '10.0.0.1', 'login_ok', 'ok', {})])

    def test_failures_after_lock_expiry_restart_count_and_success_resets(self):
        for _ in range(5):
            services.attempt_login('alice', 'wrong-password', None, now=T0)
        self.assertIsNone(services.attempt_login('alice', 'wrong-password', None, now=T0 + 901))
        self.assertEqual(self.fresh().failed, 1)
        for _ in range(3):
            services.attempt_login('alice', 'wrong-password', None, now=T0 + 902)
        self.assertIsNotNone(services.attempt_login('alice', PW, None, now=T0 + 903))
        self.assertEqual(self.fresh().failed, 0)

    def test_unknown_locked_disabled_all_return_none_and_log(self):
        self.assertIsNone(services.attempt_login('nobody', 'whatever-pw', '10.0.0.1', now=T0))
        self.assertIsNone(services.attempt_login('Bad Name!', 'whatever-pw', None, now=T0))
        self.assertIsNone(services.attempt_login(None, 'whatever-pw', None, now=T0))
        self.assertEqual(events('login_failed'), [
            (None, '10.0.0.1', 'login_failed', 'invalid', {}),
            (None, None, 'login_failed', 'invalid', {}),
            (None, None, 'login_failed', 'invalid', {})])
        services.create_user('bob', PW, 'user', must_change=False)
        User.objects.filter(username='bob').update(is_active=False)
        self.assertIsNone(services.attempt_login('bob', PW, '10.0.0.2', now=T0))
        self.assertEqual(events('login_failed')[-1],
                         ('bob', '10.0.0.2', 'login_failed', 'invalid', {'error': 'disabled'}))
        User.objects.filter(username='alice').update(failed=5, locked_until=T0 + 900)
        self.assertIsNone(services.attempt_login('alice', PW, '10.0.0.3', now=T0))
        self.assertEqual(events('login_locked'),
                         [('alice', '10.0.0.3', 'login_locked', 'locked', {})])

    def test_unknown_user_runs_dummy_hash(self):
        services._dummy_hash()  # build the dummy outside the patch
        with mock.patch.object(ScryptPasswordHasher, 'verify', autospec=True,
                               return_value=False) as verify:
            self.assertIsNone(services.attempt_login('nobody', 'whatever-pw', None, now=T0))
        self.assertEqual(verify.call_count, 1)

    def test_password_rules_10_to_1024(self):
        for bad in ('short', 'x' * 9, 'x' * 1025, None):
            with self.subTest(pw=bad), self.assertRaises(ValueError):
                services.create_user('carol', bad, 'user')
        services.create_user('carol', 'x' * 10, 'user')
        services.create_user('dave', 'x' * 1024, 'user')
        with self.assertRaises(ValueError) as cm:
            services.create_user('carol', PW, 'user')
        self.assertEqual(str(cm.exception), '이미 있는 계정입니다.')
        with self.assertRaises(ValueError) as cm:
            services.create_user('erin', PW, 'root')
        self.assertEqual(str(cm.exception), '알 수 없는 역할입니다.')
        with self.assertRaises(ValueError) as cm:
            services.change_password(self.alice, 'wrong-password', 'brand new pass 2', None)
        self.assertEqual(str(cm.exception), '현재 비밀번호가 맞지 않습니다.')
        self.assertEqual(events('password_changed'),
                         [('alice', None, 'password_changed', 'invalid', {})])
        self.assertEqual(self.fresh().failed, 1)
        for bad in ('short', 'x' * 1025):
            with self.subTest(new=bad), self.assertRaises(ValueError):
                services.change_password(self.alice, PW, bad, None)
        with self.assertRaises(ValueError) as cm:
            services.change_password(self.alice, PW, PW, None)
        self.assertEqual(str(cm.exception), '새 비밀번호는 현재 비밀번호와 달라야 합니다.')
        self.assertEqual(self.fresh().password_version, 0)
        services.change_password(self.alice, PW, 'x' * 1024, '10.0.0.1')
        user = self.fresh()
        self.assertEqual((user.password_version, user.must_change, user.failed), (1, False, 0))
        self.assertTrue(user.check_password('x' * 1024))
        self.assertEqual(events('password_changed')[-1],
                         ('alice', '10.0.0.1', 'password_changed', 'ok', {}))

    def test_username_normalized(self):
        user = services.create_user(' Alice2 ', PW, 'user')
        self.assertEqual(user.username, 'alice2')
        self.assertIsNotNone(services.attempt_login(' ALICE ', PW, None, now=T0))
        self.assertEqual(events('login_ok')[0][0], 'alice')
        with self.assertRaises(ValueError) as cm:
            services.create_user(' Alice ', PW, 'user')
        self.assertEqual(str(cm.exception), '이미 있는 계정입니다.')
        with self.assertRaises(ValueError):
            services.create_user('bad name!', PW, 'user')

    def test_last_admin_cannot_be_disabled_or_demoted(self):
        root = services.create_user('root', PW, 'admin', must_change=False)
        for kwargs in (dict(disabled=True), dict(role='user')):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as cm:
                    services.admin_update(root, 'root', ip='10.0.0.1', **kwargs)
                self.assertEqual(str(cm.exception), _LAST_ADMIN)
        self.assertEqual(self.fresh('root').role, 'admin')
        with self.assertRaises(PermissionError):
            services.admin_update(self.alice, 'alice', role='admin', ip=None)
        with self.assertRaises(ValueError) as cm:
            services.admin_update(root, 'ghost', disabled=True, ip=None)
        self.assertEqual(str(cm.exception), '없는 계정입니다.')
        services.admin_create(root, 'root2', PW, 'admin', '10.0.0.1')
        self.assertEqual(events('admin_create_user'),
                         [('root', '10.0.0.1', 'admin_create_user', 'ok',
                           {'action': 'create', 'target': 'root2'})])
        services.admin_update(root, 'root', role='user', ip='10.0.0.1')
        self.assertEqual(self.fresh('root').role, 'user')
        self.assertEqual(events('admin_update_user'),
                         [('root', '10.0.0.1', 'admin_update_user', 'ok',
                           {'action': 'role', 'target': 'root'})])

    def test_reset_sets_must_change_and_bumps_version(self):
        root = services.create_user('root', PW, 'admin', must_change=False)
        old_hash = self.fresh().get_session_auth_hash()
        services.admin_update(root, 'alice', reset_password='temporary pass 3', ip='10.0.0.1')
        user = self.fresh()
        self.assertEqual((user.must_change, user.password_version), (True, 1))
        self.assertNotEqual(user.get_session_auth_hash(), old_hash)
        self.assertEqual(services.attempt_login('alice', 'temporary pass 3', None, now=T0).username,
                         'alice')
        self.assertEqual(events('admin_reset_password'),
                         [('root', '10.0.0.1', 'admin_reset_password', 'ok',
                           {'action': 'reset', 'target': 'alice'})])
        # version alone changes the session hash, with the password hash unchanged
        before = user.get_session_auth_hash()
        user.password_version += 1
        self.assertNotEqual(user.get_session_auth_hash(), before)
        services.admin_update(root, 'alice', disabled=True, ip=None)
        self.assertFalse(self.fresh().is_active)
        services.admin_update(root, 'alice', disabled=False, ip=None)
        User.objects.filter(username='alice').update(failed=5, locked_until=T0 + 900)
        services.admin_update(root, 'alice', unlock=True, ip=None)
        self.assertEqual((self.fresh().failed, self.fresh().locked_until), (0, 0.0))
        self.assertEqual([e[4]['action'] for e in events('admin_update_user')],
                         ['disable', 'enable', 'unlock'])

    def test_password_hash_is_scrypt(self):
        self.assertTrue(self.alice.password.startswith('scrypt$'))
        self.assertTrue(self.fresh().password.startswith('scrypt$'))

    def test_no_free_text_in_audit_for_login_failures(self):
        secret = 'hunter2-secret-pw'
        attempts = ['alice', 'Bad Name!', 'evil"user', '이름', 'x' * 40, 'ghost user']
        for name in attempts:
            services.attempt_login(name, secret, '10.0.0.9', now=T0)
        services.attempt_login('alice', 'y' * 2000, None, now=T0)
        for e in Event.objects.all():
            row = ' '.join(str(v) for v in (e.username, e.ip, e.event, e.result, e.detail))
            self.assertNotIn(secret, row)
            self.assertNotIn('y' * 20, row)
            if e.username is not None:
                self.assertRegex(e.username, _SAFE)
                self.assertIn(e.username, ('alice',))
        self.assertEqual(Event.objects.filter(username__isnull=True).count(), 5)


if __name__ == '__main__':
    unittest.main()
