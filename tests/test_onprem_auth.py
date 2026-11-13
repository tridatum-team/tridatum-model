import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trid.onprem import auth as auth_mod
from trid.onprem.auth import Auth
from trid.onprem.store import Store

PW = "correct horse 1"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mono = Clock(0.0)
        self.wall = Clock(1_700_000_000.0)
        self.store = Store(Path(self.tmp.name) / "app.db", clock=self.wall)
        self.addCleanup(self.store.close)
        self.auth = Auth(self.store, clock=self.mono, wall=self.wall)
        self.store.create_user("alice", PW, "user", must_change=False)

    def advance(self, secs):
        self.mono.t += secs
        self.wall.t += secs

    def events(self, event=None):
        return self.store.query_events(event=event)

    def test_lock_after_five_failures_for_fifteen_minutes(self):
        for _ in range(5):
            self.assertEqual(self.auth.login("alice", "wrong-password", "10.0.0.1")[0], "invalid")
        self.assertEqual(self.auth.login("alice", PW, "10.0.0.1")[0], "locked")
        self.assertEqual(len(self.events("login_locked")), 1)
        self.advance(500)
        self.assertEqual(self.auth.login("alice", PW, "10.0.0.1")[0], "locked")
        self.advance(400)
        status, sess = self.auth.login("alice", PW, "10.0.0.1")
        self.assertEqual(status, "ok")
        self.assertIsNotNone(sess)

    def test_locked_attempt_does_not_extend_lock(self):
        for _ in range(5):
            self.auth.login("alice", "wrong-password", None)
        until = self.store.get_user("alice").locked_until
        self.advance(10)
        self.auth.login("alice", "wrong-password", None)
        self.assertEqual(self.store.get_user("alice").locked_until, until)

    def test_failures_after_lock_expiry_restart_count(self):
        for _ in range(5):
            self.auth.login("alice", "wrong-password", None)
        self.advance(901)
        self.assertEqual(self.auth.login("alice", "wrong-password", None)[0], "invalid")
        self.assertEqual(self.store.get_user("alice").failed, 1)

    def test_success_resets_failure_count(self):
        for _ in range(4):
            self.auth.login("alice", "wrong-password", "10.0.0.1")
        self.assertEqual(self.auth.login("alice", PW, "10.0.0.1")[0], "ok")
        self.assertEqual(self.store.get_user("alice").failed, 0)
        for _ in range(4):
            self.auth.login("alice", "wrong-password", "10.0.0.1")
        self.assertEqual(self.auth.login("alice", PW, "10.0.0.1")[0], "ok")

    def test_unknown_user_same_response_logged_as_unknown(self):
        with mock.patch.object(auth_mod, "verify_password", return_value=False) as vp:
            self.assertEqual(self.auth.login("nobody", "whatever-pw", "10.0.0.1"),
                             ("invalid", None))
            self.assertEqual(self.auth.login("Bad Name!", "whatever-pw", None),
                             ("invalid", None))
        self.assertEqual(vp.call_count, 2)
        failed = self.events("login_failed")
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(e["username"] is None for e in failed))
        self.assertIn("(unknown)", self.store.export_csv())

    def test_disabled_account_is_invalid(self):
        self.store.update_user("alice", disabled=True)
        self.assertEqual(self.auth.login("alice", PW, "10.0.0.1"), ("invalid", None))
        ev = self.events("login_failed")[0]
        self.assertEqual(ev["username"], "alice")
        self.assertEqual(ev["detail"], {"error": "disabled"})

    def test_idle_session_expires_after_1800s_and_drops_document(self):
        _, sess = self.auth.login("alice", PW, "10.0.0.1")
        sess.document = object()
        sess.document_id = "abc"
        self.advance(1000)
        self.assertIs(self.auth.session(sess.sid), sess)  # touch
        self.advance(1799)
        self.assertIs(self.auth.session(sess.sid), sess)
        self.advance(1800)
        self.assertIsNone(self.auth.session(sess.sid))
        self.assertIsNone(sess.document)
        self.assertIsNone(sess.document_id)
        self.assertEqual(len(self.events("session_expired")), 1)
        self.assertIsNone(self.auth.session(sess.sid))

    def test_prune_expires_idle_sessions(self):
        _, sess = self.auth.login("alice", PW, "10.0.0.1")
        self.advance(1800)
        self.auth.prune()
        self.assertEqual(len(self.events("session_expired")), 1)
        self.assertIsNone(self.auth.session(sess.sid))

    def test_must_change_on_first_login(self):
        self.store.create_user("bob", PW, "user")
        status, sess = self.auth.login("Bob", PW, "10.0.0.1")
        self.assertEqual(status, "must_change")
        self.assertTrue(sess.must_change)
        self.assertEqual(sess.username, "bob")
        self.auth.change_password(sess, PW, "brand new pass 2", "10.0.0.1")
        self.assertFalse(sess.must_change)
        self.assertEqual(sess.password_version, 2)
        self.assertEqual(self.auth.login("bob", "brand new pass 2", None)[0], "ok")

    def test_change_password_validates(self):
        _, sess = self.auth.login("alice", PW, "10.0.0.1")
        with self.assertRaises(ValueError):
            self.auth.change_password(sess, "wrong-password", "brand new pass 2", None)
        with self.assertRaises(ValueError):
            self.auth.change_password(sess, PW, "short", None)
        self.assertEqual(self.store.get_user("alice").password_version, 1)

    def test_password_change_ends_other_sessions(self):
        _, s1 = self.auth.login("alice", PW, "10.0.0.1")
        _, s2 = self.auth.login("alice", PW, "10.0.0.2")
        s2.document = object()
        self.auth.change_password(s1, PW, "brand new pass 2", "10.0.0.1")
        self.assertIsNone(self.auth.session(s2.sid))
        self.assertIsNone(s2.document)
        self.assertIs(self.auth.session(s1.sid), s1)
        self.assertEqual(len(self.events("password_changed")), 1)

    def test_logout(self):
        _, sess = self.auth.login("alice", PW, "10.0.0.1")
        sess.document = object()
        self.auth.logout(sess.sid, "10.0.0.1")
        self.assertIsNone(self.auth.session(sess.sid))
        self.assertIsNone(sess.document)
        self.assertEqual(len(self.events("logout")), 1)
        self.auth.logout("nonexistent", None)

    def test_disabled_user_or_version_change_invalidates_session(self):
        _, sess = self.auth.login("alice", PW, "10.0.0.1")
        self.store.update_user("alice", disabled=True)
        self.assertIsNone(self.auth.session(sess.sid))
        self.store.update_user("alice", disabled=False)
        _, s2 = self.auth.login("alice", PW, None)
        self.store.update_user("alice", password_version=5)
        self.assertIsNone(self.auth.session(s2.sid))

    def test_cannot_lock_or_demote_last_active_admin(self):
        self.store.create_user("root", PW, "admin", must_change=False)
        _, adm = self.auth.login("root", PW, "10.0.0.1")
        with self.assertRaises(ValueError):
            self.auth.admin_update(adm, "root", disabled=True, ip="10.0.0.1")
        with self.assertRaises(ValueError):
            self.auth.admin_update(adm, "root", role="user", ip="10.0.0.1")
        self.assertEqual(self.store.get_user("root").role, "admin")
        self.store.create_user("root2", PW, "admin", must_change=False)
        self.auth.admin_update(adm, "root", role="user", ip="10.0.0.1")
        self.assertEqual(self.store.get_user("root").role, "user")

    def test_non_admin_cannot_update(self):
        _, sess = self.auth.login("alice", PW, "10.0.0.1")
        with self.assertRaises(PermissionError):
            self.auth.admin_update(sess, "alice", role="admin", ip=None)

    def test_admin_reset_password_and_logging(self):
        self.store.create_user("root", PW, "admin", must_change=False)
        _, adm = self.auth.login("root", PW, "10.0.0.1")
        _, victim = self.auth.login("alice", PW, "10.0.0.2")
        self.auth.admin_update(adm, "alice", reset_password="temporary pass 3", ip="10.0.0.1")
        self.assertIsNone(self.auth.session(victim.sid))
        u = self.store.get_user("alice")
        self.assertTrue(u.must_change)
        self.assertEqual(u.password_version, 2)
        self.assertEqual(self.auth.login("alice", "temporary pass 3", None)[0], "must_change")
        ev = self.events("admin_reset_password")[0]
        self.assertEqual(ev["username"], "root")
        self.assertEqual(ev["detail"], {"target": "alice", "action": "reset"})
        self.auth.admin_update(adm, "alice", disabled=True, ip=None)
        ev = self.events("admin_update_user")[0]
        self.assertEqual(ev["detail"], {"target": "alice", "action": "disable"})
        self.auth.admin_update(adm, "alice", disabled=False, ip=None)
        self.assertEqual(self.events("admin_update_user")[0]["detail"]["action"], "enable")

    def test_admin_update_unknown_target(self):
        self.store.create_user("root", PW, "admin", must_change=False)
        _, adm = self.auth.login("root", PW, None)
        with self.assertRaises(ValueError):
            self.auth.admin_update(adm, "ghost", disabled=True, ip=None)

    def test_session_cap_returns_busy(self):
        auth = Auth(self.store, clock=self.mono, wall=self.wall, max_sessions=32)
        for _ in range(32):
            self.assertEqual(auth.login("alice", PW, "10.0.0.1")[0], "ok")
        self.assertEqual(auth.login("alice", PW, "10.0.0.1"), ("busy", None))
        self.advance(1800)  # idle sessions are pruned to make room
        self.assertEqual(auth.login("alice", PW, "10.0.0.1")[0], "ok")

    def test_session_ids_and_csrf_are_random_and_distinct(self):
        _, a = self.auth.login("alice", PW, None)
        _, b = self.auth.login("alice", PW, None)
        self.assertNotEqual(a.sid, b.sid)
        self.assertNotEqual(a.sid, a.csrf)
        self.assertGreaterEqual(len(a.sid), 40)


if __name__ == "__main__":
    unittest.main()
