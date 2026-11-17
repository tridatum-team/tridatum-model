"""Synthetic native databases exercise the account/audit cutover boundary."""

import json
import sqlite3
import tempfile
from pathlib import Path

import webapp_support as S

from django.contrib.sessions.models import Session
from django.db import DatabaseError, connection, transaction
from django.test import TestCase
from django.utils import timezone

from onprem_web.accounts import services as accounts
from onprem_web.accounts.hashers import is_legacy_hash
from onprem_web.accounts.models import User
from onprem_web.audit import services as audit
from onprem_web.audit.models import Event, PurgeGate
from onprem_web.legacy_import import LegacyImportError, import_legacy, legacy_snapshot
from trid.onprem.passwords import hash_password
from trid.onprem.store import Store


class LegacyImportTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.native_hash = hash_password(S.PW)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.source = Path(tmp.name) / 'native.db'
        self.store = Store(self.source, clock=lambda: 123.25)
        self.addCleanup(self.store.close)
        self.add_user('root', role='admin', must_change=0, failed=2, version=7)
        self.store._db.execute(
            "INSERT INTO events VALUES (42, 456.75, NULL, '10.0.0.1', 'ask', 'ok', ?)",
            ('{ "pages": 2, "status": "ok", "seconds": 0.125 }',))
        self.store._db.execute(
            "INSERT INTO events VALUES (80, 1, 'root', NULL, 'clear', 'ok', '{}')")
        self.store._db.execute('DELETE FROM events WHERE id=80')

    def add_user(self, name, *, role='user', must_change=1, failed=0, locked=0, disabled=0, version=1):
        self.store._db.execute('INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                               (name, role, self.native_hash, must_change, failed, locked, disabled, version, 123.25))

    def migrate(self):
        with legacy_snapshot(self.source, S.DATA_DIR / 'web.db') as source:
            return import_legacy(source)

    def assert_empty(self):
        self.assertEqual(User.objects.count(), 0)
        self.assertEqual(Event.objects.count(), 0)
        self.assertEqual(PurgeGate.objects.get(pk=1).open, 0)

    def test_credentials_flags_and_access_restrictions_survive(self):
        from test_webapp_admin import AdminBrowser

        self.add_user('alice', must_change=0)
        self.add_user('disabled', disabled=1, failed=3)
        self.add_user('locked', locked=4000000000.0, failed=5)
        self.add_user('change', must_change=1)
        original = self.store._db.execute('SELECT * FROM users ORDER BY username').fetchall()
        self.assertEqual(self.migrate(), (5, 1))
        self.assertFalse(Session.objects.exists())
        for row in original:
            user = User.objects.get(username=row[0])
            self.assertEqual((user.username, user.role, user.password, int(user.must_change), user.failed,
                              user.locked_until, int(not user.is_active), user.password_version, user.created_at), row)
        self.assertIsNone(accounts.attempt_login('disabled', S.PW, None, now=1000))
        self.assertIsNone(accounts.attempt_login('locked', S.PW, None, now=1000))
        for name in ('disabled', 'locked'):
            self.assertEqual(User.objects.get(username=name).password, self.native_hash)
        self.assertIsNone(accounts.attempt_login('root', 'incorrect password', None, now=1000))
        self.assertEqual(User.objects.get(username='root').password, self.native_hash)
        root = accounts.attempt_login('root', S.PW, None, now=1000)
        self.assertEqual(root.role, 'admin')
        root.refresh_from_db()
        self.assertFalse(is_legacy_hash(root.password))
        self.assertTrue(root.check_password(S.PW))
        self.assertFalse(root.check_password('incorrect password'))
        self.assertEqual((root.failed, root.password_version, root.created_at), (0, 7, 123.25))
        S.make_runtime()
        staff = AdminBrowser()
        self.assertEqual(staff.login('alice').status_code, 200)
        self.assertEqual(staff.get('/admin').status_code, 403)
        admin = AdminBrowser()
        self.assertEqual(admin.login('root').status_code, 200)
        self.assertEqual(admin.get('/admin').status_code, 200)
        forced = accounts.attempt_login('change', S.PW, None, now=1000)
        self.assertTrue(forced.must_change)
        self.assertEqual(self.store._db.execute('SELECT * FROM users ORDER BY username').fetchall(), original)

    def test_audit_fidelity_append_only_and_sequence_high_water(self):
        original = self.store._db.execute('SELECT * FROM events ORDER BY id').fetchall()
        self.migrate()
        self.assertEqual(list(Event.objects.order_by('id').values_list(
            'id', 'ts', 'username', 'ip', 'event', 'result', 'detail')), original)
        self.assertEqual(audit.query_events()[0]['detail'], json.loads(original[0][-1]))
        self.assertEqual(audit.export_csv(), b'\xef\xbb\xbf' + self.store.export_csv().encode('utf-8'))
        for mutate in (lambda: Event.objects.all().update(result='changed'),
                       lambda: Event.objects.all().delete()):
            with self.assertRaises(DatabaseError), transaction.atomic():
                mutate()
        audit.log('logout', 'root', None, 'ok')
        self.assertEqual(Event.objects.latest('id').id, 81)
        self.assertEqual(self.store._db.execute('SELECT * FROM events ORDER BY id').fetchall(), original)

    def test_source_is_read_only_and_snapshot_includes_wal_without_later_writes(self):
        with legacy_snapshot(self.source, S.DATA_DIR / 'web.db') as source:
            with self.assertRaises(sqlite3.OperationalError):
                source.execute('DELETE FROM users')
            self.add_user('later')
            self.store.log('logout', 'root', None, 'ok')
            self.assertEqual(import_legacy(source), (1, 1))
        self.assertFalse(User.objects.filter(username='later').exists())
        self.assertEqual(self.store._db.execute('SELECT COUNT(*) FROM users').fetchone()[0], 2)

    def test_existing_account_rejected_without_overwrite(self):
        original = accounts.create_user('existing', S.PW, 'user', must_change=False)
        with self.assertRaisesRegex(LegacyImportError, 'not empty'):
            self.migrate()
        self.assertEqual(list(User.objects.values_list('username', 'password')), [(original.username, original.password)])
        self.assertEqual(Event.objects.count(), 0)

    def test_existing_audit_or_session_rejected(self):
        for kind in ('event', 'session'):
            with self.subTest(kind=kind), transaction.atomic():
                if kind == 'event':
                    audit.log('logout', None, None, 'ok')
                else:
                    Session.objects.create(session_key='existing', session_data='', expire_date=timezone.now())
                with self.assertRaisesRegex(LegacyImportError, 'not empty'):
                    self.migrate()
                self.assertEqual(User.objects.count(), 0)
                self.assertEqual(Event.objects.count(), int(kind == 'event'))
                transaction.set_rollback(True)

    def test_late_invalid_audit_rolls_back_accounts_events_and_sequence(self):
        self.store._db.execute("INSERT INTO events VALUES (90, 999, NULL, NULL, 'ask', 'ok', 'invalid-json')")
        with self.assertRaisesRegex(LegacyImportError, 'audit detail'):
            self.migrate()
        self.assert_empty()
        self.store._db.execute('DELETE FROM events WHERE id=90')
        self.assertEqual(self.migrate(), (1, 1))
        audit.log('logout', 'root', None, 'ok')
        self.assertEqual(Event.objects.latest('id').id, 91)

    def test_unsupported_password_cost_is_rejected_atomically(self):
        self.add_user('zzbad')
        self.store._db.execute('UPDATE users SET pw_hash=? WHERE username=?',
                               (self.native_hash.replace('$16384$', '$1073741824$'), 'zzbad'))
        with self.assertRaisesRegex(LegacyImportError, 'account data'):
            self.migrate()
        self.assert_empty()

    def test_target_database_conflict_rolls_back_accounts(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TRIGGER reject_import BEFORE INSERT ON audit_event "
                "BEGIN SELECT RAISE(ABORT, 'synthetic conflict'); END")
        try:
            with self.assertRaisesRegex(LegacyImportError, 'database conflict'):
                self.migrate()
            self.assert_empty()
        finally:
            with connection.cursor() as cursor:
                cursor.execute('DROP TRIGGER reject_import')

    def test_unsupported_schema_and_source_target_alias_are_rejected(self):
        with self.assertRaisesRegex(LegacyImportError, 'different files'):
            with legacy_snapshot(self.source, self.source):
                self.fail('alias accepted')
        self.store._db.execute('PRAGMA user_version=2')
        with self.assertRaisesRegex(LegacyImportError, 'schema version'):
            self.migrate()
        self.store._db.execute('PRAGMA user_version=1')
        self.store._db.execute('ALTER TABLE users ADD COLUMN unsupported TEXT')
        with self.assertRaisesRegex(LegacyImportError, 'schema'):
            self.migrate()
        self.assert_empty()
