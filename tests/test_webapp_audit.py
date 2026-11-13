import tempfile
import unittest
from pathlib import Path
from unittest import mock

import webapp_support as S  # noqa: F401  (boots Django and migrates)

from django.db import transaction
from django.db.models.query import QuerySet
from django.db.utils import DatabaseError
from django.test import TestCase

from onprem_web.audit import services
from onprem_web.audit.models import Event, PurgeGate
from trid.onprem.store import Store


def add(ts, **kw):
    fields = dict(ts=ts, username='kim', ip='10.0.0.1', event='ask', result='ok', detail='{}')
    fields.update(kw)
    return Event.objects.create(**fields)


class AuditTests(TestCase):
    def test_log_rejects_free_text(self):
        bad = [
            ('ask', 'ok', dict(question='what is this')),
            ('ask', 'ok', dict(error='has space')),
            ('nope', 'ok', {}),
            ('ask', 'OK!', {}),
        ]
        for event, result, detail in bad:
            with self.subTest(event=event, result=result, detail=detail):
                with self.assertRaises(ValueError):
                    services.log(event, 'kim', '10.0.0.1', result, **detail)
        self.assertEqual(Event.objects.count(), 0)

    def test_log_stores_sorted_json(self):
        with mock.patch.object(services, '_clock', lambda: 5.0):
            services.log('pdf_upload', 'KIM', '10.0.0.1', 'ok', pages=2, sha256='ab12')
        e = Event.objects.get()
        self.assertEqual((e.ts, e.username, e.detail), (5.0, 'kim', '{"pages": 2, "sha256": "ab12"}'))

    def test_update_and_delete_blocked_at_db_level(self):
        e = add(1.0)
        attempts = [
            lambda: Event.objects.filter(pk=e.pk).update(result='x'),
            lambda: Event.objects.all().delete(),
            lambda: Event.objects.bulk_update([Event(pk=e.pk, result='x')], ['result']),
        ]
        for attempt in attempts:
            with self.subTest():
                with self.assertRaises(DatabaseError):  # IntegrityError is a subclass
                    with transaction.atomic():
                        attempt()
                self.assertEqual(Event.objects.get(pk=e.pk).result, 'ok')
        with self.assertRaises(PermissionError):
            e.save()
        with self.assertRaises(PermissionError):
            e.delete()
        self.assertEqual(Event.objects.count(), 1)

    def test_purge_deletes_only_old_rows_and_gate_closes(self):
        add(1.0)
        add(2.0)
        add(100.0)
        self.assertEqual(services.purge_before(50.0), 2)
        self.assertEqual([e.ts for e in Event.objects.all()], [100.0])
        self.assertEqual(PurgeGate.objects.get().open, 0)
        with self.assertRaises(DatabaseError):
            with transaction.atomic():
                Event.objects.all().delete()

    def test_purge_failure_keeps_gate_closed(self):
        add(1.0)
        with mock.patch.object(QuerySet, 'delete', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                services.purge_before(50.0)
        self.assertEqual(PurgeGate.objects.get().open, 0)
        self.assertEqual(Event.objects.count(), 1)

    def test_query_events_order_filters_and_paging(self):
        add(10.0, event='login_ok')
        add(20.0, username='lee')
        add(20.0, username='kim', detail='{"pages": 3}')
        rows = services.query_events()
        self.assertEqual([r['ts'] for r in rows], [20.0, 20.0, 10.0])
        self.assertEqual(rows[0]['detail'], {'pages': 3})  # id DESC tie-break
        self.assertEqual(set(rows[0]), {'ts', 'username', 'ip', 'event', 'result', 'detail'})
        self.assertEqual(len(services.query_events(username='KIM')), 2)
        self.assertEqual(len(services.query_events(event='login_ok')), 1)
        self.assertEqual(len(services.query_events(since=15.0)), 2)
        self.assertEqual(len(services.query_events(until=15.0)), 1)
        self.assertEqual(len(services.query_events(limit=1, offset=1)), 1)

    def test_csv_matches_old_store(self):
        def fixed():
            return 123.5

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / 'old.db', clock=fixed)
            try:
                store.log('login_failed', None, '10.0.0.9', 'bad_password')
                store.log('pdf_upload', 'kim', '10.0.0.1', 'ok', pages=2)
                store._db.execute(  # hostile row injected past log(), like the old suite
                    "INSERT INTO events (ts, username, ip, event, result, detail) "
                    "VALUES (123.5, '=evil', '+1', 'ask', '@x', '{}')")
                old = store.export_csv()
            finally:
                store.close()
        with mock.patch.object(services, '_clock', fixed):
            services.log('login_failed', None, '10.0.0.9', 'bad_password')
            services.log('pdf_upload', 'kim', '10.0.0.1', 'ok', pages=2)
        add(123.5, username='=evil', ip='+1', result='@x')
        new = services.export_csv()
        self.assertTrue(new.startswith(b'\xef\xbb\xbf'))
        self.assertIn(b"'=evil", new)
        self.assertEqual(new, b'\xef\xbb\xbf' + old.encode('utf-8'))


if __name__ == '__main__':
    unittest.main()
