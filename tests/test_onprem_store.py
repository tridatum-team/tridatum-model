import csv
import io
import tempfile
import unittest
from pathlib import Path

from trid.onprem.passwords import check_new_password, hash_password, verify_password
from trid.onprem.store import Store, normalize_username


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class PasswordTests(unittest.TestCase):
    def test_hash_roundtrip_and_salted(self):
        a, b = hash_password("correct horse"), hash_password("correct horse")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("scrypt$16384$8$1$"))
        self.assertTrue(verify_password("correct horse", a))
        self.assertTrue(verify_password("correct horse", b))
        self.assertFalse(verify_password("wrong horse!!", a))
        self.assertFalse(verify_password("x", "garbage"))

    def test_short_password_rejected(self):
        with self.assertRaises(ValueError):
            check_new_password("123456789")
        check_new_password("1234567890")

    def test_overlong_or_non_str_password_rejected(self):
        check_new_password("x" * 1024)
        with self.assertRaises(ValueError):
            check_new_password("x" * 1025)
        with self.assertRaises(ValueError):
            check_new_password(None)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.store = Store(Path(self.tmp.name) / "app.db", clock=self.clock)
        self.addCleanup(self.store.close)

    def test_username_normalized_on_create_and_lookup(self):
        user = self.store.create_user(" Kim ", "password-123", "user")
        self.assertEqual(user.username, "kim")
        self.assertEqual(self.store.get_user("KIM").username, "kim")
        self.assertEqual(self.store.get_user("kim").password_version, 1)
        with self.assertRaises(ValueError):
            self.store.create_user("kim", "password-123", "user")
        with self.assertRaises(ValueError):
            normalize_username("a")
        with self.assertRaises(ValueError):
            normalize_username("bad name")

    def test_update_user_and_admin_count(self):
        self.store.create_user("root", "password-123", "admin")
        self.store.create_user("kim", "password-123", "user")
        self.assertEqual(self.store.count_active_admins(), 1)
        self.store.update_user("kim", role="admin", failed=3, must_change=False)
        u = self.store.get_user("kim")
        self.assertEqual((u.role, u.failed, u.must_change), ("admin", 3, False))
        self.assertEqual(self.store.count_active_admins(), 2)
        self.store.update_user("kim", disabled=True)
        self.assertEqual(self.store.count_active_admins(), 1)
        with self.assertRaises(ValueError):
            self.store.update_user("kim", username="x")
        self.assertEqual([x.username for x in self.store.list_users()], ["kim", "root"])

    def test_log_rejects_unknown_detail_keys(self):
        with self.assertRaises(ValueError):
            self.store.log("pdf_upload", "kim", "10.0.0.1", "ok", filename="a.pdf")
        with self.assertRaises(ValueError):
            self.store.log("pdf_upload", "kim", "10.0.0.1", "ok", error=["x"])
        self.assertEqual(self.store.query_events(), [])

    def test_events_never_store_question_or_filename(self):
        self.store.log("pdf_upload", "kim", "10.0.0.1", "ok", sha256="ab" * 32, pages=3)
        self.store.log("ask", "kim", "10.0.0.1", "ok", seconds=1.5, status="answer")
        try:
            self.store.log("ask", "kim", "10.0.0.1", "ok", question="비밀질문XYZ")
        except ValueError:
            pass
        try:
            self.store.log("pdf_upload", "kim", "10.0.0.1", "ok", filename="비밀파일.pdf")
        except ValueError:
            pass
        data = b"".join(p.read_bytes() for p in Path(self.tmp.name).iterdir())
        self.assertIn(b"answer", data)
        for needle in ("비밀질문XYZ", "비밀파일", "question", "filename"):
            self.assertNotIn(needle.encode("utf-8"), data)

    def test_retention_purge_event_is_allowed(self):
        self.store.log("retention_purge", None, None, "ok", count=3)
        self.assertEqual(self.store.query_events()[0]["detail"], {"count": 3})

    def test_purge_before_removes_only_older_events(self):
        for t in (100.0, 200.0, 300.0):
            self.clock.t = t
            self.store.log("login_ok", "kim", None, "ok")
        self.assertEqual(self.store.purge_before(250.0), 2)
        rows = self.store.query_events()
        self.assertEqual([r["ts"] for r in rows], [300.0])

    def test_query_filters(self):
        self.clock.t = 100.0
        self.store.log("login_ok", "Kim", "1.1.1.1", "ok")
        self.clock.t = 200.0
        self.store.log("logout", "lee", "1.1.1.2", "ok")
        self.assertEqual(len(self.store.query_events(username="KIM")), 1)
        self.assertEqual(len(self.store.query_events(event="logout")), 1)
        self.assertEqual(len(self.store.query_events(since=150.0)), 1)
        self.assertEqual(len(self.store.query_events(until=150.0)), 1)
        self.assertEqual(len(self.store.query_events(limit=1, offset=1)), 1)

    def test_export_csv_header_and_rows(self):
        self.clock.t = 50.0
        self.store.log("login_failed", None, "10.0.0.9", "bad_password")
        self.clock.t = 60.0
        self.store.log("pdf_upload", "kim", "10.0.0.1", "ok", pages=2)
        text = self.store.export_csv()
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(rows[0], ["ts", "username", "ip", "event", "result", "detail"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][1:5], ["kim", "10.0.0.1", "pdf_upload", "ok"])
        self.assertEqual(rows[2][1], "(unknown)")
        self.assertEqual(self.store.export_csv(event="pdf_upload").count("\n"), 2)

    def test_log_rejects_free_text_in_fixed_columns(self):
        bad = [
            dict(username="x" * 33, ip="10.0.0.1", result="ok"),
            dict(username="bad name", ip="10.0.0.1", result="ok"),
            dict(username="kim", ip="not an ip", result="ok"),
            dict(username="kim", ip="10.0.0.1", result="free text answer"),
            dict(username="kim", ip="10.0.0.1", result="a" * 33),
            dict(username="kim", ip="10.0.0.1", result="ok", error="x" * 129),
            dict(username="kim", ip="10.0.0.1", result="ok", error="has space"),
            dict(username="kim", ip="10.0.0.1", result="ok", target="=cmd"),
        ]
        for kw in bad:
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    self.store.log("ask", **kw)
        self.assertEqual(self.store.query_events(), [])
        self.store.log("ask", "KIM", "::1", "ok", error="timeout:3.5")
        self.assertEqual(self.store.query_events()[0]["username"], "kim")

    def test_export_csv_escapes_formula_cells(self):
        self.store.log("login_ok", "kim", "10.0.0.1", "ok")
        # defence in depth: inject a hostile row directly, bypassing log()
        self.store._db.execute(
            "INSERT INTO events (ts, username, ip, event, result, detail) "
            "VALUES (1, '=HYPERLINK(1)', '+1', 'ask', '@x', '{}')")
        rows = list(csv.reader(io.StringIO(self.store.export_csv())))
        hostile = [r for r in rows if r[3] == "ask"][0]
        self.assertEqual(hostile[1:5], ["'=HYPERLINK(1)", "'+1", "ask", "'@x"])


if __name__ == "__main__":
    unittest.main()
