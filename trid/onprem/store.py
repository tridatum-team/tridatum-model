"""SQLite account and usage-event store (standard library only)."""

from __future__ import annotations

import csv
import io
import ipaddress
import json
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from trid.onprem.passwords import check_new_password, hash_password

EVENTS = frozenset({
    "login_ok", "login_failed", "login_locked", "logout", "session_expired",
    "password_changed", "pdf_upload", "ask", "clear", "admin_create_user",
    "admin_update_user", "admin_reset_password", "admin_export",
})
DETAIL_KEYS = frozenset({
    "sha256", "pages", "seconds", "status", "error", "target", "action", "count",
})
USER_FIELDS = frozenset({
    "role", "pw_hash", "must_change", "failed", "locked_until", "disabled",
    "password_version",
})
ROLES = frozenset({"user", "admin"})
_USERNAME_RE = re.compile(r"[a-z0-9._-]{2,32}")
_RESULT_RE = re.compile(r"[a-z_]{1,32}")
_DETAIL_STR_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_CSV_DANGER = ("=", "+", "-", "@", chr(9), chr(13))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    pw_hash TEXT NOT NULL,
    must_change INTEGER NOT NULL,
    failed INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0,
    password_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    username TEXT,
    ip TEXT,
    event TEXT NOT NULL,
    result TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


def normalize_username(name: str) -> str:
    norm = unicodedata.normalize("NFC", name).strip().lower()
    if not _USERNAME_RE.fullmatch(norm):
        raise ValueError("계정 이름은 영문 소문자, 숫자, '.', '_', '-' 2~32자여야 합니다.")
    return norm


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).strip().lower()


@dataclass
class User:
    username: str
    role: str
    pw_hash: str
    must_change: bool
    failed: int
    locked_until: float
    disabled: bool
    password_version: int


_USER_COLS = ("username, role, pw_hash, must_change, failed, locked_until, "
              "disabled, password_version")


def _row_to_user(row) -> User:
    return User(row[0], row[1], row[2], bool(row[3]), int(row[4]),
                float(row[5]), bool(row[6]), int(row[7]))


class Store:
    def __init__(self, path: Path, clock=time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False,
                                   isolation_level=None)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            if self._db.execute("PRAGMA user_version").fetchone()[0] < 1:
                self._db.executescript(_SCHEMA)
                self._db.execute("PRAGMA user_version=1")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # accounts
    def create_user(self, username, password, role, must_change=True) -> User:
        name = normalize_username(username)
        if role not in ROLES:
            raise ValueError("알 수 없는 역할입니다.")
        check_new_password(password)
        pw_hash = hash_password(password)
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO users (username, role, pw_hash, must_change, "
                    "failed, locked_until, disabled, password_version, created_at) "
                    "VALUES (?, ?, ?, ?, 0, 0, 0, 1, ?)",
                    (name, role, pw_hash, int(bool(must_change)), self._clock()))
            except sqlite3.IntegrityError:
                raise ValueError("이미 있는 계정입니다.") from None
        return User(name, role, pw_hash, bool(must_change), 0, 0.0, False, 1)

    def get_user(self, username) -> User | None:
        try:
            name = normalize_username(username)
        except ValueError:
            return None
        with self._lock:
            row = self._db.execute(
                f"SELECT {_USER_COLS} FROM users WHERE username=?", (name,)).fetchone()
        return _row_to_user(row) if row else None

    def list_users(self) -> list[User]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_USER_COLS} FROM users ORDER BY username").fetchall()
        return [_row_to_user(r) for r in rows]

    def update_user(self, username, /, **fields) -> None:
        unknown = set(fields) - USER_FIELDS
        if unknown:
            raise ValueError(f"허용되지 않는 필드: {sorted(unknown)}")
        if not fields:
            return
        if "role" in fields and fields["role"] not in ROLES:
            raise ValueError("알 수 없는 역할입니다.")
        name = normalize_username(username)
        values = [int(v) if isinstance(v, bool) else v for v in fields.values()]
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE users SET {assignments} WHERE username=?", (*values, name))
            if cur.rowcount == 0:
                raise ValueError("없는 계정입니다.")

    def count_active_admins(self) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND disabled=0"
            ).fetchone()[0]

    # events
    def log(self, event, username, ip, result, **detail) -> None:
        if event not in EVENTS:
            raise ValueError(f"알 수 없는 이벤트: {event}")
        extra = set(detail) - DETAIL_KEYS
        if extra:
            raise ValueError(f"기록할 수 없는 항목: {sorted(extra)}")
        for key, value in detail.items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"detail 값은 단순 값이어야 합니다: {key}")
            if isinstance(value, str) and not _DETAIL_STR_RE.fullmatch(value):
                raise ValueError(f"detail 문자열 형식이 올바르지 않습니다: {key}")
        name = normalize_username(username) if username is not None else None
        if ip is not None:
            try:
                ipaddress.ip_address(ip)
            except ValueError:
                raise ValueError("ip 형식이 올바르지 않습니다.") from None
        if not isinstance(result, str) or not _RESULT_RE.fullmatch(result):
            raise ValueError("result 형식이 올바르지 않습니다.")
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, username, ip, event, result, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._clock(), name, ip, event, result,
                 json.dumps(detail, ensure_ascii=False, sort_keys=True)))

    @staticmethod
    def _where(since, until, username, event):
        clauses, params = [], []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("ts <= ?")
            params.append(until)
        if username is not None:
            clauses.append("username = ?")
            params.append(_fold(username))
        if event is not None:
            clauses.append("event = ?")
            params.append(event)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query_events(self, since=None, until=None, username=None, event=None,
                     limit=200, offset=0) -> list[dict]:
        where, params = self._where(since, until, username, event)
        sql = ("SELECT ts, username, ip, event, result, detail FROM events"
               + where + " ORDER BY ts DESC, id DESC")
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [int(limit), int(offset)]
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [{"ts": r[0], "username": r[1], "ip": r[2], "event": r[3],
                 "result": r[4], "detail": json.loads(r[5])} for r in rows]

    def export_csv(self, since=None, until=None, username=None, event=None,
                   limit=None, offset=0) -> str:
        rows = self.query_events(since, until, username, event, limit, offset)
        def safe(cell):
            if isinstance(cell, str) and cell.startswith(_CSV_DANGER):
                return "'" + cell
            return cell

        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["ts", "username", "ip", "event", "result", "detail"])
        for r in rows:
            writer.writerow(safe(c) for c in [
                r["ts"],
                r["username"] if r["username"] is not None else "(unknown)",
                r["ip"] if r["ip"] is not None else "",
                r["event"], r["result"],
                json.dumps(r["detail"], ensure_ascii=False, sort_keys=True),
            ])
        return out.getvalue()

    def purge_before(self, ts: float) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM events WHERE ts < ?", (ts,)).rowcount
