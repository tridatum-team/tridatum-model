"""Read-only native v1 snapshot -> empty Django account/audit tables.

No sessions or documents cross the boundary. The source remains usable for rollback.
Errors deliberately describe the boundary, never a row, password hash or SQL value.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3

from trid.onprem.store import ROLES, normalize_username

from onprem_web.accounts.hashers import is_legacy_hash


class LegacyImportError(ValueError):
    pass


_COLUMNS = {
    'users': (
        ('username', 'TEXT', 0, 1), ('role', 'TEXT', 1, 0), ('pw_hash', 'TEXT', 1, 0),
        ('must_change', 'INTEGER', 1, 0), ('failed', 'INTEGER', 1, 0),
        ('locked_until', 'REAL', 1, 0), ('disabled', 'INTEGER', 1, 0),
        ('password_version', 'INTEGER', 1, 0), ('created_at', 'REAL', 1, 0)),
    'events': (
        ('id', 'INTEGER', 0, 1), ('ts', 'REAL', 1, 0), ('username', 'TEXT', 0, 0),
        ('ip', 'TEXT', 0, 0), ('event', 'TEXT', 1, 0), ('result', 'TEXT', 1, 0),
        ('detail', 'TEXT', 1, 0)),
}


@contextmanager
def legacy_snapshot(source: Path, target: Path):
    """Pin one SQLite read transaction, including committed WAL contents."""
    source = source.resolve(strict=True)
    target = target.resolve()
    if source == target or (target.exists() and source.samefile(target)):
        raise LegacyImportError('Legacy source and Django target must be different files.')
    db = None
    try:
        db = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        if db.execute('PRAGMA user_version').fetchone()[0] != 1:
            raise LegacyImportError('Unsupported legacy database schema version.')
        for table, expected in _COLUMNS.items():
            kind = db.execute('SELECT type FROM sqlite_master WHERE name=?', (table,)).fetchone()
            columns = tuple((r[1], r[2].upper(), r[3], r[5])
                            for r in db.execute(f'PRAGMA table_info({table})'))
            if kind != ('table',) or columns != expected:
                raise LegacyImportError('Unsupported legacy database schema.')
        yield db
    except sqlite3.Error:
        raise LegacyImportError('Legacy database could not be read; no records imported.') from None
    finally:
        if db is not None:
            db.close()


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _user(row):
    name, role, password, must_change, failed, locked, disabled, version, created = row
    try:
        normalized = normalize_username(name) if isinstance(name, str) else None
    except ValueError:
        normalized = None
    if (not isinstance(name, str) or normalized != name or role not in ROLES
            or not is_legacy_hash(password) or must_change not in (0, 1) or disabled not in (0, 1)
            or not isinstance(failed, int) or failed < 0
            or not isinstance(version, int) or version < 0
            or not _number(locked) or not _number(created)):
        raise LegacyImportError('Unsupported legacy account data; no records imported.')
    return (name, role, password, must_change, failed, locked, not disabled, version, created)


def _event(row):
    ident, ts, username, ip, event, result, detail = row
    if (not isinstance(ident, int) or not 0 < ident < 2**63 - 1 or not _number(ts)
            or any(v is not None and not isinstance(v, str) for v in (username, ip))
            or any(not isinstance(v, str) for v in (event, result, detail))):
        raise LegacyImportError('Unsupported legacy audit data; no records imported.')
    # Keep the original JSON bytes, including historical formatting and fields.
    # Consumers require an object; do not normalize history through today's log().
    try:
        valid = isinstance(json.loads(detail), dict)
    except (ValueError, TypeError, RecursionError):
        valid = False
    if not valid:
        raise LegacyImportError('Unsupported legacy audit detail; no records imported.')
    return row


def import_legacy(source: sqlite3.Connection) -> tuple[int, int]:
    """Import all records or none; target migrations must already be applied."""
    from django.db import DatabaseError, connection, transaction

    if connection.vendor != 'sqlite':
        raise LegacyImportError('Legacy import requires a SQLite Django target.')
    accounts = events = 0
    try:
        with transaction.atomic(), connection.cursor() as target:
            # Acquire the SQLite writer lock before testing emptiness. Do not open
            # the purge gate, remove triggers, or allow a concurrent bootstrap write.
            target.execute('UPDATE audit_purgegate SET open=open WHERE id=1')
            target.execute('SELECT id, open FROM audit_purgegate')
            if target.fetchall() != [(1, 0)]:
                raise LegacyImportError('Django audit protection is not ready for import.')
            for table in ('accounts_user', 'audit_event', 'django_session', 'auth_group'):
                target.execute(f'SELECT 1 FROM {table} LIMIT 1')
                if target.fetchone() is not None:
                    raise LegacyImportError('Django target is not empty; no records imported.')
            for row in source.execute('SELECT username, role, pw_hash, must_change, failed, '
                                      'locked_until, disabled, password_version, created_at FROM users '
                                      'ORDER BY username'):
                target.execute(
                    'INSERT INTO accounts_user '
                    '(username, role, password, must_change, failed, locked_until, is_active, '
                    'password_version, created_at, last_login) '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)', _user(row))
                accounts += 1
            last_id = 0
            for row in source.execute('SELECT id, ts, username, ip, event, result, detail '
                                      'FROM events ORDER BY id'):
                target.execute('INSERT INTO audit_event (id, ts, username, ip, event, result, detail) '
                               'VALUES (%s, %s, %s, %s, %s, %s, %s)', _event(row))
                last_id = row[0]
                events += 1
            sequence = source.execute("SELECT seq FROM sqlite_sequence WHERE name='events'").fetchall()
            if len(sequence) > 1 or (sequence and (
                    not isinstance(sequence[0][0], int) or sequence[0][0] < last_id
                    or sequence[0][0] >= 2**63 - 1)):
                raise LegacyImportError('Unsupported legacy audit sequence; no records imported.')
            high_water = sequence[0][0] if sequence else last_id
            if high_water:
                target.execute("UPDATE sqlite_sequence SET seq=%s WHERE name='audit_event'", [high_water])
                if not target.rowcount:
                    target.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('audit_event', %s)",
                                   [high_water])
    except (DatabaseError, sqlite3.Error):
        raise LegacyImportError('Legacy import database conflict; no records imported.') from None
    return accounts, events
