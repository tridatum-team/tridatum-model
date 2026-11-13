"""In-memory session registry: the per-session state that must never reach the database.

The Django session row only proves who is logged in. Everything held in memory for that login
(the uploaded document, its id, the per-minute call times, the idle timer) lives here, keyed by
the session key, and is gone when the process restarts. Ports the bookkeeping of
trid.onprem.auth.Auth (prune, _drop_user_sessions, idle check, rate window).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Entry:
    username: str
    touched: float
    document: object | None = None
    document_id: str | None = None
    calls: list[float] = field(default_factory=list)


def _release(entry: Entry) -> None:
    entry.document = None
    entry.document_id = None
    entry.calls = []


class SessionRegistry:
    def __init__(self, max_sessions: int, idle_seconds: float, per_minute: int, clock=time.monotonic):
        self._max = max_sessions
        self._idle = idle_seconds
        self._per_minute = per_minute
        self._clock = clock
        self._entries: dict[str, Entry] = {}
        self._lock = threading.RLock()

    def _is_idle(self, entry: Entry, now: float) -> bool:
        return now - entry.touched >= self._idle  # same boundary as Auth._is_idle

    def _drop_locked(self, key: str) -> Entry | None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            _release(entry)
        return entry

    def prune(self) -> list[tuple[str, str]]:
        """Drop idle entries; return (key, username) of each one dropped."""
        with self._lock:
            now = self._clock()
            idle = [(k, e.username) for k, e in self._entries.items() if self._is_idle(e, now)]
            for key, _ in idle:
                self._drop_locked(key)
            return idle

    def admit(self, key: str, username: str) -> bool:
        with self._lock:
            self.prune()
            entry = self._entries.get(key)
            if entry is not None and entry.username == username:
                entry.touched = self._clock()
                return True
            if entry is None and len(self._entries) >= self._max:
                return False
            self._drop_locked(key)
            self._entries[key] = Entry(username=username, touched=self._clock())
            return True

    def has_room(self, key: str | None = None) -> bool:
        """True if `key` already has an entry or another one fits (call prune() first)."""
        with self._lock:
            return key in self._entries or len(self._entries) < self._max

    def get(self, key) -> Entry | None:
        with self._lock:
            return self._entries.get(key) if isinstance(key, str) else None

    def touch(self, key) -> bool:
        with self._lock:
            entry = self.get(key)
            if entry is None:
                return False
            now = self._clock()
            if self._is_idle(entry, now):
                self._drop_locked(key)
                return False
            entry.touched = now
            return True

    def allow_call(self, key) -> bool:
        with self._lock:
            entry = self.get(key)
            if entry is None:
                return False
            now = self._clock()
            entry.calls = [t for t in entry.calls if now - t < 60]
            if len(entry.calls) >= self._per_minute:
                return False
            entry.calls.append(now)
            return True

    def rekey(self, old: str, new: str) -> None:
        """Move an entry to a new session key (cycle_key after a password change keeps the document)."""
        with self._lock:
            entry = self._entries.pop(old, None)
            if entry is not None and old != new:
                self._drop_locked(new)
            if entry is not None:
                self._entries[new] = entry

    def attach(self, key, entry: Entry, document, document_id: str) -> bool:
        """Store a document on `entry` only while `key` still maps to that same entry."""
        with self._lock:
            if self.get(key) is not entry:
                return False
            entry.document = document
            entry.document_id = document_id
            return True

    def drop(self, key) -> None:
        with self._lock:
            if isinstance(key, str):
                self._drop_locked(key)

    def drop_user(self, username: str, keep: str | None = None) -> None:
        with self._lock:
            for key in [k for k, e in self._entries.items() if e.username == username and k != keep]:
                self._drop_locked(key)

    def keys(self) -> set[str]:
        with self._lock:
            return set(self._entries)
