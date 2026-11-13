"""Login, lockout and in-memory session lifecycle (standard library only)."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from trid.onprem.passwords import check_new_password, hash_password, verify_password
from trid.onprem.store import ROLES, Store, normalize_username

_MAX_PASSWORD_CHARS = 1024
_LAST_ADMIN = "마지막 관리자는 잠그거나 권한을 바꿀 수 없습니다."


@dataclass
class Session:
    sid: str
    csrf: str
    username: str
    role: str
    password_version: int
    must_change: bool
    touched: float
    document: object = None
    document_id: str | None = None
    calls: list = field(default_factory=list)


class Auth:
    def __init__(self, store: Store, clock=time.monotonic, wall=time.time,
                 max_sessions: int = 32, idle_seconds: float = 1800,
                 max_failures: int = 5, lock_seconds: float = 900):
        self._store = store
        self._clock = clock
        self._wall = wall
        self._max_sessions = max_sessions
        self._idle = idle_seconds
        self._max_failures = max_failures
        self._lock_seconds = lock_seconds
        self._sessions: dict[str, Session] = {}
        # _lock guards the session dict (short); _acct_lock serializes account
        # mutations so failure counting cannot race (held across scrypt).
        self._lock = threading.RLock()
        self._acct_lock = threading.RLock()
        # Verified against when the account is unknown, to even out timing.
        self._dummy_hash = hash_password(secrets.token_urlsafe(16))

    # -- session bookkeeping -------------------------------------------------
    @staticmethod
    def _clear(sess: Session) -> None:
        sess.document = None
        sess.document_id = None
        sess.calls = []

    def _drop_locked(self, sid: str) -> Session | None:
        sess = self._sessions.pop(sid, None)
        if sess is not None:
            self._clear(sess)
        return sess

    def _drop_user_sessions(self, username: str, keep: str | None = None) -> None:
        with self._lock:
            for sid in [s.sid for s in self._sessions.values()
                        if s.username == username and s.sid != keep]:
                self._drop_locked(sid)

    def _expire_locked(self, sess: Session) -> None:
        self._drop_locked(sess.sid)
        self._store.log("session_expired", sess.username, None, "expired")

    def _is_idle(self, sess: Session) -> bool:
        return self._clock() - sess.touched >= self._idle

    def prune(self) -> None:
        with self._lock:
            for sess in [s for s in self._sessions.values() if self._is_idle(s)]:
                self._expire_locked(sess)

    # -- failure counting ----------------------------------------------------
    def _expire_served_lock(self, user, now: float) -> None:
        """Start counting afresh once a lock window has passed."""
        if user.locked_until and user.locked_until <= now:
            self._store.update_user(user.username, failed=0, locked_until=0.0)
            user.failed, user.locked_until = 0, 0.0

    def _record_failure(self, user, now: float) -> bool:
        """Count one failure; return True if the account is (now) locked.

        Call with _acct_lock held. An active lock is never extended.
        """
        self._expire_served_lock(user, now)
        if user.locked_until > now:
            return True
        failed = user.failed + 1
        fields = {"failed": failed}
        locked = failed >= self._max_failures
        if locked:
            fields["locked_until"] = now + self._lock_seconds
        self._store.update_user(user.username, **fields)
        return locked

    # -- login ---------------------------------------------------------------
    def login(self, username: str, password: str, ip: str | None):
        if not isinstance(password, str) or len(password) > _MAX_PASSWORD_CHARS:
            password = ""
        user = self._store.get_user(username) if isinstance(username, str) else None
        with self._acct_lock:
            if user is not None:
                user = self._store.get_user(user.username)  # fresh under lock
            now = self._wall()
            if user is None:
                verify_password(password, self._dummy_hash)
                self._store.log("login_failed", None, ip, "invalid")
                return "invalid", None
            self._expire_served_lock(user, now)
            if user.locked_until > now:
                verify_password(password, self._dummy_hash)
                self._store.log("login_locked", user.username, ip, "locked")
                return "locked", None
            ok = verify_password(password, user.pw_hash)
            if user.disabled:
                self._store.log("login_failed", user.username, ip, "invalid",
                                error="disabled")
                return "invalid", None
            if not ok:
                self._record_failure(user, now)
                self._store.log("login_failed", user.username, ip, "invalid")
                return "invalid", None
            if user.failed:
                self._store.update_user(user.username, failed=0)
            with self._lock:
                self.prune()
                if len(self._sessions) >= self._max_sessions:
                    self._store.log("login_failed", user.username, ip, "busy")
                    return "busy", None
                sess = Session(
                    sid=secrets.token_urlsafe(32), csrf=secrets.token_urlsafe(32),
                    username=user.username, role=user.role,
                    password_version=user.password_version,
                    must_change=user.must_change, touched=self._clock())
                self._sessions[sess.sid] = sess
            self._store.log("login_ok", user.username, ip, "ok")
            return ("must_change" if user.must_change else "ok"), sess

    # -- session access ------------------------------------------------------
    def session(self, sid: str) -> Session | None:
        if not isinstance(sid, str):
            return None
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return None
            if self._is_idle(sess):
                self._expire_locked(sess)
                return None
            user = self._store.get_user(sess.username)
            if (user is None or user.disabled
                    or user.password_version != sess.password_version):
                self._drop_locked(sid)
                self._store.log("session_expired", sess.username, None, "revoked")
                return None
            sess.role = user.role
            sess.must_change = user.must_change
            sess.touched = self._clock()
            return sess

    def logout(self, sid: str, ip: str | None) -> None:
        with self._lock:
            sess = self._drop_locked(sid) if isinstance(sid, str) else None
        if sess is not None:
            self._store.log("logout", sess.username, ip, "ok")

    # -- account changes -----------------------------------------------------
    def change_password(self, session: Session, current: str, new: str,
                        ip: str | None) -> None:
        with self._acct_lock:
            with self._lock:
                if self._sessions.get(session.sid) is not session:
                    raise PermissionError("세션이 유효하지 않습니다.")
            user = self._store.get_user(session.username)
            if user is None or user.disabled:
                raise PermissionError("세션이 유효하지 않습니다.")
            if not isinstance(current, str) or len(current) > _MAX_PASSWORD_CHARS \
                    or not verify_password(current, user.pw_hash):
                locked = self._record_failure(user, self._wall())
                self._store.log("password_changed", user.username, ip, "invalid")
                if locked:
                    self._drop_user_sessions(user.username)
                raise ValueError("현재 비밀번호가 맞지 않습니다.")
            check_new_password(new)
            if new == current:
                raise ValueError("새 비밀번호는 현재 비밀번호와 달라야 합니다.")
            version = user.password_version + 1
            new_hash = hash_password(new)
            with self._lock:  # store bump and session version change together, so this session cannot be revoked
                self._store.update_user(
                    user.username, pw_hash=new_hash, must_change=False,
                    password_version=version, failed=0, locked_until=0.0)
                session.password_version = version
                session.must_change = False
            self._drop_user_sessions(user.username, keep=session.sid)
            self._store.log("password_changed", user.username, ip, "ok")

    def now(self) -> float:
        return self._wall()

    def admin_unlock(self, actor: Session, target: str, ip: str | None = None) -> None:
        """Clear a failure lock without touching the password."""
        with self._acct_lock:
            with self._lock:
                live = self._sessions.get(actor.sid) is actor
            admin = self._store.get_user(actor.username) if live else None
            if admin is None or admin.disabled or admin.role != "admin":
                raise PermissionError("관리자 권한이 필요합니다.")
            name = normalize_username(target)
            if self._store.get_user(name) is None:
                raise ValueError("없는 계정입니다.")
            self._store.update_user(name, failed=0, locked_until=0.0)
            self._store.log("admin_update_user", admin.username, ip, "ok",
                            target=name, action="unlock")

    def admin_update(self, actor: Session, target: str, *, role=None,
                     disabled=None, reset_password: str | None = None,
                     ip: str | None = None) -> None:
        with self._acct_lock:
            with self._lock:
                live = self._sessions.get(actor.sid) is actor
            admin = self._store.get_user(actor.username) if live else None
            if admin is None or admin.disabled or admin.role != "admin":
                raise PermissionError("관리자 권한이 필요합니다.")
            name = normalize_username(target)
            user = self._store.get_user(name)
            if user is None:
                raise ValueError("없는 계정입니다.")
            if role is not None and role not in ROLES:
                raise ValueError("알 수 없는 역할입니다.")
            if reset_password is not None:
                check_new_password(reset_password)
            loses_admin = (user.role == "admin" and not user.disabled and (
                disabled is True or (role is not None and role != "admin")))
            if loses_admin and self._store.count_active_admins() <= 1:
                raise ValueError(_LAST_ADMIN)

            if role is not None and role != user.role:
                self._store.update_user(name, role=role)
                self._store.log("admin_update_user", admin.username, ip, "ok",
                                target=name, action="role")
            if disabled is not None and disabled != user.disabled:
                fields = {"disabled": disabled}
                if not disabled:
                    fields.update(failed=0, locked_until=0.0)
                self._store.update_user(name, **fields)
                if disabled:
                    self._drop_user_sessions(name)
                self._store.log("admin_update_user", admin.username, ip, "ok",
                                target=name,
                                action="disable" if disabled else "enable")
            if reset_password is not None:
                self._store.update_user(
                    name, pw_hash=hash_password(reset_password), must_change=True,
                    password_version=user.password_version + 1, failed=0,
                    locked_until=0.0)
                self._drop_user_sessions(name)
                self._store.log("admin_reset_password", admin.username, ip, "ok",
                                target=name, action="reset")
