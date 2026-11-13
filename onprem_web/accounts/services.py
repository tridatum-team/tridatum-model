"""Account rules: creation, login failure counting/lockout, password changes, admin edits.

Ports trid.onprem.auth.Auth (login, change_password, admin_update, admin_unlock) and
trid.onprem.admin do_create; messages, audit events and detail values are the old ones.
Sessions are not handled here: bumping password_version changes User.get_session_auth_hash().
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Callable

from django.contrib.auth.hashers import check_password, make_password
from django.db import IntegrityError, transaction

from onprem_web.audit.services import log
from trid.onprem.auth import _LAST_ADMIN, _MAX_PASSWORD_CHARS
from trid.onprem.passwords import check_new_password
from trid.onprem.store import ROLES, normalize_username

from .models import User
from .sessions import end_user_sessions

MAX_FAILURES = 5
LOCK_SECONDS = 900

_clock = time.time  # wall time for locked_until; tests patch this or pass now=
_acct_lock = threading.RLock()  # serializes account mutations (Auth._acct_lock), held across scrypt
_dummy: list[str] = []

BUSY = 'busy'  # attempt_login result: credentials fine, but no session slot (HTTP 503)

_NOT_ADMIN = '관리자 권한이 필요합니다.'
_BAD_SESSION = '세션이 유효하지 않습니다.'
_NO_USER = '없는 계정입니다.'
_BAD_ROLE = '알 수 없는 역할입니다.'


def _dummy_hash() -> str:
    """Verified against when the account is unknown or locked, to even out timing."""
    if not _dummy:
        with _acct_lock:
            if not _dummy:
                _dummy.append(make_password(secrets.token_urlsafe(16)))
    return _dummy[0]


def _get(username) -> User | None:
    if not isinstance(username, str):
        return None
    try:
        name = normalize_username(username)
    except ValueError:
        return None
    return User.objects.filter(username=name).first()


def _verify(password: str, encoded: str) -> bool:
    return check_password(password, encoded)  # no setter: never rehash/save as a side effect


def _save(user: User, **fields) -> None:
    for key, value in fields.items():
        setattr(user, key, value)
    User.objects.filter(pk=user.pk).update(**fields)


def _expire_served_lock(user: User, now: float) -> None:
    if user.locked_until and user.locked_until <= now:
        _save(user, failed=0, locked_until=0.0)


def _record_failure(user: User, now: float) -> bool:
    """Count one failure; return True if the account is (now) locked. An active lock is never extended."""
    _expire_served_lock(user, now)
    if user.locked_until > now:
        return True
    fields = {'failed': user.failed + 1}
    locked = fields['failed'] >= MAX_FAILURES
    if locked:
        fields['locked_until'] = now + LOCK_SECONDS
    _save(user, **fields)
    return locked


def _admin(actor) -> User:
    admin = User.objects.filter(pk=getattr(actor, 'pk', None)).first()
    if admin is None or not admin.is_active or admin.role != 'admin':
        raise PermissionError(_NOT_ADMIN)
    return admin


# -- creation ------------------------------------------------------------------

def create_user(username, password, role, must_change=True) -> User:
    name = normalize_username(username)
    if role not in ROLES:
        raise ValueError(_BAD_ROLE)
    check_new_password(password)
    user = User(username=name, role=role, must_change=bool(must_change))
    user.set_password(password)
    with _acct_lock:
        try:
            with transaction.atomic():
                user.save(force_insert=True)
        except IntegrityError:
            raise ValueError('이미 있는 계정입니다.') from None
    return user


def admin_create(actor, username, password, role, ip) -> User:
    with _acct_lock:
        admin = _admin(actor)
        user = create_user(username, password, role, must_change=True)
        log('admin_create_user', admin.username, ip, 'ok', target=user.username, action='create')
    return user


# -- login ---------------------------------------------------------------------

def attempt_login(username: str, password: str, ip: str | None, now: float | None = None,
                  has_capacity: Callable[[], bool] | None = None) -> User | str | None:
    """Return the user on success; None for unknown, locked, disabled or wrong password.

    `has_capacity` runs after the password is verified (as in Auth.login); if it says no,
    the attempt is logged as login_failed/busy and BUSY is returned.
    """
    if not isinstance(password, str) or len(password) > _MAX_PASSWORD_CHARS:
        password = ''
    with _acct_lock:
        user = _get(username)
        now = _clock() if now is None else now
        if user is None:
            _verify(password, _dummy_hash())
            log('login_failed', None, ip, 'invalid')
            return None
        _expire_served_lock(user, now)
        if user.locked_until > now:
            _verify(password, _dummy_hash())
            log('login_locked', user.username, ip, 'locked')
            return None
        ok = _verify(password, user.password)
        if not user.is_active:
            log('login_failed', user.username, ip, 'invalid', error='disabled')
            return None
        if not ok:
            _record_failure(user, now)
            log('login_failed', user.username, ip, 'invalid')
            return None
        if user.failed:
            _save(user, failed=0)
        if has_capacity is not None and not has_capacity():
            log('login_failed', user.username, ip, 'busy')
            return BUSY
        log('login_ok', user.username, ip, 'ok')
        return user


# -- account changes -------------------------------------------------------------

def change_password(user, current: str, new: str, ip) -> None:
    with _acct_lock:
        fresh = User.objects.filter(pk=getattr(user, 'pk', None)).first()
        if fresh is None or not fresh.is_active:
            raise PermissionError(_BAD_SESSION)
        if not isinstance(current, str) or len(current) > _MAX_PASSWORD_CHARS \
                or not _verify(current, fresh.password):
            _record_failure(fresh, _clock())
            log('password_changed', fresh.username, ip, 'invalid')
            raise ValueError('현재 비밀번호가 맞지 않습니다.')
        check_new_password(new)
        if new == current:
            raise ValueError('새 비밀번호는 현재 비밀번호와 달라야 합니다.')
        fresh.set_password(new)
        _save(fresh, password=fresh.password, must_change=False,
              password_version=fresh.password_version + 1, failed=0, locked_until=0.0)
        for key in ('password', 'must_change', 'password_version', 'failed', 'locked_until'):
            setattr(user, key, getattr(fresh, key))
        log('password_changed', fresh.username, ip, 'ok')


def admin_update(actor, target: str, *, role=None, disabled=None, reset_password=None,
                 unlock=None, ip) -> None:
    with _acct_lock:
        admin = _admin(actor)
        name = normalize_username(target)
        user = User.objects.filter(username=name).first()
        if user is None:
            raise ValueError(_NO_USER)
        if role is not None and role not in ROLES:
            raise ValueError(_BAD_ROLE)
        if reset_password is not None:
            check_new_password(reset_password)
        loses_admin = (user.role == 'admin' and user.is_active and (
            disabled is True or (role is not None and role != 'admin')))
        if loses_admin and User.objects.filter(role='admin', is_active=True).count() <= 1:
            raise ValueError(_LAST_ADMIN)

        with transaction.atomic():
            if role is not None and role != user.role:
                _save(user, role=role)
                log('admin_update_user', admin.username, ip, 'ok', target=name, action='role')
            if disabled is not None and disabled == user.is_active:
                fields = {'is_active': not disabled}
                if not disabled:
                    fields.update(failed=0, locked_until=0.0)
                _save(user, **fields)
                if disabled:
                    end_user_sessions(name)
                log('admin_update_user', admin.username, ip, 'ok', target=name,
                    action='disable' if disabled else 'enable')
            if reset_password is not None:
                _save(user, password=make_password(reset_password), must_change=True,
                      password_version=user.password_version + 1, failed=0, locked_until=0.0)
                end_user_sessions(name)
                log('admin_reset_password', admin.username, ip, 'ok', target=name, action='reset')
            if unlock:
                _save(user, failed=0, locked_until=0.0)
                log('admin_update_user', admin.username, ip, 'ok', target=name, action='unlock')
