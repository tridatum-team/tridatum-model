"""Ties Django's DB session to the in-memory SessionRegistry and applies the old server's gates.

Order per request (see the Task 5 brief): Origin on every POST, restart/revocation sync with the
registry, idle expiry, per-minute limit, must-change gate. Status codes and messages are those of
trid.onprem.server.
"""

from __future__ import annotations

import ipaddress

from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.contrib.sessions.models import Session
from django.utils import timezone

from onprem_web import runtime
from onprem_web.accounts import services
from onprem_web.audit.services import log
from onprem_web.web import json_response
from trid.onprem.server import MUST_CHANGE

LOGIN_REQUIRED = '로그인이 필요합니다.'
BAD_ORIGIN = '허용되지 않은 요청입니다.'
TOO_MANY = '요청이 많습니다. 1분 뒤 다시 시도해 주세요.'

# Never touch the session for these: no auth needed and no idle-timer refresh.
_PUBLIC_PREFIXES = ('/static/', '/fonts/')
_PUBLIC_PATHS = ('/healthz',)
_MUST_CHANGE_OPEN = ('/api/password', '/ui/password', '/api/logout', '/api/login')


def expected_origin(rt: runtime.Runtime) -> str:
    if rt.insecure_dev:
        return f'http://127.0.0.1:{rt.port}'
    return 'https://' + rt.config.hostname


def client_ip(request) -> str | None:
    raw = request.META.get('HTTP_X_REAL_IP', '').strip()
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return None


def is_api(path: str) -> bool:
    return path.startswith('/api/') or path.startswith('/ui/')


def login_required_response():
    response = json_response(401, {'error': LOGIN_REQUIRED})
    response.delete_cookie(settings.SESSION_COOKIE_NAME, path=settings.SESSION_COOKIE_PATH,
                           samesite=settings.SESSION_COOKIE_SAMESITE)
    return response


def delete_db_sessions(keys) -> None:
    keys = list(keys)
    if keys:
        Session.objects.filter(session_key__in=keys).delete()


def delete_user_db_sessions(user_pk, keep: str | None = None) -> None:
    """Delete every stored session of this user except `keep` (sessions are few: max_sessions)."""
    want = str(user_pk)
    doomed = []
    for row in Session.objects.filter(expire_date__gt=timezone.now()):
        if row.session_key != keep and row.get_decoded().get('_auth_user_id') == want:
            doomed.append(row.session_key)
    delete_db_sessions(doomed)


def prune(registry) -> None:
    """Expire idle registry entries and their DB rows, so a pruned login cannot come back as a 'restart'."""
    expired = registry.prune()
    delete_db_sessions(key for key, _ in expired)
    for _, username in expired:
        log('session_expired', username, None, 'expired')


def admit(registry, key: str, username: str) -> bool:
    prune(registry)
    return registry.admit(key, username)


class TridSessionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info
        public = path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)
        if request.method == 'POST' and request.META.get('HTTP_ORIGIN') != expected_origin(runtime.get()):
            return json_response(403, {'error': BAD_ORIGIN})
        if public:
            return self.get_response(request)

        registry = runtime.get().registry
        request.trid_ip = client_ip(request)
        request.trid_entry = None
        key = request.session.session_key  # read before request.user: a revoked session is flushed on load
        user = request.user

        if user.is_authenticated and user.locked_until > services._clock():
            # A lock (e.g. from wrong current passwords) ends the user's sessions.
            self._end(request, key)
            user = request.user

        if user.is_authenticated:
            if registry.get(key) is None:
                # The process restarted: the DB session survived, the in-memory state did not.
                if not admit(registry, key, user.username):
                    self._end(request, key)
                    return login_required_response() if is_api(path) else self.get_response(request)
            elif not registry.touch(key):
                log('session_expired', user.username, None, 'expired')
                self._end(request, key)
                return login_required_response() if is_api(path) else self.get_response(request)
        else:
            entry = registry.get(key)
            if entry is not None:
                # Password version changed or account disabled: Django already flushed the session.
                log('session_expired', entry.username, None, 'revoked')
                registry.drop(key)
            if is_api(path) and path != '/api/login':
                return login_required_response()
            return self.get_response(request)

        if request.method == 'POST' and is_api(path) and not registry.allow_call(key):
            return json_response(429, {'error': TOO_MANY})
        if user.must_change and path not in _MUST_CHANGE_OPEN \
                and not (request.method == 'GET' and path == '/'):
            return json_response(403, {'error': MUST_CHANGE, 'must_change': True})
        request.trid_entry = registry.get(key)
        return self.get_response(request)

    @staticmethod
    def _end(request, key) -> None:
        request.session.flush()
        request.user = AnonymousUser()
        runtime.get().registry.drop(key)
