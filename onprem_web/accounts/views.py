"""Auth JSON API (/api/login, /api/logout, /api/password).

Statuses and messages are those of trid.onprem.server login()/password()/post().
"""

from __future__ import annotations

import json

from django.contrib import auth
from django.views.decorators.csrf import csrf_exempt

from onprem_web import runtime
from onprem_web.accounts import middleware as mw
from onprem_web.accounts import services
from onprem_web.accounts.models import User
from onprem_web.accounts.sessions import end_user_sessions
from onprem_web.audit.services import log
from onprem_web.web import json_response
from trid.onprem.server import BAD_LOGIN

BACKEND = 'django.contrib.auth.backends.ModelBackend'
BUSY = '현재 접속 인원이 많습니다. 잠시 후 다시 시도해 주세요.'
_NOT_FOUND = '없는 페이지입니다.'


def _json_body(request):
    try:
        value = json.loads(request.body)
    except (ValueError, RecursionError):
        value = None
    return value if isinstance(value, dict) else None


def _post_only(request):
    return None if request.method == 'POST' else json_response(404, {'error': _NOT_FOUND})


@csrf_exempt  # Origin was checked by TridSessionMiddleware; there is no session token before login
def api_login(request):
    if (wrong := _post_only(request)) is not None:
        return wrong
    body = _json_body(request)
    if body is None or not isinstance(body.get('username'), str) \
            or not isinstance(body.get('password'), str):
        return json_response(400, {'error': '아이디와 비밀번호를 입력해 주세요.'})
    ip = request.trid_ip
    registry = runtime.get().registry
    old_key = request.session.session_key
    user = services.attempt_login(body['username'], body['password'], ip,
                                  has_capacity=lambda: registry.has_room(old_key))
    if user is services.BUSY:
        return json_response(503, {'error': BUSY})
    if user is None:
        return json_response(401, {'error': BAD_LOGIN})
    auth.login(request, user, backend=BACKEND)
    request.session['pv'] = user.password_version
    key = request.session.session_key
    if old_key and old_key != key:
        registry.drop(old_key)
    if not mw.admit(registry, key, user.username):  # lost a race for the last slot
        request.session.flush()
        request.META['CSRF_COOKIE_NEEDS_UPDATE'] = False  # else the rotated token re-creates a session row
        log('login_failed', user.username, ip, 'busy')
        return json_response(503, {'error': BUSY})
    return json_response(200, {'status': 'must_change' if user.must_change else 'ok'})


def api_logout(request):
    if (wrong := _post_only(request)) is not None:
        return wrong
    key = request.session.session_key
    log('logout', request.user.username, request.trid_ip, 'ok')
    auth.logout(request)
    runtime.get().registry.drop(key)
    return json_response(200, {'ok': True})


def api_password(request):
    if (wrong := _post_only(request)) is not None:
        return wrong
    body = _json_body(request)
    if body is None or not isinstance(body.get('current'), str) or not isinstance(body.get('new'), str):
        return json_response(400, {'error': '현재 비밀번호와 새 비밀번호를 입력해 주세요.'})
    outcome = change_password(request, body['current'], body['new'])
    if isinstance(outcome, str):
        return json_response(400, {'error': outcome})
    return outcome or json_response(200, {'ok': True})


def change_password(request, current: str, new: str):
    """Shared by /api/password and /ui/password: None on success, the error text, or a 401 response."""
    registry = runtime.get().registry
    user = request.user
    old_key = request.session.session_key
    try:
        services.change_password(user, current, new, request.trid_ip)
    except PermissionError:
        request.session.flush()
        registry.drop(old_key)
        return mw.login_required_response()
    except ValueError as e:
        locked_until = User.objects.filter(pk=user.pk).values_list('locked_until', flat=True).first()
        if locked_until is not None and locked_until > services._clock():
            # As in Auth.change_password: a lock from wrong current passwords ends all the user's sessions.
            end_user_sessions(user.username)
            request.session.flush()
        return str(e)
    auth.update_session_auth_hash(request, user)  # cycles the session key
    request.session['pv'] = user.password_version
    key = request.session.session_key
    registry.rekey(old_key, key)
    end_user_sessions(user.username, keep=key)
    return None
