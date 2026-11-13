"""Auth JSON API (/api/login, /api/logout, /api/password) and a placeholder home page.

Statuses and messages are those of trid.onprem.server login()/password()/post().
"""

from __future__ import annotations

import json

from django.contrib import auth
from django.http import HttpResponse
from django.middleware.csrf import get_token
from django.utils.html import escape
from django.views.decorators.csrf import csrf_exempt

from onprem_web import runtime
from onprem_web.accounts import middleware as mw
from onprem_web.accounts import services
from onprem_web.accounts.models import User
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
    user = services.attempt_login(body['username'], body['password'], ip)
    if user is None:
        return json_response(401, {'error': BAD_LOGIN})
    registry = runtime.get().registry
    old_key = request.session.session_key
    auth.login(request, user, backend=BACKEND)
    request.session['pv'] = user.password_version
    key = request.session.session_key
    if old_key and old_key != key:
        registry.drop(old_key)
    if not mw.admit(registry, key, user.username):
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
    registry = runtime.get().registry
    user = request.user
    old_key = request.session.session_key
    try:
        services.change_password(user, body['current'], body['new'], request.trid_ip)
    except PermissionError:
        request.session.flush()
        registry.drop(old_key)
        return mw.login_required_response()
    except ValueError as e:
        locked_until = User.objects.filter(pk=user.pk).values_list('locked_until', flat=True).first()
        if locked_until is not None and locked_until > services._clock():
            # As in Auth.change_password: a lock from wrong current passwords ends all the user's sessions.
            registry.drop_user(user.username)
            mw.delete_user_db_sessions(user.pk)
            request.session.flush()
        return json_response(400, {'error': str(e)})
    auth.update_session_auth_hash(request, user)  # cycles the session key
    request.session['pv'] = user.password_version
    key = request.session.session_key
    registry.rekey(old_key, key)
    registry.drop_user(user.username, keep=key)
    mw.delete_user_db_sessions(user.pk, keep=key)
    return json_response(200, {'ok': True})


def home(request):
    """Placeholder for GET / until Task 7 builds the real page; keeps the meta tags release_check reads."""
    user = request.user
    if not user.is_authenticated:
        text = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>로그인</title></head>'
                '<body><form id="login" action="/api/login" method="post"></form></body></html>')
    else:
        must = '<meta name="must-change" content="1">' if user.must_change else ''
        text = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                f'<meta name="csrf-token" content="{escape(get_token(request))}">'
                f'<meta name="username" content="{escape(user.username)}">{must}'
                '<title>문서 근거 찾기</title></head><body></body></html>')
    return HttpResponse(text.encode('utf-8'), content_type='text/html; charset=utf-8')
