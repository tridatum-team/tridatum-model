"""Korean admin screens: GET /admin, POST /ui/admin/users[/<username>], GET /ui/admin/events and
POST /api/admin/events.csv.

TridSessionMiddleware has already checked Origin, login, idle time, the per-minute limit and the
must-change gate; Django's CSRF middleware checks X-Trid-Token (htmx) or csrfmiddlewaretoken (the CSV
form, so the token never appears in a URL). Validation and messages follow trid.onprem.admin.AdminApi.
Errors go back as JSON with the old status; admin.js shows the message.

There is deliberately no route that edits or deletes usage events.
"""

from __future__ import annotations

import csv
import io
import math
import time

from django.http import HttpResponse
from django.shortcuts import render

from onprem_web.accounts import middleware as mw
from onprem_web.accounts import services as accounts
from onprem_web.accounts.models import User
from onprem_web.audit import services as audit
from onprem_web.web import json_response
from onprem_web.workspace import api
from trid.onprem.admin import BAD_REQUEST, NOT_FOUND_USER, PAGE_SIZE, BadRequest, _filters, _text
from trid.onprem.model_card import MODEL_CARD
from trid.onprem.server import ADMIN_ONLY, SMALL_BODY
from trid.onprem.store import ROLES, normalize_username

_NOT_FOUND = '없는 페이지입니다.'
_NO_ACCOUNT = '없는 계정입니다.'  # accounts.services' message for an unknown target
_METHOD = '허용되지 않은 요청입니다.'

# Copied from trid/onprem_ui/admin.js (tests compare them).
EVENT_LABELS = {
    'login_ok': '로그인', 'login_failed': '로그인 실패', 'login_locked': '로그인 잠김', 'logout': '로그아웃',
    'session_expired': '세션 만료', 'password_changed': '비밀번호 변경', 'pdf_upload': 'PDF 올리기', 'ask': '질문',
    'clear': '자료 지우기', 'admin_create_user': '계정 생성', 'admin_update_user': '계정 변경',
    'admin_reset_password': '비밀번호 초기화', 'admin_export': '기록 내보내기', 'retention_purge': '보관 기간 정리',
}
ROLE_LABELS = {'user': '직원', 'admin': '관리자'}


# -- gates ---------------------------------------------------------------------

def _denied(request):
    """401 without a live login, 403 for staff; None for an admin."""
    if getattr(request, 'trid_entry', None) is None:
        return mw.login_required_response()
    if request.user.role != 'admin':
        return json_response(403, {'error': ADMIN_ONLY})
    return None


def _admin_post(request):
    """(True, None) for an admin POST whose body passed the size checks, else (False, response)."""
    denied = _denied(request)
    if denied is not None:
        return False, denied
    entry, body = api._gate(request, SMALL_BODY)
    if entry is None:
        return False, body
    return True, None


def _error(e: Exception):
    if isinstance(e, PermissionError):
        return json_response(403, {'error': ADMIN_ONLY})
    message = str(e)
    if message == _NO_ACCOUNT:
        return json_response(404, {'error': NOT_FOUND_USER})
    return json_response(400, {'error': message})


# -- context -------------------------------------------------------------------

def _status_text(user: User, now: float) -> str:
    """statusText() of the old admin.js."""
    if not user.is_active:
        parts = ['잠금(관리자)']
    elif user.locked_until > now:
        parts = ['잠김(로그인 실패)']
    else:
        parts = ['사용 가능']
    if user.must_change:
        parts.append('비밀번호 변경 필요')
    return ' · '.join(parts)


def _users_context(message: str = '') -> dict:
    now = accounts._clock()
    rows = []
    for u in User.objects.order_by('username'):
        locked = u.locked_until > now
        other = 'user' if u.role == 'admin' else 'admin'
        rows.append({'username': u.username, 'role': ROLE_LABELS.get(u.role, u.role),
                     'status': _status_text(u, now), 'can_unlock': not u.is_active or locked,
                     'can_disable': u.is_active, 'next_role': other, 'next_role_label': ROLE_LABELS[other]})
    return {'users': rows, 'users_message': message}


def _short(key, value) -> str:
    # A sha256 is 64 characters; the table shows its first 12 (full value in the tooltip and the CSV).
    return str(value)[:12] + '…' if key == 'sha256' else _plain(value)


def _plain(value) -> str:
    """String(v) of JavaScript for the simple detail values (bool, int, float, str, None)."""
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if value is None:
        return 'null'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fallback_time(ts: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))


def _events_context(filters: dict, offset: int) -> dict:
    rows = []
    for e in audit.query_events(**filters, limit=PAGE_SIZE, offset=offset):
        entries = list((e['detail'] or {}).items())
        detail = ' · '.join(f'{k}: {_short(k, v)}' for k, v in entries)
        full = ' · '.join(f'{k}: {_plain(v)}' for k, v in entries)
        rows.append({'ts': e['ts'], 'time': _fallback_time(e['ts']),
                     'event': EVENT_LABELS.get(e['event'], e['event']),
                     'username': e['username'] if e['username'] is not None else '(알 수 없음)',
                     'result': e['result'], 'ip': e['ip'] or '', 'detail': detail,
                     'title': full if full != detail else ''})
    info = f'{offset + 1}–{offset + len(rows)}번째' if rows else '기록이 없습니다.'
    return {'events': rows, 'page_info': info, 'offset': offset,
            'prev_offset': max(0, offset - PAGE_SIZE), 'next_offset': offset + PAGE_SIZE,
            'has_prev': offset > 0, 'has_next': len(rows) >= PAGE_SIZE}


def _number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f'{name} 값이 올바르지 않습니다.') from None
    if not math.isfinite(number):
        raise BadRequest(f'{name} 값이 올바르지 않습니다.')
    return number


def _query_filters(params) -> dict:
    """since/until/username/event from a QueryDict; blank fields are left out, as parse_qs did."""
    source = {k: params.get(k) for k in ('since', 'until', 'username', 'event') if params.get(k)}
    return _filters(source, lambda v, name: None if v is None else _number(v, name))


# -- views ---------------------------------------------------------------------

def admin_page(request):
    if request.method not in ('GET', 'HEAD'):
        return json_response(404, {'error': _NOT_FOUND})
    if not request.user.is_authenticated:
        return render(request, 'login.html', {'status_text': MODEL_CARD['status']})
    denied = _denied(request)
    if denied is not None:
        return denied
    return render(request, 'admin.html', {
        'logged_in': True, 'username': request.user.username,
        'event_labels': EVENT_LABELS.items(), **_users_context(), **_events_context({}, 0)})


def create_user(request):
    ok, response = _admin_post(request)
    if not ok:
        return response
    try:
        username = _text(request.POST.get('username'), 'username', True)
        password = _text(request.POST.get('password'), 'password', True)
        role = request.POST.get('role')
        if role not in ROLES:
            raise BadRequest('알 수 없는 역할입니다.')
        user = accounts.admin_create(request.user, username, password, role, request.trid_ip)
    except (PermissionError, ValueError) as e:
        return _error(e)
    return render(request, 'partials/_users.html', _users_context(
        f'{user.username} 계정을 만들었습니다. 첫 로그인 때 비밀번호를 바꿉니다.'))


def _flag(value):
    if value is None:
        return None
    if value not in ('0', '1'):
        raise BadRequest(BAD_REQUEST)
    return value == '1'


def update_user(request, username):
    ok, response = _admin_post(request)
    if not ok:
        return response
    post = request.POST
    try:
        role = post.get('role')
        if role is not None and role not in ROLES:
            raise BadRequest('알 수 없는 역할입니다.')
        disabled, unlock = _flag(post.get('disabled')), _flag(post.get('unlock'))
        reset = _text(post.get('reset_password'), 'reset_password')
        try:
            target = normalize_username(username)
        except ValueError:
            return json_response(404, {'error': NOT_FOUND_USER})
        accounts.admin_update(request.user, target, role=role, disabled=disabled, reset_password=reset,
                              unlock=unlock, ip=request.trid_ip)
    except (PermissionError, ValueError) as e:
        return _error(e)
    # The confirmation texts of the old admin.js buttons.
    if unlock or disabled is False:
        message = f'{target} 계정의 잠금을 풀었습니다.'
    elif disabled:
        message = f'{target} 계정을 잠갔습니다.'
    elif reset is not None:
        message = f'{target}의 비밀번호를 초기화했습니다.'
    elif role is not None:
        message = f'{target}의 역할을 바꿨습니다.'
    else:
        message = ''
    return render(request, 'partials/_users.html', _users_context(message))


def events(request):
    if request.method != 'GET':
        return json_response(404, {'error': _NOT_FOUND})
    denied = _denied(request)
    if denied is not None:
        return denied
    try:
        filters = _query_filters(request.GET)
        raw = request.GET.get('offset') or '0'
        if not raw.isascii() or not raw.isdigit():
            raise BadRequest('offset 값이 올바르지 않습니다.')
        offset = int(raw)
    except BadRequest as e:
        return json_response(400, {'error': str(e)})
    return render(request, 'partials/_events.html', _events_context(filters, offset))


def events_csv(request):
    if request.method != 'POST':
        response = json_response(405, {'error': _METHOD})
        response['Allow'] = 'POST'
        return response
    ok, response = _admin_post(request)
    if not ok:
        return response
    try:
        filters = _query_filters(request.POST)
    except BadRequest as e:
        return json_response(400, {'error': str(e)})
    data = audit.export_csv(**filters)
    count = max(0, sum(1 for _ in csv.reader(io.StringIO(data.decode('utf-8-sig')))) - 1)
    audit.log('admin_export', request.user.username, request.trid_ip, 'ok', count=count)
    name = time.strftime('trid-usage-%Y%m%d.csv', time.localtime(accounts._clock()))
    response = HttpResponse(data, content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{name}"'
    return response
