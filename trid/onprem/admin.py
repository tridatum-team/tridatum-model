"""Admin routes of the on-prem server. server.py authenticates (session, CSRF, must_change)
and checks the admin role; these handlers only validate input and call Store/Auth.

There is deliberately no route that edits or deletes usage events.
"""

from __future__ import annotations

import csv
import io
import time
from urllib.parse import parse_qs

from trid.onprem.store import EVENTS, ROLES, normalize_username

PAGE_SIZE = 200
NOT_FOUND_USER = '사용자를 찾을 수 없습니다.'
BAD_REQUEST = '요청 값이 올바르지 않습니다.'
POST_ROUTES = frozenset({'/api/admin/users/list', '/api/admin/users/create',
                         '/api/admin/users/update', '/api/admin/events'})


class BadRequest(ValueError):
    pass


def _number(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
            or value in (float('inf'), float('-inf')):
        raise BadRequest(f'{name} 값이 올바르지 않습니다.')
    return value


def _text(value, name, required=False):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise BadRequest(f'{name} 값이 올바르지 않습니다.')
    return value


def _filters(source, number):
    """since/until/username/event from a dict; `number` coerces numeric fields."""
    event = _text(source.get('event'), 'event')
    if event is not None and event not in EVENTS:
        raise BadRequest('알 수 없는 이벤트입니다.')
    return {'since': number(source.get('since'), 'since'), 'until': number(source.get('until'), 'until'),
            'username': _text(source.get('username'), 'username'), 'event': event}


class AdminApi:
    def __init__(self, store, auth):
        self.store = store
        self.auth = auth

    # Returns (status, json value) for POST routes.
    def post(self, path, session, request, ip):
        try:
            return getattr(self, 'do_' + path.rsplit('/', 1)[1])(session, request, ip)
        except BadRequest as e:
            return 400, {'error': str(e)}
        except PermissionError:
            return 403, {'error': '관리자 권한이 필요합니다.'}
        except ValueError as e:
            message = str(e)
            return (404 if message == '없는 계정입니다.' else 400), {
                'error': NOT_FOUND_USER if message == '없는 계정입니다.' else message}

    def do_list(self, session, request, ip):
        now = self.auth.now()
        return 200, [{'username': u.username, 'role': u.role, 'disabled': u.disabled,
                      'locked': u.locked_until > now, 'must_change': u.must_change}
                     for u in self.store.list_users()]

    def do_create(self, session, request, ip):
        username = _text(request.get('username'), 'username', True)
        password = _text(request.get('password'), 'password', True)
        role = request.get('role')
        if role not in ROLES:
            raise BadRequest('알 수 없는 역할입니다.')
        user = self.store.create_user(username, password, role, must_change=True)
        self.store.log('admin_create_user', session.username, ip, 'ok', target=user.username, action='create')
        return 200, {'username': user.username, 'role': user.role, 'must_change': True}

    def do_update(self, session, request, ip):
        username = _text(request.get('username'), 'username', True)
        role, disabled = request.get('role'), request.get('disabled')
        reset, unlock = request.get('reset_password'), request.get('unlock')
        if role is not None and role not in ROLES:
            raise BadRequest('알 수 없는 역할입니다.')
        if disabled is not None and not isinstance(disabled, bool):
            raise BadRequest(BAD_REQUEST)
        if unlock is not None and not isinstance(unlock, bool):
            raise BadRequest(BAD_REQUEST)
        reset = _text(reset, 'reset_password')
        try:
            target = normalize_username(username)
        except ValueError:
            return 404, {'error': NOT_FOUND_USER}
        if self.store.get_user(target) is None:
            return 404, {'error': NOT_FOUND_USER}
        self.auth.admin_update(session, target, role=role, disabled=disabled,
                               reset_password=reset, ip=ip)
        if unlock:
            self.auth.admin_unlock(session, target, ip)
        return 200, {'ok': True}

    def do_events(self, session, request, ip):
        offset = request.get('offset', 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise BadRequest('offset 값이 올바르지 않습니다.')
        return 200, self.store.query_events(**_filters(request, _number), limit=PAGE_SIZE, offset=offset)

    def csv(self, session, query, ip):
        """GET /api/admin/events.csv: returns (status, value, extra) where value is bytes on success."""
        params = {k: v[-1] for k, v in parse_qs(query, keep_blank_values=False).items()}

        def number(value, name):
            if value is None:
                return None
            try:
                return _number(float(value), name)
            except ValueError:
                raise BadRequest(f'{name} 값이 올바르지 않습니다.') from None

        try:
            filters = _filters(params, number)
        except BadRequest as e:
            return 400, {'error': str(e)}, {}
        text = self.store.export_csv(**filters)
        count = max(0, sum(1 for _ in csv.reader(io.StringIO(text))) - 1)
        self.store.log('admin_export', session.username, ip, 'ok', count=count)
        name = time.strftime('trid-usage-%Y%m%d.csv', time.localtime(self.auth.now()))
        return 200, b'\xef\xbb\xbf' + text.encode('utf-8'), {
            'Content-Disposition': f'attachment; filename="{name}"'}
