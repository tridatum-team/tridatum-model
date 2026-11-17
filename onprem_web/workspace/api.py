"""JSON API over the workspace services: /api/load, /api/ask, /api/clear.

TridSessionMiddleware has already checked Origin, login, idle time, the per-minute limit and the
must-change gate, and Django's CSRF middleware the X-Trid-Token header. Body checks and messages
follow trid.onprem.server read_body().
"""

from __future__ import annotations

import json

from django.contrib.auth import HASH_SESSION_KEY
from django.contrib.sessions.models import Session
from django.core.exceptions import RequestDataTooBig
from django.http.request import RawPostDataException
from django.utils import timezone
from django.utils.crypto import constant_time_compare

from onprem_web import runtime
from onprem_web.accounts import middleware as mw
from onprem_web.accounts.models import User
from onprem_web.web import json_response
from trid.onprem.server import SMALL_BODY

from . import services

_NOT_FOUND = '없는 페이지입니다.'
_TOO_BIG = '입력이 너무 큽니다.'
_BAD_REQUEST = '허용되지 않은 요청입니다.'


class _BodyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(status)
        self.response = json_response(status, {'error': message})


def _read_body(request, limit: int) -> bytes:
    """The raw body, as read_body() of the old server; raises _BodyError with its responses."""
    if request.META.get('HTTP_TRANSFER_ENCODING'):
        raise _BodyError(400, _BAD_REQUEST)
    raw = request.META.get('CONTENT_LENGTH')
    if raw is None or raw == '':
        raise _BodyError(411, '요청 길이가 필요합니다.')
    try:
        size = int(raw)
    except ValueError:
        size = -1
    if size < 0:
        raise _BodyError(400, '요청 길이가 올바르지 않습니다.')
    if size > limit:
        raise _BodyError(413, _TOO_BIG)
    try:
        body = request.body
    except RequestDataTooBig:
        raise _BodyError(413, _TOO_BIG) from None
    except RawPostDataException:  # a form/multipart body was already parsed: never accepted here
        raise _BodyError(400, _BAD_REQUEST) from None
    if len(body) != size:
        raise _BodyError(400, '입력 전송이 완료되지 않았습니다.')
    return body


def _json(body: bytes):
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        value = None
    return value if isinstance(value, dict) else None


def _gate(request, limit: int):
    """(entry, body) for a logged-in POST, or (None, response) to send back."""
    if request.method != 'POST':
        return None, json_response(404, {'error': _NOT_FOUND})
    entry = getattr(request, 'trid_entry', None)
    if entry is None:
        return None, mw.login_required_response()
    try:
        return entry, _read_body(request, limit)
    except _BodyError as e:
        return None, e.response


def _still_valid(request, key, entry) -> bool:
    """The login that asked is still the same, live login (checked after generation)."""
    if runtime.get().registry.get(key) is not entry:
        return False
    user = User.objects.filter(pk=request.user.pk).first()
    if user is None or not user.is_active:
        return False
    if not constant_time_compare(request.session.get(HASH_SESSION_KEY, ''), user.get_session_auth_hash()):
        return False
    return Session.objects.filter(session_key=key, expire_date__gt=timezone.now()).exists()


def _end_login(request, key):
    """The login was revoked or expired meanwhile: drop its document and send 401 with a cleared cookie."""
    request.session.flush()
    runtime.get().registry.drop(key)
    return mw.login_required_response()


def load(request):
    entry, body = _gate(request, runtime.get().config.pdf.max_bytes)
    if entry is None:
        return body
    key = request.session.session_key
    registry = runtime.get().registry
    # Any X-File-Name header is ignored: the server never learns the file name.
    result = services.load_pdf(entry, body, request.user.username, request.trid_ip,
                               attach=lambda doc, doc_id: registry.attach(key, entry, doc, doc_id))
    if result.status == 401:
        return _end_login(request, key)
    if result.status == 200:
        request.session['had_document'] = True
    return json_response(result.status, result.data)


def ask(request):
    entry, body = _gate(request, SMALL_BODY)
    if entry is None:
        return body
    key = request.session.session_key
    value = _json(body) or {}
    result = services.ask(entry, value.get('question'), value.get('document_id'),
                          request.user.username, request.trid_ip,
                          still_valid=lambda: _still_valid(request, key, entry),
                          had_document=request.session.get('had_document') is True)
    if result.status == 401:
        return _end_login(request, key)
    return json_response(result.status, result.data)


def clear(request):
    entry, body = _gate(request, SMALL_BODY)
    if entry is None:
        return body
    result = services.clear(entry, request.user.username, request.trid_ip, session=request.session)
    return json_response(result.status, result.data)
