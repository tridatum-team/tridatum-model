"""HTML pages and htmx fragments: GET /, /ui/ask, /ui/document, /ui/clear, /ui/password.

Same gates as the JSON API (TridSessionMiddleware, Django CSRF via X-Trid-Token, the body checks of
api._gate). Errors other than the password form's go back as the API's JSON with its status; the
page script shows the message. Every document and answer value is autoescaped by the templates.
"""

from __future__ import annotations

from django.http import HttpResponse
from django.shortcuts import render

from onprem_web import runtime
from onprem_web.accounts import middleware as mw
from onprem_web.accounts.views import change_password
from onprem_web.web import json_response
from trid.onprem.model_card import MODEL_CARD
from trid.onprem.server import SMALL_BODY, model_card_html

from . import api, services
from .present import present_answer

_NOT_FOUND = '없는 페이지입니다.'
_PASSWORD_FIELDS = '현재 비밀번호와 새 비밀번호를 입력해 주세요.'


def home(request):
    if request.method not in ('GET', 'HEAD'):
        return json_response(404, {'error': _NOT_FOUND})
    user = request.user
    status_text = MODEL_CARD['status']
    if not user.is_authenticated:
        return render(request, 'login.html', {'status_text': status_text})
    return render(request, 'workspace.html', {
        'logged_in': True, 'username': user.username, 'role': user.role, 'must_change': user.must_change,
        'show_password': user.must_change, 'max_bytes': int(runtime.get().config.pdf.max_bytes),
        'status_text': status_text, 'model_card': model_card_html(),
    })


def ask(request):
    entry, body = api._gate(request, SMALL_BODY)
    if entry is None:
        return body
    key = request.session.session_key
    question = request.POST.get('question')
    if isinstance(question, str):
        question = question.strip()  # as the old page sent question.trim()
    result = services.ask(entry, question, request.POST.get('document_id'),
                          request.user.username, request.trid_ip,
                          still_valid=lambda: api._still_valid(request, key, entry),
                          had_document=request.session.get('had_document') is True)
    if result.status == 401:
        return api._end_login(request, key)
    if result.status != 200:
        return json_response(result.status, result.data)
    return render(request, 'partials/_answer.html', present_answer(result.data))


def document(request):
    if request.method != 'GET':
        return json_response(404, {'error': _NOT_FOUND})
    entry = request.trid_entry
    if entry is None:
        return mw.login_required_response()
    doc = entry.document
    if doc is None:
        return HttpResponse(b'', content_type='text/html; charset=utf-8')
    return render(request, 'partials/_document.html', {'pages': list(doc.pages)})


def clear(request):
    entry, body = api._gate(request, SMALL_BODY)
    if entry is None:
        return body
    services.clear(entry, request.user.username, request.trid_ip, session=request.session)
    return HttpResponse(b'', content_type='text/html; charset=utf-8')


def password(request):
    entry, body = api._gate(request, SMALL_BODY)
    if entry is None:
        return body
    current, new = request.POST.get('current'), request.POST.get('new')
    if not isinstance(current, str) or not isinstance(new, str):
        outcome = _PASSWORD_FIELDS
    else:
        outcome = change_password(request, current, new)
    if isinstance(outcome, str):
        return render(request, 'partials/_password.html', {'show_password': True, 'error': outcome})
    if outcome is not None:
        return outcome
    response = HttpResponse(b'', content_type='text/html; charset=utf-8')
    response['HX-Redirect'] = '/'
    return response
