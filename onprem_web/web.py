"""Shared HTTP helpers: JSON responses, security headers, health check and allow-listed static files."""

from __future__ import annotations

import json
from pathlib import Path

from django.http import Http404, HttpResponse

import trid
from trid.onprem.server import CSP, GENERIC_ERROR

STATIC_DIR = Path(__file__).resolve().parent / 'static'
FONT_DIR = Path(trid.__file__).resolve().parent / 'onprem_ui' / 'fonts'

# Only these names are ever served (name -> content type); nothing is resolved from the request path.
STATIC = {
    'common.css': 'text/css; charset=utf-8',
    'vendor/htmx.min.js': 'text/javascript; charset=utf-8',
}
FONTS = {'PretendardVariable.woff2': 'font/woff2'}


def json_response(status: int, data) -> HttpResponse:
    return HttpResponse(json.dumps(data, ensure_ascii=False).encode('utf-8'), status=status,
                        content_type='application/json; charset=utf-8')


class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        for key, val in {
            'Content-Security-Policy': CSP, 'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer',
        }.items():
            response[key] = val
        if 'Cache-Control' not in response:
            response['Cache-Control'] = 'no-store'
        return response


def healthz(request):
    return json_response(200, {'ok': True})


def _serve(base: Path, mime: str, name: str, cache: str) -> HttpResponse:
    try:
        data = (base / name).read_bytes()
    except OSError:
        raise Http404
    response = HttpResponse(data, content_type=mime)
    response['Cache-Control'] = cache
    return response


def static_file(request, name):
    if name not in STATIC:
        raise Http404
    return _serve(STATIC_DIR, STATIC[name], name, 'no-store')


def font_file(request, name):
    if name not in FONTS:
        raise Http404
    # The 2 MB font never changes between releases; let the browser keep it for a day.
    return _serve(FONT_DIR, FONTS[name], name, 'private, max-age=86400')


def not_found(request, exception=None):
    return json_response(404, {'error': '없는 페이지입니다.'})


def server_error(request):
    return json_response(500, {'error': GENERIC_ERROR})
