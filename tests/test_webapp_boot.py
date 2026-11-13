import unittest

import webapp_support as S  # noqa: F401  (must come before any Django import)
import django.test
from trid.onprem.server import CSP


class BootTests(django.test.TestCase):
    def test_healthz_and_security_headers(self):
        r = S.client().get('/healthz', secure=True)
        self.assertEqual(r.status_code, 200); self.assertEqual(r.json(), {'ok': True})
        self.assertEqual(r['Content-Security-Policy'], CSP)
        for k, v in {'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                     'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store'}.items():
            self.assertEqual(r[k], v)

    def test_font_cached_others_no_store(self):
        r = S.client().get('/fonts/PretendardVariable.woff2', secure=True)
        self.assertEqual(r['Content-Type'], 'font/woff2'); self.assertEqual(r['Cache-Control'], 'private, max-age=86400')
        self.assertEqual(b''.join(r.streaming_content if r.streaming else [r.content])[:4], b'wOF2')
        self.assertEqual(S.client().get('/static/vendor/htmx.min.js', secure=True)['Cache-Control'], 'no-store')
        self.assertEqual(S.client().get('/static/../settings.py', secure=True).status_code, 404)

    def test_django_defaults_are_overridden(self):
        from django.conf import settings as s
        self.assertFalse(s.DEBUG); self.assertEqual(s.ALLOWED_HOSTS, ['test.example'])
        self.assertEqual(s.CSRF_TRUSTED_ORIGINS, ['https://test.example'])
        self.assertEqual(s.FILE_UPLOAD_HANDLERS, ['django.core.files.uploadhandler.MemoryFileUploadHandler'])
        self.assertTrue(s.CSRF_USE_SESSIONS); self.assertEqual(s.CSRF_HEADER_NAME, 'HTTP_X_TRID_TOKEN')
        self.assertEqual(s.SESSION_COOKIE_NAME, '__Host-trid'); self.assertEqual(s.SESSION_COOKIE_SAMESITE, 'Strict')
        self.assertNotIn('django.contrib.admin', s.INSTALLED_APPS); self.assertNotIn('django.contrib.messages', s.INSTALLED_APPS)
        self.assertEqual(s.PASSWORD_HASHERS[0], 'django.contrib.auth.hashers.ScryptPasswordHasher')
        self.assertEqual(s.ADMINS, []); self.assertEqual(s.DATA_UPLOAD_MAX_MEMORY_SIZE, 2 * 1024 * 1024)

    def test_unknown_path_is_korean_json_404(self):
        r = S.client().get('/nope', secure=True)
        self.assertEqual((r.status_code, r.json()), (404, {'error': '없는 페이지입니다.'}))

    def test_runtime_get_requires_install(self):
        from onprem_web import runtime
        rt = S.make_runtime()
        self.assertIs(runtime.get(), rt)


if __name__ == '__main__':
    unittest.main()
