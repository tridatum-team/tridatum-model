import io
import logging
import unittest

import webapp_support as S  # noqa: F401  (must come before any Django import)
import django.test
import test_onprem_server as T
from django.test import Client
from trid.onprem.server import GENERIC_ERROR

LOGGERS = ('django', 'django.request', 'django.server', 'django.security', 'waitress')


class _Capture:
    """Point every handler of the policy loggers (and root) at one buffer."""

    def __enter__(self):
        self.buf = io.StringIO()
        self.saved = []
        seen = set()
        for name in LOGGERS + ('',):
            for h in logging.getLogger(name).handlers:
                if id(h) in seen or not hasattr(h, 'stream'):
                    continue
                seen.add(id(h))
                self.saved.append((h, h.stream))
                h.stream = self.buf
        # Anything that still reaches root through a fresh handler is captured too.
        self.root = logging.StreamHandler(self.buf)
        logging.getLogger().addHandler(self.root)
        return self

    def __exit__(self, *exc):
        logging.getLogger().removeHandler(self.root)
        for h, s in self.saved:
            h.stream = s


class LoggingTests(django.test.TestCase):
    def test_request_error_log_has_type_only(self):
        c = Client(raise_request_exception=False, enforce_csrf_checks=True,
                   HTTP_HOST=S.HOST, HTTP_ORIGIN=S.ORIGIN, HTTP_X_REAL_IP='10.0.0.5')
        with _Capture() as cap:
            r = c.get('/__boom?q=SECRET', secure=True)
        out = cap.buf.getvalue()
        self.assertEqual((r.status_code, r.json()), (500, {'error': GENERIC_ERROR}))
        self.assertIn('ValueError', out)
        for bad in (T.SECRET_QUESTION, T.SECRET_NAME, 'q=SECRET', 'Traceback'):
            self.assertNotIn(bad, out)

    def test_waitress_exception_log_has_no_message(self):
        with _Capture() as cap:
            try:
                raise ValueError(T.SECRET_NAME)
            except ValueError:
                logging.getLogger('waitress').exception('x %s', T.SECRET_NAME)
        out = cap.buf.getvalue()
        self.assertIn('waitress ERROR ValueError', out)
        self.assertNotIn(T.SECRET_NAME, out)
        self.assertNotIn('Traceback', out)

    def test_policy_shape(self):
        from django.conf import settings as s
        for name in LOGGERS:
            self.assertIn(name, s.LOGGING['loggers'])
            self.assertNotIn('mail_admins', s.LOGGING['loggers'][name]['handlers'])
        self.assertFalse(s.LOGGING['disable_existing_loggers'])


if __name__ == '__main__':
    unittest.main()
