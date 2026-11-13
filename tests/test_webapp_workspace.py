"""Upload / ask / clear services and the JSON API (/api/load, /api/ask, /api/clear).

Ports the scenarios of test_onprem_server.py and test_onprem_upload_limit.py to the Django build.
"""

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import webapp_support as S  # noqa: F401  (boots Django and migrates)
import test_onprem_server as T
from test_webapp_sessions import Browser

from django.contrib.sessions.models import Session
from django.test import TestCase

from onprem_web.accounts import services as accounts
from onprem_web.accounts.registry import SessionRegistry
from onprem_web.audit.models import Event
from onprem_web.workspace import services
from trid.document_case import CONTEXT_OVERFLOW
from trid.onprem.server import BUSY_ASK, SMALL_BODY
from trid.pdf_documents import DocumentError

PW = S.PW
PDF = b'%PDF-1.4 fake'
REUPLOAD = '문서를 다시 올려 주세요.'
FIRST = '이 브라우저에서 문서를 먼저 넣어 주세요.'


def fake_parse(body, name, tok, limits=None):
    if body.startswith(b'%PDF'):
        return T.fake_doc(name)
    raise DocumentError('PDF를 읽지 못했습니다.')


def details(event):
    return [json.loads(e.detail) for e in Event.objects.filter(event=event).order_by('id')]


def all_audit_text():
    return ''.join(f'{e.username}|{e.ip}|{e.event}|{e.result}|{e.detail}' for e in Event.objects.all())


def all_session_text():
    return ''.join(json.dumps(s.get_decoded(), ensure_ascii=False) for s in Session.objects.all())


class WorkspaceBrowser(Browser):
    def upload(self, body=PDF, **extra):
        return self.c.post('/api/load', body, content_type='application/pdf', secure=True,
                           HTTP_X_TRID_TOKEN=self.token, **extra)

    def ask(self, question, document_id):
        return self.post('/api/ask', {'question': question, 'document_id': document_id})


class WorkspaceTests(TestCase):
    answer = staticmethod(T.ok_answer)

    def setUp(self):
        self.calls = []

        def answer(document, question):
            self.calls.append(question)
            return type(self).answer(document, question)
        self.rt = S.make_runtime(answer=answer, lock_wait=0.1)
        patcher = mock.patch('onprem_web.workspace.services.parse_pdf', side_effect=fake_parse)
        self.parse = patcher.start()
        self.addCleanup(patcher.stop)

    @classmethod
    def setUpTestData(cls):  # once per class: each scrypt hash costs about 0.7 s
        accounts.create_user('alice', PW, 'user', must_change=False)
        accounts.create_user('bob', PW, 'user', must_change=False)
        cls.admin = accounts.create_user('root', PW, 'admin', must_change=False)

    def browser(self, name='alice'):
        b = WorkspaceBrowser()
        self.assertEqual(b.login(name).status_code, 200)
        return b

    def uploaded(self, name='alice'):
        b = self.browser(name)
        r = b.upload()
        self.assertEqual(r.status_code, 200, r.content)
        return b, r.json()['document_id']

    # -- load ----------------------------------------------------------------
    def test_load_ignores_file_name_header(self):
        b = self.browser()
        r = b.upload(HTTP_X_FILE_NAME=T.SECRET_NAME)
        self.assertEqual(r.status_code, 200)
        value = r.json()
        self.assertEqual(value['name'], 'document.pdf')
        self.assertEqual(value['pages'], ['본문 한 쪽'])
        self.assertEqual(value['warnings'], [])
        self.assertEqual(set(value), {'document_id', 'name', 'pages', 'warnings'})
        self.assertEqual(self.parse.call_args.args[1], 'document.pdf')
        self.assertIs(self.parse.call_args.kwargs['limits'], self.rt.config.pdf)
        self.assertNotIn(T.SECRET_NAME, r.content.decode())
        self.assertNotIn(T.SECRET_NAME, str(r.headers))
        self.assertNotIn(T.SECRET_NAME, all_audit_text())
        self.assertNotIn(T.SECRET_NAME, all_session_text())
        detail = details('pdf_upload')[0]
        self.assertEqual(len(detail['sha256']), 64)
        self.assertEqual((detail['pages'], detail['status']), (1, 'ok'))
        upload = Event.objects.get(event='pdf_upload')
        self.assertEqual((upload.username, upload.ip, upload.result), ('alice', '10.0.0.5', 'ok'))
        self.assertTrue(b.c.session.get('had_document'))
        entry = self.rt.registry.get(b.key)
        self.assertEqual(entry.document_id, value['document_id'])
        self.assertEqual(entry.document.name, 'document.pdf')

    def test_upload_failure_logged_without_file_name(self):
        b = self.browser()
        r = b.upload(b'junk', HTTP_X_FILE_NAME=T.SECRET_NAME)
        self.assertEqual((r.status_code, r.json()), (400, {'error': 'PDF를 읽지 못했습니다.'}))
        upload = Event.objects.get(event='pdf_upload')
        self.assertEqual(upload.result, 'error')
        self.assertEqual(json.loads(upload.detail)['status'], 'error')
        self.assertNotIn(T.SECRET_NAME, all_audit_text())
        self.assertIsNone(self.rt.registry.get(b.key).document)
        self.assertFalse(b.c.session.get('had_document'))

    def test_unexpected_parse_error_is_500_with_class_name_only(self):
        b = self.browser()
        err = io.StringIO()
        with mock.patch('onprem_web.workspace.services.parse_pdf', side_effect=RuntimeError(T.SECRET_NAME)), \
                contextlib.redirect_stderr(err):
            r = b.upload()
        self.assertEqual((r.status_code, r.json()), (500, {'error': '처리 중 오류가 발생했습니다.'}))
        self.assertEqual(err.getvalue().strip(), 'RuntimeError')
        self.assertEqual(Event.objects.get(event='pdf_upload').result, 'error')

    def test_upload_rejections_leave_no_files(self):
        b = self.browser()
        tmp = tempfile.gettempdir()
        before = (sorted(os.listdir(tmp)), sorted(os.listdir(S.DATA_DIR)))
        big = b'%PDF' + b'x' * (self.rt.config.pdf.max_bytes - 3)
        # The test client omits Content-Length for an empty body; a real server sends 0.
        r1, r2, r3 = b.upload(big), b.upload(b'', CONTENT_LENGTH='0'), b.upload(b'not a pdf')
        after = (sorted(os.listdir(tmp)), sorted(os.listdir(S.DATA_DIR)))
        self.assertEqual((r1.status_code, r1.json()), (413, {'error': '입력이 너무 큽니다.'}))
        self.assertEqual([r2.status_code, r3.status_code], [400, 400])
        self.assertEqual(before, after)
        self.assertIsNone(self.rt.registry.get(b.key).document)

    def test_load_too_big_by_header_is_413(self):
        b = self.browser()
        r = b.upload(PDF, CONTENT_LENGTH=str(self.rt.config.pdf.max_bytes + 1))
        self.assertEqual((r.status_code, r.json()), (413, {'error': '입력이 너무 큽니다.'}))

    def test_content_length_required_and_checked(self):
        b = self.browser()
        r = b.upload(PDF, CONTENT_LENGTH='')
        self.assertEqual((r.status_code, r.json()), (411, {'error': '요청 길이가 필요합니다.'}))
        r = b.upload(PDF, CONTENT_LENGTH='-1')
        self.assertEqual((r.status_code, r.json()), (400, {'error': '요청 길이가 올바르지 않습니다.'}))
        r = b.upload(PDF, CONTENT_LENGTH='abc')
        self.assertEqual((r.status_code, r.json()), (400, {'error': '요청 길이가 올바르지 않습니다.'}))
        r = b.upload(PDF, HTTP_TRANSFER_ENCODING='chunked')
        self.assertEqual((r.status_code, r.json()), (400, {'error': '허용되지 않은 요청입니다.'}))
        r = b.c.post('/api/clear', '{}', content_type='application/json', secure=True,
                     HTTP_X_TRID_TOKEN=b.token, CONTENT_LENGTH='')
        self.assertEqual(r.status_code, 411)

    def test_json_endpoints_limited_to_small_body(self):
        b, document_id = self.uploaded()
        big = {'question': 'q', 'document_id': document_id, 'pad': 'x' * SMALL_BODY}
        self.assertEqual((b.post('/api/ask', big).status_code), 413)
        self.assertEqual(b.post('/api/clear', {'pad': 'x' * SMALL_BODY}).status_code, 413)
        self.assertEqual(self.calls, [])
        self.assertIsNotNone(self.rt.registry.get(b.key).document)

    def test_get_is_404(self):
        b = self.browser()
        for path in ('/api/load', '/api/ask', '/api/clear'):
            self.assertEqual(b.get(path).status_code, 404, path)

    def test_load_requires_login_and_csrf(self):
        anon = WorkspaceBrowser()
        self.assertEqual(anon.upload().status_code, 401)
        b = self.browser()
        r = b.c.post('/api/load', PDF, content_type='application/pdf', secure=True)
        self.assertEqual(r.status_code, 403)
        self.assertIsNone(self.rt.registry.get(b.key).document)

    def test_must_change_blocks_load(self):
        accounts.create_user('carol', PW, 'user', must_change=True)
        b = self.browser('carol')
        self.assertEqual(b.upload().status_code, 403)
        self.assertFalse(Event.objects.filter(event='pdf_upload').exists())

    # -- ask -----------------------------------------------------------------
    def test_ask_logs_only_status_and_seconds(self):
        b, document_id = self.uploaded()
        r = b.ask(T.SECRET_QUESTION, document_id)
        self.assertEqual(r.status_code, 200)
        value = r.json()
        self.assertEqual(value['status'], 'quoted')
        self.assertNotIn('checked', value)
        self.assertNotIn('raw_output', value)
        self.assertEqual(self.calls, [T.SECRET_QUESTION])
        ask = Event.objects.get(event='ask')
        self.assertEqual(ask.result, 'ok')
        detail = json.loads(ask.detail)
        self.assertEqual(set(detail), {'status', 'seconds'})
        self.assertEqual(detail['status'], 'quoted')
        self.assertEqual(b.post('/api/clear').status_code, 200)
        for secret in (T.SECRET_QUESTION, T.SECRET_NAME, '본문 한 쪽', 'raw'):
            self.assertNotIn(secret, all_audit_text())
            self.assertNotIn(secret, all_session_text())

    def test_odd_status_is_logged_as_unknown(self):
        type(self).answer = staticmethod(lambda d, q: {'status': 'bad status!'})
        self.addCleanup(setattr, type(self), 'answer', staticmethod(T.ok_answer))
        b, document_id = self.uploaded()
        self.assertEqual(b.ask('q', document_id).status_code, 200)
        self.assertEqual(details('ask')[0]['status'], 'unknown')

    def test_model_down_is_502_korean(self):
        b, document_id = self.uploaded()
        outcomes = [ConnectionRefusedError(T.SECRET_QUESTION), None]

        def answer(document, question):
            outcome = outcomes.pop(0)
            if outcome:
                raise outcome
            return {'status': 'not_found'}
        self.rt.answer = answer
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            r = b.ask(T.SECRET_QUESTION, document_id)
            self.assertEqual((r.status_code, r.json()), (502, {'error': T.DOWN}))
            self.assertEqual(b.ask(T.SECRET_QUESTION, document_id).status_code, 200)
        self.assertNotIn('qzx', err.getvalue())
        self.assertFalse(self.rt.gen_lock.locked())
        self.assertIn('model_unavailable', [d.get('error') for d in details('ask')])
        self.assertNotIn(T.SECRET_QUESTION, all_audit_text())

    def test_context_overflow_is_422(self):
        b, document_id = self.uploaded()
        failures = iter([DocumentError(CONTEXT_OVERFLOW), DocumentError('질문을 입력해 주세요.'),
                         RuntimeError(T.SECRET_QUESTION)])

        def answer(document, question):
            raise next(failures)
        self.rt.answer = answer
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            r1, r2, r3 = (b.ask('q', document_id) for _ in range(3))
        self.assertEqual((r1.status_code, r1.json()), (422, {'error': CONTEXT_OVERFLOW}))
        self.assertEqual((r2.status_code, r2.json()), (400, {'error': '질문을 입력해 주세요.'}))
        self.assertEqual((r3.status_code, r3.json()), (500, {'error': '처리 중 오류가 발생했습니다.'}))
        self.assertEqual(err.getvalue().strip(), 'RuntimeError')
        kinds = [d.get('error') for d in details('ask')]
        self.assertEqual(sorted(k for k in kinds if k), ['bad_request', 'context_overflow'])
        self.assertFalse(self.rt.gen_lock.locked())

    def test_document_id_mismatch_is_409(self):
        b, document_id = self.uploaded()
        r = b.ask('q', document_id + 'x')
        self.assertEqual((r.status_code, r.json()), (409, {'error': FIRST}))
        other = self.browser('bob')
        r = other.ask('q', document_id)
        self.assertEqual((r.status_code, r.json()), (409, {'error': FIRST}))
        self.assertEqual(b.ask('q', document_id).status_code, 200)
        self.assertEqual(self.calls, ['q'])

    def test_question_over_2000_is_400(self):
        b, document_id = self.uploaded()
        message = {'error': '질문은 2,000자 이하로 입력해 주세요.'}
        r = b.ask('가' * 2001, document_id)
        self.assertEqual((r.status_code, r.json()), (400, message))
        self.assertEqual(b.ask('가' * 2000, document_id).status_code, 200)
        for body in ({'question': 1, 'document_id': document_id}, {'question': 'q'}, [], 'x'):
            r = b.post('/api/ask', body)
            self.assertEqual((r.status_code, r.json()), (400, message), body)
        r = b.c.post('/api/ask', b'{not json', content_type='application/json', secure=True,
                     HTTP_X_TRID_TOKEN=b.token)
        self.assertEqual((r.status_code, r.json()), (400, message))

    def test_busy_generation_lock_is_503(self):
        b, document_id = self.uploaded()
        self.rt.gen_lock.acquire()
        try:
            started = time.monotonic()
            r = b.ask('q', document_id)
            waited = time.monotonic() - started
        finally:
            self.rt.gen_lock.release()
        self.assertEqual((r.status_code, r.json()), (503, {'error': BUSY_ASK}))
        self.assertGreaterEqual(waited, 0.1)
        self.assertEqual(self.calls, [])
        self.assertEqual(details('ask')[0], {'status': 'error', 'error': 'busy',
                                             'seconds': details('ask')[0]['seconds']})

    def test_busy_parse_lock_is_503(self):
        from trid.onprem.server import BUSY_PARSE
        from trid.pdf_isolated import PdfLimits
        rt = S.make_runtime(answer=T.ok_answer, lock_wait=0.1, pdf=PdfLimits(wall_seconds=0.0))
        b = self.browser()
        rt.parse_lock.acquire()
        try:
            r = b.upload()
        finally:
            rt.parse_lock.release()
        self.assertEqual((r.status_code, r.json()), (503, {'error': BUSY_PARSE}))
        self.assertIsNone(rt.registry.get(b.key).document)

    def test_after_restart_question_asks_for_reupload(self):
        b, document_id = self.uploaded()
        self.rt.registry = SessionRegistry(self.rt.config.max_sessions, self.rt.config.idle_seconds,
                                           self.rt.config.per_minute)
        self.assertEqual(b.get('/').status_code, 200)
        r = b.ask('q', document_id)
        self.assertEqual((r.status_code, r.json()), (409, {'error': REUPLOAD}))
        # A browser that never uploaded keeps the old message.
        r = self.browser('bob').ask('q', document_id)
        self.assertEqual((r.status_code, r.json()), (409, {'error': FIRST}))

    def test_clear_drops_document_and_logs(self):
        b, document_id = self.uploaded()
        r = b.post('/api/clear')
        self.assertEqual((r.status_code, r.json()), (200, {'cleared': True}))
        entry = self.rt.registry.get(b.key)
        self.assertIsNone(entry.document)
        self.assertIsNone(entry.document_id)
        self.assertEqual(Event.objects.get(event='clear').result, 'ok')
        r = b.ask('q', document_id)
        self.assertEqual((r.status_code, r.json()), (409, {'error': REUPLOAD}))

    def test_revoked_during_generation_returns_401_and_drops_document(self):
        b, document_id = self.uploaded()
        key = b.key
        entry = self.rt.registry.get(key)
        admin = self.admin

        def revoke(document, question):
            accounts.admin_update(admin, 'alice', disabled=True, ip=None)
            return {'status': 'quoted', 'answer': 'secret-answer', 'candidates': []}
        self.rt.answer = revoke
        r = b.ask('q', document_id)
        self.assertEqual((r.status_code, r.json()), (401, {'error': '로그인이 필요합니다.'}))
        self.assertNotIn('answer', r.json())
        self.assertNotIn('secret-answer', r.content.decode())
        self.assertIsNone(self.rt.registry.get(key))
        self.assertIsNone(entry.document)
        self.assertEqual(r.cookies['__Host-trid'].value, '')
        self.assertEqual(details('ask')[-1]['error'], 'session_revoked')

    def test_password_reset_during_generation_returns_401(self):
        b, document_id = self.uploaded()
        key = b.key

        def reset(document, question):
            accounts.admin_update(self.admin, 'alice', reset_password='a brand new pw 22', ip=None)
            return {'status': 'quoted', 'answer': 'secret-answer', 'candidates': []}
        self.rt.answer = reset
        r = b.ask('q', document_id)
        self.assertEqual(r.status_code, 401)
        self.assertIsNone(self.rt.registry.get(key))

    def test_registry_drop_during_generation_returns_401(self):
        b, document_id = self.uploaded()
        key = b.key

        def drop(document, question):
            self.rt.registry.drop(key)
            return {'status': 'quoted', 'answer': 'secret-answer', 'candidates': []}
        self.rt.answer = drop
        r = b.ask('q', document_id)
        self.assertEqual(r.status_code, 401)
        self.assertNotIn('secret-answer', r.content.decode())

    def test_load_after_concurrent_drop_does_not_attach(self):
        b = self.browser()
        key = b.key
        entry = self.rt.registry.get(key)

        def parse(body, name, tok, limits=None):
            self.rt.registry.drop(key)
            return T.fake_doc(name)
        with mock.patch('onprem_web.workspace.services.parse_pdf', side_effect=parse):
            r = b.upload()
        self.assertEqual((r.status_code, r.json()), (401, {'error': '로그인이 필요합니다.'}))
        self.assertIsNone(entry.document)
        self.assertIsNone(self.rt.registry.get(key))

    def test_second_question_gets_busy_and_upload_still_works(self):
        a, document_id = self.uploaded()
        entry = self.rt.registry.get(a.key)
        bob, bob_document = self.uploaded('bob')
        started, release = threading.Event(), threading.Event()

        def blocking(document, question):
            started.set()
            release.wait(5)
            return {'status': 'quoted', 'answer': 'a'}
        self.rt.answer = blocking
        real_log = services.log
        off_thread = []

        def log(*args, **kwargs):  # thread A has no view of this test's open transaction
            if threading.current_thread() is threading.main_thread():
                real_log(*args, **kwargs)
            else:
                off_thread.append((args, kwargs))
        results = {}
        with mock.patch('onprem_web.workspace.services.log', side_effect=log):
            thread = threading.Thread(target=lambda: results.setdefault('a', services.ask(
                entry, 'qa', document_id, 'alice', '10.0.0.5', lambda: True, True)))
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                r = bob.ask('qb', bob_document)
                self.assertEqual((r.status_code, r.json()), (503, {'error': BUSY_ASK}))
                started_upload = time.monotonic()
                r = bob.upload()
                self.assertEqual(r.status_code, 200)
                self.assertLess(time.monotonic() - started_upload, 1.0)
                self.assertTrue(thread.is_alive())
            finally:
                release.set()
                thread.join(5)
        self.assertEqual(results['a'].status, 200)
        self.assertEqual(results['a'].data, {'status': 'quoted', 'answer': 'a'})
        self.assertEqual(off_thread[0][0][:4], ('ask', 'alice', '10.0.0.5', 'ok'))
        self.assertFalse(self.rt.gen_lock.locked())


class ServiceUnitTests(TestCase):
    def setUp(self):
        self.rt = S.make_runtime(answer=T.ok_answer, lock_wait=0.1)

    def test_result_and_clear(self):
        from onprem_web.accounts.registry import Entry
        entry = Entry(username='alice', touched=0.0, document=object(), document_id='x')
        result = services.clear(entry, 'alice', '10.0.0.5')
        self.assertIsInstance(result, services.Result)
        self.assertEqual((result.status, result.data), (200, {'cleared': True}))
        self.assertIsNone(entry.document)
        self.assertIsNone(entry.document_id)

    def test_still_valid_false_returns_401(self):
        from onprem_web.accounts.registry import Entry
        entry = Entry(username='alice', touched=0.0, document=T.fake_doc(), document_id='x')
        result = services.ask(entry, 'q', 'x', 'alice', None, lambda: False, True)
        self.assertEqual((result.status, result.data), (401, {'error': '로그인이 필요합니다.'}))


if __name__ == '__main__':
    unittest.main()
