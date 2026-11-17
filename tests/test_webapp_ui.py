"""Staff and login screens (templates + htmx): ports the contracts of test_onprem_ui.py to the Django build."""

import re
import unittest
from pathlib import Path
from urllib.parse import urlencode
from unittest import mock

import webapp_support as S  # noqa: F401  (boots Django and migrates)
import test_onprem_server as T
from test_webapp_workspace import PDF, WorkspaceBrowser, fake_parse

from django.test import TestCase

from onprem_web import web
from onprem_web.accounts import services as accounts
from onprem_web.audit.models import Event
from onprem_web.workspace import present
from trid.onprem.server import model_card_html
from trid.pdf_documents import Chunk, Document

PW = S.PW
NEW_PW = 'another horse 22'
WEB = Path(__file__).resolve().parent.parent / 'onprem_web'
OLD_UI = Path(__file__).resolve().parent.parent / 'trid' / 'onprem_ui'
FILES = sorted(p for d in (WEB / 'templates', WEB / 'static') for p in d.rglob('*')
               if p.is_file() and 'vendor' not in p.relative_to(d).parts)
NOTICE = '질문마다 완전한 문장으로 써 주세요. 이전 질문은 이어서 해석하지 않습니다.'
TERMS = ('모델', '상태', '구성', '기반 모델', '라이선스', '성능 측정', '한계')
EVIL_PAGE = '<script>alert(1)</script> & <b>x</b>'


def old_status_table() -> dict:
    """STATUS as written in the old app.js."""
    text = (OLD_UI / 'app.js').read_text(encoding='utf-8')
    body = re.search(r'const STATUS = \{(.*?)\};', text, re.S).group(1)
    return dict(re.findall(r"(\w+): '([^']*)'", body))


class UiFileTests(unittest.TestCase):
    def test_templates_have_no_external_urls_inline_scripts_or_innerhtml(self):
        names = {p.relative_to(WEB).as_posix() for p in FILES}
        for name in ('templates/base.html', 'templates/login.html', 'templates/workspace.html',
                     'templates/partials/_document.html', 'templates/partials/_answer.html',
                     'templates/partials/_password.html', 'static/app.js', 'static/login.js',
                     'static/common.css'):
            self.assertIn(name, names)
        for path in FILES:
            text = path.read_text(encoding='utf-8')
            for needle in ('http://', 'https://', '//fonts', '@import url(', 'X-File-Name'):
                self.assertNotIn(needle, text, f'{path.name}: {needle}')
            if path.suffix == '.html':
                for tag in re.findall(r'<script\b[^>]*>', text):
                    self.assertIn('src=', tag, path.name)
                self.assertNotRegex(text, r'\son[a-z]+\s*=', path.name)
                self.assertNotIn('hx-on', text, path.name)
                # Only the fixed, pre-escaped model card may skip autoescaping.
                self.assertEqual(text.count('|safe'), text.count('model_card|safe'), path.name)
                self.assertNotIn('autoescape off', text, path.name)
            if path.suffix == '.js':
                self.assertNotRegex(text, r'innerHTML\s*=', path.name)
                self.assertNotIn('outerHTML', text, path.name)
                self.assertNotIn('insertAdjacentHTML', text, path.name)
                self.assertNotIn('document.write', text, path.name)
                self.assertLess(len(text.splitlines()), 600, path.name)

    def test_new_scripts_are_served_from_the_allow_list(self):
        for name in ('app.js', 'login.js', 'common.css'):
            self.assertIn(name, web.STATIC)


class PresentTests(unittest.TestCase):
    def test_status_table_is_copied_from_old_app_js(self):
        self.assertEqual(present.STATUS, old_status_table())

    def test_present_answer_statuses(self):
        for key, text in present.STATUS.items():
            p = present.present_answer({'status': key, 'answer': 'a'})
            self.assertEqual(p['tag'], text)
            self.assertEqual(p['tag_bad'], key == 'model_failure', key)
            self.assertEqual(p['answer'], 'a')
        self.assertEqual(present.present_answer({'status': 'odd'})['tag'], 'odd')
        self.assertEqual(present.present_answer({})['tag'], '근거 검색')
        self.assertEqual(present.present_answer({})['answer'], '(빈 응답)')

    def test_present_answer_quote_slices_by_code_point(self):
        text = '😀가나다'
        p = present.present_answer({'status': 'quoted', 'answer': 'a', 'candidates': [
            {'text': text, 'page': 2, 'prediction': {'source_character_span': [1, 3]}},
            {'text': 'skip', 'prediction': {'source_character_span': [2, 2]}},
            {'text': 'none'},
        ]})
        self.assertEqual(len(p['quotes']), 1)
        q = p['quotes'][0]
        self.assertEqual((q['before'], q['mark'], q['after']), ('😀', '가나', '다'))
        self.assertEqual(q['place'], '2쪽')
        self.assertEqual(q['context'], '원문 일치 · 정답 검증 아님 · 2쪽')

    def test_present_answer_notes_match_old_render(self):
        p = present.present_answer({
            'status': 'review', 'answer': 'a',
            'context_limits': {'table_rows': 3, 'tables': 1, 'linked_passages': 0, 'own_passages': 2},
            'source_limits': ['그림 제외'], 'table_warning': '표 경고',
            'commentary_notes': ['일부 해설을 숨겼습니다.', '참고 메모'],
            'commentary': '해설', 'scope': '범위 A', 'context_notes': ['범위 A', '다른 메모'],
            'warnings': ['경고'], 'checked_chunks': 4, 'total_chunks': 9, 'model': 'm', 'elapsed_ms': 1234,
            'candidates': [{'text': 'abc', 'source_label': '별표 1', 'prediction': {'source_character_span': [0, 1]}}],
        })
        self.assertEqual(p['limits'], [
            '일부 근거만 반영 (표 행 3행 · 검토된 표 행 전부 · 관련도 낮은 구간 2개 제외)',
            '자료 한계: 원본 그림 제외 추출본 · 표 행·열 대응 미검증 — 자세한 내용은 아래에서 확인하세요'])
        self.assertEqual(p['warn_notes'], ['일부 해설을 숨겼습니다.'])
        self.assertEqual(p['info'], ['참고 메모', '범위 A', '다른 메모', '경고', '4/9개 구간 확인', 'm · 1.2초'])
        self.assertEqual((p['commentary'], p['table_warning']), ('해설', '표 경고'))
        self.assertEqual(p['quotes'][0]['place'], '별표 1')
        only_notes = present.present_answer({'context_notes': ['x'], 'source_limits': ['스캔'],
                                             'checked_chunks': 1})
        self.assertEqual(only_notes['limits'], [
            '일부 근거만 반영 (문맥 한도로 일부 제외)',
            '자료 한계: 원본 추출 한계 있음 — 자세한 내용은 아래에서 확인하세요'])
        self.assertEqual(only_notes['info'], ['x', '1/?개 구간 확인'])


def rich_answer(document, question):
    return {'status': 'review', 'answer': 'a', 'checked': 1, 'raw_output': 'raw',
            'context_limits': {'table_rows': 3}, 'source_limits': ['그림 제외'], 'table_warning': '표 경고',
            'commentary_notes': ['일부 해설을 숨겼습니다.', '참고 메모'], 'commentary': '해설',
            'scope': '범위', 'checked_chunks': 1, 'total_chunks': 2, 'model': 'm',
            'candidates': [{'text': '앞 <img src=x> 뒤', 'page': 1,
                            'prediction': {'source_character_span': [2, 13]}}]}


class UiBrowser(WorkspaceBrowser):
    def form(self, path, data, token=True):
        """POST as htmx does: urlencoded form fields, X-Trid-Token from <body hx-headers>."""
        extra = {'HTTP_X_TRID_TOKEN': self.token, 'HTTP_HX_REQUEST': 'true'} if token else {}
        if not data:
            extra['CONTENT_LENGTH'] = '0'  # the test client omits it for an empty body; a browser sends 0
        return self.c.post(path, urlencode(data), content_type='application/x-www-form-urlencoded',
                           secure=True, **extra)

    def ui_get(self, path):
        return self.c.get(path, secure=True, HTTP_HX_REQUEST='true')


class PageTests(TestCase):
    answer = staticmethod(rich_answer)

    @classmethod
    def setUpTestData(cls):
        accounts.create_user('alice', PW, 'user', must_change=False)
        accounts.create_user('carol', PW, 'user', must_change=True)

    def setUp(self):
        self.rt = S.make_runtime(answer=lambda d, q: type(self).answer(d, q), lock_wait=0.1)
        patcher = mock.patch('onprem_web.workspace.services.parse_pdf', side_effect=fake_parse)
        self.parse = patcher.start()
        self.addCleanup(patcher.stop)

    def browser(self, name='alice'):
        b = UiBrowser()
        self.assertEqual(b.login(name).status_code, 200)
        return b

    def assets(self, b, page):
        found = re.findall(r'<script[^>]*\ssrc="([^"]+)"', page) + re.findall(r'<link[^>]*href="([^"]+)"', page)
        self.assertTrue(found)
        for src in found:
            r = b.c.get(src, secure=True)
            self.assertEqual(r.status_code, 200, src)
            self.assertTrue(r.content, src)
        return found

    def test_login_page_and_scripts(self):
        b = UiBrowser()
        r = b.get('/')
        page = r.content.decode()
        self.assertEqual(r.status_code, 200)
        self.assertIn('<form id="login"', page)
        self.assertIn(T.STATUS_TEXT, page)
        self.assertNotIn('csrf-token', page)
        self.assertIn('/static/login.js', self.assets(b, page))
        self.assertIn('/static/vendor/htmx.min.js', self.assets(b, page))

    def test_staff_page_matches_old_contract(self):
        b = self.browser()
        r = b.get('/')
        page = r.content.decode()
        self.assertEqual(r.status_code, 200)
        self.assertIn(NOTICE, page)
        self.assertIn('maxlength="2000"', page)
        self.assertIn('<meta name="username" content="alice">', page)
        self.assertIn('<meta name="role" content="user">', page)
        self.assertIn(f'<meta name="trid-max-bytes" content="{self.rt.config.pdf.max_bytes}">', page)
        token = re.search(r'<meta name="csrf-token" content="([A-Za-z0-9]{32,})">', page).group(1)
        self.assertIn(f'<body hx-headers=\'{{"X-Trid-Token": "{token}"}}\'>', page)
        self.assertIn('<meta name="htmx-config" content=\'{"allowEval":false,"allowScriptTags":false,'
                      '"includeIndicatorStyles":false,"selfRequestsOnly":true}\'>', page)
        self.assertIn(T.STATUS_TEXT, page.split('<details', 1)[0])
        card = page.split('<div class="model-card">', 1)[1].split('</div></dl>', 1)[0]
        for term in TERMS:
            self.assertIn(f'<dt>{term}</dt>', card)
        self.assertIn(f'<strong>{T.STATUS_TEXT}</strong>', card)
        self.assertIn('<li>이미지·스캔 PDF는 읽지 못합니다.</li>', card)
        self.assertIn(model_card_html(), page)
        for element_id in ('who', 'admin-link', 'logout', 'model-status', 'model-card', 'password-view',
                           'password-form', 'current', 'new-password', 'password-message', 'work-view',
                           'file', 'load', 'clear', 'doc-status', 'doc-view', 'doc-summary', 'doc-pages',
                           'question', 'count', 'ask', 'ask-status', 'result', 'document_id'):
            self.assertIn(f'id="{element_id}"', page)
        self.assertIn('class="workspace"', page)
        self.assertIn('class="card plain doc-panel"', page)
        self.assertIn('class="status-strip"', page)
        self.assertRegex(page, r'<section id="password-view"[^>]*\shidden')
        self.assertNotIn('must-change', page)
        self.assertIn('/static/app.js', self.assets(b, page))
        self.assertIn('/static/common.css', self.assets(b, page))

    def test_must_change_page_shows_password_view(self):
        b = self.browser('carol')
        page = b.get('/').content.decode()
        self.assertIn('<meta name="must-change" content="1">', page)
        self.assertNotRegex(page, r'<section id="password-view"[^>]*\shidden')
        self.assertRegex(page, r'<div id="work-view"[^>]*\shidden')

    def test_document_and_answer_escape_markup(self):
        evil = Document('x.pdf', [EVIL_PAGE], [Chunk(1, 0, len(EVIL_PAGE), EVIL_PAGE)], [])
        self.parse.side_effect = None
        self.parse.return_value = evil
        b = self.browser()
        r = b.upload()
        self.assertEqual(r.status_code, 200)
        document_id = r.json()['document_id']
        doc = b.ui_get('/ui/document')
        self.assertEqual(doc.status_code, 200)
        answer = b.form('/ui/ask', {'question': '질문', 'document_id': document_id})
        self.assertEqual(answer.status_code, 200)
        for fragment in (doc.content.decode(), answer.content.decode()):
            self.assertIn('&lt;', fragment)
        doc_text, answer_text = doc.content.decode(), answer.content.decode()
        self.assertIn('&lt;script&gt;', doc_text)
        self.assertIn('&amp;', doc_text)
        self.assertNotIn('<script>alert', doc_text)
        self.assertNotIn('<b>x', doc_text)
        self.assertIn('&lt;img', answer_text)
        self.assertNotIn('<img src=x>', answer_text)
        self.assertNotIn('<script>alert', answer_text)

    def test_document_fragment(self):
        b = self.browser()
        r = b.ui_get('/ui/document')
        self.assertEqual((r.status_code, r.content.strip()), (200, b''))
        b.upload()
        page = b.ui_get('/ui/document').content.decode()
        self.assertIn('<summary id="doc-summary">추출 문서 보기 (1쪽)</summary>', page)
        self.assertIn('<div id="doc-pages"><div class="block"><span class="label">1쪽</span>본문 한 쪽</div>'
                      '</div>', page)
        self.assertNotIn('document.pdf', page)  # the browser shows its own file name

    def test_answer_fragment_sections_in_old_order(self):
        b, document_id = self.uploaded()
        r = b.form('/ui/ask', {'question': '  질문  ', 'document_id': document_id})
        self.assertEqual(r.status_code, 200)
        page = r.content.decode()
        order = ['<div class="answer">', '<div class="text">a</div>',
                 '<span class="tag">검증 보류 · 원문 확인 필요</span>',
                 '<div class="note warn">일부 근거만 반영 (표 행 3행 제외)</div>',
                 '<div class="note warn">자료 한계: 원본 그림 제외 추출본 · 표 행·열 대응 미검증',
                 '<div class="note warn">일부 해설을 숨겼습니다.</div>',
                 '<span class="label">확인된 인용 · 1쪽</span><mark>&lt;img src=x&gt;</mark></div>',
                 '<summary>인용 앞뒤 원문 보기 (1개)</summary>',
                 '<span class="label">원문 일치 · 정답 검증 아님 · 1쪽</span>앞 <mark>&lt;img src=x&gt;</mark> 뒤</div>',
                 '<span class="label">검증되지 않은 해설 — 원문과 대조하세요</span>해설</div>',
                 '<span class="label">자료 한계 · 표</span>표 경고</div>',
                 '<div class="note">참고 메모\n범위\n1/2개 구간 확인\nm</div>']
        positions = [page.find(s) for s in order]
        self.assertNotIn(-1, positions, [s for s, p in zip(order, positions) if p == -1])
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('raw', page)

    def uploaded(self):
        b = self.browser()
        r = b.upload()
        self.assertEqual(r.status_code, 200)
        return b, r.json()['document_id']

    def test_ui_ask_without_token_is_403(self):
        b, document_id = self.uploaded()
        r = b.form('/ui/ask', {'question': '질문', 'document_id': document_id}, token=False)
        self.assertEqual(r.status_code, 403)
        self.assertFalse(Event.objects.filter(event='ask').exists())

    def test_ui_errors_are_json_with_old_messages(self):
        b, document_id = self.uploaded()
        r = b.form('/ui/ask', {'question': '질문', 'document_id': 'wrong'})
        self.assertEqual((r.status_code, r.json()), (409, {'error': '이 브라우저에서 문서를 먼저 넣어 주세요.'}))
        r = b.form('/ui/ask', {'question': 'x' * 2001, 'document_id': document_id})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(b.ui_get('/ui/ask').status_code, 404)
        multipart = b.c.post('/ui/ask', {'question': '질문', 'document_id': document_id}, secure=True,
                             HTTP_X_TRID_TOKEN=b.token)
        self.assertEqual(multipart.status_code, 400)  # only urlencoded fields, as htmx sends them
        logged_out = UiBrowser()
        self.assertEqual(logged_out.ui_get('/ui/document').status_code, 401)

    def test_ui_clear_drops_document_and_first_message_returns(self):
        b, document_id = self.uploaded()
        r = b.form('/ui/clear', {})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(self.rt.registry.get(b.key).document)
        self.assertNotIn('had_document', b.c.session)
        self.assertEqual(b.ui_get('/ui/document').content.strip(), b'')
        r = b.form('/ui/ask', {'question': '질문', 'document_id': document_id})
        self.assertEqual((r.status_code, r.json()), (409, {'error': '이 브라우저에서 문서를 먼저 넣어 주세요.'}))

    def test_ui_password_error_and_success(self):
        b = self.browser('carol')
        self.assertEqual(b.ui_get('/ui/document').status_code, 403)
        r = b.form('/ui/password', {'current': 'wrong password', 'new': NEW_PW})
        self.assertEqual(r.status_code, 200)
        page = r.content.decode()
        self.assertIn('id="password-view"', page)
        self.assertRegex(page, r'<p id="password-message" class="msg error" role="alert">[^<]+</p>')
        r = b.form('/ui/password', {'current': PW, 'new': NEW_PW})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r['HX-Redirect'], '/')
        self.assertNotIn('must-change', b.get('/').content.decode())


if __name__ == '__main__':
    unittest.main()
