import re
import tempfile
import threading
import unittest
from pathlib import Path

import test_onprem_server as T
from trid.onprem.auth import Auth
from trid.onprem.config import Config
from trid.onprem.server import make_server
from trid.onprem.store import Store

UI = Path(__file__).resolve().parent.parent / 'trid' / 'onprem_ui'
FILES = sorted(p for p in UI.iterdir() if p.is_file())


class UiFileTests(unittest.TestCase):
    def test_ui_has_no_external_urls(self):
        self.assertTrue(FILES)
        for path in FILES:
            text = path.read_text(encoding='utf-8')
            for needle in ('http://', 'https://', '//fonts', '@import url('):
                self.assertNotIn(needle, text, f'{path.name}: {needle}')

    def test_ui_has_no_inline_script_or_innerhtml_from_server(self):
        for path in FILES:
            text = path.read_text(encoding='utf-8')
            if path.suffix == '.html':
                for tag in re.findall(r'<script\b[^>]*>', text):
                    self.assertIn('src=', tag, path.name)
                self.assertNotRegex(text, r'\son[a-z]+\s*=', path.name)
            if path.suffix == '.js':
                self.assertNotRegex(text, r'innerHTML\s*=', path.name)
                self.assertNotIn('insertAdjacentHTML', text)
                self.assertNotIn('document.write', text)

    def test_index_contains_independent_question_notice(self):
        text = (UI / 'index.html').read_text(encoding='utf-8')
        self.assertIn('질문마다 완전한 문장으로 써 주세요. 이전 질문은 이어서 해석하지 않습니다.', text)
        self.assertIn('MODEL_CARD_PLACEHOLDER', text)
        self.assertIn('maxlength="2000"', text)

    def test_js_files_stay_focused(self):
        for path in FILES:
            if path.suffix == '.js':
                self.assertLess(len(path.read_text(encoding='utf-8').splitlines()), 600, path.name)


class ServedPagesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / 'app.db')
        self.addCleanup(store.close)
        store.create_user('alice', T.PW, 'user', must_change=False)
        store.create_user('root', T.PW, 'admin', must_change=False)
        self.server = make_server(Config(hostname=T.HOST, port=0), store, Auth(store), object())
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def assets(self, client, page):
        found = re.findall(r'<script[^>]*\ssrc="([^"]+)"', page) + re.findall(r'<link[^>]*href="([^"]+)"', page)
        self.assertTrue(found)
        for src in found:
            resp, data = client.request('GET', src)
            self.assertEqual(resp.status, 200, src)
            self.assertTrue(data, src)
        return found

    def test_login_page_and_scripts(self):
        c = T.Client(self.server)
        resp, data = c.request('GET', '/')
        self.assertEqual(resp.status, 200)
        self.assertIn('/login.js', self.assets(c, data.decode()))

    def test_staff_page_and_scripts(self):
        c = T.Client(self.server)
        c.login('alice')
        resp, data = c.request('GET', '/')
        page = data.decode()
        self.assertEqual(resp.status, 200)
        self.assertIn('질문마다 완전한 문장으로', page)
        self.assertIn('<meta name="username" content="alice">', page)
        self.assertIn('/app.js', self.assets(c, page))

    def test_admin_page_and_scripts(self):
        c = T.Client(self.server)
        c.login('root')
        resp, data = c.request('GET', '/admin')
        self.assertEqual(resp.status, 200)
        self.assertIn('/admin.js', self.assets(c, data.decode()))


if __name__ == '__main__':
    unittest.main()
