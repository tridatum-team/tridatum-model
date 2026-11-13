"""The upload limit is a setting: server -> page meta -> app.js, and Caddy/compose body limit."""
import unittest
from pathlib import Path

import test_onprem_server as base
from trid.pdf_isolated import PdfLimits

ROOT = Path(__file__).resolve().parent.parent


class LimitPageTests(base.ServerTests):
    config_overrides = {'pdf': PdfLimits(max_bytes=5 * 1024 * 1024)}

    def test_page_carries_configured_max_bytes(self):
        c = self.client()
        _, data = c.request('GET', '/')
        self.assertIn('<meta name="trid-max-bytes" content="5242880">', data.decode())


class StaticTests(unittest.TestCase):
    def test_app_js_reads_limit_from_meta(self):
        js = (ROOT / 'trid' / 'onprem_ui' / 'app.js').read_text(encoding='utf-8')
        self.assertIn("meta('trid-max-bytes')", js)
        self.assertNotIn('MAX_UPLOAD = 2 * 1024 * 1024;', js)
        self.assertNotIn('2MB 이하', js)

    def test_index_html_has_placeholder(self):
        html = (ROOT / 'trid' / 'onprem_ui' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('MAX_BYTES_PLACEHOLDER', html)

    def test_caddy_and_compose_carry_trid_max_body(self):
        caddy = (ROOT / 'deploy' / 'onprem' / 'Caddyfile').read_text(encoding='utf-8')
        compose = (ROOT / 'deploy' / 'onprem' / 'compose.yaml').read_text(encoding='utf-8')
        self.assertIn('max_size {$TRID_MAX_BODY:3MB}', caddy)
        self.assertIn('TRID_MAX_BODY: ${TRID_MAX_BODY:-3MB}', compose)
        self.assertIn('TRID_MAX_BODY', (ROOT / 'docs' / 'ONPREM.md').read_text(encoding='utf-8'))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(LimitPageTests('test_page_carries_configured_max_bytes'))
    suite.addTests(loader.loadTestsFromTestCase(StaticTests))
    return suite


if __name__ == '__main__':
    unittest.main()
