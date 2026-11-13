import json
import os
import unittest
from io import BytesIO
from unittest import mock

from pypdf import PdfWriter

from trid import pdf_isolated
from trid.pdf_documents import DocumentError
from trid.pdf_isolated import DEMO_LIMITS, PdfLimits


class FakeTok:
    def encode(self, text):
        raise AssertionError('tokenizer must not be used')


def blank_pdf(pages):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


class PdfIsolatedTest(unittest.TestCase):
    def test_demo_limits_unchanged(self):
        self.assertEqual(DEMO_LIMITS, PdfLimits(2 * 1024 * 1024, 20, 50_000, 15.0, 10, 1024 ** 3))

    def test_parse_pdf_passes_limits_to_worker(self):
        limits = PdfLimits(111, 7, 22_000, 3.0, 4, 555)
        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        process.communicate.return_value = (json.dumps({'pages': [], 'warnings': []}).encode(), None)
        with mock.patch.object(pdf_isolated.subprocess, 'Popen', return_value=process) as popen, \
                mock.patch.object(pdf_isolated, 'make_chunks', return_value=[]):
            pdf_isolated.parse_pdf(b'x', 'a.pdf', FakeTok(), limits=limits)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[1:], ['-I', '-m', 'trid.pdf_extract_worker', '111', '7', '22000', '4', '555'])

    @unittest.skipUnless(os.name == 'posix', 'rlimit-based page limit is POSIX only')
    def test_parse_pdf_page_limit_message_uses_limits(self):
        limits = PdfLimits(max_pages=2)
        with self.assertRaises(DocumentError) as caught:
            pdf_isolated.parse_pdf(blank_pdf(3), 'a.pdf', FakeTok(), limits=limits)
        self.assertIn('2페이지', str(caught.exception))

    def test_parse_pdf_kills_worker_after_wall_timeout(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.returncode = -9
        process.communicate.return_value = (b'', None)
        with mock.patch.object(pdf_isolated.subprocess, 'Popen', return_value=process):
            with self.assertRaises(DocumentError) as caught:
                pdf_isolated.parse_pdf(b'x', 'a.pdf', FakeTok(), limits=PdfLimits(wall_seconds=0.2))
        self.assertIn('시간이 초과', str(caught.exception))
        process.kill.assert_called()

    def test_worker_runs_for_real_on_any_platform(self):
        with self.assertRaises(DocumentError) as caught:
            pdf_isolated.parse_pdf(b'not a pdf', 'a.pdf', FakeTok())
        self.assertIn('2MB·20페이지·5만 자', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
