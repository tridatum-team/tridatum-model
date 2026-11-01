"""ALIO reserved acquisition: every revision file must be listed, typed by content, and groups interleaved."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.collect_institutions as ci
from scripts.collect_institutions import BudgetExceeded, Metrics, NetworkError  # the classes the collector catches
from scripts.collect_institutions import files_of, kind_of, round_robin


class AlioRulesTests(unittest.TestCase):
    def test_bfiles_split_like_the_site_and_keeps_commas_in_names(self):
        self.assertEqual(files_of('69727|취업 규칙(2018년도 10월 18일 개정).hwp,78319|규정 1,2호 서식.pdf'),
                         [('69727', '취업 규칙(2018년도 10월 18일 개정).hwp'), ('78319', '규정 1,2호 서식.pdf')])
        self.assertEqual(files_of(None), [])

    def test_kind_comes_from_bytes_not_only_the_name(self):
        self.assertEqual(kind_of(b'%PDF-1.7', 'x.hwp'), 'pdf')
        self.assertEqual(kind_of(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest', 'x.hwp'), 'hwp')
        self.assertEqual(kind_of(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest', 'x.doc'), 'ole:doc')
        self.assertEqual(kind_of(b'PK\x03\x04', 'x.hwpx'), 'hwpx')
        self.assertEqual(kind_of(b'<html>error', 'x.hwp'), 'unknown:hwp')
        self.assertEqual(kind_of(b'PK\x03\x04', 'x.hwp'), 'hwpx')
        self.assertEqual(kind_of(b'PK\x03\x04', 'x.zip'), 'zip:zip')

    def test_round_robin_covers_every_group_before_repeating(self):
        fams = [{'business_group': g, 'i': i} for g, n in (('A', 3), ('B', 1), ('C', 2)) for i in range(n)]
        self.assertEqual([f['business_group'] for f in round_robin(fams)], ['A', 'B', 'C', 'A', 'C', 'A'])


class FakeResponse:
    def __init__(self, status, body=b'', ctype='application/json'):
        self.status_code, self.content, self.headers = status, body, {'Content-Type': ctype}

    def json(self):
        return json.loads(self.content)


class FakeClient:
    """Answers by URL; records every call. A detail answer can be a status, a body, or an exception to raise."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []
        self.limiter = mock.Mock(interval=1.0)

    def request(self, method, url, metrics, max_bytes=None, params=None):
        key = url if not params else f"{url}?seq={params['seq']}"
        self.calls.append(key)
        answer = self.answers[key]
        if isinstance(answer, Exception):
            metrics.record(0, 0, 0, 599)
            raise answer
        if max_bytes is not None and len(answer.content) > max_bytes:
            metrics.record(0, 0, max_bytes, answer.status_code)
            raise BudgetExceeded(url)
        metrics.record(0, 0, len(answer.content), answer.status_code)
        return answer


DETAIL = ci.BASE + '/occasional/findRuleDtl.json?seq={}'


def detail(no):
    return FakeResponse(200, json.dumps({'status': 'success', 'data': {
        'pname': '기관', 'title': '규정', 'bFiles': f'{no}|규정.hwp'}}).encode())


class LocalModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(ci, 'REPORTS', self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        rows = [{'doc_family_id': f'institutions:alio-rule:{s}', 'business_group': 'g'} for s in (1, 2, 3, 4)]
        self.reservation = self.dir / 'res-x.jsonl'
        self.reservation.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.elsewhere = self.dir / 'elsewhere.jsonl'
        self.elsewhere.write_text(json.dumps({'doc_family_id': 'institutions:alio-rule:1'}) + '\n')

    def store(self, max_disk=10 ** 6):
        return ci.LocalStore(self.dir / 'store', max_disk)

    def run_local(self, answers, max_bytes=10 ** 6, store=None, failures=3):
        client, store = FakeClient(answers), store or self.store()
        summary = ci.reserved_local(client, store, Metrics(), self.reservation, self.elsewhere, max_bytes, failures)
        return client, store, summary

    def test_keys_cannot_leave_the_store(self):
        store = self.store()
        for key in ('../x', '/etc/passwd', 'a/../../b', 'a//b', '.hidden', 'a b'):
            with self.assertRaises(ValueError, msg=key):
                store.put(key, b'x')

    def test_raw_is_never_replaced_and_writes_leave_no_temp_files(self):
        store = self.store()
        self.assertEqual(store.raw('r/1/a.hwp', b'one')[1], 'r/1/a.hwp')
        self.assertEqual(store.raw('r/1/a.hwp', b'one')[1], 'r/1/a.hwp')  # same bytes: kept as is
        digest, key = store.raw('r/1/a.hwp', b'two')
        self.assertEqual(key, f'r/1/a.{digest[:16]}.hwp')
        self.assertEqual(store.read('r/1/a.hwp'), b'one')
        self.assertEqual([p.name for p in (store.root / 'r/1').iterdir() if p.name.startswith('.')], [])

    def test_disk_budget_refuses_before_writing(self):
        store = self.store(max_disk=5)
        with self.assertRaises(ci.Stop):
            store.raw('r/big.bin', b'123456')
        self.assertFalse((store.root / 'r/big.bin').exists())

    def test_refusal_stops_after_checkpoint_and_keeps_the_family_pending(self):
        answers = {DETAIL.format(2): detail(10), ci.FILE_URL.format(10): FakeResponse(200, b'body', 'x'),
                   DETAIL.format(3): FakeResponse(403, b'no')}
        client, store, summary = self.run_local(answers)
        self.assertNotIn(DETAIL.format(1), client.calls)  # acquired elsewhere: never requested
        self.assertNotIn(DETAIL.format(4), client.calls)  # nothing after the stop
        self.assertEqual(summary['run']['stop'], 'http_403 on detail 3')
        self.assertEqual(sorted(p.stem for p in (store.root / 'corpus/reserved/res-x/state').glob('*.json')), ['2'])
        saved = json.loads((self.dir / 'res-x-local-acquisition-summary.json').read_text())
        self.assertEqual(saved['budget']['requests'], 3)

    def test_consecutive_network_failures_stop_the_run(self):
        answers = {DETAIL.format(s): NetworkError('down') for s in (2, 3, 4)}
        client, _, summary = self.run_local(answers, failures=2)
        self.assertEqual(client.calls, [DETAIL.format(2), DETAIL.format(3)])
        self.assertTrue(summary['run']['stop'].startswith('2 consecutive network failures'))

    def test_byte_budget_holds_across_runs(self):
        body = b'x' * 70000
        answers = {DETAIL.format(s): detail(10 + s) for s in (2, 3, 4)}
        answers.update({ci.FILE_URL.format(10 + s): FakeResponse(200, body, 'x') for s in (2, 3, 4)})
        store = self.store()
        _, _, first = self.run_local(answers, max_bytes=200000, store=store)  # room for two families, not three
        self.assertLessEqual(first['budget']['response_bytes'], 200000)
        self.assertTrue(first['run']['stop'].startswith('response byte budget'))
        client, _, second = self.run_local(answers, max_bytes=200000, store=ci.LocalStore(store.root, 10 ** 6))
        self.assertEqual(client.calls, [])  # the spent bytes were kept: the next run sends nothing
        self.assertEqual(second['budget']['response_bytes'], first['budget']['response_bytes'])

    def test_client_cuts_the_body_at_the_budget_and_never_retries(self):
        import datetime

        class Body:
            def __init__(self, status, size, fail=False):
                self.status_code, self.size, self.fail = status, size, fail
                self.headers, self.elapsed = {}, datetime.timedelta(seconds=0.01)

            def iter_content(self, n):
                for start in range(0, self.size, n):
                    yield b'x' * min(n, self.size - start)

            def close(self):
                pass

        class Session:
            def __init__(self, answers):
                self.answers, self.calls = list(answers), 0

            def request(self, method, url, timeout, stream=False, **kw):
                self.calls += 1
                answer = self.answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer

        def client(answers):
            c = ci.Client(ci.HostLimiter(0.0), retry=False)
            c.session = Session(answers)
            return c

        metrics = Metrics()
        c = client([Body(200, 500000)])
        with self.assertRaises(BudgetExceeded):
            c.request('GET', 'u', metrics, max_bytes=100000)
        self.assertLess(metrics.snapshot()['bytes'], 100000 + 65536)
        c = client([Body(429, 10), Body(200, 10)])
        self.assertEqual(c.request('GET', 'u', metrics, max_bytes=1000).status_code, 429)
        self.assertEqual(c.session.calls, 1)
        c = client([ConnectionError('reset'), Body(200, 10)])
        with self.assertRaises(NetworkError):
            c.request('GET', 'u', metrics, max_bytes=1000)
        self.assertEqual(c.session.calls, 1)

    def test_local_mode_never_imports_the_aws_sdk(self):
        code = ('import sys, json, tempfile, pathlib; sys.argv=["x"]\n'
                'import scripts.collect_institutions as ci\n'
                'd = pathlib.Path(tempfile.mkdtemp()); ci.REPORTS = d\n'
                's = ci.LocalStore(d / "s", 10**6); s.put("a/b.json", b"{}"); s.raw("a/c.bin", b"x")\n'
                'print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in ("boto3", "botocore"))))')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1]), [])


if __name__ == '__main__':
    unittest.main()
