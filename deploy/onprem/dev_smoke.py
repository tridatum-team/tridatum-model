"""Local HTTP smoke test of `python -m trid.onprem serve --insecure-dev` (no browser, no secrets).

Everything lives in a temp directory: a throwaway config, a tiny WordLevel tokenizer, a
hand-built one-page PDF. No llama-server runs, so /api/ask is expected to answer 502.

    python deploy/onprem/dev_smoke.py
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PASSWORD = 'smoke test pw 1'
SECRET_NAME = 'smoke-secret-file-name.pdf'
SECRET_QUESTION = 'smoke secret question zqx?'
MODEL_DOWN = '모델 서버에 연결할 수 없습니다'


def make_pdf(text: str) -> bytes:
    stream = f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode()
    objs = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R '
        b'/Resources << /Font << /F1 5 0 R >> >> >>',
        b'<< /Length %d >>\nstream\n' % len(stream) + stream + b'\nendstream',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    ]
    out = b'%PDF-1.4\n'
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b'%d 0 obj\n' % i + body + b'\nendobj\n'
    xref = len(out)
    out += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objs) + 1)
    for off in offsets:
        out += b'%010d 00000 n \n' % off
    out += b'trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objs) + 1, xref)
    return out


def make_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {'[UNK]': 0}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Client:
    def __init__(self, port):
        self.port = port
        self.sid = None
        self.csrf = None
        self.origin = f'http://127.0.0.1:{port}'

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        h = dict(headers or {})
        if method == 'POST':
            h['Origin'] = self.origin
        if self.sid:
            h['Cookie'] = f'trid-dev={self.sid}'
        if self.csrf:
            h['X-Trid-Token'] = self.csrf
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode()
            h['Content-Type'] = 'application/json'
        if method == 'POST':
            h['Content-Length'] = str(len(body or b''))
        conn.request(method, path, body=body, headers=h)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp, data


def check(label, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + label + (f'  [{detail}]' if detail else ''))
    if not ok:
        raise SystemExit(1)


def jload(data):
    return json.loads(data.decode())


def main() -> int:
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        port = free_port()
        make_tokenizer(tmp / 'tokenizer.json')
        config = tmp / 'onprem.json'
        config.write_text(json.dumps({
            'hostname': 'smoke.local', 'port': port, 'data_dir': str(tmp / 'data'),
            'tokenizer_path': str(tmp / 'tokenizer.json'), 'llm_base': 'http://127.0.0.1:1',
        }), encoding='utf-8')
        base = [sys.executable, '-X', 'utf8', '-m', 'trid.onprem']
        created = subprocess.run(
            base + ['create-admin', '--config', str(config), '--username', 'smokeadmin', '--password-stdin'],
            input=PASSWORD + '\n', text=True, capture_output=True, env=env, cwd=ROOT)
        check('create-admin', created.returncode == 0, created.stderr.strip())
        stderr_path = tmp / 'serve.stderr'
        with stderr_path.open('wb') as err:
            server = subprocess.Popen(base + ['serve', '--config', str(config), '--insecure-dev'],
                                      stdout=subprocess.DEVNULL, stderr=err, env=env, cwd=ROOT)
        try:
            for _ in range(100):
                try:
                    socket.create_connection(('127.0.0.1', port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.2)
            c = Client(port)
            r, d = c.call('GET', '/')
            check('GET / shows login page', r.status == 200 and 'login.js' in d.decode())
            r, d = c.call('POST', '/api/login', {'username': 'smokeadmin', 'password': PASSWORD})
            check('login', r.status == 200 and jload(d)['status'] == 'ok', f'{r.status} {d[:80]}')
            c.sid = r.getheader('Set-Cookie').split(';')[0].split('=', 1)[1]
            r, d = c.call('GET', '/')
            page = d.decode()
            marker = 'name="csrf-token" content="'
            check('GET / shows staff page', r.status == 200 and marker in page and 'app.js' in page)
            c.csrf = page.split(marker)[1].split('"')[0]
            for path, mime in (('/app.js', 'javascript'), ('/common.css', 'css'), ('/admin.js', 'javascript')):
                r, d = c.call('GET', path)
                check(f'GET {path}', r.status == 200 and mime in r.getheader('Content-Type') and d)
            text = ' '.join(['The retention period for usage records is three hundred sixty five days.'] * 3)
            r, d = c.call('POST', '/api/load', make_pdf(text), {'X-File-Name': SECRET_NAME})
            value = jload(d)
            check('POST /api/load', r.status == 200 and value['pages'], f'{r.status} pages={len(value.get("pages", []))}')
            r, d = c.call('POST', '/api/ask', {'question': SECRET_QUESTION, 'document_id': value['document_id']})
            check('POST /api/ask -> 502 model down', r.status == 502 and MODEL_DOWN in jload(d)['error'],
                  f'{r.status} {jload(d)}')
            r, d = c.call('GET', '/admin')
            check('GET /admin', r.status == 200 and 'admin.js' in d.decode())
            r, d = c.call('POST', '/api/admin/events', {})
            events = jload(d)
            check('POST /api/admin/events', r.status == 200, f'{r.status} {str(events)[:120]}')
            r, d = c.call('GET', f'/api/admin/events.csv?token={c.csrf}')
            csv_text = d.decode('utf-8-sig')
            check('GET /api/admin/events.csv', r.status == 200 and csv_text.startswith('ts,username'),
                  f'{len(csv_text.splitlines())} lines')
            for secret in (SECRET_NAME, SECRET_QUESTION, 'retention period'):
                check(f'event log free of {secret[:12]!r}...', secret not in csv_text)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        stderr_text = stderr_path.read_text(encoding='utf-8', errors='replace')
        print('--- serve stderr ---')
        print(stderr_text.strip() or '(empty)')
        print('--- end stderr ---')
        for secret in (SECRET_NAME, SECRET_QUESTION, 'retention period'):
            check(f'stderr free of {secret[:12]!r}...', secret not in stderr_text)
    print('SMOKE OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
