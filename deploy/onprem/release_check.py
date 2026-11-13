#!/usr/bin/env python3
"""Pre-release check of an installed on-prem service (standard library only).

Logs in as an admin, uploads a text PDF, asks the fixed questions in order and writes a
report.json with per-question status, quote count, page numbers, seconds and whether a
commentary was returned. Answers and questions are NOT written to the report: a person reads
the answers in the browser and judges them (there is no expected-value comparison).

    python3 release_check.py --base https://trid.example.internal --user admin \\
        --pdf release-document.pdf --questions release-questions.json --out report.json

The password is asked with getpass; it is never a command-line option.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import re
import ssl
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

TIMEOUT = 600
MUST_CHANGE_MESSAGE = '이 계정은 비밀번호 변경이 필요합니다. 먼저 웹에서 비밀번호를 변경하세요.'
_META = r'<meta name="%s" content="([^"]*)"'


class CheckError(Exception):
    pass


class Client:
    def __init__(self, base: str, insecure: bool, cafile: str | None):
        parts = urlsplit(base)
        if parts.scheme not in ('https', 'http') or not parts.hostname:
            raise CheckError('--base는 https://호스트 형식이어야 합니다.')
        self.origin = f'{parts.scheme}://{parts.netloc}'
        self.scheme, self.host, self.port = parts.scheme, parts.hostname, parts.port
        self.context = None
        if self.scheme == 'https':
            if insecure:
                self.context = ssl.create_default_context()
                self.context.check_hostname = False
                self.context.verify_mode = ssl.CERT_NONE
            else:
                self.context = ssl.create_default_context(cafile=cafile)
        self.cookie = ''
        self.token = ''

    def request(self, method, path, body=None, headers=None):
        """Returns (status, headers, bytes). Connection errors raise CheckError (no secrets in text)."""
        if self.scheme == 'https':
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=TIMEOUT, context=self.context)
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=TIMEOUT)
        sent = {'Origin': self.origin, **(headers or {})}
        if self.cookie:
            sent['Cookie'] = self.cookie
        if self.token:
            sent['X-Trid-Token'] = self.token
        try:
            conn.request(method, path, body=body, headers=sent)
            response = conn.getresponse()
            return response.status, response, response.read()
        except (OSError, http.client.HTTPException) as e:
            raise CheckError(f'서버에 연결할 수 없습니다. ({type(e).__name__})') from None
        finally:
            conn.close()

    def post_json(self, path, value):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        status, _, data = self.request('POST', path, body, {'Content-Type': 'application/json'})
        return status, _json(data)


def _json(data: bytes):
    try:
        value = json.loads(data)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def login(client: Client, user: str, password: str) -> str:
    body = json.dumps({'username': user, 'password': password}, ensure_ascii=False).encode('utf-8')
    status, response, data = client.request('POST', '/api/login', body, {'Content-Type': 'application/json'})
    value = _json(data)
    if status != 200:
        raise CheckError(f'로그인에 실패했습니다. (HTTP {status}) {value.get("error", "")}'.strip())
    for header in response.msg.get_all('Set-Cookie') or []:
        pair = header.split(';', 1)[0].strip()
        if pair:
            client.cookie = pair
    if not client.cookie:
        raise CheckError('로그인 응답에 세션 쿠키가 없습니다.')
    return value.get('status', 'ok')


def fetch_token(client: Client) -> bool:
    """Reads the CSRF token and the must-change flag from the page after login."""
    status, _, data = client.request('GET', '/')
    page = data.decode('utf-8', 'replace')
    token = re.search(_META % 'csrf-token', page)
    if status != 200 or not token:
        raise CheckError('로그인 뒤 페이지를 읽지 못했습니다.')
    client.token = token.group(1)
    must = re.search(_META % 'must-change', page)
    return bool(must and must.group(1) == '1')


def run_questions(client: Client, document_id: str, questions: list[dict]) -> list[dict]:
    rows = []
    for item in questions:
        started = time.monotonic()
        row = {'id': item['id'], 'kind': item.get('kind', '')}
        try:
            status, value = client.post_json('/api/ask', {'question': item['question'], 'document_id': document_id})
        except CheckError:
            status, value = 0, {}
        seconds = round(time.monotonic() - started, 2)
        if status == 200:
            candidates = [c for c in value.get('candidates') or []
                          if (c.get('prediction') or {}).get('source_character_span', [0, 0])[1]
                          > (c.get('prediction') or {}).get('source_character_span', [0, 0])[0]]
            row.update(status=str(value.get('status', 'unknown')), quotes=len(candidates),
                       pages=sorted({c['page'] for c in candidates if isinstance(c.get('page'), int)}),
                       seconds=float(seconds), has_commentary=bool(value.get('commentary')))
        else:
            row.update(status=f'http_{status}' if status else 'connection_error', quotes=0, pages=[],
                       seconds=float(seconds), has_commentary=False)
        rows.append(row)
    return rows


def run(args, password: str) -> int:
    questions = json.loads(Path(args.questions).read_text(encoding='utf-8'))
    pdf = Path(args.pdf).read_bytes()
    client = Client(args.base, args.insecure, args.cafile)
    status = login(client, args.user, password)
    try:
        must_change = fetch_token(client) or status == 'must_change'
        if must_change:
            raise CheckError(MUST_CHANGE_MESSAGE)
        started = time.monotonic()
        code, _, data = client.request('POST', '/api/load', pdf, {
            'Content-Type': 'application/pdf', 'X-File-Name': quote(Path(args.pdf).name)})
        value = _json(data)
        if code != 200:
            raise CheckError(f'PDF를 올리지 못했습니다. (HTTP {code}) {value.get("error", "")}'.strip())
        upload = {'status': 'ok', 'pages': len(value.get('pages') or []),
                  'seconds': round(time.monotonic() - started, 2), 'warnings': len(value.get('warnings') or [])}
        rows = run_questions(client, value['document_id'], questions)
    finally:
        try:
            client.post_json('/api/logout', {})
        except CheckError:
            print('경고: 로그아웃하지 못했습니다. 세션은 30분 뒤 만료됩니다.', file=sys.stderr)
    seconds = [r['seconds'] for r in rows]
    report = {
        'base': client.origin, 'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'upload': upload,
        'questions': rows,
        'summary': {'count': len(rows), 'median_seconds': round(statistics.median(seconds), 2) if seconds else None,
                    'max_seconds': max(seconds) if seconds else None,
                    'errors': sum(1 for r in rows if not r['status'] or r['status'] == 'model_failure'
                                      or r['status'].startswith(('http_', 'connection'))),
                    'status_counts': {s: [r['status'] for r in rows].count(s) for s in sorted({r['status'] for r in rows})}},
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(rows)}개 질문 완료. 처리 시간 중앙값 {report["summary"]["median_seconds"]}초, '
          f'최대 {report["summary"]["max_seconds"]}초. 보고서: {args.out}')
    print('답변 내용은 보고서에 없습니다. 웹에서 같은 문서와 질문으로 사람이 직접 확인하세요.')
    return 1 if report['summary']['errors'] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='설치본 배포 전 점검')
    p.add_argument('--base', required=True, help='https://<서버 호스트명>')
    p.add_argument('--user', required=True, help='관리자(또는 사용자) 계정 이름')
    p.add_argument('--pdf', required=True, type=Path, help='텍스트 PDF 파일')
    p.add_argument('--questions', required=True, type=Path)
    p.add_argument('--out', default='report.json', type=Path)
    p.add_argument('--insecure', action='store_true', help='자체 서명 인증서를 검증 없이 받아들임')
    p.add_argument('--cafile', help='서버 인증서(또는 CA) PEM 파일. 자체 서명이면 cert.pem')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        password = getpass.getpass('비밀번호: ')
        return run(args, password)
    except CheckError as e:
        print(str(e), file=sys.stderr)
    except (OSError, ValueError, KeyError) as e:
        print(f'오류: {type(e).__name__}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
