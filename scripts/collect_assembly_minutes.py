"""National Assembly minutes (국회회의록, record.assembly.go.kr) -> corpus contract clean-v1 (docs/CORPUS_CONTRACT.md).

  discover  walk the public meeting tree of each 대수 and meeting class (본회의, 상임위원회, 특별위원회,
            예산결산특별위원회, 국정감사, 국정조사) through the site's own list endpoints (mnts/async/*.do, the same
            POSTs its list.js issues). The index and the finished (대수, class) pairs are saved to
            corpus/state/assembly-minutes/<index> every few list requests, so an interrupted walk resumes.
  fetch     fetch each indexed meeting's text viewer (viewer/minutes/xml.do?id=N&type=view) with a small worker pool
            that shares ONE per-host request-start limiter; keep the original HTML gzip-compressed under
            corpus/raw/assembly-minutes/; extract text, redact PII (trid.korean_quality.redact) and write clean-v1
            shards with S3-verified SHA-256 to corpus/clean/assembly-minutes/clean-v1/. Each shard has a manifest
            under corpus/state/assembly-minutes/clean-v1/ listing its meeting ids, hashes and timing metrics;
            re-running skips those ids.
  aggregate verify every shard checksum, then write stats.json, provenance.json, ATTRIBUTION.txt and the _DONE marker.

Licence: the minutes site footer carries 공공누리 제1유형 (출처표시); the Assembly copyright policy
(https://open.assembly.go.kr/portal/policy/copyRightPage.do) declares Assembly-authored works free to use under
저작권법 제24조의2. robots.txt is absent (404) and no request policy is published; absence is NOT read as permission
for unlimited speed. Default request-start interval is 1 s for the whole host, slowed automatically on 429,
Retry-After, 5xx or rising latency. Never spread requests over other IPs or accounts.
"""
import argparse
import base64
import datetime
import gzip
import contextlib
import hashlib
import html
import json
import os
import re
import statistics
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKET = 'trid-training-148761639313'
SOURCE = 'assembly-minutes'
VERSION = 'clean-v5'  # v1 (B-006): old redact; v2: audit header gap; v3 (B-007): no URL-damage/listed quarantine;
# v4 (B-008 fetch of the remaining 1,142): joint-meeting headings (<h1> without 회의록) not found -> viewer UI kept
BASE = 'https://record.assembly.go.kr/assembly'
UA = 'Mozilla/5.0 (compatible; trid-collector/0.1; +public-sector research corpus)'
LICENSE = 'KOGL-Type1'
CLASSES = {'1': 'assembly_plenary', '2': 'assembly_committee', '3': 'assembly_special_committee',
           '4': 'assembly_budget_committee', '5': 'assembly_audit', '6': 'assembly_investigation'}
TERMS = [str(n) for n in range(1, 23)] + ['10.5']
MINUTE_LINK = re.compile(r'viewer/minutes/xml\.do\?id=(\d+)&(?:amp;)?type=view')
HEADER = re.compile(r'제(\d+(?:\.\d+)?)대국회\s*제(\d+)회\s*\(([^)]*)\)\s*제(\d+)차\s*(.+?)\s*\((\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\)')
AUDIT_HEADER = re.compile(r'제(\d+(?:\.\d+)?)대국회\s*(\d{4})년도\s*(국정감사|국정조사)\s*(.+?)\s*\((\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\)')
CEREMONY_HEADER = re.compile(r'제(\d+(?:\.\d+)?)대국회\s*제(\d+)회\s*\(([^)]*)\)\s*(개회식|폐회식|개원식)\s*(.+?)\s*\((\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\)')
UI_TAIL = {'top', 'down', '발언 저장', '발언 인쇄'}
SPLITS = (('test', 50), ('valid', 100))  # per-10,000 bucket bounds; dev splits, not an independent final test
# Private persons named in minutes (B-006 review): the representative petitioner of a 국민동의청원 ("홍길동 외 52,585인")
# and witness / reference / statement speakers ("증인 홍길동"). Whole meetings are quarantined, not edited: replacing names
# would break the record, and being public is not by itself a basis for training use. Public officials stay.
PETITIONER = re.compile(r'\(([가-힣]{2,4}) 외 [\d,]+인')
PRIVATE_SPEAKER = re.compile(r'^(?:증인|참고인|진술인)\s+[가-힣]{2,4}$', re.M)
# Title/date restored from the official meeting list when the minutes header has another form (e.g. 개회식); kept in
# S3 with the list request that showed it. Nothing is inferred from the body.
OFFICIAL_META_KEY = f'corpus/state/{SOURCE}/official-list-meta.json'
# Meetings held back by an outside check, e.g. partial text overlap with benchmark-reserved documents:
# {"<id>": {"reason": "reserved_overlap_partial", "evidence": ...}}
OVERLAP_QUARANTINE_KEY = f'corpus/state/{SOURCE}/listed-quarantine.json'


def payload(kind, d, term):
    """(endpoint, form) for a list node, mirroring resources/mnts/total/list.js."""
    g = lambda k: d.get(k, '')
    if kind in ('sess', 'sess2', 'mem'):
        return '/mnts/async/sess.do', {'th_sch': g('th'), 'class_id_sch': g('class'), 'sess_sch': g('sess'),
                                      'chk': g('chk1'), 'cmit_chk_cmit': g('chk2'), 'cmit_chk_sub': g('chk3'),
                                      'mnts_id_sch': g('id'), 'mnts_year_sch': g('year')}
    if kind == 'cmit':
        url = '/mnts/async/sessCmit2.do' if (g('th') == '5' and g('year') and g('class') in ('2', '4')) else '/mnts/async/sessCmit.do'
        return url, {'council_cd_sch': g('cl'), 'th_sch': g('th'), 'sess_sch': g('sess'), 'class_id_sch': g('class'),
                     'cmit_id_sch': g('cmit'), 'cmit_chk': g('chk1'), 'cmit_chk_cmit': g('chk2'),
                     'cmit_chk_sub': g('chk3'), 'cmit_chk_etc': g('chk4'), 'mnts_year_sch': g('year')}
    if kind == 'cmit2':
        return '/mnts/async/sessCmit2.do', {'th_sch': g('th'), 'class_id_sch': g('class'), 'sess_sch': g('sess'),
                                           'cmit_cd_sch': g('cmit'), 'chk': g('chk')}
    if kind == 'sub':
        return '/mnts/async/cmit.do', {'council_cd_sch': g('cl'), 'th_sch': g('th'), 'class_id_sch': g('class'),
                                      'sess_sch': g('sess'), 'cmit_cd_sch': g('cmitcd'), 'cmit_chk': g('chk1'),
                                      'cmit_chk_cmit': g('chk2'), 'cmit_chk_sub': g('chk3'), 'cmit_chk_etc': g('chk4'),
                                      'mnts_id_sch': g('id'), 'conf_year': g('dt')}
    if kind == 'ord_num':
        return '/mnts/async/ord.do', {'council_cd_sch': g('cl'), 'mnts_id_sch': g('id'), 'class_id_sch': g('class'),
                                     'th_sch': g('th')}
    raise ValueError(kind)


class NodeParser(HTMLParser):
    """Collects expandable list nodes (class tit/btn_tit + loader class) with their data-* attributes."""
    KINDS = ('sess', 'sess2', 'mem', 'cmit', 'cmit2', 'sub', 'ord_num')

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get('class') or '').split())
        if not classes & {'tit', 'btn_tit'}:
            return
        for kind in self.KINDS:
            if kind in classes:
                data = {k[5:]: (v or '') for k, v in attrs.items() if k.startswith('data-')}
                self.nodes.append((kind, data))
                return


def nodes_of(page):
    parser = NodeParser()
    parser.feed(page)
    return parser.nodes


def minute_ids(page):
    return sorted({int(i) for i in MINUTE_LINK.findall(page)})


def heading_starts(page):
    """Start offsets of the minutes title <h1>: its text contains 회의록, or it is directly followed (after an optional
    subcommittee subtitle <p class="tit_sm">) by the issue number <p class="num">제N호, as in joint meetings
    ("<h1>과학기술정보방송통신<div>…연석회의</div>위원회</h1>")."""
    starts = []
    after = re.compile(r'\s*(?:</div>\s*|<p class="tit_sm">[^<]*</p>\s*)*<p class="num">')
    for m in re.finditer(r'<h1>(.*?)</h1>', page, re.S):
        title = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        if title and ('회의록' in title or after.match(page, m.end(), m.end() + 300)):
            starts.append(m.start())
    return starts


def extract(page):
    """Viewer HTML -> (header metadata dict, minutes text). Text starts at the document title heading."""
    starts = heading_starts(page)
    head_text = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', page[:starts[-1] if starts else 0])))
    body = page[starts[-1]:] if starts else page
    body = re.sub(r'(?is)<(script|style|noscript|button)\b.*?</\1>', ' ', body)
    body = re.sub(r'(?i)<br\s*/?>|</(p|div|li|tr|h\d|dd|dt|table)>', '\n', body)
    lines = [' '.join(line.split()) for line in html.unescape(re.sub(r'<[^>]+>', ' ', body)).splitlines()]
    lines = [line for line in lines if line]
    while lines and (lines[-1] in UI_TAIL or lines[-1].startswith('Scroll :')):
        lines.pop()
    # The viewer repeats each speaker as an interactive "<name> 선택" control right before the speaker line.
    lines = [line for i, line in enumerate(lines)
             if not (line.endswith(' 선택') and i + 1 < len(lines) and lines[i + 1].startswith(line[:-3]))]
    meta = {}
    m = HEADER.search(head_text)
    a = AUDIT_HEADER.search(head_text)
    if m:
        term, session, session_kind, order, meeting, y, mo, d = m.groups()
        meta = {'term': term, 'session': int(session), 'session_kind': session_kind, 'order': int(order),
                'meeting': meeting, 'date': f'{int(y):04d}-{int(mo):02d}-{int(d):02d}',
                'title': f'제{term}대국회 제{session}회 제{order}차 {meeting}'}
    elif a:  # 국정감사·국정조사 minutes: "제22대국회 2024년도 국정감사 문화체육관광위원회 (2024.10.24.)"
        term, year, kind, meeting, y, mo, d = a.groups()
        meta = {'term': term, 'audit_year': int(year), 'session_kind': kind, 'meeting': meeting,
                'date': f'{int(y):04d}-{int(mo):02d}-{int(d):02d}', 'title': f'제{term}대국회 {year}년도 {kind} {meeting}'}
    elif c := CEREMONY_HEADER.search(head_text):  # "제22대국회 제429회 (정기회) 개회식 국회본회의 (2025.09.01.)"
        term, session, session_kind, ceremony, meeting, y, mo, d = c.groups()
        meta = {'term': term, 'session': int(session), 'session_kind': session_kind, 'ceremony': ceremony,
                'meeting': meeting, 'date': f'{int(y):04d}-{int(mo):02d}-{int(d):02d}',
                'title': f'제{term}대국회 제{session}회 {ceremony} {meeting}'}
    return meta, unicodedata.normalize('NFC', '\n'.join(lines))


def split_of(group):
    bucket = int(hashlib.sha256(f'split:{group}'.encode()).hexdigest()[:8], 16) % 10000
    return next((name for name, upper in SPLITS if bucket < upper), 'train')


def quarantine_reasons(text, meta, official, listed=None):
    from trid.korean_quality import redacted_in_url
    reasons = []
    if PETITIONER.search(text) or PRIVATE_SPEAKER.search(text):
        reasons.append('private_person_names')
    if not meta and not official:
        reasons.append('metadata_incomplete')
    if redacted_in_url(text):  # privacy handled by the replacement; the URL text itself is damaged
        reasons.append('pii_redaction_inside_url')
    if listed:  # e.g. partial overlap with benchmark-reserved texts (OVERLAP_QUARANTINE_KEY)
        reasons.append(listed)
    return reasons


def row_of(meeting, page, collected_at, official=None, listed=None):
    """-> (row, pii Counter). `official`: {'title', 'date', 'source'} from the official meeting list, used only when the
    minutes header does not parse. Quarantine reasons are flags 'quarantine:<reason>'."""
    from trid.korean_quality import redact
    meta, text = extract(page)
    text, pii = redact(text)
    title = meta.get('title') or (official or {}).get('title', '')
    date = meta.get('date') or (official or {}).get('date')
    url = f"{BASE}/viewer/minutes/xml.do?id={meeting['id']}&type=view"
    group = f"{SOURCE}:{meeting['id']}"
    flags = (['pii_redacted'] if pii else []) + ([] if meta else ['header_unparsed'])
    flags += [f'quarantine:{r}' for r in quarantine_reasons(text, meta, official, listed)]
    extra = {'metadata_source': official['source']} if official and not meta else {}
    return {'id': group, 'group': group, 'kind': CLASSES.get(str(meeting.get('class')), 'assembly_minutes'),
            'title': title, 'text': text, 'source': url, 'license': LICENSE,
            'attribution': f'출처: 국회사무처 국회회의록, {url}', 'date': date,
            'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': collected_at, 'extractor': 'html:viewer',
            'clean_version': VERSION, 'flags': flags, 'split': split_of(group),
            'provenance': {'institution': '국회', 'institution_type': '국회', 'doc_type': '회의록',
                           'committee': meta.get('meeting') or (official or {}).get('meeting'), 'published_date': date},
            'meta': {**{k: v for k, v in meta.items() if k not in ('date', 'title')}, **extra, 'minutes_id': meeting['id'],
                     'class_id': meeting.get('class'), 'list_term': meeting.get('th'), 'pii_redactions': dict(pii)}}, pii


class HostLimiter:
    """One request at a time per host, and each request starts at least `interval` after the previous request to that
    host has finished (response read or failed). Since a request's bytes go out after its start and before its end,
    consecutive actual sends are always more than `interval` apart, however late a process wakes up or stalls before
    sending (B-008 review: the old design spaced *reserved* start times; a late wake-up let the logged gap fall to
    0.9906 s). The wait is checked against the wall clock after every sleep, not computed once.

    Slows down on 429/5xx, Retry-After or time-to-first-byte above max(2x running median, median + 1 s); drifts back
    toward the base interval when healthy. Callers pass time-to-first-byte, not total transfer time.

    With `host`, the last end time and the not-before time live in a lock file (<tmp>/trid-host-limits/<host>) that is
    held, with an exclusive flock, for the whole request: every process and thread on this machine that talks to that
    host shares one gap and one request in flight. The OS drops the lock if a process dies."""

    def __init__(self, interval, max_interval=30.0, host=None, shared_dir=None):
        import tempfile
        self.base = self.interval = interval
        self.max_interval = max_interval
        self.lock, self.latencies = threading.Lock(), []
        self.last = {'end': 0.0, 'not_before': 0.0}  # without `host`: this process only
        self.shared = None
        if host:
            folder = Path(shared_dir or tempfile.gettempdir()) / 'trid-host-limits'
            folder.mkdir(parents=True, exist_ok=True)
            self.shared = folder / host
            self.shared.touch(exist_ok=True)

    @contextlib.contextmanager
    def slot(self):
        """Hold the host for one request: yields (start, waited). `start` is the wall-clock time after the last wait;
        the caller sends right after it. The end time is recorded when the block exits."""
        import fcntl
        with self.lock, (open(self.shared, 'r+') if self.shared else contextlib.nullcontext()) as f:
            if f:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                if f:
                    text = f.read().strip()
                    state = json.loads(text) if text.startswith('{') else {'end': 0.0, 'not_before': 0.0}
                else:
                    state = self.last
                t0 = time.time()
                while True:  # re-check after each sleep: a sleep may end early or late
                    gap = max(state['end'] + self.interval, state['not_before']) - time.time()
                    if gap <= 0:
                        break
                    time.sleep(gap)
                start = time.time()
                yield start, start - t0
            finally:
                end = time.time()
                self.last = {'end': end, 'not_before': end + self.interval}  # interval after observe() of this request
                if f:
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(self.last))
                    f.flush()
                    fcntl.flock(f, fcntl.LOCK_UN)

    def observe(self, status, latency, retry_after=None):
        """Call inside slot(): the interval it sets applies from this request's end, for every process."""
        self.latencies = (self.latencies + [latency])[-50:]
        median = statistics.median(self.latencies)
        if status == 429 or status >= 500 or retry_after:
            self.interval = min(self.max_interval, max(self.interval * 2, float(retry_after or 0)))
        elif len(self.latencies) >= 10 and latency > max(2 * median, median + 1.0):
            self.interval = min(self.max_interval, self.interval * 1.5)
        else:
            self.interval = max(self.base, self.interval * 0.95)


class BudgetExceeded(Exception):
    """A response body would pass the byte budget; it was not kept (the transfer stopped at the budget)."""


class NetworkError(Exception):
    """No response (connection, timeout); raised instead of retrying when the client was built with retry=False."""


class Client:
    def __init__(self, limiter, request_log=None, retry=True):
        """retry=False: one attempt per call; every response is returned whatever its status and a network error
        raises NetworkError, so the caller decides when to stop (bounded runs, MODEL-RESUME-B-026)."""
        import requests
        self.session, self.limiter = requests.Session(), limiter
        self.session.headers.update({'User-Agent': UA})
        self.request_log, self.log_lock = request_log, threading.Lock()
        self.retry = retry

    def request(self, method, url, metrics, max_bytes=None, **kw):
        """max_bytes: stream the body and stop reading once it passes this many bytes (BudgetExceeded)."""
        for attempt in range(6 if self.retry else 1):
            over = False
            with self.limiter.slot() as (start, waited):
                started = time.monotonic()
                r = error = None
                try:
                    if max_bytes is None:
                        r = self.session.request(method, url, timeout=180, **kw)
                        size = len(r.content)  # read the body inside the slot: one transfer in flight per host
                    else:
                        r = self.session.request(method, url, timeout=180, stream=True, **kw)
                        chunks, size = [], 0
                        for chunk in r.iter_content(65536):
                            size += len(chunk)
                            if size > max_bytes:
                                over = True
                                break
                            chunks.append(chunk)
                        r.close()
                        r._content, r._content_consumed = b''.join(chunks), True
                except Exception as exc:  # network error: count as a 5xx-like failure and back off
                    error = exc
                    self.limiter.observe(599, time.monotonic() - started)
                else:
                    retry_after = r.headers.get('Retry-After')
                    retry_after = float(retry_after) if retry_after and retry_after.isdigit() else None
                    self.limiter.observe(r.status_code, r.elapsed.total_seconds(), retry_after)  # elapsed = time to headers
                end = time.time()
            if self.request_log:  # one line per request: actual start (after the last wait) and end
                with self.log_lock, open(self.request_log, 'a') as log:
                    log.write(json.dumps({'t': start, 'end': end, 'pid': os.getpid(), 'method': method, 'url': url,
                                          'status': r.status_code if r is not None else None}) + '\n')
            latency = time.monotonic() - started
            if error is not None:
                metrics.record(waited, latency, 0, 599)
                if not self.retry:
                    raise NetworkError(f'{url}: {error!r}') from error
                print(f'retry after {error!r}', flush=True)
                continue
            metrics.record(waited, latency, size, r.status_code)
            if over:
                raise BudgetExceeded(f'{url}: body passed {max_bytes} bytes')
            if not self.retry or (r.status_code < 500 and r.status_code != 429):
                return r
        raise RuntimeError(f'request failed after retries: {url}')


class Metrics:
    """Thread-safe counters separating limiter wait, server response, payload, parsing and S3 time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {'requests': 0, 'wait_s': 0.0, 'response_s': 0.0, 'bytes': 0, 'parse_s': 0.0, 's3_s': 0.0,
                     'status': {}}

    def record(self, wait, response, size, status):
        with self.lock:
            d = self.data
            d['requests'] += 1
            d['wait_s'] += wait
            d['response_s'] += response
            d['bytes'] += size
            d['status'][str(status)] = d['status'].get(str(status), 0) + 1

    def add(self, field, seconds):
        with self.lock:
            self.data[field] += seconds

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data))


def s3_client():
    import boto3
    return boto3.client('s3')


def read_json_s3(s3, key, default):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)['Body'].read())
    except s3.exceptions.NoSuchKey:
        return default


def list_keys(s3, prefix):
    keys = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [item['Key'] for item in page.get('Contents', [])]
    return keys


def put_verified(s3, key, body):
    digest = hashlib.sha256(body).digest()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ChecksumSHA256=base64.b64encode(digest).decode())
    return digest.hex()


class RawExists(Exception):
    """The key already holds different bytes; raw objects are never replaced."""


def put_new(s3, key, body):
    """Write raw bytes once: S3 conditional put (If-None-Match: *) refuses to replace an existing object, so a re-run
    or a race cannot overwrite stored raw. Same bytes already stored -> returns their digest; different bytes -> raises
    RawExists (a deliberate new capture would need its own key, e.g. <id>.<sha256[:16]>.html.gz)."""
    digest = hashlib.sha256(body).digest()
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=body, ChecksumSHA256=base64.b64encode(digest).decode(), IfNoneMatch='*')
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
        if code not in ('PreconditionFailed', 'ConditionalRequestConflict'):
            raise
        head = s3.head_object(Bucket=BUCKET, Key=key, ChecksumMode='ENABLED')
        stored = base64.b64decode(head.get('ChecksumSHA256', '')).hex()
        if stored != digest.hex():
            raise RawExists(f'{key} stored sha256 {stored[:12]}') from None
    return digest.hex()


def put_raw_keep(s3, key, body):
    """put_new, but when the key already holds different bytes (a deliberate re-acquisition), keep both: the new bytes
    go to <stem>.<sha256[:16]><suffix>. Returns (sha256 hex, key written or matched)."""
    try:
        return put_new(s3, key, body), key
    except RawExists:
        digest = hashlib.sha256(body).hexdigest()
        stem, dot, suffix = key.rpartition('/')[2].partition('.')
        alt = f"{key.rpartition('/')[0]}/{stem}.{digest[:16]}{dot}{suffix}"
        return put_new(s3, alt, body), alt


def discover(client, s3, terms, classes, index, metrics):
    """Resumable walk; state = {'done': ['<term>/<class>', ...], 'meetings': {id: {...}}, 'walk': {'<term>/<class>':
    {'queue': [...], 'seen': [...], 'requests': n}}}. The walk (pending list nodes and requests already made) is saved
    every 25 requests, so a stop never repeats finished list requests (B-007)."""
    key = f'corpus/state/{SOURCE}/{index}'
    state = read_json_s3(s3, key, {'done': [], 'meetings': {}})
    walks = state.setdefault('walk', {})
    save = lambda: s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(state).encode())
    for term in terms:
        for cls in classes:
            name = f'{term}/{cls}'
            if name in state['done']:
                continue
            list_url = f'{BASE}/mnts/total/{term}.do?' + urlencode({'class_id_sch': cls, 'cmit_chk': 'all'})
            if name in walks:
                walk = walks[name]
                queue = [tuple(item) for item in walk['queue']]
                seen_requests = {(endpoint, tuple(map(tuple, form))) for endpoint, form in walk['seen']}
                requests_made = walk['requests']
            else:
                page = client.request('GET', list_url, metrics).text
                queue, seen_requests, requests_made = nodes_of(page), set(), 0
                for mid in minute_ids(page):
                    state['meetings'].setdefault(str(mid), {'id': mid, 'th': term, 'class': cls})

            def checkpoint():
                walks[name] = {'queue': [list(item) for item in queue], 'requests': requests_made,
                               'seen': [[endpoint, [list(kv) for kv in form]] for endpoint, form in seen_requests]}
                save()

            while queue:
                kind, data = queue.pop()
                data.setdefault('th', term)
                data.setdefault('class', cls)
                endpoint, form = payload(kind, data, term)
                request_key = (endpoint, tuple(sorted(form.items())))
                if request_key in seen_requests:
                    continue
                sub = client.request('POST', BASE + endpoint, metrics, data=form,
                                     headers={'X-Requested-With': 'XMLHttpRequest', 'Referer': list_url}).text
                seen_requests.add(request_key)
                requests_made += 1
                for mid in minute_ids(sub):
                    state['meetings'].setdefault(str(mid), {'id': mid, 'th': term, 'class': cls, 'sess': data.get('sess', '')})
                queue.extend(nodes_of(sub))
                if requests_made % 25 == 0:
                    checkpoint()
            state['done'].append(name)
            walks.pop(name, None)
            save()
            print(json.dumps({'term': term, 'class': cls, 'meetings_total': len(state['meetings']),
                              'list_requests': requests_made, 'metrics': metrics.snapshot()}), flush=True)


def done_ids(s3):
    ids = set()
    for key in list_keys(s3, f'corpus/state/{SOURCE}/{VERSION}/shard-'):
        ids.update(read_json_s3(s3, key, {'ids': []})['ids'])
    return ids


def stratified(meetings, seed):
    """Seeded order that alternates meeting classes (plenary, committee, audit ...) so a limited fetch is spread."""
    import random
    by_class = {}
    for m in sorted(meetings, key=lambda m: m['id']):
        by_class.setdefault(str(m.get('class')), []).append(m)
    queues = []
    for cls in sorted(by_class):
        q = by_class[cls]
        random.Random(f'{seed}:{cls}').shuffle(q)
        queues.append(q)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded, not as on disk at write time
_HASHES = {}


def code_hashes():
    """Hashes of the collector and the redactor as imported by this process (the redactor is imported here, once)."""
    if not _HASHES:
        import trid.korean_quality as kq
        _HASHES.update(code_sha256=_CODE_SHA256, korean_quality_sha256=hashlib.sha256(Path(kq.__file__).read_bytes()).hexdigest())
    return _HASHES


def write_shard(s3, client, shard, results, workers, metrics, started, origin):
    """results: [(meeting, row|None, pii|None, error|None)] -> verified docs/quarantine shards + manifest."""
    seen_text, docs, held = {}, [], []
    for _, row, _, _ in results:
        if row and row['text']:
            if row['sha256'] in seen_text:
                row['flags'].append(f"duplicate_text_of:{seen_text[row['sha256']]}")
            seen_text.setdefault(row['sha256'], row['id'])
            (held if any(f.startswith('quarantine:') for f in row['flags']) else docs).append(row)
    t = time.monotonic()
    keys = {}
    for name, rows in (('docs', docs), ('quarantine', held)):
        body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(), mtime=0)
        key = f'corpus/clean/{SOURCE}/{VERSION}/{name}-{shard:05d}.jsonl.gz'
        keys[name] = {'key': key, 'sha256': put_verified(s3, key, body), 'bytes': len(body), 'rows': len(rows),
                      'characters': sum(len(r['text']) for r in rows)}
    pii = Counter()
    for _, _, p, _ in results:
        pii.update(p or {})
    manifest = {**keys['docs'], 'quarantine': keys['quarantine'], 'origin': origin,
                **code_hashes(),
                'ids': [m['id'] for m, row, _, _ in results if row is not None],
                'failures': {str(m['id']): err for m, _, _, err in results if err},
                'empty_text_ids': [str(m['id']) for m, row, _, _ in results if row is not None and not row['text']],
                'duplicate_text_rows': sum(any(f.startswith('duplicate_text_of:') for f in r['flags']) for r in docs + held),
                'quarantine_reasons': dict(Counter(f for r in held for f in r['flags'] if f.startswith('quarantine:'))),
                'quarantined_ids': [r['meta']['minutes_id'] for r in held],
                'splits': {s: sum(r['split'] == s for r in docs) for s in ('train', 'valid', 'test')},
                'chars_by_kind': {k: sum(len(r['text']) for r in docs if r['kind'] == k) for k in {r['kind'] for r in docs}},
                'pii_redactions': dict(pii), 'metrics': metrics.snapshot(), 'elapsed_s': round(time.monotonic() - started, 1),
                'workers': workers, 'interval_s': client.limiter.interval if client else None}
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{VERSION}/shard-{shard:05d}.json',
                  Body=json.dumps(manifest, ensure_ascii=False).encode())
    metrics.add('s3_s', time.monotonic() - t)
    return manifest


def next_shard(s3):
    return 1 + max([-1] + [int(k.rsplit('-', 1)[1].split('.')[0]) for k in list_keys(s3, f'corpus/state/{SOURCE}/{VERSION}/shard-')])


def side_inputs(s3):
    """(official list metadata, listed quarantine) maps keyed by meeting id, read once per run. Also pins the code
    hashes recorded in every shard manifest of this run."""
    code_hashes()
    return read_json_s3(s3, OFFICIAL_META_KEY, {}), read_json_s3(s3, OVERLAP_QUARANTINE_KEY, {})


def processed(meeting, html_bytes, side, raw_key, raw_sha):
    official, listed = side
    mid = str(meeting['id'])
    t = time.monotonic()
    row, pii = row_of(meeting, html_bytes.decode('utf-8', 'replace'),
                      datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), official.get(mid),
                      (listed.get(mid) or {}).get('reason'))
    if mid in listed:
        row['meta']['listed_quarantine_evidence'] = listed[mid]
    row['meta'].update({'raw_key': raw_key, 'raw_gzip_sha256': raw_sha, 'raw_html_sha256': hashlib.sha256(html_bytes).hexdigest()})
    return row, pii, time.monotonic() - t


def client_error_ids(s3):
    """Meetings that any earlier run (any version) got an HTTP 4xx for: not asked again unless retry_failed."""
    ids = {}
    for key in list_keys(s3, f'corpus/state/{SOURCE}/'):
        if '/shard-' in key and key.endswith('.json'):
            for mid, err in read_json_s3(s3, key, {}).get('failures', {}).items():
                if str(err).startswith('http_4'):
                    ids[int(mid)] = err
    return ids


def fetch(client, s3, index, per_shard, workers, limit, metrics, seed=None, classes=None, retry_failed=False):
    meetings = list(read_json_s3(s3, f'corpus/state/{SOURCE}/{index}', {'meetings': {}})['meetings'].values())
    meetings = [m for m in meetings if not classes or str(m.get('class')) in classes]
    meetings = stratified(meetings, seed) if seed is not None else sorted(meetings, key=lambda m: m['id'])
    done = done_ids(s3)
    # Never download a meeting twice: raw already stored under any version is reprocessed, not fetched (B-008: a run
    # that only looked at the current version's manifests re-downloaded 10 meetings and replaced their raw bytes).
    have_raw = {int(k.rsplit('/', 1)[1].split('.')[0]) for k in list_keys(s3, f'corpus/raw/{SOURCE}/') if k.endswith('.html.gz')}
    refused = {} if retry_failed else client_error_ids(s3)
    todo = [m for m in meetings if m['id'] not in done and m['id'] not in have_raw and m['id'] not in refused][:limit]
    official = side_inputs(s3)
    print(f'{len(meetings)} meetings indexed, {len(done)} done in {VERSION}, {len(have_raw)} with stored raw, '
          f'{len(refused)} skipped after an earlier HTTP 4xx, {len(todo)} to fetch '
          f'({dict(Counter(str(m.get("class")) for m in todo))} by class)', flush=True)
    shard, started = next_shard(s3), time.monotonic()

    def one(meeting):
        r = client.request('GET', f"{BASE}/viewer/minutes/xml.do?id={meeting['id']}&type=view", metrics)
        if r.status_code != 200 or '회의록' not in r.text:
            return meeting, None, None, f'http_{r.status_code}'
        t = time.monotonic()
        raw_key = f"corpus/raw/{SOURCE}/{meeting['id']}.html.gz"
        try:
            raw_sha = put_new(s3, raw_key, gzip.compress(r.content, mtime=0))
        except RawExists as exc:  # stored bytes stay; nothing built from the new response
            return meeting, None, None, f'raw_exists_not_overwritten:{exc}'
        metrics.add('s3_s', time.monotonic() - t)
        row, pii, parse = processed(meeting, r.content, official, raw_key, raw_sha)
        metrics.add('parse_s', parse)
        return meeting, row, pii, None

    for begin in range(0, len(todo), per_shard):
        chunk = todo[begin:begin + per_shard]
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(one, chunk))
        m = write_shard(s3, client, shard, results, workers, metrics, started, 'download')
        done_n = begin + len(chunk)
        rate = done_n / max(1e-9, time.monotonic() - started)
        print(json.dumps({'shard': shard, 'rows': m['rows'], 'quarantine': m['quarantine']['rows'],
                          'failures': len(m['failures']), 'done': done_n, 'todo': len(todo),
                          'meetings_per_s': round(rate, 3), 'eta_h': round((len(todo) - done_n) / max(rate, 1e-9) / 3600, 2),
                          'interval_s': round(client.limiter.interval, 2), 'metrics': metrics.snapshot()}), flush=True)
        shard += 1


def reprocess(s3, index, per_shard, metrics):
    """Rebuild, from the stored raw HTML (no download), every indexed meeting that has raw in S3 and no row in the
    current version yet."""
    meetings = read_json_s3(s3, f'corpus/state/{SOURCE}/{index}', {'meetings': {}})['meetings']
    have_raw = {int(k.rsplit('/', 1)[1].split('.')[0]) for k in list_keys(s3, f'corpus/raw/{SOURCE}/') if k.endswith('.html.gz')}
    ids = sorted(int(i) for i in meetings if int(i) in have_raw)
    done = done_ids(s3)
    todo = [meetings[str(i)] for i in ids if i not in done]
    official = side_inputs(s3)
    print(f'{len(ids)} indexed meetings with raw, {len(todo)} to rebuild from raw', flush=True)
    shard, started = next_shard(s3), time.monotonic()
    for begin in range(0, len(todo), per_shard):
        results = []
        for meeting in todo[begin:begin + per_shard]:
            raw_key = f"corpus/raw/{SOURCE}/{meeting['id']}.html.gz"
            body = s3.get_object(Bucket=BUCKET, Key=raw_key)['Body'].read()
            row, pii, parse = processed(meeting, gzip.decompress(body), official, raw_key, hashlib.sha256(body).hexdigest())
            metrics.add('parse_s', parse)
            results.append((meeting, row, pii, None))
        m = write_shard(s3, None, shard, results, 0, metrics, started, 'raw-reprocess')
        print(json.dumps({'shard': shard, 'rows': m['rows'], 'quarantine': m['quarantine']['rows'],
                          'reasons': m['quarantine_reasons']}, ensure_ascii=False), flush=True)
        shard += 1


def aggregate(s3, index):
    keys = list_keys(s3, f'corpus/state/{SOURCE}/{VERSION}/shard-')
    manifests = [read_json_s3(s3, k, {}) for k in keys]
    for m in manifests:
        for part in (m, m.get('quarantine')):
            if not part:
                continue
            head = s3.head_object(Bucket=BUCKET, Key=part['key'], ChecksumMode='ENABLED')
            if base64.b64decode(head.get('ChecksumSHA256', '')).hex() != part['sha256'] or head['ContentLength'] != part['bytes']:
                raise RuntimeError(f"shard verification failed: {part['key']}")
    meetings = read_json_s3(s3, f'corpus/state/{SOURCE}/{index}', {'meetings': {}, 'done': []})
    chars = sum(m['characters'] for m in manifests)
    sums = lambda field: {k: sum(m[field].get(k, 0) for m in manifests) for k in {k for m in manifests for k in m[field]}}
    stats = {'source': SOURCE, 'clean_version': VERSION, 'meetings_indexed': len(meetings['meetings']),
             'index_pairs_done': len(meetings['done']), 'meetings_fetched': sum(len(m['ids']) for m in manifests),
             'rows': sum(m['rows'] for m in manifests), 'characters': chars, 'estimated_tokens': round(chars / 1.6),
             'estimate_note': 'chars / 1.6, not a tokenizer count', 'chars_by_kind': sums('chars_by_kind'),
             'splits': sums('splits'), 'pii_redactions': sums('pii_redactions'), 'shards': len(manifests),
             'quarantine_rows': sum(m.get('quarantine', {}).get('rows', 0) for m in manifests),
             'quarantine_reasons': sums('quarantine_reasons'),
             'duplicate_text_rows': sum(m.get('duplicate_text_rows', 0) for m in manifests)}
    fetched = {int(i) for m in manifests for i in m['ids']}
    stats['not_collected'] = sorted(int(i) for i in meetings['meetings'] if int(i) not in fetched)  # causes: shard failures / logs
    stats['code'] = sorted({json.dumps({k: m.get(k) for k in ('code_sha256', 'korean_quality_sha256')}) for m in manifests})
    provenance = {'source': SOURCE, 'clean_version': VERSION, 'site': 'https://record.assembly.go.kr/assembly/',
                  'license': LICENSE, 'license_evidence': ['https://record.assembly.go.kr/assembly/ footer: 공공누리 제1유형(출처표시)',
                                                           'https://open.assembly.go.kr/portal/policy/copyRightPage.do'],
                  'collector': 'scripts/collect_assembly_minutes.py', 'terms': TERMS, 'classes': CLASSES,
                  'request_policy': 'one host-wide request-start limiter (default 1 s), adaptive slow-down on 429/5xx/Retry-After/latency; no IP or account spreading',
                  'extraction': 'viewer HTML from the document <h1> heading; tags stripped; viewer controls removed; ko-pii redaction'}
    prefix = f'corpus/clean/{SOURCE}/{VERSION}'
    for name, body in (('stats.json', json.dumps(stats, ensure_ascii=False, indent=2)),
                       ('provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2)),
                       ('ATTRIBUTION.txt', '출처: 국회사무처 국회회의록 (https://record.assembly.go.kr), 공공누리 제1유형(출처표시). '
                                           '각 행의 source URL이 원문 회의록입니다.\n')):
        s3.put_object(Bucket=BUCKET, Key=f'{prefix}/{name}', Body=body.encode())
    shards = [{'manifest': k, 'origin': m['origin'], 'code_sha256': m.get('code_sha256'),
               'korean_quality_sha256': m.get('korean_quality_sha256'),
               **{part: {f: p[f] for f in ('key', 'sha256', 'bytes', 'rows', 'characters')}
                  for part, p in (('docs', m), ('quarantine', m['quarantine']))}}
              for k, m in zip(keys, manifests)]
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{VERSION}.manifest.json',
                  Body=json.dumps({**stats, 'shards': shards}, ensure_ascii=False, indent=1).encode())
    if stats['meetings_fetched'] >= stats['meetings_indexed'] and stats['index_pairs_done'] == len(TERMS) * len(CLASSES):
        s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{VERSION}/_DONE', Body=json.dumps(stats).encode())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['discover', 'fetch', 'reprocess', 'aggregate'])
    p.add_argument('--index', default='index.json', help='state file name for the resumable meeting index')
    p.add_argument('--terms', nargs='*', default=TERMS)
    p.add_argument('--classes', nargs='*', default=list(CLASSES))
    p.add_argument('--interval', type=float, default=1.0, help='host-wide minimum seconds between request starts')
    p.add_argument('--workers', type=int, default=4, help='concurrent fetches sharing the host limiter')
    p.add_argument('--per-shard', type=int, default=200)
    p.add_argument('--limit', type=int, help='fetch at most this many meetings (pilot)')
    p.add_argument('--seed', help='fetch order: seeded, alternating meeting classes (default: ascending id)')
    p.add_argument('--retry-failed', action='store_true', help='fetch: also ask again for meetings that got an HTTP 4xx')
    args = p.parse_args()
    s3, metrics = s3_client(), Metrics()
    host = BASE.split('://', 1)[1].split('/', 1)[0]
    request_log = Path(__file__).resolve().parents[1] / f'logs/requests-{host}.jsonl'
    request_log.parent.mkdir(exist_ok=True)
    client = Client(HostLimiter(args.interval, host=host), request_log)
    if args.action == 'discover':
        discover(client, s3, args.terms, args.classes, args.index, metrics)
    elif args.action == 'fetch':
        fetch(client, s3, args.index, args.per_shard, args.workers, args.limit, metrics, args.seed, set(args.classes),
              args.retry_failed)
    elif args.action == 'reprocess':
        reprocess(s3, args.index, args.per_shard, metrics)
    else:
        aggregate(s3, args.index)


if __name__ == '__main__':
    main()
