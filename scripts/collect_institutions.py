"""Public institutions (ALIO, www.alio.go.kr) inventory for the corpus contract (docs/CORPUS_CONTRACT.md).

  agencies     every institution ALIO lists (/organ/findOrganApbaList.json): name, ALIO id, type (공기업/준정부기관/
               기타공공기관), supervising ministry, homepage, parent -> corpus/state/institutions/agencies.json and the
               repo copy data/corpus-inventory/alio-agencies.json.
  rules-index  the 내부규정 수시공시 catalogue (/occasional/findRuleList.json): one entry per published rule with
               institution, rule category (insdRuleDivis), title and the page's 재·개정일자 (ruleStDa) ->
               corpus/state/institutions/rules-index.json (resumable by page). Metadata only.
  reserved     benchmark-reserved rule families (supervisor reservation JSONL, read-only): detail JSON, every revision
               and attachment file, extracted text -> corpus/reserved/<reservation>/ only, never corpus/clean.
               Per-family state corpus/reserved/<reservation>/state/<seq>.json (resume skips finished families);
               acquisition status JSONL, A's check_reserved_overlap.py `sign` input and a summary are rebuilt from
               the states after every batch.
  --local DIR  (reserved, reserved-sign-input) the same pipeline with every object under DIR on this machine instead
               of S3; boto3 is never imported and no AWS client is created. Built for a bounded resume
               (MODEL-RESUME-B-026): one request in flight, start interval >= 1 s, no retries; it stops after a
               checkpoint on 401/403/429/451, an access-restriction page, --max-network-failures consecutive network
               failures, or when --max-bytes (response bytes, kept across runs in DIR/budget.json) or --max-disk-bytes
               (bytes under DIR) would be passed. Families already acquired elsewhere (--skip-acquisition, the S3
               run's status JSONL) are skipped; files are written atomically (temp file + link, never replaced) under
               validated keys. Status goes to reports/coordination/<reservation>-local-acquisition{,-summary}.json*.
  reserved-reextract --local DIR --report STEM
               offline, no client: every file or zip member that is not 'ok' is read again from the stored raw
               bytes (sha256 checked) with content-based kinds (DRM wrapper, HWP bytes under .hwpx, xlsx), nested
               zips (up to 3 levels) and OCR for PDFs without a text layer (kordoc --ocr, cached local model,
               recorded as ocr). Protected HWP (password/배포용/DRM flags in the FileHeader) and DRM files are
               recorded as 'unextractable' with the reason; no lock is removed. 'ok' records are left as they were.
               reserved-sign-input --only-reextracted then writes just that new text (a delta manifest input).
  reserved-migrate --local DIR --profile P --manifest-out FILE [--extra FILES] [--dry-run]
               copies a local store (and extra files such as sign inputs/manifests) to S3 under reserved-only
               prefixes with put_new (never replaces an object), checks every object's stored SHA-256 and length
               (HeadObject) and GETs --roundtrip keys back. --dry-run plans offline without importing the AWS SDK.
  bucket-check --profile P
               read-only bucket settings (encryption, public access block, ownership, versioning, logging).
  rules-batch --profile P
               the next non-reserved ALIO rule families of the corpus inventory (the 1,000 reserved families are
               excluded) into S3 --root, with the bounded loop (one request, >= 1 s, stops as above), at most 1 GiB of
               responses and --max-s3-bytes of writes; every state carries training_hold licence_unconfirmed.
  AWS access (MODEL-RESUME-B-028): a named non-root --profile only; root identities and credentials in the
  environment are refused.

Licence evidence: ALIO 저작권정책 (https://www.alio.go.kr/notice/copyright.do) - public data free to use including
commercial use under 공공데이터법; ALIO-owned works under 저작권법 제24조의2; third-party rights and 비공개 정보 excluded.
Each institution's own regulation files remain that institution's public works; their per-file reuse basis is recorded
when text is collected. robots.txt: "Allow: /". Requests share one host-wide start interval (default 1 s), with
back-off on 429/5xx/Retry-After/latency (the HostLimiter of collect_assembly_minutes).
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from collect_assembly_minutes import (BUCKET, BudgetExceeded, Client, HostLimiter, Metrics, NetworkError,  # noqa: E402
                                      list_keys, put_new, put_raw_keep, put_verified, read_json_s3, s3_client)

SOURCE = 'institutions'
REPORTS = ROOT / 'reports/coordination'  # status JSONL / summary files (tests point this elsewhere)
BASE = 'https://www.alio.go.kr'


def agency_row(item):
    keep = ('apbaId', 'apbaNa', 'parnApbaId', 'parnApbaNa', 'typeNa', 'jidtNa', 'apbaType', 'jidtDptm', 'homepage',
            'addrCd', 'fdate')
    row = {k: item.get(k) for k in keep}
    row['homepage'] = (row['homepage'] or '').strip() or None
    return row


def rule_row(item):
    return {'seq': item.get('seq'), 'institution_id': item.get('apbaId'), 'institution': item.get('pname'),
            'institution_type_code': item.get('apbaType'), 'ministry_code': item.get('jidtDptm'),
            'title': item.get('title'), 'category': item.get('insdRuleDivis'),
            'revision_date': item.get('ruleStDa'),  # shown on the detail page as 재·개정일자, not 시행일
            'disclosed_date': item.get('idate'), 'base_year': item.get('critYyyy')}


def agencies(client, s3, metrics):
    rows, page = [], 1
    while True:
        r = client.request('POST', f'{BASE}/organ/findOrganApbaList.json', metrics,
                           json={'apbaNa': '', 'pageNo': page}).json()
        organ = r['data']['organList']
        rows += [agency_row(x) for x in organ['result']]
        if not organ['page'].get('nextPage_is') and page >= organ['page'].get('totalPage', page):
            break
        page += 1
    body = json.dumps({'source': f'{BASE}/guide/publicAgencyList.do', 'count': len(rows),
                       'by_type': dict(Counter(r['typeNa'] for r in rows)), 'agencies': rows,
                       'metrics': metrics.snapshot()}, ensure_ascii=False, indent=1)
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/agencies.json', Body=body.encode())
    out = ROOT / 'data/corpus-inventory/alio-agencies.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + '\n')
    print(json.dumps({'agencies': len(rows), 'by_type': dict(Counter(r['typeNa'] for r in rows))}, ensure_ascii=False))


def rules_index(client, s3, metrics, max_pages=None, workers=1):
    """Pages are fetched in batches of 50; state advances only after a whole batch, so a stop loses <50 pages.
    Workers overlap server latency; request starts still share the one host interval."""
    key = f'corpus/state/{SOURCE}/rules-index.json'
    state = read_json_s3(s3, key, {'next_page': 1, 'total_pages': None, 'rules': {}})
    fetch = lambda page: client.request('GET', f'{BASE}/occasional/findRuleList.json', metrics,
                                        params={'pageNo': page}).json()
    if state['total_pages'] is None:
        first = fetch(1)
        state['total_pages'], state['total_count'] = first['data']['page']['totalPage'], first['data']['totalCnt']
    last = min(state['total_pages'], max_pages or state['total_pages'])
    with ThreadPoolExecutor(workers) as pool:
        while state['next_page'] <= last:
            batch = range(state['next_page'], min(last, state['next_page'] + 49) + 1)
            for r in pool.map(fetch, batch):
                state['total_count'] = r['data']['totalCnt']
                for item in r['data']['result']:
                    state['rules'][str(item['seq'])] = rule_row(item)
            state['next_page'] = batch[-1] + 1
            state['metrics'] = metrics.snapshot()
            s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(state, ensure_ascii=False).encode())
            print(json.dumps({'page': batch[-1], 'total_pages': state['total_pages'], 'rules': len(state['rules']),
                              'workers': workers, 'interval_s': round(client.limiter.interval, 2),
                              'metrics': metrics.snapshot()}), flush=True)


FILE_URL = BASE + '/download/rulefiledown.json?fileNo={}'
DATE_IN_NAME = re.compile(r'\d{4}\s*(?:년도?|[.\-])\s*\d{1,2}\s*(?:월|[.\-])\s*\d{1,2}\s*일?')
LICENCE_EVIDENCE = {'site_policy': 'https://www.alio.go.kr/notice/copyright.do',
                    'site_policy_text': ('공공데이터법에 따라 알리오 제공 공공데이터는 영리 포함 자유이용; ALIO 보유 저작물은 '
                                         '저작권법 제24조의2; 제3자 권리·비공개정보는 제공 대상 제외'),
                    'original_licence': 'unconfirmed (per-file 공공누리 type not shown on ALIO)'}


def files_of(b_files):
    """The site's own split of bFiles ("fileNo|name,fileNo|name"); a name containing ',' is re-joined."""
    out = []
    for part in (b_files or '').split(','):
        if '|' in part and part.split('|', 1)[0].strip().isdigit():
            no, name = part.split('|', 1)
            out.append([no.strip(), name.strip()])
        elif out:
            out[-1][1] += ',' + part
    return [tuple(x) for x in out]


def kind_of(data, name):
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if data[:4] == b'%PDF':
        return 'pdf'
    if b'DRMONE' in data[:16]:  # DRM wrapper ("This Document is encrypted"), not a PDF/HWP stream (B-027)
        return 'drm:drmone'
    if data[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        # HWP 5 is an OLE file whose FileHeader stream starts with this signature; some institutions upload HWP
        # bytes under a .hwpx name (B-027)
        return 'hwp' if ext == 'hwp' or b'HWP Document File' in data else f'ole:{ext}'
    if data[:2] == b'PK':  # HWPX is a zip; some institutions upload HWPX bytes under a .hwp name
        return 'hwpx' if ext in ('hwpx', 'hwp') else 'xlsx' if ext == 'xlsx' else f'zip:{ext}'
    return f'unknown:{ext}'


class Unextractable(Exception):
    """The bytes are a known format whose text cannot be read without removing a protection (password, 배포용
    distribution lock, DRM) or without a reader that is not available; the reason is recorded, never hidden."""


HWP_FLAGS = {0: 'compressed', 1: 'password', 2: 'distribution', 4: 'drm', 8: 'cert_encrypted', 10: 'cert_drm'}


def hwp_flags(data):
    """Property flags of the HWP 5 FileHeader stream (signature, version, then a 32-bit flag word)."""
    i = data.find(b'HWP Document File')
    if i < 0 or len(data) < i + 40:
        return None
    prop = int.from_bytes(data[i + 36:i + 40], 'little')
    return [name for bit, name in HWP_FLAGS.items() if prop >> bit & 1]


def xlsx_text(data):
    """Cell text of every worksheet, one tab-joined line per row (shared strings and inline strings resolved)."""
    import io
    import zipfile
    from xml.etree import ElementTree as ET
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(f'{ns}si'):
            shared.append(''.join(t.text or '' for t in si.iter(f'{ns}t')))
    sheets = sorted((n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n)),
                    key=lambda n: int(re.search(r'\d+', n.rsplit('/', 1)[1]).group()))
    lines = []
    for name in sheets:
        for row in ET.fromstring(z.read(name)).iter(f'{ns}row'):
            cells = []
            for c in row.iter(f'{ns}c'):
                v, kind = c.find(f'{ns}v'), c.get('t')
                if kind == 's' and v is not None:
                    cells.append(shared[int(v.text)])
                elif kind == 'inlineStr':
                    cells.append(''.join(t.text or '' for t in c.iter(f'{ns}t')))
                elif v is not None and v.text:
                    cells.append(v.text)
            if any(x.strip() for x in cells):
                lines.append('\t'.join(cells))
    return '\n'.join(lines)


KORDOC = Path.home() / '.npm/_npx/09d8021b8ad008ca/node_modules/.bin/kordoc'  # kordoc 4.24.23, as in B-015


def ocr_text(data, timeout=1800):
    """Scanned/outlined PDF with no text layer: kordoc --ocr (built-in PP-OCRv5 korean, models cached locally under
    ~/.cache/kordoc; B-015 used the same tool). The result is OCR, recorded as such; never treated as born-digital."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / 'in.pdf', Path(d) / 'out.md'
        src.write_bytes(data)
        subprocess.run([str(KORDOC), '--ocr', '-o', str(out), str(src)], check=True, capture_output=True,
                       timeout=timeout)
        return out.read_text()


def extract(kind, data):
    from trid.service.ingest import hwpx_text, normalize, tool_text
    if kind == 'hwpx':
        return normalize(hwpx_text(data))
    if kind == 'hwp':
        flags = hwp_flags(data) or []
        for flag in ('password', 'distribution', 'drm', 'cert_encrypted', 'cert_drm'):
            if flag in flags:  # body streams are encrypted: reading them means removing the lock (not done here)
                raise Unextractable(f'hwp_{flag}_protected')
        return normalize(tool_text(kind, data))
    if kind == 'pdf':
        return normalize(tool_text(kind, data))
    if kind == 'xlsx':
        return normalize(xlsx_text(data))
    if kind.startswith('drm:'):
        raise Unextractable(f'{kind.replace(":", "_")}_encrypted')
    raise ValueError(f'no extractor for {kind}')


class Stop(Exception):
    """A condition that ends a bounded run after a checkpoint (blocked/limited by the site, budget, network)."""


STOP_STATUS = {401, 403, 429, 451}


def check_response(r, kind):
    """Bounded runs: the site refusing or limiting us ends the run; nothing is retried or worked around."""
    if r.status_code in STOP_STATUS:
        raise Stop(f'http_{r.status_code} on {kind}')
    head = r.content[:4096].decode('utf-8', 'ignore')
    if 'text/html' in (r.headers.get('Content-Type') or '') and '접근' in head and ('제한' in head or '차단' in head):
        raise Stop(f'access_restriction_page on {kind}')


class S3Store:
    """The pipeline's objects in the training bucket. Raw bytes are written once (If-None-Match; different bytes go
    beside them as <stem>.<sha16>.<ext>). `max_disk_bytes` caps the bytes this store writes in one run (Stop before
    the write that would pass it); `disk_bytes` counts them. The names match LocalStore so the bounded loop reports
    both the same way."""

    def __init__(self, s3, max_disk_bytes=None):
        self.s3, self.max_disk_bytes, self.disk_bytes = s3, max_disk_bytes, 0

    def _reserve(self, size):
        if self.max_disk_bytes is not None and self.disk_bytes + size > self.max_disk_bytes:
            raise Stop(f'S3 write budget: {self.disk_bytes} + {size} > {self.max_disk_bytes} bytes')
        self.disk_bytes += size

    def raw(self, key, body):
        self._reserve(len(body))
        return put_raw_keep(self.s3, key, body)

    def text(self, key, body):
        self._reserve(len(body))
        return put_new(self.s3, key, body)

    def put(self, key, body):
        self._reserve(len(body))
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=body)

    def read(self, key):
        with self.s3.get_object(Bucket=BUCKET, Key=key)['Body'] as body:
            return body.read()

    def exists(self, key):
        try:
            self.s3.head_object(Bucket=BUCKET, Key=key)
            return True
        except self.s3.exceptions.ClientError as exc:
            if exc.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
                return False
            raise

    def states(self, prefix):
        return [read_json_s3(self.s3, key, None) for key in list_keys(self.s3, prefix)]

    def where(self, key):
        return f's3://{BUCKET}/{key}'


class LocalStore:
    """The same objects as files under `root`. Keys are checked (no absolute path, no '..', a fixed character set)
    and every write goes to a temp file in the target folder first. Raw and text are written once: os.link refuses an
    existing name, so a re-run keeps identical bytes and puts different bytes beside them (<stem>.<sha16>.<ext>, as in
    S3). States and status files are replaced atomically with os.replace. A write that would take the folder past
    `max_disk_bytes` raises Stop before anything is written."""

    KEY = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*')

    def __init__(self, root, max_disk_bytes):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = max_disk_bytes
        self.disk_bytes = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def path(self, key):
        if not self.KEY.fullmatch(key) or '..' in key.split('/'):
            raise ValueError(f'unsafe key: {key!r}')
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f'key leaves the store: {key!r}')
        return path

    def _reserve(self, size):
        if self.disk_bytes + size > self.max_disk_bytes:
            raise Stop(f'disk budget: {self.disk_bytes} + {size} > {self.max_disk_bytes} bytes')

    def _temp(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f'.{path.name}.tmp-{os.getpid()}'
        with open(tmp, 'wb') as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        return tmp

    def _once(self, path, body):
        """Write `path` once; raises FileExistsError when it already holds different bytes."""
        if path.exists():
            if hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(body).digest():
                return
            raise FileExistsError(path)
        self._reserve(len(body))
        tmp = self._temp(path, body)
        try:
            os.link(tmp, path)  # fails if the name exists: stored bytes are never replaced
        finally:
            tmp.unlink()
        self.disk_bytes += len(body)

    def raw(self, key, body):
        digest = hashlib.sha256(body).hexdigest()
        try:
            self._once(self.path(key), body)
            return digest, key
        except FileExistsError:
            folder, _, name = key.rpartition('/')
            stem, dot, suffix = name.partition('.')
            alt = f'{folder}/{stem}.{digest[:16]}{dot}{suffix}'
            self._once(self.path(alt), body)
            return digest, alt

    def text(self, key, body):
        self._once(self.path(key), body)
        return hashlib.sha256(body).hexdigest()

    def put(self, key, body):
        path = self.path(key)
        old = path.stat().st_size if path.exists() else 0
        self._reserve(max(0, len(body) - old))
        os.replace(self._temp(path, body), path)
        self.disk_bytes += len(body) - old

    def read(self, key):
        return self.path(key).read_bytes()

    def exists(self, key):
        return self.path(key).exists()

    def states(self, prefix):
        folder = self.path(prefix.rstrip('/'))
        return [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))] if folder.is_dir() else []

    def where(self, key):
        return str(self.path(key))


def acquire_family(client, store, metrics, rid, fam, budget=None, root=None):
    """budget: bounded runs only, a callable giving the response bytes still allowed (bodies are cut there); the
    responses are then also checked for refusal or limiting (Stop). root: object prefix of the collection (default
    corpus/reserved/<rid>); a non-reserved collection records reservation: null."""
    seq = fam['doc_family_id'].rsplit(':', 1)[1]
    if not seq.isdigit():
        raise ValueError(f"unexpected family id {fam['doc_family_id']}")
    prefix = root or f'corpus/reserved/{rid}'
    limit = (lambda: {'max_bytes': budget()}) if budget else (lambda: {})
    r = client.request('GET', f'{BASE}/occasional/findRuleDtl.json', metrics, params={'seq': seq}, **limit())
    if budget:
        check_response(r, f'detail {seq}')
    t = time.monotonic()
    detail_sha, detail_key = store.raw(f'{prefix}/raw/{seq}/detail.json', r.content)  # raw is never replaced
    metrics.add('s3_s', time.monotonic() - t)
    state = {'doc_family_id': fam['doc_family_id'], 'seq': seq, 'reservation': None if root else rid,
             'reserved_business_group': fam.get('business_group'), 'business_group_review': 'pending',
             'page_url': f'{BASE}/occasional/ruleDtl.do?seq={seq}', 'detail_http': r.status_code,
             'detail_key': detail_key, 'detail_sha256': detail_sha,
             'collected_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
             'licence_evidence': LICENCE_EVIDENCE, 'licence_status': 'unconfirmed', 'files': []}
    body = r.json() if r.status_code == 200 and r.content[:1] == b'{' else {}
    d = body.get('data') if body.get('status') == 'success' else None
    if not d:
        state['status'] = 'detail_unavailable'
        return state
    state.update({'institution': d.get('pname'), 'institution_id': d.get('instNo'), 'title': d.get('title'),
                  'category': d.get('insdRuleDivis'),
                  'dates': {'alio_revision_date': d.get('retryRvsnYmd'),  # page label 재·개정일자
                            'alio_bdate': d.get('bdate'), 'alio_idate': d.get('idate'),  # unlabelled on the page
                            'effective_date': None, 'effective_date_note': '시행일 is only in the text; see effective_clauses'}})
    for index, (no, name) in enumerate(files_of(d.get('bFiles'))):
        m = DATE_IN_NAME.search(name)
        rec = {'file_no': no, 'file_name': name, 'list_index': index, 'url': FILE_URL.format(no),
               'filename_date_text': m.group(0) if m else None}
        state['files'].append(rec)
        f = client.request('GET', FILE_URL.format(no), metrics, **limit())
        if budget:
            check_response(f, f'file {no}')
        rec['http'] = f.status_code
        if f.status_code != 200 or not f.content:
            rec['download_status'] = f'http_{f.status_code}' if f.status_code != 200 else 'empty_body'
            continue
        data = f.content
        rec.update({'download_status': 'ok', 'bytes': len(data), 'content_type': f.headers.get('Content-Type'),
                    'kind': kind_of(data, name)})
        t = time.monotonic()
        ext = re.sub(r'[^a-z0-9]', '', rec['kind'].split(':')[-1].lower()) or 'bin'
        rec['raw_sha256'], rec['raw_key'] = store.raw(f'{prefix}/raw/{seq}/{no}.{ext}', data)
        metrics.add('s3_s', time.monotonic() - t)
        if rec['kind'] == 'zip:zip':
            rec['members'] = zip_members(store, metrics, prefix, seq, no, data)
            rec['extract_status'] = zip_status(rec['members'])
        else:
            rec.update(text_record(store, metrics, prefix, seq, no, rec['kind'], data))
    state['status'] = family_status(state['files'])
    return state


def family_status(files):
    ok = [x for x in files if x.get('extract_status') == 'ok']
    return ('no_files' if not files else 'text_acquired' if len(ok) == len(files)
            else 'partial' if ok else 'extraction_failed'
            if all(x.get('download_status') == 'ok' for x in files) else 'download_failed')


def redo_member(store, metrics, prefix, seq, stem, old, body, depth):
    """Re-extract one stored zip member that was not 'ok' (its bytes come from the stored zip, never the network)."""
    rec = {k: v for k, v in old.items() if k not in ('extract_status', 'extract_error', 'extract_reason', 'members')}
    rec['kind'] = kind_of(body, old.get('member') or '')
    if rec['kind'] == 'zip:zip' and depth < MAX_ZIP_DEPTH:
        rec['members'] = zip_members(store, metrics, prefix, seq, stem, body, depth + 1)
        rec['extract_status'] = zip_status(rec['members'])
    else:
        rec.update(text_record(store, metrics, prefix, seq, stem, rec['kind'], body))
    return rec


def reextract_local(store, rid, metrics, report, by):
    """Offline: for every local family with a file or zip member that is not 'ok', read the stored raw bytes again
    and extract them with the current kind detection and extractors. Records that were 'ok' are kept as they were
    (their text keys are write-once). Each changed record carries `reextract` {by, at, previous kind/status/error}."""
    import io
    import zipfile
    states, changed = load_states(store, rid), []
    at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    note = lambda old: {'by': by, 'at': at, 'previous': {k: old.get(k) for k in
                                                         ('kind', 'extract_status', 'extract_error', 'extract_reason')}}
    for seq, st in sorted(states.items(), key=lambda kv: int(kv[0])):
        prefix, before = f'corpus/reserved/{rid}', st['status']
        touched = False
        for f in st['files']:
            if f.get('extract_status') == 'ok' or f.get('download_status') != 'ok':
                continue
            data = store.read(f['raw_key'])
            if 'members' in f:
                archive = zipfile.ZipFile(io.BytesIO(data))
                infos = [m for m in archive.infolist() if not m.is_dir()]
                for j, m in enumerate(f['members']):
                    if m.get('extract_status') == 'ok' or m.get('member_index') is None:
                        continue
                    i = m['member_index']
                    body = archive.read(infos[i]) if infos[i].file_size <= 200 * 1024 * 1024 else b''
                    if hashlib.sha256(body).hexdigest() != m['member_sha256']:
                        raise ValueError(f'{seq}/{f["file_no"]} member {i}: bytes differ from the recorded sha256')
                    f['members'][j] = {**redo_member(store, metrics, prefix, seq, f'{f["file_no"]}-m{i:03d}', m, body, 1),
                                       'reextract': note(m)}
                old_status = f['extract_status']
                f['extract_status'] = zip_status(f['members'])
                if f['extract_status'] != old_status:
                    f['reextract'] = {**note({**f, 'extract_status': old_status}), 'members_only': True}
            else:
                if hashlib.sha256(data).hexdigest() != f['raw_sha256']:
                    raise ValueError(f'{seq}/{f["file_no"]}: stored bytes differ from the recorded sha256')
                old = dict(f)
                for k in ('extract_status', 'extract_error', 'extract_reason'):
                    f.pop(k, None)
                f['kind'] = kind_of(data, f['file_name'])
                f.update(text_record(store, metrics, prefix, seq, f['file_no'], f['kind'], data))
                f['reextract'] = note(old)
            touched = True
        if touched:
            st['status'] = family_status(st['files'])
            st['reextracted'] = {'by': by, 'at': at, 'previous_status': before}
            store.put(f'corpus/reserved/{rid}/state/{seq}.json', json.dumps(st, ensure_ascii=False).encode())
            changed.append({'seq': seq, 'before': before, 'after': st['status']})
    summary = publish(store, rid, states, len(states), metrics, report=report,
                      extra={'reextract': {'by': by, 'at': at, 'families_changed': len(changed),
                                           'status_changes': dict(Counter(f"{c['before']}->{c['after']}" for c in changed))}})
    print(json.dumps({'families_changed': len(changed), 'status': summary['family_status'],
                      'disk_bytes': store.disk_bytes}, ensure_ascii=False), flush=True)
    return changed


def text_docs(st):
    """(record, doc_id, source_url, re-extracted) for every file and (nested) zip member of a family state; a member
    found inside a re-extracted record counts as re-extracted too."""
    def walk(recs, base, url, inherited):
        for r in recs:
            doc_id, redo = f"{base}:member:{r['member_index']}", inherited or 'reextract' in r
            yield r, doc_id, url, redo
            yield from walk(r.get('members', []), doc_id, url, redo)
    for f in st['files']:
        base = f"{st['doc_family_id']}:file:{f['file_no']}"
        redo = 'reextract' in f and not f['reextract'].get('members_only')
        yield f, base, f['url'], redo
        yield from walk(f.get('members', []), base, f['url'], redo)


EXTRACTOR = {'hwp': 'rhwp export-doclang', 'pdf': 'pdftotext', 'hwpx': 'hwpx section XML', 'xlsx': 'xlsx sheet XML'}
OCR_EXTRACTOR = 'kordoc 4.24.23 --ocr (PP-OCRv5 korean)'
MAX_ZIP_DEPTH = 3


def text_record(store, metrics, prefix, seq, stem, kind, data):
    """Extract one document; text stored at <prefix>/text/<seq>/<stem>.txt. Failures are recorded, never promoted:
    'unextractable' (protected/encrypted, with extract_reason), 'failed' (extractor error), 'empty_text'. A PDF
    without a text layer is read by OCR and marked ocr: true."""
    from trid.service.ingest import normalize
    rec, t = {}, time.monotonic()
    extractor = EXTRACTOR.get(kind)
    try:
        text = unicodedata.normalize('NFC', extract(kind, data))
        rec['extract_status'] = 'ok' if text.strip() else 'empty_text'
    except Unextractable as exc:
        text, rec['extract_status'], rec['extract_reason'] = '', 'unextractable', str(exc)
    except Exception as exc:
        text, rec['extract_status'], rec['extract_error'] = '', 'failed', repr(exc)[:300]
    if kind == 'pdf' and rec['extract_status'] == 'empty_text':
        try:
            text = unicodedata.normalize('NFC', normalize(ocr_text(data)))
        except Exception as exc:
            rec['ocr_error'] = repr(exc)[:300]
        else:
            rec['ocr'], extractor = True, OCR_EXTRACTOR
            rec['extract_status'] = 'ok' if text.strip() else 'empty_text'
            if not text.strip():
                rec['extract_reason'] = 'ocr_found_no_text'
    metrics.add('parse_s', time.monotonic() - t)
    if text.strip():
        t = time.monotonic()
        rec['text_sha256'] = store.text(f'{prefix}/text/{seq}/{stem}.txt', text.encode())
        metrics.add('s3_s', time.monotonic() - t)
        rec.update({'text_key': f'{prefix}/text/{seq}/{stem}.txt', 'chars': len(text), 'extractor': extractor,
                    'effective_clauses': [line.strip() for line in text.splitlines() if '시행한다' in line][:8],
                    'mentions_kogl': '공공누리' in text})
    return rec


def zip_status(members):
    states = Counter(x.get('extract_status') for x in members)
    return ('empty_archive' if not members else 'ok' if states == {'ok': len(members)}
            else 'partial' if states.get('ok') else 'failed')


def zip_members(store, metrics, prefix, seq, no, data, depth=1):
    """Archived revisions/attachments (e.g. "…(개정 전).zip"): one record per file member; the zip itself is the raw.
    A zip inside the zip is opened the same way (up to MAX_ZIP_DEPTH levels; member stems <no>-m001-m002)."""
    import io
    import zipfile
    members = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        return [{'member': None, 'extract_status': 'failed', 'extract_error': repr(exc)[:300]}]
    for i, info in enumerate(m for m in archive.infolist() if not m.is_dir()):
        name = info.filename
        if not info.flag_bits & 0x800:  # no UTF-8 flag: Korean archives are usually CP949
            try:
                name = name.encode('cp437').decode('cp949')
            except UnicodeError:
                pass
        body = archive.read(info) if info.file_size <= 200 * 1024 * 1024 else b''
        m = DATE_IN_NAME.search(name)
        rec = {'member': name, 'member_index': i, 'bytes': info.file_size, 'kind': kind_of(body, name),
               'member_sha256': hashlib.sha256(body).hexdigest(), 'filename_date_text': m.group(0) if m else None}
        if rec['kind'] == 'zip:zip' and depth < MAX_ZIP_DEPTH:
            rec['members'] = zip_members(store, metrics, prefix, seq, f'{no}-m{i:03d}', body, depth + 1)
            rec['extract_status'] = zip_status(rec['members'])
        else:
            rec.update(text_record(store, metrics, prefix, seq, f'{no}-m{i:03d}', rec['kind'], body))
        members.append(rec)
    return members


def round_robin(families):
    """Interleave business groups so a partial run covers every group (supervisor order kept inside a group)."""
    groups = {}
    for fam in families:
        groups.setdefault(fam.get('business_group'), []).append(fam)
    queues, out = list(groups.values()), []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def publish(store, rid, states, reservation_count, metrics=None, report=None, extra=None, root=None):
    """report: status-file stem under reports/coordination (default <rid>; the local run writes <rid>-local).
    root: collection prefix (default corpus/reserved/<rid>)."""
    report, root = report or rid, root or f'corpus/reserved/{rid}'
    rows = [states[k] for k in sorted(states, key=lambda k: int(k))]
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (REPORTS / f'{report}-acquisition.jsonl').write_text(text)
    store.put(f'{root}/acquisition.jsonl', text.encode())
    files = [f for r in rows for f in r['files']]
    summary = {'reservation': rid, 'reserved_families': reservation_count, 'families_attempted': len(rows),
               'families_pending': reservation_count - len(rows),
               'family_status': dict(Counter(r['status'] for r in rows)),
               'family_status_by_group': dict(Counter(f"{r['reserved_business_group']} | {r['status']}" for r in rows)),
               'licence_status': dict(Counter(r['licence_status'] for r in rows)),
               'files': len(files), 'file_kinds': dict(Counter(f.get('kind') for f in files)),
               'file_download_status': dict(Counter(f.get('download_status') for f in files)),
               'file_extract_status': dict(Counter(f.get('extract_status') for f in files)),
               'archive_members': sum(len(f.get('members', [])) for f in files),
               'archive_member_extract_status': dict(Counter(m.get('extract_status') for f in files for m in f.get('members', []))),
               'text_chars': sum(f.get('chars', 0) + sum(m.get('chars', 0) for m in f.get('members', [])) for f in files),
               'files_mentioning_kogl': sum(bool(f.get('mentions_kogl')) for f in files),
               'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), **(extra or {})}
    if metrics:
        summary['metrics'] = metrics.snapshot()
    store.put(f'{root}/summary.json', json.dumps(summary, ensure_ascii=False, indent=1).encode())
    (REPORTS / f'{report}-acquisition-summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    return summary


def load_states(store, rid, root=None):
    return {st['seq']: st for st in store.states(f"{root or f'corpus/reserved/{rid}'}/state/") if st}


def reserved(client, store, metrics, reservation, workers, limit=None, retry=()):
    """retry: family statuses to fetch again (e.g. partial after an extractor fix); their old state is replaced."""
    rid = reservation.name.removesuffix('.jsonl')
    families = [json.loads(line) for line in reservation.read_text().splitlines() if line.strip()]
    families = [f for f in families if f['doc_family_id'].startswith('institutions:alio-rule:')]
    states = load_states(store, rid)
    pending = lambda f: (st := states.get(f['doc_family_id'].rsplit(':', 1)[1])) is None or st['status'] in retry
    todo = [f for f in round_robin(families) if pending(f)][:limit]
    print(json.dumps({'reserved': len(families), 'done': len(states), 'todo': len(todo)}), flush=True)
    started = time.monotonic()

    def one(fam):
        try:
            return acquire_family(client, store, metrics, rid, fam)
        except Exception as exc:  # leave the family pending; a re-run retries it
            print(f"pending {fam['doc_family_id']}: {exc!r}", flush=True)
            return None

    with ThreadPoolExecutor(workers) as pool:
        for begin in range(0, len(todo), 20):
            for st in pool.map(one, todo[begin:begin + 20]):
                if st:
                    store.put(f"corpus/reserved/{rid}/state/{st['seq']}.json", json.dumps(st, ensure_ascii=False).encode())
                    states[st['seq']] = st
            summary = publish(store, rid, states, len(families), metrics)
            done = begin + len(todo[begin:begin + 20])
            rate = done / max(1e-9, time.monotonic() - started)
            print(json.dumps({'done': len(states), 'of': len(families), 'families_per_min': round(rate * 60, 2),
                              'eta_min': round((len(todo) - done) / max(rate, 1e-9) / 60, 1),
                              'status': summary['family_status'], 'interval_s': round(client.limiter.interval, 2),
                              'metrics': metrics.snapshot()}, ensure_ascii=False), flush=True)


def reserved_local(client, store, metrics, reservation, skip_acquisition, max_bytes, max_network_failures=3,
                   limit=None, retry=()):
    """Bounded, sequential resume of a reservation into a LocalStore. Families listed in `skip_acquisition` (the
    status JSONL of an earlier run, e.g. the S3 run's 148) are not fetched again. Returns the last summary."""
    rid = reservation.name.removesuffix('.jsonl')
    families = [json.loads(line) for line in reservation.read_text().splitlines() if line.strip()]
    families = [f for f in families if f['doc_family_id'].startswith('institutions:alio-rule:')]
    elsewhere = {json.loads(line)['doc_family_id'] for line in skip_acquisition.read_text().splitlines() if line.strip()}
    return bounded_collect(client, store, metrics, families, rid, elsewhere, max_bytes, max_network_failures, limit,
                           retry, report=f'{rid}-local')


def bounded_collect(client, store, metrics, families, rid, skip_ids, max_bytes, max_network_failures=3, limit=None,
                    retry=(), root=None, report=None, budget_key='budget.json', hold=()):
    """Sequential, bounded collection of ALIO rule families into `store` under `root` (default corpus/reserved/<rid>).
    A family's state is written as soon as it is done (checkpoint); a family cut by a stop keeps no state and stays
    pending. Response bytes are kept across runs in `budget_key` and bodies are cut so the total never passes
    `max_bytes`. Families in `skip_ids` are never requested. `hold`: training holds written on every state."""
    elsewhere, prefix = skip_ids, root or f'corpus/reserved/{rid}'
    states = load_states(store, rid, root)
    spent = (json.loads(store.read(budget_key)) if store.exists(budget_key)
             else {'response_bytes': 0, 'requests': 0})
    pending = lambda f: f['doc_family_id'] not in elsewhere and (
        (st := states.get(f['doc_family_id'].rsplit(':', 1)[1])) is None or st['status'] in retry)
    todo = [f for f in round_robin(families) if pending(f)][:limit]
    base = {'bytes': 0, 'requests': 0}  # everything this process sent counts, the robots.txt check included
    used = lambda: spent['response_bytes'] + metrics.snapshot()['bytes'] - base['bytes']
    remaining = lambda: max(0, max_bytes - used() - 65536)  # one stream chunk of slack: the total stays under max_bytes
    run = {'families_elsewhere': len(elsewhere), 'local_states_before': len(states), 'todo': len(todo),
           'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
    print(json.dumps({'reserved': len(families), **run, 'response_bytes_before': spent['response_bytes']}), flush=True)
    failures, done, stop = 0, 0, None

    def checkpoint():
        snap = metrics.snapshot()
        totals = {'response_bytes': spent['response_bytes'] + snap['bytes'] - base['bytes'],
                  'requests': spent['requests'] + snap['requests'] - base['requests']}
        store.put(budget_key, json.dumps(totals).encode())
        return publish(store, rid, states, len(families), metrics, report=report or f'{rid}-local', root=root,
                       extra={'families_acquired_elsewhere': len(elsewhere),
                              'budget': {'max_response_bytes': max_bytes, 'max_disk_bytes': store.max_disk_bytes,
                                         **totals, 'disk_bytes': store.disk_bytes},
                              'run': {**run, 'done_this_run': done, 'stop': stop}})

    for fam in todo:
        if remaining() <= 0:
            stop = f'response byte budget reached ({used()} of {max_bytes})'
            break
        try:
            st = acquire_family(client, store, metrics, rid, fam, budget=remaining, root=root)
        except Stop as exc:
            stop = str(exc)
            break
        except BudgetExceeded as exc:
            stop = f'response byte budget: {exc}'
            break
        except NetworkError as exc:
            failures += 1
            print(f"pending {fam['doc_family_id']}: {exc}", flush=True)
            if failures >= max_network_failures:
                stop = f'{failures} consecutive network failures (last: {exc})'
                break
            continue
        failures = 0
        if hold:
            st['training_hold'] = list(hold)
        store.put(f"{prefix}/state/{st['seq']}.json", json.dumps(st, ensure_ascii=False).encode())
        states[st['seq']] = st
        done += 1
        if done % 10 == 0:
            summary = checkpoint()
            print(json.dumps({'done_this_run': done, 'local_states': len(states), 'status': summary['family_status'],
                              'budget': summary['budget'], 'interval_s': round(client.limiter.interval, 2)},
                             ensure_ascii=False), flush=True)
    run['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    summary = checkpoint()
    print(json.dumps({'stop': stop, 'done_this_run': done, 'local_states': len(states),
                      'status': summary['family_status'], 'budget': summary['budget']}, ensure_ascii=False), flush=True)
    return summary


def sign_input(store, rid, local_out=None, only=None):
    """A's check_reserved_overlap.py `sign` input: one row per extracted file {family_id, text, doc_id, source_url,
    group}. S3 run: written to S3 and data/<rid>/sign-input.jsonl.gz (gitignored). Local run: written only to
    `local_out`, never over an existing file. only='reextracted': just the records a re-extraction produced text for
    (a delta for a new candidate manifest). Reserved text never goes under corpus/clean."""
    states = load_states(store, rid)
    rows = []
    for st in states.values():
        for doc, doc_id, url, redo in text_docs(st):
            if doc.get('text_key') and (only != 'reextracted' or redo):
                rows.append({'family_id': st['doc_family_id'], 'text': store.read(doc['text_key']).decode(),
                             'doc_id': doc_id, 'source_url': url, 'group': st['doc_family_id']})
    body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(), mtime=0)
    if local_out:
        local, digest, remote = Path(local_out), hashlib.sha256(body).hexdigest(), None
        if local.exists():
            raise SystemExit(f'{local} exists; versions are never overwritten')
    else:
        digest = put_verified(store.s3, f'corpus/reserved/{rid}/sign-input.jsonl.gz', body)
        local, remote = ROOT / f'data/{rid}/sign-input.jsonl.gz', store.where(f'corpus/reserved/{rid}/sign-input.jsonl.gz')
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(body)
    print(json.dumps({'rows': len(rows), 'families': len({r['family_id'] for r in rows}), 'sha256': digest,
                      's3': remote, 'local': str(local)}))


CREDENTIAL_ENV = ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_PROFILE')


def s3_session_client(profile, region='ap-northeast-2'):
    """S3 client for a named, non-root profile only (MODEL-RESUME-B-028): credentials in the environment are refused
    (no bypass of the profile), the caller identity is checked with STS and a root identity is refused. HTTPS is
    boto3's default endpoint. Returns (client, caller ARN)."""
    if not profile:
        raise SystemExit('--profile is required (a non-root profile with corpus/* access)')
    if set_env := [k for k in CREDENTIAL_ENV if os.environ.get(k)]:
        raise SystemExit(f'credentials in the environment are not used ({set_env}); unset them and pass --profile')
    import boto3
    session = boto3.Session(profile_name=profile, region_name=region)
    arn = session.client('sts').get_caller_identity()['Arn']
    if arn.endswith(':root'):
        raise SystemExit(f'profile {profile} is the account root ({arn}); refusing')
    client = session.client('s3')
    if not client.meta.endpoint_url.startswith('https://'):
        raise SystemExit(f'not an HTTPS endpoint: {client.meta.endpoint_url}')
    return client, arn


def bucket_check(s3):
    """Read-only bucket settings (encryption, public access block, ownership, versioning, logging, policy status).
    A call the identity may not make is recorded as its error code, not hidden."""
    out = {}
    calls = {'encryption': 'get_bucket_encryption', 'public_access_block': 'get_public_access_block',
             'ownership': 'get_bucket_ownership_controls', 'versioning': 'get_bucket_versioning',
             'logging': 'get_bucket_logging', 'policy_status': 'get_bucket_policy_status', 'location': 'get_bucket_location'}
    for name, call in calls.items():
        try:
            resp = getattr(s3, call)(Bucket=BUCKET)
            out[name] = {k: v for k, v in resp.items() if k != 'ResponseMetadata'}
        except s3.exceptions.ClientError as exc:
            out[name] = {'error': exc.response.get('Error', {}).get('Code')}
    return out


def migration_plan(local_root, dest_prefix, extras=(), extra_prefix=None):
    """[(local path, key, bytes, sha256)] for every file under local_root (keys keep the relative path under
    dest_prefix) and every extra file (under extra_prefix/<name>). Offline; reads the files to hash them."""
    root, plan = Path(local_root).resolve(), []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if not LocalStore.KEY.fullmatch(rel):
            raise ValueError(f'unsafe local name: {rel!r}')
        plan.append((path, f'{dest_prefix}/{rel}', path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in map(Path, extras):
        plan.append((path, f'{extra_prefix}/{path.name}', path.stat().st_size,
                     hashlib.sha256(path.read_bytes()).hexdigest()))
    return plan


def migrate(s3, plan, max_bytes, roundtrip=()):
    """Upload the plan with put_new (If-None-Match: an existing key keeps its bytes; identical bytes count as done,
    different bytes stop the run), then verify every object with HeadObject (stored SHA-256 checksum and length) and
    GET a few keys back (`roundtrip`) comparing sha256. Returns per-object records and the summary."""
    import base64
    total = sum(size for _, _, size, _ in plan)
    if total > max_bytes:
        raise SystemExit(f'plan is {total} bytes > --max-upload-bytes {max_bytes}')
    records, uploaded = [], 0
    for path, key, size, sha in plan:
        try:
            put_new(s3, key, path.read_bytes())
        except Exception as exc:  # RawExists: different bytes already there; never replaced
            raise SystemExit(f'{key}: {exc!r}; stopping, nothing was overwritten')
        head = s3.head_object(Bucket=BUCKET, Key=key, ChecksumMode='ENABLED')
        stored = base64.b64decode(head.get('ChecksumSHA256', '')).hex()
        ok = stored == sha and head['ContentLength'] == size
        records.append({'key': key, 'local': str(path), 'bytes': size, 'sha256': sha, 'verified': ok,
                        'server_side_encryption': head.get('ServerSideEncryption'),
                        'storage_class': head.get('StorageClass', 'STANDARD')})
        uploaded += size
        if not ok:
            raise SystemExit(f'{key}: stored checksum/length differ ({stored[:12]}, {head["ContentLength"]})')
    trips = []
    for key in roundtrip:
        rec = next(r for r in records if r['key'] == key)
        with s3.get_object(Bucket=BUCKET, Key=key)['Body'] as body:
            got = hashlib.sha256(body.read()).hexdigest()
        trips.append({'key': key, 'sha256_match': got == rec['sha256']})
    summary = {'objects': len(records), 'bytes': uploaded, 'verified': sum(r['verified'] for r in records),
               'encryption': dict(Counter(str(r['server_side_encryption']) for r in records)),
               'storage_class': dict(Counter(r['storage_class'] for r in records)), 'roundtrip': trips}
    return records, summary


def rules_batch_families(inventory, reservation):
    """ALIO rule families of the corpus inventory that are not in the reservation (the next non-reserved batch)."""
    reserved_ids = {json.loads(line)['doc_family_id'] for line in Path(reservation).read_text().splitlines() if line.strip()}
    with gzip.open(inventory, 'rt') as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r['doc_family_id'].startswith('institutions:alio-rule:')
            and r['doc_family_id'] not in reserved_ids], len(reserved_ids)


def homepage_host(homepage):
    """'www.sonosa.or.kr', 'http://x.go.kr/main' -> 'sonosa.or.kr', 'x.go.kr' (lower case, no www., no path)."""
    from urllib.parse import urlsplit
    host = (urlsplit(homepage if '//' in homepage else '//' + homepage).hostname or '').lower().strip('.')
    return host[4:] if host.startswith('www.') else host


def reserved_hosts(s3, reservation):
    """Hosts whose pages may carry reserved texts: alio.go.kr plus each reserved institution's ALIO homepage."""
    raw = reservation.read_bytes()
    inst = {json.loads(line)['institution_id'] for line in raw.decode().splitlines() if line.strip()}
    agencies = read_json_s3(s3, f'corpus/state/{SOURCE}/agencies.json', None)['agencies']
    by_host = {'alio.go.kr': 'ALIO'}
    for a in agencies:
        host = homepage_host(a.get('homepage') or '')
        if a['apbaId'] in inst and host:
            by_host.setdefault(host, a['apbaNa'])
    with_home = {a['apbaId'] for a in agencies if homepage_host(a.get('homepage') or '')}
    out = {'reservation': str(reservation.relative_to(ROOT)), 'reservation_sha256': hashlib.sha256(raw).hexdigest(),
           'agencies': f's3://{BUCKET}/corpus/state/{SOURCE}/agencies.json', 'institutions': len(inst),
           'institutions_without_homepage': sorted(inst - with_home), 'hosts': sorted(by_host),
           'institution_of_host': by_host,
           'use': 'provisional quarantine of web rows from these hosts until a text-overlap check against reserved texts'}
    path = ROOT / f"reports/coordination/{reservation.name.removesuffix('.jsonl')}-hosts.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({'institutions': len(inst), 'without_homepage': len(out['institutions_without_homepage']),
                      'hosts': len(out['hosts'])}))


def robots_check(client, metrics, paths):
    """One robots.txt request before a bounded run; the run starts only if every path is allowed."""
    from urllib.robotparser import RobotFileParser
    r = client.request('GET', f'{BASE}/robots.txt', metrics, max_bytes=1 << 20)
    check_response(r, 'robots.txt')
    if r.status_code != 200:
        raise Stop(f'robots.txt http_{r.status_code}')
    parser = RobotFileParser()
    parser.parse(r.text.splitlines())
    blocked = [p for p in paths if not parser.can_fetch('trid-collector', BASE + p)]
    if blocked:
        raise Stop(f'robots.txt disallows {blocked}')
    return r.text


GiB = 1 << 30


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['agencies', 'rules-index', 'reserved', 'reserved-sign-input', 'reserved-hosts',
                                      'reserved-reextract', 'reserved-migrate', 'bucket-check', 'rules-batch'])
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--max-pages', type=int)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--reservation', type=Path, default=ROOT / 'reports/coordination/benchmark-reserve-001.jsonl')
    p.add_argument('--limit', type=int)
    p.add_argument('--retry-status', nargs='*', default=[], help='re-fetch families whose state has these statuses')
    p.add_argument('--local', type=Path,
                   help='store under this folder; never S3 (reserved, reserved-sign-input, reserved-reextract only)')
    p.add_argument('--skip-acquisition', type=Path,
                   default=ROOT / 'reports/coordination/benchmark-reserve-001-acquisition.jsonl',
                   help='status JSONL of families already acquired elsewhere (the S3 run); not fetched again')
    p.add_argument('--max-bytes', type=int, default=2 * GiB, help='response bytes over all local runs')
    p.add_argument('--max-disk-bytes', type=int, default=2 * GiB, help='bytes under --local')
    p.add_argument('--max-network-failures', type=int, default=3, help='consecutive network failures that stop the run')
    p.add_argument('--sign-out', type=Path, help='reserved-sign-input --local: output file (never overwritten)')
    p.add_argument('--only-reextracted', action='store_true',
                   help='reserved-sign-input: only text produced by reserved-reextract (a delta manifest input)')
    p.add_argument('--report', help='reserved-reextract: status-file stem under reports/coordination')
    p.add_argument('--by', default='reextract', help='reserved-reextract: task id recorded on every changed record')
    p.add_argument('--profile', help='reserved-migrate / bucket-check / rules-batch: named non-root AWS profile')
    p.add_argument('--dest-prefix', default='corpus/reserved/benchmark-reserve-001/local-b026',
                   help='reserved-migrate: S3 prefix for the files under --local (reserved-only prefix)')
    p.add_argument('--extra', nargs='*', default=[], help='reserved-migrate: more files (sign inputs, manifests)')
    p.add_argument('--extra-prefix', default='corpus/reserved/benchmark-reserve-001/signatures',
                   help='reserved-migrate: S3 prefix for --extra files')
    p.add_argument('--max-upload-bytes', type=int, default=10 * GiB, help='reserved-migrate: refuse a larger plan')
    p.add_argument('--manifest-out', type=Path, help='reserved-migrate: object manifest JSONL (never overwritten)')
    p.add_argument('--roundtrip', nargs='*', default=[], help='reserved-migrate: keys to GET back and compare')
    p.add_argument('--dry-run', action='store_true', help='reserved-migrate: plan only, no AWS SDK import')
    p.add_argument('--inventory', type=Path, default=ROOT / 'data/corpus-inventory/inventory-v1/doc-families.jsonl.gz')
    p.add_argument('--root', default='corpus/raw/institutions/alio-rule-b028',
                   help='rules-batch: S3 prefix of the non-reserved batch')
    p.add_argument('--max-s3-bytes', type=int, default=GiB, help='rules-batch: bytes this run may write to S3')
    args = p.parse_args()
    metrics = Metrics()
    if args.action == 'reserved-migrate':
        if not args.local or not args.manifest_out:
            p.error('reserved-migrate needs --local and --manifest-out')
        if args.manifest_out.exists():
            p.error(f'{args.manifest_out} exists; versions are never overwritten')
        plan = migration_plan(args.local, args.dest_prefix, args.extra, args.extra_prefix)
        total = sum(size for _, _, size, _ in plan)
        if total > args.max_upload_bytes:
            p.error(f'plan is {total} bytes > --max-upload-bytes {args.max_upload_bytes}')
        if args.dry_run:
            args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_out.write_text(''.join(json.dumps({'key': k, 'local': str(lp), 'bytes': s, 'sha256': h})
                                                 + '\n' for lp, k, s, h in plan))
            print(json.dumps({'dry_run': True, 'objects': len(plan), 'bytes': total, 'max': args.max_upload_bytes,
                              'put_requests': len(plan), 'head_requests': len(plan)}))
            return 0
        s3, arn = s3_session_client(args.profile)
        records, summary = migrate(s3, plan, args.max_upload_bytes, args.roundtrip)
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode()
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_bytes(body)
        put_new(s3, f'{args.dest_prefix}/_migration-manifest.jsonl', body)
        print(json.dumps({'caller': arn, **summary}, ensure_ascii=False))
        return 0
    if args.action == 'bucket-check':
        s3, arn = s3_session_client(args.profile)
        print(json.dumps({'caller': arn, 'bucket': BUCKET, **bucket_check(s3)}, ensure_ascii=False, default=str, indent=1))
        return 0
    if args.action == 'rules-batch':
        if args.workers != 1 or args.interval < 1.0:
            p.error('rules-batch runs one request at a time with --interval >= 1.0')
        families, reserved_n = rules_batch_families(args.inventory, args.reservation)
        s3, arn = s3_session_client(args.profile)
        store = S3Store(s3, args.max_s3_bytes)
        client = Client(HostLimiter(args.interval, host='www.alio.go.kr'), ROOT / 'logs/requests-www.alio.go.kr.jsonl',
                        retry=False)
        try:
            robots_check(client, metrics, ['/occasional/findRuleDtl.json', '/download/rulefiledown.json'])
        except (Stop, NetworkError, BudgetExceeded) as exc:
            print(json.dumps({'stop': f'before families: {exc}'}, ensure_ascii=False), flush=True)
            return 2
        print(json.dumps({'caller': arn, 'families': len(families), 'reserved_excluded': reserved_n}), flush=True)
        bounded_collect(client, store, metrics, families, 'alio-rule', set(), min(args.max_bytes, GiB),
                        args.max_network_failures,
                        args.limit, tuple(args.retry_status), root=args.root, report=args.report or 'alio-rule-b028',
                        budget_key=f'{args.root}/budget.json', hold=('licence_unconfirmed',))
        return 0
    if args.local:
        if args.action not in ('reserved', 'reserved-sign-input', 'reserved-reextract'):
            p.error('--local supports reserved, reserved-sign-input and reserved-reextract only')
        store = LocalStore(args.local, args.max_disk_bytes)
        rid = args.reservation.name.removesuffix('.jsonl')
        if args.action == 'reserved-sign-input':
            if not args.sign_out:
                p.error('--sign-out is required with --local')
            return sign_input(store, rid, args.sign_out, 'reextracted' if args.only_reextracted else None)
        if args.action == 'reserved-reextract':  # offline: stored bytes only, no client is created
            if not args.report:
                p.error('--report is required with reserved-reextract')
            reextract_local(store, rid, metrics, args.report, args.by)
            return 0
        if args.workers != 1 or args.interval < 1.0:
            p.error('--local runs one request at a time with --interval >= 1.0')
        free = shutil.disk_usage(store.root).free
        need = args.max_disk_bytes - store.disk_bytes
        if free < need + GiB:
            p.error(f'free space {free} bytes < remaining disk budget {need} + 1 GiB margin')
        client = Client(HostLimiter(args.interval, host='www.alio.go.kr'), ROOT / 'logs/requests-www.alio.go.kr.jsonl',
                        retry=False)
        try:
            robots_check(client, metrics, ['/occasional/findRuleDtl.json', '/download/rulefiledown.json'])
        except (Stop, NetworkError, BudgetExceeded) as exc:
            print(json.dumps({'stop': f'before families: {exc}'}, ensure_ascii=False), flush=True)
            return 2
        reserved_local(client, store, metrics, args.reservation, args.skip_acquisition, args.max_bytes,
                       args.max_network_failures, args.limit, tuple(args.retry_status))
        return 0
    s3 = s3_client()
    client = Client(HostLimiter(args.interval, host='www.alio.go.kr'), ROOT / 'logs/requests-www.alio.go.kr.jsonl')
    if args.action == 'agencies':
        agencies(client, s3, metrics)
    elif args.action == 'rules-index':
        rules_index(client, s3, metrics, args.max_pages, args.workers)
    elif args.action == 'reserved':
        reserved(client, S3Store(s3), metrics, args.reservation, args.workers, args.limit, tuple(args.retry_status))
    elif args.action == 'reserved-hosts':
        reserved_hosts(s3, args.reservation)
    else:
        sign_input(S3Store(s3), args.reservation.name.removesuffix('.jsonl'))


if __name__ == '__main__':
    sys.exit(main())
