"""Source-quality ledger for internal-staff questions on travel expenses, meals and service/leave (DATA-CONTINUE-007,
DATA-GAPS-008).

Inputs (read only): the served indexes (`--index` v5, `--inst-index` inst-reb-002), B's hand-written ledger and contract
inputs (JSON with entries, each carrying evidence quotes). Every quote must be found verbatim (whitespace-insensitive) in
the named source, or the build stops; its char span is written out. Sources:
  v5:<doc_id> / inst:<doc_id>   document text (or its passages joined) in that index
  meta:v5:<doc_id> / meta:inst:<doc_id>   the index title of that document
  file:<path>                   a local file (tags stripped from .html)
  xml:<path>                    a law.go.kr statute XML (tags and CDATA stripped)
Outputs (never overwritten):
  source-ledger.jsonl        the reviewed sources with located evidence
  term-map.jsonl             everyday ↔ official term relations with located evidence
  applies-to-backfill.jsonl  applies_to proposals for index documents, with located evidence and the current value
  rule-conditions.jsonl      which table applies under which condition (contract key `rule_conditions`), located evidence
  form-passages.jsonl        index passages whose path ends in '서식 N' (blank forms inside rules), by the index's own path
  topic-coverage.jsonl       metadata-only inventory of topic documents (title match), with source class and dates
  qa-currency-flags.jsonl    권익위 Q&A documents that state the meal limit as 3만원 together with 음식물, by answer year
  summary.json               counts, input hashes, code hash
Metadata-derived outputs (topic-coverage, qa-currency-flags, form-passages) are diagnostics, not a review;
`--skip-diagnostics` leaves them out (a later run that only adds reviewed entries).

  python3 scripts/check_topic_sources.py --index runs/workspace-index/v5 --inst-index runs/workspace-index/inst-reb-002 \
      --ledger reports/coordination/DATA-CONTINUE-007/ledger-input.json \
      --contract reports/coordination/DATA-CONTINUE-007/contract-input.json --out reports/coordination/DATA-CONTINUE-007
  python3 scripts/check_topic_sources.py ... --ledger reports/coordination/DATA-GAPS-008/ledger-input.json \
      --contract reports/coordination/DATA-GAPS-008/contract-input.json --out reports/coordination/DATA-GAPS-008 \
      --skip-diagnostics
"""
import argparse
import collections
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
TOPICS = {
    'travel': re.compile(r'여비|출장|일비|숙박비|이전비|운임'),
    'meal': re.compile(r'식비|식대|급식|음식물|업무추진비|오찬|만찬'),
    'service_leave': re.compile(r'복무|휴가|연가|병가|공가|휴직|근무시간|초과근무|시간외|유연근무|당직'),
}
DIAGNOSTICS = ('form-passages.jsonl', 'topic-coverage.jsonl', 'qa-currency-flags.jsonl')
OUTPUTS = ('source-ledger.jsonl', 'term-map.jsonl', 'applies-to-backfill.jsonl', 'rule-conditions.jsonl') + DIAGNOSTICS + (
    'summary.json',)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def doc_text(d):
    return d.get('text') or '\n'.join(p['text'] for p in d.get('passages', []))


def source_class(kind):
    if kind in ('법령', '행정규칙', '법령 별표', '법령 서식', '기관 내규(게시본)') or kind.startswith('행정규칙 첨부'):
        return 'rule_text'
    if kind.startswith('집행내역'):
        return 'actual_spending'
    if kind in ('법령해석례', '질의회신 해석사례'):
        return 'interpretation'
    if 'FAQ' in kind or '질의응답' in kind:
        return 'qa'
    if kind == '정책브리핑 보도자료':
        return 'press'
    return 'guide'


class Sources:
    def __init__(self, index, inst):
        self.docs = {'v5': {d['doc_id']: d for d in json.loads((index / 'documents.json').read_text())},
                     'inst': {d['doc_id']: d for d in json.loads((inst / 'documents.json').read_text())}}
        self.meta = {'v5': json.loads((index / 'meta.json').read_text()),
                     'inst': json.loads((inst / 'meta.json').read_text())}
        self.cache = {}

    def text(self, ref):
        if ref in self.cache:
            return self.cache[ref]
        kind, _, rest = ref.partition(':')
        if kind in ('v5', 'inst'):
            t = doc_text(self.docs[kind][rest])
        elif kind == 'meta':
            idx, _, doc = rest.partition(':')
            t = self.meta[idx][doc]['title']
        elif kind == 'file':
            t = (ROOT / rest).read_text(encoding='utf-8', errors='replace')
            if rest.endswith('.html'):
                t = html.unescape(re.sub(r'<[^>]+>', ' ', re.sub(r'<(script|style)[^>]*>.*?</\1>', '', t, flags=re.S)))
        elif kind == 'xml':
            raw = (ROOT / rest).read_text(encoding='utf-8', errors='replace')
            t = html.unescape(re.sub(r'<[^>]+>', ' ', raw.replace('<![CDATA[', '').replace(']]>', '')))
        else:
            raise SystemExit(f'unknown source kind: {ref}')
        self.cache[ref] = t
        return t

    def locate(self, ref, quote):
        """(start, end) of `quote` in the source text, ignoring whitespace on both sides."""
        text = self.text(ref)
        keep = [i for i, ch in enumerate(text) if not ch.isspace()]
        squashed = ''.join(text[i] for i in keep)
        q = re.sub(r'\s+', '', quote)
        at = squashed.find(q)
        if at < 0:
            raise SystemExit(f'quote not found in {ref}: {quote[:60]}')
        return keep[at], keep[at + len(q) - 1] + 1


def with_spans(sources, items):
    out = []
    for item in items:
        ev = []
        for e in item['evidence']:
            a, b = sources.locate(e['source'], e['quote'])
            ev.append({**e, 'char_span': [a, b]})
        out.append({**item, 'evidence': ev})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True)
    p.add_argument('--inst-index', required=True)
    p.add_argument('--ledger', required=True)
    p.add_argument('--contract', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--skip-diagnostics', action='store_true',
                   help='write only the reviewed outputs (ledger, term map, backfill, rule conditions), not the index-wide '
                        'metadata diagnostics already written by an earlier run')
    a = p.parse_args()
    out = ROOT / a.out
    for name in OUTPUTS:
        if (out / name).exists():
            raise SystemExit(f'{out / name} exists; versions are never overwritten')
    index, inst = ROOT / a.index, ROOT / a.inst_index
    sources = Sources(index, inst)
    ledger = json.loads((ROOT / a.ledger).read_text())
    contract = json.loads((ROOT / a.contract).read_text())

    ledger_rows = with_spans(sources, ledger['entries'])
    term_rows = with_spans(sources, contract.get('term_map', []))
    rule_rows = with_spans(sources, contract.get('rule_conditions', []))
    backfill_rows = []
    for row in with_spans(sources, contract.get('applies_to_backfill', [])):
        missing = [d for d in row['doc_ids'] if d not in sources.meta['v5']]
        if missing:
            raise SystemExit(f'backfill doc ids not in the index: {missing}')
        row['current_applies_to'] = {d: sources.meta['v5'][d].get('applies_to') for d in row['doc_ids']}
        backfill_rows.append(row)

    forms, form_marker_agree, coverage, flags = [], collections.Counter(), [], []
    diagnostics = not a.skip_diagnostics
    for doc_id, d in (sources.docs['v5'].items() if diagnostics else ()):
        for ps in d.get('passages', []):
            path = ps.get('path') or []
            if path and re.match(r'서식\s*\d', path[-1]):
                marker = bool(re.match(r'\s*[\(\[]\s*별지|\s*별지\s*제', ps['text']))
                form_marker_agree[marker] += 1
                forms.append({'passage_id': ps['id'], 'doc_id': doc_id, 'path': path, 'kind': sources.meta['v5'][doc_id]['kind'],
                              'starts_with_form_marker': marker})

    for idx in (('v5', 'inst') if diagnostics else ()):
        for doc_id, m in sources.meta[idx].items():
            hits = [t for t, rx in TOPICS.items() if rx.search(m.get('title') or '')]
            if hits:
                coverage.append({'index': idx, 'doc_id': doc_id, 'title': m.get('title'), 'kind': m.get('kind'),
                                 'source_class': source_class(m.get('kind') or ''), 'institution': m.get('institution'),
                                 'date': m.get('date'), 'date_type': m.get('date_type'),
                                 'effective_date': m.get('effective_date'), 'currency': m.get('currency'),
                                 'license': m.get('license'), 'source': m.get('source'),
                                 'applies_to': m.get('applies_to'), 'topics': hits, 'review': 'metadata_only'})

    for doc_id, m in (sources.meta['v5'].items() if diagnostics else {}.items()):
        if m.get('institution') != '국민권익위원회':
            continue
        t = doc_text(sources.docs['v5'][doc_id])
        if '음식물' in t and re.search(r'3만\s*원|30,000\s*원', t):
            year = (m.get('date') or '')[:4]
            flags.append({'doc_id': doc_id, 'year': year, 'source': m.get('source'),
                          'flag': 'before_2024_limit_change' if year and year <= '2023' else 'year_2024_or_later_check_date',
                          'note': '청탁금지법 시행령 별표 1 음식물 가액은 <개정 2024. 8. 27.>로 5만원. 이 답변의 3만원은 답변 시점 기준일 수 있음',
                          'review': 'regex_and_metadata_only'})

    out.mkdir(parents=True, exist_ok=True)
    def dump(name, rows):
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    dump('source-ledger.jsonl', ledger_rows)
    dump('term-map.jsonl', term_rows)
    dump('applies-to-backfill.jsonl', backfill_rows)
    dump('rule-conditions.jsonl', rule_rows)
    if diagnostics:
        dump('form-passages.jsonl', forms)
        dump('topic-coverage.jsonl', coverage)
        dump('qa-currency-flags.jsonl', flags)
    summary = {
        'code_sha256': CODE_SHA256,
        'inputs': {'index_manifest': sha(index / 'manifest.json'), 'inst_manifest': sha(inst / 'manifest.json'),
                   'ledger': sha(ROOT / a.ledger), 'contract': sha(ROOT / a.contract)},
        'source_ledger': {'entries': len(ledger_rows), 'evidence_located': sum(len(r['evidence']) for r in ledger_rows),
                          'status': dict(collections.Counter(r['status'] for r in ledger_rows))},
        'term_map': {'entries': len(term_rows), 'relations': dict(collections.Counter(r['relation'] for r in term_rows))},
        'rule_conditions': {'entries': len(rule_rows)},
        'applies_to_backfill': {'entries': len(backfill_rows), 'documents': sum(len(r['doc_ids']) for r in backfill_rows)}}
    if diagnostics:
        summary.update({
            'form_passages': {'passages': len(forms), 'starts_with_form_marker': form_marker_agree[True],
                              'continuation_without_marker': form_marker_agree[False]},
            'topic_coverage': {'documents': len(coverage),
                               'by_topic_class': {t: dict(collections.Counter(r['source_class'] for r in coverage
                                                                              if t in r['topics'])) for t in TOPICS}},
            'qa_currency_flags': {'documents': len(flags), 'by_flag': dict(collections.Counter(f['flag'] for f in flags)),
                                  'by_year': dict(sorted(collections.Counter(f['year'] for f in flags).items()))}})
    else:
        summary['diagnostics'] = 'skipped (--skip-diagnostics)'
    summary['outputs'] = {name: sha(out / name) for name in OUTPUTS if name != 'summary.json' and (out / name).exists()}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
