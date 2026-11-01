"""Internal-work case candidates from documents already collected (MODEL-RESUME-B-024), CPU only, no model calls.

Input: a case spec written by B (question, cited articles, target with [Dn], and a structure of conclusion /
conditions / exceptions / subject / missing facts / hold reason, each condition or exception carrying a span from the
article) and a screening file (documents read this round and protected items). Nothing here is approved for training.

Sources (read only): law.go.kr statutes in data/law-ko-source (dates from the stored XML) and administrative rules in
the served index runs/workspace-index/v5 (dates from its meta). Each cited article is cut verbatim from its heading to
the next article / chapter heading ('<별표>' to the end); D1 starts with a '자료 정보' provenance line (institution,
title, document id, effective date) in the format scripts/build_case_training_view.py reads.

Document gates (any one rejects every case of that document):
  eval_id / eval_title / eval_family   the document id, its title, or its family stem (title without 시행령/시행규칙 or
                                      특례 wrappers) occurs in evaluation-side files (build_pilot_manifest.EVAL_SIDE;
                                      membership test only, nothing from those files is printed or copied)
  eval_exposure_windows                passages of the document dropped from search-corpus-v2 as evaluation_exposure
  b013_reserve_overlap                 B-013 148-family overlap lists (match or partial review)
  law_byl_hold                         held in data/law-byl/v2/status.jsonl
  protected                            protected prefixes (C decision pending)
  institution_in_eval                  an institution's own rule whose institution name occurs in evaluation-side files
                                      (reserved families include institution rules; 852 of them are unsigned)
  manual                               B's written reason in the screening file
Case gates: question prefix in evaluation-side files; same document and article already in a case training view
(duplicate); spans not verbatim in the cited articles; [Dn] out of range or a passage never cited; hold behaviours
without / answers with the '확인 불가: ' prefix; a target number that neither a cited passage nor the question holds;
a PII hint (document ids and dates removed first); a protected quote inside any text; target over 256 tokens or
sequence over 4096 tokens (CASE_SYSTEM + rag_prompt, the tokenizer A trains with); A's case contract check;
near-duplicate questions (character 3-gram Jaccard >= 0.7). Rows passing all gates are then checked against the 148
signed reserved families (scripts/check_reserved_overlap.py): a match or a partial-review row is held. The 852 unsigned
families are not checked; passing here does not make a row safe for them.

  .venv/bin/python scripts/build_work_case_candidates.py --spec reports/coordination/MODEL-RESUME-B-024/case-spec.json \
      --screen reports/coordination/MODEL-RESUME-B-024/screen-and-protect.json \
      --tokenizer runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc --out data/work-case-candidates-v1
"""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from build_case_training_view import nums  # noqa: E402
from build_pilot_manifest import eval_side_text  # noqa: E402
from trid.case_prompts import CASE_SYSTEM  # noqa: E402
from trid.korean_quality import PII_HINT  # noqa: E402
from trid.prompts import rag_prompt  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
LAW_DOCS = ROOT / 'data/law-ko-source/docs.jsonl'
RAW_LAW = ROOT / 'data/law-ko-source/raw/law'
V5 = ROOT / 'runs/workspace-index/v5'
DROPPED = ROOT / 'data/search-corpus-v2/dropped.jsonl'
B013 = [ROOT / 'reports/coordination/B-013/overlap-law-admrul-expc-all-148.json',
        ROOT / 'reports/coordination/B-013/overlap-pack-documents-148.json']
VIEWS = [ROOT / 'data/case-training-view-v2', ROOT / 'data/case-training-view-v2r1']
RESERVED = ROOT / 'data/benchmark-reserve-001/signatures-partial-148.jsonl'
HOLD = {'hold_missing_basis', 'hold_time', 'lookup_needed'}
PREFIX = '확인 불가: '
NEXT = re.compile(r'\n\s*제\s*\d+\s*조(?:의\s*\d+)?\s*\(|\n\s*제\s*\d+\s*[절장]\s|\n부칙|\n\[별|\n<별')
squash = lambda s: re.sub(r'\s+', '', s or '')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def article(text, heading):
    """The article from its heading to the next article / chapter / 부칙 / 별표 heading; '<별표>' runs to the end."""
    if text.startswith(heading):
        start = 0
    else:
        idx = text.find('\n' + heading)
        if idx < 0:
            raise KeyError(heading)
        start = idx + 1
    if heading.startswith('<별표'):
        return text[start:].strip()
    nxt = NEXT.search(text[start + len(heading):])
    return text[start:start + len(heading) + nxt.start()].strip() if nxt else text[start:].strip()


def stem(title):
    t = re.sub(r'^특정조달을 위한 ', '', title)
    return re.sub(r' (시행령|시행규칙)( 특례규정| 특례규칙)?$', '', t)


def load_sources(doc_ids):
    """{doc: {title, text, institution, effective (YYYY-MM-DD), source, license, text_sha256, kind}}"""
    out = {}
    laws = {d for d in doc_ids if d.startswith('law-go-kr:law:')}
    if laws:
        with LAW_DOCS.open() as f:
            for line in f:
                if not line.startswith('{"id": "law-go-kr:law:'):
                    continue
                if line[8:8 + 40].split('"')[0] not in laws:
                    continue
                r = json.loads(line)
                xml = (RAW_LAW / f"{r['id'].rsplit(':', 1)[1]}.xml").read_text(errors='ignore')
                eff = re.search(r'<시행일자>(?:<!\[CDATA\[)?(\d{8})', xml).group(1)
                out[r['id']] = dict(title=r['title'], text=r['text'], institution='법령', kind='법령',
                                    effective=f'{eff[:4]}-{eff[4:6]}-{eff[6:]}', date_basis='시행일자(원문 XML)',
                                    source=r['source'], license=r['license'], text_sha256=r.get('sha256'))
    rules = {d for d in doc_ids if d.startswith('law-go-kr:admrul:')}
    if rules:
        meta = json.loads((V5 / 'meta.json').read_text())
        for d in json.loads((V5 / 'documents.json').read_text()):
            if d['doc_id'] in rules:
                m = meta[d['doc_id']]
                eff = m.get('effective_date') or ''
                text = d.get('text') or '\n'.join(p['text'] for p in d['passages'])
                out[d['doc_id']] = dict(title=m['title'], text=text, institution=m.get('institution'), kind=m.get('kind'),
                                        effective=f'{eff[:4]}-{eff[4:6]}-{eff[6:]}' if len(eff) == 8 else None,
                                        date_basis='시행일자(색인 meta effective_date)', source=m.get('source'),
                                        license=m.get('license'), text_sha256=hashlib.sha256(text.encode()).hexdigest())
    return out


def doc_gates(doc, title, institution, own_rule, ctx):
    g = []
    ev = ctx['eval']
    if doc in ev:
        g.append('eval_id')
    if title and title in ev:
        g.append('eval_title')
    elif title and stem(title) != title and stem(title) in ev:
        g.append('eval_family')
    if ctx['dropped'].get(doc):
        g.append('eval_exposure_windows')
    if doc in ctx['b013']:
        g.append('b013_reserve_overlap')
    if doc in ctx['byl_held']:
        g.append('law_byl_hold')
    if any(doc.startswith(p) for p in ctx['protected_prefixes']):
        g.append('protected')
    if own_rule and institution and institution in ev:
        g.append('institution_in_eval')
    return g


def protected_quotes(cfg):
    quotes = []
    for q in cfg['quote_rows']:
        lines = (ROOT / q['file']).read_text().splitlines()
        rows = [json.loads(lines[q['line'] - 1])] if 'line' in q else [json.loads(x) for x in lines if json.loads(x).get('id') == q['id']]
        if len(rows) != 1:
            raise SystemExit(f"protected row not found exactly once: {q}")
        quotes += [(q, e['quote']) for e in rows[0].get('evidence', []) if e.get('quote')]
    return quotes


def trigrams(s):
    s = squash(s)
    return {s[i:i + 3] for i in range(len(s) - 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--screen', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-answer-tokens', type=int, default=256)
    p.add_argument('--max-total-tokens', type=int, default=4096)
    a = p.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    spec = json.loads((ROOT / a.spec).read_text())
    screen = json.loads((ROOT / a.screen).read_text())
    cfg = screen['protected']

    ctx = {'eval': eval_side_text(), 'dropped': collections.Counter(), 'b013': set(), 'byl_held': set(),
           'protected_prefixes': cfg['doc_prefixes']}
    for line in DROPPED.read_text().splitlines():
        r = json.loads(line)
        if r.get('reason') == 'evaluation_exposure':
            ctx['dropped'][r['doc_id']] += 1
    for path in B013:
        r = json.loads(path.read_text())
        ctx['b013'] |= {m['corpus_id'] for m in r.get('matches', [])} | {m['corpus_id'] for m in r.get('partial_review', [])}
    for line in (ROOT / cfg['law_byl_status']).read_text().splitlines():
        r = json.loads(line)
        if r['status'] == 'held':
            ctx['byl_held'].add(r['id'])
    pquotes = protected_quotes(cfg)

    case_docs = {c['doc'] for c in spec['cases']}
    screened = {s['doc']: s for s in screen['screened']}
    sources = load_sources(case_docs | set(screened))
    # institution own rules (snapshot records) carry their institution; checked, never used as cases here
    for s in screen['screened']:
        if s.get('snapshot'):
            for line in (ROOT / s['snapshot'] / 'normalized/records-00000.jsonl').read_text().splitlines():
                r = json.loads(line)
                did = f"{r['source_id'].rsplit(':', 1)[0]}:{r['version']}"
                if did == s['doc']:
                    sources[did] = dict(title=r['title'], institution=r['issuer']['name'], own_rule=True, text=r['text'])
    doc_report = []
    for d in sorted(set(screened) | case_docs):
        s = sources.get(d, {})
        own = bool(s.get('own_rule')) or d.startswith('reb.or.kr:')
        inst = s.get('institution') or screened.get(d, {}).get('institution')
        gates = doc_gates(d, s.get('title'), inst, own, ctx)
        if screened.get(d, {}).get('manual_exclusion'):
            gates.append('manual')
        doc_report.append({'doc': d, 'title': s.get('title'), 'institution': inst, 'found': bool(s),
                           'gates': gates, 'manual_reason': screened.get(d, {}).get('manual_exclusion'),
                           'cases_in_spec': sum(c['doc'] == d for c in spec['cases'])})
    doc_gate = {r['doc']: r['gates'] for r in doc_report}

    existing = collections.defaultdict(set)   # doc -> article headings already used by a case training view
    for v in VIEWS:
        for part in ('train', 'dev', 'held'):
            for line in (v / f'{part}.jsonl').read_text().splitlines():
                r = json.loads(line)
                if r.get('doc'):
                    heads = {m for c in r['context'] for m in re.findall(r'제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)]*\)', c)}
                    existing[r['doc']] |= heads or {'*'}

    from transformers import AutoTokenizer  # noqa: E402  (local files only)
    tok = AutoTokenizer.from_pretrained(str(ROOT / a.tokenizer), local_files_only=True)

    rows, held = [], []
    seen = []
    for c in spec['cases']:
        problems = []
        src = sources.get(c['doc'])
        if doc_gate.get(c['doc']):
            problems += [f'doc:{g}' for g in doc_gate[c['doc']]]
        if not src:
            held.append({'id': c['id'], 'doc': c['doc'], 'held': sorted(set(problems + ['source_missing']))})
            continue
        try:
            arts = [article(src['text'], h) for h in c['articles']]
        except KeyError as e:
            held.append({'id': c['id'], 'doc': c['doc'], 'held': sorted(set(problems + [f'article_missing:{e}']))})
            continue
        prov = (f"자료 정보: {src['institution']} 「{src['title']}」 (문서 {c['doc']}) · 기준 {src['effective']} 시행 조문"
                f"(이후 개정 여부 미확인)\n")
        context = [prov + arts[0]] + arts[1:]
        q, t = c['question'], c['target']
        if q[:40] in ctx['eval']:
            problems.append('question_in_eval_side')
        used = existing.get(c['doc'], set())
        if '*' in used or any(squash(h) in {squash(x) for x in used} for h in c['articles']):
            problems.append('duplicate_of_existing_view')
        allart = squash('\n'.join(arts))
        st = c['structure']
        for kind in ('conditions', 'exceptions'):
            for label, span in st[kind]:
                if squash(span) not in allart:
                    problems.append(f'span_not_verbatim:{kind}:{label}')
        if st['subject'][1] and squash(st['subject'][1]) not in allart:
            problems.append('span_not_verbatim:subject')
        cited = [int(n) for n in re.findall(r'\[D(\d+)\]', t)]
        if any(n < 1 or n > len(context) for n in cited):
            problems.append('citation_out_of_range')
        if set(range(1, len(context) + 1)) - set(cited):
            problems.append('passage_never_cited')
        if (c['behavior'] in HOLD) != t.startswith(PREFIX):
            problems.append('hold_prefix_mismatch')
        allowed = nums(q) | set().union(*(nums(context[n - 1]) for n in set(cited) if 1 <= n <= len(context)))
        stray = sorted(nums(re.sub(r'\[D\d+\]', '', t)) - allowed)
        if stray:
            problems.append(f'ungrounded_number:{",".join(stray)}')
        scan = re.sub(r'law-go-kr:[a-z-]+:\d+|\d{4}-\d{2}-\d{2}', ' ', '\n'.join([q, t] + context))
        pii = len(PII_HINT.findall(scan))
        if pii:
            problems.append(f'pii_hint:{pii}')
        text_all = squash('\n'.join([q, t] + context))
        hits = [qq['why'] for qq, quote in pquotes if len(squash(quote)) >= 8 and squash(quote) in text_all]
        if hits:
            problems.append('protected_quote')
        n = len(tok(t + tok.eos_token, add_special_tokens=False)['input_ids'])
        user = rag_prompt([(f'D{i}', x) for i, x in enumerate(context, 1)], q)
        prompt = tok.apply_chat_template([{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        total = len(tok(prompt, add_special_tokens=False)['input_ids']) + n
        if n > a.max_answer_tokens:
            problems.append('over_answer_budget')
        if total > a.max_total_tokens:
            problems.append('over_context_budget')
        tg = trigrams(q)
        dup = [sid for sid, sg in seen if len(tg & sg) / max(1, len(tg | sg)) >= 0.7]
        if dup:
            problems.append(f'near_duplicate_question:{dup[0]}')
        seen.append((c['id'], tg))
        spans = []
        for h, art in zip(c['articles'], arts):
            i = src['text'].find(art)
            spans.append({'heading': h, 'char_start': i, 'char_end': i + len(art)})
        row = {
            'id': c['id'], 'split': 'candidate', 'split_group': c['doc'], 'task_type': 'statute_application',
            'behavior': c['behavior'], 'question': q, 'context': context, 'target': t,
            'answer_tokens': n, 'total_tokens': total,
            'content_sha256': hashlib.sha256((q + '\n' + t).encode()).hexdigest(),
            'time_label': f"statute text in force from {src['effective'].replace('-', '')}; later amendments unverified",
            'institution': src['institution'], 'doc': c['doc'],
            'candidate': {
                'status': 'candidate_not_approved', 'domain': c['domain'], 'structure': st,
                'source': {'doc': c['doc'], 'title': src['title'], 'kind': src['kind'], 'url': src['source'],
                           'license': src['license'], 'text_sha256': src['text_sha256'], 'articles': spans},
                'time': {'effective_from': src['effective'], 'basis': src['date_basis'], 'current': 'not labelled current'},
                'family': {'stem': stem(src['title']), 'doc_in_existing_case_view': bool(used)},
                'pii': {'hint_hits': pii, 'check': 'trid.korean_quality.PII_HINT after removing document ids and dates'},
                'review': 'B (AI) wrote question, target and structure; gates are mechanical; not human-reviewed, '
                          'not quality-confirmed, not approved for training'}}
        (held if problems else rows).append({**row, 'held': sorted(set(problems))} if problems else row)

    out.mkdir(parents=True)
    overlap_in = out / 'overlap-input.jsonl'
    overlap_in.write_text(''.join(json.dumps({'id': r['id'], 'group': r['doc'], 'source': r['doc'],
                                              'text': '\n'.join([r['question'], *r['context'], r['target']])},
                                             ensure_ascii=False) + '\n' for r in rows))
    report = out / 'overlap-148.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_reserved_overlap.py'), 'check', '--manifest', str(RESERVED),
                    '--corpus', str(overlap_in), '--report', str(report), '--workers', '4'], check=True,
                   stdout=subprocess.DEVNULL)
    rep = json.loads(report.read_text())
    flagged = {m['corpus_id']: 'reserved_overlap_match' for m in rep.get('matches') or []}
    flagged.update({m['corpus_id']: 'reserved_overlap_partial' for m in rep.get('partial_review') or []})
    kept = [r for r in rows if r['id'] not in flagged]
    held += [{**r, 'held': [flagged[r['id']]]} for r in rows if r['id'] in flagged]

    def dump(name, items):
        body = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items).encode()
        (out / name).write_bytes(body)
        return {'rows': len(items), 'sha256': hashlib.sha256(body).hexdigest()}

    files = {'candidates': dump('candidates.jsonl', kept), 'held': dump('held.jsonl', held),
             'documents': dump('documents.jsonl', doc_report)}
    reasons = collections.Counter(h.split(':')[0] if not h.startswith('doc:') else h for x in held for h in x['held'])
    manifest = {
        'version': out.name, 'code_sha256': CODE_SHA256, 'status': 'candidates only; not approved for training',
        'inputs': {'spec': {'path': a.spec, 'sha256': sha(ROOT / a.spec)}, 'screen': {'path': a.screen, 'sha256': sha(ROOT / a.screen)},
                   'v5_manifest': sha(V5 / 'manifest.json'), 'law_docs': str(LAW_DOCS.relative_to(ROOT)),
                   'reserved_manifest': {'path': str(RESERVED.relative_to(ROOT)), 'sha256': sha(RESERVED)},
                   'tokenizer': a.tokenizer},
        'files': files,
        'counts': {'cases_in_spec': len(spec['cases']), 'candidates': len(kept), 'held': len(held),
                   'held_reasons': dict(reasons),
                   'candidates_by_domain': dict(collections.Counter(r['candidate']['domain'] for r in kept)),
                   'candidates_by_behavior': dict(collections.Counter(r['behavior'] for r in kept)),
                   'documents_screened': len(doc_report),
                   'documents_with_candidates': len({r['doc'] for r in kept}),
                   'documents_rejected': {r['doc']: r['gates'] for r in doc_report if r['gates']},
                   'answer_tokens_max': max((r['answer_tokens'] for r in kept), default=0),
                   'total_tokens_max': max((r['total_tokens'] for r in kept), default=0)},
        'overlap_148': {k: rep.get(k) for k in ('reserved_families', 'corpus_rows_scanned', 'matched_corpus_rows', 'partial_review_rows')},
        'not_checked': '852 unsigned reserved families; human review; legal correctness beyond verbatim spans',
        'review': 'B (AI) authored; mechanical gates only; not quality-confirmed'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=1))
    for x in held:
        print('HELD', x['id'], x['held'])


if __name__ == '__main__':
    main()
