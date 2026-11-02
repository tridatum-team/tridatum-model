"""C-008 final six-type candidates: 100 accepted items per type (fixed seed) from the full run and the carried-over
C-007 pilot items, as a C-002 spec for scripts/build_task_pilot.py; the remaining accepted items are kept as a reserve."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

FULL, PILOT = Path('/tmp/c008/full'), Path('/tmp/c008/pilot2')
OUT = Path(sys.argv[1])


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(path):
    return {r['cid']: r for r in jsonl(path)}


pool = []
for work, keep in ((FULL, lambda c: True), (PILOT, lambda c: c.startswith('pilot60:'))):
    verified = latest(work / 'verified.jsonl')
    built = {b['cid']: b for b in jsonl(work / 'built.jsonl')}
    tables = json.loads((work / 'spec-tables.json').read_text())
    for cid, v in verified.items():
        if keep(cid) and v['state'] == 'accepted':
            b = built[cid]
            table = tables['tables'][b['item']['table']] if b['type'] == 'table' else None
            pool.append(dict(cid=cid, type=b['type'], item=b['item'], table=table, distractors=tables['distractors']))
final, reserve = [], []
for kind in sorted({p['type'] for p in pool}):
    items = sorted((p for p in pool if p['type'] == kind), key=lambda p: hashlib.sha256(('c008-final' + p['cid']).encode()).hexdigest())
    final += items[:100]
    reserve += items[100:]
docs = set()
for p in final:
    i = p['item']
    docs |= {i['doc']} if 'doc' in i else {x[0] for x in i.get('parts', [])} | ({p['table']['doc']} if p['table'] else set())
spec = dict(purpose='C-008 six-type training candidates (100 per type): model-drafted by Kanana-1.5-2.1B with numbered verbatim-span '
                    'selection (C-007 pilot items assistant-drafted), C-002 builder checks, independently verified by Qwen3-1.7B '
                    '(answer from the documents alone + judgment + content agreement). Not human-reviewed. Training candidates, '
                    'not an evaluation set.',
            source='data/korea-kr-briefing-source/docs.jsonl', domains={},
            dev_rule="sha256('c007-dev'+doc_id) mod 10 == 0 -> valid",
            dev_docs=sorted(d for d in docs if int(hashlib.sha256(('c007-dev' + d).encode()).hexdigest(), 16) % 10 == 0),
            distractors=final[0]['distractors'],
            tables={p['item']['table']: p['table'] for p in final if p['table']},
            items=[dict(p['item'], id=p['cid'].replace(':', '-')) for p in final])
for item in spec['items']:
    item.pop('span_repair', None)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'final-spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1))
(OUT / 'final-cids.json').write_text(json.dumps([p['cid'] for p in final], ensure_ascii=False))
(OUT / 'reserve-cids.json').write_text(json.dumps([p['cid'] for p in reserve], ensure_ascii=False))
print(json.dumps(dict(pool=dict(Counter(p['type'] for p in pool)), final=dict(Counter(p['type'] for p in final)),
                      reserve=len(reserve), documents=len(docs),
                      pilot60_in_final=sum(p['cid'].startswith('pilot60') for p in final)), ensure_ascii=False))
