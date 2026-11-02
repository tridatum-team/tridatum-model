"""MODEL-RESUME-A-030: conflicting-case follow-ups through Workspace.select on v5 (no model); run from the follow-up
branch checkout: PYTHONPATH=. python check_conflicting_followup.py"""
from scripts.shared_demo import Workspace, ROOT
ws = Workspace(ROOT / 'runs/workspace-index/v5', review=ROOT / 'data/search-corpus-v4/passages.jsonl')
body = next(d for d, m in ws.meta.items() if m['title'] == '공무원 여비 규정')
for q in ['그럼 국외는?', '그럼 국외 출장 숙박비는?', '국외 출장이면 숙박비는?', '그럼 숙박비는?']:
    r = ws.select(q, context_doc=body, inherited=['국내'])
    print(q, '|', r['status'], r.get('reason'), '| continued', r.get('continued'), '| carried', r.get('conditions_carried'),
          '| conditions', r.get('conditions'), '| rows', (r.get('reviewed_rows') or {}).get('annex'), '| aside',
          [t['annex'] for t in r.get('tables_set_aside') or []])
