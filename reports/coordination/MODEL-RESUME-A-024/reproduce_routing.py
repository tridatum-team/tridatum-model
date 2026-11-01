"""MODEL-RESUME-A-024 reproduction: public dev questions and conversations, operating code (6a5eb37) vs candidate
(69d4810), retrieval and institution routing only (no router or answer model; every turn is a search turn; CPU bge-m3).
usage (each checkout needs runs/ and data/ of the main repo):
  .venv/bin/python reproduce_routing.py <operating checkout> <candidate checkout> <out.json>"""
import gc
import importlib.util
import json
import sys
from pathlib import Path

SINGLE = ['공공기관 출장비', '그러면 공공기관의 출장비', '공공기관 출장비 지급 기준', '출장비', '출장비 기준', '국내 출장비', '출장비는 얼마야?',
          '공무원 출장비', '여비', '공무원 여비', '이전비 지급 기준', '평가위원 출장비', '출장비 업무추진비', '공무국외출장계획서 양식',
          '출장 신청서 서식', '국외출장 계획서 작성 방법', '문화체육관광부 출장비', '법제처 공무국외출장 경비', '부동산원 출장비',
          '부동산원 출장 식비 기준은?', '부동산원 평균 식비', '공공기관 직원 국내 출장 식비는 하루 얼마야?', '준정부기관 숙박비 기준',
          '공공기관 임원 업무추진비는 직위별로 얼마야?', '공기업 출장 일비', '공공기관 출장 여비', '공공기관 여비', '지방공무원 출장 식비',
          '국가공무원 국내 출장 숙박비', '지방공무원 여비 기준', '상시 출장 공무원 여비 지급 기준', '공무원이 아닌 사람 여비 지급',
          '국외 항공운임 지급 기준', '국외 가족여비 지급 기준', '국내 출장 식비는 하루 얼마인가요?', '공무원 국내 여비 지급표',
          # scope mixing / named rules (MODEL-RESUME-A-024)
          '지방공무원 여비 규정에서 숙박비는?', '국가공무원 여비 규정 숙박비', '공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?',
          '공무원 여비 규정 별표 2 식비']
CONVERSATIONS = {
    'switch-to-named-rule': ['한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', '공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?'],
    'switch-to-local-rule': ['한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', '지방공무원 여비 규정에서 숙박비는?'],
    'true-follow-up': ['한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', '그럼 일비는?'],
    'follow-up-then-switch': ['한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', '그럼 일비는?', '공무원 여비 규정에서 일비는?'],
    'shared-then-institution': ['공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?', '한국부동산원 여비규정에서 식비는?'],
    'shared-follow-up': ['공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?', '그럼 숙박비는?'],
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def turn(M, state, ws, q, ctx):
    iw = M.institution_indexes(state, ws, q, ctx)
    w = iw or ws
    sel = w.select(q, None, ctx if ctx in w.meta else None)
    rec = {'index': w.name, 'status': sel['status'], 'reason': sel.get('reason'), 'table_condition': (sel.get('table_condition') or {}).get('unstated')}
    if sel['status'] == 'answer':
        m = w.meta[sel['applied']]
        rec['applied'] = f"{m['institution']} | {m['title'][:60]}"
        rec['scope'] = (m.get('applies_to') or {}).get('scope') if isinstance(m.get('applies_to'), dict) else None
        rec['linked'] = sorted(w.meta[d]['title'].split(']')[0].split('[')[-1] for d in sel.get('linked_tables') or [])
        rec['rows_annex'] = (sel.get('reviewed_rows') or {}).get('annex')
    elif sel['status'] == 'clarify':
        rec['options'] = [f"{c['institution']}: {c['title'][:44]}" for c in sel.get('candidates', [])][:5]
    r = M.workspace_reply(w, sel) if sel['status'] != 'answer' else {'status': 'answer', 'answer': '(model)'}
    r['intent'] = 'search'
    M.population_notice(r, q, iw, w, sel)
    rec['class_notice'] = bool(r.get('population_scope'))
    return rec, (sel['applied'] if sel['status'] == 'answer' else None)


def side(path, name, embedder):
    M = load(path, name)
    root = Path(path).resolve().parents[1]
    ws = M.Workspace(root / 'runs/workspace-index/v5', review=root / 'data/search-corpus-v4/passages.jsonl', embedder=embedder)
    inst = M.Workspace(root / 'runs/workspace-index/inst-reb-002', embedder=ws.embedder)
    state = type('S', (), {})(); state.institution_workspaces = {'C0150': inst}
    single = {q: turn(M, state, ws, q, None)[0] for q in SINGLE}
    convs = {}
    for name_, turns in CONVERSATIONS.items():
        ctx, out = None, []
        for q in turns:
            rec, applied = turn(M, state, ws, q, ctx)
            out.append({'q': q, **rec})
            ctx = applied or ctx
        convs[name_] = out
    emb = ws.embedder
    domain = getattr(ws, 'domain', None)
    del ws, inst; gc.collect()
    return {'single': single, 'conversations': convs, 'domain': domain}, emb


operating, emb = side(sys.argv[1] + '/scripts/shared_demo.py', 'operating', None)
candidate, _ = side(sys.argv[2] + '/scripts/shared_demo.py', 'candidate', emb)
Path(sys.argv[3]).write_text(json.dumps({'operating': operating, 'candidate': candidate}, ensure_ascii=False, indent=1, default=str))
print('written', sys.argv[3])
