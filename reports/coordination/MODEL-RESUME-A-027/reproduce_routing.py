"""MODEL-RESUME-A-027 reproduction (A-026 set + C-027 public variants W1–W8 + population-qualified probes): public dev questions and conversations through Workspace selection and institution
routing (no router or answer model; every turn is a search turn; CPU bge-m3), for several checkouts of the service code.
Records per turn which document answers, the ask-back list, and which rule tables (별표) reach the model as evidence —
separate table documents, inline table passages of the statute body, and B's reviewed table rows.
usage (each checkout needs runs/ and data/ of the main repo):
  .venv/bin/python reproduce_routing.py <out.json> <label>=<checkout> [<label>=<checkout> ...]"""
import gc
import importlib.util
import json
import re
import sys
from pathlib import Path

# A-024 set (ROOT-009 public 36 + 4 named-rule questions), earlier public dev questions of A's retrieval reports
# (HYBRID-RAG-001, RETRIEVAL-DECISION-001, RETRIEVAL-CONTINUE-007), C-025's public scenario questions, and table
# controls (stated cases and other statutes' tables).
A024 = ['공공기관 출장비', '그러면 공공기관의 출장비', '공공기관 출장비 지급 기준', '출장비', '출장비 기준', '국내 출장비', '출장비는 얼마야?',
        '공무원 출장비', '여비', '공무원 여비', '이전비 지급 기준', '평가위원 출장비', '출장비 업무추진비', '공무국외출장계획서 양식',
        '출장 신청서 서식', '국외출장 계획서 작성 방법', '문화체육관광부 출장비', '법제처 공무국외출장 경비', '부동산원 출장비',
        '부동산원 출장 식비 기준은?', '부동산원 평균 식비', '공공기관 직원 국내 출장 식비는 하루 얼마야?', '준정부기관 숙박비 기준',
        '공공기관 임원 업무추진비는 직위별로 얼마야?', '공기업 출장 일비', '공공기관 출장 여비', '공공기관 여비', '지방공무원 출장 식비',
        '국가공무원 국내 출장 숙박비', '지방공무원 여비 기준', '상시 출장 공무원 여비 지급 기준', '공무원이 아닌 사람 여비 지급',
        '국외 항공운임 지급 기준', '국외 가족여비 지급 기준', '국내 출장 식비는 하루 얼마인가요?', '공무원 국내 여비 지급표',
        '지방공무원 여비 규정에서 숙박비는?', '국가공무원 여비 규정 숙박비', '공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?',
        '공무원 여비 규정 별표 2 식비']
C025 = ['상시 출장 공무원의 일비는 어떻게 지급되나요?', '공무원 국외 출장 숙박비 기준은?', '공무원 국내 출장 숙박비는 얼마인가요?',
        '공무원이 아닌 위원의 출장 여비는 어떻게 주나요?', '공기업 직원 출장비 기준', '공무원 여비 규정에서는 숙박비가 어떻게 되나요?',
        '공무원 여비 규정에서 국내 출장 숙박비는 얼마인가요?', '공무원 여비 규정에서 국외 출장 숙박비는 어떻게 정하나요?',
        '공무원 여비 규정에서 공무원이 아닌 사람의 여비는 어떻게 지급하나요?', '지방공무원 여비 규정에서 일비는?']
CONTROLS = ['성과상여금 지급액은 어떻게 정하나요?', '위험근무수당 지급 기준', '시간외근무수당 단가는?', '공무원 가족수당은 얼마인가요?',
            '직급보조비는 얼마인가요?', '국내 철도운임 지급 기준', '국외 항공운임은 어떻게 지급하나요?', '군사분계선 이북지역 여행 여비',
            '공무원 여비 규정에서 국외 가족여비는?', '공무원 여비 규정에서 상시 출장 여비는?', '국내 여비 지급표의 숙박비 상한',
            '공무원 일비는 얼마인가요?']
EARLIER = Path(__file__).resolve().parents[1]
DEV = []
for f in ('HYBRID-RAG-001-dev-eval.json', 'RETRIEVAL-DECISION-001-local.json', 'RETRIEVAL-CONTINUE-007/evidence.json'):
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ('question', 'q', 'query') and isinstance(v, str) and 3 <= len(v) <= 200:
                    DEV.append(v.strip())
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(json.loads((EARLIER / f).read_text()))
C026 = ['공무원 여비 규정에서 숙박비는 무엇을 기준으로 지급하나요?', '공무원 여비 규정 별표 4 숙박비 상한은 어떻게 되나요?',
        '공무원 여비 규정에서 서울로 출장 가면 숙박비 상한은?', '공무원 여비 규정에서 가족과 함께 국외로 이전할 때 가족여비는?',
        '법원공무원 여비 규칙에서 일비는 얼마인가요?', '공무원수당 규정에서 정근수당은 어떻게 주나요?', '국가공무원 복무규정에서 연가 일수는?',
        '공무원 복무규정 연가 일수', '한국부동산원 같은 공공기관 직원의 국내 출장 일비는?', '준정부기관 직원 국내 출장 숙박비는?',
        '공공기관 직원이 출장 가면 일비를 어떻게 받나요?']
# rule names: abbreviated held, truly not held, qualified forms (MODEL-RESUME-A-026)
RULE_NAMES = ['공무원수당 규정 정근수당', '지방공무원 수당 규정에서 정근수당은?', '공무원 보수 규정 봉급표', '공무원 임용 규정 시보 기간',
              '지방공무원 복무 규정 연가', '공무원 징계 규정 감봉', '국가공무원 여비 규정 숙박비', '지방공무원 여비 규정에서 일비는?',
              '서울시 출장 여비 규정 일비', '공공기관 여비 규정 일비', '교원 연수 규정 연수 기간', '공무원 행동강령 규정 선물']
C027 = ['공무원 수당 규정에서 가족수당은 얼마인가요?', '공무원 보수 규정에서 호봉 획정은 어떻게 하나요?', '국가공무원 수당 규정에서 정근수당은?',
        '국가공무원 복무 규정에서 연가 일수는?', '지방공무원 복무 규정에서 연가 일수는?', '교육공무원 여비 규정에서 일비는?',
        '교육공무원 승진규정에서 가산점은 어떻게 주나요?', '국가공무원 보수 규정 봉급', '지방공무원 보수 규정 봉급',
        '국가공무원 수당 규정 가족수당', '지방공무원 수당 규정 가족수당']
SINGLE = list(dict.fromkeys(A024 + C025 + CONTROLS + DEV + C026 + RULE_NAMES + C027))
REB = '한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?'
CONVERSATIONS = {
    'switch-to-named-rule': [REB, '공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?'],
    'switch-to-local-rule': [REB, '지방공무원 여비 규정에서 숙박비는?'],
    'true-follow-up': [REB, '그럼 일비는?'],
    'follow-up-then-switch': [REB, '그럼 일비는?', '공무원 여비 규정에서 일비는?'],
    'shared-then-institution': ['공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?', '한국부동산원 여비규정에서 식비는?'],
    'shared-follow-up': ['공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?', '그럼 숙박비는?'],
    'C025-S1': ['한국부동산원 여비규정에서 국내 출장 일비는 얼마인가요?', '그 규정에서 숙박비는요?', '공무원 여비 규정에서는 숙박비가 어떻게 되나요?',
                '그럼 식비는?', '한국부동산원 기준으로 다시, 식비는요?', '지방공무원 여비 규정에서 일비는?'],
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evidence_tables(w, sel):
    """Rule tables (별표) whose text or reviewed rows reach the model: '2', '4', … (rule-qualified when not the applied rule)."""
    out = set()
    rule = w.rule_of.get(sel['applied'])
    for c, _ in sel.get('chunks', []):
        d = str(c.doc_id)
        n = w.annex_id.get(d) if d in w.annex_id else None
        if n is None:
            for part in c.path or ():
                m = re.fullmatch(r'별표\s*(\d+(?:\s*의\s*\d+)?)', str(part).strip())
                if m:
                    n = re.sub(r'\s+', '', m.group(1))
        if c.derived and sel.get('reviewed_rows'):
            n = sel['reviewed_rows']['annex']
        if n is not None:
            out.add(n if w.rule_of.get(d, rule) == rule else f'{w.meta[d]["title"][:12]}:{n}')
    return sorted(out)


def turn(M, state, ws, q, ctx):
    iw = M.institution_indexes(state, ws, q, ctx)
    w = iw or ws
    sel = w.select(q, None, ctx if ctx in w.meta else None)
    rec = {'index': w.name, 'status': sel['status'], 'reason': sel.get('reason'),
           'table_condition': (sel.get('table_condition') or {}).get('unstated'),
           'tables_set_aside': [t['annex'] for t in sel.get('tables_set_aside') or []],
           'others': [o['title'][:44] for o in sel.get('others') or []]}
    if sel['status'] == 'answer':
        m = w.meta[sel['applied']]
        rec['applied'] = f"{m['institution']} | {m['title'][:60]}"
        rec['evidence_tables'] = evidence_tables(w, sel)
    elif sel['status'] == 'clarify':
        rec['options'] = [f"{c['institution']}: {c['title'][:44]}" for c in sel.get('candidates', [])][:6]
    r = M.workspace_reply(w, sel) if sel['status'] != 'answer' else {'status': 'answer', 'answer': '(model)'}
    r['intent'] = 'search'
    M.population_notice(r, q, iw, w, sel)
    rec['class_notice'] = bool(r.get('population_scope'))
    if sel['status'] != 'answer':
        rec['reply'] = r['answer'][:300]
    return rec, (sel['applied'] if sel['status'] == 'answer' else None)


def side(path, name, embedder):
    M = load(path, name)
    root = Path(path).resolve().parents[1]
    ws = M.Workspace(root / 'runs/workspace-index/v5', review=root / 'data/search-corpus-v4/passages.jsonl', embedder=embedder)
    inst = M.Workspace(root / 'runs/workspace-index/inst-reb-002', embedder=ws.embedder)
    state = type('S', (), {})()
    state.institution_workspaces = {'C0150': inst}
    single = {q: turn(M, state, ws, q, None)[0] for q in SINGLE}
    convs = {}
    for cname, turns in CONVERSATIONS.items():
        ctx, out = None, []
        for q in turns:
            rec, applied = turn(M, state, ws, q, ctx)
            out.append({'q': q, **rec})
            ctx = applied or ctx
        convs[cname] = out
    emb = ws.embedder
    del ws, inst
    gc.collect()
    return {'code': str(root), 'single': single, 'conversations': convs}, emb


out, emb = {}, None
for arg in sys.argv[2:]:
    label, checkout = arg.split('=', 1)
    out[label], emb = side(checkout + '/scripts/shared_demo.py', 'side_' + label.replace('-', '_'), emb)
Path(sys.argv[1]).write_text(json.dumps({'questions': len(SINGLE), 'sides': out}, ensure_ascii=False, indent=1, default=str))
print('written', sys.argv[1], len(SINGLE), 'questions')
