"""Answer presentation: a port of the old app.js renderAnswer / contextLimitLine / highlighted / placeOf.

Returns plain strings only; the _answer.html template places every value with autoescaping, so no
HTML from the model, the document or the request ever reaches the page unescaped.
"""

from __future__ import annotations

import math

# Status shown on every evidence answer: never hide that an answer is unverified or that the model failed.
STATUS = {
    'quoted': '원문 일치 · 정답 검증 아님', 'answer': '원문 위치 일치 · 정답 검증 아님', 'unverified': '검증 안 됨 · 해설만',
    'review': '검증 보류 · 원문 확인 필요', 'not_found': '근거 못 찾음', 'clarify': '되묻기',
    'model_failure': '모델 응답 실패',
}
LIMIT_PARTS = [('table_rows', lambda n: f'표 행 {n}행'), ('tables', lambda n: '검토된 표 행 전부'),
               ('linked_passages', lambda n: f'다른 별표 구간 {n}개'),
               ('own_passages', lambda n: f'관련도 낮은 구간 {n}개')]
HIDDEN_MARK = '숨겼습니다'


def _js(value) -> str:
    """String form of a JSON value as a JS template literal prints it."""
    if value is None:
        return 'null'
    if value is True or value is False:
        return 'true' if value else 'false'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _positive(value) -> bool:
    """JS `Number(value) > 0`."""
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and number > 0


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _strings(value) -> list[str]:
    return [v for v in _list(value) if isinstance(v, str)]


def _span(candidate) -> tuple[int, int]:
    prediction = candidate.get('prediction')
    span = prediction.get('source_character_span') if isinstance(prediction, dict) else None
    if isinstance(span, (list, tuple)) and len(span) >= 2 and all(type(v) is int for v in span[:2]):
        return span[0], span[1]
    return 0, 0


def context_limit_line(result: dict) -> str | None:
    counts = result.get('context_limits')
    counts = counts if isinstance(counts, dict) else {}
    parts = [label(_js(counts[key])) for key, label in LIMIT_PARTS if _positive(counts.get(key))]
    if parts:
        return f"일부 근거만 반영 ({' · '.join(parts)} 제외)"
    return '일부 근거만 반영 (문맥 한도로 일부 제외)' if _list(result.get('context_notes')) else None


def highlighted(text: str, span) -> tuple[str, str, str]:
    """(before, mark, after) by code point, as Array.from(text).slice(...) in the old page."""
    a, z = span or (0, 0)
    return text[:a], (text[a:z] if z > a else ''), text[z:]


def place_of(candidate: dict) -> str:
    label = candidate.get('source_label')
    if label:
        return _js(label)
    page = candidate.get('page')
    return f'{_js(page)}쪽' if page else ''


def present_answer(result: dict) -> dict:
    status = result.get('status')
    known = STATUS.get(status) if isinstance(status, str) else None
    tag = known or (_js(status) if status else '근거 검색')

    limits = []
    line = context_limit_line(result)
    if line:
        limits.append(line)
    sources = _strings(result.get('source_limits'))
    material = []
    if sources:
        material.append('원본 그림 제외 추출본' if any('그림' in t for t in sources) else '원본 추출 한계 있음')
    if result.get('table_warning'):
        material.append('표 행·열 대응 미검증')
    if material:
        limits.append(f"자료 한계: {' · '.join(material)} — 자세한 내용은 아래에서 확인하세요")
    notes = _strings(result.get('commentary_notes'))

    quotes = []
    for c in _list(result.get('candidates')):
        if not isinstance(c, dict) or not isinstance(c.get('text'), str):
            continue
        a, z = _span(c)
        if z <= a:
            continue
        where = place_of(c)
        before, mark, after = highlighted(c['text'], (a, z))
        quotes.append({'place': where, 'before': before, 'mark': mark, 'after': after,
                       'context': f"원문 일치 · 정답 검증 아님{' · ' + where if where else ''}"})

    info = [t for t in notes if HIDDEN_MARK not in t]
    scope = result.get('scope')
    if scope:
        info.append(_js(scope))
    for text in _strings(result.get('context_notes')):
        if not (isinstance(scope, str) and text in scope):
            info.append(text)
    info.extend(_strings(result.get('warnings')))
    if 'checked_chunks' in result:
        total = result.get('total_chunks')
        info.append(f"{_js(result['checked_chunks'])}/{'?' if total is None else _js(total)}개 구간 확인")
    if result.get('model'):
        elapsed = result.get('elapsed_ms')
        seconds = f' · {elapsed / 1000:.1f}초' if isinstance(elapsed, (int, float)) and elapsed else ''
        info.append(f"{_js(result['model'])}{seconds}")

    answer = result.get('answer')
    return {
        'answer': _js(answer) if answer else '(빈 응답)',
        'tag': tag,
        'tag_bad': status == 'model_failure',
        'limits': limits,
        'warn_notes': [t for t in notes if HIDDEN_MARK in t],
        'quotes': quotes,
        'commentary': _js(result['commentary']) if result.get('commentary') else '',
        'table_warning': _js(result['table_warning']) if result.get('table_warning') else '',
        'info': info,
    }
