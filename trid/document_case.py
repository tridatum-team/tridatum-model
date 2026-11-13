"""Answering a question from an attached document with a CASE-trained model, shared by the test site and the on-prem
server. The model call and the context fitting are injected, so this module needs neither torch nor numpy."""
import re
from typing import Callable

from trid.case_prompts import CASE_SYSTEM
from trid.pdf_documents import DocumentError, retrieve
from trid.prompts import rag_prompt

QA_ANSWER = re.compile(r'\n답변\s*:')


def evidence_start(text):
    """Offset where admissible evidence begins. In a Q&A passage ("질문: … 답변: …") the asker's question is not a
    source: values and quotes count only from the answer part on; a passage holding only the question has none."""
    if text.startswith('질문'):
        m = QA_ANSWER.search(text)
        return m.end() if m else len(text)
    return 0



CONTEXT_OVERFLOW = ('근거 구간 하나가 모델 문맥 한도보다 길어 답하지 못했습니다. 서버 측 근거 구성의 한계이며, 원문 링크에서 직접 '
                    '확인해 주세요.')


LEADING_TAGS = re.compile(r'^(?:\s*(?:\[[^\]\n]{1,12}\]|D[1-9]\s*:)\s*)+')
NUMBER = re.compile(r'\d[\d,.]*')
MIN_QUOTE_CHARS = 4


def _numbers(text):
    return {n.rstrip('.,').replace(',', '') for n in NUMBER.findall(text)}



NORMALIZE = re.compile(r'[\s·ㆍ・「」『』()（）\[\]〈〉<>,.\-]')


def _norm(text):
    return NORMALIZE.sub('', text or '')


def answer_document_case(document, question, base, family, context, max_tokens=256, selected=None, topical=None,
                         provenance=None, *, complete: Callable, fit_context: Callable):
    """Serving contract of CASE-trained versions (explanatory answers, trained with CASE_SYSTEM). The model is asked
    with its training instruction; its output is never shown as a verified answer. Output sentences that occur verbatim
    in a retrieved passage and contain one of the question's `topical` words (when given) are shown as quotes,
    highlighted in the source: a quote proves the wording exists in the source, not that it answers the question.
    The rest is `commentary` explicitly marked unverified; sentences with numbers that appear in no retrieved passage
    nor the question are withheld. Repeated sentences are collapsed."""
    question = question.strip()
    if not question:
        raise DocumentError('질문을 입력해 주세요.')
    selected = [(chunk, score) for chunk, score in (retrieve(document, question) if selected is None else selected)
                if '[D' not in chunk.text]
    result = {'file': document.name, 'candidates': [], 'checked_chunks': len(selected), 'answer_mode': 'case',
              'total_chunks': len(document.chunks), 'warnings': document.warnings, 'raw_output': None,
              'commentary': None, 'commentary_verified': False, 'unsupported_numbers': [],
              'scope': '관련도 상위 최대 6개 구간을 확인한 결과입니다. 미검색 구간까지 정보가 없다고 보장하지 않습니다.'}
    if not selected:
        return {**result, 'status': 'not_found', 'answer': '질문과 관련된 문서 구간을 찾지 못했습니다. 질문을 바꾸거나 추출 내용을 확인해 주세요.'}
    # Versions trained on B's case-training-view (MODEL-REPAIR-001) saw D1 start with a '자료 정보: …' line naming the
    # document, institution and date basis; they are served the same way (`provenance`), others without it.
    def build(sel):
        texts = [chunk.text for chunk, _ in sel]
        if provenance:
            texts[0] = provenance + texts[0]
        prompt = rag_prompt([(f'D{i}', text) for i, text in enumerate(texts, 1)], question)
        return [{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': prompt}]
    selected, fit_notes, limits = fit_context(selected, build, base, context, max_tokens)
    result.update(checked_chunks=len(selected), context_notes=fit_notes, context_limits=limits)
    r = complete(base, build(selected), family, max_tokens, context)
    if r.get('skipped_context'):
        raise DocumentError(CONTEXT_OVERFLOW)
    output = r['generated']
    # Echoing the question or the instructions back is a failed answer (a known defect of the pilot), not a quote.
    flat = _norm(output)
    echoed = (len(_norm(question)) >= 8 and _norm(question) in flat) or any(
        _norm(text) in flat for text in ('제공된 근거로만 답하세요', '근거에 있는 결론과 그 이유', '근거에 없는 내용은 추측하지'))
    if echoed or output.lstrip().startswith('질문:'):
        return {**result, 'raw_output': output, 'status': 'model_failure',
                'answer': '모델이 질문이나 지시문을 그대로 옮겨 적어 답변에 실패했습니다. 이 출력은 표시하지 않습니다 '
                          '(모델 출력 결함).'}
    segments, seen, repeats = [], set(), 0
    for part in re.split(r'(?<=[.!?])\s+|\n+', output):
        part = part.strip()
        if not part:
            continue
        if part in seen:
            repeats += 1
            continue
        seen.add(part)
        segments.append(part)
    candidates, quoted = [], set()
    # A length-truncated last sentence is a fragment; it can match the source verbatim but is not a full quote.
    for part in segments[:-1] if r.get('truncated_output') and segments else segments:
        quote = LEADING_TAGS.sub('', part).strip()
        for text in (quote, quote.rstrip('.')):
            if len(text) < MIN_QUOTE_CHARS or text in quoted or (topical and not any(t in text for t in topical)):
                continue
            hit = next(((i, chunk, chunk.text.find(text, evidence_start(chunk.text))) for i, (chunk, _) in enumerate(selected)
                        if text in chunk.text[evidence_start(chunk.text):]
                        and not getattr(chunk, 'derived', False)  # built from reviewed rows, not source text
                        and not (getattr(chunk, 'value_hold', False) and _numbers(text))), None)
            if hit:
                i, chunk, start = hit
                quoted.add(text)
                candidates.append({'page': chunk.page, 'text': chunk.text, 'page_text_start': chunk.start,
                                   'retrieval_score': round(selected[i][1], 3),
                                   'page_character_span': [chunk.start + start, chunk.start + start + len(text)],
                                   'prediction': {'answer': text, 'source': f'D{i + 1}', 'abstains': False, 'presence': 0,
                                                  'source_character_span': [start, start + len(text)]}})
                break
    held = [chunk for chunk, _ in selected if getattr(chunk, 'value_hold', False)]
    support = _numbers('\n'.join(chunk.text[evidence_start(chunk.text):] for chunk, _ in selected
                                 if not getattr(chunk, 'value_hold', False)) + '\n' + question)
    notes, shown, withheld = [], [], 0
    for part in segments:
        # A sentence carrying a number found in no retrieved passage nor the question is withheld, not just flagged.
        if _numbers(part) - support:
            withheld += 1
        else:
            shown.append(part)
    if withheld:
        notes.append(f'검색된 원문과 질문에 없는 숫자가 들어 있는 해설 문장 {withheld}개를 숨겼습니다.')
    if repeats:
        notes.append(f'같은 문장 반복 {repeats}회를 생략했습니다.')
    if held:
        notes.append(f'숫자 추출이 원문 쪽과 일치하지 않는 구간 {len(held)}개를 썼습니다. 이 구간의 숫자는 답 값으로 쓰지 않으니 '
                     '원문(쪽)에서 직접 확인하세요.')
    if r.get('truncated_output'):
        notes.append(f'모델 출력이 길이 한도({max_tokens}토큰)에서 잘렸습니다.')
    result.update(raw_output=output, commentary='\n'.join(shown) or None, commentary_notes=notes,
                  unsupported_numbers=sorted(_numbers(output) - support), candidates=candidates)
    if candidates:
        return {**result, 'status': 'quoted',
                'answer': f'원문과 글자가 일치하는 문장 {len(candidates)}건을 아래에 표시합니다. 질문에 대한 답인지는 원문 '
                          '맥락에서 확인하세요. 모델 해설은 검증되지 않았습니다.'}
    return {**result, 'status': 'unverified',
            'answer': '원문과 글자 그대로 일치하는 인용이 없어 검증된 답이 없습니다. 아래 모델 해설은 검증되지 않았습니다.'}


def fit_own_passages(selected, build: Callable, count_tokens: Callable, context: int, max_tokens: int):
    """(selected, notes, limits) for an attached document alone: passages are dropped from the end (weakest first, the
    best passage stays) until the prompt `build(selected)` counted by `count_tokens` leaves `max_tokens` for the answer
    within `context`. `count_tokens` returning None (cannot be asked) keeps the passages. `limits` has the same keys as
    the test site's fit_context."""
    selected, dropped = list(selected), 0
    while len(selected) > 1:
        tokens = count_tokens(build(selected))
        if tokens is None or tokens + max_tokens <= context:
            break
        selected.pop()
        dropped += 1
    notes = [f'모델 문맥 한도로 적용 문서의 관련도 낮은 구간 {dropped}개를 빼었습니다.'] if dropped else []
    return selected, notes, {'linked_passages': 0, 'table_rows': 0, 'tables': 0, 'own_passages': dropped}
