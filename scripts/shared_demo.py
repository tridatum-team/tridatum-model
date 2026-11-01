"""Temporary multi-visitor demo. Only accepts requests from the authenticated edge proxy."""
import argparse
from dataclasses import dataclass, field, replace
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import hashlib
import html
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit, unquote

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external_chat import (GRAMMAR, MAX_TOKENS, QWEN3_NO_THINK, SYSTEM, complete, digest, evidence_grammar,
                                    request as llama_request)
from trid.case_prompts import CASE_SYSTEM
from trid.extractive import load_reader, read as read_extractive, passage_spans
from trid.prompts import rag_prompt
from trid.pdf_documents import Document, DocumentError, make_chunks, answer_document, retrieve

ROOT = Path(__file__).resolve().parents[1]
TTL = 15 * 60
MAX_BODY = 2 * 1024 * 1024
MAX_SESSIONS = 32
# Source-first answers, one or two pairs like the value-first GRAMMAR: "[D1] 값" or "[D1] 값, [D2] 값".
SOURCE_FIRST_ANSWER = re.compile(r'\[(D[1-9])\]\s*([^\[\],\n]+?)(?:\s*,\s*\[(D[1-9])\]\s*([^\[\],\n]+?))?\.?')
QA_ANSWER = re.compile(r'\n답변\s*:')


def evidence_start(text):
    """Offset where admissible evidence begins. In a Q&A passage ("질문: … 답변: …") the asker's question is not a
    source: values and quotes count only from the answer part on; a passage holding only the question has none."""
    if text.startswith('질문'):
        m = QA_ANSWER.search(text)
        return m.end() if m else len(text)
    return 0


def read_chunk(model, _presence, tok, text, question, device):
    """Adapt prompt-relative extractive spans to the document reader's chunk offsets."""
    prompt = rag_prompt([('D1', text)], question)
    result = read_extractive(model, tok, prompt, device)
    abstains = result['presence'] != 0
    span = None
    answer = result['answer']
    if not abstains:
        sid, lo, hi = passage_spans(prompt)[0]
        start, end = result['span']
        if result['source'] != sid or not lo <= start < end <= hi:
            raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
        span = [start - lo, end - lo]
        answer = result['span_text']
        if text[span[0]:span[1]] != answer:
            raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
        if span[0] < evidence_start(text):  # copied from the asker's question in a Q&A passage
            abstains, span, answer = True, None, '확인 불가'
    return {**result, 'answer': answer, 'abstains': abstains, 'source_character_span': span}


def prompt_tokens(base, messages):
    """Tokens of the prompt as the serving model renders and tokenizes it (same /apply-template + /tokenize calls as
    `complete`), or None when the server cannot be asked."""
    try:
        prompt = llama_request(base, '/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
        return len(llama_request(base, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens'])
    except OSError:
        return None


def fit_context(selected, build, base, context, max_tokens):
    """(selected, notes, limits): evidence reduced until the prompt `build(selected)` — counted with the serving model's own
    template and tokenizer — leaves `max_tokens` for the answer within `context`. Reduction order: passages of other
    documents (linked tables); then whole reviewed-table rows, one at a time (never part of a row and never a note: a
    cut condition could change a value's meaning; the rendered rows say how many were left out); then the applied
    document's weakest passages (its best passage stays). `limits` counts what was left out (the API's `context_limits`:
    linked_passages, table_rows, tables — row blocks dropped whole —, own_passages; all four keys, 0 when nothing)."""
    selected, dropped = list(selected), {'linked': 0, 'rows': 0, 'tables': 0, 'own': 0}
    while selected:
        tokens = prompt_tokens(base, build(selected))
        if tokens is None or tokens + max_tokens <= context:
            break
        applied = selected[0][0].doc_id
        linked = [i for i, (c, _) in enumerate(selected) if c.doc_id != applied and not c.derived]
        derived = [i for i, (c, _) in enumerate(selected) if c.derived]
        own = [i for i, (c, _) in enumerate(selected) if c.doc_id == applied and not c.derived][1:]
        if linked:
            del selected[linked[-1]]
            dropped['linked'] += 1
        elif derived:
            i = derived[-1]
            chunk, score = selected[i]
            n = (len(chunk.rows) if chunk.shown < 0 else chunk.shown) - 1
            if n < 1:
                del selected[i]
                dropped['tables'] += 1
            else:
                selected[i] = (replace(chunk, text=render_rows(chunk.rows, n), shown=n), score)
                dropped['rows'] += 1
        elif own:
            del selected[own[-1]]
            dropped['own'] += 1
        else:
            break
    notes = []
    if dropped['linked']:
        notes.append(f"모델 문맥 한도로 함께 넣으려던 다른 별표 구간 {dropped['linked']}개를 빼었습니다.")
    if dropped['rows']:
        notes.append(f"모델 문맥 한도로 검토된 표 행 {dropped['rows']}행을 빼었습니다. 이 답은 넣은 행에 한정됩니다.")
    if dropped['tables']:
        notes.append('모델 문맥 한도로 검토된 표 행을 넣지 못했습니다. 표 값은 원문 표에서 확인하세요.')
    if dropped['own']:
        notes.append(f"모델 문맥 한도로 적용 문서의 관련도 낮은 구간 {dropped['own']}개를 빼었습니다.")
    limits = {'linked_passages': dropped['linked'], 'table_rows': dropped['rows'], 'tables': dropped['tables'],
              'own_passages': dropped['own']}
    return selected, notes, limits


CONTEXT_OVERFLOW = ('근거 구간 하나가 모델 문맥 한도보다 길어 답하지 못했습니다. 서버 측 근거 구성의 한계이며, 원문 링크에서 직접 '
                    '확인해 주세요.')

def answer_document_chat(document, question, base, family, context, constrained=False, selected=None):
    """One completion over the retrieved chunks (as D1..Dn), with the evaluation's `grammar_instructed` system prompt
    and optionally its `evidence_constrained` grammar. `value [Dn]` and source-first `[Dn] value` are accepted (the
    evaluation's content_* reading); an answer is shown only when every value is found verbatim in the chunk it
    cites, otherwise the reply is a non-answer and the raw output is not displayed as an answer."""
    question = question.strip()
    if not question:
        raise DocumentError('질문을 입력해 주세요.')
    selected = [(chunk, score) for chunk, score in (retrieve(document, question) if selected is None else selected)
                if '[D' not in chunk.text]
    result = {'file': document.name, 'candidates': [], 'checked_chunks': len(selected),
              'total_chunks': len(document.chunks), 'warnings': document.warnings, 'raw_output': None,
              'scope': '관련도 상위 최대 6개 구간을 확인한 결과입니다. 미검색 구간까지 정보가 없다고 보장하지 않습니다.'}
    if not selected:
        return {**result, 'status': 'not_found', 'answer': '질문과 관련된 문서 구간을 찾지 못했습니다. 질문을 바꾸거나 추출 내용을 확인해 주세요.'}
    build = lambda sel: [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': rag_prompt([(f'D{i}', c.text) for i, (c, _) in enumerate(sel, 1)], question)}]
    selected, fit_notes, limits = fit_context(selected, build, base, context, MAX_TOKENS)
    result.update(checked_chunks=len(selected), context_notes=fit_notes, context_limits=limits)
    prompt = rag_prompt([(f'D{i}', chunk.text) for i, (chunk, _) in enumerate(selected, 1)], question)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    r = complete(base, messages, family, MAX_TOKENS, context, evidence_grammar(prompt) if constrained else None)
    if r.get('skipped_context'):
        raise DocumentError(CONTEXT_OVERFLOW)
    output = r['generated']
    result['raw_output'] = output
    if output.removesuffix('.') == '확인 불가':
        return {**result, 'status': 'not_found', 'answer': '확인한 문서 구간에서 답을 찾지 못했습니다(모델 보류).'}
    if output.removesuffix('.') == '근거 충돌':
        return {**result, 'status': 'review', 'answer': '모델이 근거 충돌로 판단했습니다. 원문을 직접 확인해 주세요.'}
    match = GRAMMAR.fullmatch(output)
    source_first = None if match else SOURCE_FIRST_ANSWER.fullmatch(output)
    pairs = ([(match[1], match[2]), (match[3], match[4])] if match
             else [(source_first[2], source_first[1]), (source_first[4], source_first[3])] if source_first else [])
    candidates = []
    for value, sid in pairs:
        if value is None:
            continue
        value, index = value.strip(), int(sid[1:]) - 1
        chunk = selected[index][0] if index < len(selected) else None
        start = chunk.text.find(value, evidence_start(chunk.text)) if chunk and value else -1
        if start < 0:
            return {**result, 'status': 'review', 'answer': '모델 출력이 원문 위치 검증을 통과하지 못해 답으로 표시하지 않습니다.'}
        candidates.append({'page': chunk.page, 'text': chunk.text, 'page_text_start': chunk.start,
                           'retrieval_score': round(selected[index][1], 3),
                           'page_character_span': [chunk.start + start, chunk.start + start + len(value)],
                           'prediction': {'answer': value, 'source': sid, 'abstains': False, 'presence': 0,
                                          'source_character_span': [start, start + len(value)]}})
    if not candidates:
        return {**result, 'status': 'review', 'answer': '모델 출력이 형식 검증을 통과하지 못해 답으로 표시하지 않습니다.'}
    return {**result, 'status': 'answer', 'candidates': candidates,
            'answer': ', '.join(f"{c['prediction']['answer']} [{c['prediction']['source']}]" for c in candidates)}


LEADING_TAGS = re.compile(r'^(?:\s*(?:\[[^\]\n]{1,12}\]|D[1-9]\s*:)\s*)+')
NUMBER = re.compile(r'\d[\d,.]*')
MIN_QUOTE_CHARS = 4


def _numbers(text):
    return {n.rstrip('.,').replace(',', '') for n in NUMBER.findall(text)}


def answer_document_case(document, question, base, family, context, max_tokens=256, selected=None, topical=None,
                         provenance=None):
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



@dataclass
class Session:
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    touched: float = field(default_factory=time.monotonic)
    document: object = None
    document_id: str | None = None
    calls: list = field(default_factory=list)
    pinned_doc: str | None = None
    offered_docs: list = field(default_factory=list)
    history: list = field(default_factory=list)  # [{'role', 'content', 'kind'}] of this browser session only


# Session memory (cleared by /clear): the last HISTORY_MESSAGES user+assistant messages. The conversation model gets
# every one of them that fits its context (oldest dropped first); a fixed window of 6 messages cut the visitor's
# first sentence ("견과류 알레르기가 있어요") out of the 5th turn while 479 of 4096 tokens were used (GENERAL-CHAT-002).
HISTORY_MESSAGES = 24
ROUTER_SYSTEM = (
    '당신은 입력 분류기입니다. 사용자의 마지막 입력을 처리할 방법을 하나 고르세요.\n'
    '검색: 공공기관의 규정·지침·법령·질의회신 문서에 근거해야 답할 수 있는 업무 사실(금액, 기한, 비율, 절차, 자격, '
    '기준, 처분의 효력, 해석)을 묻는 경우. 앞선 업무 질문을 이어서 묻는 경우도 포함합니다.\n'
    '대화: 인사, 감사 표현, 이 서비스가 무엇을 할 수 있는지에 대한 문의, 잡담, 의견, 글쓰기·문장 다듬기·요약 요청, '
    '앞선 대화 내용에 대한 질문처럼 기관 문서를 찾지 않아도 되는 경우.\n'
    '분류 예: 인사말 → 대화 / 서비스 사용법 문의 → 대화 / 문장 다듬기 요청 → 대화 / 업무 기준 금액 문의 → 검색 / '
    '징계 처분의 효력 문의 → 검색 / 앞선 업무 질문에 이어 다른 직급·사례를 묻는 말 → 검색.\n'
    '직전 대화가 주어지면 함께 보되, 분류하는 것은 마지막 입력입니다. 직전이 일반 대화이고 마지막 입력이 그 내용을 '
    '고르기·고치기·줄이기·덧붙이기처럼 이어 가는 말이면 대화입니다. 직전이 문서 검색이고 마지막 입력이 다른 직급·금액·'
    '사례를 이어 묻는 말이면 검색입니다.\n'
    '답은 "검색" 또는 "대화" 한 단어만 쓰세요.')
ROUTER_GRAMMAR = 'root ::= "검색" | "대화"'
# General conversation (GENERAL-CHAT-001): the prompt only sets the role and the manner. Any instruction about rules
# or institutions here made the 1.7B model send everyday talk ("커피 몇 잔", "점심 추천") to "기관 규정을 확인하세요";
# work facts are kept out by the router and by the rule-claim check on the reply (RULE_CLAIM), not by the prompt.
CHAT_SYSTEM = (
    '당신은 TRID입니다. 공공기관 직원과 이야기하는 한국어 AI 비서입니다. 사용자의 말에 공감하고, 앞선 대화를 이어서 '
    '자연스럽고 구체적으로 답하세요. 대화 도중에 인사나 자기소개를 반복하지 마세요. 한국어로 답하세요.')
# Qwen3's recommended non-thinking sampling for conversation (greedy decoding made the 1.7B model repeat its previous
# answer on follow-ups); classification and document answers stay greedy.
CHAT_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5)
EXPLAIN_SYSTEM = (
    '당신은 TRID입니다. 공공기관 업무 용어·개념·일반적인 요령을 쉬운 한국어로 짧게 설명하세요. 특정 기관 규정의 금액·'
    '한도·기한·조문 번호는 말하지 말고, 어느 기관의 규정이나 자료에 따른 설명인 것처럼 기관명을 내세우지 마세요. 구체적인 '
    '기준은 기관명이나 규정명을 넣어 물으면 공개 자료에서 근거를 찾는다고 한 문장으로 안내하세요. 반드시 한국어로만 답하세요.')
# A digit followed by a unit (money, rates, periods, articles) …
UNSOURCED_FACT = re.compile(r'\d[\d,.]*\s*(?:만\s*(?:\d[\d,.]*\s*)?)?(?:천\s*)?(?:원|%|퍼센트|일|시간|개월|년|주|조|항|호|배|명|회)')
# … in a sentence about work rules, rates or entitlements is a rule claim that needs a retrieved source, unless every
# number in it was given by the visitor. Everyday figures ("커피 2잔", "30분 낮잠", "11월 3일부터 휴가") are not.
RULE_CLAIM = re.compile(r'규정|규칙|법령|법률|시행령|조례|훈령|예규|지침|고시|조문|제\s*\d+\s*조|지급|수당|여비|출장|식비|숙박비|'
                        r'일비|운임|한도|상한|하한|기준액|단가|요율|세율|세금|과세|공제|과태료|벌금|징계|감봉|정직|견책|연가|병가|'
                        r'공가|특별휴가|휴가\s*일수|휴직|근속|호봉|봉급|보수|급여|임금|퇴직|연금|보험료|업무추진비|예산|공무원|'
                        r'공공기관|기관')
# Request types the 1.7B model does not tell apart reliably (measured in reports/coordination/TRID-ANSWER-002.md):
# markers of the request itself, never of a particular question. Visitor-given numbers + an organise/compute verb →
# conversation; actual-spending/statistics words → structured values, not rule passages; definition words → a general
# explanation that cites no institution.
GIVEN_VALUES = re.compile(r'\d[\d,.]*\s*(?:만\s*)?(?:원|%|일|시간|개월|년|박|명|회|시|분|km)')
ORGANIZE = re.compile(r'정리|요약|계산|합계|합치|총\s*얼마|표로|한\s*줄|더하면')
STATISTIC = re.compile(r'평균|통계|집행|현황|추이|실제로?\s*(?:쓴|쓰는|사용|지출)|실지출|건수')
COMPUTED = re.compile(r'평균|통계|추이|합계|총\s*(?:액|얼마|지출)|건수|인당')  # a figure computed over records
DEFINITION = re.compile(r'뜻|의미|개념|차이|정의|이란\b|란\s*(?:무엇|뭐)|구성(?:되|돼|요소|은|이)|요령')
# A question about what the visitor said earlier is answered from the conversation, not searched.
RECALL_VERB = (r'(?:말한|말씀드린|얘기한|알려\s*드린|알려\s*준|적은|물어본|물어봤|물었던|질문한|답한|답해\s*준|대답한|'
               r'말해\s*준|찾아\s*준|인용한|보여\s*준|근거로\s*(?:쓴|든|삼은)|뭐라고\s*(?:했|답했|말했))')
RECALL = re.compile(rf'(?:아까|방금|앞에서|이전에|처음에|전에|마지막에)\s*(?:제가|내가|네가|니가)?\s*{RECALL_VERB}'
                    rf'|(?:제가|내가|네가|니가)\s*(?:아까\s*|방금\s*)?(?:뭐|무엇을?|뭘|어떤\s*걸?)?\s*{RECALL_VERB}'
                    r'|(?:아까|방금|앞에서|처음에)\s[^?.!]{0,30}?(?:다고|라고|냐고)\s*(?:했|말했|물었|그랬)')
ASKING = re.compile(r'\?|얼마|몇|무엇|뭐|어떻게|언제|어디|누가|나요|까요|인가|습니까|는지|알려|궁금')
# A question asking for an amount or its standard (단가·상한·기준), as opposed to how a case is handled.
AMOUNT_ASKED = re.compile(r'얼마|금액|단가|상한|한도|지급액|기준액|몇\s*원|기준(?:은|이|을|이\s*뭐|\s*알려)')
# A guideline article sending the amount to another instrument's annex ("규정 별표1의 상한액", "공무원여비규정
# 별표2") defers to that instrument: the name written before it (empty for a bare "규정"), its kind and the annex number.
DEFERS_TO = re.compile(r'(?<![가-힣])([가-힣]*?)(규정|규칙)\s*\[?\s*별표\s*(\d+)')


def request_type(question):
    """'organize' | 'recall' | 'statistic' | 'explain' | None (see the marker comments above)."""
    if GIVEN_VALUES.search(question) and (ORGANIZE.search(question) or not ASKING.search(question)):
        return 'organize'  # the visitor's own figures: to organise, or just stated ("숙박비 8만원 들었어요")
    if RECALL.search(question):
        return 'recall'
    if STATISTIC.search(question):
        return 'statistic'
    if DEFINITION.search(question):
        return 'explain'
    return None


AMOUNT = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(?:(만)\s*(?:(\d[\d,]*)\s*(천)?)?|(천))?')


def _values(text):
    """Numeric values in text with Korean units read ("8만 원", "80,000원" → 80000; "2만5천원" → 25000)."""
    out = set()
    for digits, man, rest, rest_thousand, thousand in AMOUNT.findall(text):
        try:
            value = float(digits.replace(',', ''))
            if man:
                value = value * 10000 + (float(rest.replace(',', '')) * (1000 if rest_thousand else 1) if rest else 0)
            elif thousand:
                value *= 1000
        except ValueError:
            continue
        out.add(round(value, 6))
    return out


def _derived(values):
    """Results of plain arithmetic on the visitor's own numbers (sum of all; sum, difference, product of any two) —
    computed here, never by evaluating text — so a correct total of their figures is not mistaken for an unsourced fact."""
    vals = sorted(values)[:30]
    out = {round(sum(vals), 6)}
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            out |= {round(a + b, 6), round(abs(a - b), 6), round(a * b, 6)}
    return out


def _messages(system, history, user):
    return ([{'role': 'system', 'content': system}] + [{'role': h['role'], 'content': h['content']} for h in history]
            + [{'role': 'user', 'content': user}])


def _classify(base, family, context, system, grammar, question, before=''):
    r = complete(base, [{'role': 'system', 'content': system}, {'role': 'user', 'content': f'{before}분류할 입력: {question}'}],
                 family, 4, context, grammar)
    return (r.get('generated') or '').strip()


def route(base, family, context, history, question, continues=None):
    """'search' or 'chat', decided by the serving model itself under a two-word grammar (no keyword rules). The input
    is classified as one input — given a transcript, the 1.7B model continues the conversation instead of classifying.
    After a conversation turn the previous user input is stated above it: without it "그 중에 하나만 골라줘", "두
    문장으로 줄여줘" read as search (GENERAL-CHAT-001: 6 → 2 errors of 38 labelled inputs). After a search turn it is
    not: then "출장 다니면서 체력 관리는?" or "휴가 가면 어디가 좋을까?" were pulled into search. Short follow-ups of a
    search ("그럼 감봉은?") can read as chat to the model; an input that the previous turn's document would answer
    (`continues(question, doc_id)`) is a follow-up."""
    users = [h for h in history if h['role'] == 'user']
    last = next((h for h in reversed(history) if h['role'] == 'assistant'), None)
    after_chat = users and not (last and last.get('kind') == 'search')
    before = f"직전 사용자 입력: {users[-1]['content'][:200]}\n직전 응답 종류: 일반 대화\n" if after_chat else ''
    if _classify(base, family, context, ROUTER_SYSTEM, ROUTER_GRAMMAR, question, before) == '검색':
        return 'search'
    if last and last.get('kind') == 'search' and continues and continues(question, last.get('applied_id')):
        return 'search'
    return 'chat'


def chat_complete(base, messages, family, max_tokens, context):
    """Conversation completion: the model's native template (Qwen3 without thinking) and CHAT_SAMPLING.
    {'generated', 'truncated_output'}; {'skipped_context': True} when the prompt leaves no room for `max_tokens`."""
    prompt = llama_request(base, '/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    if family == 'qwen3' and not prompt.endswith(QWEN3_NO_THINK):
        raise ValueError('Native Qwen3 no-thinking prefix was not rendered')
    tokens = llama_request(base, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
    if len(tokens) + max_tokens > context:
        return {'skipped_context': True}
    c = llama_request(base, '/completion', dict(prompt=tokens, n_predict=max_tokens, cache_prompt=False,
                                                 seed=secrets.randbelow(2 ** 31), **CHAT_SAMPLING))
    return {'generated': c['content'].strip(), 'truncated_output': c['stop_type'] == 'limit'}


def conversation_turns(history, recall):
    """The earlier turns the conversation model is given. Search exchanges are left out — given "인사혁신처 기준으로요"
    and its document note, the 1.7B model went on to invent 휴가 rules in "휴가 쓰면 어디 가면 좋을까요?" — except when
    the question is about the conversation itself, which may be about those turns (stored as their document and
    verified quotes only)."""
    return history if recall else [h for h in history if h.get('kind') != 'search']


def chat_reply(base, family, context, history, question, max_tokens=384, explain=False, first_turn=None):
    """General conversation (or, with `explain`, a general explanation citing no institution) with the model's native
    chat template and the given earlier turns — all of them that fit the context, the oldest dropped first. Not
    grounded: a sentence making a rule claim — a number with a unit in a sentence about rules, rates or entitlements
    (RULE_CLAIM) — is withheld unless every number in it was given by the visitor in this conversation (their own facts
    may be repeated or organised); everyday figures stay. On the first turn of a conversation (`first_turn`, default:
    no turns given) the model is told there is nothing earlier, so "제 이름이 뭐라고 했죠?" after a new conversation
    is not answered with a claimed memory."""
    system = EXPLAIN_SYSTEM if explain else CHAT_SYSTEM
    if (not history) if first_turn is None else first_turn:
        system += ' 이 대화는 지금 막 시작되어 앞선 내용이 없습니다. 사용자가 앞에서 한 말을 물으면 아직 들은 적이 없다고 답하세요.'
    turns = list(history)
    while True:
        r = chat_complete(base, _messages(system, turns, question), family, max_tokens, context)
        if not r.get('skipped_context') or not turns:
            break
        turns = turns[2:]
    output = (r.get('generated') or '').strip()
    given_text = ' '.join([question] + [h['content'] for h in history if h['role'] == 'user'])
    given, given_values = _numbers(given_text), _values(given_text)
    derived = _derived(given_values) - given_values
    # Sentences end at . ! ? not preceded by a digit (list numbers like "1." stay attached) or at line breaks; the
    # model's own layout (lists, line breaks) is kept and only withheld sentences are removed.
    pieces = re.split(r'((?<=[^\d][.!?])[ \t]+|\n)', output)
    kept, withheld, computed = [], 0, 0
    for i in range(0, len(pieces), 2):
        values = _values(pieces[i])
        own = _numbers(pieces[i]) <= given or values <= given_values
        calculated = not own and values and values <= given_values | derived
        claim = UNSOURCED_FACT.search(pieces[i]) and RULE_CLAIM.search(pieces[i])
        if claim and not own and not calculated:
            withheld += 1
        else:
            computed += bool(calculated and UNSOURCED_FACT.search(pieces[i]))
            kept.append(pieces[i] + (pieces[i + 1] if i + 1 < len(pieces) else ''))
    notes = []
    if withheld:
        notes.append(f'근거 없이 업무 기준(금액·기한·비율·조문)을 말한 문장 {withheld}개를 숨겼습니다(사용자가 말한 숫자와 그 계산 결과는 '
                     '숨기지 않음). 업무 기준은 기관명이나 규정명을 넣어 질문해 주세요.')
    if computed:
        notes.append(f'사용자가 준 숫자를 더하거나 빼거나 곱한 값과 일치하는 문장 {computed}개를 그대로 보였습니다. 계산 결과는 직접 '
                     '확인하세요.')
    if r.get('truncated_output'):
        notes.append(f'응답이 길이 한도({max_tokens}토큰)에서 잘렸습니다.')
    text = re.sub(r'\n{3,}', '\n\n', ''.join(kept)).strip() or '죄송합니다. 이 질문에는 근거 없이 답할 수 있는 내용이 없습니다.'
    return {'status': 'chat', 'answer': text, 'answer_mode': 'explain' if explain else 'chat', 'candidates': [],
            'checked_chunks': 0, 'commentary_notes': notes, 'warnings': [],
            'scope': ('일반 설명입니다. 특정 기관 규정을 검색하지 않았고 검증 대상이 아닙니다.' if explain else
                      '일반 대화 응답입니다. 문서 근거를 검색하지 않았고 검증 대상이 아닙니다.')}


def no_history_reply():
    """A question about earlier turns when this session has none: said so without the model (which would invent a
    conversation). With earlier turns the question is ordinary conversation over them (chat_reply)."""
    return {'status': 'chat', 'answer_mode': 'chat', 'candidates': [], 'checked_chunks': 0, 'commentary_notes': [],
            'warnings': [], 'answer': '이 대화에는 앞서 나눈 이야기가 아직 없어요. 이전 대화나 다른 사람의 대화는 볼 수 없습니다.',
            'scope': '이 브라우저 세션의 대화 기록만 확인했습니다. 모델을 실행하지 않았습니다.'}


def statistic_reply(workspace, question):
    """Actual-spending / statistics questions: answered only from structured values (`workspace.facts`, B's
    structured_facts with basis actual_spending|statistic), aggregated here, never from rule passages or embeddings.
    Without such values the reply says so and separates them from rule payment limits."""
    named, unknown, _, _ = workspace.constraints(question) if workspace else ([], [], None, None)
    facts = workspace.matching_facts(question, named + unknown) if workspace else []
    base = {'candidates': [], 'checked_chunks': 0, 'warnings': [], 'reason': 'statistic',
            'scope': '정형 값(통계·실지출)만 조회했습니다. 규정 문서나 임베딩 유사도로 평균을 계산하지 않습니다.'}
    if facts:
        unit = facts[0]['unit']
        values = [f['value'] for f in facts if f['unit'] == unit]
        lines = '\n'.join(f"- {f['subject']['name']} · {f['metric']} {f['value']:,}{unit} ({f['period']['from']}~{f['period']['to']}, "
                          f"{f['source_id']})" for f in facts if f['unit'] == unit)
        mean = sum(values) / len(values)
        return {**base, 'status': 'answer_values', 'facts': facts,
                'answer': f'정형 값 {len(values)}건의 평균은 {mean:,.0f}{unit}입니다(단순 평균).\n{lines}'}
    who = ''
    if named or unknown:
        who = f"{', '.join(named + unknown)}의 "
    return {**base, 'status': 'not_found',
            'answer': (f'이 질문은 {who}실제 지출의 평균·통계를 묻는 것으로 보입니다. 검색 자료에는 실제 지출 통계 값이 없어 답할 수 '
                       '없습니다. 규정에 정해진 지급 기준(정액·상한)은 실제 평균 지출과 다른 값입니다. 규정상 지급 기준을 '
                       "찾으려면 '…출장 식비 지급 기준'처럼 물어 주세요.")}


@dataclass(frozen=True)
class PassageChunk:
    """A workspace passage in the shape the answer functions expect (text/page/start), plus its provenance."""
    text: str
    page: int
    start: int
    doc_id: str
    path: tuple
    table: bool = False  # passage carries a table whose rows/columns B has not verified
    value_hold: bool = False  # B value review: its extracted numbers differ from the source page; not answer values
    derived: bool = False  # built here from reviewed structure (not source text): never a verbatim quote or value span
    rows: tuple = ()  # derived chunk: the reviewed rows it renders (so the context fit can drop whole rows)
    shown: int = -1  # derived chunk: how many of `rows` are rendered (-1: all)


def reader_windows(selected, tok):
    """Workspace passages cut into the reader's token windows (trid.pdf_documents.make_chunks: 128 tokens, 32 overlap),
    keeping provenance and absolute offsets. Uploaded documents are already chunked this way; whole workspace passages
    (B passages run to 1,600+ characters) overflow the 91M reader's context, which raised ValueError → HTTP 422."""
    out = []
    for chunk, score in selected:
        for w in make_chunks([chunk.text], tok):
            out.append((PassageChunk(w.text, chunk.page, chunk.start + w.start, chunk.doc_id, chunk.path,
                                     chunk.table, chunk.value_hold), score))
    return out


# Reviewed table rows (B: one row per line with caption/columns/cells/unit/notes/dates/review) and how each row
# document maps onto the indexed text: B's link file (law-byl rows → the same law-byl document, or the statute body
# `law-go-kr:law:<MST>` span of that annex), or the row's own `source` (index_doc_id, content_hash, char_span) for an
# institution snapshot. Only rows B compared with a rendering by eye (ai_visual_review_match) are used.
TABLE_ROWS = [(ROOT / 'data/law-byl/v2-structure-003-rows-001/queryable-rows.jsonl',
               ROOT / 'reports/coordination/B-030/byl-index-links.jsonl'),
              (ROOT / 'reports/coordination/B-029/reb-yeobi-reviewed-rows.jsonl', None)]
REVIEWED = 'ai_visual_review_match'
# B's DATA-CONTINUE-007 domain contract, pinned by sha256 (a changed file stops the service at startup): every term
# relation and applicability record carries verbatim source quotes that B's scripts/check_topic_sources.py located.
#   term-map.jsonl: an everyday word and the official term it equals under a stated condition ("equivalent_when",
#     e.g. 출장비 → 여비: 공무 여행의 운임·일비·숙박비·식비) or the cases where it does not (not_equivalent …).
#   applies-to-backfill.jsonl: applies_to of documents whose own record is missing (a rule's 별표 without its body).
DOMAIN_CONTRACT = ROOT / 'reports/coordination/DATA-CONTINUE-007'
TERM_MAP = (DOMAIN_CONTRACT / 'term-map.jsonl', '531b654b373f5208273fcba2fcdaeebfca091c93c3631d700e12dd0875303897')
APPLIES_BACKFILL = [(DOMAIN_CONTRACT / 'applies-to-backfill.jsonl',
                     'f649524b3acb243b41b993b2b16250c5aee120c38786fa66f89b5e0c07288de4'),
                    # DATA-GAPS-008: 선관위·방사청 국제계약지원단 (institution), 행안부 복식부기·교육비특별회계
                    # (local_government). Rows 1-3 quote v5 text verbatim; row 4 (교육비 "1. 목적") quotes a span the index
                    # leaves out for evaluation exposure — promoted as applies_to metadata only, verbatim at this sha
                    # (MODEL-RESUME-C-024 1.2); the span itself stays unindexed, and a changed quote needs C's judgment again.
                    (ROOT / 'reports/coordination/DATA-GAPS-008/applies-to-backfill.jsonl',
                     '4f45cbf3c63d279f626ee769ecbdf63b8f7c4009c1b96b1913d8407378ce5827')]
# A form (서식) passage — the index path's last part is "서식 N" (B: the index's own structure, not a word list) — is a
# blank template: what to fill in, never a rule, an amount or a standard. It is evidence only for a question asking
# for a form; otherwise documents are judged by their rule text (DATA-CONTINUE-007 T04).
FORM_PATH = re.compile(r'서식\s*\d+(?:\s*의\s*\d+)?')
FORM_ASKED = re.compile(r'양식|서식|별지|신청서|계획서')


PINNED_CONTRACT = (TERM_MAP, *APPLIES_BACKFILL)  # checked by scripts/eval_artifact_preflight.py (present, tracked, sha)


def pinned_jsonl(spec):
    """Rows of a pinned B file. A missing or changed file stops the service: a contract that silently switched itself
    off would serve different retrieval than the one evaluated (the C-030 untracked-artifact failure). Callers that
    build a Workspace without the contract pass None."""
    if not spec:
        return []
    if not Path(spec[0]).exists():
        raise ValueError(f'{spec[0]}: pinned data contract is missing; refusing to serve')
    data = Path(spec[0]).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec[1]:
        raise ValueError(f'{spec[0]}: sha256 differs from the pinned contract; refusing to serve')
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


NORMALIZE = re.compile(r'[\s·ㆍ・「」『』()（）\[\]〈〉<>,.\-]')


def _norm(text):
    return NORMALIZE.sub('', text or '')


def _names_at_word(question, name):
    """True when the normalised rule `name` occurs in `question` starting a word — not as the tail of a longer word:
    "지방공무원 여비 규정" does not name "공무원 여비 규정" (a different body's rule). Characters NORMALIZE drops may sit
    between its letters ("공무원 여비 규정", "「공무원여비규정」")."""
    gap = NORMALIZE.pattern + '*'
    return re.search(r'(?<![0-9A-Za-z가-힣])' + gap.join(map(re.escape, name)), question) is not None


def _annex(text):
    """Annex id as printed: '6의2', '2' (spaces dropped)."""
    return re.sub(r'\s+', '', text)


def _lines(path):
    """The lines of `Path(path).read_text().splitlines()`, read a line at a time instead of holding the whole file and
    its line list (SITE-MEMORY-023): the same text decoding and newline translation, then the same splitlines()
    boundaries within each read line."""
    with open(path) as f:
        for raw in f:
            yield from raw.splitlines()


def render_rows(rows, shown=-1):
    """Reviewed table rows (B review ai_visual_review_match) as evidence text: each table's caption, unit, revision and
    statute dates, one line per row pairing every value with its column header, then the table's notes in full —
    numbers never travel without their conditions, unit and date. With `shown` ≥ 0 only that many rows are rendered
    and the text says how many rows were left out and that the answer is limited to the rows given; notes are never
    cut (a cut condition could change a value's meaning)."""
    rows = list(rows)
    kept = rows if shown < 0 else rows[:shown]
    out = []
    for table in dict.fromkeys(r['table'] for r in rows):
        part = [r for r in rows if r['table'] == table]
        head = part[0]
        unit = re.sub(r'^\(?\s*단위\s*:\s*|\)$', '', head.get('unit') or '').strip()
        when = (f"법령 시행일 {head['statute_effective_date']}" if head.get('statute_effective_date') else
                f"규정 시행일 {head['rule_effective_date']}(게시본, 현행 여부 미확인)" if head.get('rule_effective_date') else
                '시행일 표기 없음')
        out += [f"[검토된 표 행 · B AI 시각 대조, 승인·법률 자문 아님] {' '.join(head['caption'])}",
                f"단위: {unit or '표에 표기 없음'} · {head.get('revised') or '개정 표기 없음'} · {when}"]
        out += [f'조건(표 앞 문장): {c}' for c in head.get('conditions') or []]
        section = None
        for r in part:
            if not any(r is k for k in kept):
                continue
            if r.get('section') and r['section'] != section:
                section = r['section']
                out.append(f'[{section}]')
            out.append('- ' + '; '.join(f'{k}: {v}' for k, v in r['cells'].items()))
        left = sum(1 for r in part if not any(r is k for k in kept))
        if left:
            out.append(f'(이 표의 나머지 {left}행은 모델 문맥 한도로 넣지 않았습니다. 이 답은 넣은 행에 한정되며, 나머지는 원문 표를 '
                       '확인하세요.)')
        notes = list(dict.fromkeys(n for r in part for n in r.get('notes') or []))
        if notes:
            out.append(('' if notes[0].startswith('비고') else '비고: ') + ' / '.join(notes))
    return '\n'.join(out)

def _alias_ok(alias):
    return len(alias) >= 3 or bool(re.fullmatch(r'[A-Z]{2}', alias))



def _longest_shared(a, b):
    """Length of the longest substring of `a` that also occurs in `b` (both normalised)."""
    best = 0
    for i in range(len(a)):
        for j in range(len(a), i + best, -1):
            if a[i:j] in b:
                best = j - i
                break
    return best


class Workspace:
    """Questions without an uploaded document search the shared index of approved public sources
    (scripts/build_workspace_index.py). Selection uses the question's own metadata before ranking:
      1. an institution named in the question restricts the search to that institution's documents; when the corpus
         has none, the reply says so (no fallback to another institution);
      2. a document title (or a long part of one) named in the question restricts it to those documents;
      3. otherwise, when the best matches belong to different institutions with comparable scores, the reply asks in
         plain language which institution's rule is meant instead of picking one.
    Each answer uses passages of ONE document, so rules of different institutions are never mixed."""
    MAX_PASSAGES = 4  # CPU prompt processing dominates latency; four 550-char passages keep answers short
    TITLE_MIN = 8      # normalised characters shared with a title to count as naming that document
    CLOSE = 0.85       # another institution's best score within this ratio of the top makes the question ambiguous
    # Hybrid ranking (HYBRID-RAG-001 dev measurement, 14 labelled questions: dense ranks the target document far
    # higher than BM25, whose kiwi split of questions like "식비는" → 식/비 loses the key word; 0.1 keyword + 0.9
    # meaning had the best doc@1/@5). Authority: laws and their tables apply generally and come first; rules, manuals
    # and official interpretations next; FAQs; citizens' Q&A boards answer one asker's situation and come last.
    LEXICAL_WEIGHT = 0.1

    @staticmethod
    def authority_of(kind):
        kind = kind or ''
        if kind.startswith('법령') and '해석' not in kind:
            return 1.0
        if '질의응답' in kind or '사내 FAQ' in kind:
            return 0.7
        return 0.85 if 'FAQ' in kind else 0.9

    # Evidence must be about the question before any model runs. Dev calibration (best-passage bge-m3 cosine): 0.62–0.79
    # for answerable questions, 0.48–0.61 for questions the corpus does not cover, so cosine alone cannot separate them.
    # Below RELEVANT_COS nothing passes; between it and STRONG_COS the passages must also contain one of the
    # question's topical nouns (the uncovered questions' nouns — 아파트 관리비, 동호회 활동비 — occur nowhere).
    RELEVANT_COS = 0.6
    STRONG_COS = 0.7
    TOPICAL_DF = 0.02  # a question noun in at most this share of passages is topical (not 출장·공무원·기준 …)

    def __init__(self, directory, embedder=None, institutions=ROOT / 'data/corpus-inventory/alio-agencies.json',
                 review=None, table_rows=TABLE_ROWS, term_map=TERM_MAP, applies_backfill=APPLIES_BACKFILL):
        from trid.service.index import Index
        root = Path(directory)
        self.index = Index.load(root)
        self.meta = json.loads((root / 'meta.json').read_text())
        self.manifest = json.loads((root / 'manifest.json').read_text())
        self.name = root.name
        self.passage_doc = np.array([p.doc_id for p in self.index.passages])
        self.by_institution = {}
        for doc_id, m in self.meta.items():
            name = (m['institution'] or '').split('(질의')[0]
            if len(_norm(name)) >= 3:
                self.by_institution.setdefault(_norm(name), set()).add(doc_id)
        self.norm_titles = {doc_id: _norm(m['title']) for doc_id, m in self.meta.items()}
        tables = root / 'tables.json'
        self.tables = set(json.loads(tables.read_text())) if tables.exists() else set()
        self.qa_docs = {d for d, m in self.meta.items() if any(x in (m['kind'] or '') for x in ('FAQ', '질의응답', '질의회신'))}
        self.authority = np.array([self.authority_of(self.meta[str(d)]['kind']) for d in self.passage_doc], np.float32)
        self.form_mask = np.array([bool(p.path) and bool(FORM_PATH.fullmatch(p.path[-1] or '')) for p in self.index.passages])
        # B's term relations (DATA-CONTINUE-007): only "equivalent_when" counts, one way (everyday → official term), and
        # only inside the rules B cited as its source (their whole rule family, set below): there the official term is
        # what the everyday word means (공무원 여비 규정: 여비 = 공무 여행의 운임·일비·숙박비·식비). Elsewhere the same
        # word may mean something else (B's not_equivalent cases: 평가위원 출장비, 업무추진비 …), so it is not widened
        # there, and the question itself is never rewritten.
        terms = pinned_jsonl(term_map)
        mine = 'inst:' if self.manifest.get('separate') else f'{self.name}:'
        cited = {}
        for t in terms:
            # only equivalent_when rows and their evidence source ids are read: the not_equivalent rows (term-map line 6
            # quotes a held REB 행동강령 text) are kept unconsumed — reading them needs C's judgment again (MODEL-RESUME-C-024)
            if t.get('relation') == 'equivalent_when':
                docs = {e['source'][len(mine):] for e in t.get('evidence', []) if e['source'].startswith(mine)}
                cited.setdefault(t['everyday_term'], {}).setdefault(t['official_term'], set()).update(d for d in docs if d in self.meta)
        # B's applicability records for documents without one (or whose record says 'unknown'); an existing stated
        # scope is never replaced. Each record quotes the source's own 적용범위/목적 text (B's checker located it).
        filled = 0
        for spec in applies_backfill or []:
            folder = Path(spec[0]).parent.name
            for entry in pinned_jsonl(spec):
                for d in entry['doc_ids']:
                    current = self.meta.get(d, {}).get('applies_to')
                    if d in self.meta and (not isinstance(current, dict) or current.get('scope') == 'unknown'):
                        self.meta[d]['applies_to'] = dict(entry['applies_to'], source=f'B backfill {folder}')
                        filled += 1
        self.domain = {'term_map': len(terms), 'applies_backfilled': filled, 'form_passages': int(self.form_mask.sum())}
        # B's per-passage review (search-corpus-v4 metadata; texts and ids unchanged, so vectors stay valid): a
        # privacy-review passage is never retrieved (no model context, answer or source text); a value-hold passage's
        # numbers are not answer values (the reply points to the original page instead).
        self.private, self.value_hold, self.review_source = set(), set(), None
        if review and Path(review).exists():
            for line in _lines(review):
                if '"value_review"' not in line:
                    continue
                row = json.loads(line)
                if row['value_review'].get('privacy_review'):
                    self.private.add(row['passage_id'])
                if row['value_review'].get('decision') == 'hold_for_value_use':
                    self.value_hold.add(row['passage_id'])
            manifest = Path(review).with_name('manifest.json')
            self.review_source = {'file': str(review), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()
                                  if manifest.exists() else None, 'privacy': len(self.private), 'value_hold': len(self.value_hold)}
        ids = np.array([p.id for p in self.index.passages])
        # Hard exclusion: a privacy passage is outside the selectable set (score −inf after ranking), not just low.
        self.private_mask = np.isin(ids, list(self.private))
        # Fail closed at startup: privacy ids that match no indexed passage mean the review and the index use different
        # passage ids, so the passages the review protects could be served under other ids.
        missing = len(self.private) - int(self.private_mask.sum())
        if missing:
            raise ValueError(f'{missing} privacy-review passage id(s) are not in the index {root}: review and index '
                             'passage ids differ; refusing to serve')
        if self.review_source:
            self.review_source['value_hold_in_index'] = int(np.isin(ids, list(self.value_hold)).sum())
        # Canonical rule names ("공무원 여비 규정" from "공무원 여비 규정 [별표 0002] …"): an exact mention names the
        # rule regardless of TITLE_MIN, separately from partial title matching of ordinary sentences.
        self.rule_names = {}
        for doc_id, m in self.meta.items():
            if doc_id in self.qa_docs:
                continue
            base = _norm(re.split(r'\s*[\[(]|\s+-\s+', m['title'])[0])
            if len(base) >= 4 and re.search(r'(법|령|규정|규칙|지침|훈령|예규|고시|편람|기준)$', base):
                self.rule_names.setdefault(base, set()).add(doc_id)
        # The tables (별표) of a rule, by the annex id printed in the table's own caption ("■ 공무원 여비 규정 [별표 6의2]"):
        # a statute body passage that says "별표 2에 따른다" is answered together with that table of the same statute
        # (one rule, not a mix of rules). The corpus title number is not used — "[별표 0006]" may be 별표 6의2 — and a
        # table whose caption gives no annex id is not linked.
        annex_docs = {d for docs in self.rule_names.values() for d in docs if '별표' in self.meta[d]['title']}
        first = {}
        for p in self.index.passages:
            if p.doc_id in annex_docs and (p.doc_id not in first or p.char_start < first[p.doc_id].char_start):
                first[p.doc_id] = p
        self.annex_id = {d: _annex(m.group(1)) for d, p in first.items()
                         for m in [re.search(r'\[별표\s*(\d+(?:\s*의\s*\d+)?)\]', p.text[:300])] if m}
        self.rule_tables = {}
        for base, docs in self.rule_names.items():
            for d in docs:
                if d in self.annex_id:
                    self.rule_tables.setdefault(base, {})[self.annex_id[d]] = d
        self.rule_of = {d: base for base, docs in self.rule_names.items() for d in docs}
        # each cited rule with all its documents (body and 별표): {everyday word: {doc id: [official terms]}}
        self.term_equivalents = {}
        for word, officials in cited.items():
            for official, docs in officials.items():
                for d in docs:
                    for f in self.rule_names.get(self.rule_of.get(d), {d}):
                        self.term_equivalents.setdefault(word, {}).setdefault(f, []).append(official)
        self.domain['term_scope'] = {w: sorted(self.meta[d]['title'][:40] for d in docs) for w, docs in self.term_equivalents.items()}
        self._link_table_rows(table_rows or [])
        self.dense, self.dense_info, self.embedder = None, None, embedder
        if (root / 'dense.f16.npy').exists():
            self.dense_info = json.loads((root / 'dense.json').read_text())
            from trid.service.embed import load_dense
            self.dense = load_dense(root)
            if len(self.dense) != len(self.index.passages):
                raise ValueError('dense.f16.npy does not match the index passages; rerun scripts/build_embeddings.py')
            if self.embedder is None:
                from trid.service.embed import load_model
                model, _ = load_model('cpu')
                self.embedder = lambda text: model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        self.registry = self._registry(Path(institutions))
        # A rule whose title starts with an institution's name is that institution's own (or its staff's) rule; a
        # question naming no institution prefers common rules over it (issuer ≠ the population a rule applies to).
        prefixes = {name for name in self.institutions if len(name) >= 3}
        self.specific = {d for d, t in self.norm_titles.items()
                         if d not in self.qa_docs and any(t[:k] in prefixes for k in range(3, min(len(t), 30) + 1))}
        # B's record that a rule applies only within its institution (제1조·제2조 적용범위) says the same as its title
        # would: an institution's own rule, so a general question prefers common rules over it.
        self.specific |= {d for d in self.meta if d not in self.qa_docs and self.applies(d).get('scope') == 'institution'}
        areas = ROOT / 'trid/service/reference/admin-areas.json'
        # City → its legal category (서울 → 특별시, 부산 → 광역시) for reading rule tables; '도' is too common to use.
        self.area_category = ({k: v for k, v in json.loads(areas.read_text())['areas'].items() if len(v) >= 3}
                              if areas.exists() else {})
        facts = root / 'facts.jsonl'  # structured_facts of API snapshot records (HYBRID-RAG-001-SCHEMA.md)
        self.facts = [json.loads(line) for line in facts.read_text().splitlines()] if facts.exists() else []

    ANNEX_CAPTION = re.compile(r'■?\s*(.*?)\s*\[별표\s*(\d+(?:\s*의\s*\d+)?)\]')

    def doc_text(self, doc):
        """A document's text rebuilt from its indexed passages at their source offsets (gaps as line breaks)."""
        out, end = [], 0
        for i in sorted(np.flatnonzero(self.passage_doc == doc), key=lambda i: self.index.passages[i].char_start):
            p = self.index.passages[i]
            if p.char_start > end:
                out.append('\n' * (p.char_start - end))
                end = p.char_start
            if p.char_start + len(p.text) > end:
                out.append(p.text[end - p.char_start:])
                end = p.char_start + len(p.text)
        return ''.join(out)

    def _link_table_rows(self, sources):
        """Link reviewed table rows (REVIEWED only; render_check_pass and unreviewed rows are never used) to the exact
        indexed text they describe. The link — B's link file entry with status 'linked' (key: law name + MST + the
        annex id printed in the table's own header + caption), or the row's own index location — is verified here:
          - the target document is in this index; a same-document link also has the rows' source hash;
          - the linked span (or the whole table document) contains every number of the rows' cells, leaving out
            revision/deletion markers, deleted items and repeated merged cells.
        Linked: `row_links[(scope, annex)]` (scope: the rule's normalised name, or the document for institution rows)
        with the document, the span, its passages and the method. Not linked (B: body_empty / mismatch, or a check
        failed here): `row_unlinked[(scope, annex)]` = reason — used to say the rows were left out, never to attach
        another table's rows instead."""
        self.row_links, self.row_unlinked, self.table_rows_source = {}, {}, []
        marker = lambda t: re.sub(r'<[^>]*>', ' ', str(t))
        deleted = lambda r: all(re.fullmatch(r'\s*[\d.]*\s*삭제\s*', marker(v)) for v in r['cells'].values())
        for rows_path, links_path in sources:
            rows_path = Path(rows_path)
            if not rows_path.exists():
                continue
            rows = [r for r in map(json.loads, rows_path.read_text().splitlines()) if r.get('review') == REVIEWED]
            links = ({l['row_doc_id']: l for l in map(json.loads, Path(links_path).read_text().splitlines())}
                     if links_path and Path(links_path).exists() else {})
            groups = {}
            for r in rows:
                cap = self.ANNEX_CAPTION.search(' '.join(r.get('caption') or []))
                groups.setdefault((r['doc_id'], _annex(cap.group(2)) if cap else None), []).append(r)
            for (doc_id, caption_annex), part in groups.items():
                src, link = part[0].get('source') or {}, links.get(doc_id)
                if src.get('index_doc_id'):           # rows that carry their own place in the index
                    spans = [r['source']['char_span'] for r in part if (r.get('source') or {}).get('char_span')]
                    target, src_hash = src['index_doc_id'], src.get('content_hash')
                    span = [min(s for s, _ in spans), max(e for _, e in spans)] if spans else None
                    annex, scope, method = caption_annex, src['index_doc_id'], 'row_source_span'
                elif link:
                    annex, scope = _annex(link['byl']), _norm(link['law'])
                    if link.get('byl_doc_in_index') == doc_id and doc_id in self.meta:
                        # the same table document is indexed: linked by its source hash (checked below), whatever
                        # B found for the statute-body span (e.g. mismatch from columns mixed in the flattened body)
                        target, span, src_hash, method = doc_id, None, src.get('kordoc_text_sha256'), 'same_source_hash'
                        if not src_hash:
                            self.row_unlinked[(scope, annex)] = 'rows carry no source hash for the indexed document'
                            continue
                    elif link.get('status') != 'linked':
                        self.row_unlinked[(scope, annex)] = f"B link status {link.get('status')}"
                        continue
                    else:
                        section = link.get('linked_section')
                        span = section['char_span'] if isinstance(section, dict) else section
                        target, src_hash, method = link['body_doc'], None, 'statute_body_segment'
                        if target != f"law-go-kr:law:{link['mst']}":
                            self.row_unlinked[(scope, annex)] = 'body document is not the linked MST'
                            continue
                else:
                    self.row_unlinked[(doc_id, caption_annex)] = 'no link to the index'
                    continue
                key = (scope, annex)
                if not annex or target not in self.meta:
                    self.row_unlinked[key] = 'target document not in this index' if annex else 'no annex id'
                    continue
                if src_hash and self.index.documents[target].sha256 != src_hash:
                    self.row_unlinked[key] = 'source hash differs from the indexed document'
                    continue
                if method == 'same_source_hash' and self.annex_id.get(target) not in (None, annex):
                    self.row_unlinked[key] = 'indexed document caption has another annex id'
                    continue
                text = self.doc_text(target)
                s, e = span or (0, len(text))
                if method == 'statute_body_segment':
                    # Two "[별표 N]" sections in one statute body (an old and a new wording): which one is current is not
                    # established by the text, so neither is used (a guess is not promoted to the current table).
                    headers = len(re.findall(r'\[별표\s*' + r'\s*'.join(map(re.escape, annex)) + r'\]', text))
                    if headers != 1:
                        self.row_unlinked[key] = (f'the statute body holds {headers} "[별표 {annex}]" sections; which one '
                                                  'is the current text is unconfirmed')
                        continue
                repeated = lambda r: set(r.get('repeated') or [])
                cells = [marker(v) for r in part if not deleted(r) for k, v in r['cells'].items() if k not in repeated(r)]
                numbers = set().union(set(), *map(_numbers, cells))
                if not numbers <= _numbers(marker(text[s:e])):
                    self.row_unlinked[key] = f'{len(numbers - _numbers(marker(text[s:e])))} row numbers not in the linked span'
                    continue
                passages = [int(i) for i in np.flatnonzero(self.passage_doc == target)
                            if self.index.passages[i].char_start < e
                            and self.index.passages[i].char_start + len(self.index.passages[i].text) > s]
                self.row_links[key] = {'doc': target, 'segment': [s, e], 'passages': passages, 'rows': tuple(part),
                                       'annex': annex, 'method': method, 'rows_doc': doc_id,
                                       'law_mst': (link or {}).get('mst'), 'statute_effective_date': part[0].get('statute_effective_date')}
            self.table_rows_source.append({'rows': str(rows_path), 'links': str(links_path) if links_path else None,
                                           'reviewed_rows': len(rows), 'rows_sha256': hashlib.sha256(rows_path.read_bytes()).hexdigest()})

    FACT_COS = 0.6  # metric name ↔ question similarity for a structured value to count as asked for

    def matching_facts(self, question, institutions):
        """Actual-spending / statistic structured values whose metric matches the question (bge-m3 cosine ≥ FACT_COS),
        limited to the named institutions when any are named."""
        pool = [f for f in self.facts if f.get('basis') in ('actual_spending', 'statistic')]
        if institutions:
            pool = [f for f in pool if self.registry.get(_norm(f['subject']['name']), _norm(f['subject']['name'])) in institutions]
        if not pool or self.embedder is None:
            return []
        q = self.embedder(question)
        return [f for f in pool if float(self.embedder(f['metric']) @ q) >= self.FACT_COS]

    def _registry(self, path, official=ROOT / 'data/agency-aliases-v1/aliases.jsonl',
                  government=ROOT / 'data/official-abbreviations/abbreviations.jsonl'):
        """alias -> official name. Sources: corpus issuers; the ALIO list of public institutions (with generic aliases:
        without a leading 한국/국립/대한 when ≥3 characters remain); and B's official alias table (agency-aliases-v1:
        기관코드 official names, ALIO names, statutory abbreviations defined in law texts, predecessor names), each row
        with its evidence. An alias marked ambiguous, or claimed by two institutions, is never resolved automatically
        (`self.ambiguous` keeps its candidates for asking back). A predecessor name stays its own (old) institution with
        `successor` noted: an old body's rules are not applied to its successor. Each official name records its corpus
        documents (issued by it, or rules/manuals whose title starts with its name) and its ALIO type and ministry.
        The official abbreviations of central administrative bodies (「정부조직 약칭과 영어 명칭에 관한 규칙」 [별표]:
        행정안전부예규 제390호 in force, plus abbreviations only an earlier version used, mapped to the body it named) are
        binding names for those bodies: each resolves to its body even where another source claims the same word (노동부
        is now 고용노동부's abbreviation, not only the pre-2010 body)."""
        self.institutions, self.ambiguous = {}, {}
        names = [(n, None) for n in self.by_institution]
        if path.exists():
            names += [(_norm(re.sub(r'^\((사|재|주)\)', '', a['apbaNa'])), a) for a in json.loads(path.read_text())['agencies']]
        aliases = {}
        entry_of = lambda name: self.institutions.setdefault(name, {'name': name, 'alio': None, 'docs': set()})
        for name, alio in names:
            entry = entry_of(name)
            if alio:
                entry['alio'] = {'id': alio['apbaId'], 'name': alio['apbaNa'], 'type': alio['typeNa'], 'ministry': alio['jidtNa']}
            for alias in {name} | {name[len(p):] for p in ('한국', '국립', '대한') if name.startswith(p) and len(name) - len(p) >= 3}:
                aliases.setdefault(alias, set()).add(name)
        self.alias_source = None
        if official.exists():
            self.alias_source = {'file': str(official), 'manifest_sha256': hashlib.sha256(
                official.with_name('manifest.json').read_bytes()).hexdigest() if official.with_name('manifest.json').exists() else None}
            for line in _lines(official):
                row = json.loads(line)
                alias, canonical, kind = _norm(row['alias']), _norm(row['canonical']), row['kind']
                # Hangul aliases need 3+ characters; Latin acronyms (LH) defined in law texts count from 2.
                if not _alias_ok(alias) or len(canonical) < 3:
                    continue
                if row['ambiguous']:
                    self.ambiguous.setdefault(alias, set()).add(canonical)
                    continue
                # 기관코드 short names include departments ("인사처" = 한국전력기술(주) 안전기획관리본부 인사처): an
                # official row counts only when its canonical name is the body's own full name (a top-level body).
                if kind in ('official_name', 'official_full_name') and _norm(row['evidence'].get('full_name')) != canonical:
                    continue
                if kind == 'predecessor_name':
                    entry_of(alias)['successor'] = row['canonical']
                    aliases.setdefault(alias, set()).add(alias)
                    continue
                entry = entry_of(canonical)
                if row.get('alio_id') and not entry['alio']:
                    entry['alio_id'] = row['alio_id']
                aliases.setdefault(alias, set()).add(canonical)
        self.government_abbreviations, self.central_bodies = {}, set()
        if government.exists():
            manifest = government.with_name('manifest.json')
            info = json.loads(manifest.read_text())
            versions = {v['id']: v for v in info['versions']}
            for line in government.read_text().splitlines():
                row = json.loads(line)
                alias, canonical = _norm(row['alias']), _norm(row['canonical'])
                entry_of(canonical)
                aliases[alias] = {canonical}
                self.ambiguous.pop(alias, None)
                v = versions[row['version']]
                # which version of the rule the abbreviation comes from, said wherever it is used
                self.government_abbreviations[alias] = {
                    'canonical': canonical, 'status': row['status'],
                    'source': f"「{info['rule']}」 [별표], {v['version']}" + ('' if row['status'] == 'current' else ' — 현행 판에는 없는 이전 약칭')}
            # the 중앙행정기관 the rule in force lists (not a statement about who of their staff is a civil servant)
            self.central_bodies = {_norm(n) for n in info['central_bodies']}
            self.alias_source = {**(self.alias_source or {}), 'government_abbreviations': str(government),
                                 'government_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
        prefixes = {}
        for d, t in self.norm_titles.items():
            if d not in self.qa_docs:
                for k in range(3, min(len(t), 40) + 1):
                    prefixes.setdefault(t[:k], set()).add(d)
        for name, entry in self.institutions.items():
            entry['docs'] = self.by_institution.get(name, set()) | prefixes.get(name, set())
        for alias, owners in aliases.items():
            if len(owners) > 1:
                self.ambiguous.setdefault(alias, set()).update(owners)
        return {alias: next(iter(owners)) for alias, owners in aliases.items()
                if len(owners) == 1 and _alias_ok(alias) and alias not in self.ambiguous}

    def scores(self, question, protected=False):
        """Hybrid passage scores in [0, 1] (authority-weighted), plus the raw cosines (None without dense vectors).
        Keyword: BM25 / its maximum. Meaning: bge-m3 cosine rescaled so the question's median passage is 0 and its best
        passage 1. Hybrid = LEXICAL_WEIGHT·keyword + (1−LEXICAL_WEIGHT)·meaning, times the passage's authority.
        Privacy passages get −inf (outside every ranking, pin and fallback); with `protected` the scores they would
        have had are returned too (0 elsewhere), so a question whose best evidence is protected is not answered from
        the rest."""
        from trid.service.index import tokenize_all
        # A city in the question is also searched as its legal category (부산 → 부산 광역시): rule tables use the
        # category, and the question's own word is kept, not replaced.
        question = question + ''.join(f' {self.area_category[a]}' for a in self.area_category if a in question)
        tokens = tokenize_all([question])[0]
        bm = self.index.bm25.get_scores(tokens) if tokens else np.zeros(len(self.passage_doc), np.float32)
        lexical = bm / bm.max() if bm.max() > 0 else bm
        cos = None
        if self.dense is not None:
            cos = self.dense @ self.embedder(question)
            floor, top = float(np.median(cos)), float(cos.max())
            semantic = np.clip((cos - floor) / max(top - floor, 1e-6), 0, 1)
            hybrid = self.LEXICAL_WEIGHT * lexical + (1 - self.LEXICAL_WEIGHT) * semantic
        else:
            hybrid = lexical
        hybrid = (hybrid * self.authority).astype(np.float32)
        hidden = np.where(self.private_mask, hybrid, 0).astype(np.float32)
        hybrid[self.private_mask] = -np.inf
        return (hybrid, cos, tokens, hidden) if protected else (hybrid, cos, tokens)

    QUESTION_WORDS = {'얼마', '무엇', '어디', '언제', '누구', '어떻게', '정도', '경우', '관련', '내용', '기준', '규정',
                      '뜻', '의미', '차이', '개념', '정의', '요령', '방법'}  # question and request words, not topics
    JOSA = re.compile(r'(?:에서는|에서|으로는|으로|에게|까지|부터|이나|이란|란|은|는|이|가|을|를|의|에|로|와|과|도|만|요)$')

    def topical(self, question, named):
        """{noun: idf} for nouns that carry the question's topic: in at most TOPICAL_DF of the passages (at least one
        passage), not part of an institution name, not a question word. A kiwi noun found in no passage at all is
        topical too (nothing in the corpus is about it) and weighs most. Words with their particle removed are added
        when the corpus knows them, since kiwi splits some questions' nouns ("식비는" → 식/비)."""
        from trid.korean_quality import kiwi
        n = len(self.passage_doc)
        limit = max(1, int(self.TOPICAL_DF * n))
        df = lambda t: int(np.count_nonzero(self.index.bm25.get_scores([t])))
        nouns = [t.form.lower() for t in kiwi().tokenize(question) if t.tag in ('NNG', 'NNP', 'SL', 'SH')]
        words = [self.JOSA.sub('', w) for w in re.findall(r'[가-힣]{2,}', question)]
        out = {}
        for t in dict.fromkeys(nouns + [w for w in words if len(w) >= 2 and w not in nouns]):
            if len(t) < 2 or t in self.QUESTION_WORDS or any(t in name for name in named):
                continue
            f = df(t)
            if (f <= limit and t in nouns) or 0 < f <= limit:
                out[t] = float(np.log((n + 1) / (f + 1)))
        return out

    COVERAGE = 0.4  # dev calibration: uncovered questions 0.34 of topical-noun idf present, answerable 0.42–1.0

    def relevant(self, indices, cos, topical, continued=False):
        """Whether these passages can answer the question: best cosine ≥ RELEVANT_COS; below STRONG_COS the passages
        must also hold ≥ COVERAGE of the question's topical-noun idf mass (without dense vectors only the noun check).
        A short follow-up of the previous document ("그럼 감봉은?") embeds poorly on its own: only the noun check."""
        best = max(float(cos[i]) for i in indices) if cos is not None and not continued else None
        if best is not None and best < self.RELEVANT_COS:
            return False
        if (best is not None and best >= self.STRONG_COS) or not topical:
            return True
        texts = ' '.join(self.index.passages[i].text for i in indices)
        # A city named in the question is covered by its legal category in a rule table (부산 → 광역시); an everyday word
        # by the official term B's contract equates it with, inside the rules B cited for that relation (출장비 → 여비 in
        # 공무원 여비 규정: its text says 여비, never 출장비). The passages here are one document's.
        doc = str(self.passage_doc[indices[0]])
        present = lambda t: (t in texts or (t in self.area_category and self.area_category[t] in texts)
                             or any(o in texts for o in self.term_equivalents.get(t, {}).get(doc, ())))
        return sum(w for t, w in topical.items() if present(t)) >= self.COVERAGE * sum(topical.values())

    INSTITUTION_SUFFIX = re.compile(r'(공사|공단|위원회|재단|진흥원|연구원|은행|센터|청|처|부)$')

    def institution_candidates(self, mention):
        """Official institutions a name outside the alias table may stand for, to ASK about (never resolved silently):
        the official table's candidates when the alias is ambiguous there; otherwise names in which its characters
        appear in order and that end in the same kind of body (건보공단 ~ 국민건강보험공단). At most six."""
        if mention in self.ambiguous:
            return sorted(self.ambiguous[mention] - {mention})[:6]
        suffix = self.INSTITUTION_SUFFIX.search(mention)
        if len(mention) < 3 or not suffix:
            return []
        pattern = re.compile('.*?'.join(map(re.escape, mention)))
        return sorted(n for n in self.institutions if n != mention and n.endswith(suffix.group(1)) and pattern.search(n))[:6]

    def scope(self):
        kinds = sorted(self.manifest['by_kind'].items(), key=lambda kv: -kv[1])
        shown = ' · '.join(f'{k} {n:,}' for k, n in kinds[:6]) + (' 등' if len(kinds) > 6 else '')
        lo, hi = self.manifest['date_range']
        return (f"검색 자료 {self.manifest['documents']:,}건·{self.manifest['passages']:,}구간({shown}; 기준일 {lo}~{hi}). "
                "법제처 국가법령정보센터·공공데이터포털 등 공개 자료이며 기관 내부 비공개 자료가 아닙니다. 질의응답 자료는 "
                "국민권익위원회 게시판 비중이 큽니다. 모든 값은 표시된 기준일 당시 값이며 현행 여부는 다시 확인하지 않았습니다.")

    def provenance(self, doc_id):
        """The '자료 정보: 문서 … · 기관 … · 기준 …' line of B's case-training-view (scripts/build_case_training_view.py
        provenance()), from this document's metadata; the date basis says that later amendments were not checked."""
        m = self.meta[doc_id]
        when = f"{m['date']} {m['date_type'] or ''}".strip() + ' 기준 자료(이후 개정 여부 미확인)' if m.get('date') else None
        parts = [f"문서 {m['title']}", f"기관 {m['institution']}" if m.get('institution') else None,
                 f'기준 {when}' if when else None]
        return f"자료 정보: {' · '.join(p for p in parts if p)}\n"

    def describe(self, doc_id, score=None):
        m = self.meta[doc_id]
        return {'doc_id': doc_id, 'title': m['title'], 'institution': m['institution'], 'kind': m['kind'],
                'date': m['date'], 'date_type': m['date_type'], 'effective_date': m['effective_date'],
                'currency': m['currency'], 'source': m['source'], 'license': m['license'],
                **{k: m[k] for k in ('source_id', 'snapshot_id', 'issuer_id', 'issuer_id_scheme') if m.get(k)},
                **({'score': round(score, 2)} if score is not None else {})}

    INSTITUTION_LIKE = re.compile(r'[가-힣]{2,}(?:공사|공단|위원회|재단|진흥원|연구원|은행|센터|특별시|광역시|특별자치시|특별자치도|청|처|부)(?![가-힣])')
    WEAK = 0.6  # best score inside the named institution below this share of the corpus-wide best: no relevant rule

    def constraints(self, question):
        """(named institutions, unknown institution names, allowed doc ids or None, named document ids or None).
        Institutions are recognised through the registry (official names and generic aliases, e.g. 부동산원 →
        한국부동산원) and, for unregistered names, INSTITUTION_LIKE. A registered institution without documents in the
        corpus, or an unregistered name, is unknown (the reply says the corpus lacks its rules). An institution's
        documents are those it issues or rules/manuals whose title starts with its name."""
        q = _norm(question)
        found = [a for a in self.registry if a in q]
        found = [a for a in found if not any(a != b and a in b for b in found)]  # longest aliases
        official = list(dict.fromkeys(self.registry[a] for a in found))
        mentioned = [_norm(w) for w in self.INSTITUTION_LIKE.findall(question)]
        mentioned = [w for w in mentioned if not any(w in a or a in w for a in found)]
        # Unregistered names count when rules/manuals start with them ("해양경찰청 …"); Q&A titles are visitors'
        # questions ("국민건강보험공단 지사장께 …") and a mention inside another body's title is not its rule either.
        # Other unregistered or ambiguous names stay unknown: select() asks which official institution is meant.
        named = [n for n in official if self.institutions[n]['docs']]
        unknown = [n for n in official if not self.institutions[n]['docs']]
        leading = {w: {d for d, t in self.norm_titles.items() if t.startswith(w) and d not in self.qa_docs} for w in mentioned}
        named += [w for w, docs in leading.items() if docs]
        # A name outside the official table that only occurs inside document titles ("환경시설설치공사매뉴얼") and
        # resembles no official institution is a topic word ("설치공사"), not an institution.
        topic = {w for w, docs in leading.items() if not docs and not self.institution_candidates(w)
                 and any(w in t for d, t in self.norm_titles.items() if d not in self.qa_docs)}
        unknown += [w for w, docs in leading.items() if not docs and w not in topic]
        allowed = None
        if named:
            allowed = set().union(*((self.institutions[n]['docs'] if n in self.institutions else leading[n]) for n in named))
        mentioned = found + mentioned
        # An exact canonical rule name ("공무원 여비 규정") names the rule whatever its length; only otherwise are
        # shared title fragments of TITLE_MIN+ characters read as naming a document.
        exact = [n for n in self.rule_names if len(n) >= 6 and n in q and _names_at_word(question, n)]
        exact = [n for n in exact if not any(n != m and n in m for m in exact)]
        titled = set().union(*(self.rule_names[n] for n in exact)) if exact else None
        if titled and allowed is not None:
            titled = titled & allowed or None
        if titled:
            return named, unknown, allowed, titled
        # Title matching ignores institution names, so naming an institution alone never counts as naming a document.
        q_title = q
        for name in named + mentioned:
            q_title = q_title.replace(name, '|')
        # Documents whose normalised title contains any TITLE_MIN-character stretch of the question title. (Formerly a
        # dictionary of every title's TITLE_MIN-grams, 1.17 M entries and 0.35 GB resident for v5; a title has a gram
        # exactly when the gram is a substring of it, so scanning the 14.7 k titles gives the same set — SITE-MEMORY-023.)
        k = self.TITLE_MIN
        grams = {q_title[i:i + k] for i in range(len(q_title) - k + 1)}
        near = {d for d, t in self.norm_titles.items() if any(g in t for g in grams)} if grams else set()
        pool = (near & allowed if allowed is not None else near) - self.qa_docs  # Q&A titles are visitors' questions
        shared = {d: _longest_shared(self.norm_titles[d], q_title) for d in pool}
        top = max(shared.values(), default=0)
        titled = {d for d, n in shared.items() if n == top} if top >= self.TITLE_MIN else None
        return named, unknown, allowed, titled

    def _context_relevant(self, everywhere, context_doc):
        # Hybrid scores are dense near the top, so a follow-up continues the previous document only when that document
        # is about as good as anything (CLOSE): "그럼 감봉은?" after 징계업무편람 0.90, a self-contained new question such as
        # "상시 출장하는 공무원 여비는?" after 법원공무원 여비 규칙 0.76 (dev measurement).
        return (context_doc in self.meta
                and everywhere[self.passage_doc == context_doc].max(initial=0) >= self.CLOSE * everywhere.max() > 0)

    def continues(self, question, context_doc):
        """True when a question that names no institution or document is answered by `context_doc` (the document
        applied in the previous conversation turn) at least CLOSE as well as by the best passage anywhere."""
        named, unknown, allowed, titled = self.constraints(question)
        if named or unknown or titled or allowed is not None or context_doc not in self.meta:
            return False
        return self._context_relevant(self.scores(question)[0], context_doc) and not self.scope_conflict(question, context_doc)

    def scope_conflict(self, question, doc):
        """True when the question names 지방공무원 / 국가공무원 and `doc`'s rule is recorded (B applies_to) for another
        population: it is then no follow-up of `doc` ("지방공무원 여비 규정에서 숙박비는?" after 공무원 여비 규정, whose
        제1조 limits it to 국가공무원, or after 한국부동산원 여비규정; MODEL-RESUME-A-024)."""
        subject = self.civil_service_subject(question)
        scope = rule_scope(self, doc) if subject and doc in self.meta else None
        return bool(scope) and scope != subject

    def select(self, question, pinned=None, context_doc=None):
        """dict(status, applied, chunks, others, institutions, retrieval, ...). status: 'answer' | 'clarify' |
        'unknown_institution' | 'institution_no_match' | 'not_found' (reason 'irrelevant' when passages were found but
        none can answer: no passage of a candidate document reaches RELEVANT_COS or contains a topical question word).
        Candidate documents, the ask-back list and the other-documents list only contain documents passing that check.
        `context_doc` (the previous turn's document) is searched instead of the whole corpus when it scores at least
        CLOSE of the corpus-wide best for a question naming no institution or document (`continued`)."""
        named, unknown, allowed, titled = self.constraints(question)
        result = {'applied': None, 'chunks': [], 'others': [], 'institutions': named, 'unknown_institutions': unknown,
                  'title_matched': bool(titled)}
        if unknown and not named and not pinned:
            # An official institution without documents: say so. A name not in the official list: offer the official
            # institutions it may stand for, or ask for the official name — never assert that its rules are absent.
            informal = [w for w in unknown if w not in self.institutions]
            if informal:
                options = sorted({n for w in informal for n in self.institution_candidates(w)})
                if options:
                    return {**result, 'status': 'clarify', 'reason': 'institution', 'institutions': options[:6],
                            'candidates': []}
                return {**result, 'status': 'unknown_institution', 'reason': 'unrecognized'}
            return {**result, 'status': 'unknown_institution'}
        restrict = {pinned} if pinned else (titled or allowed)
        everywhere, cos, tokens, hidden = self.scores(question, protected=True)
        topical = self.topical(question, named + unknown)
        result['retrieval'] = {'mode': 'hybrid' if cos is not None else 'bm25', 'topical_words': list(topical),
                               'best_cosine': round(float(cos.max()), 3) if cos is not None else None}
        if everywhere.max() <= 0:
            return {**result, 'status': 'not_found'}
        if (restrict is None and not unknown and self._context_relevant(everywhere, context_doc)
                and not self.scope_conflict(question, context_doc)):
            restrict = {context_doc}
            result['continued'] = True
        scores = everywhere
        if restrict is not None:
            scores = np.where(np.isin(self.passage_doc, list(restrict)), everywhere, -np.inf)
        # A blank form says what to fill in, not what is paid or required: for a question not asking for a form, form
        # passages are not evidence and documents are judged by their rule text (a 공무국외출장계획서 "출장경비" blank
        # outscored the 여비 rule for "공공기관 출장비"). A question asking for a form keeps them.
        forms = bool(FORM_ASKED.search(question))
        if not forms and self.form_mask.any():
            scores = np.where(self.form_mask, -np.inf, scores)
        result['retrieval']['form_passages'] = 'included (form asked)' if forms else 'not evidence'
        order = [int(i) for i in np.argsort(-scores, kind='stable')[:60] if scores[i] > 0]
        by_doc = {}
        for i in order:
            by_doc.setdefault(str(self.passage_doc[i]), []).append(i)
        passing = {d: idx[:self.MAX_PASSAGES] for d, idx in by_doc.items()
                   if self.relevant(idx[:self.MAX_PASSAGES], cos, topical, bool(result.get('continued')))}
        # Protected evidence is judged inside the searched scope (named institution, pinned or continued document):
        # when a privacy passage would have outscored (by 1/CLOSE) every usable passage — or every passage at all when
        # none is usable — the reply is a protected-evidence refusal, never a weaker neighbour (C-029: another table
        # of the same statute) nor a claim that nothing relevant exists.
        in_scope = hidden if restrict is None else np.where(np.isin(self.passage_doc, list(restrict)), hidden, 0)
        top_hidden = float(in_scope.max(initial=0))
        open_best = (max(float(scores[idx[0]]) for idx in passing.values()) if passing else
                     float(scores[order[0]]) if order else 0.0)
        if top_hidden > 0 and top_hidden * self.CLOSE >= open_best:
            return {**result, 'status': 'not_found', 'reason': 'protected'}
        if restrict is not None and named and not titled and not pinned and scores.max() < self.WEAK * everywhere.max():
            return {**result, 'status': 'institution_no_match'}
        if not order:
            return {**result, 'status': 'not_found'}
        result['retrieval']['hits'] = [{'doc_id': str(self.passage_doc[i]), 'passage': self.index.passages[i].id,
                                        'hybrid': round(float(scores[i]), 3),
                                        'cosine': round(float(cos[i]), 3) if cos is not None else None} for i in order[:5]]
        if not passing:
            nearest = str(self.passage_doc[order[0]])
            return {**result, 'status': 'institution_no_match' if named and not titled and not pinned else 'not_found',
                    'reason': 'irrelevant', 'nearest': self.describe(nearest)}
        best = {d: float(scores[idx[0]]) for d, idx in passing.items()}
        general = not (pinned or titled or named or result.get('continued'))
        subject = self.civil_service_subject(question)
        if general and subject:
            # "지방공무원 …" is not answered from a rule whose own 제1조 limits it to 국가공무원 (B applies_to), and back;
            # its 별표 carry the rule's record.
            kept = {d: s for d, s in best.items() if rule_scope(self, d) not in
                    {'central_government', 'local_government'} - {subject}}
            if kept:
                best = kept
        applied = next(iter(best))
        result['civil_service_subject'] = subject
        if general and applied in self.specific:
            # A question naming no institution: a common rule as good as (CLOSE) the top institution's own rule wins.
            common = [d for d in best if d not in self.specific and best[d] >= self.CLOSE * best[applied]]
            if common:
                applied = common[0]
        if general and applied in self.specific:
            result['scope_note'] = 'specific'
        conditioned = self.unstated_table_condition(applied, best, question)
        set_aside = None
        if conditioned:
            # "공공기관 출장비 지급 기준" ranked 공무원 여비 규정 [별표 8] (상시 출장 공무원의 여비 지급 기준표) first: a table
            # for a condition the question does not state is not answered in place of its rule, whose body cites the
            # table that applies (and brings it in as a linked table) — never that set-aside table again, whose
            # passages and reviewed rows would hand the model its figures anyway (MODEL-RESUME-A-024 smoke).
            result['table_condition'] = {'table': self.meta[applied]['title'], 'unstated': conditioned[1],
                                         'answered_from': self.meta[conditioned[0]]['title']}
            set_aside, applied = applied, conditioned[0]
        if AMOUNT_ASKED.search(question):
            standard = self.standard_source(applied, passing, best)
            if standard:
                result['standard_choice'] = {'from': applied, 'to': standard[0], 'basis': standard[1]}
                applied = standard[0]
        others = [self.describe(d, s) for d, s in best.items() if d != applied][:5]
        result.update(others=others)
        if general:
            inst = lambda d: (self.meta[d]['institution'] or self.meta[d]['title']).split('(질의')[0]
            top = best[applied]
            rivals = {}
            # Own rules don't rival a common one — a rule B recorded as applying beyond one institution (applies_to).
            # A rule with no record is not known to be common: another institution's own rule still rivals it, so a
            # general question is asked back instead of answered from one body's rule ("국외출장 계획서 작성 방법" →
            # 방위사업청 관리규정 once DG-008 recorded the rival 선관위 rule as its own; MODEL-RESUME-A-024).
            unrecorded = not rule_scope(self, applied)
            for d, s in best.items():
                if (inst(d) != inst(applied) and s >= self.CLOSE * top
                        and (applied in self.specific or d not in self.specific or unrecorded)):
                    rivals.setdefault(inst(d), d)
            if rivals:
                options = [inst(applied)] + list(rivals)[:3]
                # each option with its own annex when that passed too: a common law's 별표 holds the amounts asked
                # for, and listing only its body hides them (C-030 L1/S3) — but not a table for a condition the
                # question does not state (공무원 여비 규정 [별표 8] 상시 출장 for "출장비"; MODEL-RESUME-A-024)
                shown = [applied] + list(rivals.values())[:3]
                annexes = [next((d for d in best if d not in shown and self.meta[d]['title'].startswith(self.meta[s]['title'])
                                 and '[별표' in self.meta[d]['title'] and not self.unstated_table_condition(d, best, question)),
                                None) for s in shown]
                return {**result, 'status': 'clarify', 'institutions': options,
                        'candidates': [c for s, a in zip(shown, annexes)
                                       for c in [self.describe(s, best[s])] + ([self.describe(a, best[a])] if a else [])]}
        chunks = [(PassageChunk(self.index.passages[i].text, 1, self.index.passages[i].char_start,
                                self.passage_doc[i], self.index.passages[i].path,
                                self.index.passages[i].id in self.tables, self.index.passages[i].id in self.value_hold),
                  float(scores[i]))
                  for i in passing[applied]]
        linked = self.linked_tables(applied, [c for c, _ in chunks], everywhere, exclude={set_aside})
        if linked:
            result['linked_tables'] = sorted({str(self.passage_doc[i]) for i in linked})
            chunks += [(PassageChunk(self.index.passages[i].text, 1, self.index.passages[i].char_start,
                                     self.passage_doc[i], self.index.passages[i].path,
                                     self.index.passages[i].id in self.tables, self.index.passages[i].id in self.value_hold),
                        float(everywhere[i])) for i in linked]
        # Tables whose source text is in the evidence: a 별표 document's own annex, or an inline "[별표 N]" caption in a
        # passage. Reviewed rows are added only for such a table when its rows are linked (_link_table_rows) to this
        # very text — the same document, or this statute body at the rows' effective date — never another table.
        present = {}
        for c, s in chunks:
            d = str(c.doc_id)
            scope = self.rule_of.get(d) or d
            overlap = lambda link: link['doc'] == d and c.start < link['segment'][1] and c.start + len(c.text) > link['segment'][0]
            annexes = {self.annex_id[d]} if d in self.annex_id else set()
            annexes |= {_annex(n) for n in re.findall(r'\[별표\s*(\d+(?:\s*의\s*\d+)?)\]', c.text)}
            annexes |= {n for (sc, n), link in self.row_links.items() if sc == scope and overlap(link)}  # table span only
            # "…는 별표1에 따른다" inside the document that itself holds that table (institution rules): that table
            annexes |= {_annex(n) for n in re.findall(r'별표\s*(\d+(?:\s*의\s*\d+)?)', c.text)
                        if self.row_links.get((scope, _annex(n)), {}).get('doc') == d}
            for n in annexes - ({self.annex_id[set_aside]} if set_aside and scope == self.rule_of.get(set_aside) else set()):
                link = self.row_links.get((scope, n))
                body = (link and d not in self.annex_id and self.meta[d].get('effective_date')
                        and self.meta[d].get('effective_date') == link['statute_effective_date'])
                here = link and (overlap(link) or link['doc'] == d)
                if (scope, n) in self.row_unlinked:
                    present.setdefault((scope, n), max(s, present.get((scope, n), -1.0)))
                elif link and (here or body):
                    present[(scope, n)] = max(s, present.get((scope, n), -1.0))
        # The table the evidence is most about — the one whose own caption carries most of the question's topic words
        # (국내 vs 국외 여비 지급표), then the best passage score: its rows if linked; if its rows could not be verified,
        # no other table's rows are attached in its place.
        caption_of = lambda k: ' '.join((self.row_links.get(k) or {}).get('rows', [{}])[0].get('caption') or []) \
            if k in self.row_links else ''
        words = {self.JOSA.sub('', w) for w in re.findall(r'[가-힣]{2,}', question)} - self.QUESTION_WORDS
        about = lambda k: sum(1 for w in words if len(w) >= 2 and w in caption_of(k))
        top = sorted(present, key=lambda k: (-about(k), -present[k]))[:1]
        linked_rows = [k for k in top if k in self.row_links]
        better = [k for k in self.row_links if top and k[0] == top[0][0] and about(k) > about(top[0])]
        if linked_rows and better:
            # the question names another table of the same rule (국외여비 asked, 국내여비 in the evidence): no rows
            linked_rows = []
            result['table_rows_not_in_evidence'] = [f"별표 {self.row_links[k]['annex']}" for k in better]
        missing = sorted(n for (scope, n) in present if (scope, n) in self.row_unlinked)
        if missing:
            result['table_rows_unlinked'] = [f'별표 {n}' for n in missing]
        if linked_rows:
            # B's visually reviewed rows of that table, with their unit, notes and dates: the row/column pairing the
            # flattened table text loses. Derived text, so never a verbatim quote nor a value-version answer. A linked
            # table document's flattened passages give way to its rows; the applied document's passages stay (quotes).
            link = self.row_links[linked_rows[0]]
            result['reviewed_rows'] = {'doc': link['doc'], 'rows': len(link['rows']), 'annex': link['annex'],
                                       'method': link['method']}
            if link['doc'] != applied:
                chunks = [(c, s) for c, s in chunks if str(c.doc_id) != link['doc']]
            chunks.append((PassageChunk(render_rows(link['rows']), 1, 0, link['doc'], ('검토된 표 행',), derived=True,
                                        rows=link['rows']), 0.0))
        kept_linked = sorted({str(c.doc_id) for c, _ in chunks if str(c.doc_id) != applied and not c.derived})
        if kept_linked:
            result['linked_tables'] = kept_linked
        else:
            result.pop('linked_tables', None)
        used = list(passing[applied]) + list(linked)
        if self.private_mask[used].any():  # invariant: never reachable; fail the request rather than serve it
            raise RuntimeError('privacy-review passage reached the answer context')
        return {**result, 'status': 'answer', 'applied': applied, 'chunks': chunks}

    def unstated_table_condition(self, applied, best, question):
        """(body doc, condition nouns) when `applied` is a rule's table (별표) whose caption states a condition the
        question does not, and the rule's body also passed; else None. The condition nouns are the caption's nouns minus
        the rule's own name, the table's type (…표), nouns shared by at least half of the rule's table captions (지급,
        여비 …) and anything in the question. 별표 8 "상시 출장 공무원의 여비 지급 기준표" → 상시: a question that says
        neither is not about that table. Nothing is matched against fixed words or table numbers."""
        if applied not in self.annex_id:
            return None
        base = self.rule_of.get(applied)
        docs = self.rule_names.get(base, set())
        body = next((d for d in docs if d not in self.annex_id and d in best), None)
        if body is None:
            return None
        from trid.korean_quality import kiwi
        caption = lambda d: re.sub(r'\([^)]*\)', '', self.meta[d]['title'].split(']', 1)[-1])
        nouns = lambda text: {t.form for t in kiwi().tokenize(text) if t.tag in ('NNG', 'NNP') and len(t.form) >= 2}
        annexes = [d for d in docs if d in self.annex_id]
        counts = {}
        for d in annexes:
            for n in nouns(caption(d)):
                counts[n] = counts.get(n, 0) + 1
        shared = {n for n, c in counts.items() if c * 2 >= len(annexes)}
        name = nouns(self.meta[body]['title'])
        unstated = sorted(n for n in nouns(caption(applied)) - shared - name
                          if not n.endswith('표') and n not in question)
        return (body, unstated) if unstated else None

    def linked_tables(self, applied, chunks, scores, per_table=2, max_tables=2, exclude=()):
        """Passages of the tables (별표) that the chosen statute-body passages cite by annex id ("별표 2", "별표 6의2"),
        from the same rule — the best-scoring `per_table` of each (privacy passages excluded), for the `max_tables`
        cited tables whose best passage scores highest — so "식비는 별표 2에 따른다" comes with 별표 2."""
        tables = self.rule_tables.get(self.rule_of.get(applied), {})
        if not tables or applied in tables.values():
            return []
        cited = {_annex(n) for c in chunks for n in re.findall(r'별표\s*(\d+(?:\s*의\s*\d+)?)', c.text)}
        cited -= {_annex(n) for c in chunks for n in re.findall(r'\[별표\s*(\d+(?:\s*의\s*\d+)?)\]', c.text)}  # quoted inline
        ranked = []
        for n in sorted(cited):
            doc = tables.get(n)
            if doc and doc not in exclude:
                idx = np.flatnonzero((self.passage_doc == doc) & ~self.private_mask)
                if len(idx):
                    ranked.append([int(i) for i in idx[np.argsort(-scores[idx], kind='stable')][:per_table]])
        ranked.sort(key=lambda idx: -float(scores[idx[0]]))
        return [i for idx in ranked[:max_tables] for i in idx]

    def applies(self, doc):
        """B's structured applicability ({'scope', 'basis', ...}) of a document; {} when absent or free text only."""
        a = self.meta[doc].get('applies_to')
        return a if isinstance(a, dict) else {}

    def standard_source(self, applied, passing, best):
        """(doc, 'defers') for a question asking an amount or standard, when the chosen document is not where that
        standard is set: its passages send the amount to another instrument's annex ("규정 별표1의 상한액", "여비규정
        별표1") and a passing document of the same institution is that instrument — its title holds the name written
        before 별표 (bare "규정": any 규정 of the institution) and its text the annex header [별표1]. Otherwise None. The
        guideline's worked cases (deductions, examples) stay for questions about cases. Same institution only, and not
        statutes: those are left to their own annex linking (linked_tables, reviewed rows)."""
        inst = self.meta[applied]['institution']
        if not inst or self.meta[applied]['kind'].startswith('법령'):
            return None
        peers = sorted((d for d in best if d != applied and self.meta[d]['institution'] == inst
                        and not self.meta[d]['kind'].startswith('법령')), key=lambda d: -best[d])
        if not peers:
            return None
        texts = [self.index.passages[i].text for i in passing[applied]]
        for text in texts:
            for m in DEFERS_TO.finditer(text):
                name = _norm(m[1] + m[2])
                header = re.compile(rf'\[\s*별표\s*0*{m[3]}\s*\]|^#+\s*별표\s*0*{m[3]}\b', re.M)
                for d in peers:
                    # a bare "규정 별표1" (the guideline's defined term) is its own subject's 규정: titles share a
                    # subject word (여비업무 처리지침 / 여비규정), not just the instrument kind
                    if (name in _norm(self.meta[d]['title']) and (m[1] or self._subject(applied) & self._subject(d))
                            and any(header.search(self.index.passages[i].text) for i in np.flatnonzero(self.passage_doc == d))):
                        return d, 'defers'
        return None

    def _subject(self, doc):
        """Character pairs of a rule title's subject: the title without its institution, instrument kinds, amendment
        and posting words, and dates (한국부동산원_여비업무 처리지침 일부개정지침안(게시 …) → 여비)."""
        t = self.meta[doc]['title'].replace(self.meta[doc]['institution'] or '', '')
        t = re.sub(r'\(.*?\)|일부개정\S*|전부개정\S*|제정\S*|규정|규칙|지침|세칙|내규|요강|처리|업무|운영|시행|게시|[^가-힣]', ' ', t)
        return {w[i:i + 2] for w in t.split() for i in range(len(w) - 1)}

    @staticmethod
    def civil_service_subject(question):
        """'local_government' / 'central_government' when the question names only 지방공무원 / only 국가공무원."""
        local, central = '지방공무원' in question, '국가공무원' in question
        return 'local_government' if local and not central else 'central_government' if central and not local else None



def _withhold_figures(result, why):
    """Model commentary sentences carrying numbers are withheld (and the raw output with them); value answers become
    'review'. Verbatim quotes stay, labelled with their source."""
    shown, withheld = [], 0
    for part in re.split(r'\n+|(?<=[.!?])\s+', result.get('commentary') or ''):
        if part.strip():
            if _numbers(part):
                withheld += 1
            else:
                shown.append(part)
    result['commentary'] = '\n'.join(shown) or None
    if withheld:
        result.setdefault('commentary_notes', []).append(f'{why} 금액·일수 등 숫자가 든 해설 문장 {withheld}개를 숨겼습니다.')
    if result.get('status') == 'answer':
        result['status'] = 'review'
    if result.get('raw_output'):
        result.update(raw_output=None, raw_output_withheld=True)


# A question about public institutions as a class (not one named body).
POPULATION = re.compile(r'공공기관|공기업|준정부기관|(?:여러|모든|전체|각|다른)\s*기관|기관(?:들|별)')


def rule_scope(ws, doc):
    """B's applicability scope of a document, or of the rule it belongs to (its 별표 carry no record of their own)."""
    return ws.applies(doc).get('scope') or next(
        (ws.applies(d).get('scope') for d in ws.rule_names.get(ws.rule_of.get(doc), ()) if ws.applies(d).get('scope')), None)


def population_notice(result, question, inst_ws, ws, selection):
    """A question about public institutions as a class answered from one institution's own material (its separate
    snapshot, its structured values, or its own rule): the visible answer first says whose data it is and that it does
    not represent public institutions in general (C-030 #8). One body's figures or position list is never offered as
    the class's. The same holds for an ask-back whose options are all particular bodies' own rules or rules limited to
    civil servants (B applies_to): none is the class's common rule, which the corpus then does not hold — said first,
    so the list does not read as the public institutions' alternatives ("공공기관 출장비" → ministries' 공무국외출장 rules)."""
    if not POPULATION.search(question) or result.get('intent') == 'chat' or result.get('status') == 'chat':
        return
    if inst_ws is not None:
        name = next(iter(inst_ws.by_institution), None) or inst_ws.name
    elif selection and selection.get('status') == 'answer' and selection.get('scope_note') == 'specific':
        name = ws.meta[selection['applied']]['institution'] or '특정 기관'
    elif (selection and selection.get('status') == 'clarify' and selection.get('candidates')
          and all(c['doc_id'] in ws.specific or rule_scope(ws, c['doc_id']) in ('central_government', 'local_government')
                  for c in selection['candidates'])):
        result['population_scope'] = {'asked': 'public_institutions', 'common_rule_found': False}
        result['answer'] = ('질문은 공공기관 전반에 관한 것이지만, 검색 자료에서 공공기관 공통 기준을 찾지 못했습니다. 아래 후보는 각 기관의 '
                            '자체 규정이거나 공무원에게 적용되는 규정이며, 공공기관 전체의 기준이 아닙니다. ' + (result.get('answer') or ''))
        return
    else:
        return
    result['population_scope'] = {'asked': 'public_institutions', 'source_institution': name}
    result['answer'] = (f'질문은 공공기관 전반에 관한 것이지만, 아래는 {name} 한 기관의 자료입니다. 공공기관 전체나 다른 기관의 '
                        f'대표값·기준으로 보지 마세요. ' + (result.get('answer') or ''))


def applicability_guard(result, applies, title):
    """A question that is not about civil servants answered from a rule whose 제1조 limits it to 국가/지방공무원
    (B applies_to): the rule is shown only as the civil-service rule it is. The visible answer says so first and no
    figure of the model's reads as the asker's institution's entitlement (_withhold_figures)."""
    who = '국가공무원' if applies == 'central_government' else '지방공무원'
    _withhold_figures(result, f'이 규정의 적용 대상({who})과 질문 대상이 같다고 확인되지 않아,')
    result['answer'] = (f'「{title}」은 {who}에게 적용되는 규정입니다(제1조 목적). 질문한 대상(예: 공공기관 직원)에게 적용되는지는 '
                        '확인되지 않았으므로, 아래 원문의 금액·기준을 그 기관의 지급액으로 보지 말고 기관 자체 규정을 확인하세요. '
                        + (result.get('answer') or ''))
    result['applicability'] = {'rule_scope': applies, 'question_names_civil_servants': False, 'status': 'not_confirmed'}


def expenditure_guard(result, m):
    """An expenditure record (B: expenditure_record / actual_spend, e.g. one month of executives' 업무추진비) is a list
    of actual payments, not a rule nor a statistic: it is shown as such, its rows only as verbatim quotes, and nothing
    computed from it (sums, averages, per-employee figures) is given (_withhold_figures)."""
    _withhold_figures(result, '집행내역 문서로 계산하거나 다른 의미(직원 평균·출장 식비 등)로 옮긴 값일 수 있어')
    result['answer'] = (f"「{m['title']}」은 {m['institution']} 임원 업무추진비의 실제 집행내역({m['date']} 집행분 공개 문서)입니다. "
                        '규정상 기준이 아니며 직원 평균 식비나 출장 식비도 아닙니다. 아래에는 원문과 일치하는 건별 내역만 보이고, '
                        '평균·합계를 계산해 답하지 않습니다. ' + (result.get('answer') or ''))
    result['record_kind'] = 'expenditure_record'


def institution_indexes(state, workspace, question, context_doc):
    """The separate snapshot index (B: index institution_rules_separate) of the institution the question names by
    its official ALIO id — or, for a follow-up naming no institution and no document, of the previous answer's
    document — else None. A question that names a rule or document ("공무원 여비 규정에서 …") switches to it: it is
    not a follow-up of the previous institution's snapshot even without naming an institution. Nor is one that names
    지방공무원 / 국가공무원 when the previous document's scope (B applies_to) is not theirs ("지방공무원 여비 규정에서
    숙박비는?" after 한국부동산원 여비규정): the answer must not mix the institution's scope into theirs (MODEL-RESUME-A-024)."""
    indexes = getattr(state, 'institution_workspaces', None) or {}
    if not indexes:
        return None
    named, unknown, _, titled = workspace.constraints(question)
    for name in named + unknown:
        entry = workspace.institutions.get(name) or {}
        alio = (entry.get('alio') or {}).get('id') or entry.get('alio_id')
        if alio in indexes:
            return indexes[alio]
    if named or unknown or titled or not context_doc:
        return None
    snapshot = next((ws for ws in indexes.values() if context_doc in ws.meta), None)
    return None if snapshot is None or snapshot.scope_conflict(question, context_doc) else snapshot


def institution_statistic_reply(ws, question, shown=20):
    """A statistic question about an institution with a separate snapshot. B's structured facts there are per-officer
    monthly totals of 임원 업무추진비 (two-reader verified; `not_for`: staff meals, averages, rule limits). They are
    listed as they are — filtered by the officer and month the question names — only when the question is about
    officers' 업무추진비 and not about staff, meals or travel; no average or sum is computed. Any other statistic
    question gets what exists, told apart: rules (규정상 기준), records (실지출 문서), officer-month values."""
    name = next(iter(ws.meta.values()))['institution']
    roles = {r.group(1) for f in ws.facts for r in [re.search(r'\((.+?)\)', f.get('population') or '')] if r}
    asked = [r for r in roles if r in question]
    staff = re.search(r'직원|식비|출장|여비|1인|인당|개인', question)
    officers = asked or re.search(r'임원|업무추진비', question)
    month = re.search(r'(\d{1,2})\s*월', question)
    base = {'candidates': [], 'checked_chunks': 0, 'warnings': [], 'reason': 'statistic',
            'scope': f'모델을 실행하지 않았습니다. {name} 별도 게시 자료 색인({ws.name})의 정형 값과 문서 종류만 확인했습니다. '
                     '평균·합계는 계산하지 않습니다.'}
    if ws.facts and officers and not staff and not asked:
        return {**base, 'status': 'clarify', 'reason': 'officer',
                'answer': (f'{name}의 공개 집행액은 임원 직위별·월별 값입니다. 어느 직위의 몇 월 집행액인지 정해 주세요: '
                           + ', '.join(sorted(roles)) + '. 여러 직위·여러 달을 묶은 평균·합계는 계산하지 않습니다.')}
    if ws.facts and officers and not staff:
        facts = sorted((f for f in ws.facts
                        if (not asked or any(f'({r})' in f['population'] for r in asked))
                        and (not month or f['period']['from'][5:7] == f'{int(month.group(1)):02d}')),
                       key=lambda f: (f['period']['from'], f['population']))
        if facts:
            lines = '\n'.join(f"- {f['period']['from'][:7]} · {f['population']} · {f['value']:,}{f['unit']}" for f in facts[:shown])
            more = f'\n(외 {len(facts) - shown}건)' if len(facts) > shown else ''
            return {**base, 'status': 'answer_values', 'facts': facts,
                    'answer': (f"{name} {facts[0]['metric']} 정형 값 {len(facts)}건입니다(B 검증: {facts[0].get('verification')}). "
                               f"각 값은 해당 임원 1명의 한 달 집행액이며 {facts[0].get('not_for') or '다른 용도로 쓰지 않음'}. "
                               f'평균·합계는 계산하지 않았습니다.\n{lines}{more}')}
    rules = [m for m in ws.meta.values() if m.get('record_kind') == 'institution_rule']
    spend = sorted(m['date'] for m in ws.meta.values() if m.get('record_kind') == 'expenditure_record')
    have = [f"기관 내규 게시본 {len(rules)}건(규정상 기준: {', '.join(sorted({m['title'].split('_')[-1].split(' 일부')[0].split('(')[0] for m in rules}))})"
            if rules else None,
            f'임원 업무추진비 집행내역 {len(spend)}건({spend[0]}~{spend[-1]} 월별 실지출 공개 문서)' if spend else None,
            f'임원 1명별 월 업무추진비 집행액 정형 값 {len(ws.facts)}건' if ws.facts else None]
    answer = (f'{name}에 대해 이 질문에 맞는 계산된 통계 값(직원 평균 등)은 검색 자료에 없습니다. 있는 자료: '
              + ', '.join(h for h in have if h) + '. '
              + ('집행내역과 그 정형 값은 임원(원장·감사 등)의 업무추진비 지출이며 직원 평균 식비나 출장 식비가 아니므로, 이 '
                 '자료로 평균·합계를 계산해 답하지 않습니다. ' if spend or ws.facts else '')
              + ('출장 식비의 규정상 기준을 찾으려면 기관명과 규정명(예: 여비규정 식비)으로 물어 주세요.' if rules else ''))
    return {**base, 'status': 'not_found', 'reason': 'no_structured_statistic', 'answer': answer}


def workspace_reply(workspace, selection):
    """Reply without a model call when the question cannot be tied to one institution's rule in the corpus."""
    status = selection['status']
    if status == 'clarify' and selection.get('reason') == 'institution':
        names = selection['unknown_institutions']
        options = '\n'.join(f"- {((workspace.institutions.get(n) or {}).get('alio') or {}).get('name') or n}"
                            for n in selection['institutions'])
        answer = (f"질문의 {', '.join(repr(n) for n in names)}이(가) 어느 기관인지 공식 기관 목록에서 하나로 정해지지 않습니다. "
                  f"다음 중 어느 기관인가요?\n{options}\n정식 기관명을 넣어 다시 물어봐 주세요.")
    elif status == 'clarify':
        options = '\n'.join(f"- {c['institution'] or '기관 표기 없음'}: {c['title']}" for c in selection['candidates'])
        answer = ('이 질문은 여러 기관의 규정에서 비슷하게 검색됩니다. 기관마다 기준이 다를 수 있어 임의로 한 곳을 고르지 '
                  f'않았습니다.\n{options}\n어느 기관(또는 어떤 규정) 기준인지 질문에 기관명이나 규정명을 넣어 다시 물어봐 주세요.')
    elif status == 'unknown_institution' and selection.get('reason') == 'unrecognized':
        answer = (f"질문의 {', '.join(repr(n) for n in selection['unknown_institutions'])}을(를) 공식 기관 목록과 검색 자료에서 "
                  '찾지 못했습니다. 자료가 없다는 뜻은 아닐 수 있으니 정식 기관명으로 다시 물어봐 주세요. 다른 기관의 규정으로 '
                  '대신 답하지 않습니다.')
    elif status == 'not_found' and selection.get('reason') == 'protected':
        answer = ('이 질문과 가장 관련된 구간은 개인정보 검토로 공개 대상에서 제외되어 있어 답하지 않습니다. 같은 문서의 다른 '
                  '구간이나 다른 자료로 대신 답하지 않습니다.')
    elif status == 'unknown_institution':
        described, successors = [], []
        for name in selection['unknown_institutions']:
            entry = workspace.institutions.get(name) or {}
            alio = entry.get('alio')
            described.append(f"{alio['name']}({alio['type']}, 주무부처 {alio['ministry']})" if alio else f"'{name}'")
            if entry.get('successor'):
                successors.append(f"'{name}'은(는) 기관코드상 {entry['successor']}(으)로 승계된 옛 기관명입니다. 옛 기관의 규정을 "
                                  f"현 기관에 그대로 적용하지 않으니, {entry['successor']} 기준을 원하면 그 이름으로 물어 주세요.")
        answer = (f"{', '.join(described)}의 규정은 현재 검색 자료에 없습니다. 다른 기관의 규정으로 대신 답하지 않습니다. "
                  + (' '.join(successors) + ' ' if successors else '')
                  + '해당 기관 규정 문서를 첨부(＋)로 넣으면 그 문서에서 찾을 수 있습니다. 기관을 정하지 않고 공공기관 '
                  '일반 기준을 찾으려면 기관명 없이 물어 주세요.')
    elif selection.get('reason') == 'irrelevant':
        words = ', '.join(f"'{w}'" for w in selection['retrieval']['topical_words'])
        where = f"'{', '.join(selection['institutions'])}' 자료" if status == 'institution_no_match' else '검색 자료'
        answer = (f"{where}에서 이 질문에 답할 만큼 관련된 구간을 찾지 못했습니다"
                  + (f"(핵심어 {words}을 충분히 다루는 구간 없음)" if words else '') + '. '
                  '관련 없는 문서로 대신 답하지 않습니다. 기관명이나 규정명을 넣거나 다른 표현으로 물어 주세요.')
    elif status == 'institution_no_match':
        names = ', '.join(selection['institutions'])
        answer = (f"'{names}' 자료에서 이 질문과 관련된 규정을 찾지 못했습니다. 다른 기관의 규정으로 대신 답하지 않습니다.")
    else:
        answer = '검색 자료에서 질문과 관련된 구간을 찾지 못했습니다. 질문을 바꾸거나 기관명·규정명을 넣어 주세요.'
    return {'status': status, 'answer': answer, 'candidates': [], 'checked_chunks': 0, 'warnings': [],
            'reason': selection.get('reason'), 'retrieval': selection.get('retrieval'),
            'scope': '모델을 실행하지 않았습니다. 검색 자료와 질문의 기관·규정명, 근거 적합성을 확인한 결과입니다.'}


PUBLIC_FIELDS = ('id', 'label', 'kind', 'base_model', 'license', 'version', 'status', 'evaluation', 'limits')


class State:
    """Serves the versions listed in a versions file. The first `extractive` entry supplies the chunking tokenizer for
    uploaded documents; `llama` entries talk to a loopback llama.cpp server whose model file hash must match the
    file (versions on the same model file share one server). The entry marked `"default": true` (else the first) is
    preselected on the page."""
    def __init__(self, versions):
        self.versions, self.children, servers = {}, [], {}
        if not any(v['kind'] == 'extractive' for v in versions):
            raise ValueError('추출 독해기 버전이 하나는 있어야 합니다(업로드 문서 구간 분할 토크나이저).')
        try:
            for meta in versions:
                if meta['kind'] == 'extractive':
                    metadata = json.loads((ROOT / meta['run'] / 'training.json').read_text())
                    if metadata.get('status') != 'complete':
                        raise ValueError('학습 완료 모델만 공유할 수 있습니다.')
                    model, tok, device = load_reader(ROOT / meta['run'], device='cpu')
                    self.versions[meta['id']] = dict(meta, model=model, tok=tok, device=device)
                elif meta['kind'] == 'llama':
                    key = (meta['model_file'], meta['llama_server'], meta['context'])
                    if key not in servers:
                        servers[key] = self.start_llama(meta)
                    self.versions[meta['id']] = dict(meta, **servers[key])
                else:
                    raise ValueError(f"unknown version kind {meta['kind']}")
        except BaseException:
            self.close()
            raise
        self.tok = next(v['tok'] for v in self.versions.values() if v['kind'] == 'extractive')
        self.default = next((v['id'] for v in versions if v.get('default')), versions[0]['id'])
        self.run = self.versions[self.default]['label']
        self.sessions = {}
        self.lock = threading.RLock()
        self.work = threading.Lock()

    def start_llama(self, meta):
        model_file = ROOT / meta['model_file']
        if digest(model_file) != meta['model_sha256']:
            raise ValueError(f"{meta['id']}: model file hash mismatch")
        port = meta['port']
        try:
            llama_request(f'http://127.0.0.1:{port}', '/health')
            raise RuntimeError(f"{meta['id']}: port {port} is already serving; refusing to adopt an unmanaged server")
        except OSError:
            pass
        runtime = Path(os.environ.get('TRID_DEMO_RUNTIME', ROOT / '.shared-demo'))  # an isolated instance logs apart
        log = open(runtime / f"llama-{meta['id']}.log", 'wb') if runtime.is_dir() else subprocess.DEVNULL
        child = subprocess.Popen([str(ROOT / meta['llama_server']), '-m', str(model_file), '--host', '127.0.0.1', '--port', str(port),
                                  '-c', str(meta['context']), '-np', '1', '-t', str(meta.get('threads', 4)),
                                  '-tb', str(meta.get('threads', 4)), '--jinja'],
                                 stdout=log, stderr=log)
        self.children.append(child)
        base = f'http://127.0.0.1:{port}'
        for _ in range(240):
            if child.poll() is not None:
                raise RuntimeError(f"{meta['id']}: llama-server exited")
            try:
                props = llama_request(base, '/props')
                if Path(props['model_path']).resolve() == model_file.resolve():
                    return {'base': base, 'context': props['default_generation_settings']['n_ctx']}
                raise RuntimeError(f"{meta['id']}: another model is served on port {port}")
            except OSError:
                time.sleep(.5)
        raise RuntimeError(f"{meta['id']}: llama-server readiness timeout")

    def public_versions(self):
        return [{k: v[k] for k in PUBLIC_FIELDS} for v in self.versions.values()]

    def close(self, grace=10):
        """Stop the llama.cpp servers: all are signalled at once and share one `grace` period (sequential waits used
        to outlast the supervisor's wait for this process), then any still running is killed and reaped."""
        running = [c for c in self.children if c.poll() is None]
        for child in running:
            child.terminate()
        deadline = time.monotonic() + grace
        for child in running:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def prune(self):
        with self.lock:
            now = time.monotonic()
            for sid in list(self.sessions):
                if now - self.sessions[sid].touched > TTL:
                    del self.sessions[sid]


def parse_pdf(body, name, tok):
    # Parsing untrusted PDFs happens in a separate process with a wall-clock timeout.
    try:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / 'scripts/shared_pdf_extract.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=ROOT,
        )
        output = []
        def communicate():
            output.append(process.communicate(body)[0])
        communication = threading.Thread(target=communicate, daemon=True)
        communication.start()
        deadline = time.monotonic() + 15
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise DocumentError('PDF 처리 시간이 초과됐습니다. 더 작은 문서를 넣어 주세요.')
                if sys.platform == 'darwin':
                    memory = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, timeout=2).stdout.strip()
                    if memory and int(memory) > 512 * 1024:
                        raise DocumentError('PDF 처리 메모리 한도를 넘었습니다. 더 작은 문서를 넣어 주세요.')
                communication.join(.1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            communication.join(3)
        if process.returncode or not output:
            raise DocumentError('PDF를 읽지 못했습니다. 작은 텍스트 PDF로 다시 시도해 주세요.')
        result = json.loads(output[0])
    except (subprocess.SubprocessError, ValueError):
        raise DocumentError('PDF를 읽지 못했습니다. 텍스트가 있는 작은 PDF로 다시 시도해 주세요.')
    if 'error' in result:
        raise DocumentError(result['error'])
    name = name.replace('\\', '/').rsplit('/', 1)[-1][:180] or 'document.pdf'
    return Document(name, result['pages'], make_chunks(result['pages'], tok), result['warnings'])


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args)

    def process_request(self, request, address):
        if not self.slots.acquire(False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def make_server(state, secret, port=0, hostname='test.tridatum.co'):
    if len(secret) < 32:
        raise ValueError('Proxy secret must contain at least 32 characters')
    assets = ROOT / 'trid/shared_ui'
    origin = 'https://' + hostname
    workspace = getattr(state, 'workspace', None)
    workspace_scope = html.escape(workspace.scope()) if workspace else '공유 검색 자료가 없습니다. 문서를 직접 넣어 주세요.'
    version_options = ''.join(
        f'<option value="{html.escape(v["id"], quote=True)}"{" selected" if v["id"] == state.default else ""}>'
        f'{html.escape(v["label"])}</option>'
        for v in state.public_versions()
    )
    version_cards = ''.join(
        f'<pre class="muted version-card" data-version="{html.escape(v["id"], quote=True)}"'
        f'{"" if v["id"] == state.default else " hidden"}>'
        + html.escape(f'{v["label"]} · {v["version"]} · {v["status"]}\n기반: {v["base_model"]} · 라이선스: {v["license"]}\n'
                      f'실측: {v["evaluation"]}\n한계: {v["limits"]}') + '</pre>'
        for v in state.public_versions()
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = 'trid'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass

        def authenticated(self):
            supplied = self.headers.get('X-Trid-Proxy', '')
            return (supplied.isascii() and hmac.compare_digest(supplied, secret)
                    and self.headers.get('X-Trid-Host') == hostname)

        def respond(self, status, value, mime='application/json; charset=utf-8', cookie=None):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            for key, val in {
                'Content-Type': mime, 'Content-Length': str(len(body)), 'Cache-Control': 'no-store',
                'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                'Referrer-Policy': 'no-referrer', 'Connection': 'close',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'",
            }.items():
                self.send_header(key, val)
            if cookie:
                self.send_header('Set-Cookie', f'__Host-trid={cookie}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age={TTL}')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def session(self, create=False):
            jar = SimpleCookie()
            try:
                jar.load(self.headers.get('Cookie', ''))
                sid = jar['__Host-trid'].value if '__Host-trid' in jar else ''
            except Exception:
                sid = ''
            with state.lock:
                state.prune()
                session = state.sessions.get(sid)
                if session is None and create and len(state.sessions) < MAX_SESSIONS:
                    sid = secrets.token_urlsafe(32)
                    session = Session()
                    state.sessions[sid] = session
                if session:
                    session.touched = time.monotonic()
                return sid, session

        def do_GET(self):
            if not self.authenticated():
                self.respond(403, {'error': '허용되지 않은 연결입니다.'}); return
            path = urlsplit(self.path).path
            if path == '/healthz':
                self.respond(200, {'ok': True, 'model': state.run, 'versions': list(state.versions)}); return
            if path == '/app.js':
                self.respond(200, (assets / 'app.js').read_bytes(), 'text/javascript; charset=utf-8'); return
            if path != '/':
                self.respond(404, {'error': '없는 페이지입니다.'}); return
            sid, session = self.session(create=True)
            if session is None:
                self.respond(503, {'error': '현재 테스트 인원이 많습니다. 잠시 후 다시 접속해 주세요.'}); return
            page = ((assets / 'index.html').read_text()
                    .replace('TOKEN_PLACEHOLDER', session.csrf)
                    .replace('MODEL_PLACEHOLDER', html.escape(state.run))
                    .replace('VERSION_OPTIONS_PLACEHOLDER', version_options)
                    .replace('VERSION_CARDS_PLACEHOLDER', version_cards)
                    .replace('WORKSPACE_SCOPE_PLACEHOLDER', workspace_scope))
            self.respond(200, page.encode(), 'text/html; charset=utf-8', cookie=sid)

        def do_POST(self):
            if not self.authenticated() or self.headers.get('Origin') != origin:
                self.respond(403, {'error': '허용되지 않은 요청입니다.'}); return
            sid, session = self.session()
            token = self.headers.get('X-Trid-Token', '')
            if session is None or not token.isascii() or not hmac.compare_digest(token, session.csrf):
                self.respond(403, {'error': '세션이 만료됐습니다. 페이지를 새로고침해 주세요.'}); return
            with state.lock:
                now = time.monotonic()
                session.calls = [t for t in session.calls if now - t < 60]
                if len(session.calls) >= 20:
                    self.respond(429, {'error': '요청이 많습니다. 1분 뒤 다시 시도해 주세요.'}); return
                session.calls.append(now)
            try:
                try:
                    size = int(self.headers.get('Content-Length', '-1'))
                except ValueError:
                    size = -1
                if self.headers.get('Transfer-Encoding') or not 0 <= size <= MAX_BODY:
                    self.respond(413, {'error': '입력은 2MB 이하로 넣어 주세요.'}); return
                body = self.rfile.read(size)
                if len(body) != size:
                    raise DocumentError('입력 전송이 완료되지 않았습니다.')
                path = urlsplit(self.path).path
                if not state.work.acquire(timeout=2):
                    self.respond(503, {'error': '다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.'}); return
                try:
                    if path in {'/api/load', '/api/sample', '/api/text'}:
                        if path == '/api/load':
                            doc = parse_pdf(body, unquote(self.headers.get('X-File-Name', 'document.pdf')), state.tok)
                        elif path == '/api/sample':
                            content = (ROOT / 'data/reading-demo-v3/travel-named.txt').read_text()
                            doc = Document('가상 출장 규정 · 샘플', [content], make_chunks([content], state.tok), ['시연용 가상 자료입니다. 실제 기관 규정이 아닙니다.'])
                        else:
                            request = json.loads(body)
                            content = request.get('text') if isinstance(request, dict) else None
                            if not isinstance(content, str) or not 10 <= len(content.strip()) <= 5000:
                                raise DocumentError('문서는 10~5,000자로 입력해 주세요.')
                            import unicodedata
                            content = unicodedata.normalize('NFC', content.strip())
                            doc = Document('직접 입력한 문서', [content], make_chunks([content], state.tok), [])
                        with state.lock:
                            session.document = doc
                            session.document_id = secrets.token_urlsafe(24)
                        result = {'document_id': session.document_id, 'name': doc.name, 'pages': doc.pages, 'warnings': doc.warnings}
                    elif path == '/api/pin':
                        if workspace is None:
                            raise DocumentError('공유 검색 자료가 준비되지 않았습니다.')
                        request = json.loads(body)
                        choice = request.get('choice') if isinstance(request, dict) else None
                        with state.lock:
                            if choice is None:
                                session.pinned_doc = None
                            elif isinstance(choice, int) and 1 <= choice <= len(session.offered_docs):
                                session.pinned_doc = session.offered_docs[choice - 1]
                            else:
                                raise DocumentError('직전 답변에 표시된 다른 후보 문서 번호를 골라 주세요.')
                        result = {'pinned': workspace.describe(session.pinned_doc) if session.pinned_doc else None}
                    elif path == '/api/clear':
                        # a new chat starts with no context: the document, the conversation, and the pinned/offered
                        # documents of the previous chat (a pin kept answering from that document; MODEL-RESUME-A-024)
                        with state.lock:
                            session.document = session.document_id = session.pinned_doc = None
                            session.offered_docs = []
                            session.history.clear()
                        result = {'cleared': True}
                    elif path == '/api/ask':
                        request = json.loads(body)
                        if not isinstance(request, dict):
                            raise DocumentError('질문 형식을 확인해 주세요.')
                        workspace_mode = request.get('scope') == 'workspace'
                        if workspace_mode and workspace is None:
                            raise DocumentError('공유 검색 자료가 준비되지 않았습니다. 문서를 직접 넣어 주세요.')
                        if not workspace_mode and (session.document is None or request.get('document_id') != session.document_id):
                            self.respond(409, {'error': '이 브라우저에서 문서를 먼저 넣어 주세요.'}); return
                        question = request.get('question')
                        if not isinstance(question, str) or len(question) > 500:
                            raise DocumentError('질문은 500자 이하로 입력해 주세요.')
                        version_id = request.get('version', state.default)
                        if not isinstance(version_id, str) or version_id not in state.versions:
                            raise DocumentError('목록에 있는 모델 버전을 선택해 주세요.')
                        version = state.versions[version_id]
                        started = time.monotonic()
                        history = list(session.history)
                        intent, kind = 'search', None
                        last = next((h for h in reversed(history) if h['role'] == 'assistant'), None)
                        context_doc = last.get('applied_id') if last and last.get('kind') == 'search' else None
                        selected, others, selection = None, [], None
                        document = session.document
                        ws = workspace  # the shared index, or a named institution's separate snapshot index
                        inst_ws = institution_indexes(state, workspace, question, context_doc) if workspace_mode else None
                        continues = ((lambda q, d: (inst_ws or workspace).continues(q, d)) if workspace_mode else None)
                        explain = False
                        if version.get('assistant'):
                            kind = request_type(question)
                            if kind in ('organize', 'recall'):
                                # the visitor's own figures, or this conversation itself: answered from its turns
                                intent = 'chat'
                            else:
                                # Markers (statistic, definition) only refine a work question: the router decides
                                # first, so "인생의 의미가 뭘까?" or "요즘 실제로 쓰는 앱" stay conversation.
                                intent = route(version['base'], version['family'], version['context'], history, question,
                                               continues)
                                if intent == 'search' and kind == 'explain' and workspace_mode and not inst_ws:
                                    # A definition is answered from the corpus when it covers the term (견책 →
                                    # 징계업무편람); only otherwise as a general explanation citing no institution.
                                    selection = workspace.select(question, session.pinned_doc, context_doc)
                                    if selection['status'] not in ('answer', 'clarify'):
                                        intent, selection, explain = 'chat', None, True
                        if intent == 'chat' and kind == 'recall' and not history:
                            result = no_history_reply()
                        elif intent == 'chat':
                            result = chat_reply(version['base'], version['family'], version['context'],
                                                conversation_turns(history, kind == 'recall'), question, explain=explain,
                                                first_turn=not history)
                        elif workspace_mode and ((kind == 'statistic' and not (inst_ws and not inst_ws.facts
                                                                               and not COMPUTED.search(question)))
                                                 or (inst_ws and inst_ws.facts and '업무추진비' in question)):
                            # an institution without structured facts: asking for a computed figure gets what exists
                            # instead; asking what a record lists ("9월 집행내역") is searched in its documents
                            result = (institution_statistic_reply(inst_ws, question) if inst_ws
                                      else statistic_reply(workspace, question))
                        else:
                            if workspace_mode:
                                if inst_ws:
                                    # a named institution whose posted rules/records are a separate snapshot (B: never in
                                    # the shared or evaluation index) is answered from that index only
                                    ws = inst_ws
                                    selection = inst_ws.select(question, None, context_doc if context_doc in inst_ws.meta else None)
                                else:
                                    selection = selection or workspace.select(question, session.pinned_doc, context_doc)
                                others = selection['others']
                            if (version.get('assistant') and selection and selection['status'] == 'not_found'
                                    and selection.get('reason') != 'protected' and not inst_ws and not session.pinned_doc
                                    and not (last and last.get('kind') == 'search')
                                    and not (selection.get('institutions') or selection.get('unknown_institutions')
                                             or selection.get('title_matched'))
                                    and not RULE_CLAIM.search(question)):
                                # The router sent it to search, nothing in the corpus answers it, it does not follow a
                                # search turn, and it names no institution, rule or rule topic: everyday talk the
                                # router misread ("마지막 문장은 빼줘", "커피 몇 잔까지 괜찮을까?"), answered as
                                # conversation and said so — a search failure is not a reply to it. Work questions
                                # and follow-ups of a search ("다른 등급은?") keep the not-found reply.
                                intent, selection, others = 'chat', None, []
                                result = chat_reply(version['base'], version['family'], version['context'],
                                                    conversation_turns(history, False), question, first_turn=not history)
                                result['scope'] = ('업무 자료에서 맞는 근거를 찾지 못했고 업무 기준을 묻는 말로 보이지 않아 일반 대화로 '
                                                   '답했습니다. ' + result['scope'])
                            elif selection and selection['status'] != 'answer':
                                result = workspace_reply(ws, selection)
                            else:
                                held_out = 0
                                if selection:
                                    selected = selection['chunks']
                                    if version['kind'] == 'extractive' or version.get('answer_mode') != 'case':
                                        # versions whose answer is a value: held passages' numbers are not values, so
                                        # those passages are not given to the model at all
                                        held_out = sum(c.value_hold for c, _ in selected)
                                        selected = [(c, s) for c, s in selected if not c.value_hold and not c.derived]
                                    if version['kind'] == 'extractive':
                                        selected = reader_windows(selected, version['tok'])
                                    document = Document(ws.meta[selection['applied']]['title'], [],
                                                        [c for c, _ in selected], [])
                                if version['kind'] == 'extractive':
                                    result = answer_document(document, question, version['model'], None, version['tok'],
                                                             version['device'], reader=read_chunk, selected=selected)
                                    if any(item['prediction']['presence'] == 2 for item in result['checked']):
                                        result.update(status='review', answer='모델이 근거 충돌로 판단한 구간이 있습니다. 원문을 직접 확인해 주세요.')
                                elif version.get('answer_mode') == 'case':
                                    result = answer_document_case(document, question, version['base'], version['family'],
                                                                  version['context'], selected=selected,
                                                                  topical=selection['retrieval']['topical_words'] if selection else None,
                                                                  provenance=ws.provenance(selection['applied'])
                                                                  if selection and version.get('provenance_line') else None)
                                else:
                                    result = answer_document_chat(document, question, version['base'], version['family'],
                                                                  version['context'], version['decoding'] == 'evidence_constrained',
                                                                  selected=selected)
                                result.pop('checked', None)
                                if held_out:
                                    hold_note = (f'숫자 추출이 원문 쪽과 일치하지 않는 것으로 검토된 구간 {held_out}개는 값으로 답하지 않도록 '
                                                 '모델에 넣지 않았습니다. 그 구간의 숫자는 원문(쪽)에서 직접 확인하세요.')
                                    if not selected:
                                        result.update(status='review', reason='value_hold', candidates=[],
                                                      answer='관련 구간의 숫자가 원문 쪽과 일치하지 않는 것으로 검토되어 값으로 답하지 '
                                                             '않습니다. 원문(쪽)에서 직접 확인하세요.')
                                    result.setdefault('commentary_notes', []).append(hold_note)
                        result['intent'] = intent
                        if kind:
                            result['request_type'] = kind
                        if selection and selection.get('retrieval'):
                            result['retrieval'] = {**selection['retrieval'], 'index': ws.name,
                                                   'dense': ws.dense_info and {k: ws.dense_info[k] for k in ('model', 'snapshot', 'vectors_sha256')}}
                        if selection:
                            with state.lock:
                                session.offered_docs = [o['doc_id'] for o in others] if ws is workspace else []
                            for c in result.get('candidates', []):
                                chunk = next(ch for ch, _ in selected if ch.text == c['text'])
                                c['source_label'] = ' · '.join(x for x in (
                                    ws.meta[chunk.doc_id]['institution'], ws.meta[chunk.doc_id]['title'],
                                    ' '.join(chunk.path[1:]) or None) if x)
                                c['page_character_span'] = [chunk.start + c['prediction']['source_character_span'][0],
                                                            chunk.start + c['prediction']['source_character_span'][1]]
                            applied = selection['applied']
                            result['workspace'] = {'applied': ws.describe(applied) if applied else None,
                                                   'pinned': bool(session.pinned_doc), 'others': others,
                                                   'selection': selection['status'], 'institutions': selection['institutions'],
                                                   'title_matched': selection['title_matched'],
                                                   'continued': bool(selection.get('continued')), 'scope': ws.scope(),
                                                   'separate_snapshot': ws.manifest.get('separate') or False}
                            result['total_chunks'] = len(ws.index.passages)
                            if selection['status'] == 'answer' and any(c.table for c, _ in selected):
                                result['table_warning'] = ('표가 들어 있는 구간을 썼습니다. 표의 행·열 대응은 검증되지 않았으므로 표 값을 '
                                                           '해석하거나 계산한 내용은 근거로 쓰지 말고 원문 링크에서 표를 직접 확인하세요.'
                                                           + (' 표의 〃(위와 같음) 표시는 원문 그대로이며 다른 값으로 바꿔 읽지 '
                                                              '않았습니다.' if any('〃' in c.text for c, _ in selected) else ''))
                            if selection['status'] == 'answer':
                                how = ('질문에 적힌 문서명' if selection['title_matched'] else
                                       f"질문에 적힌 기관({', '.join(selection['institutions'])})" if selection['institutions'] else
                                       '앞선 대화에서 쓴 문서(질문에 기관·문서명이 없어 이어서 찾음)' if selection.get('continued') else
                                       '관련도 판정')
                                result['scope'] = (f'{how}으로 고른 문서 1건의 구간만 모델에 넣었습니다. 다른 기관·문서의 규정을 '
                                                   '섞지 않습니다. 검색되지 않은 자료까지 정보가 없다고 보장하지 않습니다.')
                                m = ws.meta[selection['applied']]
                                scope_text = ws.applies(selection['applied']).get('note')
                                if scope_text:
                                    result['scope'] += f" 이 자료의 적용 범위(B 기록): {scope_text}."
                                elif selection.get('scope_note') == 'specific':
                                    noun = '규정' if re.search(r'법령|규정|규칙|훈령|예규|지침|행정규칙', f"{m['kind']} {m['title']}") else '자료'
                                    result['scope'] += (f" 이 문서는 {m['institution'] or '특정 기관'}의 자체 {noun}(또는 그 소속·특정 대상 "
                                                        f'{noun})입니다. 질문에 기관을 정하지 않았고 공통 규정은 찾지 못했으므로, 질문한 대상에 '
                                                        f"이 {noun}{'이' if noun == '규정' else '가'} 적용되는지는 확인되지 않았습니다.")
                                for note in m.get('limitations') or []:
                                    result['scope'] += ' 자료 한계: ' + note
                                if m.get('limitations'):
                                    result['source_limits'] = list(m['limitations'])  # for a visible warning (UI: B)
                                # a 별표 has no record of its own: its rule's applicability (공무원 여비 규정 [별표 8] → 국가공무원)
                                applies = rule_scope(ws, selection['applied'])
                                central = (applies == 'central_government' and selection['institutions']
                                           and all(n in ws.central_bodies for n in selection['institutions']))
                                if central and '공무원' not in question:
                                    # 인사처 → 인사혁신처, a 중앙행정기관: the rule covers its 국가공무원 — not every person
                                    # working there (공무직 and other non-civil-servant staff are not established here)
                                    names = ', '.join(selection['institutions'])
                                    result['applicability'] = {'rule_scope': applies, 'named_central_body': selection['institutions'],
                                                               'status': 'civil_servants_only'}
                                    result['answer'] = (f"「{m['title']}」은 국가공무원에게 적용되는 규정입니다(제1조 목적). {names} 소속 "
                                                        '국가공무원 기준이며, 같은 기관의 공무직 등 공무원이 아닌 인력에게 적용되는지는 '
                                                        '확인되지 않았습니다. ' + (result.get('answer') or ''))
                                elif applies in ('central_government', 'local_government') and '공무원' not in question:
                                    applicability_guard(result, applies, m['title'])
                                if m.get('record_kind') == 'expenditure_record':
                                    expenditure_guard(result, m)
                                if ws is not workspace:
                                    typo = ('원문 부칙의 시행일 표기를 그대로 옮겼으며 오기 여부는 확인하지 않았습니다. '
                                            if '오기' in (m.get('date_type') or '') else '')
                                    result['answer'] = (f"{m['institution']} 누리집 게시 자료(「{m['title']}」, {m['date']} · "
                                                        f"{m['currency']})에서 찾았습니다. {typo}" + (result.get('answer') or ''))
                                for note in m.get('source_notes') or []:
                                    result['scope'] += f' 출처 메모(B): {note}'
                                if selection.get('reviewed_rows'):
                                    rr = selection['reviewed_rows']
                                    result['scope'] += (f" 원문 표 구간과 함께 B가 원본 렌더링과 대조한 별표 {rr['annex']}의 표 행(단위·비고·"
                                                        f"시점 포함, {rr['rows']}행; 원문 위치 확인 방식 {rr['method']})을 넣었습니다. 이 행은 "
                                                        '원문 인용으로 표시하지 않으며, 승인된 값이나 법률 자문이 아닙니다.')
                                if selection.get('table_rows_unlinked'):
                                    result['scope'] += (f" {', '.join(selection['table_rows_unlinked'])}의 검토된 표 행은 현재 색인 문서의 원문 "
                                                        '위치와 맞는지 확인하지 못해 넣지 않았습니다(다른 별표로 대신하지 않음). 표 값은 원문 '
                                                        '표에서 확인하세요.')
                                if selection.get('table_rows_not_in_evidence'):
                                    result['scope'] += (f" 질문이 가리키는 {', '.join(selection['table_rows_not_in_evidence'])} 구간이 검색 근거에 "
                                                        '없어 그 표의 검토된 행을 넣지 않았고, 다른 별표의 행으로 대신하지도 않았습니다. 정보가 '
                                                        '부족합니다. 별표 번호나 표 이름을 넣어 다시 물어 주세요.')
                                lacking = (selection.get('table_rows_not_in_evidence') or []) + (selection.get('table_rows_unlinked') or [])
                                if lacking and not selection.get('reviewed_rows'):
                                    result['answer'] = (result.get('answer') or '') + (
                                        f" {', '.join(dict.fromkeys(lacking))}의 표 값은 이번 근거에 확인된 형태로 들어 있지 않아 정보가 "
                                        '부족합니다. 다른 별표 값으로 대신 답하지 않았으니 원문 표를 확인하세요.')
                                for note in result.get('context_notes') or []:
                                    result['scope'] += ' ' + note
                                if selection.get('linked_tables'):
                                    result['scope'] += (' 본문이 인용한 같은 규정의 별표 구간을 함께 넣었습니다'
                                                        f"({len(selection['linked_tables'])}건).")
                                if selection.get('standard_choice'):
                                    sc = selection['standard_choice']
                                    result['scope'] += (f" 질문이 금액·기준을 물었고, 처음 고른 「{ws.meta[sc['from']]['title']}」의 구간이 "
                                                        f"그 금액을 「{ws.meta[sc['to']]['title']}」의 별표에 따르도록 정하고 있어 같은 기관의 "
                                                        '그 규정을 근거로 골랐습니다. 처리지침의 사례·공제 예시는 사례를 묻는 질문에 씁니다.')
                                if selection.get('table_condition'):
                                    # RETRIEVAL-ROOT-009: why the rule's body, not its best-scoring table, was used. The
                                    # caption nouns stay in the response field; the visible note names the table itself.
                                    tc = result['table_condition'] = selection['table_condition']
                                    result['scope'] += (f" 가장 가깝게 검색된 「{tc['table']}」는 질문에 적히지 않은 대상·조건에 "
                                                        f"관한 표라서, 그 표를 정한 규정 본문 「{tc['answered_from']}」을 근거로 "
                                                        '골랐습니다.')
                        population_notice(result, question, inst_ws, ws, selection)
                        if workspace_mode and result.get('status') != 'chat':
                            asked = _norm(question)
                            used = [{'alias': a, **info} for a, info in workspace.government_abbreviations.items()
                                    if a in asked and info['canonical'] not in asked]
                            if used:
                                # which version of the official abbreviation rule the name was read with
                                result['abbreviations'] = used
                                result['scope'] = (result.get('scope') or '') + ''.join(
                                    f" 약칭 인식: {u['alias']} → {u['canonical']}({u['source']} 기준)." for u in used)
                        result['model'] = version['label']
                        result['version'] = version_id
                        result['elapsed_ms'] = round((time.monotonic() - started) * 1000)
                        if result.get('raw_output') and any('숨겼습니다' in n for n in result.get('commentary_notes') or []):
                            # what the reply withholds is not sent in the API either
                            result.update(raw_output=None, raw_output_withheld=True)
                        if version.get('assistant'):
                            applied_doc = (result.get('workspace') or {}).get('applied')
                            if intent == 'chat':
                                said = result['answer']
                            else:
                                # Memory keeps only what is verified, with its provenance: the document the answer was
                                # chosen from and its verified quotes. Unverified commentary, ask-back lists and
                                # institution listings are not remembered, so the conversation model cannot replay them
                                # as facts.
                                quotes = ' '.join(f"“{c['prediction']['answer']}”" for c in result.get('candidates', [])[:3])
                                said = '[문서 검색 답변] ' + (
                                    f"근거 문서 {applied_doc['institution'] or ''} 「{applied_doc['title']}」, "
                                    + (f'확인된 원문: {quotes}' if quotes else '확인된 원문 인용 없음')
                                    if applied_doc and (result.get('workspace') or {}).get('selection') == 'answer'
                                    else '근거 문서를 정하지 않음(보류·되묻기·자료 없음)')
                            with state.lock:
                                session.history += [{'role': 'user', 'content': question, 'kind': intent},
                                                    {'role': 'assistant', 'content': said[:1500 if intent == 'chat' else 600],
                                                     'kind': intent,
                                                     'applied_id': applied_doc['doc_id'] if applied_doc else None}]
                                del session.history[:-HISTORY_MESSAGES]
                    else:
                        self.respond(404, {'error': '없는 기능입니다.'}); return
                finally:
                    state.work.release()
                self.respond(200, result, cookie=sid)
            except DocumentError as error:
                self.respond(422, {'error': str(error)})
            except (json.JSONDecodeError, UnicodeError):
                self.respond(422, {'error': '입력 형식을 확인해 주세요.'})
            except (ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception:
                # A server-side failure is not the visitor's input: logged for the operator, reported as such.
                traceback.print_exc()
                self.respond(500, {'error': '서버 처리 중 오류가 발생했습니다. 운영자가 확인해야 하는 문제입니다.'})

    return BoundedServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', type=Path, default=ROOT / 'deploy/shared-demo/versions.json',
                        help='JSON list of served versions; the first is the default and must be extractive')
    parser.add_argument('--secret-file', required=True)
    parser.add_argument('--port', type=int, default=51873)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'runs/workspace-index/v5',
                        help='shared search index of approved public sources (scripts/build_workspace_index.py)')
    parser.add_argument('--passage-review', type=Path, default=ROOT / 'data/search-corpus-v4/passages.jsonl',
                        help="B's per-passage review metadata (value_review: hold_for_value_use, privacy_review)")
    parser.add_argument('--institution-index', type=Path, action='append', default=None,
                        help='separate institution snapshot index (build_workspace_index.py --separate-snapshot), served '
                             'only for questions naming that institution (by its ALIO id); default: '
                             'runs/workspace-index/inst-reb-002 when present')
    args = parser.parse_args()
    torch.set_num_threads(4)
    def stop_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    state = None
    try:
        state = State(json.loads(args.versions.read_text()))
        state.workspace = (Workspace(args.workspace, review=args.passage_review)
                           if (args.workspace / 'manifest.json').exists() else None)
        state.institution_workspaces = {}
        for path in args.institution_index or [p for p in [ROOT / 'runs/workspace-index/inst-reb-002'] if p.exists()]:
            inst = Workspace(path, embedder=state.workspace.embedder)
            ids = {m.get('issuer_id') for m in inst.meta.values()}
            if len(ids) != 1 or not inst.manifest.get('separate'):
                raise ValueError(f'{path}: an institution index must be a separate snapshot of exactly one institution')
            state.institution_workspaces[ids.pop()] = inst
        server = make_server(state, Path(args.secret_file).read_text().strip(), args.port)
        stop = threading.Event()
        def prune():
            while not stop.wait(30):
                state.prune()
        threading.Thread(target=prune, daemon=True).start()
        print(f'trid temporary demo ready on 127.0.0.1:{server.server_port} versions={list(state.versions)}', flush=True)
        try:
            server.serve_forever()
        finally:
            stop.set()
            server.server_close()
            state.sessions.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if state is not None:
            state.close()


if __name__ == '__main__':
    main()
