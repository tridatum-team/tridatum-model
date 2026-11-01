import http.client
import json
import re
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.shared_demo import make_server, State, TTL, MAX_BODY, read_chunk, answer_document_chat, answer_document_case
from trid.pdf_documents import Document, make_chunks
from trid.pdf_documents import DocumentError
from trid.prompts import rag_prompt
from trid.extractive import passage_spans
from test_pdf_documents import CharTokenizer

# Tests use 'http://127.0.0.1:1' as a model server that does not exist and rely on the call failing at once. Where
# loopback connections are dropped instead of refused, an unpatched call (e.g. prompt_tokens) would wait for the
# request timeout; the failure is raised directly here, without a socket, so the suite does not depend on that.
_PLACEHOLDER_MODEL = 'http://127.0.0.1:1'


def setUpModule():
    import scripts.shared_demo as sd
    real = sd.llama_request

    def no_network(base, *args, **kwargs):
        if base == _PLACEHOLDER_MODEL:
            raise ConnectionRefusedError('placeholder model server (test)')
        return real(base, *args, **kwargs)
    patcher = patch('scripts.shared_demo.llama_request', side_effect=no_network)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class SharedDemoTests(unittest.TestCase):
    def setUp(self):
        self.secret = 'x' * 40
        meta = dict(id='v', label='test', kind='extractive', base_model='b', license='l', version='0', status='s',
                    evaluation='e', limits='<script>')
        self.state = SimpleNamespace(sessions={}, lock=threading.RLock(), work=threading.Lock(), tok=CharTokenizer(),
                                     run='test', default='v', versions={'v': meta})
        self.state.prune = lambda: State.prune(self.state)
        self.state.public_versions = lambda: State.public_versions(self.state)
        self.server = make_server(self.state, self.secret)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(1)

    def request(self, method, path, data=None, headers=None, trusted=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        h = {'X-Trid-Proxy': self.secret, 'X-Trid-Host': 'test.tridatum.co'} if trusted else {}
        h.update(headers or {})
        body = json.dumps(data).encode() if data is not None else None
        connection.request(method, path, body, h)
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def visitor(self):
        status, body, headers = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn('HttpOnly', headers['Set-Cookie'])
        self.assertIn('Secure', headers['Set-Cookie'])
        self.assertEqual(headers['Cache-Control'], 'no-store')
        token = re.search(rb'name="trid-token" content="([^"]+)"', body).group(1).decode()
        return {'Cookie': headers['Set-Cookie'].split(';')[0], 'X-Trid-Token': token, 'Origin': 'https://test.tridatum.co'}

    def test_proxy_csrf_origin_and_private_files(self):
        self.assertEqual(self.request('GET', '/', trusted=False)[0], 403)
        h = self.visitor()
        self.assertEqual(self.request('GET', '/runs/model.safetensors')[0], 404)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**h, 'Origin': 'https://evil.test'})[0], 403)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**h, 'X-Trid-Token': 'wrong'})[0], 403)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**h, 'Content-Length': str(MAX_BODY + 1)})[0], 413)

    def test_documents_are_isolated_and_clear_only_own(self):
        a, b = self.visitor(), self.visitor()
        loaded = json.loads(self.request('POST', '/api/sample', {}, a)[1])
        question = {'document_id': loaded['document_id'], 'question': '해외 식비는?'}
        self.assertEqual(self.request('POST', '/api/ask', question, b)[0], 409)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**b, 'X-Trid-Token': a['X-Trid-Token']})[0], 403)
        self.request('POST', '/api/sample', {}, b)
        sessions = list(self.state.sessions.values())
        sessions[0].pinned_doc, sessions[0].offered_docs = 'law', ['law', 't2']
        sessions[1].pinned_doc = 'law'
        self.request('POST', '/api/clear', {}, a)
        self.assertIsNone(sessions[0].document)
        self.assertEqual((sessions[0].pinned_doc, sessions[0].offered_docs, sessions[0].history), (None, [], []))
        self.assertIsNotNone(sessions[1].document)  # the other visitor's chat keeps its document and pin
        self.assertEqual(sessions[1].pinned_doc, 'law')
        self.assertEqual(self.request('POST', '/api/ask', question, a)[0], 409)

    def test_expired_session_and_invalid_input(self):
        h = self.visitor()
        self.assertEqual(self.request('POST', '/api/ask', [], h)[0], 422)
        self.assertEqual(self.request('POST', '/api/text', {'text': 'x'}, h)[0], 422)
        session = next(iter(self.state.sessions.values()))
        session.touched = time.monotonic() - TTL - 1
        self.state.prune()
        self.assertFalse(self.state.sessions)
        self.assertEqual(self.request('POST', '/api/sample', {}, h)[0], 403)

    def test_version_cards_are_escaped_and_unknown_version_is_rejected(self):
        status, body, _ = self.request('GET', '/')
        self.assertNotIn(b'<script>', body)
        h = self.visitor()
        loaded = json.loads(self.request('POST', '/api/sample', {}, h)[1])
        ask = {'document_id': loaded['document_id'], 'question': '해외 식비는?', 'version': 'missing'}
        self.assertEqual(self.request('POST', '/api/ask', ask, h)[0], 422)


class ChatBackendTests(unittest.TestCase):
    def ask(self, generated):
        text = '출장 담당은 운영지원팀입니다. 식비는 하루 3만 원입니다.'
        document = Document('d', [text], make_chunks([text], CharTokenizer()), [])
        with patch('scripts.shared_demo.complete', return_value={'generated': generated}):
            return answer_document_chat(document, '식비는 얼마?', 'http://127.0.0.1:1', 'qwen3', 4096), text

    def test_verified_value_is_answered_with_its_exact_span(self):
        for generated in ('3만 원 [D1]', '[D1] 3만 원', '[D1] 3만 원.'):
            result, text = self.ask(generated)
            self.assertEqual(result['status'], 'answer', generated)
            start, end = result['candidates'][0]['page_character_span']
            self.assertEqual(text[start:end], '3만 원')

    def test_two_source_first_pairs_are_each_verified(self):
        result, text = self.ask('[D1] 3만 원, [D1] 운영지원팀')
        self.assertEqual(result['status'], 'answer')
        self.assertEqual([text[a:b] for a, b in (c['page_character_span'] for c in result['candidates'])], ['3만 원', '운영지원팀'])
        result, _ = self.ask('[D1] 3만 원, [D1] 9만 원')  # one unverifiable value withholds the whole answer
        self.assertNotEqual(result['status'], 'answer')

    def test_value_absent_from_cited_passage_is_not_an_answer(self):
        for generated in ('5만 원 [D1]', '3만 원 [D2]', '3만 원', '[D1] 식비는 하루 3만 원이고 담당은 운영지원팀입니다, 확인'):
            result, _ = self.ask(generated)
            self.assertNotEqual(result['status'], 'answer', generated)
            self.assertEqual(result['candidates'], [])

    def test_abstention_and_conflict_are_not_answers(self):
        self.assertEqual(self.ask('확인 불가')[0]['status'], 'not_found')
        self.assertEqual(self.ask('근거 충돌')[0]['status'], 'review')

    def case(self, generated):
        text = '출장 담당은 운영지원팀입니다. 식비는 하루 3만 원입니다.'
        document = Document('d', [text], make_chunks([text], CharTokenizer()), [])
        with patch('scripts.shared_demo.complete', return_value={'generated': generated, 'truncated_output': False}):
            return answer_document_case(document, '식비는 얼마?', 'http://127.0.0.1:1', 'qwen3', 4096), text

    def test_case_mode_highlights_only_verbatim_sentences_and_never_verifies_commentary(self):
        result, text = self.case('[근거] [D1] 식비는 하루 3만 원입니다.\n따라서 하루 3만 원을 받습니다.')
        self.assertEqual(result['status'], 'quoted')
        self.assertFalse(result['commentary_verified'])
        self.assertEqual([text[a:b] for a, b in (c['page_character_span'] for c in result['candidates'])],
                         ['식비는 하루 3만 원입니다.'])

    def test_case_mode_withholds_commentary_sentences_with_invented_numbers(self):
        result, _ = self.case('식비는 하루 5만 원이며 담당자 번호는 02-3456-7890입니다.\n담당은 운영지원팀입니다.')
        self.assertEqual(result['status'], 'quoted')  # the second sentence is verbatim source text
        self.assertEqual(result['commentary'], '담당은 운영지원팀입니다.')
        self.assertNotIn('3456', result['commentary'])
        self.assertIn('3456', result['unsupported_numbers'])
        self.assertTrue(any('1개를 숨겼습니다' in n for n in result['commentary_notes']))

    def test_case_mode_collapses_repeated_sentences(self):
        result, _ = self.case('식비는 하루 3만 원입니다. 식비는 하루 3만 원입니다. 식비는 하루 3만 원입니다.')
        self.assertEqual(result['commentary'], '식비는 하루 3만 원입니다.')
        self.assertTrue(any('반복 2회' in n for n in result['commentary_notes']))

    def test_case_mode_reports_question_echo_as_failure(self):
        result, _ = self.case('질문: 식비는 얼마?\n식비는 하루 3만 원입니다.')
        self.assertEqual(result['status'], 'model_failure')
        self.assertEqual(result['candidates'], [])

    def test_qa_question_part_is_not_evidence(self):
        text = '질문: 강의료 100만원을 줘도 되나요?\n답변: 상한액은 직급별로 40만원입니다.'
        document = Document('d', [text], make_chunks([text], CharTokenizer()), [])
        for generated, status in (('[D1] 100만원', 'review'), ('[D1] 40만원', 'answer')):
            with patch('scripts.shared_demo.complete', return_value={'generated': generated}):
                result = answer_document_chat(document, '상한은?', 'http://127.0.0.1:1', 'qwen3', 4096)
            self.assertEqual(result['status'], status, generated)
        with patch('scripts.shared_demo.complete', return_value={'generated': '강의료는 100만원까지 줄 수 있습니다.', 'truncated_output': False}):
            result = answer_document_case(document, '상한은?', 'http://127.0.0.1:1', 'qwen3', 4096)
        self.assertNotIn('100만원', result['commentary'] or '')

    def test_chat_reply_withholds_unsourced_rule_facts(self):
        from scripts.shared_demo import chat_reply
        with patch('scripts.shared_demo.chat_complete', return_value={'generated': '안녕하세요. 출장 식비는 하루 3만 원입니다.\n무엇을 도와드릴까요?'}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, [], '안녕')
        self.assertEqual(r['status'], 'chat')
        self.assertNotIn('3만', r['answer'])
        self.assertIn('무엇을 도와드릴까요?', r['answer'])
        self.assertTrue(any('1개를 숨겼습니다' in n for n in r['commentary_notes']))

    def test_chat_keeps_everyday_figures_but_not_rule_claims(self):
        from scripts.shared_demo import chat_reply
        reply = '커피는 하루 2잔 정도가 무난해요. 점심 뒤에 20분 산책해 보세요. 공무원 출장 일비는 하루 2만5천원입니다.'
        with patch('scripts.shared_demo.chat_complete', return_value={'generated': reply}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, [], '어제 잠을 못 잤는데 오후 버티는 팁 있어?')
        self.assertIn('커피는 하루 2잔 정도가 무난해요.', r['answer'])
        self.assertIn('20분 산책', r['answer'])
        self.assertNotIn('2만5천원', r['answer'])

    def test_chat_drops_the_oldest_turns_when_the_conversation_does_not_fit(self):
        from scripts.shared_demo import chat_reply
        history = [{'role': r, 'content': f'{r} {i}'} for i in range(3) for r in ('user', 'assistant')]
        sent = []
        def complete(base, messages, *args):
            sent.append([m['content'] for m in messages])
            return {'skipped_context': True} if len(messages) > 4 else {'generated': '네.'}
        with patch('scripts.shared_demo.chat_complete', side_effect=complete):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, history, '그래서?')
        self.assertEqual(r['answer'], '네.')
        self.assertEqual(sent[-1][1:], ['user 2', 'assistant 2', '그래서?'])  # the latest turn kept, the oldest gone

    def test_chat_keeps_the_visitors_first_statement_while_the_conversation_fits(self):
        # GENERAL-CHAT-002: a fixed 6-message window dropped "견과류 알레르기가 있어요" on the 5th turn although the
        # prompt used 479 of 4096 tokens; the whole stored conversation goes in while it fits.
        from scripts.shared_demo import chat_reply
        history = [{'role': 'user', 'content': '저는 견과류 알레르기가 있어요.'}, {'role': 'assistant', 'content': '알겠어요.'}]
        history += [{'role': r, 'content': f'{r} {i}'} for i in range(4) for r in ('user', 'assistant')]
        sent = []
        def complete(base, messages, *args):
            sent.append([m['content'] for m in messages])
            return {'generated': '네.'}
        with patch('scripts.shared_demo.chat_complete', side_effect=complete):
            chat_reply('http://127.0.0.1:1', 'qwen3', 4096, history, '비 오는 날이면 메뉴를 바꾸는 게 나을까요?')
        self.assertEqual(sent[0][1], '저는 견과류 알레르기가 있어요.')
        self.assertEqual(len(sent), 1)

    def test_value_hold_passage_numbers_are_neither_quotes_nor_support(self):
        from scripts.shared_demo import PassageChunk
        text = '식비는 하루 3만 원입니다. 담당은 운영지원팀입니다.'
        held = [(PassageChunk(text, 1, 0, 'd', ('d',), False, True), 1.0)]
        document = Document('d', [text], [c for c, _ in held], [])
        with patch('scripts.shared_demo.complete', return_value={'generated': '식비는 하루 3만 원입니다.', 'truncated_output': False}):
            r = answer_document_case(document, '식비는 얼마?', 'http://127.0.0.1:1', 'qwen3', 4096, selected=held)
        self.assertEqual((r['candidates'], r['commentary']), ([], None))
        self.assertTrue(any('원문(쪽)에서 직접 확인' in n for n in r['commentary_notes']))

    def test_long_workspace_passage_is_windowed_for_the_reader(self):
        from scripts.shared_demo import PassageChunk, reader_windows
        source = '가' * 50 + ''.join(f'제{i}조 출장 식비는 하루 {i}만원으로 한다. ' for i in range(1, 40))
        chunk = PassageChunk(source[50:], 1, 50, 'd', ('d',), False, True)
        windows = reader_windows([(chunk, 0.9)], CharTokenizer())
        self.assertGreater(len(windows), 1)
        for w, score in windows:
            self.assertLessEqual(len(w.text), 128)  # CharTokenizer: one token per character
            self.assertEqual(source[w.start:w.start + len(w.text)], w.text)
            self.assertEqual((w.doc_id, w.value_hold, score), ('d', True, 0.9))

    def test_chat_keeps_numbers_the_visitor_gave(self):
        from scripts.shared_demo import chat_reply
        history = [{'role': 'user', 'content': '이번 달 연차가 2일 남았어요.'}, {'role': 'assistant', 'content': '알겠습니다.'}]
        with patch('scripts.shared_demo.chat_complete', return_value={'generated': '연차가 2일 남으셨어요. 연가는 연 21일입니다.'}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, history, '제 연차 며칠 남았죠?')
        self.assertIn('2일 남으셨어요', r['answer'])
        self.assertNotIn('21일', r['answer'])

    def test_chat_keeps_arithmetic_of_visitor_figures_but_not_other_numbers(self):
        from scripts.shared_demo import chat_reply
        history = [{'role': 'user', 'content': '숙박비 8만원, 식비 2만원이에요.'}, {'role': 'assistant', 'content': '네.'}]
        reply = '합계는 10만원입니다. 하루 일비는 2만5천원입니다.'
        with patch('scripts.shared_demo.chat_complete', return_value={'generated': reply}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, history, '둘을 더하면 얼마예요?')
        self.assertIn('합계는 10만원입니다.', r['answer'])
        self.assertNotIn('일비', r['answer'])
        self.assertTrue(any('계산 결과는 직접' in n for n in r['commentary_notes']))

    def test_request_type_markers(self):
        from scripts.shared_demo import request_type
        self.assertEqual(request_type('회의는 오후 3시, 예산은 20만원이에요. 한 줄로 정리해 주세요.'), 'organize')
        self.assertEqual(request_type('직원들이 실제로 쓴 평균 식비는?'), 'statistic')
        self.assertEqual(request_type('여비와 출장비는 무슨 차이야?'), 'explain')
        self.assertIsNone(request_type('국내 출장 식비 지급 기준은?'))
        self.assertEqual(request_type('아까 제가 말한 출장비 예산이 얼마였죠?'), 'recall')




class WorkspaceTests(unittest.TestCase):
    TEXTS = {'a': ('가기관 출장 규정', '제1조(식비) 국외 출장 식비는 하루 8만원으로 한다.\n제2조(숙박) 국외 출장 숙박비는 1박 25만원으로 한다.'),
             'b': ('나기관 출장 규정', '제1조(식비) 국외 출장 식비는 하루 6만원으로 한다.\n제2조(숙박) 숙박비는 실비로 한다.')}

    def setUp(self):
        import tempfile
        from trid.service.index import Index
        from trid.service.structure import parse_document
        from trid.service.types import Document as ServiceDocument
        from scripts.shared_demo import Workspace
        texts = self.TEXTS
        docs, meta = [], {}
        for key, (title, text) in texts.items():
            d = ServiceDocument(key, title, key, {'public'}, text)
            d.passages, d.refs = parse_document(text, key, title)
            docs.append(d)
            meta[key] = {'title': title, 'kind': '행정규칙', 'institution': title[:3], 'date': '2026-01-01', 'date_type': '발령일자',
                         'effective_date': None, 'currency': None, 'source': f'https://example.invalid/{key}', 'license': 'test',
                         **getattr(self, 'META', {}).get(key, {})}
        self.dir = tempfile.TemporaryDirectory(); self.addCleanup(self.dir.cleanup)
        Index.build(docs).save(self.dir.name)
        Path = __import__('pathlib').Path
        (Path(self.dir.name) / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
        (Path(self.dir.name) / 'manifest.json').write_text(json.dumps({'documents': len(texts), 'by_kind': {'행정규칙': len(texts)}, 'date_range': ['2026-01-01', '2026-01-01']}))
        self.ws = Workspace(self.dir.name)

    def test_ambiguous_institution_is_asked_back_instead_of_picked(self):
        r = self.ws.select('국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'clarify')
        self.assertEqual(r['chunks'], [])
        self.assertEqual(set(r['institutions']), {'가기관', '나기관'})

    def test_named_institution_restricts_passages_to_its_document(self):
        r = self.ws.select('나기관 국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'answer')
        self.assertEqual(r['applied'], 'b')
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {'b'})

    def test_institution_missing_from_corpus_does_not_fall_back(self):
        r = self.ws.select('다라공단 국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'unknown_institution')
        self.assertEqual(r['chunks'], [])

    def test_pinned_document_is_the_only_source(self):
        r = self.ws.select('국외 출장 식비는 하루 얼마인가요?', pinned='b')
        self.assertEqual(r['applied'], 'b')
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {'b'})

    def test_follow_up_without_names_continues_previous_document(self):
        r = self.ws.select('그럼 숙박비는요?', context_doc='b')
        self.assertEqual(r['status'], 'answer')
        self.assertTrue(r['continued'])
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {'b'})
        self.assertTrue(self.ws.continues('그럼 숙박비는요?', 'b'))
        self.assertFalse(self.ws.continues('그럼 숙박비는요?', None))

    def test_named_institution_overrides_previous_document(self):
        r = self.ws.select('가기관 숙박비는 얼마인가요?', context_doc='b')
        self.assertEqual(r['applied'], 'a')
        self.assertNotIn('continued', r)
        self.assertFalse(self.ws.continues('가기관 숙박비는 얼마인가요?', 'b'))

    def test_previous_document_without_relevant_text_is_not_continued(self):
        r = self.ws.select('국외 출장 숙박비 1박', context_doc='b')
        self.assertNotIn('continued', r)
        self.assertEqual(r['applied'], 'a')
        self.assertFalse(self.ws.continues('국외 출장 숙박비 1박', 'b'))

    def test_official_alias_without_documents_is_named_not_substituted(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('부동산원 국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'unknown_institution')
        self.assertEqual(r['unknown_institutions'], ['한국부동산원'])
        self.assertIn('한국부동산원(공기업', workspace_reply(self.ws, r)['answer'])

    def test_question_about_absent_topic_lists_no_unrelated_documents(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('국외 출장 항공권 좌석 등급은?')
        self.assertEqual((r['status'], r.get('reason')), ('not_found', 'irrelevant'))
        self.assertEqual((r['chunks'], r['others']), ([], []))
        self.assertNotIn('기관 출장 규정', workspace_reply(self.ws, r)['answer'])

    def test_statistic_question_uses_only_structured_values(self):
        from scripts.shared_demo import statistic_reply
        self.assertEqual(statistic_reply(self.ws, '직원 평균 출장 식비는?')['status'], 'not_found')
        self.ws.facts = [{'metric': '출장 식비 실지출 평균', 'value': v, 'unit': '원', 'basis': 'actual_spending', 'source_id': f's{v}',
                          'subject': {'id': None, 'name': n}, 'period': {'from': '2025-01', 'to': '2025-12'}}
                         for n, v in (('가기관', 20000), ('나기관', 30000))]
        self.ws.embedder = lambda text: __import__('numpy').ones(4) / 2  # every metric matches
        r = statistic_reply(self.ws, '직원 평균 출장 식비는?')
        self.assertEqual(r['status'], 'answer_values')
        self.assertIn('25,000원', r['answer'])

    def test_short_form_resolves_to_official_institution_instead_of_no_data(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('건보공단 직원 출장 식비는?')
        self.assertIn('국민건강보험공단', r['unknown_institutions'])
        self.assertIn('국민건강보험공단', workspace_reply(self.ws, r)['answer'])

    def test_short_form_without_official_alias_is_asked_not_resolved(self):
        r = self.ws.select('근복공단 직원 출장 식비는?')  # in neither B's table nor the government abbreviation rule
        self.assertEqual((r['status'], r['reason']), ('clarify', 'institution'))
        self.assertIn('근로복지공단', r['institutions'])

    def test_government_abbreviation_rule_names_its_body(self):
        # 「정부조직 약칭과 영어 명칭에 관한 규칙」 [별표]: 국토부 = 국토교통부; 노동부 is now 고용노동부's abbreviation
        # (before: the pre-2010 body); 기재부 comes from the superseded version and still names 기획재정부, whose
        # rules are not given to its successor.
        from scripts.shared_demo import workspace_reply
        for asked, body in (('국토부', '국토교통부'), ('노동부', '고용노동부'), ('기재부', '기획재정부')):
            r = self.ws.select(f'{asked} 국외 출장 식비는?')
            self.assertEqual((r['status'], r['unknown_institutions']), ('unknown_institution', [body]), asked)
        self.assertIn('재정경제부', workspace_reply(self.ws, self.ws.select('기재부 국외 출장 식비는?'))['answer'])

    def test_predecessor_name_is_not_answered_with_successor_rules(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('기획재정부 국외 출장 식비는?')
        self.assertEqual((r['status'], r['unknown_institutions']), ('unknown_institution', ['기획재정부']))
        self.assertIn('재정경제부', workspace_reply(self.ws, r)['answer'])

    def test_ambiguous_official_alias_is_asked_back_with_its_candidates(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('교육청 국외 출장 식비는?')  # 기관코드 table marks 교육청 ambiguous
        self.assertEqual((r['status'], r['reason']), ('clarify', 'institution'))
        self.assertNotIn('교육청', r['institutions'])
        self.assertIn('경기도교육청', workspace_reply(self.ws, r)['answer'])

    def test_exact_short_rule_name_is_recognised(self):
        r = self.ws.select('나기관 출장 규정에서 숙박비는?')
        self.assertTrue(r['title_matched'])
        self.assertEqual(r['applied'], 'b')

    def test_privacy_review_passages_are_never_retrieved(self):
        from scripts.shared_demo import Workspace
        review = __import__('pathlib').Path(self.dir.name) / 'review.jsonl'
        review.write_text(''.join(json.dumps({'passage_id': p.id, 'value_review': {'privacy_review': True}}) + '\n'
                                  for p in self.ws.index.passages if p.doc_id == 'b'))
        ws = Workspace(self.dir.name, review=review)
        r = ws.select('국외 출장 식비는 하루 얼마인가요?', pinned='b')
        self.assertEqual(r['chunks'], [])
        self.assertNotIn('b', {c.doc_id for c, _ in ws.select('국외 출장 식비는 하루 얼마인가요?')['chunks']})


class CivilServiceScopeTests(WorkspaceTests.__base__):
    TEXTS = {'a': ('가기관 수당 규정', '제1조(목적) 이 영은 국가공무원 및 지방공무원의 출장 식비를 정한다. 출장 식비는 하루 2만원.'),
             'b': ('나기관 수당 규정', '제1조(목적) 이 영은 국가공무원 및 지방공무원의 출장 식비를 정한다. 출장 식비는 하루 3만원.')}
    setUp = WorkspaceTests.setUp

    def test_rule_for_the_other_civil_service_population_is_not_applied(self):
        self.ws.meta['a']['applies_to'] = {'scope': 'central_government', 'basis': '제1조(목적) names 국가공무원'}
        self.ws.meta['b']['applies_to'] = {'scope': 'local_government', 'basis': '제1조(목적) names 지방공무원'}
        r = self.ws.select('지방공무원 출장 식비는 하루 얼마인가요?')
        self.assertEqual((r['status'], r['applied']), ('answer', 'b'))
        self.assertEqual(self.ws.select('공무원 출장 식비는 하루 얼마인가요?')['status'], 'clarify')  # no population named

    def test_follow_up_naming_the_other_civil_service_population_leaves_the_previous_rule(self):
        self.ws.meta['a']['applies_to'] = {'scope': 'central_government', 'basis': '제1조(목적) names 국가공무원'}
        self.ws.meta['b']['applies_to'] = {'scope': 'local_government', 'basis': '제1조(목적) names 지방공무원'}
        r = self.ws.select('지방공무원 출장 식비는 하루 얼마인가요?', context_doc='a')
        self.assertEqual((r['status'], r['applied'], r.get('continued')), ('answer', 'b', None))
        self.assertFalse(self.ws.continues('지방공무원 출장 식비는 하루 얼마인가요?', 'a'))
        self.assertTrue(self.ws.select('그럼 출장 식비는 하루 얼마인가요?', context_doc='a').get('continued'))


class RuleRoleTests(WorkspaceTests.__base__):
    """Within one institution's rule set, an amount question goes to the instrument that sets the amount, not to a
    guideline whose articles send it there or whose top passages are worked cases (C-030 #3)."""
    TEXTS = {'g': ('다기관 여비 처리지침', '제6조(식비 지급기준) 출장 식비는 규정 별표1의 식비 정액으로 지급한다.\n'
                                    '○ 1박 2일 출장 중 식사가 제공된 경우 식비 지급액은? - 제공된 끼니의 식비를 빼고 지급한다.'),
             'r': ('다기관 여비 규정', '제9조(국내 여비) 국내 여비는 별표1에 따른다.\n[별표1] 국내여비\n출장 식비 지급액 1일 25,000원')}
    setUp = WorkspaceTests.setUp

    def test_amount_question_goes_to_the_rule_the_guideline_defers_to(self):
        r = self.ws.select('다기관 출장 식비 지급액은 얼마야?')
        self.assertEqual(r['retrieval']['hits'][0]['doc_id'], 'g')  # the guideline ranks first …
        self.assertEqual((r['status'], r['applied']), ('answer', 'r'))  # … and hands the amount to the rule
        self.assertEqual(r['standard_choice'], {'from': 'g', 'to': 'r', 'basis': 'defers'})

    def test_case_question_keeps_the_guideline(self):
        r = self.ws.select('다기관 출장 중 식사가 제공된 경우 식비 처리는?')
        self.assertEqual(r['applied'], 'g')
        self.assertNotIn('standard_choice', r)


class TitleFragmentTests(WorkspaceTests.__base__):
    """A question naming TITLE_MIN (8) consecutive normalised characters of a title names that document; 7 do not, and a
    Q&A title never does (SITE-MEMORY-023 replaced the title n-gram dictionary with a substring scan)."""
    TEXTS = {'a': ('공무국외출장 운영지침', '제1조(목적) 이 지침은 공무국외출장의 운영에 필요한 사항을 정한다.'),
             'b': ('국외 업무협의 처리요령', '제1조(목적) 이 요령은 국외 업무협의 처리에 필요한 사항을 정한다.'),
             'q': ('공무국외출장 운영 질의', '공무국외출장 운영에 관한 질의와 회신이다.')}
    META = {'a': {'institution': '가기관'}, 'b': {'institution': '가기관'}, 'q': {'institution': '가기관', 'kind': '질의응답'}}
    setUp = WorkspaceTests.setUp

    def test_eight_shared_title_characters_name_the_document(self):
        self.assertEqual(self.ws.constraints('공무국외출장운영 기준이 궁금해요')[3], {'a'})
        self.assertIsNone(self.ws.constraints('공무국외출장운 관련 기준이 궁금해요')[3])
        self.assertEqual(self.ws.constraints('국외 업무협의 처리 방법')[3], {'b'})


class RuleNameBoundaryTests(WorkspaceTests.__base__):
    """A rule's name names it only from a word start: "지방공무원 여비 규정" is not "공무원 여비 규정" (whose 제1조 limits
    it to 국가공무원), so a local-government question is not answered from the central rule (MODEL-RESUME-A-024)."""
    TEXTS = {'law': ('공무원 여비 규정', '제1조(목적) 이 영은 국가공무원의 여비를 정한다. 제16조(숙박비) 숙박비는 별표 2에 따른다.'),
             'b': ('나기관 출장 규정', '제1조(숙박) 출장 숙박비는 실비로 한다.')}
    setUp = WorkspaceTests.setUp

    def test_rule_name_inside_a_longer_word_does_not_name_the_rule(self):
        for q in ('공무원 여비 규정에서 숙박비는?', '「공무원여비규정」의 숙박비', '공무원 여비규정상 숙박비'):
            self.assertEqual(self.ws.constraints(q)[3], {'law'}, q)
        for q in ('지방공무원 여비 규정에서 숙박비는?', '국가공무원 여비 규정 숙박비'):
            self.assertFalse(self.ws.constraints(q)[3], q)



class AskBackAnnexTests(WorkspaceTests.__base__):
    """An ask-back over institutions lists each one's own annex with its body when both passed (C-030 L1/S3)."""
    TEXTS = {'a': ('가기관 여비 규정', '제10조(식비) 국내 출장 식비는 하루 단위로 별표 2에 따라 국내 출장 식비를 지급한다.'),
             'a2': ('가기관 여비 규정 [별표 2] 국내 여비 지급표', '식비 하루 25,000원'),
             'b': ('나기관 여비 규칙', '제10조(식비) 국내 출장 식비는 하루 25,000원으로 한다.')}
    setUp = WorkspaceTests.setUp

    def test_common_annex_is_offered_with_its_body(self):
        r = self.ws.select('국내 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'clarify')
        titles = [c['title'] for c in r['candidates']]
        # the top institution's body, then its own annex (where the amounts are), then the other institution's rule
        self.assertEqual(titles, ['가기관 여비 규정', '가기관 여비 규정 [별표 2] 국내 여비 지급표', '나기관 여비 규칙'])

    def test_annex_for_an_unstated_condition_is_not_offered_with_its_body(self):
        # "출장비" listed 공무원 여비 규정 with its [별표 8] 상시 출장 table (MODEL-RESUME-A-024)
        from unittest.mock import patch
        with patch.object(type(self.ws), 'unstated_table_condition',
                          lambda ws, d, best, q: ('a', ['상시']) if d == 'a2' else None):
            r = self.ws.select('국내 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'clarify')
        self.assertEqual([c['title'] for c in r['candidates']], ['가기관 여비 규정', '나기관 여비 규칙'])

    def test_own_rule_rivals_a_rule_not_recorded_as_common(self):
        # 나기관's own rule (title prefix or B applies_to institution); 가기관's rule read as not its own
        self.ws.specific = (self.ws.specific | {'b'}) - {'a', 'a2'}
        r = self.ws.select('국내 출장 식비는 하루 얼마인가요?')
        self.assertEqual((r['status'], r['institutions'][-1]), ('clarify', '나기관'))  # 가기관 rule: applicability unrecorded
        self.ws.meta['a']['applies_to'] = {'scope': 'central_government', 'basis': '제1조(목적)'}
        r = self.ws.select('국내 출장 식비는 하루 얼마인가요?')
        self.assertEqual((r['status'], r['applied']), ('answer', 'a'))  # a recorded common rule is not rivalled by it


class TableConditionTests(WorkspaceTests.__base__):
    """RETRIEVAL-ROOT-009: a rule's table whose caption states a condition the question does not (상시 출장) is not
    answered in place of the rule's body; a question stating it keeps the table."""
    TEXTS = {'a': ('가기관 여비 규정', '제10조(식비) 국내 출장 식비는 별표 2에 따른다.\n제29조(상시 출장) 상시 출장 직원의 여비는 별표 8에 따른다.'),
             'a1': ('가기관 여비 규정 [별표 1] 여비 지급 구분표', '■ 가기관 여비 규정 [별표 1] 여비 지급 구분표 1호 2호'),
             'a2': ('가기관 여비 규정 [별표 2] 국내 여비 지급표', '■ 가기관 여비 규정 [별표 2] 국내 여비 지급표 식비 하루 25,000원'),
             'a8': ('가기관 여비 규정 [별표 8] 상시 출장 직원의 여비 지급 기준표',
                    '■ 가기관 여비 규정 [별표 8] 상시 출장 직원의 여비 지급 기준표 월 15일 이상 전액')}
    setUp = WorkspaceTests.setUp

    def test_special_table_gives_way_to_its_rule_unless_the_question_states_its_condition(self):
        both = {'a8': 0.9, 'a': 0.8}
        body, unstated = self.ws.unstated_table_condition('a8', both, '출장 여비 지급 기준')
        self.assertEqual(body, 'a')
        self.assertIn('상시', unstated)
        self.assertIsNone(self.ws.unstated_table_condition('a8', both, '상시 출장 직원 여비 지급 기준'))
        self.assertIsNone(self.ws.unstated_table_condition('a2', {'a2': 0.9, 'a': 0.8}, '국내 출장 식비'))
        self.assertIsNone(self.ws.unstated_table_condition('a8', {'a8': 0.9}, '출장 여비 지급 기준'))  # body did not pass

    def test_set_aside_table_is_not_linked_back_through_its_body(self):
        # the body cites 별표 2 and 별표 8; the table set aside for its unstated condition is not brought back
        from types import SimpleNamespace
        everywhere = self.ws.scores('출장 여비 지급 기준')[0]
        body = [SimpleNamespace(text=p.text) for p in self.ws.index.passages if p.doc_id == 'a']
        linked = lambda exclude: {str(self.ws.passage_doc[i]) for i in self.ws.linked_tables('a', body, everywhere, exclude=exclude)}
        self.assertEqual(linked(()), {'a2', 'a8'})
        self.assertEqual(linked({'a8'}), {'a2'})


class DomainContractTests(WorkspaceTests.__base__):
    """RETRIEVAL-ROOT-008: a blank form is evidence only for a question asking for a form; an everyday word counts as
    covered by its official term only inside the rules B cited for that relation (DATA-CONTINUE-007 T01)."""
    TEXTS = {'a': ('가기관 여비 규정', '제1조(여비) 공무로 여행하는 직원의 여비는 운임, 일비, 숙박비 및 식비로 한다.'),
             'b': ('나기관 국외출장 규정', '[별지 제1호 서식] 국외출장 계획서 출장비 출장경비 금액 부담기관 출장기간')}

    def setUp(self):
        WorkspaceTests.setUp(self)
        self.ws.form_mask = self.ws.passage_doc == 'b'  # b's passages stand for the index path "서식 1"

    def test_form_passages_are_evidence_only_when_a_form_is_asked(self):
        asked = self.ws.select('국외출장 계획서 서식')
        self.assertEqual(asked['retrieval']['form_passages'], 'included (form asked)')
        self.assertIn('b', {asked.get('applied')} | {c['doc_id'] for c in asked.get('candidates', [])})
        paid = self.ws.select('출장비 금액 부담')
        self.assertEqual(paid['retrieval']['form_passages'], 'not evidence')
        self.assertNotIn('b', {paid.get('applied')} | {o['doc_id'] for o in paid.get('others', [])})

    def test_everyday_word_is_covered_by_the_official_term_only_in_the_cited_rule(self):
        topical = {'출장비': 10.0}
        a = [i for i, d in enumerate(self.ws.passage_doc) if d == 'a']
        self.assertFalse(self.ws.relevant(a, None, topical))  # the rule text says 여비, never 출장비
        self.ws.term_equivalents = {'출장비': {'a': ['여비']}}
        self.assertTrue(self.ws.relevant(a, None, topical))
        self.ws.term_equivalents = {'출장비': {'b': ['여비']}}  # cited elsewhere: a stays uncovered
        self.assertFalse(self.ws.relevant(a, None, topical))


class ConversationScopeTests(unittest.TestCase):
    def test_recall_markers(self):
        from scripts.shared_demo import request_type
        for q in ('아까 내가 뭐 물어봤지?', '방금 답한 규정이 뭐였어?', '앞에서 뭐라고 했지?', '아까 내가 어디 간다고 했지?'):
            self.assertEqual(request_type(q), 'recall', q)
        self.assertIsNone(request_type('출장 전에 신청해야 하나요?'))

    def test_class_question_answered_from_one_institution_says_so(self):
        from types import SimpleNamespace
        from scripts.shared_demo import population_notice
        inst = SimpleNamespace(by_institution={'한국부동산원': set()}, name='inst-reb-002')
        result = {'status': 'clarify', 'answer': '어느 직위인지 정해 주세요: 원장, 감사.'}
        population_notice(result, '공공기관 임원 업무추진비는 직위별로 얼마야?', inst, None, None)
        self.assertTrue(result['answer'].startswith('질문은 공공기관 전반에 관한 것이지만, 아래는 한국부동산원 한 기관의 자료입니다.'))
        named = {'status': 'answer', 'answer': 'x'}
        population_notice(named, '한국부동산원 임원 업무추진비는?', inst, None, None)
        self.assertEqual(named['answer'], 'x')  # one named institution: no class claim to correct

    def test_class_ask_back_among_own_or_civil_service_rules_says_no_common_rule(self):
        from types import SimpleNamespace
        from scripts.shared_demo import population_notice
        scopes = {'law': 'central_government'}  # B applies_to sits on the rule body only
        ws = SimpleNamespace(specific={'ministry-own'}, rule_names={'여비규정': {'law', 'law-annex'}},
                             rule_of={'law': '여비규정', 'law-annex': '여비규정'},
                             applies=lambda d: {'scope': scopes[d]} if d in scopes else {})
        ask = lambda *docs: {'status': 'clarify', 'candidates': [{'doc_id': d} for d in docs]}
        result = {'status': 'clarify', 'answer': '어느 기관 기준인지 정해 주세요.'}
        population_notice(result, '공공기관 출장비', None, ws, ask('ministry-own', 'law-annex'))
        self.assertTrue(result['answer'].startswith('질문은 공공기관 전반에 관한 것이지만, 검색 자료에서 공공기관 공통 기준을 찾지 못했습니다.'))
        self.assertFalse(result['population_scope']['common_rule_found'])
        unrecorded = {'status': 'clarify', 'answer': 'x'}  # an option whose applicability is not recorded may be common
        population_notice(unrecorded, '공공기관 출장비', None, ws, ask('ministry-own', 'unknown-scope'))
        self.assertEqual(unrecorded['answer'], 'x')
        not_class = {'status': 'clarify', 'answer': 'x'}
        population_notice(not_class, '출장비 기준', None, ws, ask('ministry-own', 'law'))
        self.assertEqual(not_class['answer'], 'x')


class RuleTableLinkTests(WorkspaceTests.__base__):
    """A statute body passage citing 별표 N is answered with that table of the same statute — identified by the annex id
    in the table's own caption, never by the corpus title number ("[별표 0006]" may be 별표 6의2)."""
    LAW = 'law-go-kr:law:1'
    BODY = ('제16조(식비) 국내 출장 식비는 별표 2에 따라 지급하고, 가족 여비는 별표 6의2에 따른다.\n'
            '제17조(국외) 국외 여비는 별표 4에 따른다.\n'
            '■ 공무원 여비 규정 [별표 4]\n국외 여비 지급표 가 60 실비(상한액: 471) 186\n'
            '■ 공무원 여비 규정 [별표 5]\n이전비 지급 기준표 실비 390,000\n'
            '■ 공무원 여비 규정 [별표 5]\n이전비 지급 기준표 실비 포함 390,000\n'
            '■ 공무원 여비 규정 [별표 7]\n기타 지급 100')
    TEXTS = {LAW: ('공무원 여비 규정', BODY),
             't2': ('공무원 여비 규정 [별표 0002] 국내 여비 지급표', '■ 공무원 여비 규정 [별표 2]\n국내 출장 식비 하루 25,000원'),
             't6': ('공무원 여비 규정 [별표 0006] 가족 여비 지급표', '■ 공무원 여비 규정 [별표 6의2]\n국외 가족 여비 지급표'),
             't6x': ('공무원 여비 규정 [별표 0006] 다른 표', '■ 공무원 여비 규정 [별표 6]\n다른 식비 표'),
             't8': ('공무원 여비 규정 [별표 0008] 상시 출장 공무원의 여비 지급 기준표', '■ 공무원 여비 규정 [별표 8]\n상시 출장 식비 월 지급'),
             'other': ('다른 여비 규정 [별표 0002] 지급표', '■ 다른 여비 규정 [별표 2]\n식비 하루 1원')}
    META = {LAW: {'kind': '법령', 'effective_date': '2026-10-02'}}
    setUp = WorkspaceTests.setUp

    def test_cited_table_of_the_same_rule_is_added_by_its_caption_annex_id(self):
        self.assertEqual((self.ws.annex_id['t6'], self.ws.annex_id['t6x']), ('6의2', '6'))
        r = self.ws.select('국내 출장 식비 지급', pinned=self.LAW)
        self.assertEqual(r['linked_tables'], ['t2', 't6'])
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {self.LAW, 't2', 't6'})

    def span(self, annex):
        s = self.BODY.index(f'■ 공무원 여비 규정 [별표 {annex}]')
        return [s, self.BODY.find('■', s + 1) if self.BODY.find('■', s + 1) > 0 else len(self.BODY)]

    def rows_workspace(self, groups, links):
        """Workspace over this fixture with a reviewed-row file and B's link file.
        groups = [(rows doc id, annex, cell rows, review, source)]; links = [B link entries]."""
        from pathlib import Path
        from scripts.shared_demo import Workspace
        d = Path(self.dir.name)
        rows = [{'id': f'{doc}:t0:r{i}', 'doc_id': doc, 'table': 0, 'review': review, 'cells': cells, 'source': source,
                 'caption': [f'■ 공무원 여비 규정 [별표 {annex}] <개정 2021. 5. 25.>', '지급표'], 'unit': '단위: 원',
                 'revised': '개정 2021. 5. 25.', 'statute_effective_date': '2026-10-02', 'section': None,
                 'notes': ['비고 1. 대통령의 식비는 실비로 한다.']}
                for doc, annex, cell_rows, review, source in groups for i, cells in enumerate(cell_rows)]
        (d / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        (d / 'links.jsonl').write_text(''.join(json.dumps(l, ensure_ascii=False) + '\n' for l in links))
        return Workspace(self.dir.name, table_rows=[(d / 'rows.jsonl', d / 'links.jsonl')])

    def link(self, doc, annex, status='linked', in_index=None, span=None):
        return {'row_doc_id': doc, 'law': '공무원 여비 규정', 'mst': '1', 'byl': annex, 'status': status,
                'byl_doc_in_index': in_index, 'body_doc': self.LAW, 'linked_section': span}

    def test_reviewed_rows_link_only_to_their_verified_source_text(self):
        ws = self.rows_workspace(
            [('t2', '2', [{'구분': '제2호', '식비(1일당)': '25,000'}], 'ai_visual_review_match', {'kordoc_text_sha256': 't2'}),
             ('t8', '8', [{'구분': '1', '지급': '99,999'}], 'render_check_pass', {'kordoc_text_sha256': 't8'}),
             ('byl4', '4', [{'등급': '가', '일비': '60', '숙박비': '실비(상한액: 471)', '식비': '186'},
                            {'구분': '6. 삭제 <2021. 5. 25.>'}], 'ai_visual_review_match', {}),
             ('byl5', '5', [{'구분': '국내 이전비', '지급액': '390,000'}], 'ai_visual_review_match', {}),
             ('byl7', '7', [{'구분': '기타', '지급액': '200'}], 'ai_visual_review_match', {}),
             ('t6', '6의2', [{'지급 사유': '부임', '지급액': '3분의 2'}], 'ai_visual_review_match', {'kordoc_text_sha256': 'stale'}),
             ('byl11', '11', [{'호봉': '1', '금액': '7'}], 'ai_visual_review_match', {}),
             ('t6x', '6', [{'구분': '다른', '식비': '없음'}], 'ai_visual_review_match', {'kordoc_text_sha256': 't6x'})],
            [self.link('t2', '2', in_index='t2'), self.link('t8', '8', in_index='t8'),
             self.link('byl4', '4', span=self.span(4)), self.link('byl5', '5', span=self.span(5)),
             self.link('byl7', '7', span=self.span(7)),
             self.link('t6', '6의2', in_index='t6'), self.link('byl11', '11', status='mismatch'),
             self.link('t6x', '6', status='mismatch', in_index='t6x')])
        self.assertEqual(ws.row_links[('공무원여비규정', '2')]['method'], 'same_source_hash')       # same document, same hash
        self.assertEqual(ws.row_links[('공무원여비규정', '4')]['method'], 'statute_body_segment')  # B span, values verified
        self.assertEqual(ws.row_links[('공무원여비규정', '4')]['segment'], self.span(4))
        self.assertNotIn(('공무원여비규정', '8'), ws.row_links)                                   # render_check_pass rows never
        self.assertIn('2 "[별표 5]" sections', ws.row_unlinked[('공무원여비규정', '5')])  # B's pick of one is not promoted
        self.assertIn('numbers not in the linked span', ws.row_unlinked[('공무원여비규정', '7')])
        self.assertIn('hash differs', ws.row_unlinked[('공무원여비규정', '6의2')])
        self.assertIn('mismatch', ws.row_unlinked[('공무원여비규정', '11')])
        self.assertEqual(ws.row_links[('공무원여비규정', '6')]['method'], 'same_source_hash')  # body span mismatch, same doc hash
        r = ws.select('국외 여비 지급표', pinned=self.LAW)
        derived = [c for c, _ in r['chunks'] if c.derived]
        self.assertEqual((len(derived), r['reviewed_rows']['annex']), (1, '4'))
        for part in ('단위: 원', '법령 시행일 2026-10-02', '일비: 60', '비고 1. 대통령의 식비는 실비로 한다.'):
            self.assertIn(part, derived[0].text)
        self.assertIn('별표 5', r.get('table_rows_unlinked', []))  # its span is in the evidence: said, not replaced

    def test_rows_are_not_attached_for_a_table_the_question_does_not_ask_about(self):
        ws = self.rows_workspace(
            [('t2', '2', [{'구분': '제2호', '식비(1일당)': '25,000'}], 'ai_visual_review_match', {'kordoc_text_sha256': 't2'}),
             ('t8', '8', [{'구분': '별표 8', '지급': '월'}], 'ai_visual_review_match', {'kordoc_text_sha256': 't8'})],
            [self.link('t2', '2', in_index='t2'), self.link('t8', '8', in_index='t8')])
        ws.row_links[('공무원여비규정', '2')]['rows'][0]['caption'][1] = '국내 여비 지급표'
        ws.row_links[('공무원여비규정', '8')]['rows'][0]['caption'][1] = '상시 출장 지급표'
        r = ws.select('국내 출장 식비 지급', pinned=self.LAW)
        self.assertEqual(r['reviewed_rows']['annex'], '2')
        # another table of the rule is named by the question better than the one in the evidence: no rows at all
        ws.row_links[('공무원여비규정', '8')]['rows'][0]['caption'][1] = '국내 출장 식비 지급 기준표'
        r = ws.select('국내 출장 식비 지급', pinned=self.LAW)
        self.assertNotIn('reviewed_rows', r)
        self.assertEqual(r['table_rows_not_in_evidence'], ['별표 8'])

    def test_derived_rows_are_never_quoted_and_shrink_by_whole_rows(self):
        from dataclasses import replace
        from scripts.shared_demo import CASE_SYSTEM, answer_document_case, rag_prompt, render_rows
        from trid.pdf_documents import Document as Doc
        ws = self.rows_workspace([('byl4', '4', [{'등급': g, '일비': '60', '숙박비': '실비(상한액: 471)', '식비': '186'}
                                                for g in '가나다라'], 'ai_visual_review_match', {})],
                                 [self.link('byl4', '4', span=self.span(4))])
        r = ws.select('국외 여비 지급표', pinned=self.LAW)
        derived = next(c for c, _ in r['chunks'] if c.derived)
        # the prompt once other documents' passages are gone and two of the four rows remain: the budget
        kept = [(c, s) for c, s in r['chunks'] if c.doc_id == self.LAW and not c.derived] + [
            (replace(derived, text=render_rows(derived.rows, 2), shown=2), 0.0)]
        counter = lambda base, messages: sum(len(m['content']) for m in messages)  # a stand-in tokenizer: 1 char = 1 token
        budget = counter(None, [{'content': CASE_SYSTEM},
                                {'content': rag_prompt([(f'D{i}', c.text) for i, (c, _) in enumerate(kept, 1)], '일비')}])
        document = Doc('d', [], [c for c, _ in r['chunks']], [])
        with patch('scripts.shared_demo.prompt_tokens', side_effect=counter), \
             patch('scripts.shared_demo.complete', return_value={'generated': '- 등급: 가; 일비: 60', 'truncated_output': False}) as call:
            result = answer_document_case(document, '일비', 'http://127.0.0.1:1', 'qwen3', budget, max_tokens=0,
                                          selected=r['chunks'])
        sent = call.call_args[0][1][1]['content']
        self.assertNotIn('[별표 2]', sent)                                  # other documents' passages go first
        self.assertIn('등급: 나', sent)
        self.assertNotIn('등급: 다', sent)                                   # then whole rows, from the end
        self.assertIn('나머지 2행', sent)                                    # and the text says so
        self.assertIn('비고 1. 대통령의 식비는 실비로 한다.', sent)          # notes kept whole
        self.assertIn('나머지', sent)                                       # dropped rows are stated
        self.assertTrue(any('행을 빼었습니다' in n for n in result['context_notes']))
        other = sum(1 for c, _ in r['chunks'] if c.doc_id != self.LAW and not c.derived)
        self.assertEqual(result['context_limits'],                         # counts of what was actually left out
                         {'linked_passages': other, 'table_rows': 2, 'tables': 0, 'own_passages': 0})
        self.assertEqual(result['candidates'], [])  # a line of the derived rows is not a verbatim source quote


class ExtractiveChunkTests(unittest.TestCase):
    def prediction(self, text, question, occurrence, presence=0):
        prompt = rag_prompt([('D1', text)], question)
        _, lo, _ = passage_spans(prompt)[0]
        return dict(presence=presence, answer='운영지원팀 [D1]', span_text='운영지원팀',
                    source='D1', span=[lo + occurrence, lo + occurrence + len('운영지원팀')])

    def test_repeated_answer_uses_predicted_offset_not_first_match(self):
        text = '이전 담당: 운영지원팀. 현재 담당: 운영지원팀.'
        start = text.rindex('운영지원팀')
        with patch('scripts.shared_demo.read_extractive', return_value=self.prediction(text, '현재 담당은?', start)):
            result = read_chunk(None, None, None, text, '현재 담당은?', 'cpu')
        self.assertEqual(result['source_character_span'], [start, start + len('운영지원팀')])
        self.assertEqual(result['answer'], '운영지원팀')

    def test_abstention_does_not_expose_latent_span(self):
        text = '담당은 운영지원팀입니다.'
        prediction = self.prediction(text, '기한은?', text.index('운영지원팀'), presence=1)
        prediction['answer'] = '확인 불가'
        with patch('scripts.shared_demo.read_extractive', return_value=prediction):
            result = read_chunk(None, None, None, text, '기한은?', 'cpu')
        self.assertTrue(result['abstains'])
        self.assertIsNone(result['source_character_span'])
        self.assertEqual(result['answer'], '확인 불가')

    def test_span_outside_passage_is_rejected(self):
        prediction = self.prediction('담당은 운영지원팀입니다.', '담당은?', 4)
        prediction['span'] = [0, 5]
        with patch('scripts.shared_demo.read_extractive', return_value=prediction):
            with self.assertRaises(DocumentError):
                read_chunk(None, None, None, '담당은 운영지원팀입니다.', '담당은?', 'cpu')


class ProtectedPassageApiTests(unittest.TestCase):
    """C-029: privacy-review passages never reach selection, model context or any response field, and value-hold
    numbers are never answer values — through the real /api/ask path, for case, value (llama) and reader versions."""
    PHONE, HELD, ALL_HELD, TABLE = '010-1111-2222', '5일', '30만원', '100,000원'
    DOCS = [('priv', '가기관 출장 연락 지침', '가기관', None,
             '제1조(담당자 연락처) 출장 담당자 연락처는 전화 010-1111-2222이다.\n제2조(신청) 출장 신청은 운영지원팀에 한다.'),
            ('hold', '나기관 경조사 휴가 규정', '나기관', None,
             '제1조(휴가일수) 본인 결혼 경조사 휴가는 5일로 한다.\n제2조(신청) 경조사 휴가는 미리 신청한다.'),
            ('allhold', '다기관 포상 규정', '다기관', None, '제1조(포상금) 우수 직원 포상금은 30만원으로 한다.'),
            ('law', '공무원 여비 규정', '인사혁신처', 'central_government',
             '제1조(목적) 이 영은 국가공무원의 여비를 정한다.\n제16조(식비) 국내 출장 식비는 하루 25,000원으로 하고, 숙박비는 별표 2에 따른다.'),
            ('t2', '공무원 여비 규정 [별표 0002] 국내 여비 지급표', '인사혁신처', None, '■ 공무원 여비 규정 [별표 2]\n국내 출장 숙박비 상한 서울 100,000원')]

    def setUp(self):
        import tempfile
        from pathlib import Path
        from trid.service.index import Index
        from trid.service.structure import parse_document
        from trid.service.types import Document as ServiceDocument
        from scripts.shared_demo import Workspace
        docs, meta = [], {}
        for key, title, inst, applies, text in self.DOCS:
            d = ServiceDocument(key, title, key, {'public'}, text)
            d.passages, d.refs = parse_document(text, key, title)
            docs.append(d)
            meta[key] = {'title': title, 'kind': '법령' if key == 'law' else '행정규칙', 'institution': inst, 'date': '2026-01-01',
                         'date_type': '발령일자', 'effective_date': None, 'currency': None, 'source': 'https://example.invalid',
                         'license': 'test', **({'applies_to': {'scope': applies, 'basis': '제1조'}} if applies else {})}
        d = Path(tempfile.mkdtemp())
        index = Index.build(docs)
        index.save(d)
        (d / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
        (d / 'manifest.json').write_text(json.dumps({'documents': len(docs), 'passages': len(index.passages), 'by_kind': {},
                                                     'date_range': ['2026-01-01', '2026-01-01']}))
        pid = lambda key, needle: next(p.id for p in index.passages if p.doc_id == key and needle in p.text)
        self.private_ids = [pid('priv', self.PHONE), pid('t2', self.TABLE)]
        review = [{'passage_id': i, 'value_review': {'privacy_review': True}} for i in self.private_ids]
        review += [{'passage_id': pid('hold', self.HELD), 'value_review': {'decision': 'hold_for_value_use'}},
                   {'passage_id': pid('allhold', self.ALL_HELD), 'value_review': {'decision': 'hold_for_value_use'}}]
        (d / 'review.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in review))
        workspace = Workspace(d, review=d / 'review.jsonl')
        common = dict(label='t', base_model='b', license='l', version='0', status='s', evaluation='e', limits='l',
                      base='http://127.0.0.1:1', family='qwen3', context=4096)
        versions = {'case': dict(common, id='case', kind='llama', answer_mode='case'),
                    'value': dict(common, id='value', kind='llama', decoding='grammar_instructed'),
                    'reader': dict(common, id='reader', kind='extractive', model=None, tok=CharTokenizer(), device='cpu')}
        self.secret = 'x' * 40
        self.state = SimpleNamespace(sessions={}, lock=threading.RLock(), work=threading.Lock(), tok=CharTokenizer(),
                                     run='t', default='case', versions=versions, workspace=workspace)
        self.state.prune = lambda: State.prune(self.state)
        self.state.public_versions = lambda: State.public_versions(self.state)
        self.server = make_server(self.state, self.secret)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    close, request, visitor = SharedDemoTests.close, SharedDemoTests.request, SharedDemoTests.visitor

    def ask(self, question, version, generated='확인 불가', pinned=None):
        """(response body text, response dict, texts the model was given)."""
        h = self.visitor()
        if pinned:
            next(iter(self.state.sessions.values())).pinned_doc = pinned
        seen = []
        def complete(base, messages, *args, **kwargs):
            seen.append('\n'.join(m['content'] for m in messages))
            return {'generated': generated, 'truncated_output': False}
        def reader(model, presence, tok, text, question, device):
            seen.append(text)
            return {'abstains': True, 'answer': '확인 불가', 'source_character_span': None, 'presence': 1}
        with patch('scripts.shared_demo.complete', side_effect=complete), patch('scripts.shared_demo.read_chunk', side_effect=reader):
            status, body, _ = self.request('POST', '/api/ask', {'question': question, 'scope': 'workspace', 'version': version}, h)
        self.assertEqual(status, 200, body.decode())
        return body.decode(), json.loads(body), '\n'.join(seen)

    def test_privacy_passage_reaches_no_context_and_no_response_field(self):
        for version in ('case', 'value', 'reader'):
            # best evidence protected: refused without a model call
            body, r, seen = self.ask('가기관 출장 담당자 연락처 전화', version)
            self.assertEqual((r['status'], r['reason']), ('not_found', 'protected'), version)
            self.assertEqual(seen, '')
            # the same document pinned, and a statute whose cited table is protected: other passages only
            for question, pinned in (('출장 신청은 어디에 하나요?', 'priv'), ('공무원 여비 규정 국내 출장 식비와 숙박비', None)):
                body2, r2, seen2 = self.ask(question, version, '확인 불가', pinned)
                self.assertEqual(r2['workspace']['selection'], 'answer', (version, question))
                body, seen = body + body2, seen + seen2
            for secret in (self.PHONE, self.TABLE, *self.private_ids):
                self.assertNotIn(secret, body, version)
                self.assertNotIn(secret, seen, version)

    def test_value_hold_numbers_are_not_given_to_value_versions_nor_shown_as_answers(self):
        for version in ('value', 'reader'):
            body, r, seen = self.ask('나기관 경조사 휴가 일수와 신청', version, '5일 [D1]')
            self.assertNotIn('본인 결혼 경조사 휴가는 5일', seen, version)  # held passage not in the model input
            self.assertNotEqual(r['status'], 'answer', version)
            self.assertTrue(any('원문(쪽)' in n for n in r['commentary_notes']), version)
            body, r, seen = self.ask('다기관 우수 직원 포상금', version, '30만원 [D1]')  # every passage held
            self.assertEqual((r['status'], r['reason'], seen), ('review', 'value_hold', ''), version)
            self.assertNotIn(self.ALL_HELD, r['answer'])
        # case version reads the passage but its numbers are neither quotes nor commentary, nor left in raw_output
        body, r, _ = self.ask('나기관 경조사 휴가 일수와 신청', 'case', '[D1] 본인 결혼 경조사 휴가는 5일로 한다.\n따라서 휴가는 5일입니다.')
        self.assertEqual(r['candidates'], [])
        self.assertNotIn(self.HELD, (r['commentary'] or '') + r['answer'] + (r['raw_output'] or ''))

    def test_civil_service_amount_is_not_given_as_another_population_entitlement(self):
        question = '국내 출장 식비는 하루 얼마인가요?'  # names no population; the rule is for 국가공무원 only
        body, r, _ = self.ask(question, 'case', '[D1] 국내 출장 식비는 하루 25,000원으로 하고, 숙박비는 별표 2에 따른다.\n'
                                                '따라서 공공기관 직원의 식비는 하루 25,000원입니다.')
        self.assertTrue(r['answer'].startswith('「공무원 여비 규정」은 국가공무원에게 적용되는 규정'))
        self.assertNotIn('25,000', (r['commentary'] or '') + (r['raw_output'] or ''))
        self.assertTrue(any('숨겼습니다' in n for n in r['commentary_notes']))
        self.assertEqual(r['applicability']['status'], 'not_confirmed')
        body, r, _ = self.ask(question, 'value', '하루 25,000원입니다.')
        self.assertEqual(r['status'], 'review')  # the value is not presented as this population's answer
        self.assertNotIn('25,000', body)  # not in any field, raw model output included
        _, r, _ = self.ask('공무원 국내 출장 식비는 하루 얼마인가요?', 'case', '식비는 하루 25,000원입니다.')
        self.assertNotIn('applicability', r)


class ConversationApiTests(unittest.TestCase):
    """GENERAL-CHAT-001, through the real /api/ask path of the default (assistant) version: everyday talk the router
    sends to search is not turned into a search failure; work questions keep their not-found reply; questions about
    the conversation are answered by the model over this session's own turns only."""
    setUp_shared = ProtectedPassageApiTests.setUp
    close, request, visitor = SharedDemoTests.close, SharedDemoTests.request, SharedDemoTests.visitor
    PHONE, HELD, ALL_HELD, TABLE, DOCS = (ProtectedPassageApiTests.PHONE, ProtectedPassageApiTests.HELD,
                                          ProtectedPassageApiTests.ALL_HELD, ProtectedPassageApiTests.TABLE,
                                          ProtectedPassageApiTests.DOCS)

    def setUp(self):
        self.setUp_shared()
        self.state.versions['assist'] = dict(self.state.versions['case'], id='assist', assistant=True)

    def say(self, visitor, question, routed='search'):
        """(response, the texts the conversation model was given)."""
        seen = []
        def chat(base, messages, *args):
            seen.append([m['content'] for m in messages])
            return {'generated': '네, 그렇게 해 볼게요.', 'truncated_output': False}
        with patch('scripts.shared_demo.route', return_value=routed), patch('scripts.shared_demo.chat_complete', side_effect=chat), \
             patch('scripts.shared_demo.complete', return_value={'generated': '확인 불가', 'truncated_output': False}):
            status, body, _ = self.request('POST', '/api/ask', {'question': question, 'scope': 'workspace', 'version': 'assist'},
                                           visitor)
        self.assertEqual(status, 200, body.decode())
        return json.loads(body), seen

    def test_everyday_input_routed_to_search_without_evidence_is_answered_as_conversation(self):
        r, seen = self.say(self.visitor(), '마지막 문장은 빼 줘')
        self.assertEqual((r['intent'], r['status'], r['answer']), ('chat', 'chat', '네, 그렇게 해 볼게요.'))
        self.assertTrue(r['scope'].startswith('업무 자료에서 맞는 근거를 찾지 못했고'))
        self.assertEqual(len(seen), 1)

    def test_work_question_without_evidence_keeps_its_not_found_reply(self):
        visitor = self.visitor()
        r, seen = self.say(visitor, '국외 출장 항공권 좌석 등급은?')
        self.assertEqual((r['intent'], r['status']), ('search', 'not_found'))
        self.assertEqual(seen, [])
        r, seen = self.say(visitor, '그럼 다른 건?')  # a follow-up of that search, not small talk
        self.assertEqual((r['intent'], r['status'], seen), ('search', 'not_found', []))

    def test_question_about_the_conversation_uses_this_sessions_turns_only(self):
        first, other = self.visitor(), self.visitor()
        self.say(first, '다음 주에 제주도로 가족 여행 가요', routed='chat')
        r, seen = self.say(first, '아까 내가 어디 간다고 했지?')
        self.assertEqual(r['status'], 'chat')
        self.assertIn('다음 주에 제주도로 가족 여행 가요', seen[0])  # the model answers over the stored turn
        r, seen = self.say(other, '아까 내가 어디 간다고 했지?')
        self.assertEqual((r['status'], seen), ('chat', []))  # another session: nothing earlier, no model call
        self.assertIn('앞서 나눈 이야기가 아직 없어요', r['answer'])


class InstitutionSnapshotApiTests(unittest.TestCase):
    """A named institution with a separate snapshot (B: institution_rules_separate) is answered from that index only;
    an expenditure record is never turned into a statistic or a rule; the shared index never contains the snapshot."""
    setUp_shared = ProtectedPassageApiTests.setUp
    close, request, visitor, ask = SharedDemoTests.close, SharedDemoTests.request, SharedDemoTests.visitor, ProtectedPassageApiTests.ask
    PHONE, HELD, ALL_HELD, TABLE, DOCS = (ProtectedPassageApiTests.PHONE, ProtectedPassageApiTests.HELD,
                                          ProtectedPassageApiTests.ALL_HELD, ProtectedPassageApiTests.TABLE,
                                          ProtectedPassageApiTests.DOCS)

    def setUp(self):
        import hashlib, tempfile, time
        from pathlib import Path
        from scripts.build_workspace_index import separate_index
        from scripts.shared_demo import Workspace
        self.setUp_shared()
        def rec(n, kind, basis, title, text, dates, note):
            return {'source_id': f'reb.or.kr:{n}', 'version': str(n), 'content_hash': str(n), 'kind': kind, 'basis': basis,
                    'title': title, 'text': text, 'issuer': {'id': 'C0150', 'id_scheme': 'alio_apbaId', 'name': '한국부동산원'},
                    'applies_to': {'scope': 'institution', 'ids': ['C0150'], 'note': note}, 'dates': dates,
                    'date_kinds': {'effective_from': '마지막 부칙의 시행일(본문 그대로)', 'spending_period': '집행월. 실제 지출'},
                    'currency': '게시본. 이후 개정 여부는 확인하지 않았고 현행이라고 표시하지 않음', 'rights': {'license': 'test'},
                    'location': {'url': 'https://example.invalid'}, 'structured_facts': []}
        rows = [rec(1, 'institution_rule', 'rule_text', '한국부동산원_여비규정', '제10조(식비) 국내 출장 식비는 1일 25,000원으로 한다.',
                    {'effective_from': '2026-07-01'}, '한국부동산원 임직원'),
                rec(2, 'expenditure_record', 'actual_spend', '한국부동산원_2026년 9월 임원 업무추진비',
                    '원장 업무추진비 내역: 2026-09-01 유관기관 업무 간담회 9명 340,000원. 합계 790,000원',
                    {'spending_period': '2026-09'}, '임원의 실제 집행 내역이며 규정상 기준이 아님')]
        rows[1]['structured_facts'] = [{'metric': '임원 업무추진비 월 집행액', 'unit': '원', 'value': 790000,
                                        'population': '임원(원장) 1명 분', 'basis': 'actual_spending',
                                        'not_for': '직원 1인 식비·평균, 규정상 한도로 쓰지 않음', 'verification': '두 판독기 일치',
                                        'period': {'from': '2026-09-01', 'to': '2026-09-30'},
                                        'subject': {'id': 'C0150', 'name': '한국부동산원'}}]
        snap = Path(tempfile.mkdtemp()) / 'inst'
        (snap / 'normalized').mkdir(parents=True)
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (snap / 'normalized/records-00000.jsonl').write_text(body)
        (snap / 'manifest.json').write_text(json.dumps({
            'snapshot_id': 'inst-test', 'index': 'institution_rules_separate', 'holds': ['evaluation', 'training'],
            'files': [{'path': 'normalized/records-00000.jsonl', 'sha256': hashlib.sha256(body.encode()).hexdigest()}]}))
        out = snap.parent / 'index'
        separate_index(snap, out, time.time())
        self.state.institution_workspaces = {'C0150': Workspace(out)}
        self.state.versions['assist'] = dict(self.state.versions['case'], id='assist', assistant=True)

    def test_named_institution_is_answered_from_its_own_snapshot_with_its_date_and_currency(self):
        with patch('scripts.shared_demo.route', return_value='search'):
            _, r, seen = self.ask('한국부동산원 국내 출장 식비 기준', 'assist', '[D1] 국내 출장 식비는 1일 25,000원으로 한다.')
        self.assertEqual((r['workspace']['separate_snapshot'], r['file']), (True, '한국부동산원_여비규정'))
        self.assertTrue(r['answer'].startswith('한국부동산원 누리집 게시 자료(「한국부동산원_여비규정」, 2026-07-01'))
        self.assertIn('현행이라고 표시하지 않음', r['answer'])
        self.assertNotIn('applicability', r)
        _, r, _ = self.ask('국내 출장 식비는 하루 얼마인가요?', 'case', '확인 불가')  # names no institution: shared index
        self.assertFalse(r['workspace']['separate_snapshot'])

    def test_expenditure_records_and_officer_values_are_never_staff_statistics(self):
        with patch('scripts.shared_demo.route', return_value='search'):
            _, r, seen = self.ask('한국부동산원 직원 평균 식비는 얼마인가요?', 'assist')
            self.assertEqual((r['status'], r['reason'], seen), ('not_found', 'no_structured_statistic', ''))
            self.assertIn('임원 업무추진비 집행내역 1건(2026-09~2026-09', r['answer'])
            self.assertNotIn('790,000', r['answer'])
            _, r, _ = self.ask('한국부동산원 임원 업무추진비 평균은?', 'assist')  # no officer named: asked, not averaged
            self.assertEqual((r['status'], r['reason']), ('clarify', 'officer'))
            _, r, seen = self.ask('한국부동산원 9월 원장 업무추진비는?', 'assist')
            self.assertEqual((r['status'], seen), ('answer_values', ''))
            self.assertIn('2026-09 · 임원(원장) 1명 분 · 790,000원', r['answer'])
            self.assertIn('평균·합계는 계산하지 않았습니다', r['answer'])
            body, r, _ = self.ask('한국부동산원 9월 유관기관 간담회 내역', 'assist',
                                  '[D1] 2026-09-01 유관기관 업무 간담회 9명 340,000원.\n따라서 1인당 평균 37,778원입니다.')
        self.assertIn('임원 업무추진비의 실제 집행내역(2026-09 집행분', r['answer'])
        self.assertNotIn('37,778', body)  # a computed figure appears nowhere, raw output included
        self.assertNotEqual(r['status'], 'answer')

    def test_explicit_rule_or_population_switch_leaves_the_snapshot_and_a_follow_up_stays(self):
        from scripts.shared_demo import institution_indexes
        shared, inst = self.state.workspace, self.state.institution_workspaces['C0150']
        ctx = next(d for d, m in inst.meta.items() if m['title'] == '한국부동산원_여비규정')
        pick = lambda q, c=ctx: institution_indexes(self.state, shared, q, c)
        self.assertIs(pick('그럼 숙박비는?'), inst)  # a follow-up keeps the institution
        self.assertIs(pick('한국부동산원 숙박비는?', None), inst)
        self.assertIsNone(pick('공무원 여비 규정에서 국내 출장 식비는?'))  # names another rule
        self.assertIsNone(pick('지방공무원 여비 규정에서 숙박비는?'))  # names a population the institution's rule is not for
        self.assertIsNone(pick('그럼 숙박비는?', None))  # new chat: no previous document

if __name__ == '__main__':
    unittest.main()
