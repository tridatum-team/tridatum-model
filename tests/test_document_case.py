import unittest

from trid.document_case import CONTEXT_OVERFLOW, answer_document_case, fit_own_passages
from trid.pdf_documents import Chunk, Document, DocumentError

SOURCE = '해외 출장 식비는 하루 3만원이다. 숙박비는 실비로 정산한다.'


def document(text=SOURCE):
    return Document('a.pdf', [text], [Chunk(1, 0, len(text), text)], [])


def answer(generated, question='해외 출장 식비', **extra):
    seen = {}

    def complete(base, messages, family, max_tokens, context):
        seen['messages'] = messages
        return {'generated': generated, **extra}
    result = answer_document_case(document(), question, 'http://x', 'qwen3', 4096, complete=complete,
                                  fit_context=lambda selected, build, base, context, max_tokens: (selected, [], {}))
    return result, seen


class FitOwnPassagesTests(unittest.TestCase):
    def test_fit_own_passages_keeps_best_and_drops_weakest_first(self):
        chunks = [(Chunk(1, i, i + 1, f'구간{i}'), 1.0 - i / 10) for i in range(3)]
        counts = iter([5000, 4000, 3000])
        selected, notes, limits = fit_own_passages(chunks, lambda sel: sel, lambda _: next(counts), 4096, 512)
        self.assertEqual(selected, chunks[:1])
        self.assertEqual(limits, {'linked_passages': 0, 'table_rows': 0, 'tables': 0, 'own_passages': 2})
        self.assertEqual(notes, ['모델 문맥 한도로 적용 문서의 관련도 낮은 구간 2개를 빼었습니다.'])

    def test_fit_own_passages_works_on_plain_pdf_chunks(self):
        chunks = [(Chunk(1, 0, 3, 'abc'), 0.9), (Chunk(1, 3, 6, 'def'), 0.5)]
        selected, _, limits = fit_own_passages(chunks, lambda sel: sel, lambda _: 9999, 4096, 512)
        self.assertEqual(len(selected), 1)
        self.assertEqual(limits['own_passages'], 1)

    def test_fit_own_passages_leaves_everything_when_it_fits_or_count_unknown(self):
        chunks = [(Chunk(1, 0, 3, 'abc'), 0.9), (Chunk(1, 3, 6, 'def'), 0.5)]
        for count in (100, None):
            selected, notes, limits = fit_own_passages(chunks, lambda sel: sel, lambda _: count, 4096, 512)
            self.assertEqual((selected, notes, limits['own_passages']), (chunks, [], 0))


class CaseAnswerTests(unittest.TestCase):
    def test_case_quote_requires_verbatim_match(self):
        result, seen = answer('해외 출장 식비는 하루 3만원이다. 숙박비는 정액으로 지급한다.')
        self.assertEqual(result['status'], 'quoted')
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0]['prediction']['answer'], '해외 출장 식비는 하루 3만원이다.')
        self.assertIn('숙박비는 정액으로 지급한다.', result['commentary'])
        self.assertEqual(seen['messages'][0]['role'], 'system')

    def test_case_withholds_sentence_with_unsupported_number(self):
        result, _ = answer('해외 출장 식비는 하루 3만원이다. 식비는 하루 9만원이다.')
        self.assertNotIn('9만원', result['commentary'])
        self.assertEqual(result['unsupported_numbers'], ['9'])
        self.assertTrue(any('숨겼습니다' in n for n in result['commentary_notes']))

    def test_case_overflow_raises_context_overflow(self):
        with self.assertRaises(DocumentError) as raised:
            answer('', skipped_context=True)
        self.assertEqual(str(raised.exception), CONTEXT_OVERFLOW)


if __name__ == '__main__':
    unittest.main()
