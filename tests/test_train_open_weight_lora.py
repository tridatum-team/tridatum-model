"""Pinned LoRA rows preserve prior dialogue without supervising prior answers.

Uses a real byte-level tokenizer and chat template, never model weights or downloads.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from scripts import train_open_weight_lora as trainer
from trid.prompts import rag_prompt


TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}"
    "{% endif %}{% endif %}"
)


def local_tokenizer():
    # Complete byte alphabet: Korean history cannot collapse into a shared unknown token.
    tokens = ['<|endoftext|>', '<|im_start|>', '<|im_end|>'] + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={text: i for i, text in enumerate(tokens)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>'], chat_template=TEMPLATE,
        clean_up_tokenization_spaces=False,
    )


class HistoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = local_tokenizer()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def pinned(self, rows, adapter='prompt_answer', **entry_fields):
        path = self.base / 'synthetic.jsonl'
        # Preserve physical line numbering even when the pinned file starts with a blank line.
        path.write_text('\n' + ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        entry = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), adapter=adapter)
        entry.update(entry_fields)
        return entry

    def encode_row(self, row, adapter='prompt_answer'):
        loaded, = trainer.load_rows(self.pinned([row], adapter), self.base)
        return loaded, trainer.encode(self.tok, loaded, 'qwen3', 10000)

    def assert_encoded(self, item, prompt, answer):
        prefix = self.tok(prompt, add_special_tokens=False)['input_ids']
        target = self.tok(answer + self.tok.eos_token, add_special_tokens=False)['input_ids']
        self.assertEqual(item['ids'], prefix + target)
        self.assertEqual(item['labels'], [-100] * len(prefix) + target)
        self.assertEqual(item['labels'][-1], self.tok.eos_token_id)

    def test_prior_context_and_corrections_preserved_but_only_final_answer_supervised(self):
        raw = {
            'id': 'history-correction', 'task': 'synthetic_history',
            'history': [
                {'user': '상자 표시는 청록색이야.', 'assistant': '청록색으로 기억할게요.'},
                {'user': '아니, 자주색으로 정정해.', 'assistant': '상자 표시를 자주색으로 정정했어요.'},
            ],
            'prompt': '정정한 색 이름만 알려줘.', 'answer': '자주색',
        }
        loaded, item = self.encode_row(raw)
        expected = (
            '<|im_start|>user\n상자 표시는 청록색이야.<|im_end|>\n'
            '<|im_start|>assistant\n청록색으로 기억할게요.<|im_end|>\n'
            '<|im_start|>user\n아니, 자주색으로 정정해.<|im_end|>\n'
            '<|im_start|>assistant\n상자 표시를 자주색으로 정정했어요.<|im_end|>\n'
            '<|im_start|>user\n정정한 색 이름만 알려줘.<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        )
        self.assert_encoded(item, expected, '자주색')
        self.assertEqual(self.tok.decode(item['ids']), expected + '자주색<|im_end|>')
        # The context limit rejects the whole sample rather than removing an old turn.
        self.assertIsNone(trainer.encode(self.tok, loaded, 'qwen3', len(item['ids']) - 1))
        self.assertEqual(trainer.encode(self.tok, loaded, 'qwen3', len(item['ids'])), item)

    def test_rag_system_stays_before_history(self):
        question = rag_prompt([('D1', '표식은 은색이다.')], '표식은?')
        _, item = self.encode_row({
            'id': 'rag-history', 'prompt': question, 'answer': '은색 [D1]',
            'history': [{'user': '표식만 확인하자.', 'assistant': '표식을 확인하겠습니다.'}],
        })
        expected = (
            f'<|im_start|>system\n{trainer.SYSTEM}<|im_end|>\n'
            '<|im_start|>user\n표식만 확인하자.<|im_end|>\n'
            '<|im_start|>assistant\n표식을 확인하겠습니다.<|im_end|>\n'
            f'<|im_start|>user\n{question}<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        )
        self.assert_encoded(item, expected, '은색 [D1]')

    def test_history_free_prompt_and_case_keep_existing_tokens_and_labels(self):
        rows = [
            ('prompt_answer', {'id': 'plain', 'prompt': '짧게 인사해.', 'answer': '반갑습니다.'}, None),
            ('prompt_answer', {'id': 'rag', 'prompt': rag_prompt([('D1', '색은 은색이다.')], '색은?'),
                               'answer': '은색 [D1]'}, trainer.SYSTEM),
            ('case', {'id': 'case', 'question': '색은?', 'context': ['색은 은색이다.'],
                      'target': '은색입니다 [D1].'}, trainer.CASE_SYSTEM),
        ]
        for adapter, raw, system in rows:
            with self.subTest(adapter=adapter, row=raw['id']):
                loaded, item = self.encode_row(raw, adapter)
                # Independent pre-history rendering contract, not messages_of().
                messages = ([{'role': 'system', 'content': system}] if system else [])
                messages += [{'role': 'user', 'content': loaded['user']}]
                prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
                self.assert_encoded(item, prompt, loaded['answer'])
                _, empty_history = self.encode_row(dict(raw, history=[]), adapter)
                self.assertEqual(empty_history, item)
                if adapter == 'case':
                    count = sum(label != -100 for label in item['labels'])
                    self.assertIsNone(trainer.case_contract_violation(loaded, count, 1000))
                    self.assertEqual(trainer.case_contract_violation(loaded, count, count - 1), 'over_answer_budget')
                    self.assertEqual(trainer.case_contract_violation(dict(loaded, answer='은색입니다.'), 5, 1000),
                                     'no_citation')

    def test_invalid_history_fails_with_file_physical_line_and_row_identity(self):
        invalid = [
            None,
            {'user': '앞 질문', 'assistant': '앞 답변'},
            ['not a turn'],
            [{'role': 'user', 'content': '역할별 메시지 형식은 이 스키마가 아님'}],
            [{'user': '앞 질문'}],
            [{'user': '앞 질문', 'assistant': '앞 답변', 'system': '추가 역할'}],
            [{'user': ' \n', 'assistant': '앞 답변'}],
            [{'user': '앞 질문', 'assistant': 7}],
            [{'user': '앞 질문', 'assistant': '<|assistant|>예약 제어 토큰'}],
        ]
        for history in invalid:
            with self.subTest(history=history):
                entry = self.pinned([{'id': 'malformed-history', 'prompt': '현재 질문', 'answer': '현재 답변',
                                      'history': history}])
                with self.assertRaises(ValueError) as caught:
                    trainer.load_rows(entry, self.base)
                message = str(caught.exception)
                self.assertIn(f'{self.base / entry["path"]}:2:', message)
                self.assertIn("row 'malformed-history'", message)

    def test_case_history_is_rejected_instead_of_silently_dropped(self):
        for history in ([{'user': '앞 질문', 'assistant': '앞 답변'}], None):
            with self.subTest(history=history):
                entry = self.pinned([{'id': 'case-history', 'question': '색은?', 'context': ['색은 은색이다.'],
                                      'target': '은색 [D1]', 'history': history}], 'case')
                with self.assertRaises(ValueError) as caught:
                    trainer.load_rows(entry, self.base)
                self.assertIn(f'{self.base / entry["path"]}:2:', str(caught.exception))
                self.assertIn("row 'case-history'", str(caught.exception))
                if history:
                    self.assertIn('case adapter does not support history', str(caught.exception))

    def test_pinned_hash_and_selected_split_still_gate_history_rows(self):
        entry = self.pinned([
            {'id': 'excluded', 'split': 'hold', 'history': None},
            {'id': 'selected', 'split': 'train', 'prompt': '색은?', 'answer': '은색',
             'history': [{'user': '은색으로 정해.', 'assistant': '은색입니다.'}]},
        ], split='train')
        rows = trainer.load_rows(entry, self.base)
        self.assertEqual([row['id'] for row in rows], ['selected'])
        path = self.base / entry['path']
        path.write_text(path.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'sha256 does not match'):
            trainer.load_rows(entry, self.base)

    def test_shifted_batch_loss_masks_history_and_padding_not_current_answer(self):
        _, long = self.encode_row({
            'id': 'long', 'prompt': '색은?', 'answer': '자주색',
            'history': [{'user': '표시는 자주색으로 정해.', 'assistant': '자주색으로 기억합니다.'}],
        })
        _, short = self.encode_row({'id': 'short', 'prompt': '인사해.', 'answer': '안녕.'})
        items = [long, short]
        width = max(len(item['ids']) for item in items)
        # Analytic logits, no model: the production loss must give gradient only at shifted final-target positions.
        logits = torch.zeros((2, width, len(self.tok)), requires_grad=True)
        loss = trainer.batch_loss(lambda **kwargs: SimpleNamespace(logits=logits), items, 'cpu', self.tok.pad_token_id)
        loss.backward()
        for index, item in enumerate(items):
            expected = [label != -100 for label in item['labels'][1:]] + [False] * (width - len(item['labels']) + 1)
            actual = logits.grad[index].abs().sum(-1).ne(0).tolist()
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
