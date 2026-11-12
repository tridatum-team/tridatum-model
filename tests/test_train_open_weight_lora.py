"""Pinned LoRA rows preserve prior dialogue without supervising prior answers; --checkpoint-every snapshots.

Uses a real byte-level tokenizer and chat template; CheckpointTests train a randomly initialised tiny Qwen3 saved to
a temporary directory. No downloads.
"""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
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

    def grounded_row(self):
        return {
            'id': 'grounded-example', 'task': 'grounded_explanation', 'split': 'train',
            'messages': [
                {'role': 'system', 'content': '  근거에 따라 JSON 문단으로 설명하세요.\n'},
                {'role': 'user', 'content': '\n질문: 표식은?\n[D1] 표식은 은색이다.  '},
            ],
            'answer': '{"paragraphs":[{"text":"은색 표식이라는 최종 설명입니다.","citations":["D1"]}]}',
        }

    def test_grounded_messages_preserve_literal_prompt_and_supervise_only_answer_and_eos(self):
        raw = self.grounded_row()
        loaded, item = self.encode_row(raw, 'grounded_messages')
        self.assertEqual(trainer.messages_of(loaded), raw['messages'])
        expected = (
            '<|im_start|>system\n  근거에 따라 JSON 문단으로 설명하세요.\n<|im_end|>\n'
            '<|im_start|>user\n\n질문: 표식은?\n[D1] 표식은 은색이다.  <|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        )
        self.assert_encoded(item, expected, raw['answer'])
        prompt_ids = [token for token, label in zip(item['ids'], item['labels']) if label == -100]
        prompt = self.tok.decode(prompt_ids)
        self.assertEqual(prompt, expected)
        self.assertNotIn(raw['answer'], prompt)
        self.assertNotIn('최종 설명', prompt)

    def test_grounded_messages_reject_malformed_roles_content_and_history(self):
        raw = self.grounded_row()
        system, user = raw['messages']
        invalid = [
            {'messages': None}, {'messages': {}}, {'messages': []},
            {'messages': [system]}, {'messages': [user, system]},
            {'messages': [system, dict(user, role='assistant')]},
            {'messages': [system, user, {'role': 'assistant', 'content': raw['answer']}]},
            {'messages': [system, user, system, user]},
            {'messages': [system, 'not a message']},
            {'messages': [system, {'role': 'user'}]},
            {'messages': [system, dict(user, history=[])]},
        ]
        for history in ([], None, [{'user': '앞 질문', 'assistant': '앞 답변'}]):
            invalid.append({'history': history})
        for content in ('', ' \n\t', None, 7, [], {}):
            invalid.append({'messages': [dict(system, content=content), user]})
            invalid.append({'messages': [system, dict(user, content=content)]})
            for field in ('id', 'task', 'split', 'answer'):
                invalid.append({field: content})
        malformed = [dict(raw, **fields) for fields in invalid]
        malformed.extend({key: value for key, value in raw.items() if key != field}
                         for field in ('id', 'task', 'split', 'messages', 'answer'))
        for row in malformed:
            with self.subTest(row=row):
                entry = self.pinned([row], 'grounded_messages')
                with self.assertRaises(ValueError) as caught:
                    trainer.load_rows(entry, self.base)
                self.assertIn(f'{self.base / entry["path"]}:2:', str(caught.exception))
                self.assertIn(f"row {row.get('id')!r}", str(caught.exception))

    def test_grounded_budgets_include_eos_and_reject_instead_of_dropping_rows(self):
        raw = self.grounded_row()
        _, encoded = self.encode_row(raw, 'grounded_messages')
        context_tokens = len(encoded['ids'])
        answer_tokens = sum(label != -100 for label in encoded['labels'])
        self.assertEqual(answer_tokens, len(self.tok(raw['answer'], add_special_tokens=False)['input_ids']) + 1)
        entry = self.pinned([raw], 'grounded_messages', epochs=2)
        items, dropped = trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                             context_tokens, answer_tokens, True)
        self.assertEqual(items, [encoded, encoded])
        self.assertEqual(dropped, 0)
        valid, valid_dropped = trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                                   context_tokens, answer_tokens, False)
        self.assertEqual(valid, [encoded])
        self.assertEqual(valid_dropped, 0)
        # A fitting row must not let the builder return a partial dataset when the next row is lossy.
        first = dict(raw, id='fits', answer='{}')
        entry = self.pinned([first, raw], 'grounded_messages')
        for context_limit, answer_limit, reason in (
            (context_tokens - 1, answer_tokens, 'over_context_budget'),
            (context_tokens, answer_tokens - 1, 'over_answer_budget'),
            (context_tokens, 0, 'over_answer_budget'),
        ):
            for expand in (False, True):
                with self.subTest(context=context_limit, answer=answer_limit, expand=expand):
                    with self.assertRaisesRegex(ValueError, reason):
                        trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                            context_limit, answer_limit, expand)

    def test_legacy_manifest_build_retains_whole_row_drop_and_case_answer_contract(self):
        raw = {'id': 'legacy', 'prompt': '인사해.', 'answer': '반갑습니다.'}
        _, encoded = self.encode_row(raw)
        entry = self.pinned([raw], epochs=2)
        self.assertEqual(trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                             len(encoded['ids']) - 1, 1, True), ([], 2))
        self.assertEqual(trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                             len(encoded['ids']), 1, True), ([encoded, encoded], 0))
        case = {'id': 'case', 'question': '색은?', 'context': ['색은 은색이다.'], 'target': '은색 [D1]'}
        _, encoded = self.encode_row(case, 'case')
        entry = self.pinned([case], 'case')
        answer_tokens = sum(label != -100 for label in encoded['labels'])
        self.assertEqual(trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                             len(encoded['ids']), answer_tokens, False), ([encoded], 0))
        with self.assertRaisesRegex(ValueError, 'case:over_answer_budget'):
            trainer.build_items(self.tok, [entry], self.base, 'qwen3',
                                len(encoded['ids']), answer_tokens - 1, False)

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

    def tiny_qwen3(self, tie_word_embeddings=True):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(self.tok), hidden_size=32, intermediate_size=64,
                                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                               head_dim=16, tie_word_embeddings=tie_word_embeddings,
                                               attention_dropout=0.0))
        model.eval()  # deterministic; gradients still flow
        return model

    def two_rows(self):
        """A history row and a short row: different answer spans (union path), right padding, and a supervised
        count that is not a multiple of the chunk size (ragged last chunk)."""
        _, long = self.encode_row({
            'id': 'long', 'prompt': '색은?', 'answer': '자주색',
            'history': [{'user': '표시는 자주색으로 정해.', 'assistant': '자주색으로 기억합니다.'}],
        })
        _, short = self.encode_row({'id': 'short', 'prompt': '인사해.', 'answer': '안녕.'})
        return [long, short]

    def compare_families(self, model, items):
        """(loss, grads, context grad) for the generic full-logits path and the qwen3 chunked path, with a chunk
        size that splits the supervised union into several chunks and a ragged last one."""
        context_grads = []

        def keep_context(_module, _inputs, output):  # the embedded sequence the model actually read
            output.retain_grad()
            context_grads.append(output)

        hook = model.get_input_embeddings().register_forward_hook(keep_context)
        self.addCleanup(hook.remove)
        supervised = len({p for item in items for p, label in enumerate(item['labels'][1:]) if label != -100})
        chunk = next(c for c in (3, 4, 5) if supervised % c)
        self.assertGreater(supervised, chunk)
        self.addCleanup(setattr, trainer, 'LOGIT_CHUNK', trainer.LOGIT_CHUNK)
        trainer.LOGIT_CHUNK = chunk

        def run(family):
            model.zero_grad(set_to_none=True)
            loss = trainer.batch_loss(model, items, 'cpu', self.tok.pad_token_id, family)
            loss.backward()
            grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            return loss.detach(), grads, context_grads[-1].grad.clone()

        return run('generic'), run('qwen3')

    def test_answer_position_chunked_loss_equals_full_logits_loss_and_every_gradient(self):
        """--family qwen3 runs the head only at supervised positions, in checkpointed chunks; loss, every parameter
        gradient (tied head included) and the gradient reaching the context embeddings must equal the full-width
        computation on a real causal LM with two rows, padding and a ragged last chunk."""
        model = self.tiny_qwen3()
        items = self.two_rows()
        (full_loss, full_grads, full_context), (opt_loss, opt_grads, opt_context) = self.compare_families(model, items)
        self.assertTrue(torch.allclose(full_loss, opt_loss, atol=1e-5))
        self.assertEqual(set(full_grads), set(dict(model.named_parameters())))
        self.assertEqual(set(full_grads), set(opt_grads))
        for name in full_grads:
            self.assertTrue(torch.allclose(full_grads[name], opt_grads[name], atol=1e-5), name)
        self.assertTrue(torch.allclose(full_context, opt_context, atol=1e-5))
        first_answer = next(p for p, label in enumerate(items[0]['labels']) if label != -100)
        self.assertTrue(full_context[0, :first_answer].abs().sum().item() > 0)

    def test_chunked_loss_matches_full_path_through_a_frozen_base_with_trainable_lora(self):
        """The production shape: frozen base, trainable LoRA on the seven projections, head untouched. The qwen3
        chunked path must give the same loss, the same gradient on every adapter parameter and the same context
        gradient as the full-logits path, and no gradient on frozen parameters in either."""
        from peft import LoraConfig, get_peft_model
        for tied in (True, False):
            with self.subTest(tie_word_embeddings=tied):
                base = self.tiny_qwen3(tie_word_embeddings=tied)
                self.assertEqual(base.lm_head.weight is base.get_input_embeddings().weight, tied)
                model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                                       target_modules=trainer.TARGETS, task_type='CAUSAL_LM'))
                with torch.no_grad():  # nonzero B so the adapter changes the forward pass
                    for name, p in model.named_parameters():
                        if 'lora_B' in name:
                            p.add_(torch.randn_like(p))
                model.enable_input_require_grads()  # the embedding output carries the context gradient
                model.eval()
                items = self.two_rows()
                (full_loss, full_grads, full_context), (opt_loss, opt_grads, opt_context) = self.compare_families(model, items)
                trainable = {name for name, p in model.named_parameters() if p.requires_grad}
                self.assertTrue(trainable and all('lora_' in name for name in trainable))
                self.assertEqual(set(full_grads), trainable)
                self.assertEqual(set(opt_grads), trainable)
                self.assertTrue(torch.allclose(full_loss, opt_loss, atol=1e-5))
                for name in trainable:
                    self.assertTrue(torch.allclose(full_grads[name], opt_grads[name], atol=1e-5), name)
                self.assertTrue(torch.allclose(full_context, opt_context, atol=1e-5))


class PackedHeadTests(unittest.TestCase):
    """Real tensor losses/gradients and the supervised-token logits workspace contract."""

    def model(self):
        class Backbone(torch.nn.Embedding):
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=super().forward(input_ids))

        class SyntheticLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone(24, 4)
                self.lm_head = torch.nn.Linear(4, 11)

            def forward(self, input_ids, attention_mask):
                hidden = self.model(input_ids, attention_mask).last_hidden_state
                return SimpleNamespace(logits=self.lm_head(hidden))

        with torch.random.fork_rng():
            torch.manual_seed(19)
            return SyntheticLM()

    def test_packed_losses_and_signed_sequence_gradients_match_dense_with_holes_and_empty_rows(self):
        ids = torch.arange(24).reshape(3, 8)
        mask = torch.ones_like(ids)
        keep = torch.tensor([0, 2, 3, 6, 7])
        dense = torch.arange(15).reshape(3, 5) % 11
        holes = dense.clone()
        holes[0, [0, 2, 4]] = -100
        holes[2, [1, 3]] = -100
        empty_row = holes.clone()
        empty_row[1] = -100
        with unittest.mock.patch.object(trainer, 'LOGIT_CHUNK', 2):
            for target in (dense, holes, empty_row):
                with self.subTest(target=target.tolist()):
                    model = self.model()
                    hidden = model.model(ids, mask).last_hidden_state
                    expected = trainer.chunk_cross_entropy(hidden[:, keep], model.lm_head, target)
                    weights = torch.tensor([0.7, -0.2, 0.4])
                    (expected.sum(1) * weights).sum().backward()
                    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
                    model.zero_grad(set_to_none=True)
                    head_inputs = []
                    hook = model.lm_head.register_forward_pre_hook(
                        lambda _module, args: head_inputs.append(args[0].detach().clone()))
                    try:
                        actual = trainer.chunked_answer_loss(model, ids, mask, keep, target)
                    finally:
                        hook.remove()
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(actual[target == -100].count_nonzero().item(), 0)
                    # The head sees every supervised token once, no ignored row-position pair, and never a
                    # logits workspace larger than the previous batch_size * LOGIT_CHUNK budget.
                    rows, columns = (target != -100).nonzero(as_tuple=True)
                    observed = torch.cat([chunk.reshape(-1, 4) for chunk in head_inputs])
                    self.assertCountEqual(observed.tolist(), hidden.detach()[rows, keep[columns]].tolist())
                    self.assertTrue(all(chunk.numel() // 4 <= ids.shape[0] * trainer.LOGIT_CHUNK
                                        for chunk in head_inputs))
                    (actual.sum(1) * weights).sum().backward()
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)

    def test_all_ignored_sft_preserves_zero_gradients_for_trainable_and_frozen_heads(self):
        items = [dict(ids=[0, 1, 2, 3], labels=[-100] * 4),
                 dict(ids=[4, 5], labels=[-100] * 2)]
        for frozen_head in (False, True):
            with self.subTest(frozen_head=frozen_head):
                model = self.model()
                model.lm_head.requires_grad_(not frozen_head)
                results = []
                for family in ('generic', 'qwen3'):
                    model.zero_grad(set_to_none=True)
                    loss = trainer.batch_loss(model, items, 'cpu', 0, family)
                    loss.backward()
                    self.assertEqual(loss.item(), 0)
                    results.append({name: None if p.grad is None else p.grad.clone()
                                    for name, p in model.named_parameters()})
                for name, parameter in model.named_parameters():
                    before, after = results[0][name], results[1][name]
                    if parameter.requires_grad:
                        self.assertIsNotNone(after, name)
                        torch.testing.assert_close(after, before, atol=0, rtol=0)
                        self.assertEqual(after.count_nonzero().item(), 0, name)
                    else:
                        self.assertIsNone(before, name)
                        self.assertIsNone(after, name)


class CheckpointTests(unittest.TestCase):
    """--checkpoint-every through main() on a real tiny Qwen3 saved to disk: snapshots are distinct, attributed to
    their step, reload independently to the recorded validation loss, and an interrupted run never looks complete."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        tok = local_tokenizer()
        cls.model_dir = cls.base / 'tiny-qwen3'
        tok.save_pretrained(cls.model_dir)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                         tie_word_embeddings=True, attention_dropout=0.0)).save_pretrained(cls.model_dir)
        rows = {
            'train.jsonl': [{'id': f'train-{i}', 'task': ('color', 'greeting')[i % 2], 'prompt': f'{i}번 질문', 'answer': ('자주색', '안녕.')[i % 2],
                             'history': [{'user': '색은 자주색.', 'assistant': '기억합니다.'}] if i % 2 == 0 else []} for i in range(4)],
            'valid.jsonl': [{'id': 'valid-color', 'task': 'color', 'prompt': '색은?', 'answer': '자주색'},
                            {'id': 'valid-greeting', 'task': 'greeting', 'prompt': '인사해.', 'answer': '안녕.'}],
        }
        entries = {}
        for name, items in rows.items():
            path = cls.base / name
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items), encoding='utf-8')
            entries[name] = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), adapter='prompt_answer')
        cls.manifest = cls.base / 'manifest.json'
        cls.manifest.write_text(json.dumps({'id': 'tiny-checkpoints', 'train': [entries['train.jsonl']],
                                            'valid': [entries['valid.jsonl']]}))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_trainer(self, out, *extra):
        argv = ['train_open_weight_lora.py', '--model', str(self.model_dir), '--model-id', 'tiny/qwen3',
                '--model-revision', 'rev-tiny', '--manifest', str(self.manifest), '--out', str(out), '--device', 'cpu',
                '--batch-size', '1', '--micro-batch-size', '1', '--lr', '1e-2', '--rank', '2', '--alpha', '4',
                '--valid-every', '50', *extra]
        with unittest.mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            trainer.main()

    @staticmethod
    def weights(adapter):
        from safetensors.torch import load_file
        return load_file(str(adapter / 'adapter_model.safetensors'))

    def assert_same_weights(self, x, y, equal):
        self.assertEqual(set(x), set(y))
        self.assertEqual(all(torch.equal(x[k], y[k]) for k in x), equal)

    def test_checkpoints_are_distinct_attributed_and_reload_to_their_recorded_validation(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        out = self.base / 'run-checkpoints'
        self.run_trainer(out, '--steps', '4', '--checkpoint-every', '2')
        self.assertEqual(sorted(p.name for p in (out / 'checkpoints').iterdir()), ['step-000002', 'step-000004'])
        step2, step4 = (out / 'checkpoints' / f'step-00000{s}' / 'adapter' for s in (2, 4))
        self.assert_same_weights(self.weights(step2), self.weights(step4), equal=False)  # progression, not copies
        self.assert_same_weights(self.weights(step4), self.weights(out / 'adapter'), equal=True)  # last step stays exact
        manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        meta = json.loads((step2.parent / 'training.json').read_text())
        self.assertEqual((meta['status'], meta['steps'], meta['planned_steps']), ('checkpoint', 2, 4))
        self.assertEqual((meta['base_model_id'], meta['base_revision'], meta['manifest_sha256'], meta['manifest_id']),
                         ('tiny/qwen3', 'rev-tiny', manifest_sha, 'tiny-checkpoints'))
        self.assertEqual([h['step'] for h in meta['history']], [1, 2])  # nothing from after this step leaks in
        self.assertEqual(meta['checkpoint']['step'], 2)
        self.assertIsInstance(meta['checkpoint']['elapsed_seconds'], float)
        recorded = meta['checkpoint']['valid_loss_by_task']
        self.assertEqual(sorted(recorded), ['color', 'greeting'])
        # The saved snapshot on a fresh base reproduces the validation loss recorded for its own step.
        tok = AutoTokenizer.from_pretrained(self.model_dir)
        valid = [trainer.encode(tok, r, 'qwen3', 1536)
                 for r in trainer.load_rows(json.loads(self.manifest.read_text())['valid'][0], self.base)]
        base = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32)
        reloaded = PeftModel.from_pretrained(base, str(step2), is_trainable=False)
        measured = trainer.validate(reloaded, valid, 'cpu', tok.pad_token_id, 1, 'qwen3')
        for task in recorded:
            self.assertAlmostEqual(measured[task], recorded[task], places=3, msg=task)
        final = json.loads((out / 'training.json').read_text())
        self.assertEqual(final['status'], 'complete')
        self.assertEqual([c['step'] for c in final['checkpoints']], [2, 4])
        self.assertEqual(final['selection']['status'], 'unselected')
        self.assertTrue(final['last_step']['equals_last_checkpoint'])
        ledger = json.loads((out / 'checkpoints.json').read_text())
        self.assertEqual(ledger['run_status'], 'complete')
        self.assertEqual(ledger['checkpoints'], final['checkpoints'])

    def test_interrupted_run_keeps_earlier_checkpoints_but_is_never_complete(self):
        out = self.base / 'run-interrupted'
        with self.assertRaisesRegex(SystemExit, 'max-train-seconds'):
            self.run_trainer(out, '--steps', '8', '--checkpoint-every', '2', '--max-train-seconds', '1e-9')
        self.assertEqual(sorted(p.name for p in (out / 'checkpoints').iterdir()), ['step-000002', 'step-000004'])
        self.assertFalse((out / 'adapter').exists())
        self.assertFalse((out / 'training.json').exists())
        ledger = json.loads((out / 'checkpoints.json').read_text())
        self.assertEqual((ledger['run_status'], ledger['planned_steps']), ('in_progress', 8))
        self.assertEqual([c['step'] for c in ledger['checkpoints']], [2, 4])
        self.assertEqual(ledger['selection']['status'], 'unselected')
        for folder in (out / 'checkpoints').iterdir():
            self.assertEqual(json.loads((folder / 'training.json').read_text())['status'], 'checkpoint')

    def test_default_run_writes_no_checkpoint_artifacts(self):
        out = self.base / 'run-default'
        self.run_trainer(out, '--steps', '2')
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['adapter', 'training.json'])
        final = json.loads((out / 'training.json').read_text())
        self.assertEqual((final['status'], final['steps']), ('complete', 2))
        self.assertNotIn('checkpoints', final)


class PreadLoaderTests(unittest.TestCase):
    """load_qwen3_pread on a real tiny Qwen3 (CPU, explicit device): every parameter, the tied storage, buffers,
    generation config, dtype, eval mode and logits equal from_pretrained; a tied lm_head in the checkpoint is
    shape-checked but its bytes never used; missing, unexpected or mis-shaped weights are refused before loading."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        tok = local_tokenizer()
        cls.model_dir = cls.base / 'tiny-qwen3'
        tok.save_pretrained(cls.model_dir)
        with torch.random.fork_rng():
            torch.manual_seed(1)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                                 num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                                 tie_word_embeddings=True, attention_dropout=0.0))
        model.to(torch.bfloat16).save_pretrained(cls.model_dir)  # bf16 on disk, as the pinned base
        generation = cls.model_dir / 'generation_config.json'
        payload = json.loads(generation.read_text())
        payload['max_new_tokens'] = 7  # distinctive: must come from the file, not from the model config
        generation.write_text(json.dumps(payload))
        cls.ids = torch.tensor([tok('<|im_start|>user\n색은?<|im_end|>\n', add_special_tokens=False)['input_ids']])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def canonical(model_dir, dtype):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)

    def sharded_copy(self, name, tensors, shard2=None):
        """A checkpoint directory like the pinned base: shard 1 = unique weights, shard 2 = the tensors in `shard2`."""
        from safetensors.torch import save_file
        folder = self.base / name
        folder.mkdir()
        for f in self.model_dir.iterdir():
            if not f.name.startswith('model'):
                (folder / f.name).write_bytes(f.read_bytes())
        shards = {'model-00001-of-00002.safetensors': tensors, 'model-00002-of-00002.safetensors': shard2 or {}}
        weight_map = {}
        for file, items in shards.items():
            if items:
                save_file({k: v.contiguous() for k, v in items.items()}, folder / file, metadata={'format': 'pt'})
                weight_map.update({k: file for k in items})
        (folder / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}))
        return folder

    def unique_tensors(self):
        from safetensors.torch import load_file
        return load_file(str(self.model_dir / 'model.safetensors'))

    def assert_equivalent(self, model, canonical, dtype, name_or_path=None):
        ours, theirs = dict(model.state_dict()), dict(canonical.state_dict())
        self.assertEqual(set(ours), set(theirs))
        for key in theirs:
            self.assertEqual(ours[key].dtype, dtype, key)
            self.assertEqual(ours[key].device.type, 'cpu', key)
            self.assertTrue(torch.equal(ours[key], theirs[key]), key)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)  # tied storage, one unique tensor
        self.assertEqual(sum(1 for _ in model.parameters()), sum(1 for _ in canonical.parameters()))
        buffers = dict(model.named_buffers())
        self.assertEqual(set(buffers), set(dict(canonical.named_buffers())))
        self.assertTrue(buffers)
        for key, value in canonical.named_buffers():
            self.assertEqual(buffers[key].dtype, value.dtype, key)
            self.assertTrue(torch.equal(buffers[key], value), key)
        self.assertFalse(model.training)
        self.assertEqual(model.generation_config.max_new_tokens, 7)
        self.assertEqual(model.generation_config.to_dict(), canonical.generation_config.to_dict())
        self.assertEqual(model.config.dtype, dtype)
        self.assertEqual(model.config._attn_implementation, canonical.config._attn_implementation)
        self.assertEqual(model.name_or_path, str(name_or_path or self.model_dir))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(input_ids=self.ids).logits, canonical(input_ids=self.ids).logits))

    def test_cpu_load_equals_from_pretrained_in_checkpoint_and_cast_dtype(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                model, report = trainer.load_qwen3_pread(self.model_dir, dtype, 'cpu')
                self.assert_equivalent(model, self.canonical(self.model_dir, dtype), dtype)
                unique = len(model.state_dict()) - 1
                self.assertEqual((report['backend'], report['staging'], report['tied_output_keys'], report['input_key']),
                                 ('pread', 'cpu-per-tensor', ['lm_head.weight'], 'model.embed_tokens.weight'))
                self.assertEqual(report['parameters'], unique)
                self.assertEqual(report['cast'], 0 if dtype == torch.bfloat16 else unique)
                self.assertEqual([s['file'] for s in report['shards']], ['model.safetensors'])
                self.assertGreater(report['bytes_loaded'], 0)

    def test_tied_lm_head_shard_is_shape_checked_but_never_read(self):
        tensors = self.unique_tensors()
        self.assertNotIn('lm_head.weight', tensors)
        embed = tensors['model.embed_tokens.weight']
        garbage = torch.full_like(embed, 3.0)  # would change every logit if the loader used it
        folder = self.sharded_copy('tied-head', tensors, {'lm_head.weight': garbage})
        model, report = trainer.load_qwen3_pread(folder, torch.bfloat16, 'cpu')
        self.assert_equivalent(model, self.canonical(self.model_dir, torch.bfloat16), torch.bfloat16, name_or_path=folder)
        second = report['shards'][1]
        self.assertEqual((second['file'], second['loaded'], second['skipped_tied']),
                         ('model-00002-of-00002.safetensors', 0, ['lm_head.weight']))
        self.assertEqual(second['bytes_skipped'], embed.numel() * 2)
        self.assertEqual(report['bytes_skipped'], embed.numel() * 2)
        self.assertEqual(report['parameters'], len(tensors))

    def test_missing_unexpected_or_misshaped_weights_are_refused_before_loading(self):
        tensors = self.unique_tensors()
        embed = tensors['model.embed_tokens.weight']
        cases = {
            'missing': (dict((k, v) for k, v in tensors.items() if k != 'model.norm.weight'), None, 'missing parameters.*model.norm.weight'),
            'unexpected': (dict(tensors, **{'model.extra.weight': torch.ones(2)}), None, 'unexpected key model.extra.weight'),
            'misshaped': (dict(tensors, **{'model.norm.weight': torch.ones(3, dtype=torch.bfloat16)}), None, 'model.norm.weight: checkpoint shape \\[3\\]'),
            'misshaped-tied-head': (tensors, {'lm_head.weight': embed[:, :8].clone()}, 'lm_head.weight: checkpoint shape'),
        }
        for name, (shard1, shard2, message) in cases.items():
            with self.subTest(name):
                folder = self.sharded_copy(name, shard1, shard2)
                with self.assertRaisesRegex(ValueError, message):
                    trainer.load_qwen3_pread(folder, torch.bfloat16, 'cpu')


class Q8GgufBaseTests(unittest.TestCase):
    """q8_gguf_base on a tiny synthetic Q8_0 GGUF (CPU, float32 compute): dequantised values equal gguf's reference
    dequantisation, the head shares the embedding's int8 weight AND fp16 scales, forward logits / input gradients /
    LoRA gradients through the chunked Q8 matmul equal a dense model holding the dequantised weights, malformed
    metadata is refused before any row is read, the trainer records the Q8 base and refuses --merge, and --init-adapter
    warm-starts from a saved checkpoint adapter (weights preserved, then trained on; mismatches refused before loading)."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.tok = local_tokenizer()
        cls.model_dir = cls.base / 'tiny-qwen3'
        cls.tok.save_pretrained(cls.model_dir)
        cls.config = Qwen3Config(vocab_size=len(cls.tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
                                 attention_dropout=0.0)
        with torch.random.fork_rng():
            torch.manual_seed(2)
            model = Qwen3ForCausalLM(cls.config)
        model.to(torch.bfloat16).save_pretrained(cls.model_dir)
        cls.unique = {k: v.float() for k, v in model.state_dict().items() if k != 'lm_head.weight'}
        cls.gguf = cls.base / 'tiny.q8_0.gguf'
        cls.sha = cls.write_gguf(cls.gguf, cls.unique)
        rows = {
            'train.jsonl': [{'id': f'train-{i}', 'task': ('color', 'greeting')[i % 2], 'prompt': f'{i}번 질문',
                             'answer': ('자주색', '안녕.')[i % 2]} for i in range(4)],
            'valid.jsonl': [{'id': 'valid-color', 'task': 'color', 'prompt': '색은?', 'answer': '자주색'}],
        }
        entries = {}
        for name, items in rows.items():
            path = cls.base / name
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items), encoding='utf-8')
            entries[name] = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), adapter='prompt_answer')
        cls.manifest = cls.base / 'manifest.json'
        cls.manifest.write_text(json.dumps({'id': 'tiny-q8', 'train': [entries['train.jsonl']], 'valid': [entries['valid.jsonl']]}))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def write_gguf(path, unique, arch='qwen3', file_type=7, block_count=2, drop=(), override=None, extra=None):
        """A Q8_0 GGUF as convert_hf_to_gguf writes it: 2-D tensors quantised by gguf.quants (the reference encoder),
        1-D tensors F32, llama.cpp names, ggml dims. `override` replaces an HF key's (numpy array, ggml type),
        `extra` adds GGUF-named tensors, `drop` omits HF keys. Returns the file's sha256."""
        import gguf
        import numpy as np
        name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, block_count)
        writer = gguf.GGUFWriter(str(path), arch)
        writer.add_block_count(block_count)
        writer.add_file_type(file_type)
        for key, value in unique.items():
            if key in drop:
                continue
            name = name_map.get_name(key, try_suffixes=('.weight',))
            if override and key in override:
                data, kind = override[key]
            elif value.dim() == 2:
                data, kind = gguf.quants.quantize(value.numpy().astype(np.float32), gguf.GGMLQuantizationType.Q8_0), gguf.GGMLQuantizationType.Q8_0
            else:
                data, kind = value.numpy().astype(np.float32), gguf.GGMLQuantizationType.F32
            writer.add_tensor(name, data, raw_dtype=kind)
        for name, (data, kind) in (extra or {}).items():
            writer.add_tensor(name, data, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def reference_state(cls):
        """What the GGUF holds, dequantised by gguf's own reference: Q8_0 2-D tensors, exact 1-D norms."""
        import gguf
        import numpy as np
        state = {}
        for key, value in cls.unique.items():
            if value.dim() == 2:
                blocks = gguf.quants.quantize(value.numpy().astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
                state[key] = torch.from_numpy(gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.Q8_0)).reshape(value.shape)
            else:
                state[key] = value.clone()
        return state

    @classmethod
    def dense_reference(cls):
        """A float32 Qwen3ForCausalLM holding exactly the dequantised GGUF values (tied head)."""
        from transformers import Qwen3ForCausalLM
        state = cls.reference_state()
        state['lm_head.weight'] = state['model.embed_tokens.weight']
        model = Qwen3ForCausalLM(cls.config)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def load(self, gguf_path=None, sha=None):
        from scripts import q8_gguf_base as q8
        return q8.load_qwen3_q8_gguf(self.model_dir, gguf_path or self.gguf, sha or self.sha, torch.float32, 'cpu')

    def items(self):
        rows = [dict(id='long', task='color', system=None, user='색은?', answer='자주색', file='f',
                     history=[{'user': '표시는 자주색으로 정해.', 'assistant': '자주색으로 기억합니다.'}]),
                dict(id='short', task='greeting', system=None, user='인사해.', answer='안녕.', file='f', history=[])]
        return [trainer.encode(self.tok, row, 'qwen3', 10000) for row in rows]

    def test_dequantised_values_match_reference_and_head_shares_weight_and_scales(self):
        from scripts import q8_gguf_base as q8
        model, report = self.load()
        reference = self.reference_state()
        q8_keys, f32_keys = [], []
        for key, value in reference.items():
            module = model.get_submodule(key.rpartition('.')[0])
            if value.dim() == 2:
                q8_keys.append(key)
                self.assertIsInstance(module, q8.Q8Embedding if key.endswith('embed_tokens.weight') else q8.Q8Linear)
                self.assertEqual((module.weight.dtype, module.scales.dtype), (torch.int8, torch.float16), key)
                self.assertEqual(tuple(module.scales.shape), (value.shape[0], value.shape[1] // 32), key)
                self.assertFalse(module.weight.requires_grad or module.scales.requires_grad, key)
                self.assertTrue(torch.equal(module.dequantize(), value), key)  # float32: int8 * fp16 is exact
                self.assertFalse(torch.equal(value, self.unique[key]), key)  # the Q8 base is not the bf16 base
            else:
                f32_keys.append(key)
                self.assertEqual(module.weight.dtype, torch.float32)
                self.assertTrue(torch.equal(module.weight, value), key)
        self.assertEqual((len(q8_keys), len(f32_keys)), (1 + 7 * 2, 1 + 4 * 2))
        head, embed = model.lm_head, model.model.embed_tokens
        self.assertIsInstance(head, q8.Q8Linear)
        self.assertIs(head.weight, embed.weight)
        self.assertIs(head.scales, embed.scales)
        self.assertEqual((head.in_features, head.out_features), (32, len(self.tok)))
        self.assertEqual(trainer.tied_output_keys(model), ({'lm_head.weight'}, 'model.embed_tokens.weight'))
        all_params = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(list(model.parameters())), len(all_params) - 2)  # the head adds no storage of its own
        self.assertTrue(all(not p.requires_grad for _, p in all_params))
        self.assertEqual([n for n, t in all_params if t.device.type != 'cpu'], [])
        self.assertFalse(model.training)
        self.assertEqual(model.name_or_path, str(self.model_dir))
        self.assertEqual((report['backend'], report['staging'], report['representation'], report['tensors'], report['skipped_tied']),
                         ('gguf-pread', 'cpu-row-chunk', 'q8_0', {'q8_0': 15, 'f32': 9}, []))
        self.assertEqual(report['gguf'], dict(path=str(self.gguf), sha256=self.sha, architecture='qwen3', file_type=7, block_count=2))
        self.assertEqual((report['parameters'], report['bytes_skipped']), (24, 0))
        self.assertGreater(report['bytes_loaded'], 0)
        # Full forward through Qwen3ForCausalLM (Q8 embedding gather, Q8 projections, tied Q8 head) equals the dense model.
        ids = torch.tensor([self.items()[0]['ids']])
        with torch.no_grad():
            ours, theirs = model(input_ids=ids).logits, self.dense_reference()(input_ids=ids).logits
        self.assertTrue(torch.allclose(ours, theirs, atol=1e-5, rtol=1e-5))

    def test_chunked_q8_matmul_gives_reference_outputs_and_input_gradients(self):
        from scripts import q8_gguf_base as q8
        model, _ = self.load()
        for module in (model.lm_head, model.model.layers[0].mlp.down_proj):
            dense = module.dequantize()
            for rows in (module.out_features, 7):  # one chunk, and ragged multi-chunk accumulation
                with self.subTest(out_features=module.out_features, rows=rows):
                    module.chunk_rows = rows
                    x = torch.randn(2, 5, module.in_features, requires_grad=True)
                    out = module(x)
                    g = torch.randn_like(out)
                    out.backward(g)
                    self.assertTrue(torch.allclose(out, torch.nn.functional.linear(x, dense), atol=1e-5))
                    self.assertTrue(torch.allclose(x.grad, g @ dense, atol=1e-5))
                    self.assertIsNone(module.weight.grad)
                    self.assertIsNone(module.scales.grad)
        embed = model.model.embed_tokens
        self.assertIsInstance(embed, q8.Q8Embedding)
        ids = torch.tensor([[3, 0, 258], [258, 3, 3]])
        self.assertTrue(torch.equal(embed(ids), embed.dequantize()[ids]))

    def test_lora_gradients_through_q8_base_equal_dense_dequantised_reference(self):
        """The production shape on the Q8 base: frozen int8 base, trainable LoRA on the seven projections, Q8 head
        shared with the embedding, trainer loss path (qwen3 chunked head). Loss, every LoRA gradient and the
        gradient reaching the embedded context must equal the same LoRA on a dense model holding the dequantised
        values, with the Q8 matmuls split into several ragged row chunks (head included)."""
        from peft import LoraConfig, get_peft_model
        from scripts import q8_gguf_base as q8
        q8_model, _ = self.load()
        for module in q8_model.modules():
            if isinstance(module, q8.Q8Linear):
                module.chunk_rows = 5
        lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=trainer.TARGETS, task_type='CAUSAL_LM')
        with torch.random.fork_rng():
            torch.manual_seed(5)
            q8_model = get_peft_model(q8_model, lora)
            with torch.no_grad():
                for name, p in q8_model.named_parameters():
                    if 'lora_B' in name:
                        p.add_(torch.randn_like(p))
        dense = get_peft_model(self.dense_reference(), lora)
        adapter = {n: p for n, p in q8_model.named_parameters() if 'lora_' in n}
        with torch.no_grad():
            for name, p in dense.named_parameters():
                if 'lora_' in name:
                    p.copy_(adapter[name])
        self.assertEqual({n for n, p in q8_model.named_parameters() if p.requires_grad}, set(adapter))
        self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in adapter.values()))
        items = self.items()
        self.addCleanup(setattr, trainer, 'LOGIT_CHUNK', trainer.LOGIT_CHUNK)
        trainer.LOGIT_CHUNK = 3

        def run(model):
            model.enable_input_require_grads()
            model.eval()
            context = []

            def keep_context(_module, _inputs, output):
                output.retain_grad()
                context.append(output)

            hook = model.get_input_embeddings().register_forward_hook(keep_context)
            self.addCleanup(hook.remove)
            model.zero_grad(set_to_none=True)
            loss = trainer.batch_loss(model, items, 'cpu', self.tok.pad_token_id, 'qwen3')
            loss.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            return loss.detach(), grads, context[-1].grad.clone()

        q8_loss, q8_grads, q8_context = run(q8_model)
        dense_loss, dense_grads, dense_context = run(dense)
        self.assertEqual(set(q8_grads), set(adapter))
        self.assertEqual(set(dense_grads), set(adapter))
        self.assertTrue(torch.allclose(q8_loss, dense_loss, atol=1e-5))
        for name in adapter:
            self.assertGreater(q8_grads[name].abs().sum().item(), 0, name)
            self.assertTrue(torch.allclose(q8_grads[name], dense_grads[name], atol=1e-5, rtol=1e-4), name)
        self.assertTrue(torch.allclose(q8_context, dense_context, atol=1e-5, rtol=1e-4))
        inner = q8_model.get_base_model()
        self.assertIsNone(inner.lm_head.weight.grad)
        self.assertIs(inner.lm_head.scales, inner.model.embed_tokens.scales)

    def test_malformed_metadata_is_refused_before_any_row_is_read(self):
        import gguf
        import numpy as np
        from scripts import q8_gguf_base as q8
        f32, q8t = gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q8_0
        embed = self.unique['model.embed_tokens.weight']
        q_proj = self.unique['model.layers.0.self_attn.q_proj.weight'].numpy()
        cases = {
            'architecture': (dict(arch='llama'), 'general.architecture'),
            'file-type': (dict(file_type=int(gguf.LlamaFileType.ALL_F32)), 'general.file_type'),
            'block-count': (dict(block_count=3), 'block_count'),
            'missing': (dict(drop=('model.layers.1.mlp.down_proj.weight',)), 'missing GGUF tensor blk.1.ffn_down.weight'),
            'unexpected': (dict(extra={'blk.0.extra.weight': (np.ones(4, dtype=np.float32), f32)}), 'unexpected GGUF tensors'),
            'misshaped': (dict(override={'model.norm.weight': (np.ones(3, dtype=np.float32), f32)}), 'output_norm.weight: GGUF shape \\[3\\]'),
            'not-q8': (dict(override={'model.layers.0.self_attn.q_proj.weight': (q_proj, f32)}), 'blk.0.attn_q.weight: type F32, expected Q8_0'),
            'misshaped-tied-head': (dict(extra={'output.weight': (gguf.quants.quantize(q_proj, q8t), q8t)}), 'output.weight: GGUF shape'),
        }
        never = dict(side_effect=AssertionError('tensor rows were read'))
        for name, (kwargs, message) in cases.items():
            with self.subTest(name):
                path = self.base / f'{name}.gguf'
                sha = self.write_gguf(path, self.unique, **kwargs)
                with unittest.mock.patch.object(q8, '_read_q8_rows', **never) as rows:
                    with self.assertRaisesRegex(ValueError, message):
                        self.load(path, sha)
                rows.assert_not_called()
        with unittest.mock.patch.object(q8, '_read_q8_rows', **never):
            with self.assertRaisesRegex(ValueError, 'sha256 .* != expected'):
                self.load(self.gguf, '0' * 64)
        # A tied output.weight of the right shape is shape-checked and skipped; its bytes never reach the model.
        garbage = gguf.quants.quantize(np.full(embed.shape, 3.0, dtype=np.float32), q8t)
        path = self.base / 'tied-head.gguf'
        sha = self.write_gguf(path, self.unique, extra={'output.weight': (garbage, q8t)})
        model, report = self.load(path, sha)
        self.assertEqual((report['skipped_tied'], report['bytes_skipped']), (['output.weight'], garbage.nbytes))
        self.assertTrue(torch.equal(model.lm_head.dequantize(), self.reference_state()['model.embed_tokens.weight']))

    def test_trainer_records_q8_base_in_every_metadata_file_and_refuses_merge(self):
        from peft import PeftModel
        from safetensors.torch import load_file
        out = self.base / 'run-q8'
        argv = ['train_open_weight_lora.py', '--model', str(self.model_dir), '--model-id', 'tiny/qwen3',
                '--model-revision', 'rev-tiny', '--manifest', str(self.manifest), '--device', 'cpu', '--out', str(out),
                '--batch-size', '1', '--micro-batch-size', '1', '--lr', '1e-2', '--rank', '2', '--alpha', '4']
        q8_args = ['--base-gguf', str(self.gguf), '--base-gguf-sha256', self.sha]
        for name, extra in {'merge': q8_args + ['--merge'], 'no-sha': q8_args[:2], 'generic': q8_args + ['--family', 'generic']}.items():
            with self.subTest(name), unittest.mock.patch.object(sys, 'argv', argv + extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.main()
        self.assertFalse(out.exists())
        with unittest.mock.patch.object(sys, 'argv', argv + q8_args + ['--steps', '2', '--checkpoint-every', '1']), \
                contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
        expected = dict(representation='q8_0', gguf=str(self.gguf), gguf_sha256=self.sha, block_size=32, scale_dtype='float16',
                        compute_dtype='torch.float32', quantized_tensors=15, f32_tensors=9, merge_supported=False)
        for meta in (out / 'training.json', out / 'checkpoints' / 'step-000001' / 'training.json'):
            payload = json.loads(meta.read_text())
            self.assertEqual({k: payload['base_weights'][k] for k in expected}, expected, meta)
            self.assertIn('not the ggml Q8 kernel', payload['base_weights']['note'])
        ledger = json.loads((out / 'checkpoints.json').read_text())
        self.assertEqual(ledger['base_weights']['gguf_sha256'], self.sha)
        self.assertEqual([c['step'] for c in ledger['checkpoints']], [1, 2])
        weights = load_file(str(out / 'adapter' / 'adapter_model.safetensors'))
        self.assertTrue(weights and all('lora_' in k for k in weights))  # a plain PEFT adapter: no base rows or scales
        self.assertTrue(all(v.dtype == torch.float32 for v in weights.values()))
        adapter_config = json.loads((out / 'adapter' / 'adapter_config.json').read_text())
        self.assertEqual(adapter_config['base_model_name_or_path'], str(self.model_dir))
        # The saved adapter reloads onto a fresh Q8 base and reproduces the validation loss recorded for its step.
        step1 = json.loads((out / 'checkpoints' / 'step-000001' / 'training.json').read_text())
        base, _ = self.load()
        reloaded = PeftModel.from_pretrained(base, str(out / 'checkpoints' / 'step-000001' / 'adapter'), is_trainable=False)
        valid = [trainer.encode(self.tok, r, 'qwen3', 1536)
                 for r in trainer.load_rows(json.loads(self.manifest.read_text())['valid'][0], self.base)]
        measured = trainer.validate(reloaded, valid, 'cpu', self.tok.pad_token_id, 1, 'qwen3')
        self.assertAlmostEqual(measured['color'], step1['checkpoint']['valid_loss_by_task']['color'], places=3)

    def q8_argv(self, out, *extra):
        return ['train_open_weight_lora.py', '--model', str(self.model_dir), '--model-id', 'tiny/qwen3',
                '--model-revision', 'rev-tiny', '--manifest', str(self.manifest), '--device', 'cpu', '--out', str(out),
                '--batch-size', '1', '--micro-batch-size', '1', '--lr', '1e-2', '--rank', '2', '--alpha', '4',
                '--base-gguf', str(self.gguf), '--base-gguf-sha256', self.sha, *extra]

    def run_q8(self, out, *extra):
        with unittest.mock.patch.object(sys, 'argv', self.q8_argv(out, *extra)), contextlib.redirect_stdout(io.StringIO()):
            trainer.main()

    def source_checkpoint(self):
        """A completed Q8 run with per-step checkpoints; the step-2 checkpoint adapter is the warm-start source."""
        out = self.base / 'run-q8-source'
        if not out.exists():
            self.run_q8(out, '--steps', '2', '--checkpoint-every', '1')
        return out / 'checkpoints' / 'step-000002' / 'adapter'

    @staticmethod
    def tree_hashes(folder):
        return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()}

    def test_init_adapter_loads_saved_weights_then_a_real_update_moves_only_the_lora(self):
        """load_init_adapter on the Q8 base: every trainable tensor equals the saved checkpoint tensor (the reloaded
        model reproduces the validation loss recorded for the source step), and one real AdamW step afterwards changes
        the LoRA while the frozen int8 weights and scales stay bit-identical; through main(), an lr=0 segment keeps
        the source weights exactly and an lr>0 segment moves them, with the ancestry recorded in every training.json
        and the source directory untouched."""
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        source = self.source_checkpoint()
        source_meta = json.loads((source.parent / 'training.json').read_text())
        self.assertEqual((source_meta['status'], source_meta['steps'], source_meta['cumulative_steps']), ('checkpoint', 2, 2))
        self.assertIsNone(source_meta['init_adapter'])
        saved = load_file(str(source / 'adapter_model.safetensors'))
        base, _ = self.load()
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.05, target_modules=trainer.TARGETS, task_type='CAUSAL_LM'))
        frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        self.assertEqual(trainer.load_init_adapter(model, source, 'cpu'), len(saved))
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(len(trainable), len(saved))
        for name, p in trainable.items():
            key = name.replace('.default.', '.')
            self.assertIn(key, saved, name)
            self.assertTrue(torch.equal(p.detach(), saved[key]), name)
        self.assertTrue(any(p.abs().sum().item() > 0 for n, p in trainable.items() if 'lora_B' in n))  # trained, not fresh zeros
        valid = [trainer.encode(self.tok, r, 'qwen3', 1536)
                 for r in trainer.load_rows(json.loads(self.manifest.read_text())['valid'][0], self.base)]
        measured = trainer.validate(model, valid, 'cpu', self.tok.pad_token_id, 1, 'qwen3')
        self.assertAlmostEqual(measured['color'], source_meta['checkpoint']['valid_loss_by_task']['color'], places=3)
        # One real update with a fresh optimizer: LoRA moves, the Q8 base does not.
        model.train()
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-2, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss = trainer.batch_loss(model, self.items(), 'cpu', self.tok.pad_token_id, 'qwen3')
        loss.backward()
        optimizer.step()
        self.assertEqual({n for n, p in trainable.items() if p.grad is not None}, set(trainable))
        self.assertTrue(any(not torch.equal(p.detach(), saved[n.replace('.default.', '.')]) for n, p in trainable.items()))
        for name, p in model.named_parameters():
            if not p.requires_grad:
                self.assertIsNone(p.grad, name)
                self.assertTrue(torch.equal(p.detach(), frozen_before[name]), name)
        # Through the CLI: lr=0 proves the saved tensors are what training starts from; lr>0 proves it trains on.
        source_tree = self.tree_hashes(source.parent)
        kept = self.base / 'run-q8-warm-kept'
        self.run_q8(kept, '--init-adapter', str(source), '--steps', '1', '--lr', '0')
        kept_weights = load_file(str(kept / 'adapter' / 'adapter_model.safetensors'))
        self.assertEqual(set(kept_weights), set(saved))
        self.assertTrue(all(torch.equal(kept_weights[k], saved[k]) for k in saved))
        moved = self.base / 'run-q8-warm-moved'
        self.run_q8(moved, '--init-adapter', str(source), '--steps', '2', '--checkpoint-every', '1')
        step1 = load_file(str(moved / 'checkpoints' / 'step-000001' / 'adapter' / 'adapter_model.safetensors'))
        self.assertEqual(set(step1), set(saved))
        self.assertFalse(all(torch.equal(step1[k], saved[k]) for k in saved))
        self.assertEqual(self.tree_hashes(source.parent), source_tree)  # the source checkpoint is never written
        expected = dict(path=str(source), adapter_weights_sha256=hashlib.sha256((source / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                        adapter_config_sha256=hashlib.sha256((source / 'adapter_config.json').read_bytes()).hexdigest(),
                        training_json_sha256=hashlib.sha256((source.parent / 'training.json').read_bytes()).hexdigest(),
                        source_status='checkpoint', source_steps=2, source_planned_steps=2, source_cumulative_steps=2,
                        source_manifest_id='tiny-q8', tensors_loaded=len(saved))
        for meta, steps in ((moved / 'checkpoints' / 'step-000001' / 'training.json', 1),
                            (moved / 'checkpoints' / 'step-000002' / 'training.json', 2), (moved / 'training.json', 2),
                            (kept / 'training.json', 1)):
            payload = json.loads(meta.read_text())
            self.assertEqual({k: payload['init_adapter'][k] for k in expected}, expected, meta)
            self.assertTrue(payload['init_adapter']['optimizer'].startswith('reset'), meta)
            self.assertTrue(payload['init_adapter']['scheduler'].startswith('reset'), meta)
            self.assertEqual((payload['steps'], payload['cumulative_steps']), (steps, 2 + steps), meta)
            self.assertEqual(payload['base_weights']['gguf_sha256'], self.sha)
        ledger = json.loads((moved / 'checkpoints.json').read_text())
        self.assertEqual((ledger['run_status'], ledger['planned_steps'], ledger['init_adapter']['source_steps']), ('complete', 2, 2))
        self.assertEqual([h['step'] for h in json.loads((moved / 'training.json').read_text())['history']], [1, 2])

    def test_init_adapter_compatibility_mismatch_is_refused_before_any_weight_loads(self):
        """Every refusal happens in init_adapter_ancestry on metadata alone: nothing is loaded, no output exists."""
        import shutil
        import q8_gguf_base  # the module name the trainer imports (scripts/ is on sys.path), not scripts.q8_gguf_base
        source = self.source_checkpoint()
        other_manifest = self.base / 'manifest-other.json'
        other_manifest.write_text(json.dumps(dict(json.loads(self.manifest.read_text()), id='tiny-q8-other')))
        orphan = self.base / 'orphan' / 'adapter'  # a bare adapter directory: interrupted or unattributed
        shutil.copytree(source, orphan)
        tampered = self.base / 'tampered'
        shutil.copytree(source.parent, tampered)
        config = json.loads((tampered / 'adapter' / 'adapter_config.json').read_text())
        config['modules_to_save'] = ['lm_head']
        (tampered / 'adapter' / 'adapter_config.json').write_text(json.dumps(config))
        swapped = self.base / 'swapped'
        shutil.copytree(source.parent, swapped)
        shutil.copyfile(source.parent.parent / 'step-000001' / 'adapter' / 'adapter_model.safetensors',
                        swapped / 'adapter' / 'adapter_model.safetensors')

        def edited_copy(name, edit):
            """The source checkpoint folder copied with its training.json altered by `edit(meta)`; returns the adapter dir."""
            folder = self.base / f'edited-{name}'
            shutil.copytree(source.parent, folder)
            meta = json.loads((folder / 'training.json').read_text())
            edit(meta)
            (folder / 'training.json').write_text(json.dumps(meta, ensure_ascii=False))
            return ['--init-adapter', str(folder / 'adapter')]

        def unrecorded(meta):
            del meta['checkpoint']['adapter_weights_sha256']

        def planned_as_actual(meta):
            meta['steps'] = meta['planned_steps'] + 1  # a plan or a later count is not what this adapter was trained for

        def history_overrun(meta):
            meta['history'].append(dict(meta['history'][-1], step=meta['steps'] + 1))

        def bool_steps(meta):
            meta['steps'] = True
            meta['checkpoint']['step'] = True
            meta['history'][-1]['step'] = True

        def cumulative_below(meta):
            meta['cumulative_steps'] = meta['steps'] - 1

        def unattributed_complete(meta):  # a complete run saved without --checkpoint-every: no last_step sha256
            meta['status'] = 'complete'
            del meta['checkpoint']
        init = ['--init-adapter', str(source)]
        cases = {
            'rank': (init + ['--rank', '4', '--alpha', '8'], 'training.json lora'),
            'dropout': (init + ['--dropout', '0.0'], 'training.json lora'),
            'manifest': (init + ['--manifest', str(other_manifest)], 'training.json manifest_sha256'),
            'revision': (init + ['--model-revision', 'rev-other'], 'training.json base_revision'),
            'model-id': (init + ['--model-id', 'tiny/other'], 'training.json base_model_id'),
            'gguf-sha': (init + ['--base-gguf-sha256', '0' * 64], 'training.json base_weights'),
            'orphan': (['--init-adapter', str(orphan)], 'no adjacent'),
            'not-an-adapter': (['--init-adapter', str(self.base)], 'not a saved PEFT adapter'),
            'modules-to-save': (['--init-adapter', str(tampered / 'adapter')], 'modules_to_save'),
            'swapped-weights': (['--init-adapter', str(swapped / 'adapter')], 'adapter_weights_sha256 recorded'),
            'unrecorded-sha': (edited_copy('unrecorded', unrecorded), 'records no adapter_weights_sha256'),
            'unattributed-complete': (edited_copy('complete', unattributed_complete), 'records no adapter_weights_sha256'),
            'planned-as-actual': (edited_copy('planned', planned_as_actual), 'disagrees with the recorded checkpoint step'),
            'history-overrun': (edited_copy('history', history_overrun), 'disagrees with the recorded checkpoint step'),
            'bool-steps': (edited_copy('bool', bool_steps), 'not a positive step count'),
            'cumulative-below-steps': (edited_copy('cumulative', cumulative_below), 'cumulative_steps .* is not an integer >= steps'),
        }
        never = dict(side_effect=AssertionError('adapter weights were loaded'))
        no_base = dict(side_effect=AssertionError('the Q8 base was loaded'))
        for name, (extra, message) in cases.items():
            out = self.base / f'run-q8-refused-{name}'
            with self.subTest(name), unittest.mock.patch.object(trainer, 'load_init_adapter', **never) as loader, \
                    unittest.mock.patch.object(q8_gguf_base, 'load_qwen3_q8_gguf', **no_base) as base_loader, \
                    unittest.mock.patch.object(sys, 'argv', self.q8_argv(out, *extra)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, message):
                    trainer.main()
                loader.assert_not_called()
                base_loader.assert_not_called()
                self.assertFalse(out.exists())
        # BF16/generic/--merge warm starts are refused by the CLI, never silently downgraded.
        plain = self.q8_argv(self.base / 'run-q8-refused-cli')[:-4]  # without --base-gguf/--base-gguf-sha256
        for name, argv in {'bf16': plain + init, 'generic': plain + init + ['--family', 'generic'],
                           'merge': self.q8_argv(self.base / 'run-q8-refused-cli', *init, '--merge')}.items():
            with self.subTest(name), unittest.mock.patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.main()
        self.assertFalse((self.base / 'run-q8-refused-cli').exists())


if __name__ == '__main__':
    unittest.main()
