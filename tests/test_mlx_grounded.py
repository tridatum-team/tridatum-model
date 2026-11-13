"""Local-only regressions; no downloads, fake tokenizer, or mocked MLX wiring.

Small-model tests require the isolated MLX environment. Set
TRID_LOCAL_MLX_TOKENIZER_DIR to the pinned Qwen3-4B directory for official native
framing/whitespace tests. These tests do not load its model weights.
"""
import os
from pathlib import Path
import sys
import unittest

os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mlx_grounded import (NO_THINK, RECIPE, assemble_training_tokens, attach_lora,
                          collect_generation, decode_generation, encode_training,
                          packed_answer_loss, prompt_tokens)


class TokenBudgetTests(unittest.TestCase):
    def test_answer_eos_only_mask_and_exact_boundary(self):
        prompt, answer = [11, 12, 13], [21, 22, 2]
        item = assemble_training_tokens(prompt, answer, context=6, answer_budget=3)
        self.assertEqual(item['ids'], [11, 12, 13, 21, 22, 2])
        self.assertEqual(item['labels'], [-100, -100, -100, 21, 22, 2])
        self.assertEqual(item['prompt_tokens'], 3)
        self.assertEqual(item['answer_tokens'], 3)
        self.assertEqual(prompt, [11, 12, 13])
        self.assertEqual(answer, [21, 22, 2])

    def test_neither_budget_silently_truncates(self):
        with self.assertRaisesRegex(RuntimeError, 'Answer plus EOS'):
            assemble_training_tokens([3], [4, 5, 2], context=10, answer_budget=2)
        with self.assertRaisesRegex(RuntimeError, 'no truncation'):
            assemble_training_tokens([3, 4, 5], [6, 2], context=4, answer_budget=2)
        with self.assertRaisesRegex(RuntimeError, 'Empty'):
            assemble_training_tokens([], [2], context=4, answer_budget=2)


class RealQwen3ArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models.qwen3 import Model, ModelArgs
            from mlx.utils import tree_flatten
        except ImportError as exc:
            raise unittest.SkipTest(f'Isolated local MLX runtime required: {exc}')
        cls.mx, cls.nn = mx, nn
        cls.Model, cls.ModelArgs, cls.flatten = Model, ModelArgs, staticmethod(tree_flatten)
        # Tests must not raise the workstation wired limit to its recommendation.
        mx.set_cache_limit(0)
        mx.set_memory_limit(6442450944)
        mx.set_wired_limit(6442450944)

    def make_model(self, tied):
        self.mx.random.seed(17)
        args = self.ModelArgs(model_type='qwen3', hidden_size=64, num_hidden_layers=2,
                              intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=16, rms_norm_eps=1e-6, vocab_size=96,
                              max_position_embeddings=64, tie_word_embeddings=tied, rope_theta=10000.0)
        model = self.Model(args)
        self.nn.quantize(model, group_size=32, bits=4)
        recipe = dict(RECIPE, rank=2, num_layers=1)
        attach_lora(model, recipe)
        # Exercise gradients through both adapter factors, not only initially-zero B.
        weights = [(name, self.mx.random.normal(value.shape) * 0.01)
                   for name, value in self.flatten(model.trainable_parameters()) if name.endswith('lora_b')]
        model.load_weights(weights, strict=False)
        self.mx.eval(model.parameters())
        return model

    def test_packed_head_loss_and_all_lora_gradients_match_full_mask(self):
        mx, nn = self.mx, self.nn
        ids = mx.array([3, 7, 11, 9, 5, 17, 21, 2], dtype=mx.int32)
        prompt_length = 4

        def full_mask_loss(model, sequence):
            logits = model(sequence[None, :-1]).astype(mx.float32)
            losses = nn.losses.cross_entropy(logits, sequence[None, 1:], reduction='none')
            mask = mx.arange(sequence.shape[0] - 1) >= prompt_length - 1
            return mx.sum(losses * mask[None]) / mx.sum(mask)

        for tied in (True, False):
            with self.subTest(tied_embedding=tied):
                model = self.make_model(tied)
                full_loss, full_grad = nn.value_and_grad(model, full_mask_loss)(model, ids)
                packed_loss, packed_grad = nn.value_and_grad(
                    model, lambda m, x: packed_answer_loss(m, x, prompt_length, chunk_size=3))(model, ids)
                mx.eval(full_loss, packed_loss, full_grad, packed_grad)
                self.assertAlmostEqual(full_loss.item(), packed_loss.item(), places=5)
                full, packed = dict(self.flatten(full_grad)), dict(self.flatten(packed_grad))
                self.assertEqual(set(full), set(packed))
                self.assertTrue(full)
                for key in full:
                    self.assertTrue(bool(mx.allclose(full[key], packed[key], rtol=2e-4, atol=2e-5).item()), key)
                self.assertTrue(any(float(mx.max(mx.abs(value)).item()) > 0 for value in packed.values()))

    def test_real_generate_step_preserves_eos_and_exact_token_cap(self):
        # Genuine native Qwen3 forward + quantized backbone. A zero untied head
        # creates a deterministic argmax tie at token 0, without a fake generator.
        model = self.make_model(False)
        model.lm_head = self.nn.Linear(64, 96, bias=False)
        model.lm_head.weight = self.mx.zeros((96, 64))
        model.eval()
        prompt = [3, 7, 11, 9, 5]
        events, eos_output, length_output = [], [], []
        collect_generation(model, prompt, {0}, 3, lambda event, **fields: events.append((event, fields)), eos_output)
        self.assertEqual(eos_output, [0])
        self.assertTrue(any(event == 'prefill' and fields['n'] == 0 for event, fields in events))
        self.assertTrue(any(event == 'prefill' and fields['n'] == len(prompt) for event, fields in events))
        collect_generation(model, prompt, {95}, 3, lambda event, **fields: None, length_output)
        self.assertEqual(length_output, [0, 0, 0])


class OfficialTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('TRID_LOCAL_MLX_TOKENIZER_DIR')
        if not path:
            raise unittest.SkipTest('Set TRID_LOCAL_MLX_TOKENIZER_DIR to the pinned local Qwen3-4B directory')
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def test_native_template_separate_answer_encoding_and_eos_mask(self):
        row = dict(id='synthetic', task='synthetic', split='train', answer=' 답변입니다.\n',
                   messages=[dict(role='system', content='근거만 사용하세요.\n'),
                             dict(role='user', content='  질문입니다.\n')])
        text, prompt = prompt_tokens(self.tokenizer, row['messages'])
        self.assertTrue(text.endswith(NO_THINK))
        self.assertIn(row['messages'][0]['content'], text)
        self.assertIn(row['messages'][1]['content'], text)
        item = encode_training(self.tokenizer, row, RECIPE)
        answer = self.tokenizer.encode(row['answer'] + self.tokenizer.eos_token, add_special_tokens=False)
        self.assertEqual(item['ids'], prompt + answer)
        self.assertEqual(item['labels'], [-100] * len(prompt) + answer)
        self.assertEqual(item['labels'][-1], self.tokenizer.eos_token_id)
        with self.assertRaisesRegex(RuntimeError, 'no truncation'):
            encode_training(self.tokenizer, row, dict(RECIPE, train_context=len(item['ids']) - 1))
        with self.assertRaisesRegex(RuntimeError, 'Answer plus EOS'):
            encode_training(self.tokenizer, row, dict(RECIPE, answer_tokens=len(answer) - 1))

    def test_terminal_eos_only_is_removed_even_at_cap(self):
        tokenizer = self.tokenizer
        body = ' \n```json\n<|im_start|><think>내부</think>\n{"x":1}\n```\t '
        ids = tokenizer.encode(body, add_special_tokens=False) + [tokenizer.eos_token_id]
        decoded = decode_generation(tokenizer, ids, [tokenizer.eos_token_id], len(ids))
        self.assertEqual(decoded['response_text'], body)
        self.assertEqual(decoded['raw_output'], body + tokenizer.eos_token)
        self.assertEqual(decoded['termination_reason'], 'eos')
        self.assertEqual(decoded['generated_tokens'], len(ids))
        self.assertEqual(decoded['terminal_stop_token_id'], tokenizer.eos_token_id)

    def test_length_and_invalid_early_termination_never_repair(self):
        tokenizer = self.tokenizer
        text = ' \n{"x":"끝"}\t '
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = decode_generation(tokenizer, ids, [tokenizer.eos_token_id], len(ids))
        self.assertEqual(decoded['response_text'], text)
        self.assertEqual(decoded['raw_output'], text)
        self.assertEqual(decoded['termination_reason'], 'length')
        self.assertIsNone(decoded['terminal_stop_token_id'])
        with self.assertRaisesRegex(RuntimeError, 'Unexplained'):
            decode_generation(tokenizer, ids, [tokenizer.eos_token_id], len(ids) + 1)
        with self.assertRaisesRegex(RuntimeError, 'envelope'):
            decode_generation(tokenizer, [], [tokenizer.eos_token_id], 1)
        with self.assertRaisesRegex(RuntimeError, 'envelope'):
            decode_generation(tokenizer, ids, [tokenizer.eos_token_id], len(ids) - 1)


if __name__ == '__main__':
    unittest.main()
