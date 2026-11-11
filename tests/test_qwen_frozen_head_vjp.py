"""Generated-only regressions for the experimental first-order frozen Q8 answer head."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import q8_gguf_base as q8
from scripts import qwen_frozen_head_vjp as candidate
from scripts import train_qwen_preferences as dpo

sys.path.insert(0, str(Path(__file__).parent))
import test_train_qwen_preferences as preference_fixtures  # noqa: E402
from test_train_open_weight_lora import local_tokenizer  # noqa: E402

T = dpo.T


def generated_head(vocab=19, module=q8):
    with torch.random.fork_rng():
        torch.manual_seed(29)
        head = module.Q8Linear(32, vocab, torch.float32, 'cpu')
        with torch.no_grad():
            head.weight.copy_(torch.randint(-30, 31, head.weight.shape, dtype=torch.int8))
            head.scales.fill_(0.003)
    head.chunk_rows = 7  # Several row chunks and a ragged final chunk.
    return head


def generated_model():
    class Backbone(torch.nn.Embedding):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=super().forward(input_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone(24, 32)
            self.lm_head = generated_head()

    with torch.random.fork_rng():
        torch.manual_seed(31)
        return Model()


class FrozenHeadCrossEntropyTests(unittest.TestCase):
    def test_losses_and_signed_nonuniform_vjp_match_original_with_noncontiguous_inputs(self):
        head = generated_head()
        with torch.random.fork_rng():
            torch.manual_seed(37)
            source = torch.randn(2, 32, 5)
        target = (torch.arange(10).reshape(5, 2) % head.out_features).t()
        target[0, 1] = -100
        target[1, 3] = -100
        coefficients = torch.tensor([[0.7, -0.2], [-1.1, 0.0], [0.3, 1.7], [-0.8, 0.4], [1.2, -0.5]]).t()
        self.assertFalse(target.is_contiguous())
        self.assertFalse(coefficients.is_contiguous())
        expected_hidden = source.clone().transpose(1, 2).requires_grad_(True)
        actual_hidden = source.clone().transpose(1, 2).requires_grad_(True)
        self.assertFalse(actual_hidden.is_contiguous())
        expected = T.chunk_cross_entropy(expected_hidden, head, target)
        actual = candidate.frozen_head_cross_entropy(actual_hidden, head, target)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.shape, target.shape)
        expected_grad, = torch.autograd.grad(expected, expected_hidden, coefficients)
        actual_grad, = torch.autograd.grad(actual, actual_hidden, coefficients)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
        self.assertEqual(actual_grad.shape, actual_hidden.shape)
        self.assertEqual(actual_grad[target == -100].count_nonzero().item(), 0)
        self.assertEqual(actual_grad[coefficients == 0].count_nonzero().item(), 0)
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(head.scales.grad)

    def test_saves_only_hidden_derivative_and_never_reprojects_during_backward(self):
        head = generated_head(vocab=71)
        hidden = torch.ones(2, 3, 32, requires_grad=True)
        target = torch.arange(6).reshape(2, 3)
        forwards = []
        hook = head.register_forward_pre_hook(lambda _head, args: forwards.append(tuple(args[0].shape)))
        self.addCleanup(hook.remove)
        losses = candidate.frozen_head_cross_entropy(hidden, head, target)
        saved = losses.grad_fn.saved_tensors
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].shape, hidden.shape)
        self.assertEqual(saved[0].dtype, torch.float32)
        self.assertFalse(saved[0].requires_grad)
        self.assertEqual(forwards, [(2, 3, 32)])
        losses.sum().backward()
        self.assertEqual(forwards, [(2, 3, 32)])
        self.assertGreater(hidden.grad.abs().sum().item(), 0)

    def test_no_grad_inference_and_unrequired_hidden_use_original_scoring_without_vjp(self):
        head = generated_head()
        hidden = torch.ones(2, 3, 32, requires_grad=True)
        target = torch.arange(6).reshape(2, 3)
        expected = T.chunk_cross_entropy(hidden, head, target).detach()
        for mode in (torch.no_grad, torch.inference_mode):
            observed_grad_modes = []
            hook = head.register_forward_pre_hook(
                lambda _head, _args: observed_grad_modes.append(torch.is_grad_enabled()))
            try:
                with mode():
                    actual = candidate.frozen_head_cross_entropy(hidden, head, target)
                self.assertEqual(observed_grad_modes, [False])
                self.assertFalse(actual.requires_grad)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            finally:
                hook.remove()
        actual = candidate.frozen_head_cross_entropy(hidden.detach(), head, target)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_higher_order_derivatives_fail_explicitly(self):
        hidden = torch.ones(1, 2, 32, requires_grad=True)
        losses = candidate.frozen_head_cross_entropy(hidden, generated_head(), torch.tensor([[1, 2]]))
        with self.assertRaisesRegex(RuntimeError, 'higher-order'):
            torch.autograd.grad(losses.sum(), hidden, create_graph=True)

    def test_script_import_identity_is_also_an_exact_supported_head(self):
        head = generated_head(module=candidate.command_q8)
        hidden = torch.ones(1, 2, 32, requires_grad=True)
        target = torch.tensor([[1, 2]])
        actual = candidate.frozen_head_cross_entropy(hidden, head, target)
        expected = T.chunk_cross_entropy(hidden, head, target)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        self.assertGreater(hidden.grad.abs().sum().item(), 0)


class FrozenHeadBoundaryTests(unittest.TestCase):
    def test_unsupported_heads_fail_even_during_no_grad_scoring(self):
        class Subclass(q8.Q8Linear):
            pass

        heads = []
        heads.append(('exact Q8Linear', torch.nn.Linear(32, 19)))
        heads.append(('exact Q8Linear', Subclass(32, 19, torch.float32, 'cpu')))
        biased = generated_head()
        biased.bias = torch.nn.Parameter(torch.zeros(19), requires_grad=False)
        heads.append(('bias-free', biased))
        trainable = generated_head()
        trainable.scales.requires_grad_(True)
        heads.append(('frozen', trainable))
        wrong_weight = generated_head()
        wrong_weight.weight = torch.nn.Parameter(wrong_weight.weight.float(), requires_grad=False)
        heads.append(('int8 weights', wrong_weight))
        wrong_scales = generated_head()
        wrong_scales.scales = torch.nn.Parameter(wrong_scales.scales.float(), requires_grad=False)
        heads.append(('FP16 scales', wrong_scales))
        wrong_compute = generated_head()
        wrong_compute.compute_dtype = torch.bfloat16
        heads.append(('FP32', wrong_compute))
        heads.append(('CPU only', q8.Q8Linear(32, 19, torch.float32, 'meta')))
        unbounded = generated_head()
        unbounded.chunk_rows = q8.chunk_rows(32, q8.Q8_DEQUANT_CHUNK_BYTES) + 1
        heads.append(('bounded', unbounded))
        for message, head in heads:
            for grad_enabled in (False, True):
                with self.subTest(message=message, grad_enabled=grad_enabled), torch.set_grad_enabled(grad_enabled):
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        candidate.frozen_head_cross_entropy(torch.ones(1, 2, 32, requires_grad=True), head,
                                                            torch.tensor([[1, 2]]))

    def test_unsupported_hidden_and_autocast_fail_clearly(self):
        head = generated_head()
        target = torch.tensor([[1, 2]])
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, 'FP32'):
                candidate.frozen_head_cross_entropy(torch.ones(1, 2, 32, dtype=dtype), head, target)
        with self.assertRaisesRegex(ValueError, 'CPU only'):
            candidate.frozen_head_cross_entropy(torch.empty(1, 2, 32, device='meta'), head, target)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            candidate.frozen_head_cross_entropy(torch.ones(1, 2, 16), head, target)
        with torch.autocast('cpu', dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, 'autocast'):
            candidate.frozen_head_cross_entropy(torch.ones(1, 2, 32), head, target)

    def test_all_ignored_batch_does_not_bypass_frozen_head_requirement(self):
        model = generated_model()
        model.lm_head.scales.requires_grad_(True)
        ids = torch.tensor([[0, 1]])
        with self.assertRaisesRegex(ValueError, 'frozen'):
            candidate.chunked_answer_loss(model, ids, torch.ones_like(ids), torch.tensor([0]),
                                          torch.tensor([[-100]]))


class FrozenHeadPackingTests(unittest.TestCase):
    def test_packing_preserves_holes_empty_rows_ownership_capacity_and_gradients(self):
        ids = torch.arange(24).reshape(3, 8)
        mask = torch.ones_like(ids)
        keep = torch.tensor([0, 2, 3, 6, 7])
        dense = torch.arange(15).reshape(3, 5) % 19
        holes = dense.clone()
        holes[0, [0, 2, 4]] = -100
        holes[2, [1, 3]] = -100
        empty_row = holes.clone()
        empty_row[1] = -100
        coefficients = torch.tensor([[0.7, -0.2, 0.0, 0.4, -1.1],
                                     [-0.3, 0.9, 0.2, -0.5, 0.6],
                                     [1.2, -0.8, 0.1, 0.0, -0.4]])
        with patch.object(T, 'LOGIT_CHUNK', 2):
            for target in (dense, holes, empty_row):
                with self.subTest(target=target.tolist()):
                    model = generated_model()
                    baseline = T.chunked_answer_loss(model, ids, mask, keep, target)
                    (baseline * coefficients).sum().backward()
                    expected_grad = model.model.weight.grad.clone()
                    model.zero_grad(set_to_none=True)
                    observed = []
                    hook = model.lm_head.register_forward_pre_hook(
                        lambda _head, args: observed.append(args[0].detach().clone()))
                    try:
                        actual = candidate.chunked_answer_loss(model, ids, mask, keep, target)
                        forward_count = len(observed)
                        (actual * coefficients).sum().backward()
                        self.assertEqual(len(observed), forward_count)
                    finally:
                        hook.remove()
                    torch.testing.assert_close(actual, baseline, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(model.model.weight.grad, expected_grad, atol=1e-6, rtol=1e-5)
                    self.assertEqual(actual[target == -100].count_nonzero().item(), 0)
                    rows, columns = (target != -100).nonzero(as_tuple=True)
                    wanted = model.model.weight.detach()[ids[rows, keep[columns]]]
                    torch.testing.assert_close(torch.cat(observed, dim=1).squeeze(0), wanted, atol=0, rtol=0)
                    self.assertTrue(all(chunk.shape[1] <= target.shape[0] * T.LOGIT_CHUNK for chunk in observed))
                    self.assertIsNone(model.lm_head.weight.grad)
                    self.assertIsNone(model.lm_head.scales.grad)

    def test_all_ignored_preserves_exact_zero_gradients_and_sft_fallback(self):
        items = [dict(ids=[0, 1, 2, 3], labels=[-100] * 4),
                 dict(ids=[4, 5], labels=[-100] * 2)]
        model = generated_model()
        baseline = T.batch_loss(model, items, 'cpu', 0, 'qwen3')
        baseline.backward()
        expected = model.model.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        with patch.object(T, 'chunked_answer_loss', candidate.chunked_answer_loss):
            actual = T.batch_loss(model, items, 'cpu', 0, 'qwen3')
        self.assertTrue(actual.requires_grad)
        self.assertEqual(actual.item(), 0)
        actual.backward()
        self.assertIsNotNone(model.model.weight.grad)
        torch.testing.assert_close(model.model.weight.grad, expected, atol=0, rtol=0)
        self.assertEqual(model.model.weight.grad.count_nonzero().item(), 0)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNone(model.lm_head.scales.grad)


class FrozenHeadPreferenceIntegrationTests(unittest.TestCase):
    def test_generated_qwen3_q8_head_lora_combined_step_and_optimizer_agree(self):
        tok = local_tokenizer()
        fixture = preference_fixtures.CombinedStepTests()
        fixture.tok = tok
        pairs, replay = fixture.batches()
        baseline = preference_fixtures.tiny_lora_qwen3(len(tok))
        baseline.get_base_model().lm_head = generated_head(len(tok))
        experimental = copy.deepcopy(baseline)
        with torch.no_grad():
            reference = [tuple(v + shift for v, shift in zip(
                dpo.pair_logprobs(baseline, pair, 'cpu', tok.pad_token_id).tolist(), offsets))
                for pair, offsets in zip(pairs, ((0.5, 0.0), (0.0, 0.25)))]
        paired = [(f'p{i}', pair, ref) for i, (pair, ref) in enumerate(zip(pairs, reference))]
        replayed = [(f'r{i}', row) for i, row in enumerate(replay)]
        models = (baseline, experimental)
        optimizers = [torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05, momentum=0.9)
                      for model in models]
        original = T.chunked_answer_loss
        with patch.object(T, 'LOGIT_CHUNK', 3):
            for step in range(2):
                results = []
                for model, optimizer, loss_fn in zip(models, optimizers, (original, candidate.chunked_answer_loss)):
                    optimizer.zero_grad(set_to_none=True)
                    with patch.object(T, 'chunked_answer_loss', loss_fn):
                        numbers = dpo.accumulate_step(model, paired, replayed, 0.1, 0.7, 'cpu', tok.pad_token_id,
                                                      replay_micro_batch=2)
                    self.assertIs(T.chunked_answer_loss, original)
                    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
                    self.assertTrue(gradients)
                    self.assertTrue(all('lora_' in name for name in gradients))
                    self.assertTrue(any(grad.count_nonzero().item() for grad in gradients.values()))
                    self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                    optimizer.step()
                    results.append((numbers, gradients))
                for key in ('preference_loss', 'replay_ce', 'train_loss', 'reward_margin'):
                    self.assertAlmostEqual(results[0][0][key], results[1][0][key], places=5, msg=f'{step}: {key}')
                for key in ('pairs', 'replay_rows', 'replay_micro_batches'):
                    self.assertEqual(results[0][0][key], results[1][0][key])
                self.assertEqual(results[0][1].keys(), results[1][1].keys())
                for name in results[0][1]:
                    torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=1e-6, rtol=1e-5,
                                               msg=f'{step}: {name}')
                for (name, before), (_, after) in zip(baseline.named_parameters(), experimental.named_parameters()):
                    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5, msg=f'{step}: {name}')
                for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
                    torch.testing.assert_close(left['momentum_buffer'], right['momentum_buffer'], atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
