"""Local-only regressions; no downloads, fake tokenizer, or mocked MLX wiring.

Small-model tests require the isolated MLX environment. Set
TRID_LOCAL_MLX_TOKENIZER_DIR to the pinned Qwen3-4B directory for official native
framing/whitespace tests. These tests do not load its model weights.
"""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mlx_grounded import (LIMITS, MODEL_REPO, MODEL_REVISION, NO_THINK, RECIPE, Runtime,
                          admit_training_rows, assemble_training_tokens, attach_lora, audit_inputs,
                          canonical, collect_generation, decode_generation, encode_training,
                          packed_answer_loss, prompt_tokens, run_training, training_coverage,
                          training_schedule, validate_plan, validate_training_declaration)


def mixed_rows():
    return [dict(id=f'train-{i}', task=task, split='train', answer=f'답변 {i}.\n',
                 messages=[dict(role='system', content='질문에 답하세요.\n'),
                           dict(role='user', content=f'질문 {i}.\n')])
            for i, task in enumerate(('mois_grounded_explanation', 'mois_grounded_explanation',
                                      'general_instruction', 'general_instruction'))]


def training_declaration():
    return dict(path='/unused/train.jsonl', sha256='0' * 64, rows=4,
                cohorts=dict(grounded_messages=2, general_instruction=2))


class TrainingAdmissionTests(unittest.TestCase):
    def test_explicit_one_and_two_epochs_and_maximum_population(self):
        for rows, epochs in ((4, 1), (4, 2), (192, 2)):
            declaration = training_declaration()
            declaration.update(rows=rows, cohorts=dict(grounded_messages=rows // 2,
                                                       general_instruction=rows // 2))
            validate_training_declaration(declaration, dict(RECIPE, epochs=epochs, steps=rows * epochs))

    def test_invalid_epochs_steps_recipe_keys_and_invariants_fail_plan(self):
        valid = dict(RECIPE, epochs=2, steps=8)
        recipes = [dict(valid, epochs=value) for value in (0, 3, True, 1.0, None)]
        recipes += [dict(valid, steps=value) for value in (4, 7, 9, 385, True, 8.0)]
        recipes += [dict(valid, seed=15), dict(valid, rank=16), dict(valid, retry=True)]
        for key in ('epochs', 'steps', 'seed'):
            candidate = dict(valid)
            del candidate[key]
            recipes.append(candidate)
        for recipe in recipes:
            plan = dict(schema='trid-local-mlx-v2', run_id='admission',
                        model=dict(repo_id=MODEL_REPO, revision=MODEL_REVISION),
                        train=training_declaration(), recipe=recipe, limits=LIMITS)
            with self.subTest(recipe=recipe), self.assertRaises(RuntimeError):
                validate_plan(plan)

    def test_declaration_rejects_wrong_counts_missing_unknown_or_empty_cohorts(self):
        for updates in (dict(rows=0), dict(rows=193), dict(rows=True), dict(rows=4.0),
                        dict(cohorts={'grounded_messages': 4}),
                        dict(cohorts={'grounded_messages': 4, 'general_instruction': 0}),
                        dict(cohorts={'grounded_messages': 2, 'general_instruction': 1}),
                        dict(cohorts={'grounded_messages': 2, 'general_instruction': 2, 'unknown': 1}),
                        dict(cohorts={'grounded_messages': True, 'general_instruction': 3})):
            with self.subTest(updates=updates), self.assertRaises(RuntimeError):
                validate_training_declaration(dict(training_declaration(), **updates),
                                              dict(RECIPE, epochs=2, steps=8))
        for key in ('rows', 'cohorts'):
            declaration = training_declaration()
            del declaration[key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                validate_training_declaration(declaration, dict(RECIPE, epochs=2, steps=8))

    def test_mixed_rows_are_all_admitted_without_mutation(self):
        rows = mixed_rows()
        before = copy.deepcopy(rows)
        admit_training_rows(rows, training_declaration())
        self.assertEqual(rows, before)

    def test_duplicate_missing_extra_rows_and_wrong_task_mix_fail(self):
        rows = mixed_rows()
        candidates = [rows[:-1], rows + [rows[0]], [rows[0], rows[0], *rows[2:]],
                      [*rows[:3], dict(rows[3], task='mois_grounded_explanation')]]
        for candidate in candidates:
            with self.subTest(rows=candidate), self.assertRaises(RuntimeError):
                admit_training_rows(candidate, training_declaration())

    def test_unknown_tasks_cohorts_unsupervised_or_malformed_rows_fail(self):
        mutations = [dict(task='unknown'), dict(cohort='unknown'),
                     dict(cohort='general_instruction'), dict(answer=' \n'),
                     dict(split='valid'), dict(history=[]), dict(id=''),
                     dict(messages=[dict(role='user', content='question')]),
                     dict(messages=[dict(role='system', content='system'),
                                    dict(role='assistant', content='answer')])]
        for update in mutations:
            rows = mixed_rows()
            rows[0].update(update)
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                admit_training_rows(rows, training_declaration())
        for field in ('id', 'task', 'answer', 'messages', 'split'):
            rows = mixed_rows()
            del rows[0][field]
            with self.subTest(missing=field), self.assertRaises(RuntimeError):
                admit_training_rows(rows, training_declaration())

    def test_deterministic_complete_epoch_schedule_and_exposure_receipt(self):
        rows = mixed_rows()
        original = copy.deepcopy(rows)
        for epochs in (1, 2):
            recipe = dict(RECIPE, epochs=epochs, steps=len(rows) * epochs)
            schedule = list(training_schedule(rows, recipe))
            self.assertEqual(schedule, list(training_schedule(rows, recipe)))
            self.assertEqual(rows, original)
            history = [dict(id=row['id'], task=row['task'], epoch=epoch, global_step=step,
                            supervised_tokens=2)
                       for step, (epoch, row) in enumerate(schedule, 1)]
            coverage = training_coverage(history, rows, training_declaration(), recipe)
            self.assertEqual(coverage['unique_rows'], 4)
            self.assertEqual(coverage['exposures'], 4 * epochs)
            self.assertEqual(coverage['row_exposures'], {row['id']: epochs for row in rows})
            self.assertEqual(coverage['cohort_exposures'],
                             dict(grounded_messages=2 * epochs, general_instruction=2 * epochs))
            for epoch in range(1, epochs + 1):
                self.assertEqual(Counter(row['id'] for e, row in schedule if e == epoch),
                                 Counter(row['id'] for row in rows))
        broken = copy.deepcopy(history)
        broken[0]['id'] = broken[1]['id']
        for candidate in (history[:-1], broken):
            with self.assertRaises(RuntimeError):
                training_coverage(candidate, rows, training_declaration(), recipe)
        for key, value in (('epoch', 1), ('task', 'unknown'), ('global_step', 0), ('supervised_tokens', 0)):
            broken = copy.deepcopy(history)
            broken[-1][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                training_coverage(broken, rows, training_declaration(), recipe)


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

    def test_real_two_epoch_updates_include_both_cohorts_and_save_final_only(self):
        from mlx_lm.tuner.utils import remove_lora_layers
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(False)
            remove_lora_layers(model)
            recipe = dict(RECIPE, rank=2, num_layers=1, epochs=2, steps=8)
            plan = dict(run_id='tiny-mixed-epochs', output_dir=directory, limits=LIMITS,
                        train=training_declaration(), recipe=recipe)
            runtime = Runtime(plan, 'train', '0' * 64)
            runtime.mx = self.mx
            items = [dict(assemble_training_tokens([3, 7, 11], [9 + i, 2], 16, 4),
                          id=row['id'], task=row['task']) for i, row in enumerate(mixed_rows())]
            run_training(runtime, model, items, capacity=False)
            receipt = runtime.receipt
            self.assertEqual(receipt['epochs'], 2)
            self.assertEqual(receipt['unique_rows'], 4)
            self.assertEqual(receipt['updates'], 8)
            self.assertEqual(receipt['exposures'], 8)
            self.assertEqual(receipt['cohort_exposures'], dict(grounded_messages=4, general_instruction=4))
            self.assertEqual([row['epoch'] for row in receipt['history']], [1] * 4 + [2] * 4)
            self.assertEqual([row['global_step'] for row in receipt['history']], list(range(1, 9)))
            self.assertEqual(receipt['row_exposures'], {row['id']: 2 for row in items})
            self.assertTrue(all(row['supervised_tokens'] == 2 for row in receipt['history']))
            self.assertTrue(any(value > 0 for value in receipt['changed_trainable_max_abs'].values()))
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {'adapter'})
            self.assertEqual({Path(pin['path']).name for pin in receipt['artifacts']},
                             {'adapter_config.json', 'adapters.safetensors'})

    def test_real_capacity_uses_longest_general_row_then_discards_adapter(self):
        from mlx_lm.tuner.utils import remove_lora_layers
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(False)
            remove_lora_layers(model)
            recipe = dict(RECIPE, rank=2, num_layers=1, epochs=2, steps=8)
            plan = dict(run_id='tiny-capacity', output_dir=directory, limits=LIMITS,
                        train=training_declaration(), recipe=recipe)
            runtime = Runtime(plan, 'capacity', '0' * 64)
            runtime.mx = self.mx
            items = [dict(assemble_training_tokens([3] * (i + 2), [9, 2], 16, 4),
                          id=row['id'], task=row['task']) for i, row in enumerate(mixed_rows())]
            run_training(runtime, model, items, capacity=True)
            receipt = runtime.receipt
            self.assertEqual(receipt['capacity_row_id'], items[-1]['id'])
            self.assertEqual(receipt['capacity_row_tokens'], len(items[-1]['ids']))
            self.assertEqual(len(receipt['history']), 1)
            self.assertEqual(receipt['history'][0]['task'], 'general_instruction')
            self.assertTrue(receipt['capacity_adapter_discarded'])
            self.assertTrue(receipt['reload']['exact_saved_tensor_parity'])
            self.assertEqual(list(Path(directory).iterdir()), [])


class OfficialTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('TRID_LOCAL_MLX_TOKENIZER_DIR')
        if not path:
            raise unittest.SkipTest('Set TRID_LOCAL_MLX_TOKENIZER_DIR to the pinned local Qwen3-4B directory')
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def test_native_template_separate_answer_encoding_and_eos_mask(self):
        row = dict(id='synthetic', task='general_instruction', split='train', answer=' 답변입니다.\n',
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

    def test_mixed_audit_preserves_all_answers_and_blind_hash_contract(self):
        rows = mixed_rows()
        blind = [dict(id=f'blind-{i}', messages=[
            dict(role='system', content='Answer the question.'),
            dict(role='user', content=f'What is {i} plus one?')]) for i in range(18)]
        pins = [dict(id=row['id'], kind='grounded' if i < 12 else 'synthetic-general-retention',
                     prompt_sha256=hashlib.sha256(canonical(row['messages'])).hexdigest())
                for i, row in enumerate(blind)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, blind_path = root / 'train.jsonl', root / 'blind.jsonl'
            train_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            blind_path.write_text(''.join(json.dumps(row) + '\n' for row in blind))
            declaration = dict(training_declaration(), path=str(train_path))
            plan = dict(train=declaration, recipe=dict(RECIPE, epochs=2, steps=8),
                        blind_inputs=dict(path=str(blind_path)), blind_cases=pins)
            encoded, audited = audit_inputs(plan, self.tokenizer)
            self.assertEqual(len(encoded), 4)
            self.assertEqual(len(audited), 18)
            for row, item in zip(rows, encoded):
                answer = self.tokenizer.encode(row['answer'] + self.tokenizer.eos_token,
                                               add_special_tokens=False)
                self.assertEqual(item['labels'][item['prompt_tokens']:], answer)
                self.assertEqual(item['task'], row['task'])
            for mutation in ('overlap', 'hash', 'blind-count', 'blind-mix', 'gold'):
                candidate = copy.deepcopy(plan)
                changed_blind = copy.deepcopy(blind)
                if mutation == 'overlap':
                    changed_blind[0]['id'] = rows[0]['id']
                    candidate['blind_cases'][0]['id'] = rows[0]['id']
                elif mutation == 'hash':
                    changed_blind[0]['messages'][1]['content'] += ' changed'
                elif mutation == 'blind-count':
                    changed_blind.pop()
                elif mutation == 'blind-mix':
                    candidate['blind_cases'][0]['kind'] = 'synthetic-general-retention'
                else:
                    changed_blind[0]['answer'] = 'not allowed'
                blind_path.write_text(''.join(json.dumps(row) + '\n' for row in changed_blind))
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                    audit_inputs(candidate, self.tokenizer)


if __name__ == '__main__':
    unittest.main()
