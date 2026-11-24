"""Offline shared inference contracts using real tokenizers and tiny native MLX arrays.

No model downloads or large weight loads. Optional official Qwen3 tokenizer cases
use only TRID_LOCAL_MLX_TOKENIZER_DIR; array cases need the isolated MLX runtime.
"""
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
import mlx_inference as inference


class ProfileTests(unittest.TestCase):
    def test_exact_historical_defaults_and_independent_profiles(self):
        expected = dict(name='historical-greedy-no-thinking', thinking=False,
                        temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                        presence_penalty=0.0, repetition_penalty=1.0,
                        penalty_context_size=0, seed=14, max_new_tokens=768,
                        context_tokens=3072, prefill_step_size=128)
        profile = inference.historical_profile()
        self.assertEqual(profile, expected)
        inference.validate_profile(profile)
        self.assertEqual(inference.historical_profile(max_new_tokens=12, context_tokens=512),
                         dict(expected, max_new_tokens=12, context_tokens=512))
        profile['temperature'] = 0.9
        self.assertEqual(inference.historical_profile(), expected)

    def test_every_field_is_required_and_unknown_fields_are_rejected(self):
        profile = inference.historical_profile()
        for field in profile:
            candidate = dict(profile)
            del candidate[field]
            with self.subTest(missing=field), self.assertRaises(RuntimeError):
                inference.validate_profile(candidate)
        for candidate in (None, [], 'greedy', dict(profile, retry=True)):
            with self.subTest(candidate=candidate), self.assertRaises(RuntimeError):
                inference.validate_profile(candidate)

    def test_numeric_types_ranges_and_finiteness_are_strict(self):
        profile = inference.historical_profile()
        invalid = {
            'name': ['', '   ', None, 1, True],
            'thinking': [0, 1, 'false', None],
            'temperature': [-0.01, '0'],
            'top_p': [0, -0.1, 1.01, '1'],
            'top_k': [-1, 1.5, 1.0, '1'],
            'min_p': [-0.01, 1.01, '0'],
            'presence_penalty': [-2.01, 2.01, '0'],
            'repetition_penalty': [0, -1, '1'],
            'penalty_context_size': [-1, 20.0, '20'],
            'seed': [-1, 2 ** 32, 14.0, '14'],
            'max_new_tokens': [0, -1, 1.0, '1'],
            'context_tokens': [0, -1, 3072.0, '3072'],
            'prefill_step_size': [0, -1, 128.0, '128', 3073],
        }
        for field, values in invalid.items():
            if field not in ('name', 'thinking'):
                values = values + [True, False, None, float('nan'), float('inf'), -float('inf')]
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    inference.validate_profile(dict(profile, **{field: value}))

    def test_valid_boundary_values_do_not_mutate_profile(self):
        profile = dict(inference.historical_profile(), seed=2 ** 32 - 1,
                       top_k=1, min_p=1, top_p=0.001, temperature=0,
                       presence_penalty=-2, repetition_penalty=0.001,
                       max_new_tokens=1, context_tokens=2, prefill_step_size=1,
                       penalty_context_size=0, thinking=True)
        original = copy.deepcopy(profile)
        inference.validate_profile(profile)
        self.assertEqual(profile, original)
        inference.validate_profile(dict(profile, presence_penalty=2, min_p=0, seed=0,
                                        penalty_context_size=64))

    def test_generation_budget_must_leave_room_for_a_nonempty_prompt(self):
        for maximum in (3072, 3073):
            with self.subTest(maximum=maximum), self.assertRaises(RuntimeError):
                inference.validate_profile(dict(inference.historical_profile(), max_new_tokens=maximum))

    def test_missing_local_model_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (root / 'missing-model', root):
                with self.subTest(path=path), self.assertRaises(RuntimeError):
                    inference.load_local(path)
            not_directory = root / 'not-a-directory'
            not_directory.write_text('{}', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                inference.load_local(not_directory)


def byte_tokenizer(closing_special=False):
    """A genuine HF fast tokenizer with lossless byte-level whitespace handling."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate(['[UNK]', '[EOS]', *alphabet])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]')
    if closing_special:
        tokenizer.add_special_tokens({'additional_special_tokens': ['</think>']})
    else:
        tokenizer.add_tokens(['</think>'], special_tokens=False)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    return tokenizer


class NativeTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = byte_tokenizer()
        except ImportError as exc:
            raise unittest.SkipTest(f'Local transformers/tokenizers runtime required: {exc}')

    def test_instruct_only_native_template_needs_no_qwen_suffix(self):
        messages = [dict(role='system', content='Keep whitespace.\n'),
                    dict(role='user', content='  question\t ')]
        original = copy.deepcopy(messages)
        profile = inference.historical_profile(max_new_tokens=4)
        text, ids = inference.render_prompt(self.tokenizer, messages, profile)
        expected = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(text, expected)
        self.assertTrue(text.endswith('assistant: '))
        self.assertNotIn('<think>', text)
        self.assertEqual(ids, self.tokenizer.encode(expected, add_special_tokens=False))
        self.assertEqual(messages, original)
        with self.assertRaisesRegex(RuntimeError, 'did not open thinking'):
            inference.render_prompt(self.tokenizer, messages, dict(profile, thinking=True))

    def test_thinking_native_template_must_leave_verified_reasoning_open(self):
        tokenizer = byte_tokenizer(closing_special=True)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<think>', '</think>']})
        messages = [dict(role='user', content='Question?')]
        tokenizer.chat_template += "{{ '<think>\\n' if add_generation_prompt and enable_thinking else '' }}"
        profile = dict(inference.historical_profile(max_new_tokens=4), thinking=True)
        text, ids = inference.render_prompt(tokenizer, messages, profile)
        self.assertEqual(text, tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        self.assertEqual(ids, tokenizer.encode(text, add_special_tokens=False))
        self.assertTrue(text.endswith('<think>\n'))
        tokenizer.chat_template += "{{ '</think>' if add_generation_prompt else '' }}"
        with self.assertRaisesRegex(RuntimeError, 'did not leave thinking open'):
            inference.render_prompt(tokenizer, messages, profile)

    def test_prompt_budget_exact_boundary_and_overflow_no_truncation(self):
        messages = [dict(role='system', content='system'), dict(role='user', content='question')]
        profile = inference.historical_profile(max_new_tokens=4)
        text, ids = inference.render_prompt(self.tokenizer, messages, profile)
        exact = dict(profile, context_tokens=len(ids) + 4, prefill_step_size=1)
        self.assertEqual(inference.render_prompt(self.tokenizer, messages, exact), (text, ids))
        with self.assertRaisesRegex(RuntimeError, 'no truncation'):
            inference.render_prompt(self.tokenizer, messages, dict(exact, context_tokens=len(ids) + 3))

    def test_raw_whitespace_and_only_terminal_eos_are_preserved(self):
        tokenizer = self.tokenizer
        body = ' \n```json\n[UNK]</think>\n{"x":1}\n```\t '
        ids = tokenizer.encode(body, add_special_tokens=False) + [tokenizer.eos_token_id]
        before = list(ids)
        decoded = inference.decode_generation(tokenizer, ids, {tokenizer.eos_token_id}, len(ids))
        self.assertEqual(ids, before)
        self.assertEqual(decoded['raw_output'], body + tokenizer.eos_token)
        self.assertEqual(decoded['response_text'], body)
        self.assertEqual(decoded['termination_reason'], 'eos')
        self.assertEqual(decoded['terminal_stop_token_id'], tokenizer.eos_token_id)
        self.assertEqual(decoded['generated_tokens'], len(ids))
        self.assertEqual(decoded['framing_status'], 'complete')
        self.assertTrue(decoded['reasoning_complete'])

    def test_ordinary_closing_token_cannot_release_thinking_response(self):
        tokenizer = self.tokenizer
        closing = tokenizer.convert_tokens_to_ids('</think>')
        self.assertNotIn(closing, tokenizer.all_special_ids)
        self.assertEqual(tokenizer.encode('</think>', add_special_tokens=False), [closing])
        body = 'private</think> public\n'
        ids = tokenizer.encode(body, add_special_tokens=False) + [tokenizer.eos_token_id]
        with self.assertRaisesRegex(RuntimeError, 'verified native thinking closing token'):
            inference.decode_generation(tokenizer, ids, {tokenizer.eos_token_id}, len(ids), thinking=True)

    def test_verified_native_closing_token_releases_only_exact_suffix(self):
        tokenizer = byte_tokenizer(closing_special=True)
        closing = tokenizer.convert_tokens_to_ids('</think>')
        self.assertIn(closing, tokenizer.all_special_ids)
        prefix, suffix = ' private\n', ' \nanswer</think>\t '
        ids = (tokenizer.encode(prefix, add_special_tokens=False) + [closing]
               + tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id])
        decoded = inference.decode_generation(tokenizer, ids, {tokenizer.eos_token_id}, len(ids), thinking=True)
        self.assertEqual(decoded['response_text'], suffix)
        self.assertEqual(decoded['raw_output'], prefix + '</think>' + suffix + tokenizer.eos_token)
        self.assertEqual(decoded['framing_status'], 'complete')
        self.assertTrue(decoded['reasoning_complete'])

    def test_native_added_markers_need_not_be_flagged_special(self):
        tokenizer = byte_tokenizer()
        tokenizer.add_tokens(['<think>'], special_tokens=False)
        opening = tokenizer.convert_tokens_to_ids('<think>')
        closing = tokenizer.convert_tokens_to_ids('</think>')
        self.assertNotIn(opening, tokenizer.all_special_ids)
        self.assertNotIn(closing, tokenizer.all_special_ids)
        original_template = tokenizer.chat_template
        messages = [dict(role='user', content='Question?')]
        profile = dict(inference.historical_profile(max_new_tokens=8), thinking=True)
        for preopen in (False, True):
            tokenizer.chat_template = original_template + (
                "{% if add_generation_prompt %}{% if enable_thinking %}"
                + ("{{ '<think>\\n' }}" if preopen else '')
                + "{% else %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
            )
            with self.subTest(preopen=preopen):
                text, ids = inference.render_prompt(tokenizer, messages, profile)
                self.assertEqual(opening in ids, preopen)
                self.assertNotIn(closing, ids)
                self.assertEqual(text, tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True))
                answer = '\n exact answer\t '
                generated = tokenizer.encode('reason</think>' + answer, add_special_tokens=False)
                decoded = inference.decode_generation(
                    tokenizer, generated, {tokenizer.eos_token_id}, len(generated), thinking=True)
                self.assertEqual(decoded['response_text'], answer)
                self.assertEqual(decoded['raw_output'], 'reason</think>' + answer)
                self.assertTrue(decoded['reasoning_complete'])

    def test_invalid_generated_envelopes_are_not_repaired(self):
        tokenizer = self.tokenizer
        ids = tokenizer.encode(' \nanswer\t ', add_special_tokens=False)
        decoded = inference.decode_generation(tokenizer, ids, {tokenizer.eos_token_id}, len(ids))
        self.assertEqual(decoded['response_text'], ' \nanswer\t ')
        self.assertEqual(decoded['raw_output'], decoded['response_text'])
        self.assertEqual(decoded['termination_reason'], 'length')
        self.assertIsNone(decoded['terminal_stop_token_id'])
        for candidate, maximum in (([], 1), (ids, len(ids) - 1), (ids, len(ids) + 1),
                                   ([True], 1), ([-1], 1), ([1.0], 1),
                                   ([tokenizer.eos_token_id, 3], 2)):
            with self.subTest(ids=candidate, maximum=maximum), self.assertRaises(RuntimeError):
                inference.decode_generation(tokenizer, candidate, {tokenizer.eos_token_id}, maximum)


class TinyNativeQwen3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx.utils import tree_flatten
            from mlx_lm.models.qwen3 import Model, ModelArgs
        except ImportError as exc:
            raise unittest.SkipTest(f'Isolated local MLX runtime required: {exc}')
        cls.mx, cls.nn = mx, nn
        cls.Model, cls.ModelArgs = Model, ModelArgs
        cls.flatten = staticmethod(tree_flatten)
        mx.set_cache_limit(0)
        mx.set_memory_limit(6442450944)
        mx.set_wired_limit(6442450944)

    def make_model(self, lora=False):
        self.mx.random.seed(17)
        args = self.ModelArgs(model_type='qwen3', hidden_size=64, num_hidden_layers=2,
                              intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=16, rms_norm_eps=1e-6, vocab_size=96,
                              max_position_embeddings=64, tie_word_embeddings=False, rope_theta=10000.0)
        model = self.Model(args)
        self.nn.quantize(model, group_size=32, bits=4)
        if lora:
            from mlx_grounded import RECIPE, attach_lora
            attach_lora(model, dict(RECIPE, rank=2, num_layers=1))
        self.mx.eval(model.parameters())
        model.eval()
        return model

    def profile(self, **updates):
        profile = dict(inference.historical_profile(max_new_tokens=24),
                       context_tokens=64, prefill_step_size=8)
        profile.update(updates)
        return profile

    def collect(self, model, prompt, profile, stops=None):
        output = []
        inference.collect_generation(model, prompt, {96} if stops is None else stops,
                                     profile, lambda event, **fields: None, output)
        return output

    def test_seed_is_reset_and_prompt_cache_is_fresh_for_each_generation(self):
        model = self.make_model()
        profile = self.profile(temperature=0.8, top_p=0.95, top_k=32, min_p=0.05,
                               presence_penalty=0.4, repetition_penalty=1.1)
        before = copy.deepcopy(profile)
        prompt = [3, 7, 11, 9, 5]
        first = self.collect(model, prompt, profile)
        self.mx.random.seed(987)
        self.mx.eval(self.mx.random.normal((128,)))
        self.collect(model, [8, 6, 4], dict(profile, seed=99))
        self.assertEqual(first, self.collect(model, prompt, profile))
        self.assertNotEqual(first, self.collect(model, prompt, dict(profile, seed=15)))
        self.assertEqual(len(first), profile['max_new_tokens'])
        self.assertEqual(profile, before)

    def test_native_terminal_eos_is_included_even_when_it_is_the_cap(self):
        model = self.make_model()
        model.lm_head = self.nn.Linear(64, 96, bias=False)
        model.lm_head.weight = self.mx.zeros((96, 64))
        self.assertEqual(self.collect(model, [3, 7, 11], self.profile(max_new_tokens=1), {0}), [0])
        self.assertEqual(self.collect(model, [3, 7, 11], self.profile(max_new_tokens=3), {0}), [0])
        self.assertEqual(self.collect(model, [3, 7, 11], self.profile(max_new_tokens=3), {95}), [0, 0, 0])

    def test_profile_and_actual_model_context_budgets_reject_before_tokens(self):
        model = self.make_model()
        prompt = [3, 7, 11, 9, 5]
        for profile in (self.profile(max_new_tokens=60),
                        self.profile(max_new_tokens=3, context_tokens=7, prefill_step_size=1),
                        self.profile(max_new_tokens=60, context_tokens=128)):
            output, events = [], []
            with self.subTest(profile=profile), self.assertRaises(RuntimeError):
                inference.collect_generation(model, prompt, {96}, profile,
                                             lambda event, **fields: events.append(event), output)
            self.assertEqual(output, [])
            self.assertNotIn('token', events)
        exact = self.profile(max_new_tokens=3, context_tokens=8)
        self.assertEqual(len(self.collect(model, prompt, exact)), 3)

    def test_exact_actual_lora_tensor_coverage_and_shapes(self):
        model = self.make_model(lora=True)
        weights = dict(self.flatten(model.trainable_parameters()))
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(name.endswith(('lora_a', 'lora_b')) for name in weights))
        inference.validate_adapter_tensors(model, weights)
        first = next(iter(weights))
        missing = dict(weights)
        del missing[first]
        extra = dict(weights, unexpected=self.mx.zeros((2, 2)))
        wrong_shape = dict(weights)
        shape = weights[first].shape
        wrong_shape[first] = self.mx.zeros((shape[0] + 1, *shape[1:]))
        for candidate in ({}, missing, extra, wrong_shape):
            with self.subTest(keys=tuple(candidate)), self.assertRaises(RuntimeError):
                inference.validate_adapter_tensors(model, candidate)

    def test_native_penalties_exclude_prompt_and_honor_full_generated_history(self):
        mx = self.mx
        prompt = [7, 7, 7]
        generated = [3] + [5] * 24
        tokens = mx.array(prompt + generated, dtype=mx.int32)
        for field, value, expected in (('presence_penalty', 0.75, 3.25),
                                       ('repetition_penalty', 2.0, 2.0)):
            for context in (0, 20):
                profile = self.profile(**{field: value, 'penalty_context_size': context})
                processors = inference.make_generation_processors(profile, len(prompt))
                self.assertTrue(processors)
                logits = mx.full((1, 96), 4.0)
                for processor in processors:
                    logits = processor(tokens, logits)
                mx.eval(logits)
                with self.subTest(field=field, context=context):
                    self.assertAlmostEqual(float(logits[0, 7].item()), 4.0)
                    self.assertAlmostEqual(float(logits[0, 3].item()), expected if context == 0 else 4.0)
                    self.assertAlmostEqual(float(logits[0, 5].item()), expected)
                    self.assertAlmostEqual(float(logits[0, 9].item()), 4.0)
                # Native processors must also leave the first sampling step alone.
                empty_logits = mx.full((1, 96), 4.0)
                for processor in processors:
                    empty_logits = processor(mx.array(prompt, dtype=mx.int32), empty_logits)
                self.assertTrue(bool(mx.all(empty_logits == 4.0).item()))


class OfficialThinkingTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('TRID_LOCAL_MLX_TOKENIZER_DIR')
        if not path:
            raise unittest.SkipTest('Set TRID_LOCAL_MLX_TOKENIZER_DIR to the pinned local Qwen3 tokenizer')
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def test_native_template_receives_explicit_thinking_setting(self):
        messages = [dict(role='system', content='Be precise.\n'), dict(role='user', content='  1+1?\n')]
        rendered = []
        for thinking in (False, True):
            profile = dict(inference.historical_profile(max_new_tokens=8), thinking=thinking)
            text, ids = inference.render_prompt(self.tokenizer, messages, profile)
            self.assertEqual(text, self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking))
            self.assertEqual(ids, self.tokenizer.encode(text, add_special_tokens=False))
            rendered.append(text)
        self.assertNotEqual(*rendered)

    def test_unclosed_reasoning_stays_private_at_length_and_eos(self):
        tokenizer = self.tokenizer
        body = ' \n아직 생각 중입니다.\t '
        body_ids = tokenizer.encode(body, add_special_tokens=False)
        for eos in (False, True):
            ids = body_ids + ([tokenizer.eos_token_id] if eos else [])
            decoded = inference.decode_generation(
                tokenizer, ids, {tokenizer.eos_token_id}, len(ids), thinking=True)
            with self.subTest(eos=eos):
                self.assertEqual(decoded['raw_output'], body + (tokenizer.eos_token if eos else ''))
                self.assertEqual(decoded['response_text'], '')
                self.assertEqual(decoded['framing_status'], 'incomplete_reasoning')
                self.assertFalse(decoded['reasoning_complete'])
                self.assertEqual(decoded['termination_reason'], 'eos' if eos else 'length')
                self.assertEqual(decoded['generated_tokens'], len(ids))
                self.assertEqual(decoded['terminal_stop_token_id'], tokenizer.eos_token_id if eos else None)

    def test_native_closing_boundary_preserves_answer_whitespace_and_raw_reasoning(self):
        tokenizer = self.tokenizer
        closing = tokenizer.convert_tokens_to_ids('</think>')
        self.assertEqual(tokenizer.get_added_vocab()['</think>'], closing)
        self.assertIn('</think>', tokenizer.get_chat_template())
        self.assertEqual(tokenizer.convert_ids_to_tokens(closing), '</think>')
        self.assertEqual(tokenizer.encode('</think>', add_special_tokens=False), [closing])
        reasoning, answer = '  비공개 추론\n', '\n\n  답변입니다.\t '
        for eos in (False, True):
            ids = (tokenizer.encode(reasoning, add_special_tokens=False) + [closing]
                   + tokenizer.encode(answer, add_special_tokens=False)
                   + ([tokenizer.eos_token_id] if eos else []))
            decoded = inference.decode_generation(
                tokenizer, ids, {tokenizer.eos_token_id}, len(ids), thinking=True)
            with self.subTest(eos=eos):
                self.assertEqual(decoded['raw_output'], reasoning + '</think>' + answer
                                 + (tokenizer.eos_token if eos else ''))
                self.assertEqual(decoded['response_text'], answer)
                self.assertEqual(decoded['framing_status'], 'complete')
                self.assertTrue(decoded['reasoning_complete'])


class DiagnosticPlanTests(unittest.TestCase):
    """Exercise file admission only; fixture weight bytes are never loaded."""

    def setUp(self):
        import diagnose_mlx_inference as diagnostic
        self.diagnostic = diagnostic
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.config = dict(model_type='qwen3', max_position_embeddings=4096)
        (self.model / 'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        (self.model / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
        (self.model / 'tokenizer.json').write_text('{}', encoding='utf-8')
        # Admission checks hashes and presence, not model or tokenizer execution.
        (self.model / 'model.safetensors').write_bytes(b'file-admission-only; never load')
        self.cases = [dict(id='one', messages=[dict(role='user', content='Question?')])]
        self.case_path = self.root / 'cases.json'
        self.case_path.write_text(json.dumps(self.cases), encoding='utf-8')
        source_root = Path(diagnostic.__file__).resolve().parent
        self.plan = dict(
            schema=diagnostic.SCHEMA, run_id='local-plan-test',
            model=self.directory_pin(self.model), adapter=None, cases=self.pin(self.case_path),
            profiles=[inference.historical_profile()],
            limits=dict(rss_stop_bytes=6 * 1024 ** 3, mlx_stop_bytes=6 * 1024 ** 3,
                        swap_growth_stop_bytes=256 * 1024 ** 2, pressure_required=1,
                        question_seconds=180, inference_seconds=1800),
            output_dir=str(self.root / 'output'),
            source_pins=[self.pin(source_root / name) for name in diagnostic.SOURCE_NAMES])

    @staticmethod
    def pin(path, root=None):
        raw = path.read_bytes()
        return dict(path=str(path.relative_to(root) if root else path.resolve()),
                    sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    def directory_pin(self, path):
        return dict(path=str(path), files=[self.pin(item, path) for item in sorted(path.iterdir())])

    def test_valid_pinned_plan_returns_config_and_cases_without_loading(self):
        before = copy.deepcopy(self.plan)
        self.assertEqual(self.diagnostic.validate_plan(self.plan), (self.config, self.cases))
        self.assertEqual(self.plan, before)
        self.assertFalse((self.root / 'output').exists())
        self.diagnostic.validate_plan(dict(self.plan, parity=True))

    def test_corrupted_model_case_and_source_hashes_and_sizes_are_rejected(self):
        for target in ('model', 'cases', 'source'):
            for field, bad in (('sha256', '0' * 64), ('bytes', 0)):
                candidate = copy.deepcopy(self.plan)
                pin = (candidate['model']['files'][0] if target == 'model' else
                       candidate['cases'] if target == 'cases' else candidate['source_pins'][0])
                pin[field] = bad
                with self.subTest(target=target, field=field), self.assertRaises(RuntimeError):
                    self.diagnostic.validate_plan(candidate)

    def test_missing_duplicate_and_incomplete_source_pins_are_rejected(self):
        for pins in ([], self.plan['source_pins'][:-1],
                     self.plan['source_pins'] + self.plan['source_pins'][:1],
                     [dict(self.plan['source_pins'][0], path=str(self.root / 'missing.py')),
                      *self.plan['source_pins'][1:]]):
            with self.subTest(pins=pins), self.assertRaises(RuntimeError):
                self.diagnostic.validate_plan(dict(self.plan, source_pins=pins))

    def test_missing_unpinned_and_duplicate_model_files_are_rejected(self):
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'):
            path = self.model / name
            raw = path.read_bytes()
            path.unlink()
            try:
                candidate = copy.deepcopy(self.plan)
                candidate['model'] = self.directory_pin(self.model)
                with self.subTest(missing=name), self.assertRaises(RuntimeError):
                    self.diagnostic.validate_plan(candidate)
            finally:
                path.write_bytes(raw)
        candidate = copy.deepcopy(self.plan)
        candidate['model']['files'].append(candidate['model']['files'][0])
        with self.assertRaises(RuntimeError):
            self.diagnostic.validate_plan(candidate)
        (self.model / 'unregistered.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.diagnostic.validate_plan(self.plan)

    def test_profiles_limits_and_parity_fail_closed(self):
        invalid = [dict(self.plan, profiles=[]),
                   dict(self.plan, profiles=self.plan['profiles'] * 2),
                   dict(self.plan, profiles=[dict(inference.historical_profile(), context_tokens=4097)]),
                   dict(self.plan, parity=True, profiles=[inference.historical_profile(max_new_tokens=12)]),
                   dict(self.plan, parity='true')]
        for field, value in (('rss_stop_bytes', 6 * 1024 ** 3 + 1), ('mlx_stop_bytes', True),
                             ('swap_growth_stop_bytes', 0), ('pressure_required', 2),
                             ('question_seconds', 181), ('inference_seconds', 1801),
                             ('inference_seconds', float('nan'))):
            invalid.append(dict(self.plan, limits=dict(self.plan['limits'], **{field: value})))
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises((RuntimeError, ValueError)):
                self.diagnostic.validate_plan(candidate)

    def test_invalid_cases_and_missing_case_file_are_rejected(self):
        invalid = [[], self.cases * 2, [dict(self.cases[0], answer='leaked gold')],
                   [dict(id='', messages=self.cases[0]['messages'])],
                   [dict(id='one', messages=[])],
                   [dict(id='one', messages=[dict(role='assistant', content='answer')])],
                   [dict(id='one', messages=[dict(role='user', content='  ')])],
                   [dict(id='one', messages=[dict(role='tool', content='result')])]]
        for cases in invalid:
            self.case_path.write_text(json.dumps(cases), encoding='utf-8')
            candidate = dict(self.plan, cases=self.pin(self.case_path))
            with self.subTest(cases=cases), self.assertRaises(RuntimeError):
                self.diagnostic.validate_plan(candidate)
        self.case_path.unlink()
        with self.assertRaises(RuntimeError):
            self.diagnostic.validate_plan(self.plan)

    def test_adapter_missing_file_and_output_overwrite_are_rejected(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{"fine_tune_type":"lora"}', encoding='utf-8')
        candidate = dict(self.plan, adapter=self.directory_pin(adapter))
        with self.assertRaises(RuntimeError):
            self.diagnostic.validate_plan(candidate)
        (adapter / 'adapters.safetensors').write_bytes(b'file-admission-only; never load')
        candidate['adapter'] = self.directory_pin(adapter)
        self.assertEqual(self.diagnostic.validate_plan(candidate), (self.config, self.cases))
        for output in (self.model, self.model / 'nested', adapter, adapter / 'nested'):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                self.diagnostic.validate_plan(dict(candidate, output_dir=str(output)))


if __name__ == '__main__':
    unittest.main()
