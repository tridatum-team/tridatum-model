"""Real, local official-tokenizer framing tests; no mocks, downloads, or model weights.

Set TRID_NATIVE_TOKENIZER_DIR to the already downloaded, pinned Qwen3-14B
metadata directory. Missing opt-in/assets are not replaced by an echo decoder.
"""
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
from grounded_hf_worker import decode_generation


class OfficialTokenizerFramingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('TRID_NATIVE_TOKENIZER_DIR')
        if not directory:
            raise unittest.SkipTest('Set TRID_NATIVE_TOKENIZER_DIR to the existing pinned official tokenizer metadata')
        from transformers import AutoTokenizer, GenerationConfig
        cls.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        cls.generation = GenerationConfig.from_pretrained(directory, local_files_only=True)
        eos = cls.generation.eos_token_id
        cls.stops = [eos] if type(eos) is int else eos
        if not isinstance(cls.stops, list) or not cls.stops or any(type(value) is not int for value in cls.stops):
            raise AssertionError('Official generation configuration must supply actual EOS IDs for this regression')

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def stop_text(self, token):
        return self.tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def test_each_actual_generation_eos_preserves_raw_protocol_and_decodes_json_body(self):
        body = '\n  {"synthetic":true,"text":"가나다"}\t \n'
        for stop in self.stops:
            with self.subTest(stop=stop):
                ids = self.encode(body) + [stop]
                original = ids.copy()
                result = decode_generation(self.tokenizer, ids, self.generation.eos_token_id, len(ids) + 10)
                self.assertEqual(result['raw_output'], body + self.stop_text(stop))
                self.assertEqual(result['response_text'], body)
                self.assertEqual(json.loads(result['response_text']), {'synthetic': True, 'text': '가나다'})
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(result['raw_output'])
                self.assertEqual(result['termination_reason'], 'eos')
                self.assertEqual(result['terminal_stop_token_id'], stop)
                self.assertEqual(result['generated_tokens'], len(ids))
                self.assertEqual(ids, original)

    def test_actual_eos_at_token_cap_is_still_eos_not_length(self):
        body = '{"synthetic":"at-cap"}'
        ids = self.encode(body) + [self.stops[0]]
        result = decode_generation(self.tokenizer, ids, self.generation.eos_token_id, len(ids))
        self.assertEqual(result['termination_reason'], 'eos')
        self.assertEqual(result['response_text'], body)
        self.assertEqual(result['terminal_stop_token_id'], ids[-1])

    def test_nonstop_length_case_preserves_every_token(self):
        body = ' \n{"synthetic":"length"}\t '
        ids = self.encode(body)
        self.assertNotIn(ids[-1], self.stops)
        result = decode_generation(self.tokenizer, ids, self.generation.eos_token_id, len(ids))
        self.assertEqual(result['raw_output'], body)
        self.assertEqual(result['response_text'], body)
        self.assertEqual(result['termination_reason'], 'length')
        self.assertIsNone(result['terminal_stop_token_id'])
        self.assertEqual(result['generated_tokens'], len(ids))

    def test_unconfigured_eos_is_not_inferred_from_tokenizer(self):
        body = '{"synthetic":"no-configured-stop"}'
        ids = self.encode(body) + [self.stops[0]]
        for unconfigured in (None, []):
            with self.subTest(eos=unconfigured):
                result = decode_generation(self.tokenizer, ids, unconfigured, len(ids))
                self.assertEqual(result['termination_reason'], 'length')
                self.assertIsNone(result['terminal_stop_token_id'])
                self.assertEqual(result['response_text'], body + self.stop_text(ids[-1]))
                self.assertEqual(result['raw_output'], result['response_text'])

    def test_internal_special_think_fences_and_whitespace_are_not_repaired(self):
        marker = '<|im_start|>'
        self.assertIn(marker, self.tokenizer.get_added_vocab())
        body = ' \n```json\n' + marker + '<think>synthetic inner text</think>\n{"synthetic":true}\n```\t '
        ids = self.encode(body) + [self.stops[0]]
        result = decode_generation(self.tokenizer, ids, self.generation.eos_token_id, len(ids) + 5)
        self.assertEqual(result['response_text'], body)
        self.assertEqual(result['raw_output'], body + self.stop_text(ids[-1]))
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result['response_text'])

    def test_short_nonstop_empty_or_overflow_sequences_cannot_fabricate_termination(self):
        ids = self.encode('{"synthetic":"incomplete-stop-contract"}')
        with self.assertRaisesRegex(RuntimeError, 'Unexplained generation termination'):
            decode_generation(self.tokenizer, ids, self.generation.eos_token_id, len(ids) + 1)
        with self.assertRaisesRegex(RuntimeError, 'registered envelope'):
            decode_generation(self.tokenizer, [], self.generation.eos_token_id, 1)
        with self.assertRaisesRegex(RuntimeError, 'registered envelope'):
            decode_generation(self.tokenizer, ids + [self.stops[0]], self.generation.eos_token_id, len(ids))


if __name__ == '__main__':
    unittest.main()
