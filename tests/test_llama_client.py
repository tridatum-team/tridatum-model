import unittest
from unittest.mock import patch

from trid import llama_client

MESSAGES = [{'role': 'user', 'content': 'q'}]


class LlamaClientTests(unittest.TestCase):
    def test_complete_rejects_missing_qwen3_no_think_prefix(self):
        with patch('trid.llama_client.request', return_value={'prompt': '...'}):
            with self.assertRaises(ValueError):
                llama_client.complete('http://x', MESSAGES, 'qwen3', 512, 4096)

    def test_complete_skips_when_prompt_plus_max_tokens_exceeds_context(self):
        calls = []

        def fake(base, endpoint, payload=None, timeout=180):
            calls.append(endpoint)
            if endpoint == '/apply-template':
                return {'prompt': 'p' + llama_client.QWEN3_NO_THINK}
            if endpoint == '/tokenize':
                return {'tokens': list(range(4000))}
            raise AssertionError(endpoint)

        with patch('trid.llama_client.request', side_effect=fake):
            result = llama_client.complete('http://x', MESSAGES, 'qwen3', 512, 4096)
        self.assertIs(result['skipped_context'], True)
        self.assertEqual(result['prompt_tokens'], 4000)
        self.assertNotIn('/completion', calls)

    def test_complete_payload_is_greedy(self):
        payloads = {}

        def fake(base, endpoint, payload=None, timeout=180):
            payloads[endpoint] = payload
            if endpoint == '/apply-template':
                return {'prompt': 'p' + llama_client.QWEN3_NO_THINK}
            if endpoint == '/tokenize':
                return {'tokens': [1, 2, 3]}
            return {'content': ' a ', 'tokens': [4], 'stop_type': 'eos', 'truncated': False, 'timings': {}}

        with patch('trid.llama_client.request', side_effect=fake):
            result = llama_client.complete('http://x', MESSAGES, 'qwen3', 512, 4096)
        payload = payloads['/completion']
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['seed'], 42)
        self.assertEqual(payload['samplers'], ['temperature'])
        self.assertEqual(payload['repeat_penalty'], 1)
        self.assertIs(payload['cache_prompt'], False)
        self.assertEqual(result['generated'], 'a')

    def test_count_prompt_tokens_returns_none_on_oserror(self):
        with patch('trid.llama_client.request', side_effect=OSError('down')):
            self.assertIsNone(llama_client.count_prompt_tokens('http://x', MESSAGES))

    def test_count_prompt_tokens_counts_tokens(self):
        def fake(base, endpoint, payload=None, timeout=180):
            return {'prompt': 'p'} if endpoint == '/apply-template' else {'tokens': [1, 2, 3]}

        with patch('trid.llama_client.request', side_effect=fake):
            self.assertEqual(llama_client.count_prompt_tokens('http://x', MESSAGES), 3)


if __name__ == '__main__':
    unittest.main()
