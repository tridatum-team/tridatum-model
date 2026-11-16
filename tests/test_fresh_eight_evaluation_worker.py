"""Exercise real paired loops, journals, decoder and greedy observation offline."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import evaluate_local_eight as H
import grounded_hf_worker as core
import fresh_eight_evaluation as E
import fresh_eight_evaluation_worker as W


class Config:
    do_sample = False
    num_beams = 1
    max_new_tokens = 768
    return_dict_in_generate = True
    eos_token_id = [9]

    def to_dict(self):
        return dict(do_sample=self.do_sample, num_beams=self.num_beams, max_new_tokens=768,
            return_dict_in_generate=True, eos_token_id=[9], use_cache=True)

    def get_generation_mode(self):
        return SimpleNamespace(value='sample' if self.do_sample else 'greedy_search')


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def __getitem__(self, key):
        row, column = key
        return SimpleNamespace(tolist=lambda: self.values[row][column])


class Torch:
    long = 'long'
    seeds = []

    def __init__(self):
        self.cuda = SimpleNamespace(manual_seed_all=self.manual_seed, synchronize=lambda: None,
            memory_allocated=lambda device: 10, memory_reserved=lambda device: 20,
            max_memory_allocated=lambda device: 30, max_memory_reserved=lambda device: 40)
        self.seeds = []

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def tensor(self, values, **kwargs):
        if kwargs.get('device') != 'cuda:0':
            raise AssertionError('wrong device')
        return Tensor(values)

    def ones_like(self, value):
        return Tensor([[1] * value.shape[1]])

    def inference_mode(self):
        return contextlib.nullcontext()


class Model:
    def __init__(self, fail_at=None, blank=False, bypass=False):
        self.disabled = False
        self.calls = []
        self.fail_at = fail_at
        self.blank = blank
        self.bypass = bypass

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def _prepare_generation_config(self, requested, use_model_defaults=None, **kwargs):
        if use_model_defaults is not False or kwargs != dict(do_sample=False, num_beams=1):
            raise AssertionError('wrong generation dispatch')
        return requested, {}

    def generate(self, input_ids, attention_mask, generation_config, **kwargs):
        self.calls.append((self.disabled, list(input_ids.values[0])))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('synthetic generation failure')
        if not self.bypass:
            self._prepare_generation_config(generation_config, **kwargs)
        tokens = [9] if self.blank else [100 + len(self.calls), 9]
        return SimpleNamespace(sequences=Tensor([input_ids.values[0] + tokens]))


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return ''.join('<eos>' if t == 9 else str(t) for t in tokens)

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != dict(tokenize=False, add_generation_prompt=True, enable_thinking=False):
            raise AssertionError('wrong template')
        return messages[0]['content'] + '<think>\n\n</think>\n\n'

    def __call__(self, prompt, **kwargs):
        if kwargs != dict(add_special_tokens=False, truncation=False):
            raise AssertionError('truncation attempted')
        return dict(input_ids=[1] * (3329 if prompt.startswith('oversized') else 2))


def cases(count):
    return H.preflight_prompts(Tokenizer(), [dict(id=E.SMOKE_ID if i == 0 else 'case-' + str(i),
        messages=[dict(role='user', content='question ' + str(i))]) for i in range(count)])


class EvaluationWorkerTests(unittest.TestCase):
    def test_smoke_then_all200_are_402_new_calls(self):
        torch, model = Torch(), Model()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode, count in [('smoke', 1), ('evaluation', 200)]:
                folder = root / mode
                folder.mkdir()
                W.run_pairs(core, H, torch, model, Tokenizer(), cases(count), Config(), dict(mode=mode), folder)
                rows = [json.loads(s) for s in (folder / 'base-outputs.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), count)
                self.assertEqual(len((folder / 'attempts.jsonl').read_text().splitlines()), count * 2)
            smoke = json.loads((root / 'smoke/base-outputs.jsonl').read_text())
            full = json.loads((root / 'evaluation/base-outputs.jsonl').read_text().splitlines()[0])
            self.assertEqual(smoke['id'], full['id'])
            self.assertNotEqual(smoke['output_token_ids'], full['output_token_ids'])
        self.assertEqual(len(model.calls), 402)
        self.assertEqual([disabled for disabled, _ in model.calls], [True, False] * 201)
        self.assertEqual(torch.seeds, [14] * 804)
        self.assertFalse(model.disabled)
        self.assertNotIn('_prepare_generation_config', model.__dict__)

    def test_generation_exception_stops_without_retry_and_keeps_rows(self):
        model = Model(fail_at=3)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaisesRegex(RuntimeError, 'synthetic generation failure'):
                W.run_pairs(core, H, Torch(), model, Tokenizer(), cases(200), Config(), dict(mode='evaluation'), folder)
            self.assertEqual(len(model.calls), 3)
            base_rows = [json.loads(s) for s in (folder / 'base-outputs.jsonl').read_text().splitlines()]
            self.assertEqual(len(base_rows), 2)
            self.assertEqual(base_rows[-1]['termination_reason'], 'error')
            self.assertEqual(base_rows[-1]['exception']['type'], 'RuntimeError')
            self.assertEqual(len((folder / 'candidate-outputs.jsonl').read_text().splitlines()), 1)
            attempts = [json.loads(s) for s in (folder / 'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['state'] for r in attempts], ['started', 'started', 'started', 'failed'])
            with self.assertRaises(FileExistsError):
                W.run_pairs(core, H, Torch(), model, Tokenizer(), cases(1), Config(), dict(mode='evaluation'), folder)
        self.assertFalse(model.disabled)

    def test_smoke_blank_stops_but_full_empty_stays_in_denominator(self):
        for mode, accepted in [('smoke', False), ('evaluation', True)]:
            model = Model(blank=True)
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                if accepted:
                    W.run_pairs(core, H, Torch(), model, Tokenizer(), cases(2), Config(), dict(mode=mode), folder)
                    self.assertEqual(len(model.calls), 4)
                else:
                    with self.assertRaisesRegex(ValueError, 'Smoke'):
                        W.run_pairs(core, H, Torch(), model, Tokenizer(), cases(1), Config(), dict(mode=mode), folder)
                    self.assertEqual(len(model.calls), 1)
                row = json.loads((folder / 'base-outputs.jsonl').read_text().splitlines()[0])
                self.assertEqual(row['response_text'], '')
                self.assertEqual(row['termination_reason'], 'eos')

    def test_effective_generation_bypass_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'bypassed'):
                W.run_pairs(core, H, Torch(), Model(bypass=True), Tokenizer(), cases(1), Config(),
                    dict(mode='evaluation'), Path(temp))
        config = Config()
        config.do_sample = True
        model = Model()
        with self.assertRaisesRegex(ValueError, 'greedy'):
            with H.observe_generation_policy(model, config):
                model._prepare_generation_config(config, **H.GREEDY_DISPATCH)

    def test_preflight_refuses_overlength_and_unknown_phase(self):
        rows = [dict(id='oversized', messages=[dict(role='user', content='oversized')])]
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            H.preflight_prompts(Tokenizer(), rows)
        for phase in ('train', 'capacity', 'baseline', 'adapter', 'select', 'unknown'):
            self.assertIsNone(W.phase_kind(phase))
            with self.assertRaises(ValueError):
                W.child_stage(None, dict(schema=E.SCHEMA), '.', phase)

    def test_durable_metadata_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            W.save_new(path, dict(status='first'))
            with self.assertRaises(FileExistsError):
                W.save_new(path, dict(status='replacement'))
            self.assertEqual(json.loads(path.read_bytes()), dict(status='first'))


if __name__ == '__main__':
    unittest.main()
