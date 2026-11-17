"""Worker contract tests: no cloud, downloaded weights or fabricated model inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import grounded_hf_worker as core
import public_evidence_experiment as P
import public_evidence_worker as W


class TokenizerFixture:
    """Token-budget fixture only; never used to stand in for model generation."""
    def apply_chat_template(self, messages, **kwargs):
        if kwargs != dict(tokenize=False, add_generation_prompt=True, enable_thinking=False):
            raise AssertionError('Non-native template arguments')
        return messages[-1]['content'] + '<think>\n\n</think>\n\n'

    def __call__(self, text, **kwargs):
        if kwargs != dict(add_special_tokens=False, truncation=False):
            raise AssertionError('Tokenization may not truncate')
        return {'input_ids': list(text.encode())}


def cases():
    return [dict(id='first', messages=[dict(role='user', content='A')]),
            dict(id='second', messages=[dict(role='user', content='B')])]


def item(index):
    return dict(ids=[10, 11, 12, 1], labels=[-100, -100, 12, 1],
                identity=dict(id=f'row-{index}', source_sha256='a' * 64, source_line=index + 1))


class PublicWorkerContracts(unittest.TestCase):
    def test_full_preflight_rejects_late_overflow_without_partial_result(self):
        public = cases()
        public[1]['messages'][0]['content'] = 'X' * 100
        with self.assertRaisesRegex(ValueError, 'Full-corpus'):
            W.preflight_prompts(TokenizerFixture(), public, context_limit=64, max_new_tokens=8)

    def test_public_only_unique_ids_and_native_policy(self):
        for change in ('private', 'duplicate', 'assistant'):
            public = cases()
            if change == 'private':
                public[1]['gold'] = 'must not enter the worker'
            elif change == 'duplicate':
                public[1]['id'] = public[0]['id']
            else:
                public[1]['messages'][-1]['role'] = 'assistant'
            with self.subTest(change=change), self.assertRaises(ValueError):
                W.preflight_prompts(TokenizerFixture(), public, context_limit=64, max_new_tokens=8)
        result = W.preflight_prompts(TokenizerFixture(), cases(), context_limit=64, max_new_tokens=8)
        self.assertEqual([p['id'] for p in result], ['first', 'second'])
        self.assertNotEqual(result[0]['rendered_prompt_sha256'], result[1]['rendered_prompt_sha256'])

    def test_journal_is_durable_and_never_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'responses.jsonl'
            with path.open('x') as stream:
                W.append_row(stream, {'id': 'first', 'finish_reason': 'exception', 'error': {'type': 'RuntimeError'}})
                self.assertEqual(json.loads(path.read_text())['id'], 'first')
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                W.save_new(path, {'id': 'replacement'})
            self.assertEqual(path.read_bytes(), before)

    def test_ordered_preflight_cannot_admit_subset(self):
        public = cases()
        prepared = W.preflight_prompts(TokenizerFixture(), public, context_limit=64, max_new_tokens=8)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'responses.jsonl'
            with self.assertRaisesRegex(ValueError, 'Complete ordered'):
                W.stream_responses(None, None, public, prepared[:1], output,
                                   context_limit=64, max_new_tokens=8)
            self.assertFalse(output.exists())

    def test_exact_training_command_uses_plan_without_checkpoints(self):
        plan = dict(recipe=dict(P.RECIPE), model=dict(id=P.MODEL_ID, revision=P.MODEL_REVISION),
                    data=dict(manifest=dict(path='public/training-manifest.json')), schema=P.SCHEMA)
        argv = core.train_argv(plan, Path('/owned'))
        for flag, expected in (('--steps', '315'), ('--batch-size', '2'), ('--micro-batch-size', '1'),
                               ('--max-tokens', '4096'), ('--max-answer-tokens', '768'), ('--device', 'cuda')):
            self.assertEqual(argv[argv.index(flag) + 1], expected)
        self.assertNotIn('--checkpoint-every', argv)
        self.assertNotIn('--init-adapter', argv)
        self.assertNotIn('--merge', argv)

    def test_observer_requires_each_physical_row_once(self):
        plan = dict(recipe=dict(P.RECIPE))
        items = [item(i) for i in range(plan['recipe']['rows'])]
        observer, receipt = W.training_observer(plan, None, 1)
        observer('admission', train=items, valid=[], dropped=0, valid_dropped=0)
        coverage = [entry['identity'] for entry in reversed(items)]
        observer('completion', coverage=coverage, steps=plan['recipe']['steps'])
        self.assertTrue(receipt['coverage_exactly_once'])
        self.assertEqual(receipt['masking']['rows'], 630)
        self.assertTrue(receipt['masking']['eos_supervised'])
        with self.assertRaisesRegex(ValueError, 'wrapped, skipped or repeated'):
            observer('completion', coverage=coverage[:-1] + [coverage[0]], steps=315)

    def test_observer_rejects_mask_eos_drops_and_repeated_source(self):
        for failure in ('mask', 'eos', 'duplicate', 'dropped'):
            plan = dict(recipe=dict(P.RECIPE))
            items = [item(i) for i in range(plan['recipe']['rows'])]
            if failure == 'mask':
                items[-1]['labels'][0] = 10
            elif failure == 'eos':
                items[-1]['ids'][-1] = items[-1]['labels'][-1] = 2
            elif failure == 'duplicate':
                items[-1]['identity'] = copy.deepcopy(items[0]['identity'])
            observer, _ = W.training_observer(plan, None, 1)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                observer('admission', train=items, valid=[], dropped=int(failure == 'dropped'), valid_dropped=0)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'tokenizers', 'peft')),
                     'Real local HF/PEFT runtime required for tiny CPU transport proof')
class RealTinyCPUTransport(unittest.TestCase):
    def test_actual_native_generation_and_unmerged_peft_have_honest_smoke_receipts(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        tokenizer_backend = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, '<pad>': 2, 'A': 3, 'B': 4}, unk_token='<unk>'))
        tokenizer_backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token='<unk>',
                                           eos_token='<eos>', pad_token='<pad>')
        # Emit the suffix as a Jinja expression: literal trailing template
        # newlines are trimmed by the tokenizer's Jinja environment.
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt and enable_thinking == false %}"
            "{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
        )
        rendered = tokenizer.apply_chat_template(cases()[0]['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(rendered.endswith('<think>\n\n</think>\n\n'))
        torch.manual_seed(14)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=5, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=128, eos_token_id=1, pad_token_id=2))
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / 'baseline.jsonl'
            receipt = W.tiny_cpu_generation(model, tokenizer, cases(), baseline,
                                            max_new_tokens=2, context_limit=128)
            self.assertEqual(receipt['role'], 'smoke_only')
            self.assertEqual(receipt['device'], 'cpu')
            self.assertFalse(receipt['gpu_admission'])
            self.assertEqual(receipt['rows'], 2)
            self.assertEqual(receipt['exception_rows'], 0)
            model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
                                                    task_type='CAUSAL_LM'))
            candidate = Path(tmp) / 'candidate.jsonl'
            candidate_receipt = W.tiny_cpu_generation(model, tokenizer, cases(), candidate,
                                                      max_new_tokens=2, context_limit=128)
            self.assertEqual(candidate_receipt['exception_rows'], 0)
            records = [[json.loads(line) for line in path.read_text().splitlines()]
                       for path in (baseline, candidate)]
            self.assertEqual([r['seed'] for r in records[0]], [r['seed'] for r in records[1]])
            self.assertTrue(all(r['generation_receipt'] == 'hf_native_v1' and r['mode'] == 'smoke'
                                for arm in records for r in arm))
            self.assertTrue(all(r['rendered_prompt'].endswith('<think>\n\n</think>\n\n')
                                for arm in records for r in arm))
            self.assertTrue(all('gpu_uuid' not in r and 'binding' not in r for arm in records for r in arm))
            self.assertFalse(any(getattr(module, 'merged', False) for module in model.modules()))
            saved = Path(tmp) / 'adapter'
            model.save_pretrained(saved, safe_serialization=True)
            weights = saved / 'adapter_model.safetensors'
            W.verify_loaded_adapter(model, weights, torch)
            with torch.no_grad():
                next(p for name, p in model.named_parameters() if '.lora_B.' in name).add_(1)
            with self.assertRaisesRegex(ValueError, 'tensor differs'):
                W.verify_loaded_adapter(model, weights, torch)
            with self.assertRaises(FileExistsError):
                W.tiny_cpu_generation(model, tokenizer, cases(), baseline, max_new_tokens=2, context_limit=128)


if __name__ == '__main__':
    unittest.main()
