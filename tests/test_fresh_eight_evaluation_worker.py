"""Exercise real paired loops, journals, decoder and greedy observation offline."""
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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
    bfloat16 = 'torch.bfloat16'
    float32 = 'torch.float32'
    seeds = []

    def get_num_threads(self):
        return 1

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


class Device:
    type = 'cuda'
    index = 0

    def __str__(self):
        return 'cuda:0'


class Model:
    def __init__(self, fail_at=None, blank=False, bypass=False):
        self.disabled = False
        self.calls = []
        self.fail_at = fail_at
        self.blank = blank
        self.bypass = bypass
        self.generation_config = Config()
        self.config = SimpleNamespace(_attn_implementation='sdpa')

    def eval(self):
        return self

    def modules(self):
        return [self]

    def named_parameters(self):
        return [(name, SimpleNamespace(dtype=dtype, device=Device(),
            requires_grad=False, numel=lambda: 1, element_size=lambda size=size: size))
            for name, dtype, size in [('weight', Torch.bfloat16, 2), ('layer.lora_A.weight', Torch.float32, 4)]]

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def buffers(self):
        return []

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
    pad_token_id = None
    eos_token_id = 9
    bos_token_id = 1

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


@contextlib.contextmanager
def child_fixture(base):
    """Synthetic model/admission only; real journals, hashes and receipt validators.

    This fixture exercises control flow on CPU, not CUDA or model equivalence.
    """
    root = base / 'output'
    root.mkdir()
    (base / 'model').mkdir()
    model_file = base / 'model/weights'
    model_file.write_bytes(b'synthetic weights, not a model')
    sources = []
    for name in sorted(E.REQUIRED_SOURCES):
        path = base / 'code' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        sources.append(E.pin(path, base / 'code'))
    adapter_files = [dict(path='adapter/' + name, bytes=size, sha256=sha)
                     for name, (size, sha) in E.ADAPTER_FILES.items()]
    lineage = dict(schema=E.LINEAGE_SCHEMA, candidate=dict(bytes=414, sha256=E.CANDIDATE_SHA),
        adapter_sha256=E.ADAPTER_SHA, adapter_files=adapter_files, provenance=dict(test_fixture=True))
    W.save_new(root / 'lineage-binding.json', lineage)
    W.save_new(root / 'source-manifest.json', dict(files=sources))
    W.save_new(base / 'code/lineage-binding.json', lineage)
    W.save_new(base / 'code/source-manifest.json', dict(files=sources))
    plan = dict(schema=E.SCHEMA, limits=dict(E.LIMITS),
        lifetime=dict(total_seconds=19800, termination_grace_seconds=900, automatic_extension=False),
        cost=dict(termination_grace_seconds=900), runtime=dict(gpu_name='synthetic GPU'),
        model=dict(files=[E.pin(model_file, base / 'model')]),
        sources={p['path']: p['sha256'] for p in sources},
        workload=dict(adapter_files=adapter_files, acquisition_manifest=dict(test_fixture=True),
            evaluation_inputs=dict(path='inputs.json', bytes=0, sha256=E.INPUT_SHA),
            lineage_binding=E.pin(root / 'lineage-binding.json', root),
            source_manifest=E.pin(root / 'source-manifest.json', root)))
    W.save_new(base / 'plan.json', plan)
    W.save_new(root / 'plan.json', plan)
    release = dict(gpu=dict(uuid='GPU-synthetic'), instance_id='synthetic-instance')
    W.save_new(base / 'release.json', release)
    W.save_new(root / 'runtime.json', dict(packages=E.RUNTIME_PACKAGES,
        plan_sha256=core.digest(base / 'plan.json'), instance_id=release['instance_id'],
        torch=E.RUNTIME_PACKAGES['torch'], cuda='12.8', count=1, bf16=True,
        name=plan['runtime']['gpu_name'], run_id=E.RUN_ID))
    acquisition = dict(status='authenticated', model_id=E.MODEL_ID, revision=E.MODEL_REVISION,
        plan_sha256=core.digest(base / 'plan.json'), files=plan['model']['files'])
    W.save_new(root / 'model-acquisition.json', acquisition)
    admitted = dict(evaluation_inputs=[dict(id=p['id'], messages=p['messages']) for p in cases(200)],
        lineage=lineage, adapter_config={}, sources=sources)
    model, torch = Model(), Torch()
    distributions = [SimpleNamespace(metadata={'Name': name}, version=version)
                     for name, version in E.RUNTIME_PACKAGES.items()]
    modules = dict(transformers=SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        GenerationConfig=Config),
        peft=SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *a, **kw: model)))
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES=release['gpu']['uuid']))
        stack.enter_context(patch.object(W.importlib.metadata, 'distributions', return_value=distributions))
        stack.enter_context(patch.object(E, 'validate_data', return_value=admitted))
        stack.enter_context(patch.object(core, 'validate_model', return_value=None))
        stack.enter_context(patch.object(core, 'native_runtime', return_value=torch))
        stack.enter_context(patch.object(W, 'helpers', return_value=H))
        yield plan, model, acquisition


class EvaluationWorkerTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix' and os.geteuid() != 0,
                         'Real chmod denial requires an unprivileged POSIX user')
    def test_child_completes_with_private_receipt_but_parent_authenticates_it(self):
        for tampered in (False, True):
            with self.subTest(tampered=tampered), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                with child_fixture(base) as (plan, model, acquisition):
                    root = base / 'output'
                    private = root / 'model-acquisition.json'
                    # run_experiment owns creation of these parent receipts.
                    for name in ('plan.json', 'lineage-binding.json', 'source-manifest.json',
                                 'model-acquisition.json'):
                        (root / name).unlink()

                    def supervise(plan, base, phase, *args, **kwargs):
                        private.chmod(0o000)
                        try:
                            with self.assertRaises(PermissionError):
                                private.read_bytes()
                            W.child_stage(core, plan, base, phase)
                            self.assertEqual(private.stat().st_mode & 0o777, 0)
                            with self.assertRaises(PermissionError):
                                private.read_bytes()
                            self.assertFalse((root / 'evaluation/failure.json').exists())
                            self.assertFalse((root / 'smoke/failure.json').exists())
                            worker = json.loads((root / 'evaluation/evaluation-worker.json').read_bytes())
                            self.assertEqual(worker['generation_calls'], 402)
                            self.assertTrue(worker['evaluation_success'])
                            self.assertFalse(worker['quality_accepted'])
                        finally:
                            # Restore only the synthetic parent's access, never world-readability.
                            private.chmod(0o600)
                        self.assertEqual(json.loads(private.read_bytes()), acquisition)
                        if tampered:
                            private.write_bytes(E.canonical(dict(acquisition, status='unauthenticated')))
                        return dict(phase=phase, state='complete', cleanup_verified=True,
                            invocation='synthetic-invocation', run_id=E.RUN_ID,
                            instance_id='synthetic-instance', plan_sha256=core.digest(base / 'plan.json'),
                            stopped_state={'LoadState': 'not-found'})

                    try:
                        with patch.object(core, 'download'), \
                             patch.object(core, 'supervise_phase', side_effect=supervise), \
                             patch.object(os, 'chown'):
                            expected = (self.assertRaisesRegex(ValueError, 'model acquisition receipt')
                                        if tampered else contextlib.nullcontext())
                            with expected:
                                W.run_experiment(core, plan, base, dict(stages=[]), {},
                                    SimpleNamespace(pw_uid=os.geteuid(), pw_gid=os.getegid()),
                                    9999999999, 9999999999, [], set(), [])
                        self.assertEqual(len(model.calls), 402)
                        self.assertEqual(private.stat().st_mode & 0o777, 0o600)
                        self.assertEqual((root / 'evaluation/supervisor-result.json').exists(), not tampered)
                        if not tampered:
                            self.assertEqual(E.validate_supervision(root, plan)['status'], 'accepted')
                    finally:
                        # Parent acceptance freezes output directories; restore this fixture only.
                        for path in root.rglob('*'):
                            if path.is_dir():
                                path.chmod(0o755)

    def test_child_rejects_invalid_mode_evidence_before_worker_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with child_fixture(base) as (plan, model, _):
                run_pairs = W.run_pairs

                def corrupt_mode(*args):
                    run_pairs(*args)
                    manifest_path = args[-1] / 'run-manifest.json'
                    manifest = json.loads(manifest_path.read_bytes())
                    manifest['retries'] = True
                    manifest_path.write_bytes(E.canonical(manifest))

                with patch.object(W, 'run_pairs', side_effect=corrupt_mode):
                    with self.assertRaisesRegex(ValueError, 'Paired mode contract'):
                        W.child_stage(core, plan, base, 'evaluation')
                self.assertEqual(len(model.calls), 2)
                failure = json.loads((base / 'output/smoke/failure.json').read_bytes())
                self.assertEqual(failure['type'], 'ValueError')
                self.assertTrue(failure['completed_rows_retained'])
                self.assertFalse((base / 'output/evaluation/evaluation-worker.json').exists())
                self.assertFalse((base / 'output/evaluation/base-outputs.jsonl').exists())

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
