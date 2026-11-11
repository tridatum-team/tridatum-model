"""External tracing with tiny CPU tensors only; no model downloads or training data."""
import contextlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from scripts import trace_qwen_preferences as tracing


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
TRAINER_AVAILABLE = TORCH_AVAILABLE and importlib.util.find_spec('transformers') is not None
ROOT = Path(__file__).resolve().parents[1]


def trainer_module():
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(ROOT / 'scripts'))
        return importlib.import_module('train_qwen_preferences')
    finally:
        sys.path[:] = previous


def tiny_model():
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10)

        def forward(self, input_ids, attention_mask):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.weight[input_ids])

        def save_pretrained(self, destination):
            destination = Path(destination)
            destination.mkdir()
            torch.save(self.state_dict(), destination / 'adapter.pt')

    return TinyModel()


def accumulate(trainer, model):
    chosen = dict(ids=[0, 1, 2], labels=[-100, 1, 2])
    rejected = dict(ids=[0, 3], labels=[-100, 3])
    return trainer.accumulate_step(
        model, [('private-pair-id', dict(chosen=chosen, rejected=rejected), (-1.0, -2.0))],
        [('private-replay-id', chosen), ('other-private-id', rejected)],
        0.1, 0.7, 'cpu', 0, family='generic', replay_micro_batch=1)


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def termination_child(trace_path, out_path):
    """Called only by the SIGTERM test's isolated Python subprocess."""
    import torch
    trainer = trainer_module()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def validation(*args, **kwargs):
        print('VALIDATION_ENTERED', flush=True)
        threading.Event().wait()

    def training_main():
        accumulate(trainer, model)
        optimizer.step()
        trainer.T.validate(model, [], 'cpu', 0, 1)

    with mock.patch.object(trainer, 'main', training_main), mock.patch.object(trainer.T, 'validate', validation):
        tracing.run_traced(['--mode', 'train', '--out', out_path], trace_path)


class TraceCliTests(unittest.TestCase):
    def test_help_does_not_attempt_heavy_imports(self):
        program = '''
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'q8_gguf_base', 'train_qwen_preferences'}:
            raise AssertionError('help attempted a heavy import')
sys.meta_path.insert(0, RejectHeavy())
from scripts.trace_qwen_preferences import main
main(['--help'])
'''
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--trace', result.stdout)

    def test_refuses_existing_sidecar_cache_mode_and_output_descendants_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'new-run'
            existing = root / 'existing.jsonl'
            existing.write_text('keep this byte-for-byte')
            with self.assertRaises(FileExistsError):
                tracing.run_traced(['--mode', 'train', '--out', str(out)], existing)
            self.assertEqual(existing.read_text(), 'keep this byte-for-byte')
            for option in ('--out', '--ou', '--o'):
                for target in (out, out / 'trace.jsonl', out / 'nested' / 'trace.jsonl'):
                    with self.subTest(option=option, target=target), self.assertRaises(ValueError):
                        tracing.run_traced(['--mode=train', option, str(out)], target)
            self.assertFalse(out.exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tracing.run_traced(['--mode', 'cache', '--out', str(out)], root / 'cache.jsonl')
            self.assertFalse((root / 'cache.jsonl').exists())
            alias = root / 'alias'
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                tracing.run_traced(['--mode=train', '--out=' + str(out)], alias / 'new-run' / 'trace.jsonl')
            self.assertFalse(out.exists())


@unittest.skipUnless(TRAINER_AVAILABLE, 'requires optional torch and transformers training dependencies')
class TraceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.trainer = trainer_module()
        previous = sys.path[:]
        try:
            sys.path.insert(0, str(ROOT / 'scripts'))
            cls.q8 = importlib.import_module('q8_gguf_base')
        finally:
            sys.path[:] = previous

    def test_gradients_and_adamw_effects_unchanged_and_backward_phases_distinct(self):
        torch, trainer = self.torch, self.trainer
        plain, traced = tiny_model(), tiny_model()
        baseline = torch.optim.AdamW(plain.parameters(), lr=0.02)
        optimizer = torch.optim.AdamW(traced.parameters(), lr=0.02)
        expected = accumulate(trainer, plain)
        gradient = plain.weight.grad.clone()
        baseline.step()
        stream = io.StringIO()
        trace = tracing.TrainingTrace(stream)
        # Even while accumulation is active, nested backwards from model code are
        # not the two direct backward callsites in the original accumulation code.
        def nested_backward(module, inputs, result):
            torch.tensor(3.0, requires_grad=True).square().backward()

        handle = traced.register_forward_hook(nested_backward)
        self.addCleanup(handle.remove)
        with trace.install(trainer, self.q8):
            # A direct backward outside accumulate_step must not produce a phase.
            torch.tensor(2.0, requires_grad=True).square().backward()
            actual = accumulate(trainer, traced)
            self.assertTrue(torch.equal(gradient, traced.weight.grad))
            optimizer.step()
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(plain.weight, traced.weight))
        for key in baseline.state[plain.weight]:
            self.assertTrue(torch.equal(baseline.state[plain.weight][key], optimizer.state[traced.weight][key]))
        events = records(stream)
        names = [e['event'] for e in events]
        self.assertEqual(names.count('preference_backward_enter'), 1)
        self.assertEqual(names.count('replay_backward_enter'), 2)
        self.assertEqual(names.count('preference_forward_enter'), 1)
        self.assertEqual(names.count('replay_forward_enter'), 2)
        self.assertLess(names.index('preference_backward_returned'), names.index('replay_forward_enter'))
        self.assertLess(names.index('accumulation_returned'), names.index('optimizer_update_enter'))
        self.assertEqual(names[-1], 'optimizer_update_returned')
        self.assertEqual(events[-1]['optimizer_updates'], 1)
        self.assertEqual([e['seq'] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual([e['monotonic_ns'] for e in events], sorted(e['monotonic_ns'] for e in events))
        self.assertNotIn('private', stream.getvalue())
        self.assertNotIn('labels', stream.getvalue())
        self.assertNotIn('checkpoint', stream.getvalue())

    def test_optimizer_identity_scope_and_existing_hooks_survive(self):
        torch, trainer = self.torch, self.trainer
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
        model = tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        unrelated = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.01)
        seen = []
        pre = register_optimizer_step_pre_hook(lambda opt, args, kwargs: seen.append(('pre', id(opt))))
        post = register_optimizer_step_post_hook(lambda opt, args, kwargs: seen.append(('post', id(opt))))
        stream = io.StringIO()
        try:
            with tracing.TrainingTrace(stream).install(trainer, self.q8):
                optimizer.step()  # Before an actual accumulation, not this run's update.
                accumulate(trainer, model)
                unrelated.step()  # Must not claim the first optimizer identity.
                optimizer.step()
                unrelated.step()
            before = stream.getvalue()
            optimizer.step()
            self.assertEqual(stream.getvalue(), before)
            self.assertEqual([kind for kind, ident in seen if ident == id(optimizer)], ['pre', 'post'] * 3)
            self.assertEqual([kind for kind, ident in seen if ident == id(unrelated)], ['pre', 'post'] * 2)
        finally:
            pre.remove()
            post.remove()
        self.assertEqual([e['event'] for e in records(stream) if e['event'].startswith('optimizer_')],
                         ['optimizer_update_enter', 'optimizer_update_returned'])

    def test_optimizer_exception_is_never_marked_returned(self):
        torch, trainer = self.torch, self.trainer
        failure = RuntimeError('SECRET optimizer error content')

        class ExplodingOptimizer(torch.optim.Optimizer):
            def __init__(self, params):
                super().__init__(params, {})

            def step(self, closure=None):
                raise failure

        model = tiny_model()
        optimizer = ExplodingOptimizer(model.parameters())
        stream = io.StringIO()
        trace = tracing.TrainingTrace(stream)
        with self.assertRaises(RuntimeError) as caught:
            with trace.phase('run'), trace.install(trainer, self.q8):
                accumulate(trainer, model)
                optimizer.step()
        self.assertIs(caught.exception, failure)
        names = [e['event'] for e in records(stream)]
        self.assertIn('optimizer_update_enter', names)
        self.assertNotIn('optimizer_update_returned', names)
        self.assertEqual(names[-1], 'run_error')
        self.assertEqual(trace.optimizer_updates, 0)
        self.assertNotIn('SECRET', stream.getvalue())

    def test_backward_exception_preserves_identity_and_restores_every_patch(self):
        torch, trainer = self.torch, self.trainer
        model = tiny_model()
        failure = RuntimeError('SECRET backward detail')
        originals = [(obj, name, getattr(obj, name)) for obj, name in (
            (trainer, 'verify_cache_shards'), (trainer, 'pair_logprobs'), (trainer, 'sequence_losses'),
            (trainer, 'accumulate_step'), (trainer, 'check_initial_agreement'),
            (trainer.T, 'load_init_adapter'), (trainer.T, 'validate'), (self.q8, 'load_qwen3_q8_gguf'),
            (torch.Tensor, 'backward'))]
        stream = io.StringIO()

        def fail(gradient):
            raise failure

        handle = model.weight.register_hook(fail)
        try:
            with self.assertRaises(RuntimeError) as caught:
                with tracing.TrainingTrace(stream).install(trainer, self.q8):
                    accumulate(trainer, model)
            self.assertIs(caught.exception, failure)
        finally:
            handle.remove()
        for obj, name, original in originals:
            self.assertIs(getattr(obj, name), original)
        names = [e['event'] for e in records(stream)]
        self.assertIn('preference_backward_error', names)
        self.assertNotIn('preference_backward_returned', names)
        self.assertNotIn('replay_backward_enter', names)
        self.assertFalse(any('SECRET' in json.dumps(e) for e in records(stream)))
        # A later, ordinary computation works after the failing tracing context.
        accumulate(trainer, model)
        self.assertIsNotNone(model.weight.grad)

    def test_actual_bound_adapter_serialization_is_not_checkpoint_completion(self):
        model, trainer = tiny_model(), self.trainer
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'adapter'
            # The adapter import requires PEFT; only this unrelated loading boundary
            # is replaced. The actual bound model serializer performs real disk I/O.
            with mock.patch.object(trainer.T, 'load_init_adapter', return_value=3):
                with tracing.TrainingTrace(stream).install(trainer, self.q8):
                    self.assertEqual(trainer.T.load_init_adapter(model, 'unused', 'cpu'), 3)
                    model.save_pretrained(checkpoint)
                    self.assertTrue((checkpoint / 'adapter.pt').is_file())
                    with self.assertRaises(FileExistsError):
                        model.save_pretrained(checkpoint)
            self.assertNotIn('save_pretrained', vars(model))
        names = [e['event'] for e in records(stream)]
        self.assertEqual(names, ['init_adapter_load_enter', 'init_adapter_load_returned',
                                 'adapter_save_enter', 'adapter_save_returned', 'adapter_save_enter', 'adapter_save_error'])
        self.assertFalse(any('checkpoint' in name for name in names))

    def test_initial_cache_and_validation_boundaries_and_exception_privacy(self):
        trainer, model = self.trainer, tiny_model()
        stream = io.StringIO()
        with tracing.TrainingTrace(stream).install(trainer, self.q8):
            self.assertEqual(trainer.verify_cache_shards([], {}, {}), {})
            chosen = dict(ids=[0, 1], labels=[-100, 1])
            with self.torch.no_grad():
                trainer.pair_logprobs(model, dict(chosen=chosen, rejected=chosen), 'cpu', 0, 'generic')
            trainer.check_initial_agreement([dict(id='SECRET', policy=dict(chosen=-1., rejected=-2.),
                                                 reference=dict(chosen=-1., rejected=-2.))], 0.1)
            # Empty validation still exercises the real no-grad decorated helper.
            self.assertEqual(trainer.T.validate(model, [], 'cpu', 0, 1, 'generic'), {})
            with self.assertRaises(ValueError):
                trainer.check_initial_agreement([dict(id='SECRET', policy=dict(chosen=-100., rejected=-2.),
                                                     reference=dict(chosen=-1., rejected=-2.))], 0.1)
        names = [e['event'] for e in records(stream)]
        for phase in ('cache_verification', 'initial_pair_scoring', 'initial_agreement', 'validation'):
            self.assertIn(phase + '_enter', names)
            self.assertIn(phase + '_returned', names)
        self.assertEqual(names[-1], 'initial_agreement_error')
        self.assertNotIn('SECRET', stream.getvalue())

    def test_run_preserves_literal_arguments_cwd_and_restores_process_state_on_error(self):
        trainer = self.trainer
        argv_object, path_object = sys.argv, sys.path
        saved_argv, saved_path, cwd = sys.argv[:], sys.path[:], os.getcwd()
        failure = RuntimeError('SECRET run detail')
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'trace.jsonl'
            args = ['--mode=train', '--out', './literal-new-output', '--lr=0.01', '--lr', '0.02']
            observed = []

            def training_main():
                observed.append((sys.argv[:], os.getcwd()))
                accumulate(trainer, tiny_model())
                raise failure

            with mock.patch.object(trainer, 'main', training_main), self.assertRaises(RuntimeError) as caught:
                tracing.run_traced(args, trace_path)
            self.assertIs(caught.exception, failure)
            self.assertEqual(observed, [([str(tracing.SCRIPTS / 'train_qwen_preferences.py'), *args], cwd)])
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'run_error')
            metadata = events[0]
            self.assertEqual(metadata['argv'][1:], args)
            self.assertEqual(metadata['cwd'], cwd)
            self.assertEqual(set(metadata['engine_sources']), set(tracing.ENGINE_NAMES))
            self.assertEqual(len(metadata['tracer']['sha256']), 64)
            runtime = next(e for e in events if e['event'] == 'runtime_identity')
            self.assertEqual(runtime['torch_version'], str(self.torch.__version__))
            self.assertEqual(metadata['engine_sources'], runtime['modules'])
            self.assertNotIn('SECRET', trace_path.read_text())
        self.assertIs(sys.argv, argv_object)
        self.assertIs(sys.path, path_object)
        self.assertEqual(sys.argv, saved_argv)
        self.assertEqual(sys.path, saved_path)
        self.assertEqual(os.getcwd(), cwd)

    def test_successful_run_tail_follows_optimizer_and_real_validation(self):
        torch, trainer = self.torch, self.trainer
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        argv_object, path_object = sys.argv, sys.path
        saved_path = sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / 'trace.jsonl'

            def training_main():
                accumulate(trainer, model)
                optimizer.step()
                item = dict(ids=[0, 1, 2], labels=[-100, 1, 2], task='synthetic')
                measured = trainer.T.validate(model, [item], 'cpu', 0, 1, 'generic')
                self.assertGreater(measured['synthetic'], 0)
                return measured

            with mock.patch.object(trainer, 'main', training_main):
                result = tracing.run_traced(['--mode', 'train', '--out', str(Path(directory) / 'out')], sidecar)
            self.assertIn('synthetic', result)
            events = [json.loads(line) for line in sidecar.read_text().splitlines()]
            names = [event['event'] for event in events]
            self.assertLess(names.index('optimizer_update_returned'), names.index('validation_enter'))
            self.assertLess(names.index('validation_returned'), names.index('run_returned'))
            self.assertEqual(names[-1], 'run_returned')
            self.assertEqual(events[-1]['status'], 'success')
            self.assertEqual(events[-1]['optimizer_updates'], 1)
        self.assertIs(sys.argv, argv_object)
        self.assertIs(sys.path, path_object)
        self.assertEqual(sys.path, saved_path)

    def _check_sidecar_close_failure(self, training_fails):
        trainer, torch = self.trainer, self.torch
        training_error = RuntimeError('private training failure')
        close_error = OSError('private close failure')
        opened = []
        real_open = Path.open
        argv_object, path_object, saved_path = sys.argv, sys.path, sys.path[:]
        original_backward = torch.Tensor.backward

        class FailingStream:
            def __init__(self, raw):
                self.raw = raw
                self.fail_flush = False
                self.flush_failed = False
                self.close_calls = 0

            def write(self, value):
                return self.raw.write(value)

            def flush(self):
                if self.fail_flush:
                    self.flush_failed = True
                    raise OSError('private flush failure')
                return self.raw.flush()

            def close(self):
                self.close_calls += 1
                # Exercise an actual file and release it even when simulating a
                # buffered close/flush failure; never fill the filesystem.
                self.raw.close()
                raise close_error

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / 'trace.jsonl'

            def opening(path, mode='r', *args, **kwargs):
                raw = real_open(path, mode, *args, **kwargs)
                if path == sidecar and mode == 'x':
                    stream = FailingStream(raw)
                    opened.append(stream)
                    return stream
                return raw

            def training_main():
                model = tiny_model()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                accumulate(trainer, model)
                optimizer.step()
                if training_fails:
                    opened[0].fail_flush = True
                    raise training_error

            expected = training_error if training_fails else close_error
            with mock.patch.object(Path, 'open', opening), mock.patch.object(trainer, 'main', training_main):
                with self.assertRaises(type(expected)) as caught:
                    tracing.run_traced(['--mode', 'train', '--out', str(Path(directory) / 'out')], sidecar)
            self.assertIs(caught.exception, expected)
            self.assertEqual(len(opened), 1)
            self.assertEqual(opened[0].close_calls, 1)
            self.assertTrue(opened[0].raw.closed)
            self.assertEqual(opened[0].flush_failed, training_fails)
            events = [json.loads(line) for line in sidecar.read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'run_error' if training_fails else 'run_returned')
        self.assertIs(sys.argv, argv_object)
        self.assertIs(sys.path, path_object)
        self.assertEqual(sys.path, saved_path)
        self.assertIs(torch.Tensor.backward, original_backward)

    def test_training_exception_survives_sidecar_flush_and_close_failures(self):
        self._check_sidecar_close_failure(training_fails=True)

    def test_sidecar_close_failure_propagates_after_successful_training(self):
        self._check_sidecar_close_failure(training_fails=False)

    @unittest.skipUnless(os.name == 'posix', 'SIGTERM semantics require POSIX')
    def test_sigterm_during_validation_leaves_flushed_incomplete_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / 'trace.jsonl'
            program = ('import sys; sys.path.insert(0, sys.argv[1]); '
                       'from test_trace_qwen_preferences import termination_child; '
                       'termination_child(sys.argv[2], sys.argv[3])')
            child = subprocess.Popen([sys.executable, '-B', '-c', program, str(ROOT / 'tests'),
                                      str(sidecar), str(Path(directory) / 'out')], cwd=ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                ready, _, _ = select.select([child.stdout], [], [], 60)
                self.assertTrue(ready, 'tiny child did not reach validation')
                line = child.stdout.readline()
                if line != 'VALIDATION_ENTERED\n':
                    child.kill()
                    _, errors = child.communicate(timeout=10)
                    self.fail('child failed before validation: ' + errors)
                # The phase entry is already readable while validation is blocked.
                before = [json.loads(line) for line in sidecar.read_text().splitlines()]
                self.assertEqual(before[-1]['event'], 'validation_enter')
                self.assertIn('optimizer_update_returned', [e['event'] for e in before])
                child.terminate()
                child.communicate(timeout=10)
                self.assertEqual(child.returncode, -signal.SIGTERM)
                after = [json.loads(line) for line in sidecar.read_text().splitlines()]
                self.assertEqual(after, before)
                self.assertFalse(any(e['event'] in ('validation_returned', 'run_returned', 'run_error') for e in after))
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=10)
                child.stdout.close()
                child.stderr.close()


if __name__ == '__main__':
    unittest.main()
