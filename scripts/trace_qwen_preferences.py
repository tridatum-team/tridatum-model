"""External, train-only phase sidecar; the three cache-bound engine sources stay untouched.

Usage: python scripts/trace_qwen_preferences.py --trace NEW.jsonl -- ORIGINAL_ARGS
The sidecar must be outside --out. Events are flushed, not fsynced. A missing
run_returned/run_error tail (including SIGTERM) means incomplete observation, not
successful completion. optimizer_update_returned means only that the optimizer
implementation returned: not scheduler, validation, or checkpoint completion.
adapter_save_returned means serialization only, not checkpoint bookkeeping.
Tracing adds Python/I/O overhead; it is not uninstrumented throughput profiling.
"""
import argparse
from contextlib import ExitStack, contextmanager
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time


SCRIPTS = Path(__file__).resolve().parent
ENGINE_NAMES = ('train_qwen_preferences', 'train_open_weight_lora', 'q8_gguf_base')


def _source_identity(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


class TrainingTrace:
    """One synchronous trainer invocation; no tensor values or frames are retained."""

    def __init__(self, stream):
        self.stream = stream
        self.started = time.monotonic_ns()
        self.seq = 0
        self.step = 0
        self.optimizer_updates = 0
        self.phase_counts = {}
        self.in_accumulation = False
        self.backward_phase = None
        self.pending_update = False
        self.parameter_ids = set()
        self.optimizer_id = None
        self.entered_optimizer = None
        self.thread_id = threading.get_ident()

    def emit(self, event, **fields):
        now = time.monotonic_ns()
        self.seq += 1
        record = dict(event=event, seq=self.seq, monotonic_ns=now,
                      elapsed_seconds=(now - self.started) / 1e9,
                      step=self.step, optimizer_updates=self.optimizer_updates, **fields)
        self.stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.stream.flush()

    @contextmanager
    def phase(self, name):
        count = self.phase_counts.get(name, 0) + 1
        self.phase_counts[name] = count
        self.emit(name + '_enter', phase=name, phase_count=count, status='enter')
        try:
            yield
        except BaseException as error:
            # A sidecar write failure during unwind must not replace the training error.
            try:
                self.emit(name + '_error', phase=name, phase_count=count,
                          status='error', exception_type=type(error).__name__)
            except Exception:
                pass
            raise
        else:
            self.emit(name + '_returned', phase=name, phase_count=count, status='success')

    @contextmanager
    def install(self, trainer, q8=None):
        """Install only known boundaries; restore attributes and public hooks on exit."""
        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

        if q8 is None:
            q8 = importlib.import_module('q8_gguf_base')
        accumulate = trainer.accumulate_step
        accumulate_code = accumulate.__code__
        pair = trainer.pair_logprobs
        sequence = trainer.sequence_losses
        backward = torch.Tensor.backward

        def on_thread():
            return threading.get_ident() == self.thread_id

        with ExitStack() as cleanup:
            def patch(obj, name, replacement):
                # Restore inherited descriptors by removing the instance override.
                own = name in vars(obj)
                old = vars(obj).get(name)
                setattr(obj, name, replacement)
                if own:
                    cleanup.callback(setattr, obj, name, old)
                else:
                    cleanup.callback(delattr, obj, name)

            def wrap(obj, name, phase):
                original = getattr(obj, name)

                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    if not on_thread():
                        return original(*args, **kwargs)
                    with self.phase(phase):
                        return original(*args, **kwargs)
                patch(obj, name, wrapped)

            attached_models = set()
            load_adapter = trainer.T.load_init_adapter

            @functools.wraps(load_adapter)
            def adapter_load(model, *args, **kwargs):
                if not on_thread():
                    return load_adapter(model, *args, **kwargs)
                if id(model) not in attached_models:
                    wrap(model, 'save_pretrained', 'adapter_save')
                    attached_models.add(id(model))
                with self.phase('init_adapter_load'):
                    return load_adapter(model, *args, **kwargs)

            @functools.wraps(accumulate)
            def accumulation(model, *args, **kwargs):
                if not on_thread():
                    return accumulate(model, *args, **kwargs)
                self.step += 1
                self.pending_update = False
                self.parameter_ids = {id(p) for p in model.parameters() if p.requires_grad}
                self.in_accumulation = True
                self.backward_phase = None
                try:
                    with self.phase('accumulation'):
                        result = accumulate(model, *args, **kwargs)
                    self.pending_update = True
                    return result
                finally:
                    self.in_accumulation = False
                    self.backward_phase = None

            @functools.wraps(pair)
            def pair_forward(*args, **kwargs):
                if not on_thread():
                    return pair(*args, **kwargs)
                if self.in_accumulation:
                    self.backward_phase = 'preference'
                    phase = 'preference_forward'
                elif self.step == 0:
                    phase = 'initial_pair_scoring'
                else:
                    return pair(*args, **kwargs)
                with self.phase(phase):
                    return pair(*args, **kwargs)

            @functools.wraps(sequence)
            def sequence_forward(*args, **kwargs):
                if (on_thread() and self.in_accumulation
                        and sys._getframe(1).f_code is accumulate_code):
                    self.backward_phase = 'replay'
                    with self.phase('replay_forward'):
                        return sequence(*args, **kwargs)
                return sequence(*args, **kwargs)

            @functools.wraps(backward)
            def direct_backward(tensor, *args, **kwargs):
                if (on_thread() and self.in_accumulation
                        and sys._getframe(1).f_code is accumulate_code):
                    with self.phase(self.backward_phase + '_backward'):
                        return backward(tensor, *args, **kwargs)
                return backward(tensor, *args, **kwargs)

            def optimizer_pre(optimizer, args, kwargs):
                if not on_thread() or not self.pending_update or self.in_accumulation:
                    return
                if self.optimizer_id is None:
                    if not any(id(p) in self.parameter_ids for group in optimizer.param_groups for p in group['params']):
                        return
                    self.optimizer_id = id(optimizer)
                if id(optimizer) != self.optimizer_id:
                    return
                self.entered_optimizer = id(optimizer)
                self.emit('optimizer_update_enter', phase='optimizer_update',
                          phase_count=self.optimizer_updates + 1, status='enter')

            def optimizer_post(optimizer, args, kwargs):
                if not on_thread() or self.entered_optimizer != id(optimizer):
                    return
                self.entered_optimizer = None
                self.pending_update = False
                self.optimizer_updates += 1
                self.emit('optimizer_update_returned', phase='optimizer_update',
                          phase_count=self.optimizer_updates, status='success')

            wrap(trainer, 'verify_cache_shards', 'cache_verification')
            wrap(q8, 'load_qwen3_q8_gguf', 'q8_load')
            patch(trainer.T, 'load_init_adapter', adapter_load)
            wrap(trainer, 'check_initial_agreement', 'initial_agreement')
            wrap(trainer.T, 'validate', 'validation')
            patch(trainer, 'accumulate_step', accumulation)
            patch(trainer, 'pair_logprobs', pair_forward)
            patch(trainer, 'sequence_losses', sequence_forward)
            patch(torch.Tensor, 'backward', direct_backward)
            cleanup.callback(register_optimizer_step_pre_hook(optimizer_pre).remove)
            cleanup.callback(register_optimizer_step_post_hook(optimizer_post).remove)
            try:
                yield self
            finally:
                self.parameter_ids.clear()
                self.pending_update = False
                self.entered_optimizer = None


def _check_destination(trainer_argv, trace_path):
    # --o and --ou are the canonical trainer's unambiguous --out abbreviations.
    # Other trainer arguments remain literal and are validated by its own parser.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--out', '--ou', '--o', required=True)
    options, _ = parser.parse_known_args(trainer_argv)
    if options.mode != 'train':
        parser.error('tracing supports --mode train only')
    trace_path = Path(trace_path)
    resolved = trace_path.resolve()
    out = Path(options.out).resolve()
    if resolved == out or out in resolved.parents:
        raise ValueError('trace must be outside the trainer --out directory')
    if trace_path.exists() or trace_path.is_symlink():
        raise FileExistsError('trace sidecar already exists')
    return trace_path


def run_traced(trainer_argv, trace_path):
    """Run the canonical trainer once, preserving argument spelling, order and CWD."""
    trainer_argv = list(trainer_argv)
    trace_path = _check_destination(trainer_argv, trace_path)
    original_argv, original_path = sys.argv, sys.path
    saved_path = sys.path[:]
    # No parent mkdir: in particular this can never pre-create the trainer's output.
    stream = trace_path.open('x', encoding='utf-8')
    try:
        trace = TrainingTrace(stream)
        trace.emit('metadata', schema_version=1, status='metadata',
                   tracer=_source_identity(__file__),
                   engine_sources={name: _source_identity(SCRIPTS / (name + '.py')) for name in ENGINE_NAMES},
                   argv=[str(SCRIPTS / 'train_qwen_preferences.py'), *trainer_argv],
                   launcher_argv=list(original_argv), cwd=os.getcwd(),
                   runtime=dict(executable=sys.executable, python=sys.version,
                                platform=platform.platform(), pid=os.getpid()))
        try:
            sys.path.insert(0, str(SCRIPTS))
            sys.argv = [str(SCRIPTS / 'train_qwen_preferences.py'), *trainer_argv]
            with trace.phase('run'):
                with trace.phase('trainer_import'):
                    trainer = importlib.import_module('train_qwen_preferences')
                    q8 = importlib.import_module('q8_gguf_base')
                modules = (trainer, trainer.T, q8)
                for name, module in zip(ENGINE_NAMES, modules):
                    if Path(module.__file__).resolve() != SCRIPTS / (name + '.py'):
                        raise RuntimeError('canonical trainer module identity mismatch')
                versions = {}
                for name in ('torch', 'transformers', 'peft', 'safetensors', 'gguf', 'numpy', 'tokenizers', 'accelerate'):
                    try:
                        versions[name] = importlib.metadata.version(name)
                    except importlib.metadata.PackageNotFoundError:
                        versions[name] = None
                trace.emit('runtime_identity', status='metadata', versions=versions,
                           torch_version=str(trainer.torch.__version__), torch_file=str(trainer.torch.__file__),
                           modules={name: _source_identity(module.__file__) for name, module in zip(ENGINE_NAMES, modules)})
                with trace.install(trainer, q8):
                    result = trainer.main()
        finally:
            sys.argv = original_argv
            original_path[:] = saved_path
            sys.path = original_path
    except BaseException:
        # Closing can retry a failed buffered flush. Preserve the active training
        # exception rather than replacing it with a second sidecar I/O failure.
        try:
            stream.close()
        except BaseException:
            pass
        raise
    else:
        # With no active failure, an unsuccessful close must remain observable.
        stream.close()
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trace', required=True, type=Path)
    if '--' not in argv:
        parser.parse_args(argv)  # --help exits without any heavy import.
        parser.error('separate original trainer arguments with --')
    split = argv.index('--')
    options = parser.parse_args(argv[:split])
    return run_traced(argv[split + 1:], options.trace)


if __name__ == '__main__':
    main()
