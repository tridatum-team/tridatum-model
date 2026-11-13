"""Pinned, offline Qwen3 quantized LoRA; one exclusive process per phase.

CLI: python scripts/mlx_grounded.py --plan /absolute/plan.json --phase
     audit|capacity|baseline|train|adapter
Run in that order under mlx_grounded_guard.py. The allocator limit is a guideline,
not a hard cap; the guard owns sampled RSS, pressure, swap and process-group wall
limits. This module imports neither MLX nor Torch until main's startup gate (and
never imports Torch). Tests may import the pure encoding/decoding helpers.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time

SCHEMA = 'trid-local-mlx-v1'
RECEIPT_SCHEMA = 'trid-local-mlx-receipt-v1'
PHASES = ('audit', 'capacity', 'baseline', 'train', 'adapter')
NO_THINK = '<think>\n\n</think>\n\n'
MODEL_REPO = 'mlx-community/Qwen3-4B-4bit'
MODEL_REVISION = '4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25'
RECIPE = dict(seed=14, rank=8, scale=2.0, dropout=0.0, num_layers=4,
              keys=['self_attn.q_proj', 'self_attn.v_proj'], learning_rate=0.00002,
              epochs=1, steps=34, train_context=2048, eval_context=3072, answer_tokens=768)
LIMITS = dict(rss_stop_bytes=6442450944, mlx_stop_bytes=6442450944,
              swap_growth_stop_bytes=268435456, pressure_required=1,
              capacity_seconds=180, question_seconds=180, train_seconds=900,
              evaluation_seconds=1800)
PACKED_HEAD_CHUNK = 64


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_pin(pin, base=None):
    path = Path(pin['path'])
    if base is not None:
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe model relative path')
        path = base / path
    else:
        require(path.is_absolute(), 'Pinned paths must be absolute')
    require(path.is_file() and sha256(path) == pin['sha256'], f'File hash mismatch: {path}')
    if 'bytes' in pin:
        require(path.stat().st_size == pin['bytes'], f'File size mismatch: {path}')
    return path


def source_provenance():
    packages = ('mlx', 'mlx-lm', 'transformers', 'tokenizers', 'safetensors')
    versions = {name: importlib.metadata.version(name) for name in packages}
    require(versions['mlx'] == '0.32.4' and versions['mlx-lm'] == '0.32.2',
            'This worker requires the inspected MLX 0.32.4 / mlx-lm 0.32.2 APIs')
    distribution = importlib.metadata.distribution('mlx-lm')
    files = [Path(__file__).resolve()] + [distribution.locate_file('mlx_lm/' + name)
             for name in ('models/qwen3.py', 'tuner/lora.py', 'tuner/utils.py',
                          'generate.py', 'utils.py', 'tokenizer_utils.py')]
    return versions, [dict(path=str(Path(path).resolve()), sha256=sha256(path)) for path in files]


def validate_plan(plan):
    require(plan.get('schema') == SCHEMA and isinstance(plan.get('run_id'), str)
            and bool(plan['run_id']), 'Invalid local plan identity')
    require(plan['model']['repo_id'] == MODEL_REPO and plan['model']['revision'] == MODEL_REVISION,
            'Only the registered new Qwen3-4B quantized base is allowed')
    require(plan['recipe'] == RECIPE and plan['limits'] == LIMITS, 'Local recipe/limits differ')
    root = Path(plan['model']['path'])
    require(root.is_absolute() and root.is_dir(), 'Local pinned model directory required')
    names = [item['path'] for item in plan['model']['files']]
    require(len(names) == len(set(names)) and {'config.json', 'tokenizer_config.json', 'tokenizer.json'} <= set(names),
            'Incomplete or repeated model pins')
    # Reject unpinned local files that HF/MLX could discover (including alternate weights/templates).
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    require(actual == set(names), 'Model directory must contain exactly the registered files')
    for pin in plan['model']['files']:
        verify_pin(pin, root)
    config = json.loads((root / 'config.json').read_text())
    require(config.get('model_type') == 'qwen3' and not config.get('model_file'), 'Native Qwen3 only')
    quant = config.get('quantization', {})
    require(quant.get('bits') == 4 and any(name.endswith('.safetensors') for name in names),
            'A real MLX 4-bit checkpoint is required')
    for key in ('train', 'blind_inputs'):
        verify_pin(plan[key])
    require(Path(plan['output_dir']).is_absolute(), 'Absolute output directory required')
    return config


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_messages(messages):
    require(isinstance(messages, list) and len(messages) == 2, 'Exactly system/user messages required')
    for message, role in zip(messages, ('system', 'user')):
        require(isinstance(message, dict) and set(message) == {'role', 'content'}
                and message['role'] == role and isinstance(message['content'], str)
                and bool(message['content'].strip()), f'Invalid {role} message')


def prompt_tokens(tokenizer, messages):
    check_messages(messages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    require(text.endswith(NO_THINK), 'Native Qwen3 no-thinking prefix was not rendered')
    ids = tokenizer.encode(text, add_special_tokens=False)
    require(bool(ids), 'Empty native prompt')
    return text, ids


def assemble_training_tokens(prompt_ids, answer_ids, context, answer_budget):
    require(bool(prompt_ids) and bool(answer_ids), 'Empty prompt or supervised answer')
    require(len(answer_ids) <= answer_budget, 'Answer plus EOS exceeds answer budget')
    require(len(prompt_ids) + len(answer_ids) <= context, 'Training context budget exceeded; no truncation')
    return dict(ids=prompt_ids + answer_ids, labels=[-100] * len(prompt_ids) + answer_ids,
                prompt_tokens=len(prompt_ids), answer_tokens=len(answer_ids))


def encode_training(tokenizer, row, recipe):
    require('history' not in row, 'Training history is forbidden')
    for field in ('id', 'task', 'split', 'answer'):
        require(isinstance(row.get(field), str) and bool(row[field].strip()), f'Missing training {field}')
    require(row['split'] == 'train', 'Only admitted training rows may be optimized')
    text, prompt_ids = prompt_tokens(tokenizer, row['messages'])
    require(isinstance(tokenizer.eos_token, str) and type(tokenizer.eos_token_id) is int,
            'Tokenizer EOS is unavailable')
    # Deliberately identical to train_open_weight_lora.encode: separate prompt and answer+EOS encodes.
    answer_ids = tokenizer.encode(row['answer'] + tokenizer.eos_token, add_special_tokens=False)
    require(bool(answer_ids) and answer_ids[-1] == tokenizer.eos_token_id,
            'Answer encoding did not end in actual tokenizer EOS')
    item = assemble_training_tokens(prompt_ids, answer_ids, recipe['train_context'], recipe['answer_tokens'])
    return dict(item, id=row['id'], task=row['task'], prompt_sha256=hashlib.sha256(canonical(row['messages'])).hexdigest(),
                rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest())


def eos_ids(tokenizer, config):
    value = config.get('eos_token_id', [])
    values = [value] if type(value) is int else value
    require(isinstance(values, list) and all(type(x) is int for x in values), 'Invalid config EOS IDs')
    values = sorted(set(values + [tokenizer.eos_token_id]))
    require(values and all(type(x) is int and x >= 0 for x in values), 'Invalid native EOS IDs')
    return values


def decode_generation(tokenizer, generated_ids, eos_token_id, max_new_tokens):
    stops = {eos_token_id} if type(eos_token_id) is int else set(eos_token_id or [])
    require(0 < len(generated_ids) <= max_new_tokens and all(type(x) is int and x >= 0 for x in generated_ids),
            'Generated IDs violate registered envelope')
    terminal = generated_ids[-1] if generated_ids[-1] in stops else None
    require(terminal is not None or len(generated_ids) == max_new_tokens, 'Unexplained generation termination')
    kwargs = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return dict(raw_output=tokenizer.decode(generated_ids, **kwargs),
                response_text=tokenizer.decode(generated_ids[:-1] if terminal is not None else generated_ids, **kwargs),
                generated_tokens=len(generated_ids), terminal_stop_token_id=terminal,
                termination_reason='eos' if terminal is not None else 'length')


def audit_inputs(plan, tokenizer):
    train = read_jsonl(plan['train']['path'])
    require(len(train) == 34 and len({row.get('id') for row in train}) == 34, 'Exactly 34 unique training rows required')
    encoded = [encode_training(tokenizer, row, plan['recipe']) for row in train]
    blind = read_jsonl(plan['blind_inputs']['path'])
    pins = plan['blind_cases']
    require(len(blind) == len(pins) == 18 and len({p['id'] for p in pins}) == 18, 'Exactly 18 unique blind cases required')
    require(sum(p['kind'] == 'grounded' for p in pins) == 12
            and sum(p['kind'] == 'synthetic-general-retention' for p in pins) == 6, 'Wrong blind domain/retention mix')
    for row, pin in zip(blind, pins):
        require(set(row) == {'id', 'messages'} and row['id'] == pin['id'], 'Blind inputs/order differ or contain gold')
        check_messages(row['messages'])
        require(hashlib.sha256(canonical(row['messages'])).hexdigest() == pin['prompt_sha256'], 'Blind prompt pin differs')
        text, ids = prompt_tokens(tokenizer, row['messages'])
        require(len(ids) + plan['recipe']['answer_tokens'] <= plan['recipe']['eval_context'],
                'Blind prompt plus full answer budget exceeds context; no truncation')
        row['prompt_ids'] = ids
        row['rendered_prompt_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    require(not ({row['id'] for row in train} & {row['id'] for row in blind}), 'Train/blind IDs overlap')
    return encoded, blind


class Deadline:
    """Independent wall watchdog: includes blocked prefill, not just token callbacks.

    Python/Metal work may not return to a token callback in time. The watchdog
    fails the phase and exits the owned worker; the external guard kills its group.
    """
    def __init__(self, seconds, callback):
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(seconds, callback), daemon=True)

    def _run(self, seconds, callback):
        if not self.cancelled.wait(seconds):
            try:
                callback()
            finally:
                os._exit(124)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.cancelled.set()
        self.thread.join()


class Runtime:
    def __init__(self, plan, phase, plan_sha):
        self.plan, self.phase = plan, phase
        self.output = Path(plan['output_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.receipt_path = self.output / f'{phase}-receipt.json'
        require(not self.receipt_path.exists(), 'Phase receipt already exists; retries are forbidden')
        self.start = time.monotonic()
        self.mx = None
        self.lock = threading.Lock()
        self.finished = False
        self.memory = dict(mlx_active_bytes=0, mlx_cache_bytes=0, mlx_peak_bytes=0)
        self.versions, sources = source_provenance()
        self.receipt = dict(schema=RECEIPT_SCHEMA, status='running', phase=phase, run_id=plan['run_id'],
                            plan_sha256=plan_sha, worker_source_sha256=sha256(__file__),
                            versions=self.versions, source_files=sources, started_at=timestamp(),
                            prerequisites=[], artifacts=[], sampled_limits=True,
                            allocator_limit_is_guideline=True)
        self.partial = None

    def prerequisites(self):
        position = PHASES.index(self.phase)
        if not position:
            return
        # Verify the complete chain, not just a claimed immediate predecessor.
        previous_pin = None
        for phase in PHASES[:position]:
            path = self.output / f'{phase}-receipt.json'
            receipt = json.loads(path.read_text())
            require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('status') == 'complete'
                    and receipt.get('phase') == phase and receipt.get('run_id') == self.plan['run_id']
                    and receipt.get('plan_sha256') == self.receipt['plan_sha256']
                    and receipt.get('source_files') == self.receipt['source_files']
                    and receipt.get('versions') == self.versions, f'Invalid prerequisite receipt: {phase}')
            require(receipt.get('prerequisites') == ([] if previous_pin is None else [previous_pin]),
                    f'Broken phase chain: {phase}')
            for pin in receipt['artifacts']:
                verify_pin(pin)
            previous_pin = dict(phase=phase, path=str(path), sha256=sha256(path))
        self.receipt['prerequisites'] = [previous_pin]

    def emit(self, event, **fields):
        print(json.dumps(dict(phase=self.phase, event=event, **self.memory, **fields),
                         ensure_ascii=False, allow_nan=False), flush=True)

    def check(self, event='progress', **fields):
        if self.mx is not None:
            self.memory = dict(mlx_active_bytes=self.mx.get_active_memory(),
                               mlx_cache_bytes=self.mx.get_cache_memory(), mlx_peak_bytes=self.mx.get_peak_memory())
        self.emit(event, **fields)
        cap = self.plan['limits']['mlx_stop_bytes']
        if self.memory['mlx_active_bytes'] + self.memory['mlx_cache_bytes'] > cap or self.memory['mlx_peak_bytes'] > cap:
            raise MemoryError('mlx_memory_limit: sampled active+cache or active peak exceeds limit')
        require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'Torch import forbidden')

    def finish(self, status, **fields):
        with self.lock:
            if self.finished:
                return
            self.receipt.update(status=status, finished_at=timestamp(), elapsed_seconds=time.monotonic() - self.start,
                                memory=self.memory, **fields)
            if self.partial is not None:
                self.receipt['partial_generation'] = self.partial
            write_json(self.receipt_path, self.receipt)
            self.finished = True
            self.emit('complete' if status == 'complete' else 'failure', **fields)

    def deadline(self, reason):
        self.finish('failed', stop_reason=reason, error=reason)


def configure_mlx(runtime):
    import mlx.core as mx
    runtime.mx = mx
    mx.set_cache_limit(0)
    mx.set_memory_limit(runtime.plan['limits']['mlx_stop_bytes'])
    mx.set_wired_limit(runtime.plan['limits']['mlx_stop_bytes'])
    mx.random.seed(runtime.plan['recipe']['seed'])
    runtime.check('mlx_ready')
    return mx


def load_base(runtime):
    from mlx_lm.utils import load_model
    from mlx_lm.models.qwen3 import Model
    import mlx.nn as nn
    model, _ = load_model(Path(runtime.plan['model']['path']), lazy=False, strict=True, trust_remote_code=False)
    require(type(model) is Model, 'Loaded model is not the inspected native Qwen3 implementation')
    require(len(model.layers) >= 4, 'Not enough Qwen3 layers')
    for layer in model.layers[-4:]:
        require(isinstance(layer.self_attn.q_proj, nn.QuantizedLinear)
                and isinstance(layer.self_attn.v_proj, nn.QuantizedLinear)
                and layer.self_attn.q_proj.bits == layer.self_attn.v_proj.bits == 4,
                'Q/V targets must remain genuine 4-bit base linears')
    model.freeze()
    runtime.check('base_loaded')
    return model


def attach_lora(model, recipe):
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    model.freeze()
    config = {key: recipe[key] for key in ('rank', 'scale', 'dropout', 'keys')}
    linear_to_lora_layers(model, recipe['num_layers'], config)
    expected = {f'model.layers.{i}.{key}.{leaf}'
                for i in range(len(model.layers) - recipe['num_layers'], len(model.layers))
                for key in recipe['keys'] for leaf in ('lora_a', 'lora_b')}
    actual = dict(tree_flatten(model.trainable_parameters()))
    require(set(actual) == expected, 'Trainable tensor coverage differs from last-four Q/V LoRA only')
    return actual


def packed_answer_loss(model, ids, prompt_length, chunk_size=PACKED_HEAD_CHUNK):
    """Full causal backbone, answer+EOS predecessor positions only, token-mean CE.

    Checkpoint each frozen-head chunk so backward never retains a prompt-wide
    vocabulary tensor. Frozen head weights are not differentiation arguments.
    """
    import mlx.core as mx
    import mlx.nn as nn
    require(ids.ndim == 1 and 0 < prompt_length < ids.shape[0] and chunk_size > 0, 'Invalid supervised positions')
    hidden = model.model(ids[None, :-1])[:, prompt_length - 1:, :]
    targets = ids[prompt_length:]
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head

    @mx.checkpoint
    def chunk_loss(h, y):
        # Checkpoint's VJP includes every positional argument; integer class
        # targets are constants, not differentiation inputs to cross_entropy.
        return nn.losses.cross_entropy(head(h).astype(mx.float32), mx.stop_gradient(y), reduction='sum')

    total = mx.array(0.0, dtype=mx.float32)
    for start in range(0, targets.shape[0], chunk_size):
        total = total + chunk_loss(hidden[:, start:start + chunk_size, :], targets[None, start:start + chunk_size])
    return total / targets.shape[0]


def finite_tree(tree):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    arrays = [value for _, value in tree_flatten(tree)]
    require(bool(arrays), 'Empty differentiable tensor tree')
    value = mx.all(mx.stack([mx.all(mx.isfinite(array)) for array in arrays]))
    return bool(value.item())


def train_step(runtime, model, optimizer, item):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    mx = runtime.mx
    runtime.check('before_update', row_id=item['id'])
    ids = mx.array(item['ids'], dtype=mx.int32)
    loss, gradients = nn.value_and_grad(model, packed_answer_loss)(model, ids, item['prompt_tokens'])
    mx.eval(loss, gradients)
    runtime.check('gradients_evaluated', row_id=item['id'])
    value = loss.item()
    require(math.isfinite(value) and finite_tree(gradients), 'Nonfinite training loss/gradient')
    grad_norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item()) for _, g in tree_flatten(gradients)))
    require(math.isfinite(grad_norm) and grad_norm > 0, 'Zero or nonfinite gradient norm')
    optimizer.update(model, gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    require(finite_tree(model.trainable_parameters()), 'Nonfinite updated adapter')
    runtime.check('update', row_id=item['id'], loss=value, gradient_norm=grad_norm)
    return dict(id=item['id'], loss=value, gradient_norm=grad_norm, supervised_tokens=item['answer_tokens'])


def adapter_config(recipe):
    return dict(fine_tune_type='lora', num_layers=recipe['num_layers'],
                lora_parameters={key: recipe[key] for key in ('rank', 'scale', 'dropout', 'keys')})


def save_adapter(model, directory, recipe):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    write_json(directory / 'adapter_config.json', adapter_config(recipe))
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(directory / 'adapters.safetensors'), weights)
    return [file_pin(directory / name) for name in ('adapter_config.json', 'adapters.safetensors')]


def reload_adapter(model, directory, recipe):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    directory = Path(directory)
    require(json.loads((directory / 'adapter_config.json').read_text()) == adapter_config(recipe), 'Adapter config differs')
    expected = dict(tree_flatten(model.trainable_parameters()))
    weights = mx.load(str(directory / 'adapters.safetensors'))
    require(set(expected) == set(weights) and all(expected[k].shape == weights[k].shape for k in expected),
            'Adapter reload requires exact tensor coverage and shapes')
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())
    require(all(bool(mx.array_equal(v, weights[k]).item()) for k, v in tree_flatten(model.trainable_parameters())),
            'Loaded adapter differs from saved tensors')


def changed_trainables(initial, model):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    result = {key: float(mx.max(mx.abs(value - initial[key])).item())
              for key, value in tree_flatten(model.trainable_parameters())}
    require(all(math.isfinite(value) for value in result.values()) and any(value > 0 for value in result.values()),
            'No finite trainable value changed')
    return result


def run_training(runtime, model, items, capacity):
    import mlx.optimizers as optim
    from mlx_lm.tuner.utils import remove_lora_layers
    mx, recipe = runtime.mx, runtime.plan['recipe']
    initial = attach_lora(model, recipe)
    mx.eval(initial)
    model.train()
    optimizer = optim.Adam(learning_rate=recipe['learning_rate'])
    if capacity:
        item = max(items, key=lambda row: len(row['ids']))
        history = [train_step(runtime, model, optimizer, item)]
        changes = changed_trainables(initial, model)
        model.eval()
        before = packed_answer_loss(model, mx.array(item['ids'], dtype=mx.int32), item['prompt_tokens']).item()
        runtime.check('capacity_before_save')
        with tempfile.TemporaryDirectory(prefix='capacity-adapter-', dir=runtime.output) as temp:
            directory = Path(temp) / 'adapter'
            saved = save_adapter(model, directory, recipe)
            # Replace LoRA modules (not the immutable base) before loading saved bytes.
            remove_lora_layers(model)
            attach_lora(model, recipe)
            reload_adapter(model, directory, recipe)
            model.eval()
            after = packed_answer_loss(model, mx.array(item['ids'], dtype=mx.int32), item['prompt_tokens']).item()
            require(math.isfinite(before) and math.isfinite(after) and math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-6),
                    'Capacity save/reload loss parity failed')
            runtime.check('capacity_reload')
            proof = dict(saved_files=saved, loss_before_reload=before, loss_after_reload=after,
                         exact_saved_tensor_parity=True, reattached_to_same_frozen_base=True)
        runtime.receipt.update(capacity_row_id=item['id'], capacity_row_tokens=len(item['ids']),
                               capacity_selection='longest_audited_training_row', history=history,
                               changed_trainable_max_abs=changes, reload=proof, capacity_adapter_discarded=True)
    else:
        order = list(items)
        random.Random(recipe['seed']).shuffle(order)
        history = []
        for step, item in enumerate(order, 1):
            record = train_step(runtime, model, optimizer, item)
            record.update(step=step, elapsed_seconds=time.monotonic() - runtime.start)
            history.append(record)
            runtime.receipt['history'] = history
        require(len(history) == recipe['steps'] == len({row['id'] for row in history}), 'Final row coverage is not one pass')
        changes = changed_trainables(initial, model)
        runtime.receipt.update(epochs=1, updates=len(history), row_coverage=[row['id'] for row in history],
                               changed_trainable_max_abs=changes, optimizer='mlx.optimizers.Adam',
                               learning_rate=recipe['learning_rate'], fresh_base=True, capacity_adapter_reused=False)
        runtime.receipt['artifacts'] = save_adapter(model, runtime.output / 'adapter', recipe)
        runtime.check('adapter_saved')


def collect_generation(model, prompt_ids, stops, max_tokens, check, generated_ids):
    """generate_step yields the actual EOS (stream_generate suppresses it)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    iterator = generate_step(mx.array(prompt_ids, dtype=mx.int32), model, max_tokens=max_tokens,
                             prefill_step_size=128, prompt_progress_callback=lambda n, total: check('prefill', n=n, total=total))
    try:
        for token, _logprobs in iterator:
            generated_ids.append(int(token))
            check('token', generated_tokens=len(generated_ids))
            if token in stops:
                break
    finally:
        iterator.close()
        mx.synchronize()
    require(bool(generated_ids), 'Generation yielded no tokens')


def run_evaluation(runtime, model, tokenizer, blind, stops):
    adapted = runtime.phase == 'adapter'
    if adapted:
        attach_lora(model, runtime.plan['recipe'])
        reload_adapter(model, runtime.output / 'adapter', runtime.plan['recipe'])
    model.eval()
    path = runtime.output / f'{runtime.phase}-outputs.jsonl'
    maximum = runtime.plan['recipe']['answer_tokens']
    runtime.receipt['questions'] = []
    with path.open('x', encoding='utf-8') as stream:
        for row, pin in zip(blind, runtime.plan['blind_cases']):
            start = time.monotonic()
            generated = []
            runtime.partial = dict(id=row['id'], output_token_ids=generated, prompt_token_ids=row['prompt_ids'])
            deadline = start + runtime.plan['limits']['question_seconds']
            runtime.emit('case_started', case_id=row['id'], question_started_monotonic=start,
                         question_deadline_monotonic=deadline)
            with Deadline(max(0, deadline - time.monotonic()), lambda: runtime.deadline('question_deadline')):
                collect_generation(model, row['prompt_ids'], set(stops), maximum,
                                   lambda event, **fields: runtime.check(event, case_id=row['id'], **fields), generated)
                decoded = decode_generation(tokenizer, generated, stops, maximum)
                record = dict(id=row['id'], kind=pin['kind'], family=pin['family'],
                              cluster_id=runtime.plan['run_id'], adapter=adapted,
                              prompt_sha256=pin['prompt_sha256'], rendered_prompt_sha256=row['rendered_prompt_sha256'],
                              prompt_token_ids=row['prompt_ids'], output_token_ids=generated, **decoded)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                runtime.check('case_finished', case_id=row['id'], question_finished_monotonic=time.monotonic(),
                              termination_reason=decoded['termination_reason'])
                runtime.receipt['questions'].append(dict(id=row['id'], elapsed_seconds=time.monotonic() - start,
                                                         generated_tokens=len(generated), **runtime.memory))
            runtime.partial = None
    runtime.receipt['artifacts'] = [file_pin(path)]
    if adapted:
        runtime.receipt['artifacts'].extend(file_pin(runtime.output / 'adapter' / name)
                                            for name in ('adapter_config.json', 'adapters.safetensors'))
    runtime.receipt.update(case_ids=[row['id'] for row in blind], eos_token_ids=stops, greedy=True,
                           enable_thinking=False, max_new_tokens=maximum)


def startup_gate():
    fd = os.environ.pop('TRID_UI_START_FD', None)
    if fd is not None:
        descriptor = int(fd)
        try:
            require(os.read(descriptor, 1) == b'G', 'Guard startup authorization missing')
        finally:
            os.close(descriptor)
    # Enforce offline tokenizer/model access and disable Transformers' Torch backend before import.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES, required=True)
    args = parser.parse_args(argv)
    startup_gate()
    require(args.plan.is_absolute(), '--plan must be absolute')
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    runtime = Runtime(plan, args.phase, hashlib.sha256(raw).hexdigest())
    seconds = plan['limits'].get({'capacity': 'capacity_seconds', 'train': 'train_seconds',
                                 'baseline': 'evaluation_seconds', 'adapter': 'evaluation_seconds'}.get(args.phase))
    try:
        with Deadline(seconds, lambda: runtime.deadline('phase_deadline')) if seconds else contextlib.nullcontext():
            config = validate_plan(plan)
            runtime.prerequisites()
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(plan['model']['path'], local_files_only=True, trust_remote_code=False)
            train, blind = audit_inputs(plan, tokenizer)
            stops = eos_ids(tokenizer, config)
            if args.phase == 'audit':
                runtime.receipt.update(train_rows=train,
                                       blind_rows=[dict(id=row['id'], prompt_tokens=len(row['prompt_ids']),
                                                        rendered_prompt_sha256=row['rendered_prompt_sha256']) for row in blind],
                                       eos_token_ids=stops, native_template_no_thinking=True,
                                       answer_encoding='separate answer + tokenizer.eos_token',
                                       masked_prompt=True, truncation=False)
            else:
                configure_mlx(runtime)
                model = load_base(runtime)
                if args.phase in ('capacity', 'train'):
                    run_training(runtime, model, train, args.phase == 'capacity')
                else:
                    run_evaluation(runtime, model, tokenizer, blind, stops)
            runtime.check('phase_complete')
            runtime.finish('complete')
        return 0
    except Exception as exc:
        runtime.finish('failed', stop_reason='mlx_memory_limit' if isinstance(exc, MemoryError) else 'error',
                       error=f'{type(exc).__name__}: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
