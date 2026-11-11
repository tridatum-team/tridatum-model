"""CUDA FP32 derivative of the frozen preference trainer, sharing its P/T mathematics.

Cache and train require --cuda-runtime PATH --cuda-runtime-sha256 SHA. Generate that
manifest first with --inspect-runtime NEW_PATH in the same UUID-isolated environment.
There is no CPU fallback, CPU-cache conversion, alternate head, or optimizer resume.
All source/data/reference, DPO/replay, validation and checkpoint contracts of the
frozen trainer remain in force. CUDA math explicitly disables TF32. Cache shards,
checkpoint reports and final reports bind the derivative source and actual runtime.
load_timing uses time.monotonic; history elapsed_seconds includes loading once.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_qwen_preferences as P

T = P.T
torch = P.torch
TRAINER_ENTRY = 'scripts/train_qwen_preferences_cuda.py'
RUNTIME_SCHEMA = 'trid-cuda-runtime-v1'
GPU_UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def inspect_runtime():
    """Inspect actual CUDA hardware/packages, configure FP32, and bind all sources.

    Only nonvolatile GPU identity is included: memory usage and active PIDs belong
    in the independent guard receipt, not a reusable runtime identity.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not GPU_UUID.fullmatch(visible):
        raise ValueError('CUDA_VISIBLE_DEVICES must be one explicit full GPU UUID')
    if platform.machine() != 'x86_64':
        raise ValueError('CUDA training requires x86_64')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.version.cuda:
        raise ValueError('exactly one actual CUDA device is required; no CPU fallback')
    from qwen_linux_guard import gpu_metrics
    observed = gpu_metrics(visible)
    gpu = {key: observed[key] for key in ('uuid', 'name', 'pci_bus_id', 'total_bytes', 'driver_version')}
    if gpu['uuid'] != visible or torch.cuda.get_device_name(0) != gpu['name']:
        raise ValueError('CUDA device identity differs from the UUID-bound GPU observation')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    settings = dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision())
    if settings != dict(cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, float32_matmul_precision='highest'):
        raise ValueError('CUDA FP32 settings could not be applied')
    import peft
    import transformers
    versions = dict(torch=str(torch.__version__), transformers=transformers.__version__, peft=peft.__version__)
    engine = P.engine_identity(versions, 'cuda', torch.float32)
    engine['code'][Path(__file__).name] = T.sha256(Path(__file__).resolve())
    engine.update(trainer_entry=TRAINER_ENTRY, architecture='x86_64', cuda_runtime=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(), gpu=gpu, math=settings)
    return dict(schema=RUNTIME_SCHEMA, trainer_entry=TRAINER_ENTRY, device='cuda', dtype=str(torch.float32),
                architecture='x86_64', versions=versions, gpu_uuid=visible, gpu=gpu, engine=engine)


def authenticate_runtime(path, expected_sha256):
    """Authenticate the exact manifest bytes and current identity before loading weights."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('--cuda-runtime-sha256 must be a lowercase SHA256')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('CUDA runtime manifest SHA256 mismatch')
    expected = json.loads(raw)
    actual = inspect_runtime()
    # JSON type identity matters: Python equality alone equates False with 0.
    if json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(actual, sort_keys=True, allow_nan=False):
        raise ValueError('CUDA runtime manifest differs from the actual source/package/device/math identity')
    return actual


def write_json(path, payload):
    """Reports are immutable; never overwrite a pre-existing output."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--inspect-runtime':
        parser = argparse.ArgumentParser(description='Write an exclusive CUDA runtime identity manifest')
        parser.add_argument('--inspect-runtime', type=Path, required=True)
        args = parser.parse_args(argv)
        if args.inspect_runtime.exists():
            parser.error('Use a new --inspect-runtime path')
        runtime = inspect_runtime()
        args.inspect_runtime.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.inspect_runtime, runtime)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cache', 'train'], required=True)
    p.add_argument('--cuda-runtime', type=Path, required=True)
    p.add_argument('--cuda-runtime-sha256', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--base-gguf', type=Path, required=True)
    p.add_argument('--base-gguf-sha256', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--source-dropout', type=float, default=0.05)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--ids', nargs='+')
    p.add_argument('--cache', type=Path, action='append', default=[])
    p.add_argument('--cache-sha256', action='append', default=[])
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--batch-pairs', type=int, default=4)
    p.add_argument('--replay-micro-batch', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--valid-every', type=int, default=50)
    p.add_argument('--checkpoint-every', type=int)
    p.add_argument('--max-steps', type=int)
    a = p.parse_args(argv)
    if a.out.exists():
        p.error('Use a new --out path')
    if not (math.isfinite(a.beta) and a.beta > 0 and math.isfinite(a.lr) and a.lr > 0):
        p.error('--beta and --lr must be finite and > 0')
    if a.valid_every < 1:
        p.error('--valid-every must be >= 1')
    if a.mode == 'train':
        if not a.checkpoint_every or a.checkpoint_every < 1:
            p.error('--mode train requires --checkpoint-every >= 1')
        if a.max_steps is not None and a.max_steps < 1:
            p.error('--max-steps must be >= 1')
        if a.batch_pairs < 1 or a.replay_micro_batch < 1:
            p.error('--batch-pairs and --replay-micro-batch must be >= 1')
        if not a.cache or len(a.cache) != len(a.cache_sha256):
            p.error('--mode train requires --cache with corresponding --cache-sha256')
        if a.ids:
            p.error('--ids applies to --mode cache only')
    elif a.cache or a.cache_sha256 or a.max_steps is not None or a.checkpoint_every is not None:
        p.error('--cache/--cache-sha256/--max-steps/--checkpoint-every apply to train only')
    started = time.monotonic()
    runtime = authenticate_runtime(a.cuda_runtime, a.cuda_runtime_sha256)
    device, dtype = 'cuda', torch.float32
    versions = runtime['versions']
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    base = a.manifest.resolve().parent
    manifest = json.loads(a.manifest.read_text())
    manifest_sha256 = T.sha256(a.manifest)
    ref = manifest['reference']
    source_manifest_path = P.pinned_file(ref['source_manifest'], base, 'source manifest')
    source_manifest = json.loads(source_manifest_path.read_text())
    checkpoint = P.resolve(ref['checkpoint'], base)
    source_lora = dict(rank=a.rank, alpha=a.alpha, dropout=a.source_dropout, targets=T.TARGETS)
    ancestry = T.init_adapter_ancestry(checkpoint / 'adapter', a.model_id, a.model_revision, a.base_gguf_sha256,
                                     ref['source_manifest']['sha256'], source_lora)
    for key in ('adapter_weights_sha256', 'adapter_config_sha256', 'training_json_sha256'):
        if ref.get(key) != ancestry[key]:
            raise ValueError(f'manifest reference.{key} {ref.get(key)!r} != {ancestry[key]} on disk')
    reference = dict(checkpoint=str(checkpoint), adapter=str(checkpoint / 'adapter'),
                     source_manifest=dict(path=str(source_manifest_path), sha256=ref['source_manifest']['sha256'],
                                          id=source_manifest.get('id')),
                     adapter_weights_sha256=ancestry['adapter_weights_sha256'],
                     adapter_config_sha256=ancestry['adapter_config_sha256'],
                     training_json_sha256=ancestry['training_json_sha256'],
                     source_steps=ancestry['source_steps'], source_cumulative_steps=ancestry['source_cumulative_steps'],
                     source_lora=source_lora)

    def valid_key(entry, root):
        return (str(P.resolve(entry['path'], root).resolve()), entry['sha256'], entry['adapter'], entry.get('split'))

    valid_entries = manifest.get('valid', [])
    if [valid_key(e, base) for e in valid_entries] != [valid_key(e, source_manifest_path.parent)
                                                   for e in source_manifest.get('valid', [])]:
        raise ValueError('manifest valid entries are not exactly the source manifest valid entries')
    replay = P.replay_contract(manifest, base, source_manifest_path, ref['source_manifest']['sha256'])
    tok = P.AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tokenizer = P.tokenizer_identity(a.model, tok)
    source, ambiguous = P.source_rows(source_manifest, source_manifest_path.parent)
    pairs = P.load_pairs(manifest, base, source, ambiguous)
    encoded = {pair['id']: P.encode_pair(tok, pair, a.max_tokens) for pair in pairs}
    digests = {pid: P.pair_digest(encoded[pid]) for pid in encoded}
    replay_encoded = {}
    for row in P.replay_rows(source_manifest, source_manifest_path.parent):
        item = T.encode(tok, row, P.FAMILY, a.max_tokens)
        if item is None:
            raise ValueError(f'replay source row {row["id"]!r} ({row["file"]}) exceeds --max-tokens {a.max_tokens}; '
                             'every canonical source TRAIN row must be replayed, nothing is dropped')
        replay_encoded[row['id']] = item
    replay_record = dict(replay, kind=P.OBJECTIVE_KIND, seed=a.seed + 1, rows_admitted=len(replay_encoded),
                         source_files=[{k: e.get(k) for k in ('path', 'sha256', 'adapter', 'split')}
                                       for e in source_manifest['train']],
                         rows_by_file={f: sum(1 for x in replay_encoded.values() if x['file'] == f)
                                       for f in sorted({x['file'] for x in replay_encoded.values()})},
                         token_label_sha256=P.replay_digest(list(replay_encoded.values())))
    base_identity = dict(model_id=a.model_id, revision=a.model_revision, gguf_sha256=a.base_gguf_sha256)
    bindings = dict(kind=P.CACHE_KIND, manifest_sha256=manifest_sha256, reference=reference, base=base_identity,
                    tokenizer=tokenizer, family=P.FAMILY, max_tokens=a.max_tokens, objective=P.OBJECTIVE,
                    engine=runtime['engine'], cuda_runtime=runtime, cuda_runtime_sha256=a.cuda_runtime_sha256)
    if a.mode == 'cache':
        ids = list(encoded)
        if a.ids:
            unknown, counts = sorted(set(a.ids) - set(encoded)), {}
            for pid in a.ids:
                counts[pid] = counts.get(pid, 0) + 1
            duplicates = sorted(pid for pid, count in counts.items() if count > 1)
            if unknown or duplicates:
                raise ValueError(f'--ids: not admitted pairs {unknown[:10]}, repeated {duplicates[:10]}')
            ids = list(a.ids)
    else:
        shards = []
        for path, expected_sha in zip(a.cache, a.cache_sha256):
            actual = T.sha256(path)
            if actual != expected_sha:
                raise ValueError(f'--cache {path}: sha256 {actual} != {expected_sha}')
            shards.append(json.loads(path.read_text()))
        cached = P.verify_cache_shards(shards, bindings, digests)
        cache_record = [dict(path=str(path), sha256=sha, ids=len(shard['ids']))
                        for path, sha, shard in zip(a.cache, a.cache_sha256, shards)]
        valid = []
        for entry in valid_entries:
            for row in T.load_rows(entry, base):
                item = T.encode(tok, row, P.FAMILY, a.max_tokens)
                if item is None:
                    raise ValueError(f'valid row {row.get("id")!r} ({row["file"]}) exceeds --max-tokens {a.max_tokens}; '
                                     'every source valid row must stay represented')
                valid.append(item)
        replay_bounds = P.replay_partition(len(replay_encoded), math.ceil(len(encoded) / a.batch_pairs))

    from peft import LoraConfig, get_peft_model
    from q8_gguf_base import base_weights_record, load_qwen3_q8_gguf
    load_start = time.monotonic()
    print(json.dumps(dict(event='load_start', monotonic=load_start, cuda_runtime_sha256=a.cuda_runtime_sha256)), flush=True)
    model, loading = load_qwen3_q8_gguf(a.model, a.base_gguf, a.base_gguf_sha256, dtype, device)
    base_weights = base_weights_record(loading)
    print(json.dumps({'weight_loading': loading}), flush=True)
    lora = dict(rank=a.rank, alpha=a.alpha, dropout=0.0, targets=T.TARGETS)
    if a.mode == 'train':
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0,
                                           target_modules=T.TARGETS, task_type='CAUSAL_LM'))
    ancestry['tensors_loaded'] = T.load_init_adapter(model, checkpoint / 'adapter', device)
    # Packed Q8 storage intentionally contains int8/fp16; trainable LoRA must be CUDA FP32.
    for name, param in model.named_parameters():
        if param.device.type != 'cuda' or (param.requires_grad and param.dtype != dtype):
            raise ValueError(f'loaded parameter {name} violates CUDA/FP32 training placement')
    torch.cuda.synchronize()
    load_timing = dict(load_start_monotonic=load_start, load_end_monotonic=time.monotonic(),
                       elapsed_origin_monotonic=started)
    print(json.dumps({'init_adapter': ancestry, 'load_timing': load_timing}, ensure_ascii=False), flush=True)

    if a.mode == 'cache':
        model.eval()
        scores = {}
        with torch.no_grad():
            for n, pid in enumerate(ids, 1):
                chosen, rejected = (float(v) for v in P.pair_logprobs(model, encoded[pid], device, pad_id).tolist())
                if not (math.isfinite(chosen) and math.isfinite(rejected)):
                    raise FloatingPointError(f'non-finite reference score for pair {pid!r}; nothing written')
                scores[pid] = dict(chosen=chosen, rejected=rejected, pair_sha256=digests[pid])
                print(json.dumps(dict(cached=pid, n=n, of=len(ids), chosen=chosen, rejected=rejected,
                                      elapsed_seconds=time.monotonic() - started), ensure_ascii=False), flush=True)
        shard = dict(bindings, manifest=str(a.manifest), manifest_id=manifest.get('id'), base_weights=base_weights,
                     init_adapter=ancestry, admitted_pairs=len(encoded), ids=ids, scores=scores, seed=a.seed,
                     replay=replay_record, device=device, dtype=str(dtype), versions=versions,
                     load_timing=load_timing, elapsed_seconds=time.monotonic() - started, status='complete')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(a.out, shard)
        print(f'cache complete: {len(ids)} pairs -> {a.out} sha256 {T.sha256(a.out)}', flush=True)
        return

    order = list(encoded)
    random.shuffle(order)
    batches_full = P.pair_batches(order, a.batch_pairs)
    planned = len(batches_full)
    steps = min(planned, a.max_steps) if a.max_steps else planned
    batches = batches_full[:steps]
    replay_order = list(replay_encoded)
    random.Random(a.seed + 1).shuffle(replay_order)
    replay_batches = [replay_order[start:stop] for start, stop in replay_bounds][:steps]
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    model.eval()
    initial = []
    with torch.no_grad():
        for pid in batches[0]:
            chosen, rejected = P.pair_logprobs(model, encoded[pid], device, pad_id).tolist()
            initial.append(dict(id=pid, policy=dict(chosen=chosen, rejected=rejected),
                                reference=dict(chosen=cached[pid][0], rejected=cached[pid][1])))
    initial_check = P.check_initial_agreement(initial, a.beta)
    print(json.dumps({'initial_check': initial_check}, ensure_ascii=False), flush=True)
    model.train()
    history, checkpoints = [], []
    replay_plan = dict(replay_record, full_epoch_steps=planned, planned_steps=steps,
                       rows_planned=sum(len(b) for b in replay_batches), rows_per_step=[len(b) for b in replay_batches],
                       partition='batch i = shuffled_order[floor(i*N/S):floor((i+1)*N/S)], S = full-epoch pair steps; '
                                 '--max-steps keeps a prefix of these batches',
                       order_sha256=hashlib.sha256(json.dumps(replay_order).encode()).hexdigest(),
                       order_sha256_over='the random.Random(seed + 1) shuffled assignment order of source row ids, i.e. '
                                         'which rows each update carries, BEFORE the within-update length sorting below',
                       micro_batch=a.replay_micro_batch,
                       micro_batch_order='within each update the assigned rows are stable-sorted by encoded input length '
                                         '(ties keep assignment order) and chunked into consecutive micro-batches of at '
                                         'most micro_batch rows; rows never move across updates',
                       micro_batches_per_step=[len(P.replay_microbatches([(rid, replay_encoded[rid]) for rid in b],
                                                                       a.replay_micro_batch)) for b in replay_batches],
                       loss='per row the answer+EOS token-mean cross-entropy; each micro-batch backpropagates weight * '
                            'sum of its rows / rows of the update, so replay_ce is the mean over all rows of the update '
                            'whatever the micro-batch size')
    objective = dict(kind=P.OBJECTIVE_KIND, beta=a.beta, reference_scores=P.OBJECTIVE, reference=reference, cache=cache_record,
                     sft_auxiliary_loss=None, replay=replay_plan, planned_epochs=1, pairs_admitted=len(order),
                     batch_pairs=a.batch_pairs, replay_micro_batch=a.replay_micro_batch,
                     micro_batch='one pair (two sequences) per forward, then the update\'s replay rows length-sorted in '
                                 'micro-batches of at most replay_micro_batch rows per forward',
                     final_partial_batch='pairs divided by the actual pair count, replay rows by the actual row count of '
                                         'the update (a short final micro-batch contributes only its rows)',
                     train_loss='preference_loss + replay.weight * replay_ce', grad_clip=1.0)
    a.out.mkdir(parents=True)
    (a.out / 'checkpoints').mkdir()

    def consumed_through(step):
        return sum(len(b) for b in batches[:step])

    def replay_consumed_through(step):
        return sum(len(b) for b in replay_batches[:step])

    def consumption(step):
        consumed, replayed = consumed_through(step), replay_consumed_through(step)
        return dict(pairs_admitted=len(order), pairs_planned=consumed_through(steps), pairs_consumed_through_step=consumed,
                    epoch_fraction_consumed=round(consumed / len(order), 4), full_epoch_steps=planned, planned_steps=steps,
                    bounded_by_max_steps=steps < planned, epoch_status='complete' if consumed == len(order) else 'partial',
                    replay_rows_admitted=len(replay_order), replay_rows_planned=replay_consumed_through(steps),
                    replay_rows_consumed_through_step=replayed,
                    replay_epoch_fraction_consumed=round(replayed / len(replay_order), 4),
                    replay_epoch_status='complete' if replayed == len(replay_order) else 'partial')

    identity = dict(engine=runtime['engine'], cuda_runtime=runtime, cuda_runtime_sha256=a.cuda_runtime_sha256,
                    load_timing=load_timing)

    def report(status, **extra):
        payload = dict(
            protocol=__doc__, base_model_id=a.model_id, base_revision=a.model_revision, family=P.FAMILY,
            manifest=str(a.manifest), manifest_sha256=manifest_sha256, manifest_id=manifest.get('id'),
            files=[{k: e.get(k) for k in ('path', 'sha256', 'adapter', 'epochs', 'split')}
                   for e in manifest['train'] + valid_entries], replay_files=replay_record['source_files'],
            train_rows=len(order), train_pairs_admitted=len(order), replay_rows=len(replay_order), valid_rows=len(valid),
            task_counts={t: sum(1 for pid in order if encoded[pid]['chosen']['task'] == t)
                         for t in sorted({encoded[pid]['chosen']['task'] for pid in order})},
            steps=steps, planned_steps=steps, batch_size=a.batch_pairs, micro_batch_size=1,
            replay_micro_batch_size=a.replay_micro_batch, lr=a.lr, beta=a.beta, lora=lora,
            max_tokens=a.max_tokens, seed=a.seed, device=device, dtype=str(dtype), base_weights=base_weights,
            init_adapter=ancestry, reference=reference, objective=objective, tokenizer=tokenizer,
            initial_check=initial_check, versions=versions, history=list(history),
            elapsed_seconds=time.monotonic() - started, status=status, selection=T.SELECTION, **identity)
        payload.update(extra)
        payload['cumulative_steps'] = ancestry['source_cumulative_steps'] + payload['steps']
        return payload

    def ledger(run_status, **extra):
        # The ledger is the sole mutable index inside this exclusively created run.
        payload = dict(run_status=run_status, planned_steps=steps, checkpoint_every=a.checkpoint_every,
                       base_model_id=a.model_id, base_revision=a.model_revision, manifest_sha256=manifest_sha256,
                       base_weights=base_weights, init_adapter=ancestry, reference=reference, objective=objective,
                       checkpoints=list(checkpoints), selection=T.SELECTION, **identity, **extra)
        (a.out / 'checkpoints.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    def save_checkpoint(step, record):
        folder = a.out / 'checkpoints' / f'step-{step:06d}'
        model.save_pretrained(folder / 'adapter')
        checkpoints.append(dict(step=step, path=str(folder / 'adapter'),
                                adapter_weights_sha256=T.sha256(folder / 'adapter' / 'adapter_model.safetensors'),
                                train_loss=record['train_loss'], preference_loss=record['preference_loss'],
                                replay_ce=record['replay_ce'], reward_margin=record['reward_margin'],
                                valid_loss_by_task=record.get('valid_loss_by_task'), pairs_consumed=consumed_through(step),
                                replay_rows_consumed=replay_consumed_through(step), elapsed_seconds=record['elapsed_seconds']))
        write_json(folder / 'training.json', report('checkpoint', steps=step, planned_steps=steps,
                                                   checkpoint=checkpoints[-1], consumption=consumption(step)))
        ledger('in_progress')
        print(json.dumps({'checkpoint': checkpoints[-1]}, ensure_ascii=False), flush=True)

    for step, (batch, replay_batch) in enumerate(zip(batches, replay_batches), 1):
        optimizer.zero_grad(set_to_none=True)
        try:
            numbers = P.accumulate_step(model, [(pid, encoded[pid], cached[pid]) for pid in batch],
                                        [(rid, replay_encoded[rid]) for rid in replay_batch], a.beta, replay['weight'],
                                        device, pad_id, replay_micro_batch=a.replay_micro_batch)
        except FloatingPointError as exc:
            raise FloatingPointError(f'{exc} at step {step}; stopping without saving') from None
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        # CUDA optimizer launches are asynchronous: timing must include their actual completion.
        torch.cuda.synchronize()
        checkpoint_step = step % a.checkpoint_every == 0 or step == steps
        record = dict(step=step, train_loss=round(numbers['train_loss'], 5),
                      preference_loss=round(numbers['preference_loss'], 5), replay_ce=round(numbers['replay_ce'], 5),
                      replay_weight=replay['weight'], reward_margin=round(numbers['reward_margin'], 5),
                      preferred_fraction=round(numbers['preferred_fraction'], 4), pairs=len(batch),
                      replay_rows=len(replay_batch), replay_micro_batches=numbers['replay_micro_batches'],
                      lr=schedule.get_last_lr()[0], elapsed_seconds=time.monotonic() - started)
        if valid and (step % a.valid_every == 0 or step == steps or checkpoint_step):
            record['valid_loss_by_task'] = T.validate(model, valid, device, pad_id, 1, P.FAMILY)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if checkpoint_step:
            save_checkpoint(step, record)
    model.save_pretrained(a.out / 'adapter')
    last = dict(step=steps, path=str(a.out / 'adapter'),
                adapter_weights_sha256=T.sha256(a.out / 'adapter' / 'adapter_model.safetensors'))
    last['equals_last_checkpoint'] = last['adapter_weights_sha256'] == checkpoints[-1]['adapter_weights_sha256']
    ledger('complete', last_step=last)
    write_json(a.out / 'training.json', report('complete', checkpoints=checkpoints, last_step=last,
                                             consumption=consumption(steps)))
    print(f'complete: {steps} steps, {consumed_through(steps)} of {len(order)} admitted pairs and '
          f'{replay_consumed_through(steps)} of {len(replay_order)} replay rows consumed -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
