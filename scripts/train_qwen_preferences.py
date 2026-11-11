"""DPO (direct preference optimisation) plus canonical-source replay of the TRID Qwen3 LoRA from a pinned pair manifest.

Both the reference and the policy are the SAME pinned, already-trained SFT checkpoint (never the bare base, never zero
adapters). One model exists per process, in one of two modes:

  --mode cache   loads the reference checkpoint, scores every admitted pair (or the explicit --ids subset) and writes one
                 immutable cache shard: summed answer log-probabilities (answer tokens + EOS, NOT length-normalised),
                 model.eval()/no_grad, one padded [chosen, rejected] pair at a time through the existing answer-only
                 shifted cross-entropy with chunked head (train_open_weight_lora.chunked_answer_loss).
  --mode train   verifies the cache shards against the manifest, the reference pins, the tokenizer identity and the
                 encoded pair token/label hashes BEFORE any weight is loaded (exact id coverage, no duplicates, finite
                 scores), then loads the Q8 base + the reference adapter as the trainable policy and optimises, per update,
                   mean over the batch's pairs of -logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)))
                   + replay.weight * mean over the update's replay rows of the answer+EOS token-mean cross-entropy
                 with fresh AdamW + warmup/cosine, sequential micro-batches on the one model (one pair = two sequences
                 per forward; then the update's replay rows, stable-sorted by encoded length and chunked into at most
                 --replay-micro-batch rows per forward, default 4), separately normalised gradients accumulated (the
                 final partial pair batch is divided by its actual pair count; every replay micro-batch backpropagates
                 the sum of ITS rows' per-example token-mean cross-entropy divided by the update's total replay rows,
                 so a short final micro-batch and a long answer never weigh more), gradient clipping 1.0 once, one
                 optimizer update. One epoch of pairs and exactly one pass over the canonical
                 source TRAIN rows. A missing or stale cache is an error, never a recomputation. --replay-micro-batch
                 is computational batching only: which rows an update carries and the objective are identical for any
                 value >= 1 (1 = one row per forward, the profiling/testing setting).

Isolated environment (the project venv pins transformers 5.x and has no peft), as train_open_weight_lora.py:

  uv run --no-project --python 3.12 --with 'torch==2.9.1' --with 'transformers==4.57.6' --with 'peft==0.17.1' \
      --with 'safetensors==0.8.0' --with 'gguf==0.19.0' --with accelerate --with numpy --with tokenizers \
      python scripts/train_qwen_preferences.py --mode cache --model <hf dir> --model-id Qwen/Qwen3-1.7B \
      --model-revision <sha> --base-gguf <q8_0.gguf> --base-gguf-sha256 <hex> --manifest <pairs.json> \
      --out <cache-shard.json> [--ids ID ...]
  ... --mode train ... --out runs/<name> --cache <shard.json> --cache-sha256 <hex> [--cache ... --cache-sha256 ...] \
      --checkpoint-every N [--max-steps N] [--replay-micro-batch N]

Pair manifest (JSON): {"id", "train": [{"path", "sha256", "adapter": "preference_pair", "split": "train", "epochs": 1}],
"valid": [the SOURCE manifest's valid entries, unchanged], "reference": {"checkpoint": <dir with training.json and
adapter/>, "source_manifest": {"path", "sha256"}, "adapter_weights_sha256", "adapter_config_sha256",
"training_json_sha256"}, "replay": {"source_manifest": {"path", "sha256"}, "weight", "epochs": 1}}. Paths resolve
relative to the manifest. The `replay` block is REQUIRED in both modes (no default, no implicit replay): its source
manifest must be the same resolved file and sha256 as reference.source_manifest, weight finite and > 0, epochs exactly
1; it is checked before any weight is loaded. Replay rows are ALL canonical source TRAIN rows (every train entry of the
source manifest, as train_open_weight_lora.load_rows normalises them, exact system/user/history/answer, file
provenance kept; source `epochs` ignored), each exactly once per full run, no exclusion or oversampling; a missing id
or an id occurring twice in the source (identical or not) is an error. Valid rows are never replayed. Replay order is
an independent random.Random(seed + 1) shuffle of source manifest order, partitioned into the full-epoch number of pair
steps S = ceil(pairs / --batch-pairs) as batch i = order[floor(i*N/S):floor((i+1)*N/S)] (nonempty, N >= S required);
--max-steps takes a prefix of these full-epoch batches, so a smoke run consumes a prefix of the replay rows, never all.
The source manifest is the SFT manifest whose TRAIN
rows the pairs come from; init_adapter_ancestry is checked against the SOURCE manifest hash (the reference adapter was
trained on it) with the source LoRA shape (--source-dropout, default 0.05), the reference pins must equal the hashes
on disk, and the pair manifest's valid entries must be exactly the source manifest's valid entries (same resolved
file, sha256, adapter, split). Pair rows (JSONL): {"id", "task", "system"?, "user", "history"?, "chosen",
"rejected"}; id/task/system/user/history (system None and history [] when absent) must equal the source TRAIN row
of that id as train_open_weight_lora.load_rows normalises it, and chosen must equal that row's answer exactly. Duplicate
ids, identical or empty answers, missing/invalid fields and ambiguous source ids are errors; extra provenance fields are
ignored and never become prompts. Every chosen and rejected sequence (native template + answer + EOS) must fit
--max-tokens (default 1024, the explicit planned cap); an over-length pair, replay or valid row is an error, nothing is
dropped. The policy's LoRA dropout is 0.0 (recorded as the effective lora.dropout; the source checkpoint's 0.05 under
reference.source_lora); the frozen base is the serving Q8_0 GGUF (q8_gguf_base), CPU float32.

Outputs (train): <out>/checkpoints/step-NNNNNN/adapter + training.json at every --checkpoint-every step and the last
step, <out>/checkpoints.json, and <out>/adapter + <out>/training.json at the end, in the layout train_open_weight_lora
writes (status "checkpoint"/"complete", `checkpoint`/`last_step` with adapter_weights_sha256, `steps` = updates of this
run, `cumulative_steps` = source cumulative + steps, `init_adapter` = the reference ancestry, `manifest_sha256` = THIS
pair manifest; the source manifest hash is under `reference.source_manifest`). history[*].valid_loss_by_task is the
original per-task answer cross-entropy (train_open_weight_lora.validate) on the unchanged source valid rows;
history[*].train_loss is preference_loss + replay.weight * replay_ce (`preference_loss` = mean DPO loss, `replay_ce` =
mean over the update's replay rows of the per-row cross-entropy, `replay_rows` = rows in that update,
`replay_micro_batches` = forwards they took) with `reward_margin` (mean policy-vs-reference margin).
Before the first update the policy's pair log-probabilities of the first batch are compared with the cache
(`initial_check`: the policy IS the reference, so a max absolute error above INITIAL_CACHE_MAX_ABS_ERROR or any
non-finite value stops the run before any update; the observed error and margins are recorded). Cache shards also
bind the scoring engine (the packed-supervised-token head layout, library versions, device/dtype, sha256 of this
file, train_open_weight_lora.py and q8_gguf_base.py): any code change invalidates them, nothing recomputes implicitly.
Every training.json records
`consumption` (admitted pairs, pairs planned under --max-steps, pairs consumed through the saved step, epoch status
partial/complete, and the same for replay rows): a --max-steps smoke or an intermediate checkpoint has seen a prefix
of the shuffled pairs and a prefix of the replay rows only. objective.kind is "dpo_with_canonical_replay" with the
replay weight, seed, source pin, admitted/planned rows, the sha256 of the replay order and of the exact replay
tokens/labels. No optimizer state is saved; a run never overwrites an existing --out or cache file.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_open_weight_lora as T  # noqa: E402
from trid.text import validate_history  # noqa: E402

FAMILY = 'qwen3'
CACHE_KIND = 'preference_reference_cache'
OBJECTIVE = 'summed_answer_logprob_including_eos'
OBJECTIVE_KIND = 'dpo_with_canonical_replay'
PAIR_ADAPTER = 'preference_pair'
PAIR_FIELDS = ('id', 'task', 'system', 'user', 'history')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
                   'added_tokens.json')


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def pinned_file(entry, base, what):
    """The file an {path, sha256} entry pins, hash verified."""
    path = resolve(entry['path'], base)
    actual = T.sha256(path)
    if actual != entry['sha256']:
        raise ValueError(f'{what} {path}: sha256 {actual} != pinned {entry["sha256"]}')
    return path


def tokenizer_identity(model_dir, tok):
    """What the cache binds the encoding to: the tokenizer files of the model directory and the special ids."""
    model_dir = Path(model_dir)
    files = {name: T.sha256(model_dir / name) for name in TOKENIZER_FILES if (model_dir / name).is_file()}
    if not files:
        raise ValueError(f'{model_dir}: no tokenizer files to bind the cache to')
    return dict(files=files, vocab_size=len(tok), eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)


def normalized_pair(row, path, line_number):
    """A pair row's identity fields as the source rows are normalised, or ValueError naming file/line/id."""
    def refuse(why):
        raise ValueError(f'{path}:{line_number}: row {row.get("id")!r}: {why}')
    for key in ('id', 'task', 'user', 'chosen', 'rejected'):
        if not isinstance(row.get(key), str) or not row[key]:
            refuse(f'{key} must be a nonempty string')
    system = row.get('system')
    if system is not None and (not isinstance(system, str) or not system):
        refuse('system must be a nonempty string or absent')
    history = row.get('history', [])
    try:
        validate_history(history)
    except ValueError as exc:
        refuse(str(exc))
    if row['chosen'] == row['rejected']:
        refuse('chosen and rejected are identical')
    return dict(id=row['id'], task=row['task'], system=system, user=row['user'], history=history,
                chosen=row['chosen'], rejected=row['rejected'], file=str(path))


def source_rows(source_manifest, base):
    """id -> normalised source TRAIN row (train_open_weight_lora.load_rows, epochs ignored); ids whose rows differ
    between occurrences are recorded as ambiguous."""
    rows, ambiguous = {}, set()
    for entry in source_manifest['train']:
        for row in T.load_rows(entry, base):
            key = {k: row[k] for k in PAIR_FIELDS + ('answer',)}
            if row['id'] in rows and rows[row['id']] != key:
                ambiguous.add(row['id'])
            rows[row['id']] = key
    return rows, ambiguous


def load_pairs(manifest, base, source, ambiguous):
    """Admitted pairs in manifest order, each proven to be a source TRAIN row plus a rejected answer."""
    pairs, seen, problems = [], set(), []
    for entry in manifest['train']:
        if entry.get('adapter') != PAIR_ADAPTER:
            raise ValueError(f'train entry {entry.get("path")}: adapter must be {PAIR_ADAPTER!r}')
        if entry.get('epochs', 1) != 1:
            raise ValueError(f'train entry {entry.get("path")}: epochs must be 1 (one pass over the pairs)')
        path = pinned_file(entry, base, 'pair file')
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if entry.get('split') and row.get('split') != entry['split']:
                continue
            pair = normalized_pair(row, path, line_number)
            where = f'{path}:{line_number}: row {pair["id"]!r}'
            if pair['id'] in seen:
                problems.append(f'{where}: duplicate pair id')
            seen.add(pair['id'])
            if pair['id'] in ambiguous:
                problems.append(f'{where}: ambiguous source id (differing source rows share it)')
            elif pair['id'] not in source:
                problems.append(f'{where}: not a source TRAIN row')
            else:
                src = source[pair['id']]
                differing = [k for k in PAIR_FIELDS if pair[k] != src[k]]
                if differing:
                    problems.append(f'{where}: differs from the source row in {differing}')
                if pair['chosen'] != src['answer']:
                    problems.append(f'{where}: chosen is not the source answer')
            pairs.append(pair)
    if problems:
        raise ValueError(f'{len(problems)} pair rows are not admitted: {problems[:20]}')
    if not pairs:
        raise ValueError('No preference pairs')
    return pairs


def replay_contract(manifest, base, source_manifest_path, source_sha256):
    """The REQUIRED manifest `replay` block, normalised: {"source_manifest": {path, sha256} naming exactly the
    reference's source manifest (same resolved file, same sha256), "weight": finite > 0, "epochs": 1}. An absent or
    deviating block is a ValueError before any weight is loaded; there is no default and no implicit replay."""
    replay = manifest.get('replay')
    if not isinstance(replay, dict):
        raise ValueError('manifest.replay is required: {"source_manifest": {"path", "sha256"}, "weight": > 0, "epochs": 1}')
    pin = replay.get('source_manifest')
    if not isinstance(pin, dict) or not isinstance(pin.get('path'), str) or not isinstance(pin.get('sha256'), str):
        raise ValueError('manifest.replay.source_manifest must be {"path", "sha256"}')
    path = resolve(pin['path'], base).resolve()
    if path != Path(source_manifest_path).resolve():
        raise ValueError(f'manifest.replay.source_manifest {path} is not reference.source_manifest {source_manifest_path}')
    if pin['sha256'] != source_sha256:
        raise ValueError(f'manifest.replay.source_manifest sha256 {pin["sha256"]} != reference.source_manifest {source_sha256}')
    weight = replay.get('weight')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f'manifest.replay.weight must be a finite number > 0, not {weight!r}')
    epochs = replay.get('epochs')
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 1:
        raise ValueError(f'manifest.replay.epochs must be exactly 1, not {epochs!r}')
    return dict(source_manifest=dict(path=str(path), sha256=pin['sha256']), weight=float(weight), epochs=1)


def replay_rows(source_manifest, base):
    """Every canonical source TRAIN row exactly once, in source manifest order (train entries, then lines), as
    train_open_weight_lora.load_rows normalises it (system/user/history/answer, file provenance kept). The source
    entries' `epochs` are ignored: replay.epochs governs. A missing id or an id that occurs twice (identical rows or
    not) is an error: nothing is collapsed, excluded or oversampled."""
    rows, seen, problems = [], {}, []
    for entry in source_manifest['train']:
        for row in T.load_rows(entry, base):
            rid = row['id']
            where = f'{row["file"]}: row {rid!r}'
            if not isinstance(rid, str) or not rid:
                problems.append(f'{where}: missing id')
            elif rid in seen:
                problems.append(f'{where}: duplicate source id (first in {seen[rid]})')
            else:
                seen[rid] = row['file']
            rows.append(row)
    if problems:
        raise ValueError(f'{len(problems)} source TRAIN rows cannot be replayed: {problems[:20]}')
    if not rows:
        raise ValueError('No source TRAIN rows to replay')
    return rows


def replay_digest(items):
    """sha256 over the exact token ids and labels of the replay sequences, in the given (source manifest) order."""
    return hashlib.sha256(json.dumps([[x['ids'], x['labels']] for x in items]).encode()).hexdigest()


def pair_batches(order, batch_pairs):
    """Consecutive --batch-pairs slices of the shuffled pair order: ceil(n / batch_pairs) batches, the last possibly
    partial, every pair in exactly one batch."""
    if batch_pairs < 1:
        raise ValueError('batch_pairs must be >= 1')
    return [order[i:i + batch_pairs] for i in range(0, len(order), batch_pairs)]


def replay_partition(n_rows, full_epoch_steps):
    """(start, stop) of replay batch i = [floor(i*N/S), floor((i+1)*N/S)) for i in range(S): S contiguous, disjoint,
    nonempty slices (sizes differ by at most one) covering all N rows exactly once. Fewer rows than full-epoch steps
    is an error: every update of the epoch carries replay. --max-steps takes a prefix of THESE batches, never a
    re-partition over fewer steps."""
    if full_epoch_steps < 1:
        raise ValueError('full_epoch_steps must be >= 1')
    if n_rows < full_epoch_steps:
        raise ValueError(f'{n_rows} replay rows cannot give {full_epoch_steps} full-epoch steps a nonempty replay batch each')
    return [(i * n_rows // full_epoch_steps, (i + 1) * n_rows // full_epoch_steps) for i in range(full_epoch_steps)]


def encode_pair(tok, pair, max_tokens):
    """{'chosen': item, 'rejected': item} with train_open_weight_lora.encode; over-length is an error."""
    encoded = {}
    for side in ('chosen', 'rejected'):
        row = dict(pair, answer=pair[side])
        item = T.encode(tok, row, FAMILY, max_tokens)
        if item is None:
            raise ValueError(f'pair {pair["id"]!r}: {side} exceeds --max-tokens {max_tokens}')
        encoded[side] = item
    return encoded


def pair_digest(encoded):
    """sha256 over the exact chosen/rejected token ids and labels the model scores."""
    payload = [encoded[side][key] for side in ('chosen', 'rejected') for key in ('ids', 'labels')]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def sequence_losses(model, items, device, pad_id, family=FAMILY):
    """(summed answer cross-entropy [n], supervised token count [n]) of `items` (train_open_weight_lora.encode dicts,
    any number, any lengths) under `model`, right-padded to the longest and scored in ONE forward: for row i,
    sum over its shifted answer positions (answer tokens + EOS; prompt, history and padding carry label -100 and are
    masked out) of the fp32 per-token cross-entropy, and the number of those positions (>= 1 for every encoded row).
    The qwen3 path sends only supervised row-position pairs through the chunked head (never [n, T, V] logits);
    `keep` remains the union of supervised positions as the coordinate mapping for targets and returned losses.
    Cross-entropy is position-wise, so the result equals the full-logits computation for each row independently of
    which other rows share the batch. Both pair scoring and replay micro-batches use this one function."""
    width = max(len(x['ids']) for x in items)
    n = len(items)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    labels = torch.full((n, width), -100, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    for i, x in enumerate(items):
        k = len(x['ids'])
        ids[i, :k] = torch.tensor(x['ids']); labels[i, :k] = torch.tensor(x['labels']); mask[i, :k] = 1
    target = labels[:, 1:].to(device)
    counts = (target != -100).sum(1)
    if family == 'qwen3':
        keep = (target != -100).any(0).nonzero().squeeze(1)
        target = target[:, keep]
        per_token = T.chunked_answer_loss(model, ids.to(device), mask.to(device), keep, target)
    else:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, :-1].float()
        per_token = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction='none')
    return per_token.sum(1), counts


def pair_logprobs(model, encoded, device, pad_id, family=FAMILY):
    """Summed log-probability of the answer tokens (EOS included) of [chosen, rejected] under `model`, as a
    differentiable tensor [2]. Padding, prompt and history positions are masked by the answer-only labels; the sum
    is the negative of the per-token cross-entropy (chunked head on the qwen3 path), not length-normalised."""
    return -sequence_losses(model, [encoded['chosen'], encoded['rejected']], device, pad_id, family)[0]


def dpo_loss(policy, reference, beta):
    """(loss, margin): policy = tensor [chosen, rejected] log-probs, reference = (chosen, rejected) cached floats."""
    margin = (policy[0] - policy[1]) - (reference[0] - reference[1])
    return -torch.nn.functional.logsigmoid(beta * margin), margin


def replay_microbatches(replay_batch, micro):
    """The rows of ONE update's replay batch [(id, encoded item)], stable-sorted by encoded input length (ids order
    breaks ties) and chunked into consecutive micro-batches of at most `micro` rows (the last possibly shorter). Pure
    computational batching: every row of the batch is in exactly one micro-batch, nothing crosses an update, and the
    objective (see accumulate_step) does not depend on the chunking. micro = 1 is one row per forward (the
    profiling/testing setting); micro < 1 is an error."""
    if micro < 1:
        raise ValueError('replay_micro_batch must be >= 1')
    ordered = sorted(replay_batch, key=lambda x: len(x[1]['ids']))
    return [ordered[i:i + micro] for i in range(0, len(ordered), micro)]


def accumulate_step(model, pair_batch, replay_batch, beta, weight, device, pad_id, family=FAMILY, replay_micro_batch=1):
    """Backward of one update's objective, accumulated over sequential micro-batches:
        mean over pair_batch of -logsigmoid(beta * margin)                 (one pair = two sequences per forward)
        + weight * mean over replay_batch of the answer+EOS token-mean cross-entropy of each row
          (replay_microbatches of at most replay_micro_batch rows per forward; each micro-batch backpropagates
           weight * sum of ITS rows' per-example token-mean CE / len(replay_batch), so every row weighs 1/len(replay_batch)
           whatever micro-batch it lands in, a short final micro-batch counts its rows only, and a long answer never
           weighs more than a short one).
    pair_batch = [(id, encoded pair, (ref_chosen, ref_rejected))], replay_batch = [(id, encoded item)], both nonempty.
    Gradients accumulate on the model; the caller zeroes before, clips once and steps once after. Returns the numbers
    of the record; raises FloatingPointError at the first non-finite loss, before any further micro-batch."""
    if not pair_batch or not replay_batch:
        raise ValueError('an update needs at least one pair and one replay sequence')
    micro_batches = replay_microbatches(replay_batch, replay_micro_batch)  # validated before any forward
    preference, margins, correct = 0.0, [], 0
    for pid, encoded, reference in pair_batch:
        loss, margin = dpo_loss(pair_logprobs(model, encoded, device, pad_id, family), reference, beta)
        loss = loss / len(pair_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite preference loss (pair {pid!r})')
        loss.backward()
        preference += loss.item(); margins.append(margin.item()); correct += margin.item() > 0
    replay_ce = 0.0
    for chunk in micro_batches:
        summed, counts = sequence_losses(model, [item for _, item in chunk], device, pad_id, family)
        per_example = summed / counts  # token-mean answer+EOS cross-entropy of each row (counts >= 1 by encoding)
        finite = torch.isfinite(per_example)
        if not bool(finite.all()):
            bad = [rid for (rid, _), ok in zip(chunk, finite.tolist()) if not ok]
            raise FloatingPointError(f'non-finite replay cross-entropy (source rows {bad!r})')
        share = per_example.sum() / len(replay_batch)
        (weight * share).backward()
        replay_ce += share.item()
    return dict(preference_loss=preference, replay_ce=replay_ce, train_loss=preference + weight * replay_ce,
                reward_margin=sum(margins) / len(margins), preferred_fraction=correct / len(pair_batch),
                pairs=len(pair_batch), replay_rows=len(replay_batch), replay_micro_batches=len(micro_batches))


def verify_cache_shards(shards, expected, digests):
    """Every shard (parsed JSON, in order) must carry the exact bindings in `expected` (kind, manifest_sha256,
    reference, base, tokenizer, family, max_tokens, objective, engine) and together cover exactly the ids of `digests`
    (id -> pair sha256) once, each with finite float chosen/rejected scores and the current pair digest.
    Returns id -> (chosen, rejected). Raises ValueError naming the first class of problem."""
    scores, problems = {}, []
    for index, shard in enumerate(shards):
        for key, want in expected.items():
            if shard.get(key) != want:
                problems.append(f'shard {index}: {key} differs from the current run')
        if shard.get('status') != 'complete':
            problems.append(f'shard {index}: status {shard.get("status")!r}')
        ids, recorded = shard.get('ids') or [], shard.get('scores') or {}
        if sorted(ids) != sorted(recorded):
            problems.append(f'shard {index}: ids and scores disagree')
        for pid in ids:
            if pid in scores:
                problems.append(f'shard {index}: {pid!r} scored twice')
                continue
            if pid not in digests:
                problems.append(f'shard {index}: {pid!r} is not an admitted pair')
                continue
            entry = recorded.get(pid) or {}
            if entry.get('pair_sha256') != digests[pid]:
                problems.append(f'shard {index}: {pid!r} pair_sha256 differs from the current encoding')
                continue
            values = [entry.get(side) for side in ('chosen', 'rejected')]
            if not all(isinstance(v, float) and math.isfinite(v) for v in values):
                problems.append(f'shard {index}: {pid!r} scores are not finite floats')
                continue
            scores[pid] = (values[0], values[1])
    missing = sorted(set(digests) - set(scores))
    if missing:
        problems.append(f'{len(missing)} admitted pairs have no cached reference score: {missing[:10]}')
    if problems:
        raise ValueError('reference cache rejected: ' + '; '.join(problems[:20]))
    return scores


def engine_identity(versions, device, dtype):
    """The scoring engine a cache is bound to: head layout, library versions, device/dtype and the sha256 of this
    scorer and the two modules it computes with. Any change (comments included) invalidates the cache on purpose:
    these are short-lived immutable experiment caches, and nothing recomputes them implicitly."""
    here = Path(__file__).resolve()
    code = {here.name: T.sha256(here), 'train_open_weight_lora.py': T.sha256(Path(T.__file__).resolve()),
            'q8_gguf_base.py': T.sha256(here.parent / 'q8_gguf_base.py')}
    return dict(answer_head='packed_supervised_tokens_v1', versions=versions, device=device, dtype=str(dtype), code=code)


INITIAL_CACHE_MAX_ABS_ERROR = 1e-3  # summed log-prob: the policy IS the reference, so only numerical noise is allowed


def check_initial_agreement(pairs, beta):
    """`pairs` = [{id, policy: {chosen, rejected}, reference: {chosen, rejected}}] scored by the untouched policy.
    Returns the diagnostics recorded as `initial_check`; raises ValueError (before any update) when a score, error or
    margin is not finite or the largest absolute error exceeds INITIAL_CACHE_MAX_ABS_ERROR."""
    rows = []
    for x in pairs:
        pc, pr, rc, rr = x['policy']['chosen'], x['policy']['rejected'], x['reference']['chosen'], x['reference']['rejected']
        rows.append(dict(x, cache_abs_error=max(abs(pc - rc), abs(pr - rr)), policy_margin=pc - pr,
                         preference_margin=(pc - pr) - (rc - rr)))
    bad = [r['id'] for r in rows if not all(math.isfinite(v) for v in (r['policy']['chosen'], r['policy']['rejected'],
                                                                        r['reference']['chosen'], r['reference']['rejected'],
                                                                        r['cache_abs_error'], r['preference_margin']))]
    if bad:
        raise ValueError(f'initial policy/reference scores are not finite for pairs {bad[:10]}')
    worst = max(rows, key=lambda r: r['cache_abs_error'])
    check = dict(pairs=rows, tolerance=INITIAL_CACHE_MAX_ABS_ERROR,
                 initial_cache_max_abs_error=worst['cache_abs_error'], worst_pair=worst['id'],
                 initial_preference_margin=sum(r['preference_margin'] for r in rows) / len(rows),
                 initial_policy_margin=sum(r['policy_margin'] for r in rows) / len(rows),
                 initial_preference_loss=sum(-torch.nn.functional.logsigmoid(torch.tensor(beta * r['preference_margin'])).item()
                                             for r in rows) / len(rows))
    if check['initial_cache_max_abs_error'] > INITIAL_CACHE_MAX_ABS_ERROR:
        raise ValueError(f'initial policy scores disagree with the reference cache: max abs error '
                         f'{check["initial_cache_max_abs_error"]:.6g} > {INITIAL_CACHE_MAX_ABS_ERROR} (pair {worst["id"]!r}); '
                         'the cache was not produced by this checkpoint/engine, nothing is trained')
    return check


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['cache', 'train'], required=True)
    p.add_argument('--model', type=Path, required=True, help='HF directory (config + tokenizer) of the base')
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--base-gguf', type=Path, required=True, help='the serving Q8_0 GGUF (frozen base)')
    p.add_argument('--base-gguf-sha256', required=True)
    p.add_argument('--manifest', type=Path, required=True, help='preference-pair manifest')
    p.add_argument('--out', type=Path, required=True, help='cache: new shard file; train: new run directory')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--source-dropout', type=float, default=0.05,
                   help='LoRA dropout the reference checkpoint was trained with (its adapter_config value); the policy trains with 0.0')
    p.add_argument('--max-tokens', type=int, default=1024, help='pair or valid sequences above this are an error, never dropped')
    p.add_argument('--ids', nargs='+', help='cache: score only these admitted pair ids (one bounded shard)')
    p.add_argument('--cache', type=Path, action='append', default=[], help='train: reference cache shard (repeat; pair with --cache-sha256)')
    p.add_argument('--cache-sha256', action='append', default=[], help='train: expected sha256 of the --cache shard in the same position')
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--batch-pairs', type=int, default=4, help='pairs per optimizer update (one pair per micro-batch)')
    p.add_argument('--replay-micro-batch', type=int, default=4,
                   help='train: replay rows per forward within an update (rows of the update stable-sorted by encoded '
                        'length, then chunked); computational batching only, the objective is identical for any value >= 1')
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--valid-every', type=int, default=50)
    p.add_argument('--checkpoint-every', type=int, help='train: required; save every N steps and at the last step')
    p.add_argument('--max-steps', type=int, help='train: explicit smoke bound on the number of updates')
    a = p.parse_args()
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
        if a.batch_pairs < 1:
            p.error('--batch-pairs must be >= 1')
        if a.replay_micro_batch < 1:
            p.error('--replay-micro-batch must be >= 1')
        if not a.cache or len(a.cache) != len(a.cache_sha256):
            p.error('--mode train requires at least one --cache with a --cache-sha256 in the same position')
        if a.ids:
            p.error('--ids applies to --mode cache only')
    elif a.cache or a.cache_sha256 or a.max_steps is not None or a.checkpoint_every is not None:
        p.error('--cache/--cache-sha256/--max-steps/--checkpoint-every apply to --mode train only')
    started = time.time()
    random.seed(a.seed); torch.manual_seed(a.seed)
    device, dtype = 'cpu', torch.float32
    base = a.manifest.resolve().parent
    manifest = json.loads(a.manifest.read_text())
    manifest_sha256 = T.sha256(a.manifest)
    ref = manifest['reference']

    # Reference ancestry: the SOURCE manifest (what the reference adapter was trained on), its LoRA shape, the pins.
    source_manifest_path = pinned_file(ref['source_manifest'], base, 'source manifest')
    source_manifest = json.loads(source_manifest_path.read_text())
    checkpoint = resolve(ref['checkpoint'], base)
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
        return (str(resolve(entry['path'], root).resolve()), entry['sha256'], entry['adapter'], entry.get('split'))
    valid_entries = manifest.get('valid', [])
    if [valid_key(e, base) for e in valid_entries] != [valid_key(e, source_manifest_path.parent) for e in source_manifest.get('valid', [])]:
        raise ValueError('manifest valid entries are not exactly the source manifest valid entries')
    # Replay is a REQUIRED part of the contract in both modes: the block is validated against the reference's source
    # pin before the tokenizer, and every canonical source TRAIN row is loaded and encoded before any weight is loaded.
    replay = replay_contract(manifest, base, source_manifest_path, ref['source_manifest']['sha256'])

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tokenizer = tokenizer_identity(a.model, tok)
    source, ambiguous = source_rows(source_manifest, source_manifest_path.parent)
    pairs = load_pairs(manifest, base, source, ambiguous)
    encoded = {pair['id']: encode_pair(tok, pair, a.max_tokens) for pair in pairs}
    digests = {pid: pair_digest(encoded[pid]) for pid in encoded}
    replay_encoded = {}
    for row in replay_rows(source_manifest, source_manifest_path.parent):
        item = T.encode(tok, row, FAMILY, a.max_tokens)
        if item is None:
            raise ValueError(f'replay source row {row["id"]!r} ({row["file"]}) exceeds --max-tokens {a.max_tokens}; '
                             'every canonical source TRAIN row must be replayed, nothing is dropped')
        replay_encoded[row['id']] = item
    replay_record = dict(replay, kind=OBJECTIVE_KIND, seed=a.seed + 1, rows_admitted=len(replay_encoded),
                         source_files=[{k: e.get(k) for k in ('path', 'sha256', 'adapter', 'split')} for e in source_manifest['train']],
                         rows_by_file={f: sum(1 for x in replay_encoded.values() if x['file'] == f)
                                       for f in sorted({x['file'] for x in replay_encoded.values()})},
                         token_label_sha256=replay_digest(list(replay_encoded.values())))
    base_identity = dict(model_id=a.model_id, revision=a.model_revision, gguf_sha256=a.base_gguf_sha256)
    bindings = {'kind': CACHE_KIND, 'manifest_sha256': manifest_sha256, 'reference': reference, 'base': base_identity,
                'tokenizer': tokenizer, 'family': FAMILY, 'max_tokens': a.max_tokens, 'objective': OBJECTIVE}
    import peft, transformers
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)
    bindings['engine'] = engine_identity(versions, device, dtype)

    if a.mode == 'cache':
        ids = list(encoded)
        if a.ids:
            unknown, counts = sorted(set(a.ids) - set(encoded)), {}
            for pid in a.ids:
                counts[pid] = counts.get(pid, 0) + 1
            duplicates = sorted(pid for pid, n in counts.items() if n > 1)
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
        cached = verify_cache_shards(shards, bindings, digests)
        cache_record = [dict(path=str(path), sha256=sha, ids=len(shard['ids']))
                        for path, sha, shard in zip(a.cache, a.cache_sha256, shards)]
        valid = []
        for entry in valid_entries:
            for row in T.load_rows(entry, base):
                item = T.encode(tok, row, FAMILY, a.max_tokens)
                if item is None:
                    raise ValueError(f'valid row {row.get("id")!r} ({row["file"]}) exceeds --max-tokens {a.max_tokens}; '
                                     'every source valid row must stay represented')
                valid.append(item)
        # The replay partition depends only on counts: rejected here (N < full-epoch steps) before any weight is loaded.
        replay_bounds = replay_partition(len(replay_encoded), math.ceil(len(encoded) / a.batch_pairs))

    # One model per process: the Q8 base plus the reference adapter (policy in train mode, scorer in cache mode).
    from peft import LoraConfig, get_peft_model
    from q8_gguf_base import base_weights_record, load_qwen3_q8_gguf
    model, loading = load_qwen3_q8_gguf(a.model, a.base_gguf, a.base_gguf_sha256, dtype, device)
    base_weights = base_weights_record(loading)
    print(json.dumps({'weight_loading': loading}), flush=True)
    lora = dict(rank=a.rank, alpha=a.alpha, dropout=0.0, targets=T.TARGETS)
    if a.mode == 'train':
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=T.TARGETS,
                                             task_type='CAUSAL_LM'))
    ancestry['tensors_loaded'] = T.load_init_adapter(model, checkpoint / 'adapter', device)
    print(json.dumps({'init_adapter': ancestry}, ensure_ascii=False), flush=True)

    def write_json(path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    if a.mode == 'cache':
        model.eval()
        scores = {}
        with torch.no_grad():
            for n, pid in enumerate(ids, 1):
                chosen, rejected = (float(v) for v in pair_logprobs(model, encoded[pid], device, pad_id).tolist())
                if not (math.isfinite(chosen) and math.isfinite(rejected)):
                    raise FloatingPointError(f'non-finite reference score for pair {pid!r}; nothing written')
                scores[pid] = dict(chosen=chosen, rejected=rejected, pair_sha256=digests[pid])
                print(json.dumps({'cached': pid, 'n': n, 'of': len(ids), 'chosen': chosen, 'rejected': rejected,
                                  'elapsed_seconds': round(time.time() - started, 1)}, ensure_ascii=False), flush=True)
        shard = dict(bindings, manifest=str(a.manifest), manifest_id=manifest.get('id'), base_weights=base_weights,
                     init_adapter=ancestry, admitted_pairs=len(encoded), ids=ids, scores=scores, seed=a.seed, replay=replay_record,
                     device=device, dtype=str(dtype), versions=versions, elapsed_seconds=round(time.time() - started, 1),
                     status='complete')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(a.out, shard)
        print(f'cache complete: {len(ids)} pairs -> {a.out} sha256 {T.sha256(a.out)}', flush=True)
        return

    order = list(encoded)
    random.shuffle(order)  # the global seeded stream, as before: the pair order is unchanged by replay
    batches_full = pair_batches(order, a.batch_pairs)
    planned = len(batches_full)  # = full_epoch_steps = ceil(pairs / batch_pairs), computed before model load too
    steps = min(planned, a.max_steps) if a.max_steps else planned
    batches = batches_full[:steps]
    # Replay: an independent shuffle of source manifest order, partitioned over the FULL epoch; --max-steps keeps the
    # prefix of those batches, so a smoke run sees a prefix of the replay rows (never all of them).
    replay_order = list(replay_encoded)
    random.Random(a.seed + 1).shuffle(replay_order)
    replay_batches = [replay_order[start:stop] for start, stop in replay_bounds][:steps]
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))

    # The policy starts AS the reference: its scores of the first batch must reproduce the cache up to numerical noise,
    # checked and recorded before the first update (initial_cache_max_abs_error, initial_preference_margin).
    model.eval()
    initial = []
    with torch.no_grad():
        for pid in batches[0]:
            chosen, rejected = pair_logprobs(model, encoded[pid], device, pad_id).tolist()
            initial.append(dict(id=pid, policy=dict(chosen=chosen, rejected=rejected),
                                reference=dict(chosen=cached[pid][0], rejected=cached[pid][1])))
    initial_check = check_initial_agreement(initial, a.beta)
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
                       micro_batches_per_step=[len(replay_microbatches([(rid, replay_encoded[rid]) for rid in b], a.replay_micro_batch))
                                               for b in replay_batches],
                       loss='per row the answer+EOS token-mean cross-entropy; each micro-batch backpropagates weight * '
                            'sum of its rows / rows of the update, so replay_ce is the mean over all rows of the update '
                            'whatever the micro-batch size')
    objective = dict(kind=OBJECTIVE_KIND, beta=a.beta, reference_scores=OBJECTIVE, reference=reference, cache=cache_record,
                     sft_auxiliary_loss=None, replay=replay_plan, planned_epochs=1, pairs_admitted=len(order),
                     batch_pairs=a.batch_pairs, replay_micro_batch=a.replay_micro_batch,
                     micro_batch='one pair (two sequences) per forward, then the update\'s replay rows length-sorted in '
                                 'micro-batches of at most replay_micro_batch rows per forward',
                     final_partial_batch='pairs divided by the actual pair count, replay rows by the actual row count of '
                                         'the update (a short final micro-batch contributes only its rows)',
                     train_loss='preference_loss + replay.weight * replay_ce', grad_clip=1.0)
    (a.out / 'checkpoints').mkdir(parents=True)

    def consumed_through(step):
        return sum(len(b) for b in batches[:step])

    def replay_consumed_through(step):
        return sum(len(b) for b in replay_batches[:step])

    def consumption(step):
        """What the adapter saved at `step` has actually seen: pairs and replay rows are consumed once each in their
        shuffled orders, so a --max-steps run or an intermediate checkpoint has trained on a prefix of the admitted
        pairs and a prefix of the replay rows, not the epoch."""
        consumed, replayed = consumed_through(step), replay_consumed_through(step)
        return dict(pairs_admitted=len(order), pairs_planned=consumed_through(steps), pairs_consumed_through_step=consumed,
                    epoch_fraction_consumed=round(consumed / len(order), 4), full_epoch_steps=planned, planned_steps=steps,
                    bounded_by_max_steps=steps < planned,
                    epoch_status='complete' if consumed == len(order) else 'partial',
                    replay_rows_admitted=len(replay_order), replay_rows_planned=replay_consumed_through(steps),
                    replay_rows_consumed_through_step=replayed,
                    replay_epoch_fraction_consumed=round(replayed / len(replay_order), 4),
                    replay_epoch_status='complete' if replayed == len(replay_order) else 'partial')

    def report(status, **extra):
        payload = dict(
            protocol=__doc__, base_model_id=a.model_id, base_revision=a.model_revision, family=FAMILY,
            manifest=str(a.manifest), manifest_sha256=manifest_sha256, manifest_id=manifest.get('id'),
            files=[{k: e.get(k) for k in ('path', 'sha256', 'adapter', 'epochs', 'split')} for e in manifest['train'] + valid_entries],
            replay_files=replay_record['source_files'],
            train_rows=len(order), train_pairs_admitted=len(order), replay_rows=len(replay_order), valid_rows=len(valid),
            task_counts={t: sum(1 for pid in order if encoded[pid]['chosen']['task'] == t)
                         for t in sorted({encoded[pid]['chosen']['task'] for pid in order})},
            steps=steps, planned_steps=steps, batch_size=a.batch_pairs, micro_batch_size=1, replay_micro_batch_size=a.replay_micro_batch,
            lr=a.lr, beta=a.beta,
            lora=lora, max_tokens=a.max_tokens, seed=a.seed, device=device, dtype=str(dtype), base_weights=base_weights,
            init_adapter=ancestry, reference=reference, objective=objective, tokenizer=tokenizer,
            initial_check=initial_check, versions=versions, history=list(history),
            elapsed_seconds=round(time.time() - started, 1), status=status, selection=T.SELECTION)
        payload.update(extra)
        payload['cumulative_steps'] = ancestry['source_cumulative_steps'] + payload['steps']
        return payload

    def ledger(run_status, **extra):
        write_json(a.out / 'checkpoints.json', dict(
            run_status=run_status, planned_steps=steps, checkpoint_every=a.checkpoint_every,
            base_model_id=a.model_id, base_revision=a.model_revision, manifest_sha256=manifest_sha256, base_weights=base_weights,
            init_adapter=ancestry, reference=reference, objective=objective,
            checkpoints=list(checkpoints), selection=T.SELECTION, **extra))

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
            numbers = accumulate_step(model, [(pid, encoded[pid], cached[pid]) for pid in batch],
                                      [(rid, replay_encoded[rid]) for rid in replay_batch], a.beta, replay['weight'], device, pad_id,
                                      replay_micro_batch=a.replay_micro_batch)
        except FloatingPointError as exc:
            raise FloatingPointError(f'{exc} at step {step}; stopping without saving') from None
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step(); schedule.step()
        checkpoint_step = step % a.checkpoint_every == 0 or step == steps
        record = {'step': step, 'train_loss': round(numbers['train_loss'], 5), 'preference_loss': round(numbers['preference_loss'], 5),
                  'replay_ce': round(numbers['replay_ce'], 5), 'replay_weight': replay['weight'],
                  'reward_margin': round(numbers['reward_margin'], 5), 'preferred_fraction': round(numbers['preferred_fraction'], 4),
                  'pairs': len(batch), 'replay_rows': len(replay_batch), 'replay_micro_batches': numbers['replay_micro_batches'],
                  'lr': schedule.get_last_lr()[0],
                  'elapsed_seconds': round(time.time() - started, 1)}
        if valid and (step % a.valid_every == 0 or step == steps or checkpoint_step):
            record['valid_loss_by_task'] = T.validate(model, valid, device, pad_id, 1, FAMILY)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if checkpoint_step:
            save_checkpoint(step, record)
    model.save_pretrained(a.out / 'adapter')
    last = dict(step=steps, path=str(a.out / 'adapter'), adapter_weights_sha256=T.sha256(a.out / 'adapter' / 'adapter_model.safetensors'))
    last['equals_last_checkpoint'] = last['adapter_weights_sha256'] == checkpoints[-1]['adapter_weights_sha256']
    ledger('complete', last_step=last)
    write_json(a.out / 'training.json', report('complete', checkpoints=checkpoints, last_step=last, consumption=consumption(steps)))
    print(f'complete: {steps} steps, {consumed_through(steps)} of {len(order)} admitted pairs and '
          f'{replay_consumed_through(steps)} of {len(replay_order)} replay rows consumed -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
