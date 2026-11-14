"""LoRA SFT of an open-weight chat model (TRID specialization) from a pinned data manifest.

Isolated environment (the project venv pins transformers 5.x and has no peft):

  uv run --no-project --python 3.12 --with 'torch==2.9.1' --with 'transformers==4.57.6' --with 'peft==0.17.1' \
      --with 'safetensors==0.8.0' --with numpy --with tokenizers python scripts/train_open_weight_lora.py \
      --model <hf dir> --model-id Qwen/Qwen3-1.7B --model-revision <sha> --manifest <manifest.json> --out runs/<name>

Manifest (JSON): {"id": str, "train": [file...], "valid": [file...]} with file = {"path", "sha256", "adapter",
"epochs" (default 1), "split" (optional: use only rows whose `split` equals it)}. Every file hash is verified before training; the manifest decides eligibility, this script
never filters for quality. Adapters:
  - `prompt_answer`: rows {"prompt", "answer", "history"?: [{"user": str, "assistant": str}], "task"?, "id"?}
    (trid curricula). RAG prompts (trid.prompts.rag_prompt format) get the evaluation's grammar_instructed system
    prompt; other rows get none. History is ordered prior dialogue, not additional supervised targets.
  - `case`: rows {"id", "question", "context": [str], "target", "task_type"} (C cases). The user turn is
    rag_prompt over the context passages as D1..Dn with CASE_SYSTEM (cases answer with conclusion, basis,
    conditions and exceptions; the value-only SYSTEM would contradict those targets); the answer is `target`.
    Contract (MODEL-REPAIR-001, shared with B's scripts/build_case_training_view.py): the user turn demands source
    numbers and '확인 불가', so every case target must carry a [Dn] citation of an existing passage or start with
    '확인 불가: ', and fit --max-answer-tokens (the serving answer budget). A violating file stops training with the
    offending ids instead of being trained (pilot-sft-001 trained 57 targets without any [Dn], 36 over 256 tokens).
    Nonempty case history is unsupported and rejected. History must be a list of user/assistant pairs with nonempty
    string content (trid.text.validate_history); invalid history reports the file, line and row id before weights load.
  - `grounded_messages`: rows {"id", "task", "split", "messages": [...], "answer": str}.
    Native messages may start with a system turn, then alternate user/assistant and end with the current user.
    All content is nonempty and preserved verbatim; prior assistant turns are masked context, not targets.
    A separate `history` field is forbidden. Context or answer+EOS over budget rejects rather than truncates.
Each sample is the native chat template with the generation prompt (Qwen3: the no-thinking prefix must render)
followed by the answer and the tokenizer's EOS (end of turn); loss covers only those answer tokens. The system,
all prior user/assistant turns, current user and batch padding stay masked. With --family qwen3 the vocabulary head
runs only on packed supervised row-position pairs, with at most batch_size * LOGIT_CHUNK tokens per
non-reentrant checkpoint (chunked_answer_loss). Logits stay within that chunk budget; losses and gradients
agree with the full-width computation within floating-point tolerance. Rows longer than
--max-tokens are dropped whole and counted for legacy adapters (history is never truncated); grounded_messages
rejects them. No upsampling beyond the per-file epochs; the per-file share of trained answer
tokens is recorded. Validation loss per task is recorded at fixed steps only; nothing is selected on evaluation sets.
Writes the LoRA adapter, optionally the merged model (--merge) for GGUF conversion, and training.json.
With --checkpoint-every N the adapter is also saved every N steps and at the last step under
<out>/checkpoints/step-NNNNNN/adapter, each with its own training.json (status "checkpoint", `steps` = the steps that
adapter was trained for, `planned_steps`, the validation loss per task measured at that step, elapsed time, base
revision and manifest hash; diagnose_lora_pipeline.py --adapter reads that training.json next to the adapter).
<out>/checkpoints.json lists the saved checkpoints after every save, so a stopped run leaves usable, attributed
snapshots; its run_status stays "in_progress" and no <out>/adapter or <out>/training.json exists until the last step
finishes. <out>/adapter stays the exact last step, never a "best" pick; which checkpoint serves is selected
elsewhere (development behaviour plus task-balanced validation, not training loss), so training.json records
selection status "unselected". No optimizer state is saved; checkpoints are not resume points.
On MPS with --family qwen3 the base weights are loaded by load_qwen3_pread: each tensor is read on its own with
safetensors backend='pread' into CPU memory (no mmap-backed weight storage retained), moved to the device through
PyTorch's allocator (the supervisor's memory cap applies), assigned into a meta-built model and its CPU allocation
released (OS page-cache/RSS behaviour not guaranteed; no memory fit claimed); the tied lm_head shard is never read,
checkpoint keys/shapes are checked before any read and no initialised stand-in exists for a missing weight. Other
devices use from_pretrained(...).to(device) unchanged.
With --base-gguf <q8_0.gguf> --base-gguf-sha256 <hex> (qwen3 only) the frozen base is instead the serving GGUF's
Q8_0 blocks, loaded by q8_gguf_base.load_qwen3_q8_gguf: int8 weights + fp16 block-32 scales held as frozen
Parameters (the exact bytes llama.cpp serves), dequantised per bounded row chunk on the fly in forward and backward,
the head sharing the embedding's weight and scales. This is an explicit Q8 base, NOT the BF16 base, and the PyTorch
dequantised matmul (CPU float32 / MPS bfloat16) is not the ggml Q8 kernel; training.json/checkpoints.json record it under
`base_weights` (representation, GGUF path and sha256). --merge is refused on this path (LoRA cannot be merged into
int8); serving stays the pinned GGUF plus the converted adapter. The file sha256 is verified before loading.
The Q8 path additionally requires gguf==0.19.0 (add --with "gguf==0.19.0" to the isolated uv command above).
With --init-adapter <dir> (Q8 base only: refused without --base-gguf, with --family generic or with --merge, never a
fallback) the LoRA starts from a complete adapter this trainer saved (<run>/adapter or
<run>/checkpoints/step-NNNNNN/adapter). Before any weight is loaded, init_adapter_ancestry requires the adjacent
training.json (status "checkpoint"/"complete", written by this trainer; a bare adapter directory is interrupted or
unattributed and refused) to record the same base model id and revision, `base_weights` representation q8_0 with the
same GGUF sha256, the same manifest sha256 and the same LoRA rank/alpha/dropout/targets, an adapter_weights_sha256
(`checkpoint` or `last_step`; a complete run saved without --checkpoint-every records none and is refused) equal to
the file, `steps` a positive integer equal to the recorded checkpoint step and the last history step with
`cumulative_steps` (if present) an integer >= `steps`, and adapter_config.json to be a plain LoRA (bias none, no
modules_to_save, no DoRA/rsLoRA/patterns).
load_init_adapter then copies the saved tensors into the freshly built trainable adapter through PEFT's state-dict
loader on the already loaded Q8 base (no second base copy) and verifies exact coverage: every trainable tensor equals
one saved tensor and nothing is left over. The source directory is never written. This is a warm start, not a resume:
AdamW and the warmup+cosine schedule start fresh over this segment (the source run never saved optimizer state).
Every training.json and checkpoints.json records `init_adapter` (path, adapter/config/training.json sha256, source
status, source_steps = the updates the source adapter was trained for in its own run, source_cumulative_steps, the
optimizer/scheduler reset) and `cumulative_steps` = source_cumulative_steps + `steps`; `steps` stays the updates of
this run only, so step-bound checkpoints and their readers keep their meaning. Runs without --init-adapter record
init_adapter null and cumulative_steps = steps.
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.utils.checkpoint
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trid.prompts import RAG_PREFIX, rag_prompt  # noqa: E402
from trid.text import validate_history  # noqa: E402

SYSTEM = ('주어진 근거만 사용하세요. 답은 값 [Dn] 형식으로 쓰고, 여러 값은 쉼표로 구분하세요. '
          '근거가 없으면 확인 불가, 근거가 충돌하면 근거 충돌만 출력하세요. 설명은 덧붙이지 마세요.')
from trid.case_prompts import CASE_SYSTEM  # noqa: E402
QWEN3_NO_THINK = '<think>\n\n</think>\n\n'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
SELECTION = dict(status='unselected', default_artifact='adapter (last step)',
                 rule='development behavioural evaluation plus task-balanced validation loss; never training loss alone')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_rows(entry, base):
    path = Path(entry['path'])
    path = path if path.is_absolute() else base / path  # relative paths resolve against the manifest's directory
    if sha256(path) != entry['sha256']:
        raise ValueError(f'{path}: sha256 does not match the manifest')
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if entry.get('split') and row.get('split') != entry['split']:
            continue  # one pinned file may hold several splits; only the named one is used
        if entry['adapter'] == 'grounded_messages':
            try:
                if 'history' in row:
                    raise ValueError('grounded_messages adapter does not support history')
                for field in ('id', 'task', 'split', 'answer'):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f'grounded_messages requires nonempty string {field}')
                messages = row.get('messages')
                if not isinstance(messages, list) or not messages:
                    raise ValueError('grounded_messages requires nonempty native chat messages')
                has_system = isinstance(messages[0], dict) and messages[0].get('role') == 'system'
                for index, message in enumerate(messages):
                    role = ('system' if has_system and index == 0 else
                            'user' if (index - int(has_system)) % 2 == 0 else 'assistant')
                    if not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] != role:
                        raise ValueError(f'grounded_messages requires a {role} message with only role and content')
                    if not isinstance(message['content'], str) or not message['content'].strip():
                        raise ValueError(f'grounded_messages requires nonempty string {role} content')
                if messages[-1]['role'] != 'user':
                    raise ValueError('grounded_messages must end with the current user turn')
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: row {row.get('id')!r}: {exc}") from exc
            rows.append(dict(id=row['id'], task=row['task'], messages=messages, answer=row['answer'],
                             adapter='grounded_messages', file=str(path)))
            continue
        history = row.get('history', [])
        try:
            validate_history(history)
            if entry['adapter'] == 'case' and history:
                raise ValueError('case adapter does not support history')
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: row {row.get('id')!r}: {exc}") from exc
        if entry['adapter'] == 'prompt_answer':
            system = SYSTEM if row['prompt'].startswith(RAG_PREFIX) else None
            rows.append(dict(id=row.get('id'), task=row.get('task', 'unknown'), system=system,
                             user=row['prompt'], history=history, answer=row['answer'], file=str(path)))
        elif entry['adapter'] == 'case':
            passages = [(f'D{i}', text) for i, text in enumerate(row['context'], 1)]
            rows.append(dict(id=row['id'], task=row.get('task_type', 'unknown'), system=CASE_SYSTEM,
                             user=rag_prompt(passages, row['question']), answer=row['target'], file=str(path)))
        else:
            raise ValueError(f"unknown adapter {entry['adapter']}")
    return rows


def case_contract_violation(row, answer_tokens, max_answer_tokens):
    """Why a case row breaks the training contract, or None."""
    target = row['answer']
    cited = {int(n) for n in re.findall(r'\[D(\d+)\]', target)}
    passages = len(re.findall(r'(?:^|\n)\[D\d+\] ', row['user']))
    if not cited and not target.startswith('확인 불가: '):
        return 'no_citation'
    if cited and (min(cited) < 1 or max(cited) > passages):
        return 'citation_out_of_range'
    if max_answer_tokens and answer_tokens > max_answer_tokens:
        return 'over_answer_budget'
    return None


def messages_of(row):
    """Native-template input; prior assistant turns are context, never current targets."""
    if row.get('adapter') == 'grounded_messages':
        return row['messages']
    messages = [{'role': 'system', 'content': row['system']}] if row['system'] else []
    for turn in row.get('history', []):
        messages += [{'role': 'user', 'content': turn['user']}, {'role': 'assistant', 'content': turn['assistant']}]
    return messages + [{'role': 'user', 'content': row['user']}]


def encode(tok, row, family, max_tokens):
    prompt = tok.apply_chat_template(messages_of(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if family == 'qwen3' and not prompt.endswith(QWEN3_NO_THINK):
        raise ValueError('Native Qwen3 no-thinking prefix was not rendered')
    prompt_ids = tok(prompt, add_special_tokens=False)['input_ids']
    answer_ids = tok(row['answer'] + tok.eos_token, add_special_tokens=False)['input_ids']
    ids = prompt_ids + answer_ids
    if len(ids) > max_tokens:
        if row.get('adapter') == 'grounded_messages':
            raise ValueError(f"{row['file']}: row {row['id']!r}: over_context_budget ({len(ids)} > {max_tokens})")
        return None
    return dict(ids=ids, labels=[-100] * len(prompt_ids) + answer_ids, task=row['task'], file=row['file'])


def build_items(tok, entries, base, family, max_tokens, max_answer_tokens, expand):
    """Build a manifest split, rejecting lossy grounded input before any weights are loaded."""
    items, dropped, violations = [], 0, []
    for entry in entries:
        for _ in range(entry.get('epochs', 1) if expand else 1):
            for row in load_rows(entry, base):
                x = encode(tok, row, family, max_tokens)
                if x is None:
                    dropped += 1
                    continue
                answer_tokens = sum(t != -100 for t in x['labels'])
                if entry['adapter'] == 'grounded_messages' and answer_tokens > max_answer_tokens:
                    raise ValueError(f"{row['file']}: row {row['id']!r}: "
                                     f"over_answer_budget ({answer_tokens} > {max_answer_tokens}, EOS included)")
                if entry['adapter'] == 'case':
                    why = case_contract_violation(row, answer_tokens, max_answer_tokens)
                    if why:
                        violations.append(f"{row['id']}:{why}")
                        continue
                items.append(x)
    if violations:
        raise ValueError(f'{len(violations)} case rows break the training contract: {sorted(set(violations))[:20]}')
    return items, dropped


LOGIT_CHUNK = 64  # per-row token budget; packed head chunks contain at most batch_size * LOGIT_CHUNK tokens


def causal_lm(model):
    """The Qwen3ForCausalLM under an optional PEFT wrapper: `.model` yields the final-norm hidden states that
    `.lm_head` (the real head, tied or untied, not a LoRA target) consumes. LoRA layers live inside `.model`, so
    calling it directly runs the same adapted computation; PeftModelForCausalLM.forward only filters kwargs."""
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def chunk_cross_entropy(hidden, head, target):
    """Per-token fp32 cross-entropy of head(hidden) for one chunk of supervised positions."""
    logits = head(hidden).float()
    return torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction='none')


def chunked_answer_loss(model, ids, mask, keep, target):
    """Per-token cross-entropy [b, len(keep)] with only supervised row-position pairs reaching the head.
    `keep` maps target columns to backbone positions. Packed chunks contain at most b * LOGIT_CHUNK tokens,
    preserving the old logits workspace bound. Non-reentrant checkpointing recomputes each chunk's head/CE
    during backward; scattering its losses preserves sequence ownership and ignored-position zero gradients."""
    inner = causal_lm(model)
    hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
    head = inner.lm_head
    rows, columns = (target != -100).nonzero(as_tuple=True)
    if rows.numel() == 0:
        # batch_loss supplies one fallback column for all-ignored batches. Preserve its head computation and
        # zero gradients (including a trainable head), rather than replacing the graph with a hidden-only zero.
        selected = hidden[:, keep]
        return torch.cat([torch.utils.checkpoint.checkpoint(chunk_cross_entropy, selected[:, s:s + LOGIT_CHUNK], head,
                                                            target[:, s:s + LOGIT_CHUNK], use_reentrant=False)
                          for s in range(0, selected.shape[1], LOGIT_CHUNK)], dim=1)
    packed_hidden = hidden[rows, keep[columns]].unsqueeze(0)
    packed_target = target[rows, columns].unsqueeze(0)
    capacity = target.shape[0] * LOGIT_CHUNK
    losses = torch.cat([torch.utils.checkpoint.checkpoint(chunk_cross_entropy, packed_hidden[:, s:s + capacity], head,
                                                        packed_target[:, s:s + capacity], use_reentrant=False)
                        for s in range(0, rows.numel(), capacity)], dim=1)
    return losses.new_zeros(target.shape).index_put((rows, columns), losses.reshape(-1))


def batch_loss(model, items, device, pad_id, family='generic'):
    width = max(len(x['ids']) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x['ids'])
        ids[i, :n] = torch.tensor(x['ids']); labels[i, :n] = torch.tensor(x['labels']); mask[i, :n] = 1
    target = labels[:, 1:].to(device)
    if family == 'qwen3':
        # The union supplies a shared coordinate map; chunked_answer_loss packs only supervised (row, position)
        # pairs before the head, so ignored entries within that union incur no logits work either. Backbone
        # context, per-example normalization and the ignored-position zero gradients remain unchanged.
        keep = (target != -100).any(0).nonzero().squeeze(1)
        if keep.numel() == 0:
            keep = torch.tensor([target.shape[1] - 1], device=device)  # all-ignored batch: zero loss stays differentiable
        target = target[:, keep]
        per_token = chunked_answer_loss(model, ids.to(device), mask.to(device), keep, target)
    else:
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
        logits = out.logits[:, :-1].float()
        per_token = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction='none')
    # Example-normalized answer loss: each example weighs the same regardless of answer length.
    counts = (target != -100).sum(1).clamp(min=1)
    return (per_token.sum(1) / counts).mean()


@torch.no_grad()
def validate(model, items, device, pad_id, micro, family='generic'):
    model.eval()
    by_task = {}
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        for x in chunk:
            by_task.setdefault(x['task'], []).append(batch_loss(model, [x], device, pad_id, family).item())
    model.train()
    return {task: round(sum(v) / len(v), 5) for task, v in sorted(by_task.items())}


def tied_output_keys(model):
    """State-dict keys that `tie_weights()` rebinds to the input embedding (Qwen3: `lm_head.weight` when
    `tie_word_embeddings`), so bytes stored under them are provably never used; and the key they alias."""
    output, inputs = model.get_output_embeddings(), model.get_input_embeddings()
    if output is None or output.weight is not inputs.weight:
        return set(), None
    input_key = next(name for name, module in model.named_modules() if module is inputs) + '.weight'
    return {name for name, p in model.named_parameters(remove_duplicate=False) if p is inputs.weight} - {input_key}, input_key


@torch.no_grad()
def load_qwen3_pread(model_dir, dtype, device):
    """Qwen3ForCausalLM from a local safetensors directory, one tensor at a time: safetensors' official
    `backend='pread'` reads a single tensor into CPU memory (no mmap-backed weight storage is retained, unlike
    `from_pretrained(...).to(device)`, which keeps the mapped shard alive while it copies module by module), the
    tensor is moved to `device`/`dtype` through PyTorch's own allocator (so the supervisor's MPS memory fraction cap
    applies to every device allocation), assigned into a meta-built model and the per-tensor CPU allocation released
    before the next tensor is read. Peak transient CPU staging is one tensor (Qwen3-1.7B: the 622 MB embedding); the
    kernel may still cache file pages, so OS page-cache and RSS behaviour are not guaranteed and no memory fit is
    claimed. The model is built on the meta device (buffers such as
    rotary `inv_freq` are computed normally), checkpoint headers are checked against every expected key and shape
    BEFORE any tensor bytes are read, the canonical `tie_weights()` is re-applied and generation_config.json, dtype
    and eval mode follow from_pretrained. A tied output key (`lm_head.weight`) present in the checkpoint is only
    shape-checked; its bytes are never read, because tie_weights() would discard them. Any missing unique parameter,
    unexpected key or shape mismatch raises before loading: nothing is ever initialised in place of a weight.
    Returns (model, report) with report listing per-shard keys and bytes read/skipped."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != 'qwen3':
        raise ValueError(f'load_qwen3_pread supports model_type qwen3 only, got {config.model_type!r}')
    config.name_or_path = model_dir  # from_pretrained does the same; PEFT records it as base_model_name_or_path
    config.dtype = dtype
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.tie_weights()
    redundant, input_key = tied_output_keys(model)
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}

    index = model_dir / SAFE_WEIGHTS_INDEX_NAME
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    elif (model_dir / SAFE_WEIGHTS_NAME).is_file():
        shards = [SAFE_WEIGHTS_NAME]
    else:
        raise FileNotFoundError(f'{model_dir} has neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME}')
    sizes = {'F32': 4, 'BF16': 2, 'F16': 2}
    plan, seen, problems = [], {}, []
    for shard in shards:
        with safe_open(model_dir / shard, framework='pt', device='cpu', backend='pread') as f:
            keys = list(f.keys())
            header = {k: (f.get_slice(k).get_shape(), f.get_slice(k).get_dtype()) for k in keys}
        for key, (shape, kind) in header.items():
            if key in seen:
                problems.append(f'{key} appears in {seen[key]} and {shard}')
            seen[key] = shard
            want = expected.get(input_key if key in redundant else key)
            if want is None:
                problems.append(f'unexpected key {key} in {shard}')
            elif tuple(shape) != want:
                problems.append(f'{key}: checkpoint shape {list(shape)} != model shape {list(want)}')
        nbytes = {k: math.prod(shape) * sizes.get(kind, 0) for k, (shape, kind) in header.items()}
        plan.append(dict(file=shard, keys=keys, bytes=nbytes))
    missing = sorted(set(expected) - redundant - set(seen))
    if missing:
        problems.append(f'missing parameters: {missing}')
    if problems:
        raise ValueError(f'{model_dir} does not match {type(model).__name__}: ' + '; '.join(problems))

    report = dict(backend='pread', staging='cpu-per-tensor', device=str(device), dtype=str(dtype), tied_output_keys=sorted(redundant),
                  input_key=input_key, shards=[], parameters=0, bytes_loaded=0, bytes_skipped=0, cast=0)
    for entry in plan:
        wanted = [k for k in entry['keys'] if k not in redundant]
        skipped = [k for k in entry['keys'] if k in redundant]
        record = dict(file=entry['file'], loaded=len(wanted), skipped_tied=skipped,
                      bytes_loaded=sum(entry['bytes'][k] for k in wanted), bytes_skipped=sum(entry['bytes'][k] for k in skipped))
        report['shards'].append(record)
        report['bytes_skipped'] += record['bytes_skipped']
        if not wanted:
            continue  # the whole shard is redundant (Qwen3-1.7B shard 2 = lm_head.weight only): never read
        with safe_open(model_dir / entry['file'], framework='pt', device='cpu', backend='pread') as f:
            for key in wanted:  # redundant keys of a mixed shard are simply never fetched
                module_name, _, attr = key.rpartition('.')
                module = model.get_submodule(module_name)
                target = getattr(module, attr)
                staged = f.get_tensor(key)  # one tensor on CPU via pread(2)
                report['cast'] += int(staged.dtype != target.dtype)
                tensor = staged.to(device=device, dtype=target.dtype)  # PyTorch-allocated device copy (cap-accounted)
                del staged  # the CPU staging copy is released before the next tensor is read
                module.load_state_dict({attr: tensor}, strict=False, assign=True)
                report['parameters'] += 1
                report['bytes_loaded'] += entry['bytes'][key]
                del tensor
    model.tie_weights()
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == 'meta']
    if still_meta:
        raise RuntimeError(f'parameters left uninitialised after loading: {still_meta}')
    if redundant and model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise RuntimeError('tied output embedding lost its shared storage')
    model.to(device)  # parameters are already on `device` (no copy); this moves the CPU-initialised buffers
    model.eval()
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
    except OSError:  # as from_pretrained: keep the config-derived generation config
        pass
    return model, report


def init_adapter_ancestry(path, model_id, model_revision, base_gguf_sha256, manifest_sha256, lora):
    """Validate --init-adapter before any weight is loaded: `path` must be a complete adapter saved by this trainer
    (adapter_config.json + adapter_model.safetensors with the training.json this trainer writes next to the adapter
    directory), attributed to the same Q8 base (model id, revision, GGUF sha256), the same manifest hash and the same
    LoRA shape (`lora` = the current rank/alpha/dropout/targets; no bias, no modules_to_save). Returns the ancestry
    record written into every training.json of the warm-started run; raises ValueError naming the first mismatch."""
    def refuse(why):
        raise ValueError(f'--init-adapter {path}: {why}')
    config_path, weights_path, meta_path = path / 'adapter_config.json', path / 'adapter_model.safetensors', path.parent / 'training.json'
    if not (config_path.is_file() and weights_path.is_file()):
        refuse('not a saved PEFT adapter directory (adapter_config.json + adapter_model.safetensors)')
    if not meta_path.is_file():
        refuse(f'no adjacent {meta_path} (an interrupted or unattributed adapter directory is not a warm start)')
    meta = json.loads(meta_path.read_text())
    if not {'protocol', 'lora', 'base_weights', 'steps', 'history'} <= set(meta) or meta.get('status') not in ('checkpoint', 'complete'):
        refuse(f"{meta_path.name} is not a complete checkpoint record of this trainer (status {meta.get('status')!r})")
    weights_sha = sha256(weights_path)
    recorded = meta.get('checkpoint') if meta['status'] == 'checkpoint' else meta.get('last_step')
    if not isinstance(recorded, dict) or not recorded.get('adapter_weights_sha256'):
        refuse('training.json records no adapter_weights_sha256 for this adapter (unattributed; complete runs need '
               '--checkpoint-every to record last_step)')
    if recorded['adapter_weights_sha256'] != weights_sha:
        refuse('adapter_model.safetensors does not match the adapter_weights_sha256 recorded in training.json')
    source_steps, cumulative = meta['steps'], meta.get('cumulative_steps', meta['steps'])
    if not (isinstance(source_steps, int) and not isinstance(source_steps, bool) and source_steps > 0):
        refuse(f'training.json steps {source_steps!r} is not a positive step count')
    if recorded.get('step') != source_steps or not meta['history'] or meta['history'][-1].get('step') != source_steps:
        refuse(f'training.json steps {source_steps} disagrees with the recorded checkpoint step {recorded.get("step")!r} '
               f'or the last history step {(meta["history"] or [{}])[-1].get("step")!r}')
    if not (isinstance(cumulative, int) and not isinstance(cumulative, bool) and cumulative >= source_steps):
        refuse(f'training.json cumulative_steps {cumulative!r} is not an integer >= steps {source_steps}')
    for key, want in dict(family='qwen3', base_model_id=model_id, base_revision=model_revision,
                          manifest_sha256=manifest_sha256, lora=lora).items():
        if meta.get(key) != want:
            refuse(f'training.json {key} {meta.get(key)!r} != current {want!r}')
    base = meta['base_weights']
    if base.get('representation') != 'q8_0' or base.get('gguf_sha256') != base_gguf_sha256:
        refuse(f"training.json base_weights {base.get('representation')!r}/{base.get('gguf_sha256')!r} "
               f'!= current q8_0/{base_gguf_sha256!r}')
    cfg = json.loads(config_path.read_text())
    shape = dict(peft_type=cfg.get('peft_type'), task_type=cfg.get('task_type'), r=cfg.get('r'),
                 lora_alpha=cfg.get('lora_alpha'), lora_dropout=cfg.get('lora_dropout'),
                 target_modules=sorted(cfg.get('target_modules') or []), bias=cfg.get('bias'),
                 modules_to_save=cfg.get('modules_to_save') or None, use_dora=bool(cfg.get('use_dora')),
                 use_rslora=bool(cfg.get('use_rslora')), rank_pattern=cfg.get('rank_pattern') or {},
                 alpha_pattern=cfg.get('alpha_pattern') or {})
    want = dict(peft_type='LORA', task_type='CAUSAL_LM', r=lora['rank'], lora_alpha=lora['alpha'],
                lora_dropout=lora['dropout'], target_modules=sorted(lora['targets']), bias='none', modules_to_save=None,
                use_dora=False, use_rslora=False, rank_pattern={}, alpha_pattern={})
    for key in want:
        if shape[key] != want[key]:
            refuse(f'adapter_config.json {key} {shape[key]!r} != expected {want[key]!r}')
    return dict(path=str(path), adapter_weights_sha256=weights_sha, adapter_config_sha256=sha256(config_path),
                training_json_sha256=sha256(meta_path), source_status=meta['status'], source_steps=meta['steps'],
                source_planned_steps=meta.get('planned_steps', meta['steps']),
                source_cumulative_steps=meta.get('cumulative_steps', meta['steps']), source_manifest_id=meta.get('manifest_id'),
                optimizer='reset: fresh AdamW (the source run never saved optimizer state)',
                scheduler='reset: fresh warmup+cosine over this segment only')


def load_init_adapter(model, path, device):
    """Copy the saved LoRA tensors of `path` into the trainable 'default' adapter of the freshly built PEFT model through
    PEFT's own state-dict loader. Coverage is checked BEFORE the loader runs (every trainable parameter has exactly one
    saved tensor of its shape and no saved tensor is left over, so a foreign or base-named key can never touch the
    model) and verified after it (no unexpected keys, every trainable parameter equals its saved tensor). The base is
    the model already loaded; no second base copy exists. Returns the number of tensors loaded."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    saved = load_file(str(path / 'adapter_model.safetensors'))  # CPU tensors; load_state_dict copies to each parameter
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    expected = {re.sub(r'\.(lora_[AB])\.default\.', r'.\1.', name): name for name in trainable}  # saved key -> parameter
    missing, unexpected = sorted(set(expected) - set(saved)), sorted(set(saved) - set(expected))
    if missing or unexpected or not expected:
        raise ValueError(f'--init-adapter {path}: tensor coverage mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}')
    misshaped = [key for key, name in expected.items() if tuple(saved[key].shape) != tuple(trainable[name].shape)]
    if misshaped:
        raise ValueError(f'--init-adapter {path}: saved tensor shape differs from the parameter for {misshaped[:5]}')
    result = set_peft_model_state_dict(model, saved, adapter_name='default')
    if getattr(result, 'unexpected_keys', []):
        raise ValueError(f'--init-adapter {path}: PEFT loader reported unexpected keys {list(result.unexpected_keys)[:5]}')
    for key, name in expected.items():
        p = trainable[name]
        if not torch.equal(p.detach(), saved[key].to(p.device, p.dtype)):
            raise ValueError(f'--init-adapter {path}: {key} was not loaded into {name}')
    return len(expected)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--family', choices=['qwen3', 'generic'], default='qwen3')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, help='default: one pass over the epoch-expanded training rows')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--micro-batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--max-tokens', type=int, default=1536)
    p.add_argument('--max-answer-tokens', type=int, default=256,
                   help='case and grounded_messages targets above this (serving answer budget, EOS included) break the contract')
    p.add_argument('--valid-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--merge', action='store_true', help='also write the merged model to <out>/merged')
    p.add_argument('--base-gguf', type=Path, help='qwen3 only: train on the frozen Q8_0 blocks of this serving GGUF '
                                                  '(q8_gguf_base.load_qwen3_q8_gguf) instead of the safetensors base')
    p.add_argument('--base-gguf-sha256', help='required with --base-gguf: expected sha256 of the GGUF, verified before loading')
    p.add_argument('--init-adapter', type=Path,
                   help='Q8 base only: warm-start the LoRA from this complete adapter saved by this trainer (its adjacent '
                        'training.json must record the same base, GGUF sha256, manifest sha256 and LoRA shape); fresh '
                        'AdamW and schedule, not an optimizer resume')
    p.add_argument('--max-train-seconds', type=float,
                   help='after 5 steps, stop (without saving) if the projected training time exceeds this')
    p.add_argument('--checkpoint-every', type=int,
                   help='also save the adapter every N steps and at the last step under <out>/checkpoints/step-NNNNNN/ '
                        '(each with its own training.json); <out>/adapter stays the last step, nothing is selected')
    a = p.parse_args()
    if a.out.exists():
        p.error('Use a new --out directory')
    if a.checkpoint_every is not None and a.checkpoint_every < 1:
        p.error('--checkpoint-every must be a positive step count')
    if a.base_gguf is not None:
        if not a.base_gguf_sha256:
            p.error('--base-gguf requires --base-gguf-sha256')
        if a.family != 'qwen3':
            p.error('--base-gguf supports --family qwen3 only')
        if a.merge:
            p.error('--merge is unsupported with --base-gguf: LoRA cannot be merged into a Q8_0 base')
    if a.init_adapter is not None and a.base_gguf is None:
        p.error('--init-adapter supports the Q8 base only (--base-gguf/--base-gguf-sha256); BF16/generic warm starts are refused')
    started = time.time()
    random.seed(a.seed); torch.manual_seed(a.seed)
    manifest = json.loads(a.manifest.read_text())
    manifest_sha256 = sha256(a.manifest)
    ancestry = None
    if a.init_adapter is not None:
        ancestry = init_adapter_ancestry(a.init_adapter, a.model_id, a.model_revision, a.base_gguf_sha256, manifest_sha256,
                                         dict(rank=a.rank, alpha=a.alpha, dropout=a.dropout, targets=TARGETS))
        print(json.dumps({'init_adapter': ancestry}, ensure_ascii=False), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train, dropped = build_items(tok, manifest['train'], a.manifest.resolve().parent, a.family,
                                 a.max_tokens, a.max_answer_tokens, True)
    valid, valid_dropped = build_items(tok, manifest.get('valid', []), a.manifest.resolve().parent, a.family,
                                       a.max_tokens, a.max_answer_tokens, False)
    if not train:
        raise ValueError('No training rows')
    random.shuffle(train)
    steps = a.steps or math.ceil(len(train) / a.batch_size)
    accumulation = a.batch_size // a.micro_batch_size
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base_weights = dict(representation='hf-safetensors', dtype=str(dtype))
    if a.base_gguf is not None:
        from q8_gguf_base import base_weights_record, load_qwen3_q8_gguf
        model, loading = load_qwen3_q8_gguf(a.model, a.base_gguf, a.base_gguf_sha256, dtype, a.device)
        base_weights = base_weights_record(loading)
        print(json.dumps({'weight_loading': loading}), flush=True)
    elif a.family == 'qwen3' and a.device == 'mps':
        model, loading = load_qwen3_pread(a.model, dtype, a.device)
        print(json.dumps({'weight_loading': loading}), flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=TARGETS, task_type='CAUSAL_LM'))
    if ancestry is not None:
        ancestry['tensors_loaded'] = load_init_adapter(model, a.init_adapter, a.device)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    history, cursor = [], 0
    trained_tokens = {}
    model.train()
    train_started = time.time()
    checkpoints = []
    import peft, transformers
    if a.checkpoint_every:
        (a.out / 'checkpoints').mkdir(parents=True)

    def report(status, **extra):
        """training.json for the adapter saved at this moment: identity, data provenance and the history so far."""
        token_total = sum(trained_tokens.values())
        payload = dict(
            protocol=__doc__, base_model_id=a.model_id, base_revision=a.model_revision, family=a.family,
            manifest=str(a.manifest), manifest_sha256=manifest_sha256, manifest_id=manifest.get('id'),
            files=[{k: e.get(k) for k in ('path', 'sha256', 'adapter', 'epochs')} for e in manifest['train'] + manifest.get('valid', [])],
            train_rows=len(train), valid_rows=len(valid), dropped_over_max_tokens=dropped, valid_dropped=valid_dropped,
            task_counts={t: sum(1 for x in train if x['task'] == t) for t in sorted({x['task'] for x in train})},
            answer_token_share={f: round(n / token_total, 4) for f, n in sorted(trained_tokens.items())},
            steps=steps, batch_size=a.batch_size, micro_batch_size=a.micro_batch_size, lr=a.lr,
            lora=dict(rank=a.rank, alpha=a.alpha, dropout=a.dropout, targets=TARGETS), max_tokens=a.max_tokens,
            max_answer_tokens=a.max_answer_tokens,
            seed=a.seed, device=a.device, dtype=str(dtype), base_weights=base_weights, init_adapter=ancestry,
            system_prompt=SYSTEM, case_system_prompt=CASE_SYSTEM,
            versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
            history=list(history), elapsed_seconds=round(time.time() - started, 1), status=status)
        payload.update(extra)  # a checkpoint overrides `steps` with the steps its adapter was trained for
        payload['cumulative_steps'] = (ancestry['source_cumulative_steps'] if ancestry else 0) + payload['steps']
        return payload

    def write_json(path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    def ledger(run_status, **extra):
        """<out>/checkpoints.json after every checkpoint save: what exists so far, and that nothing is selected."""
        write_json(a.out / 'checkpoints.json', dict(
            run_status=run_status, planned_steps=steps, checkpoint_every=a.checkpoint_every,
            base_model_id=a.model_id, base_revision=a.model_revision, manifest_sha256=manifest_sha256, base_weights=base_weights,
            init_adapter=ancestry,
            checkpoints=list(checkpoints), selection=SELECTION, **extra))

    def save_checkpoint(step, record):
        folder = a.out / 'checkpoints' / f'step-{step:06d}'
        model.save_pretrained(folder / 'adapter')
        checkpoints.append(dict(step=step, path=str(folder / 'adapter'),
                                adapter_weights_sha256=sha256(folder / 'adapter' / 'adapter_model.safetensors'),
                                train_loss=record['train_loss'], valid_loss_by_task=record.get('valid_loss_by_task'),
                                elapsed_seconds=record['elapsed_seconds']))
        # Bound to this step only: `steps` is what the saved adapter was trained for, never the plan.
        write_json(folder / 'training.json', report('checkpoint', steps=step, planned_steps=steps, checkpoint=checkpoints[-1]))
        ledger('in_progress')
        print(json.dumps({'checkpoint': checkpoints[-1]}, ensure_ascii=False), flush=True)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            chunk = [train[(cursor + i) % len(train)] for i in range(a.micro_batch_size)]
            cursor += a.micro_batch_size
            for x in chunk:
                trained_tokens[x['file']] = trained_tokens.get(x['file'], 0) + sum(1 for t in x['labels'] if t != -100)
            loss = batch_loss(model, chunk, a.device, pad_id, a.family) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {step}; stopping without saving')
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step(); schedule.step()
        if a.max_train_seconds and step == 5:
            projected = (time.time() - train_started) / 5 * steps
            if projected > a.max_train_seconds:
                raise SystemExit(f'projected training time {projected:.0f}s exceeds --max-train-seconds; stopping early')
        checkpoint_step = bool(a.checkpoint_every) and (step % a.checkpoint_every == 0 or step == steps)
        record = {'step': step, 'train_loss': round(total, 5), 'lr': schedule.get_last_lr()[0],
                  'elapsed_seconds': round(time.time() - started, 1)}
        if valid and (step % a.valid_every == 0 or step == steps or checkpoint_step):
            record['valid_loss_by_task'] = validate(model, valid, a.device, pad_id, a.micro_batch_size, a.family)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if checkpoint_step:
            save_checkpoint(step, record)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out / 'adapter')
    if a.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(a.out / 'merged', safe_serialization=True)
        tok.save_pretrained(a.out / 'merged')
    extra = {}
    if a.checkpoint_every:
        last = dict(step=steps, path=str(a.out / 'adapter'), adapter_weights_sha256=sha256(a.out / 'adapter' / 'adapter_model.safetensors'))
        last['equals_last_checkpoint'] = last['adapter_weights_sha256'] == checkpoints[-1]['adapter_weights_sha256']
        extra = dict(checkpoints=checkpoints, last_step=last, selection=SELECTION)
        ledger('complete', last_step=last)
    write_json(a.out / 'training.json', report('complete', **extra))
    print(f'complete: {steps} steps -> {a.out}')


if __name__ == '__main__':
    main()
