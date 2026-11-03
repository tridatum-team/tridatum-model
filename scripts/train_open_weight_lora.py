"""LoRA SFT of an open-weight chat model (TRID specialization) from a pinned data manifest.

Isolated environment (the project venv pins transformers 5.x and has no peft):

  uv run --no-project --python 3.12 --with 'torch==2.9.*' --with 'transformers>=4.51,<5' --with 'peft==0.17.*' \
      --with safetensors --with numpy --with tokenizers python scripts/train_open_weight_lora.py \
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
Each sample is the native chat template with the generation prompt (Qwen3: the no-thinking prefix must render)
followed by the answer and the tokenizer's EOS (end of turn); loss covers only those answer tokens. The system,
all prior user/assistant turns, current user and batch padding stay masked. Rows longer than
--max-tokens are dropped whole and counted (history is never truncated). No upsampling beyond the per-file epochs; the per-file share of trained answer
tokens is recorded. Validation loss per task is recorded at fixed steps only; nothing is selected on evaluation sets.
Writes the LoRA adapter, optionally the merged model (--merge) for GGUF conversion, and training.json.
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
        return None
    return dict(ids=ids, labels=[-100] * len(prompt_ids) + answer_ids, task=row['task'], file=row['file'])


def batch_loss(model, items, device, pad_id):
    width = max(len(x['ids']) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x['ids'])
        ids[i, :n] = torch.tensor(x['ids']); labels[i, :n] = torch.tensor(x['labels']); mask[i, :n] = 1
    out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
    logits = out.logits[:, :-1].float()
    target = labels[:, 1:].to(device)
    # Example-normalized answer loss: each example weighs the same regardless of answer length.
    per_token = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction='none')
    counts = (target != -100).sum(1).clamp(min=1)
    return (per_token.sum(1) / counts).mean()


@torch.no_grad()
def validate(model, items, device, pad_id, micro):
    model.eval()
    by_task = {}
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        for x in chunk:
            by_task.setdefault(x['task'], []).append(batch_loss(model, [x], device, pad_id).item())
    model.train()
    return {task: round(sum(v) / len(v), 5) for task, v in sorted(by_task.items())}


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
                   help='case targets above this (serving answer budget, EOS included) break the contract')
    p.add_argument('--valid-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--merge', action='store_true', help='also write the merged model to <out>/merged')
    p.add_argument('--max-train-seconds', type=float,
                   help='after 5 steps, stop (without saving) if the projected training time exceeds this')
    a = p.parse_args()
    if a.out.exists():
        p.error('Use a new --out directory')
    started = time.time()
    random.seed(a.seed); torch.manual_seed(a.seed)
    manifest = json.loads(a.manifest.read_text())
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def build(entries, expand):
        items, dropped, violations = [], 0, []
        for entry in entries:
            for _ in range(entry.get('epochs', 1) if expand else 1):
                for row in load_rows(entry, a.manifest.resolve().parent):
                    x = encode(tok, row, a.family, a.max_tokens)
                    if x is None:
                        dropped += 1
                        continue
                    if entry['adapter'] == 'case':
                        why = case_contract_violation(row, sum(t != -100 for t in x['labels']), a.max_answer_tokens)
                        if why:
                            violations.append(f"{row['id']}:{why}")
                            continue
                    items.append(x)
        if violations:
            raise ValueError(f'{len(violations)} case rows break the training contract: {sorted(set(violations))[:20]}')
        return items, dropped

    train, dropped = build(manifest['train'], True)
    valid, valid_dropped = build(manifest.get('valid', []), False)
    if not train:
        raise ValueError('No training rows')
    random.shuffle(train)
    steps = a.steps or math.ceil(len(train) / a.batch_size)
    accumulation = a.batch_size // a.micro_batch_size
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=TARGETS, task_type='CAUSAL_LM'))
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    history, cursor = [], 0
    trained_tokens = {}
    model.train()
    train_started = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            chunk = [train[(cursor + i) % len(train)] for i in range(a.micro_batch_size)]
            cursor += a.micro_batch_size
            for x in chunk:
                trained_tokens[x['file']] = trained_tokens.get(x['file'], 0) + sum(1 for t in x['labels'] if t != -100)
            loss = batch_loss(model, chunk, a.device, pad_id) / accumulation
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
        record = {'step': step, 'train_loss': round(total, 5), 'lr': schedule.get_last_lr()[0],
                  'elapsed_seconds': round(time.time() - started, 1)}
        if valid and (step % a.valid_every == 0 or step == steps):
            record['valid_loss_by_task'] = validate(model, valid, a.device, pad_id, a.micro_batch_size)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    a.out.mkdir(parents=True)
    model.save_pretrained(a.out / 'adapter')
    if a.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(a.out / 'merged', safe_serialization=True)
        tok.save_pretrained(a.out / 'merged')
    token_total = sum(trained_tokens.values())
    import peft, transformers
    report = dict(
        protocol=__doc__, base_model_id=a.model_id, base_revision=a.model_revision, family=a.family,
        manifest=str(a.manifest), manifest_sha256=sha256(a.manifest), manifest_id=manifest.get('id'),
        files=[{k: e.get(k) for k in ('path', 'sha256', 'adapter', 'epochs')} for e in manifest['train'] + manifest.get('valid', [])],
        train_rows=len(train), valid_rows=len(valid), dropped_over_max_tokens=dropped, valid_dropped=valid_dropped,
        task_counts={t: sum(1 for x in train if x['task'] == t) for t in sorted({x['task'] for x in train})},
        answer_token_share={f: round(n / token_total, 4) for f, n in sorted(trained_tokens.items())},
        steps=steps, batch_size=a.batch_size, micro_batch_size=a.micro_batch_size, lr=a.lr,
        lora=dict(rank=a.rank, alpha=a.alpha, dropout=a.dropout, targets=TARGETS), max_tokens=a.max_tokens,
        max_answer_tokens=a.max_answer_tokens,
        seed=a.seed, device=a.device, dtype=str(dtype), system_prompt=SYSTEM, case_system_prompt=CASE_SYSTEM,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
        history=history, elapsed_seconds=round(time.time() - started, 1), status='complete')
    (a.out / 'training.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f'complete: {steps} steps -> {a.out}')


if __name__ == '__main__':
    main()
