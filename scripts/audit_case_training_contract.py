"""CPU audit of what a case LoRA run actually supervised (TRAIN-CONTRACT-AUDIT-001). No model weights are loaded and
nothing runs forward: rows go through the trainer's own `load_rows` and `encode`, and the label tensor is built and
shifted exactly as `batch_loss` does, so the answer boundary seen here is the one the loss saw.

Per train row: prompt/answer/total tokens, whether the last supervised target is the EOS token (and which token
that is), how many EOS tokens the answer holds, the last supervised tokens, whether the row was dropped or cut
(max_tokens, max_answer_tokens), whether tokenizing prompt and answer separately equals tokenizing them joined (no
merge across the boundary), sentence repeats inside the target, and the target's format. Run-level: manifest and file
hashes, tokenizer special tokens and generation_config, the chat template of the HF tokenizer against the template
and EOG token ids stored in the served GGUF files, and the serving budgets the answers are compared with.

  uv run --no-project --python 3.12 --with 'torch==2.9.*' --with 'transformers>=4.51,<5' --with 'peft==0.17.*' \
      --with safetensors --with numpy --with tokenizers python scripts/audit_case_training_contract.py \
      --manifest data/case-training-view-v1r1/a-train-manifest.json --model runs/model-repair-002-qwen3-1p7b/base-resaved \
      --merged runs/model-repair-002-qwen3-1p7b/train/merged --training-json runs/model-repair-002-qwen3-1p7b/train/training.json \
      --gguf runs/model-repair-002-qwen3-1p7b/model-repair-002-qwen3-1p7b.q8_0.gguf \
      --gguf runs/model-repair-002-qwen3-1p7b/qwen3-1.7b-base-resaved.q8_0.gguf \
      --gguf-py /tmp/llama.cpp/gguf-py --out reports/coordination/TRAIN-CONTRACT-AUDIT-001/audit.json
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import train_open_weight_lora as trainer  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
SENT = re.compile(r'[^.!?\n]+[.!?]?')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supervised(item):
    """Targets the loss sees for one item at micro batch 1: labels shifted by one, as in trainer.batch_loss."""
    labels = torch.tensor([item['labels']])
    target = labels[:, 1:][0]
    return [int(t) for t in target if t != -100]


def padded_batch_check(items, pad_id, eos_id):
    """Two rows of different length padded as batch_loss pads them: each row's supervised targets end with EOS and the
    padding adds no supervised position."""
    width = max(len(x['ids']) for x in items)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x['ids'])
        ids[i, :n] = torch.tensor(x['ids'])
        labels[i, :n] = torch.tensor(x['labels'])
    target = labels[:, 1:]
    out = []
    for i, x in enumerate(items):
        row = [int(t) for t in target[i] if t != -100]
        out.append({'width': width, 'length': len(x['ids']), 'supervised': len(row), 'last_is_eos': row[-1] == eos_id,
                    'pad_positions_supervised': int(((ids[i] == pad_id) & (torch.arange(width) >= len(x['ids']))
                                                     & torch.cat([target[i] != -100, torch.tensor([False])])).sum())})
    return out


def gguf_meta(path, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    r = GGUFReader(path)

    def field(name):
        f = r.fields.get(name)
        if f is None:
            return None
        v = f.parts[f.data[0]]
        return bytes(v).decode() if f.types and f.types[0].name == 'STRING' else v.tolist()[0]

    tokens = r.fields['tokenizer.ggml.tokens']
    text = lambda i: bytes(tokens.parts[tokens.data[i]]).decode() if i is not None else None
    meta = {k: field(k) for k in ('tokenizer.ggml.eos_token_id', 'tokenizer.ggml.eot_token_id', 'tokenizer.ggml.eom_token_id',
                                  'tokenizer.ggml.padding_token_id', 'tokenizer.ggml.bos_token_id',
                                  'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.add_eos_token')}
    meta['token_text'] = {k: text(v) for k, v in meta.items() if k.endswith('_id') and v is not None}
    meta['chat_template_sha256'] = hashlib.sha256((field('tokenizer.chat_template') or '').encode()).hexdigest()
    meta['file_sha256'] = sha(path)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', required=True, help='the HF dir the run trained from (tokenizer + template)')
    p.add_argument('--merged', required=True)
    p.add_argument('--training-json', required=True)
    p.add_argument('--gguf', action='append', default=[])
    p.add_argument('--gguf-py', required=True)
    p.add_argument('--serving-case-max-tokens', type=int, default=256)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    run = json.loads((ROOT / a.training_json).read_text())
    manifest_path = ROOT / a.manifest
    manifest = json.loads(manifest_path.read_text())
    if sha(manifest_path) != run['manifest_sha256']:
        raise SystemExit('manifest differs from the one the run recorded')
    tok = AutoTokenizer.from_pretrained(ROOT / a.model)
    eos_id, pad_id = tok.eos_token_id, (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    model_dir, merged_dir = ROOT / a.model, ROOT / a.merged
    tokenizer_files = {f: {'model': sha(model_dir / f), 'merged': sha(merged_dir / f)}
                       for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')}

    rows_out, items = [], []
    for entry in manifest['train']:
        rows = trainer.load_rows(entry, manifest_path.parent)  # verifies the pinned sha256
        for row in rows:
            x = trainer.encode(tok, row, run['family'], run['max_tokens'])
            raw = next(json.loads(l) for l in (manifest_path.parent / entry['path']).read_text().splitlines()
                       if l.strip() and json.loads(l)['id'] == row['id'])
            prompt = tok.apply_chat_template(trainer.messages_of(row), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            joint = tok(prompt + row['answer'] + tok.eos_token, add_special_tokens=False)['input_ids']
            sents = [s.strip() for s in SENT.findall(re.sub(r'\[D\d+\]', '', row['answer'])) if len(s.strip()) > 8]
            rec = {'id': row['id'], 'task': row['task'], 'adapter': entry['adapter'], 'epochs': entry.get('epochs', 1),
                   'system_is_case': row['system'] == trainer.CASE_SYSTEM, 'dropped_over_max_tokens': x is None}
            if x is not None:
                sup = supervised(x)
                n_answer = sum(t != -100 for t in x['labels'])
                rec.update(
                    prompt_tokens=len(x['ids']) - n_answer, answer_tokens=n_answer, total_tokens=len(x['ids']),
                    supervised_targets=len(sup), last_supervised_is_eos=sup[-1] == eos_id,
                    eos_in_answer=sum(1 for t in sup if t == eos_id),
                    endoftext_in_answer=sum(1 for t in sup if t == tok.convert_tokens_to_ids('<|endoftext|>')),
                    last_supervised_tokens=[tok.decode([t]) for t in sup[-4:]],
                    last_label_equals_last_id=x['labels'][-1] == x['ids'][-1],
                    over_answer_budget=n_answer > run['max_answer_tokens'],
                    joint_equals_split=joint == x['ids'],
                    target_tail=row['answer'][-24:], last_line=row['answer'].strip().split('\n')[-1][:40],
                    lines=row['answer'].count('\n') + 1, target_ends_with_space=row['answer'] != row['answer'].rstrip(),
                    special_token_text_in_target=any(s in row['answer'] for s in ('<|im_end|>', '<|endoftext|>', '<think>')),
                    stored_answer_tokens=raw.get('answer_tokens'),
                    repeated_sentences=sum(c - 1 for c in Counter(sents).values() if c > 1),
                    starts_with_conclusion_label=row['answer'].startswith('결론:'),
                    cites=bool(re.search(r'\[D\d+\]', row['answer'])), no_evidence_prefix=row['answer'].startswith('확인 불가'))
                items.append(x)
            rows_out.append(rec)
    kept = [r for r in rows_out if not r['dropped_over_max_tokens']]
    answers = sorted(r['answer_tokens'] for r in kept)
    by_len = sorted(items, key=lambda x: len(x['ids']))
    summary = {
        'rows': len(rows_out), 'kept': len(kept), 'dropped_over_max_tokens': len(rows_out) - len(kept),
        'last_supervised_is_eos': sum(r['last_supervised_is_eos'] for r in kept),
        'exactly_one_eos_in_answer': sum(r['eos_in_answer'] == 1 for r in kept),
        'endoftext_supervised': sum(r['endoftext_in_answer'] for r in kept),
        'joint_equals_split': sum(r['joint_equals_split'] for r in kept),
        'over_answer_budget': sum(r['over_answer_budget'] for r in kept),
        'answer_tokens': {'min': answers[0], 'median': statistics.median(answers), 'max': answers[-1],
                          'ge_200': sum(n >= 200 for n in answers), 'ge_240': sum(n >= 240 for n in answers)},
        'stored_counts_match': sum(r['answer_tokens'] == r['stored_answer_tokens'] for r in kept),
        'total_tokens_max': max(r['total_tokens'] for r in kept),
        'repeated_sentence_rows': sum(r['repeated_sentences'] > 0 for r in kept),
        'conclusion_label_rows': sum(r['starts_with_conclusion_label'] for r in kept),
        'no_evidence_rows': sum(r['no_evidence_prefix'] for r in kept),
        'task_counts_unique_rows': dict(Counter(r['task'] for r in rows_out)),
        'adapters': dict(Counter(r['adapter'] for r in rows_out)),
        'system_is_case': sum(r['system_is_case'] for r in rows_out),
        'padded_batch_check': padded_batch_check([by_len[0], by_len[-1]], pad_id, eos_id),
    }
    report = {
        'code_sha256': CODE_SHA256, 'trainer_sha256': sha(ROOT / 'scripts/train_open_weight_lora.py'),
        'manifest': {'path': a.manifest, 'sha256': sha(manifest_path), 'train': manifest['train'],
                     'valid': manifest.get('valid')},
        'run': {k: run[k] for k in ('manifest_sha256', 'train_rows', 'dropped_over_max_tokens', 'task_counts', 'steps',
                                    'batch_size', 'micro_batch_size', 'max_tokens', 'max_answer_tokens', 'versions')},
        'tokenizer': {'eos_token': tok.eos_token, 'eos_id': eos_id, 'pad_token': tok.pad_token, 'pad_id': pad_id,
                      'im_end_id': tok.convert_tokens_to_ids('<|im_end|>'),
                      'endoftext_id': tok.convert_tokens_to_ids('<|endoftext|>'),
                      'generation_config': json.loads((model_dir / 'generation_config.json').read_text()),
                      'files_model_vs_merged': tokenizer_files,
                      'chat_template_sha256': hashlib.sha256((model_dir / 'chat_template.jinja').read_bytes()).hexdigest()},
        'gguf': {Path(g).name: gguf_meta(str(ROOT / g), a.gguf_py) for g in a.gguf},
        'serving_budgets': {'case_serving_max_tokens': a.serving_case_max_tokens,
                            'training_max_answer_tokens_incl_eos': run['max_answer_tokens']},
        'summary': summary, 'rows': rows_out}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('tokenizer', 'gguf', 'summary')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
