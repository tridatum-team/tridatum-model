"""Pretraining token folder: the public mix (law, admin rules, ordinances, press releases, Wikipedia) plus a
token-budgeted sample of FineWeb-2 Korean web text (kind `fineweb2_ko`).

FineWeb-2 kor_Hang (ODC-By 1.0; https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) already applies the
FineWeb-2 Korean quality, repetition and MinHash dedup filters, the same thresholds as trid.korean_quality.FINEWEB2.
Each FineWeb-2 document additionally gets, in this order:
  1. NFC, trid.korean_quality.clean_lines (C4-style line cleanup);
  2. trid.korean_quality script-ratio check (Hangul >= 0.6, Hanja <= 0.3, Latin <= 0.3 of letters);
  3. trid.korean_quality.redact (ko-pii, format-checked RRN/FRN/PHONE/EMAIL/ACCOUNT/CARD/PASSPORT/DRIVER_LICENSE);
  4. decontamination: dropped if it shares any whitespace-word 8-gram with an evaluation text
     (RAG310 passages, KLUE MRC official dev contexts, institution-qa-frozen-01, public-document-sft-source).
Parquet files are taken in a seeded order, downloaded one at a time to --cache and deleted after use, until the
train token budget is reached. A held-out FineWeb-2 slice from its `test` split is appended to valid.bin.

Output keeps the trid token format (uint16, <bos> text <eos>, manifest.json, sources.json), so
`pretrain(..., sampling='epoch', source_weights={...})` can up-weight the public kinds.

  python scripts/prepare_fineweb2_ko.py --output data/pretrain-mix-v4-tokens --train-tokens 5500000000
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = 'HuggingFaceFW/fineweb-2'
REVISION_NOTE = 'main at download time; per-file sha256 recorded'
KIND = 'fineweb2_ko'
NGRAM = 8
EVAL_SOURCES = ('data/korean-rag-eval-v1/test.jsonl', 'data/institution-qa-frozen-01', 'data/public-document-sft-source')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def grams(text):
    """Stable 64-bit hashes of whitespace-word n-grams (builtin hash() differs between worker processes)."""
    words = unicodedata.normalize('NFC', text).split()
    return {int.from_bytes(hashlib.blake2b(' '.join(words[i:i + NGRAM]).encode(), digest_size=8).digest(), 'little')
            for i in range(len(words) - NGRAM + 1)}


def eval_texts():
    texts = [p['context'] for rec in json.loads((ROOT / 'data/klue-mrc-source/dev.json').read_text())['data']
             for p in rec['paragraphs']]
    for source in EVAL_SOURCES:
        path = ROOT / source
        files = [path] if path.is_file() else sorted(path.rglob('*.jsonl'))
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    texts += [v for v in row.values() if isinstance(v, str) and len(v) >= 80]
    return texts


_state = {}


def init_worker(tokenizer, banned):
    from tokenizers import Tokenizer
    _state['tok'] = Tokenizer.from_file(tokenizer)
    _state['banned'] = banned


def process(texts):
    """-> (list of uint16 arrays, Counter). Runs in a worker."""
    from trid.korean_quality import FINEWEB2, clean_lines, redact, script_ratios
    from trid.text import SPECIAL
    tok, banned = _state['tok'], _state['banned']
    kept, counts = [], Counter()
    for text in texts:
        text, _ = clean_lines(unicodedata.normalize('NFC', text))
        if not text.strip():
            counts['empty_after_line_cleanup'] += 1
            continue
        hangul, hanja, latin = script_ratios(text)
        if hangul < FINEWEB2.hangul_min or hanja > FINEWEB2.hanja_max or latin > FINEWEB2.latin_max:
            counts['script_ratio'] += 1
            continue
        if any(s in text for s in SPECIAL):
            counts['special_token_text'] += 1
            continue
        if grams(text) & banned:
            counts['eval_overlap'] += 1
            continue
        text, pii = redact(text)
        counts.update({f'pii_{k}': v for k, v in pii.items()})
        kept.append(text)
    bos, eos = tok.token_to_id('<|bos|>'), tok.token_to_id('<|eos|>')
    arrays = [np.asarray([bos] + e.ids + [eos], dtype='<u2') for e in tok.encode_batch(kept)]
    counts['kept'] += len(arrays)
    return arrays, counts


def batches(path, size, limit, read):
    """Stream the text column in record batches (a whole 4.8 GB file does not fit comfortably in memory)."""
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=['text']):
        texts = batch.column('text').to_pylist()
        if limit is not None:
            texts = texts[:max(0, limit - read[0])]
        read[0] += len(texts)
        if not texts:
            return
        yield texts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--public', type=Path, default=ROOT / 'data/mix-v3-basetok-tokens')
    p.add_argument('--train-tokens', type=int, required=True, help='FineWeb-2 train token budget (incl. BOS/EOS)')
    p.add_argument('--valid-docs', type=int, default=3000)
    p.add_argument('--cache', type=Path, default=Path('/tmp/fineweb2-ko-cache'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=291030)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new output directory')
    from huggingface_hub import HfApi, hf_hub_download
    public = json.loads((a.public / 'manifest.json').read_text())
    public_sources = json.loads((a.public / 'sources.json').read_text())
    tokenizer = a.public / 'tokenizer.json'
    if sha256(tokenizer) != public['tokenizer_sha256']:
        raise ValueError('Public tokenizer changed')
    a.output.mkdir(parents=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer, a.output / 'tokenizer.json')
    started = time.time()

    texts = eval_texts()
    banned = set().union(*(grams(t) for t in texts))
    print(f'decontamination: {len(texts):,} evaluation texts, {len(banned):,} word {NGRAM}-grams', flush=True)

    kinds = list(public_sources['kinds']) + [KIND]
    fw_kind = kinds.index(KIND)
    splits, counts, files_used = {}, Counter(), []
    # Public splits first, byte-identical, then FineWeb-2 appended.
    for split in ('train', 'valid', 'test'):
        shutil.copyfile(a.public / f'{split}.bin', a.output / f'{split}.bin')
        s = public_sources['splits'][split]
        splits[split] = {'offsets': list(s['offsets']), 'kind': list(s['kind'])}
    api = HfApi()
    listing = {split: sorted(f.path for f in api.list_repo_tree(REPO, repo_type='dataset', path_in_repo=f'data/kor_Hang/{split}'))
               for split in ('train', 'test')}
    order = listing['train'][:]
    random.Random(a.seed).shuffle(order)
    plan = [('valid', listing['test'][0], a.valid_docs, None)] + [('train', f, None, a.train_tokens) for f in order]
    totals = {'train': 0, 'valid': 0}
    with Pool(a.workers, initializer=init_worker, initargs=(str(tokenizer), banned)) as pool:
        for split, name, doc_limit, token_limit in plan:
            if split == 'train' and totals['train'] >= a.train_tokens:
                break
            local = Path(hf_hub_download(REPO, name, repo_type='dataset', cache_dir=str(a.cache)))
            files_used.append({'split': split, 'path': name, 'bytes': local.stat().st_size, 'sha256': sha256(local)})
            read = [0]
            offset = (a.output / f'{split}.bin').stat().st_size // 2
            with (a.output / f'{split}.bin').open('ab') as out:
                source = batches(local, 2000, doc_limit, read)
                done = False
                while not done:
                    # Bounded window: Pool.imap would read the whole file into its task queue.
                    window = list(islice(source, 2 * a.workers))
                    if not window:
                        break
                    for arrays, c in pool.map(process, window):
                        counts.update(c)
                        for arr in arrays:
                            if token_limit and totals[split] >= token_limit:
                                done = True
                                break
                            arr.tofile(out)
                            splits[split]['offsets'].append(offset)
                            splits[split]['kind'].append(fw_kind)
                            offset += len(arr)
                            totals[split] += len(arr)
                        if done:
                            break
            files_used[-1]['documents_read'] = read[0]
            shutil.rmtree(a.cache, ignore_errors=True)
            a.cache.mkdir(parents=True, exist_ok=True)
            print(f'{name}: {split} fineweb tokens {totals[split]:,} | {json.dumps(dict(counts))} | {time.time() - started:.0f}s', flush=True)

    tokens = {s: (a.output / f'{s}.bin').stat().st_size // 2 for s in ('train', 'valid', 'test')}
    docs = {s: len(splits[s]['offsets']) for s in splits}
    from tokenizers import Tokenizer
    tok_bos = Tokenizer.from_file(str(tokenizer)).token_to_id('<|bos|>')
    for s in ('train', 'valid'):
        arr = np.memmap(a.output / f'{s}.bin', dtype='<u2', mode='r')
        offsets = np.asarray(splits[s]['offsets'], dtype=np.int64)
        bos_count = sum(int((arr[i:i + (1 << 28)] == tok_bos).sum()) for i in range(0, len(arr), 1 << 28))
        if bos_count != len(offsets) or not (arr[offsets] == tok_bos).all():
            raise ValueError(f'{s}: document offsets do not match <|bos|> positions')
    (a.output / 'sources.json').write_text(json.dumps({'format': 'document-start-offsets', 'kinds': kinds, 'splits': splits}))
    by_kind = {}
    for s in ('train', 'valid'):
        offs = np.asarray(splits[s]['offsets'] + [tokens[s]], dtype=np.int64)
        lengths = np.diff(offs)
        by_kind[s] = {kinds[k]: int(lengths[np.asarray(splits[s]['kind']) == k].sum()) for k in range(len(kinds))}
    manifest = {
        'format': 'uint16-little-endian', 'vocab_size': public['vocab_size'], 'documents': docs, 'tokens': tokens,
        'tokens_by_kind': by_kind, 'tokenizer_sha256': sha256(a.output / 'tokenizer.json'),
        'tokenizer_origin': str(tokenizer), 'sha256': {s: sha256(a.output / f'{s}.bin') for s in ('train', 'valid', 'test')},
        'objective': 'continuous next-token prediction; BOS/EOS document separators; all targets supervised',
        'public_part': {'source': str(a.public), 'manifest_sha256': sha256(a.public / 'manifest.json'),
                        'note': 'byte-identical prefix of every split'},
        'fineweb2': {'repo': REPO, 'config': 'kor_Hang', 'license': 'ODC-By 1.0 (https://opendatacommons.org/licenses/by/1-0/); '
                     'underlying web pages keep their own copyright; per-document URLs are in the upstream parquet',
                     'revision': REVISION_NOTE, 'files': files_used, 'train_token_budget': a.train_tokens,
                     'valid_docs_from_test_split': a.valid_docs, 'seed': a.seed, 'filters': (__doc__ or '').split('\n\n')[1],
                     'decontamination': {'ngram_words': NGRAM, 'evaluation_texts': len(texts), 'sources': ['data/klue-mrc-source/dev.json', *EVAL_SOURCES]},
                     'counts': dict(counts)},
        'elapsed_seconds': round(time.time() - started)}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ('documents', 'tokens', 'tokens_by_kind')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
