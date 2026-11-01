"""BAAI/bge-m3 dense embeddings (MIT) from the local Hugging Face snapshot; never downloads at run time.

The indexer (scripts/build_embeddings.py) and the service (scripts/shared_demo.py Workspace) share `passage_input`
and the model settings, so query and passage vectors come from the same model, truncation and normalisation.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL = 'BAAI/bge-m3'
MAX_TOKENS = 512


def passage_input(meta, passage):
    """The string embedded for a passage (also its cache key): document identity first, then the verbatim text."""
    m = meta[passage.doc_id]
    return f"{m.get('institution') or ''} {m['title']}\n{passage.text}".strip()


def load_model(device='cpu'):
    """(SentenceTransformer, snapshot path). float16 off the CPU; float32 on the CPU (query-time encoding)."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    path = snapshot_download(MODEL, local_files_only=True)
    model = SentenceTransformer(path, device=device)
    model.max_seq_length = MAX_TOKENS
    if device != 'cpu':
        model.half()
    return model, path


def weight_sha256(path):
    with open(Path(path) / 'pytorch_model.bin', 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_dense(root):
    """The index's passage vectors as float32. `dense.f32.npy` — the exact float32 conversion of `dense.f16.npy`, written
    by `write_dense_f32` with the float16 file's sha256 in `dense.f32.json` — is memory-mapped read-only when that
    record matches: the same values and layout as converting in memory, held as reclaimable file pages instead of
    process memory (SITE-MEMORY-022). Otherwise the float16 file is converted in memory (also read through a map, so
    the float16 copy is not held in process memory during the conversion)."""
    root = Path(root)
    source, copy, record = root / 'dense.f16.npy', root / 'dense.f32.npy', root / 'dense.f32.json'
    if copy.exists() and record.exists() and json.loads(record.read_text()).get('source_sha256') == _sha256(source):
        return np.load(copy, mmap_mode='r')
    return np.asarray(np.load(source, mmap_mode='r').astype(np.float32))  # in memory (astype keeps the memmap type)


def write_dense_f32(root):
    """Write `dense.f32.npy` (exact float16 → float32 conversion) and its record beside `dense.f16.npy`."""
    root = Path(root)
    source = root / 'dense.f16.npy'
    vectors = np.load(source, mmap_mode='r').astype(np.float32)
    temporary = root / 'dense.f32.npy.tmp'
    with open(temporary, 'wb') as f:
        np.save(f, vectors)
    os.replace(temporary, root / 'dense.f32.npy')
    (root / 'dense.f32.json').write_text(json.dumps({'source': source.name, 'source_sha256': _sha256(source),
                                                     'shape': list(vectors.shape), 'dtype': 'float32'}, indent=1) + '\n')


if __name__ == '__main__':  # python -m trid.service.embed write-dense-f32 <index directory>...
    if len(sys.argv) < 3 or sys.argv[1] != 'write-dense-f32':
        sys.exit('usage: python -m trid.service.embed write-dense-f32 <index directory>...')
    for directory in sys.argv[2:]:
        write_dense_f32(directory)
        print(f'{directory}: dense.f32.npy written')
