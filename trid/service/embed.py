"""BAAI/bge-m3 dense embeddings (MIT) from the local Hugging Face snapshot; never downloads at run time.

The indexer (scripts/build_embeddings.py) and the service (scripts/shared_demo.py Workspace) share `passage_input`
and the model settings, so query and passage vectors come from the same model, truncation and normalisation.
"""
import hashlib
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


def load_dense(root):
    """The index's passage vectors as float32. `dense.f32.npy` — the exact float32 conversion of `dense.f16.npy` — is
    memory-mapped read-only when it is current (`trid.service.derived.current`) and has the source's shape: the same
    values and layout as converting in memory, held as reclaimable file pages instead of process memory
    (SITE-MEMORY-022). Otherwise, including a stale, partial or damaged copy or record, the float16 file is converted
    in memory as before (read through a map, so the float16 copy is not held during the conversion)."""
    from trid.service import derived
    root = Path(root)
    source = np.load(root / 'dense.f16.npy', mmap_mode='r')
    if derived.current(root / 'dense.f16.npy', root / 'dense.f32.npy', root / 'dense.f32.json'):
        try:
            copy = np.load(root / 'dense.f32.npy', mmap_mode='r')
        except (OSError, ValueError):
            copy = None
        if copy is not None and copy.dtype == np.float32 and copy.shape == source.shape:
            return copy
    return np.asarray(source.astype(np.float32))


def write_dense_f32(root):
    """Write `dense.f32.npy` (exact float16 → float32 conversion of `dense.f16.npy`) and its record `dense.f32.json`."""
    from trid.service import derived
    root = Path(root)
    source = root / 'dense.f16.npy'

    def produce(path):
        with open(path, 'wb') as f:
            np.save(f, np.load(source, mmap_mode='r').astype(np.float32))
    derived.write(source, root / 'dense.f32.npy', root / 'dense.f32.json', produce, dtype='float32',
                  shape=list(np.load(source, mmap_mode='r').shape))
