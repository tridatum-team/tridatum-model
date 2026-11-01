"""Derived index files: exact, cheaper-to-load forms of an index file, used only while provably current (SITE-MEMORY-023).

A derived file (`copy`) is written from `source` and installed with a record (`record`, JSON) naming both files' sha256.
The producer hashes the source before and after producing the copy and installs nothing if it changed, so a record
never pairs a copy with a different source. Copy and record are each installed by an atomic rename. A consumer uses
the copy only when the record parses and both hashes match the files now on disk; anything else (missing, stale,
partial or damaged copy or record) means the original source file is read instead, exactly as before.

    python -m trid.service.derived <index directory>...   writes dense.f32.npy and documents.jsonl with their records
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def current(source, copy, record):
    """True when `copy` is the complete derived form of the current `source` (see the module docstring)."""
    try:
        r = json.loads(Path(record).read_text())
        return r.get('source_sha256') == sha256(source) and r.get('copy_sha256') == sha256(copy)
    except (OSError, ValueError, AttributeError):  # missing or unreadable file, malformed record
        return False


def write(source, copy, record, produce, **extra):
    """Write `copy` with `produce(path)` from `source` and install it with its record; raise (installing nothing) if the
    source changed while the copy was produced."""
    source, copy, record = Path(source), Path(copy), Path(record)
    before = sha256(source)
    temporary = copy.with_name(copy.name + '.tmp')
    produce(temporary)
    if sha256(source) != before:
        temporary.unlink()
        raise RuntimeError(f'{source} changed while {copy.name} was written; nothing installed')
    os.replace(temporary, copy)
    entry = {'source': source.name, 'source_sha256': before, 'copy': copy.name, 'copy_sha256': sha256(copy), **extra}
    temporary = record.with_name(record.name + '.tmp')
    temporary.write_text(json.dumps(entry, ensure_ascii=False, indent=1) + '\n')
    os.replace(temporary, record)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python -m trid.service.derived <index directory>...')
    from trid.service.embed import write_dense_f32
    from trid.service.index import write_documents_jsonl
    for directory in sys.argv[1:]:
        if (Path(directory) / 'dense.f16.npy').exists():
            write_dense_f32(directory)
        write_documents_jsonl(directory)
        print(f'{directory}: derived files written')
