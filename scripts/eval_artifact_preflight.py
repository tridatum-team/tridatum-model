"""Artifact preflight for an offline service evaluation (EVAL-PREFLIGHT-001).

An evaluation that runs the service code from a pinned checkout (git worktree) must read the same artifacts the live
service reads. Files the code reads that git does not track (institution aliases, the ALIO list, reviewed table rows,
link files, …) are absent from a fresh worktree, and code that guards them with `exists()` falls back silently; C-030
measured an alias "regression" that way. This script builds the service `Workspace` from the code under test while
recording every file it opens and every existence check it makes, runs `select()` on fixed neutral probe questions
(not evaluation items; no generation model is called), and classifies each path under the code root:

  ok                  present in the evaluated checkout (sha256, size, git-tracked recorded)
  missing             absent in the checkout but present at the same relative path in the reference checkout (the
                      deployed repository): an accidental gap -> exit status 2, the evaluation must not run
  absent_intentional  absent in both: the code at this version treats it as optional (recorded, not an error)

Paths outside the code root (index directories passed explicitly, model caches) are recorded with their sha256.
The manifest is written to --output (JSON). Usage:

  .venv/bin/python scripts/eval_artifact_preflight.py --code /tmp/trid-<commit> --index runs/workspace-index/v5 \
      --review data/search-corpus-v4/passages.jsonl --output runs/c-private/preflight-<tag>.json
"""
import argparse
import builtins
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1]
PROBES = ('안녕하세요', '출장 여비 지급 기준', '연가 사용 원칙', '별표 지급표 금액')  # neutral, not evaluation items
SKIP_SHA_BYTES = None  # hash every file; large index files are hashed by streaming


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def trace(code, index, review):
    """Import the service from `code`, construct the Workspace and run the probes; return {path: {'read','checked'}}."""
    seen = {}

    def note(path, kind):
        try:
            p = Path(os.fspath(path))
        except TypeError:
            return
        if not p.is_absolute():
            p = Path.cwd() / p
        seen.setdefault(str(p), set()).add(kind)

    real_open, real_io_open = builtins.open, io.open
    real_exists, real_os_exists, real_is_file = Path.exists, os.path.exists, Path.is_file

    def t_open(file, *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)):
            note(file, 'read')
        return real_open(file, *a, **k)

    def t_io_open(file, *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)):
            note(file, 'read')
        return real_io_open(file, *a, **k)

    def t_exists(self, *a, **k):
        r = real_exists(self, *a, **k)
        note(self, 'checked')
        return r

    def t_os_exists(path):
        r = real_os_exists(path)
        note(path, 'checked')
        return r

    def t_is_file(self, *a, **k):
        r = real_is_file(self, *a, **k)
        note(self, 'checked')
        return r

    builtins.open, io.open = t_open, t_io_open
    Path.exists, os.path.exists, Path.is_file = t_exists, t_os_exists, t_is_file
    try:
        sys.path[:0] = [str(code), str(code / 'scripts')]
        import shared_demo as sd
        if Path(sd.__file__).resolve().parents[1] != code.resolve():
            raise SystemExit(f'imported shared_demo from {sd.__file__}, not from {code}')
        ws = sd.Workspace(index, review=review)
        for q in PROBES:
            ws.select(q)
        pinned = [(Path(path), sha) for path, sha in getattr(sd, 'PINNED_CONTRACT', ())]
    finally:
        builtins.open, io.open = real_open, real_io_open
        Path.exists, os.path.exists, Path.is_file = real_exists, real_os_exists, real_is_file
    return seen, pinned


def check_pinned(pinned, code):
    """The B data contracts the service pins by sha256 (shared_demo.PINNED_CONTRACT) must be in the evaluated checkout,
    git-tracked and byte-identical: a missing or untracked contract file is never an optional artifact (C-030)."""
    rows = []
    for path, sha in pinned:
        rel = path.resolve().relative_to(code.resolve())
        here = code / rel
        row = {'path': str(rel), 'pinned_sha256': sha, 'exists': here.is_file(), 'git_tracked': tracked(code, str(rel))}
        row['sha256_equal'] = here.is_file() and sha256(here) == sha
        row['status'] = 'ok' if row['exists'] and row['git_tracked'] and row['sha256_equal'] else 'fail'
        rows.append(row)
    return rows


def tracked(code, rel):
    r = subprocess.run(['git', '-C', str(code), 'ls-files', '--error-unmatch', rel], capture_output=True)
    return r.returncode == 0


def classify(seen, code, reference):
    rows = []
    roots = (code, code.resolve())
    ref_r = reference.resolve()
    for raw, kinds in sorted(seen.items()):
        p = Path(raw)
        rel = None
        for root in roots:  # unresolved first: a file inside a symlinked data directory still belongs to the checkout
            try:
                rel = p.relative_to(root)
                break
            except ValueError:
                continue
        if rel is None:
            # outside the code root: explicit index paths, model caches, interpreter files
            if any(s in raw for s in ('/site-packages/', '/lib/python', '/.venv/', '/__pycache__/')):
                continue
            if not p.exists():
                # an index or data path the code looked for and the deployment does not have either (e.g. facts.jsonl
                # of an index built before structured facts): the version's own optional artifact, not a gap
                for base in {ref_r, reference}:
                    try:
                        if p.relative_to(base).parts[0] in ('data', 'runs', 'deploy'):
                            rows.append({'path': raw, 'scope': 'external', 'status': 'absent_intentional',
                                         'kinds': sorted(kinds)})
                            break
                    except (ValueError, IndexError):
                        continue
                continue
            if not p.is_file():
                continue
            rows.append({'path': raw, 'scope': 'external', 'status': 'ok', 'kinds': sorted(kinds),
                         'bytes': p.stat().st_size, 'sha256': sha256(p)})
            continue
        if any(part in ('__pycache__',) for part in rel.parts) or rel.suffix == '.py':
            continue
        here, there = code / rel, ref_r / rel
        row = {'path': str(rel), 'scope': 'code_root', 'kinds': sorted(kinds), 'git_tracked': tracked(code, str(rel)),
               'symlink': here.is_symlink()}
        if here.is_file():
            row.update(status='ok', bytes=here.stat().st_size, sha256=sha256(here))
            if there.is_file() and not row['git_tracked']:
                row['reference_sha256_equal'] = sha256(there) == row['sha256']
        elif here.is_dir():
            row.update(status='ok', directory=True)
        elif there.exists():
            row.update(status='missing', reference_exists=True)
        else:
            row.update(status='absent_intentional', reference_exists=False)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--code', type=Path, required=True, help='checkout whose service code is evaluated')
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--review', type=Path, default=None)
    p.add_argument('--reference', type=Path, default=REFERENCE, help='deployed repository (default: this one)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code = a.code if a.code.is_absolute() else Path.cwd() / a.code
    index = a.index if a.index.is_absolute() else Path.cwd() / a.index
    review = (a.review if a.review.is_absolute() else Path.cwd() / a.review) if a.review else None
    commit = subprocess.run(['git', '-C', str(code), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    seen, pinned = trace(code, index, review)
    rows = classify(seen, code, a.reference)
    contract = check_pinned(pinned, code)
    missing = [r['path'] for r in rows if r['status'] == 'missing']
    mismatched = [r['path'] for r in rows if r.get('reference_sha256_equal') is False]
    contract_failed = [r['path'] for r in contract if r['status'] != 'ok']
    report = {'code': str(code), 'commit': commit, 'index': str(index), 'review': str(review) if review else None,
              'reference': str(a.reference), 'probes': list(PROBES), 'artifacts': rows, 'pinned_contract': contract,
              'summary': {'ok': sum(r['status'] == 'ok' for r in rows), 'missing': missing,
                          'absent_intentional': [r['path'] for r in rows if r['status'] == 'absent_intentional'],
                          'untracked_differs_from_reference': mismatched, 'pinned_contract_failed': contract_failed},
              'verdict': 'fail' if missing or mismatched or contract_failed else 'pass'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({'commit': commit[:12], 'verdict': report['verdict'], 'ok': report['summary']['ok'],
                      'missing': missing, 'absent_intentional': report['summary']['absent_intentional'],
                      'untracked_differs': mismatched, 'pinned_contract_failed': contract_failed}, ensure_ascii=False))
    sys.exit(2 if report['verdict'] == 'fail' else 0)


if __name__ == '__main__':
    main()
