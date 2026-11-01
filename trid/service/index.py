"""BM25 over passages (bm25s + kiwipiepy morphemes, same tag set as scripts/benchmark_retrieval.py) plus a
structure graph: reference edges from trid.service.structure (within a document and 「○○규정」 제N조 across
documents by title) and the 조 head of every sub-passage. `search` is ACL-filtered BM25 top-k, then each hit
pulls in its 조 head and the passages it references (hop 1; `hops=2` follows references of references).
Persisted as plain files (json + bm25s arrays); loading never touches the network.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acl import visible
from .structure import external_references, H_JO
from .types import Document, Passage

KIWI_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'SN', 'SL', 'SH'}  # scripts/benchmark_retrieval.py
SUBLABEL = re.compile(r"^[①-⑳]|^\d{1,2}\.|^[가-하]\.|^\d+$")
FORMAT = 1


def tokenize_all(texts):
    """Lower-cased morphemes with a KIWI_TAGS tag, one list per text (kiwi runs multi-threaded on a batch)."""
    from ..korean_quality import kiwi
    return [[t.form.lower() for t in toks if t.tag in KIWI_TAGS] for toks in kiwi().tokenize(list(texts))]


def block_key(path):
    """Path prefix up to the structural unit (조/별표/부칙/¶): sub-labels (항/호/목/split numbers) are cut off."""
    for k in range(1, len(path)):
        if SUBLABEL.match(path[k]):
            return path[:k]
    return path


def _normal_title(title):
    return re.sub(r"[\s「」『』()]", "", title)


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    via: str | None  # None for a BM25 hit; 'head←<pid>' / 'ref←<pid>' for graph expansions


class Index:
    def __init__(self, documents, edges, bm25, texts):
        self.documents = {d.doc_id: d for d in documents}
        self.passages = [p for d in documents for p in d.passages]
        self.by_id = {p.id: p for p in self.passages}
        self.texts = texts
        self.bm25 = bm25
        self.edges = edges  # [(from_pid, to_pid)] references, within and across documents
        self.refs = {}
        for src, dst in edges:
            if src in self.by_id and dst in self.by_id and dst not in self.refs.setdefault(src, []):
                self.refs[src].append(dst)
        self.head = {}
        first = {}
        for p in self.passages:
            key = (p.doc_id, block_key(p.path))
            self.head[p.id] = first.setdefault(key, p.id)

    # ----------------------------------------------------------------------- build / persist

    @classmethod
    def build(cls, documents):
        if not documents:
            raise ValueError("색인할 문서가 없습니다.")
        import bm25s
        passages = [p for d in documents for p in d.passages]
        if not passages:
            raise ValueError("문단이 없습니다.")
        texts = [p.text for p in passages]
        tokens = [t or ["<빈문단>"] for t in tokenize_all(texts)]
        bm25 = bm25s.BM25()
        bm25.index(tokens, show_progress=False)
        edges = [r for d in documents for r in d.refs] + cls.cross_references(documents)
        return cls(documents, edges, bm25, texts)

    @staticmethod
    def cross_references(documents):
        """(from, to) for 「title」 제N조 mentions resolved to another document's 조 passage."""
        by_title = {}
        for d in documents:
            by_title.setdefault(_normal_title(d.title), d)
        jo_of = {}
        for d in documents:
            for p in d.passages:
                jo = next((x for x in p.path[1:] if H_JO.match(x)), None)
                if jo and p.path[1] != "부칙":
                    jo_of.setdefault((d.doc_id, jo), p.id)
        edges = []
        for d in documents:
            for p in d.passages:
                for title, jo in external_references(p.text):
                    key = _normal_title(title)
                    target = by_title.get(key)
                    if target is None:
                        matches = [doc for t, doc in by_title.items() if key in t or t in key]
                        target = matches[0] if len(matches) == 1 else None
                    if target is None or target.doc_id == d.doc_id:
                        continue
                    if (pid := jo_of.get((target.doc_id, jo))) and (p.id, pid) not in edges:
                        edges.append((p.id, pid))
        return edges

    def save(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "documents.json").write_text(json.dumps([d.to_dict() for d in self.documents.values()],
                                                        ensure_ascii=False), encoding="utf-8")
        (root / "graph.json").write_text(json.dumps({"format": FORMAT, "edges": [list(e) for e in self.edges],
                                                     "kiwi_tags": sorted(KIWI_TAGS)}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        self.bm25.save(str(root / "bm25"), show_progress=False)

    @classmethod
    def load(cls, directory):
        import bm25s
        root = Path(directory)
        if not (root / "graph.json").exists():
            raise FileNotFoundError(f"색인이 없습니다: {root} (trid ingest로 먼저 만드세요)")
        graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
        if graph.get("format") != FORMAT:
            raise ValueError("색인 형식이 다릅니다. trid ingest로 다시 만드세요.")
        documents = read_documents(root)
        bm25 = bm25s.BM25.load(str(root / "bm25"), show_progress=False)
        return cls(documents, [tuple(e) for e in graph["edges"]], bm25, [p.text for d in documents for p in d.passages])

    # ----------------------------------------------------------------------- search

    def visible_ids(self, tags):
        return {p.id for p in self.passages if visible(self.documents[p.doc_id].acl, tags)}

    def search(self, query, tags, k=3, hops=1, per_seed=3):
        """[Hit]: ACL-filtered BM25 top-k seeds in score order, then for every seed its 조 head and referenced
        passages (hop 1), then their references (hop 2) ... all ACL-filtered and deduplicated."""
        allowed = self.visible_ids(tags)
        if not allowed:
            return []
        tokens = tokenize_all([query])[0]
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        mask = np.array([p.id in allowed for p in self.passages])
        scores = np.where(mask, scores, -np.inf)
        order = [int(i) for i in np.argsort(-scores, kind="stable")[:k] if scores[i] > 0]
        hits, seen = [], set()
        for i in order:
            p = self.passages[i]
            hits.append(Hit(p, float(scores[i]), None))
            seen.add(p.id)
        pos = {p.id: i for i, p in enumerate(self.passages)}

        def add(pid, via):
            if pid in allowed and pid not in seen:
                seen.add(pid)
                hits.append(Hit(self.by_id[pid], max(float(scores[pos[pid]]), 0.0), via))
                return True
            return False

        frontier = [h.passage.id for h in hits]
        for _ in range(hops):
            next_frontier = []
            for pid in frontier:
                added = 0
                if add(self.head[pid], f"head←{pid}"):
                    added += 1
                for dst in self.refs.get(pid, []):
                    if added >= per_seed:
                        break
                    if add(dst, f"ref←{pid}"):
                        next_frontier.append(dst)
                        added += 1
            frontier = next_frontier
        return hits


def read_documents(root):
    """The documents of `documents.json`, in file order. A current `documents.jsonl` (one document per line, the same
    JSON values; see trid.service.derived) is parsed a line at a time instead of decoding the whole file into one
    string and one parse tree, which held about 0.3 GB of text (4 bytes per character for this corpus) plus the tree
    at startup (SITE-MEMORY-023). Without a current one the file is read exactly as before."""
    from . import derived
    root = Path(root)
    source, copy = root / "documents.json", root / "documents.jsonl"
    if derived.current(source, copy, root / "documents.jsonl.json"):
        with open(copy, encoding="utf-8", newline="\n") as f:
            return [Document.from_dict(json.loads(line)) for line in f]
    return [Document.from_dict(d) for d in json.loads(source.read_text(encoding="utf-8"))]


def write_documents_jsonl(root):
    """Write `documents.jsonl` (each element of `documents.json` as one JSON line) and its record."""
    from . import derived
    root = Path(root)
    source = root / "documents.json"

    def produce(path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for d in json.loads(source.read_text(encoding="utf-8")):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    derived.write(source, root / "documents.jsonl", root / "documents.jsonl.json", produce)
