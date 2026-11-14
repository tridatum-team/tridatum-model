"""Attribute shared reserved signatures to candidate-side official source spans only.

Never opens reserved plaintext, evaluation questions/answers, or private review packets.
The immutable source allocation and original source-screen hashes define the scope.
Produces independent-review evidence, not a source-admission decision. Full-source
contexts can include held articles and are review-only, never author/training material.

  .venv/bin/python scripts/attribute_expanded_source_overlap.py \
    --run runs/mois-expanded-development-20261114T121509Z
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from check_reserved_overlap import Signer, exact_hash, ngrams
from prepare_expanded_source_inventory import normalized_body, file_pin

ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gram_spans(text):
    """Same NFC/whitespace-word eight-grams as check_reserved_overlap.ngrams."""
    words = list(re.finditer(r"\S+", text))
    spans = defaultdict(list)
    for index in range(len(words) - 7):
        start, end = words[index].start(), words[index + 7].end()
        joined = " ".join(word.group() for word in words[index:index + 8])
        key = int.from_bytes(hashlib.blake2b(joined.encode(), digest_size=8).digest(), "little")
        spans[key].append((start, end))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = args.run if args.run.is_absolute() else root / args.run
    destination = run / "source-overlap-attribution.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite frozen attribution")
    allocation_path = run / "candidate-source-allocation.json"
    screen_path = run / "source-overlap-screen.json"
    material_path = run / "candidate-source-material.jsonl"
    allocation = json.loads(allocation_path.read_text())
    screen = json.loads(screen_path.read_text())
    material = defaultdict(dict)
    with material_path.open() as f:
        for line in f:
            row = json.loads(line)
            material[row["family_id"]][row["raw_body_node_ordinal"]] = row
    units = []
    for family in allocation["families"]:
        raw_path = root / family["raw_source"]["path"]
        raw = raw_path.read_bytes()
        if sha(raw) != family["raw_source"]["sha256"]:
            raise ValueError(f"Frozen raw source changed: {raw_path}")
        tree = ET.fromstring(raw)
        nodes = tree.findall("./조문내용") or tree.findall("./조문/조문내용")
        body = "\n\n".join(normalized_body("".join(node.itertext())) for node in nodes)
        if sha(body.encode()) != family["body_sha256"]:
            raise ValueError(f"Frozen body projection disagrees: {family['family_id']}")
        candidates = [(family["family_id"] + ":full_body", body, None)]
        for ordinal, article in material[family["family_id"]].items():
            text = article["text"]
            if sha(text.encode()) != article["quote_sha256"] or body[article["char_start"]:article["char_end"]] != text:
                raise ValueError(f"Source excerpt pin disagrees: {family['family_id']}:{ordinal}")
            candidates.append((family["family_id"] + f":article_{ordinal}", text, article))
        for unit_id, text, article in candidates:
            prior = screen["units"][unit_id]
            if not prior["shared_ngram8_total"]:
                continue
            units.append({"unit_id": unit_id, "family_id": family["family_id"], "split": family["split"],
                "doc_id": family["doc_id"], "issuer": family["issuer"], "source_url": family["source_url"],
                "version": family["version"], "raw_source": family["raw_source"], "body_sha256": family["body_sha256"],
                "text_sha256": sha(text.encode()), "exact_sha256": exact_hash(text),
                "source_scope": "FULL_BODY_EXCLUSION_REVIEW_ONLY" if article is None else "SELECTED_ARTICLE_REVIEW_ONLY",
                "passage_pin": None if article is None else {k: article[k] for k in
                    ("article_id", "raw_body_node_ordinal", "char_start", "char_end", "quote_sha256")},
                "prior_screen": prior, "text": text, "matches": [], "shared_hashes": set()})
    owners = defaultdict(list)
    spans_by_unit = []
    signer = Signer()
    signatures = []
    for index, unit in enumerate(units):
        spans = gram_spans(unit["text"])
        # This checks source attribution against the existing fingerprint convention,
        # not a new approximate tokenizer or an alternate overlap algorithm.
        if set(spans) != ngrams(unit["text"]):
            raise ValueError(f"Fingerprint convention disagrees: {unit['unit_id']}")
        spans_by_unit.append(spans)
        unit["ngram8_count"] = len(spans)
        signatures.append(signer(unit["text"]))
        for gram in spans:
            owners[gram].append(index)
    signatures = np.asarray(signatures, dtype=np.uint64)
    manifest_pins = []
    total_matches = 0
    for manifest in screen["manifests"]:
        path = manifest["path"]
        digest = hashlib.sha256()
        with (root / path).open("rb") as f:
            for line_number, line in enumerate(f, 1):
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                shared = defaultdict(list)
                reserved_grams = set(row["ngram8"])
                for gram in reserved_grams:
                    for index in owners.get(gram, ()):
                        shared[index].append(gram)
                if not shared:
                    continue
                sig = np.asarray(row["minhash"], dtype=np.uint64)
                for index, grams in shared.items():
                    unit = units[index]
                    unit["shared_hashes"].update(grams)
                    unit["matches"].append({
                        "manifest_path": path, "manifest_sha256": manifest["sha256"],
                        "signature_row": line_number, "signature_row_sha256": sha(line),
                        "reserved_signature_metadata": {k: row.get(k) for k in
                            ("family_id", "doc_id", "group", "source_url", "exact_sha256")},
                        "signature_minhash_sha256": sha(json.dumps(row["minhash"], separators=(",", ":")).encode()),
                        "shared_ngram8": len(grams), "shared_ngram8_hashes": sorted(grams),
                        "of_reserved": len(reserved_grams), "of_candidate": unit["ngram8_count"],
                        "reserved_containment": len(grams) / max(1, len(reserved_grams)),
                        "candidate_containment": len(grams) / max(1, unit["ngram8_count"]),
                        "minhash_estimate": float((signatures[index] == sig).mean()),
                        "review_status": "PENDING_INDEPENDENT_REVIEW_NOT_ADMITTED"})
                    total_matches += 1
        if digest.hexdigest() != manifest["sha256"]:
            raise ValueError(f"Frozen signature manifest changed: {path}")
        manifest_pins.append({"path": path, "sha256": digest.hexdigest(), "bytes": (root / path).stat().st_size,
            "rows": manifest["rows"], "unique_family_count": manifest["unique_family_count"]})
    for index, unit in enumerate(units):
        total = sum(match["shared_ngram8"] for match in unit["matches"])
        if total != unit["prior_screen"]["shared_ngram8_total"]:
            raise ValueError(f"Frozen source-screen attribution count disagrees: {unit['unit_id']}")
        text = unit.pop("text")
        hashes = unit.pop("shared_hashes")
        excerpts = []
        for gram in sorted(hashes):
            occurrences = []
            for start, end in spans_by_unit[index][gram]:
                context_start, context_end = max(0, start - 120), min(len(text), end + 120)
                occurrences.append({"char_start": start, "char_end": end,
                    "candidate_side_text": text[start:end], "candidate_side_text_sha256": sha(text[start:end].encode()),
                    "context_char_start": context_start, "context_char_end": context_end,
                    "candidate_side_context": text[context_start:context_end],
                    "context_sha256": sha(text[context_start:context_end].encode())})
            excerpts.append({"ngram8_hash": gram, "occurrences": occurrences})
        unit["candidate_side_shared_ngrams"] = excerpts
        unit["shared_ngram8_total"] = total
        unit["distinct_shared_ngram8"] = len(hashes)
        unit["independent_verdict"] = "PENDING_NOT_ADMITTED"
    def pin(path):
        return file_pin(root, str(path.relative_to(root)))
    result = {"schema": "trid-expanded-source-overlap-attribution-v1", "state": "SOURCE_REVIEW_ONLY_NOT_ADMITTED",
        "source_allocation": pin(allocation_path), "source_screen": pin(screen_path), "source_material": pin(material_path),
        "implementation": pin(Path(__file__).resolve()), "signature_manifests": manifest_pins,
        "thresholds": screen["thresholds"], "counts": {"selected_families": len(allocation["families"]),
            "attributed_selected_units": len(units), "signature_unit_match_pairs": total_matches,
            "shared_ngram8_total_with_signature_repetition": sum(u["shared_ngram8_total"] for u in units),
            "distinct_candidate_ngram8_summed_over_units": sum(u["distinct_shared_ngram8"] for u in units),
            "units_by_partition": dict(Counter(u["split"] for u in units))},
        "verified_scope": "Pinned selected source bodies/excerpts against four reserved text-free manifests and pinned old source fingerprints",
        "limitations": ["Shared snippets and contexts are candidate-side source text only; no reserved plaintext was opened.",
            "Repeated signatures, full-body/article units and repeated occurrences are not independent examples.",
            "Absence of threshold hits is not an admission or semantic-safety decision.",
            "Signature extraction gaps, paraphrases, unobserved versions/agencies and global copy closure remain unverified.",
            "Full-source contexts may include held articles: independent review only, never authoring/training evidence.",
            "Independent reviewers must decide generic boilerplate versus substantive protected-content overlap in this bounded scope."],
        "units": units}
    with destination.open("x") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(destination.relative_to(root)), "sha256": sha(destination.read_bytes()),
        "bytes": destination.stat().st_size, **result["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
