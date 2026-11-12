#!/usr/bin/env python3
"""Publish a source-validated immutable MOIS RDF release without loading models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trid.service.mois_graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "runs/workspace-index/v5")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mois-graph/v1")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/mois-ontology-v1/facts.jsonl")
    parser.add_argument("--expected-mois-documents", type=int, default=143,
                        help="Fail before publication if exact MOIS inventory changed")
    args = parser.parse_args()
    metadata = json.loads((args.index / "meta.json").read_text(encoding="utf-8"))
    count = sum(row.get("institution") == "행정안전부" for row in metadata.values())
    if count != args.expected_mois_documents:
        parser.error(f"Expected {args.expected_mois_documents} exact MOIS documents, found {count}")
    del metadata
    print(json.dumps(build_graph(args.index, args.output, args.facts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
