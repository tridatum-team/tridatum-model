"""Prepare a source-only, NOT_ADMITTED expansion proposal from local official DRF bodies.

No evaluation questions, targets, training examples, or review packets are read.
Only original administrative-rule XML bodies are exported, never annexes or tables.
Coverage is a mechanical article inventory, NOT a count of authorable/admitted questions.
Historical TRAIN labels are recorded separately from checkpoint consumption (unknown).

Example (corpus analysis, not training or evaluation):
  .venv/bin/python scripts/prepare_expanded_source_inventory.py \
    --output runs/mois-expanded-development-20261114T121509Z

Existing outputs are never overwritten. Re-run into a fresh directory to compare pins.
Admission requires independent source rights/currentness, substantive exclusion, copy
closure, checkpoint-history and question-capacity reviews. This tool cannot admit data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

from check_reserved_overlap import exact_hash, ngrams
from expanded_source_overlap import scan_units

ROOT = Path(__file__).resolve().parents[1]
HISTORY = "runs/mois-grounded-training-20261112T081606Z"
PREP = "runs/mois-eight-experiment-prep-20261114T053819Z"
SIGNATURES = ["data/benchmark-reserve-001/" + name for name in (
    "signatures-partial-148.jsonl", "signatures-candidate-local-852-b026.jsonl",
    "signatures-candidate-local-reextract-b027.jsonl", "signatures-candidate-local-image-pages-b030.jsonl")]
EXECUTION = "runs/mois-eight-sft-20261114T071851Z/eight-execution-state/eight-export/training/"
METADATA = [
    "data/law-ko-source/provenance.json", "data/internal-rules/v3/docs.jsonl",
    "reports/coordination/benchmark-reserve-001.jsonl",
    HISTORY + "/source-family-allocation.json", PREP + "/source-family-allocation.json",
    HISTORY + "/final24-domain-exclusion-keys.recovered.json",
    HISTORY + "/dev-domain-exclusion-metadata.jsonl",
    HISTORY + "/final24-behavior-exclusion-patterns.recovered.json",
    HISTORY + "/dev-behavior-exclusion-metadata.jsonl",
    "data/qwen-specialization-003/domain-admission-003.json",
    "data/qwen-specialization-001/review-ledger-002.jsonl",
    EXECUTION + "exposure-receipt.json", EXECUTION + "checkpoints.json", EXECUTION + "training.json",
]
FIXED_EXCLUDED = {"law:008715", "law:014318", "admrul:83472", "admrul:70014",
                  "law:009916", "law:005046", "law:005048", "admrul:68101"}
# These are conservative source holds, not an attempted semantic equivalent of old
# private questions. Numerical paraphrases still require independent review.
SUBSTANTIVE_HOLD = re.compile(
    r"국외출장|공무국외|국외여행|채용|연차.{0,12}휴가|취업제한|수의계약|"
    r"금품.{0,12}수수|음식물.{0,12}(?:선물|경조)|정보공개.{0,20}(?:기간|결정)|"
    r"지방공무원.{0,20}(?:보수|수당|복무)")
ASSET_HOLD = re.compile(r"별표|별지|서식|<\s*(?:table|img|iframe)|\t|첨부파일", re.I)
ARTICLE = re.compile(r"^제\s*\d+\s*조(?:의\s*\d+)?\s*\(([^)]+)\)")
NON_SUBSTANTIVE = re.compile(r"^(?:목적|정의|시행일|재검토기한|재검토\s*기한|유효기간|폐지|준용)$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_pin(root: Path, path: str) -> dict:
    h = hashlib.sha256()
    with (root / path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": path, "sha256": h.hexdigest(), "bytes": (root / path).stat().st_size}


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact(text: str) -> str:
    return re.sub(r"[\s·ㆍ]", "", unicodedata.normalize("NFC", text))


def normalized_body(text: str) -> str:
    # Unlike the historical cleaned corpus, retain amendment notes and punctuation.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    return unicodedata.normalize("NFC", html.unescape(text)).strip()


def metadata_exclusions(root: Path):
    families = set(FIXED_EXCLUDED)
    exposure = {}
    for rel in (HISTORY + "/source-family-allocation.json", PREP + "/source-family-allocation.json"):
        data = json.loads((root / rel).read_text())
        for row in data["families"]:
            family = row["family_id"]
            families.add(family)
            exposure.setdefault(family, []).append({"path": rel, "split": row.get("split"),
                "declared_exposure": row.get("exposure", row.get("historical_exposure")),
                "checkpoint_consumption_proved_by_this_artifact": False})
    final = json.loads((root / METADATA[5]).read_text())
    names = {compact(row["statute_key"]) for row in final["keys"]}
    for row in rows(root / (HISTORY + "/dev-domain-exclusion-metadata.jsonl")):
        names.update(compact(v) for v in row.get("statute_keys", []))
    older = json.loads((root / "data/qwen-specialization-003/domain-admission-003.json").read_text())
    # Metadata-only older exclusions are traversed narrowly, never referenced data files.
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "statute" in key and isinstance(child, list):
                    names.update(compact(x) for x in child if isinstance(x, str))
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child)
    visit(older.get("exclusions_applied", {}))
    # Controlled metadata projection only; never follow or emit review text.
    for row in rows(root / "data/qwen-specialization-001/review-ledger-002.jsonl"):
        if row.get("final") == "eval_only_held8":
            names.update(compact(v) for v in row.get("laws_question_answer", []))
    return families, names, exposure


def parse_body(root, serial, source, listing, excluded_names):
    rel = f"data/law-ko-source/raw/admrul/{serial}.xml"
    raw = (root / rel).read_bytes()
    tree = ET.fromstring(raw)
    basic = {child.tag: child.text for child in tree.find("행정규칙기본정보")}
    if basic.get("행정규칙ID") != source["group"].split(":", 1)[1]:
        raise ValueError(f"Stable family disagreement: {rel}")
    if basic.get("소관부처명") != listing.get("소관부처명"):
        raise ValueError(f"Issuer disagreement: {rel}")
    nodes = tree.findall("./조문내용") or tree.findall("./조문/조문내용")
    texts = [normalized_body("".join(node.itertext())) for node in nodes]
    body = "\n\n".join(texts)
    articles, held = [], Counter()
    offset = 0
    seen = set()
    for ordinal, (node, text) in enumerate(zip(nodes, texts), 1):
        reasons = []
        match = ARTICLE.match(text)
        if not match:
            reasons.append("not_numbered_article_body")
        elif NON_SUBSTANTIVE.fullmatch(match[1].strip()):
            reasons.append("non_substantive_heading")
        if len(text) < 180:
            reasons.append("under_180_chars")
        if ASSET_HOLD.search(text) or len(list(node)) or "<" in text and re.search(r"<\s*[A-Za-z]", text):
            reasons.append("attachment_table_or_unverified_markup")
        if SUBSTANTIVE_HOLD.search(text):
            reasons.append("old_substantive_topic_hold")
        if any(name and name in compact(text) for name in excluded_names):
            reasons.append("old_exclusion_statute_reference_hold")
        eh = exact_hash(text)
        if eh in seen:
            reasons.append("duplicate_article_text")
        seen.add(eh)
        if reasons:
            held.update(reasons)
        else:
            article_id = text[:match.end()].split("(", 1)[0].strip()
            articles.append({"article_id": article_id, "article_title": match[1],
                "raw_body_node_ordinal": ordinal, "char_start": offset, "char_end": offset + len(text),
                "quote_sha256": digest(text.encode()), "text": text})
        offset += len(text) + 2
    return {"raw_source": {"path": rel, "sha256": digest(raw), "bytes": len(raw)},
        "body_sha256": digest(body.encode()), "body_chars": len(body), "body_node_count": len(nodes),
        "article_count": sum(bool(ARTICLE.match(t)) for t in texts), "eligible_articles": articles,
        "held_article_reason_counts": dict(held), "body": body,
        "issuer": basic.get("소관부처명"), "effective_date": basic.get("시행일자"),
        "listed_current": basic.get("현행여부"), "version": serial}


def source_unit(record, text, suffix):
    return {"unit_id": record["family_id"] + ":" + suffix, "family_id": record["family_id"],
            "doc_id": record["doc_id"], "source_url": record["source_url"], "text": text}


def copy_components(records, issuers):
    """Conservative lexical components; undiscovered/semantic relationships stay pending."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    issuer_pattern = re.compile("|".join(re.escape(x) for x in sorted(issuers, key=len, reverse=True) if x))
    owner = defaultdict(list)
    sizes = {}
    edges = []
    for i, record in enumerate(records):
        for article in record["eligible_articles"]:
            # Normalize explicit agency identities only to detect institutional copies.
            text = issuer_pattern.sub("기관", article["text"])
            text = re.sub(r"^제\s*\d+\s*조(?:의\s*\d+)?", "조문", text)
            grams = ngrams(text)
            shared = Counter(unit for gram in grams for unit in owner.get(gram, ()))
            linked = set()
            for (other, ordinal), count in shared.items():
                if other != i and count >= 20 and count / max(1, min(len(grams), sizes[(other, ordinal)])) >= .5:
                    linked.add(other)
            for other in sorted(linked):
                a, b = find(i), find(other)
                if a != b:
                    parent[a] = b
                edges.append({"a": record["family_id"], "b": records[other]["family_id"],
                    "method": "issuer_normalized_article_ngram8", "article_id": article["article_id"]})
            key = (i, article["raw_body_node_ordinal"])
            sizes[key] = len(grams)
            for gram in grams:
                owner[gram].append(key)
    return {record["family_id"]: find(i) for i, record in enumerate(records)}, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-families", type=int, default=20)
    parser.add_argument("--train-families", type=int, default=30)
    parser.add_argument("--eval-articles", type=int, default=8)
    parser.add_argument("--train-articles", type=int, default=16)
    parser.add_argument("--pool-factor", type=int, default=3)
    parser.add_argument("--source-only-packets", action="store_true",
        help="Write partition-isolated preliminary source packets; stop before text-signature analysis")
    args = parser.parse_args()
    if min(args.eval_families, args.train_families, args.eval_articles, args.train_articles, args.pool_factor) < 1:
        parser.error("capacities must be positive")
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    names = ["source-inventory.json", "candidate-source-allocation.json", "candidate-source-material.jsonl", "source-overlap-screen.json", "source-file-list.json"]
    if any((out / name).exists() for name in names):
        raise ValueError("Refusing to overwrite existing source freeze; choose a fresh output directory")
    provenance = json.loads((root / METADATA[0]).read_text())
    if provenance["license"] != "KR-Copyright-Act-Art7 (law.go.kr)":
        raise ValueError("Unexpected source license")
    listings = {}
    list_pins = []
    for path in sorted((root / "data/law-ko-source/raw/admrul/lists").glob("page-*.xml")):
        rel = str(path.relative_to(root))
        list_pins.append(file_pin(root, rel))
        for node in ET.parse(path).getroot().findall("admrul"):
            entry = {child.tag: child.text for child in node}
            entry["list_path"] = rel
            listings[entry["행정규칙일련번호"]] = entry
    legacy = {row["id"]: {"split": row.get("split"), "group": row.get("group"),
        "meta": row.get("meta", {}), "source_sha256": row.get("sha256")} for row in rows(root / METADATA[1])}
    reservation = list(rows(root / METADATA[2]))
    reserved = {row["doc_family_id"] for row in reservation}
    excluded, excluded_names, exposure = metadata_exclusions(root)
    source_counts = Counter()
    source_hash = hashlib.sha256()
    source_metadata = []
    retained = {}
    with (root / "data/law-ko-source/docs.jsonl").open("rb") as f:
        for line in f:
            source_hash.update(line)
            row = json.loads(line)
            source_counts[row.get("kind")] += 1
            if row.get("kind") != "admrul":
                continue
            serial = row["id"].rsplit(":", 1)[1]
            if serial not in listings:
                continue
            entry = listings[serial]
            meta = {key: row.get(key) for key in ("id", "group", "title", "sha256", "source", "date", "license")}
            meta["issuer"] = entry.get("소관부처명")
            source_metadata.append(meta)
            if meta["issuer"] == "행정안전부" or row["id"] in legacy:
                retained[serial] = meta
    records, skipped = [], Counter()
    for serial, source in sorted(retained.items()):
        family = source["group"]
        if family in reserved or family in excluded:
            skipped["reserved_or_old_family"] += 1
            continue
        if source["license"] != provenance["license"] or SUBSTANTIVE_HOLD.search(source["title"]):
            skipped["rights_or_substantive_title_hold"] += 1
            continue
        if not source["issuer"]:
            skipped["missing_issuer_provenance_hold"] += 1
            continue
        if re.search(r"[,·ㆍ/]", source["issuer"] or ""):
            skipped["joint_issuer_one_institution_scope_hold"] += 1
            continue
        parsed = parse_body(root, serial, source, listings[serial], excluded_names)
        record = {"family_id": family, "group_key": family, "doc_id": source["id"], "source_id": source["id"],
            "title": source["title"], "source_url": source["source"], "source_record": "data/law-ko-source/docs.jsonl",
            "legacy_cleaned_source_sha256": source["sha256"], "license": source["license"],
            "list_path": listings[serial]["list_path"], "legacy_labels": [legacy[source["id"]]] if source["id"] in legacy else [],
            **parsed}
        records.append(record)
    versions = defaultdict(list)
    for record in records:
        versions[record["family_id"]].append(record)
    canonical = []
    for group in versions.values():
        chosen = max(group, key=lambda r: (r["effective_date"] or "", int(r["version"])))
        chosen["observed_versions"] = [{"doc_id": r["doc_id"], "version": r["version"], "body_sha256": r["body_sha256"],
            "raw_source": r["raw_source"]} for r in group]
        chosen["legacy_labels"] = [label for r in group for label in r["legacy_labels"]]
        canonical.append(chosen)
    eligible_eval = [r for r in canonical if r["issuer"] == "행정안전부" and len(r["eligible_articles"]) >= args.eval_articles]
    eligible_train = [r for r in canonical if r["issuer"] != "행정안전부" and len(r["eligible_articles"]) >= args.train_articles]
    # Moderate-sized bodies are easier to independently review than giant manuals.
    def rank(record):
        return (abs(len(record["eligible_articles"]) - 24), record["body_chars"], record["family_id"])
    eval_pool = sorted(eligible_eval, key=rank)[:args.eval_families * args.pool_factor]
    # Round-robin issuer diversity; never combine rules from institutions in one case.
    by_issuer = defaultdict(list)
    for r in sorted(eligible_train, key=rank):
        by_issuer[r["issuer"]].append(r)
    train_pool = []
    while len(train_pool) < args.train_families * args.pool_factor and any(by_issuer.values()):
        for issuer in sorted(by_issuer):
            if by_issuer[issuer] and len(train_pool) < args.train_families * args.pool_factor:
                train_pool.append(by_issuer[issuer].pop(0))
    pool = eval_pool + train_pool
    old_copy_sources = []
    for serial, source in sorted(retained.items()):
        if source["group"] in excluded and source["group"].startswith("admrul:"):
            parsed = parse_body(root, serial, source, listings[serial], set())
            old_copy_sources.append({"family_id": source["group"], **parsed})
    components, edges = copy_components(pool + old_copy_sources, {r["issuer"] for r in canonical})
    old_components = {components[r["family_id"]] for r in old_copy_sources}
    if args.source_only_packets:
        packet_index = []
        eval_pool_components = {components[r["family_id"]] for r in eval_pool}
        for split, candidates, target in (("EVAL_CANDIDATE", eval_pool, args.eval_articles),
                                         ("TRAIN_CANDIDATE", train_pool, args.train_articles)):
            for record in candidates:
                component = components[record["family_id"]]
                if component in old_components or split == "TRAIN_CANDIDATE" and component in eval_pool_components:
                    continue
                packet = {k: v for k, v in record.items() if k not in ("body", "eligible_articles")}
                packet.update({"split": split, "state": "NOT_ADMITTED", "planned_rows": target,
                    "article_coverage_count": len(record["eligible_articles"]), "copy_component": component,
                    "rights": "DECLARED_ARTICLE7_OFFICIAL_BODY_ONLY_INDEPENDENT_REVIEW_PENDING",
                    "history": "CHECKPOINT_CONSUMPTION_UNPROVEN_NOT_HISTORICALLY_UNSEEN",
                    "overlap": "PENDING_NOT_ADMITTED_FINAL_ALLOCATION_MAY_HOLD_THIS_FAMILY",
                    "behavior_coverage": "EIGHT_BEHAVIOR_AUTHORABILITY_NOT_ESTABLISHED_BY_ARTICLE_COUNT",
                    "case_scope": "ONE_INSTITUTION_ONE_FAMILY_PROVIDED_EVIDENCE_ONLY",
                    "articles": record["eligible_articles"]})
                rel = Path("candidate-source-packets") / split / (record["family_id"].replace(":", "-") + ".json")
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as f:
                    f.write(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
                packet_index.append({"family_id": record["family_id"], "split": split, "path": str(rel),
                    "sha256": digest(path.read_bytes()), "article_coverage_count": len(record["eligible_articles"])})
        index = {"state": "NOT_ADMITTED", "families": packet_index,
            "mechanically_rich_mois_families": len(eligible_eval),
            "mechanically_rich_other_issuer_families": len(eligible_train),
            "warning": "Preliminary packets; final signature and semantic decisions can hold them. Do not expose EVAL packets to TRAIN authors."}
        with (out / "source-only-authoring-packet-index.json").open("x") as f:
            f.write(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"packets_by_split": dict(Counter(r["split"] for r in packet_index)),
            "mechanically_rich_mois_families": len(eligible_eval),
            "mechanically_rich_other_issuer_families": len(eligible_train)}, ensure_ascii=False))
        return
    units = []
    for record in pool:
        units.append(source_unit(record, record["body"], "full_body"))
        for article in record["eligible_articles"]:
            units.append(source_unit(record, article["text"], f"article_{article['raw_body_node_ordinal']}"))
    # All four signature sets plus the source-only old final exclusion closure.
    signature_paths = SIGNATURES + [HISTORY + "/evaluation-source-closure-v1/source-fingerprints.jsonl"]
    overlap = scan_units(root, units, signature_paths)
    risky = {r["family_id"] for r in pool if components[r["family_id"]] in old_components}
    by_family = defaultdict(list)
    for unit in units:
        result = overlap["units"][unit["unit_id"]]
        by_family[unit["family_id"]].append({"unit_id": unit["unit_id"], **result})
        if result["hits"] or result["partial_review"]:
            risky.add(unit["family_id"])
    selected = []
    # Keep all preliminary EVAL packet components out of TRAIN, even if not
    # selected later: source-only packets may already have reached EVAL authors.
    eval_components = {components[r["family_id"]] for r in eval_pool}
    for record in eval_pool:
        if record["family_id"] not in risky and len(selected) < args.eval_families:
            selected.append(("EVAL_CANDIDATE", record, args.eval_articles))
    train_count = 0
    copy_blocked = []
    for record in train_pool:
        if components[record["family_id"]] in eval_components:
            copy_blocked.append(record["family_id"])
            continue
        if record["family_id"] not in risky and train_count < args.train_families:
            selected.append(("TRAIN_CANDIDATE", record, args.train_articles))
            train_count += 1
    allocation, material = [], []
    for split, record, target in selected:
        clean = {k: v for k, v in record.items() if k not in ("body", "eligible_articles")}
        clean.update({"split": split, "admission_status": "NOT_ADMITTED", "planned_rows": target,
            "article_coverage_count": len(record["eligible_articles"]), "copy_component": components[record["family_id"]],
            "rights": {"declared_basis": provenance["license"], "scope": "official numbered administrative-rule body only",
                "independent_review": "PENDING_NOT_ADMITTED", "attachments_tables_private_material": "EXCLUDED"},
            "historical_exposure": {"checkpoint_consumption": "UNKNOWN_NOT_CERTIFIED", "historically_unseen": False,
                "legacy_train_label_is_consumption_proof": False},
            "copy_risk": {"observed_cross_partition_edges": 0, "semantic_cross_version_cross_agency_closure": "PENDING_NOT_ADMITTED",
                "signature_screen": "NO_THRESHOLD_HIT_OBSERVED_NOT_CLEARANCE", "report": "source-overlap-screen.json"},
            "currentness": "OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED",
            "evidence_admission": "PENDING_NOT_ADMITTED", "question_capacity_review": "PENDING_NOT_ADMITTED",
            "case_scope": "ONE_INSTITUTION_ONE_FAMILY_PROVIDED_EVIDENCE_ONLY",
            "passage_pins": [{k: v for k, v in a.items() if k != "text"} for a in record["eligible_articles"]]})
        allocation.append(clean)
        for article in record["eligible_articles"]:
            material.append({"family_id": record["family_id"], "doc_id": record["doc_id"], "issuer": record["issuer"],
                "source_url": record["source_url"], "source_version": record["version"], "split": split,
                "admission_status": "NOT_ADMITTED", "use": "SOURCE_ONLY_AUTHOR_REVIEW_NOT_TRAINING", **article})
    inventory_records = [{k: v for k, v in r.items() if k not in ("body", "eligible_articles")} |
        {"eligible_article_count": len(r["eligible_articles"]), "signature_screened": r in pool}
        for r in canonical]
    counts = {"original_corpus_rows_by_kind": dict(source_counts), "listed_administrative_rule_versions": len(listings),
        "listed_mois_versions": sum(v.get("소관부처명") == "행정안전부" for v in listings.values()),
        "retained_administrative_rule_records": len(source_metadata),
        "retained_mois_records": sum(v["issuer"] == "행정안전부" for v in source_metadata),
        "retained_mois_unique_families": len({v["group"] for v in source_metadata if v["issuer"] == "행정안전부"}),
        "parsed_rule_records": len(records), "parsed_unique_families": len(canonical), "preparse_holds": dict(skipped),
        "mechanically_rich_mois_families": len(eligible_eval), "mechanically_rich_other_issuer_families": len(eligible_train),
        "signature_screened_families": len(pool), "signature_screened_units": len(units),
        "signature_threshold_or_partial_hold_families": len(risky), "cross_partition_copy_hold_families": len(copy_blocked),
        "allocated_eval_families": len(selected) - train_count, "allocated_train_families": train_count,
        "allocated_eval_article_spans": sum(len(r["eligible_articles"]) for s,r,_ in selected if s == "EVAL_CANDIDATE"),
        "allocated_train_article_spans": sum(len(r["eligible_articles"]) for s,r,_ in selected if s == "TRAIN_CANDIDATE"),
        "authored_questions": 0, "admitted_questions": 0, "admitted_training_rows": 0,
        "general_behavior_sources_or_rows_prepared": 0}
    pins = [file_pin(root, rel) for rel in METADATA]
    pins.append({"path": "data/law-ko-source/docs.jsonl", "sha256": source_hash.hexdigest(),
        "bytes": (root / "data/law-ko-source/docs.jsonl").stat().st_size})
    out.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    limitations = ["No question-count or historical-unseen claim follows from source/article counts.",
        "Rights, official concordance/currentness, semantic exclusion and copy closure remain unreviewed.",
        "All reserved and old allocated family identities are held; substantive lexical holds are conservative, not semantic proof.",
        "Observed copies are component-bound; unobserved copies and linked assets remain NOT_ADMITTED.",
        "Non-MOIS pool uses existing internal-rule metadata membership joined to original official bodies.",
        "Signatures can have extraction gaps; zero threshold hits are not an admission decision.",
        "General 40 EVAL / 120 TRAIN rows require separate behavior source/authoring preparation."]
    receipt = json.loads((root / (EXECUTION + "exposure-receipt.json")).read_text())
    checkpoints = json.loads((root / (EXECUTION + "checkpoints.json")).read_text())
    known_consumption = {"status": "NARROW_LOCAL_SFT_RECEIPT_NOT_BROADER_CORPUS_PROOF",
        "unique_training_rows": receipt["unique_training_rows"], "training_exposures": receipt["training_exposures"],
        "development_rows": receipt["development_rows"], "manifest_sha256": receipt["manifest_sha256"],
        "base_model_id": checkpoints["base_model_id"], "base_revision": checkpoints["base_revision"],
        "init_adapter": checkpoints.get("init_adapter"), "selection": checkpoints.get("selection"),
        "family_allocation_evidence": PREP + "/source-family-allocation.json",
        "broader_original_corpus_consumption": "UNPROVEN", "base_pretraining_exposure": "UNPROVEN",
        "historically_unseen_claim": False}
    save("source-inventory.json", {"schema_version": 1, "state": "SOURCE_ONLY_NOT_ADMITTED", "counts": counts,
        "inputs": pins, "list_inputs": list_pins, "original_administrative_rule_metadata": source_metadata,
        "body_coverage_inventory": inventory_records, "old_family_exclusions": sorted(excluded),
        "old_statute_reference_holds": sorted(excluded_names), "old_declared_exposure_metadata": exposure,
        "known_checkpoint_consumption_metadata": known_consumption,
        "reserved_family_count": len(reserved), "limitations": limitations,
        "body_projection": "NFC and HTML entity decoding; br to newline; raw article nodes joined by two newlines; amendment notes retained"})
    save("candidate-source-allocation.json", {"schema_version": 1, "state": "NOT_ADMITTED", "counts": counts,
        "proposed_capacity": {"eval_families": args.eval_families, "eval_grounded_questions": args.eval_families * args.eval_articles,
            "train_families": args.train_families, "train_grounded_rows": args.train_families * args.train_articles,
            "general_eval_questions": 40, "general_train_rows": 120},
        "capacity_basis": "Distinct standalone numbered body articles; proposed one row per distinct article, independent authorability review pending",
        "families": allocation, "copy_relationships": edges, "copy_blocked_train_families": copy_blocked,
        "signature_held_families": sorted(risky), "limitations": limitations,
        "required_admission": ["source-specific Article7 body rights/concordance", "full reserved/old substantive exclusion review",
            "cross-version and cross-agency semantic copy closure", "actual checkpoint consumption provenance",
            "independent distinct-question capacity review", "later authored candidate overlap and semantic review"],
        "immutable_history": "Original8B scores, no-selection, final24 and exact24/8 contracts untouched"})
    save("source-overlap-screen.json", overlap)
    with (out / "candidate-source-material.jsonl").open("w") as f:
        for row in material:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_raw = {v["raw_source"]["path"]: v["raw_source"] for _,r,_ in selected for v in r["observed_versions"]}
    save("source-file-list.json", {"metadata_inputs": pins, "list_inputs": list_pins,
        "signature_inputs": overlap["manifests"], "selected_raw_source_inputs": list(selected_raw.values()),
        "implementation": [file_pin(root, "scripts/prepare_expanded_source_inventory.py"), file_pin(root, "scripts/expanded_source_overlap.py")],
        "generated_outputs": [file_pin(root, str((out / name).relative_to(root))) for name in names if name != "source-file-list.json"]})
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
