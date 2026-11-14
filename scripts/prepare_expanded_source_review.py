"""Build exclusively-created, partition-isolated source review evidence; never admit it.

Usage: python scripts/prepare_expanded_source_review.py --run runs/NEW_RUN --packet-suffix v2
This is offline source concordance analysis, not a test, evaluation, or model call.
Only frozen source material, official XML, text-free signatures and provenance
metadata are read. Questions, answers and independent review packets are not read.
The 50-family scope is bounded: unknown global semantic copies and foundation
training history remain disclosed limitations, not universal closure prerequisites.
Exactly six source verdict dimensions are requested. Question capacity is deferred
to actual authored cases and their independent semantic reviews. Attribution
projection retains every candidate-side gram/context and matched family identity;
only repeated per-signature associations are aggregated into family summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from check_reserved_overlap import url_key
from prepare_expanded_source_inventory import (
    ARTICLE, ASSET_HOLD, EXECUTION, NON_SUBSTANTIVE, ROOT, SIGNATURES,
    SUBSTANTIVE_HOLD, compact, digest, file_pin, normalized_body, rows,
)

DIMENSIONS = (
    "rights", "evidence", "bounded_copy_review",
    "protected_exclusions", "signature_review", "history_review",
)
LIMITATIONS = [
    "Source-only mechanical findings are not independent verdicts or admission.",
    "Currentness is the official collected snapshot, not rechecked; no live fetch.",
    "Known reserved identity/signature and observed copy-component checks do not prove semantic exclusion.",
    "Unresolved global, cross-agency and cross-version semantic copy closure is a disclosed limit, not an impossible universal prerequisite.",
    "Below-threshold overlap needs independent source-overlap-attribution.json review; zero threshold hits are not clearance.",
    "Source-specific public rights, currentness fitness and old substantive/semantic exclusions need independent review.",
    "Article count does not establish distinct question capacity: actual later cases and semantic reviews are required.",
    "Foundation/base-pretraining history and broader original-corpus checkpoint consumption are unknown; no historically-unseen claim.",
]


def save_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(root: Path, run: Path, packet_suffix: str = "") -> dict:
    def load(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    allocation = load("candidate-source-allocation.json")
    inventory = load("source-inventory.json")
    overlap = load("source-overlap-screen.json")
    files = load("source-file-list.json")
    families = allocation["families"]
    selected = {f["family_id"]: f for f in families}
    if len(families) != 50 or len(selected) != 50:
        raise ValueError("Review scope requires precisely 50 unique frozen allocation families")
    if any(f["split"] not in {"EVAL_CANDIDATE", "TRAIN_CANDIDATE"} for f in families):
        raise ValueError("Unexpected source partition")
    if packet_suffix and not packet_suffix.replace("-", "").isalnum():
        raise ValueError("Packet suffix must contain only letters, digits or hyphens")
    suffix = "-" + packet_suffix if packet_suffix else ""
    targets = [run / ("source-review-packets" + suffix) / f["split"] /
               (f["family_id"].replace(":", "-") + ".json") for f in families]
    index_path = run / ("source-review-packet-index" + suffix + ".json")
    if any(p.exists() for p in [*targets, index_path]):
        raise FileExistsError("Refusing to replace existing review packets or index")

    pin_cache = {}
    def pin(path):
        rel = str(Path(path).relative_to(root)) if Path(path).is_absolute() else str(path)
        if rel not in pin_cache:
            pin_cache[rel] = file_pin(root, rel)
        return pin_cache[rel]

    mismatches = []
    def compare(label, actual, expected, family=None):
        same = actual == expected
        if not same:
            mismatches.append({"family_id": family, "check": label,
                               "expected": expected, "actual": actual})
        return same

    checked_pins = []
    def check_pin(record):
        actual = pin(record["path"])
        agrees = compare("file_sha256:" + record["path"], actual["sha256"], record["sha256"])
        if "bytes" in record:
            agrees = compare("file_bytes:" + record["path"], actual["bytes"], record["bytes"]) and agrees
        checked_pins.append({"recorded": record, "observed": actual, "concordant": agrees})

    # Hash metadata as opaque bytes; never follow references to historical questions.
    recorded = (inventory["inputs"] + inventory["list_inputs"] + files["generated_outputs"]
                + files["implementation"] + files["selected_raw_source_inputs"] + overlap["manifests"])
    seen_pins = set()
    for record in recorded:
        key = (record["path"], record["sha256"], record.get("bytes"))
        if key not in seen_pins:
            check_pin(record)
            seen_pins.add(key)
    inputs = {name: pin(run / name) for name in (
        "candidate-source-allocation.json", "source-inventory.json",
        "candidate-source-material.jsonl", "source-overlap-screen.json", "source-file-list.json")}
    provenance_path = "data/law-ko-source/provenance.json"
    provenance = json.loads((root / provenance_path).read_text(encoding="utf-8"))
    reservation_path = "reports/coordination/benchmark-reserve-001.jsonl"
    reservations = list(rows(root / reservation_path))
    reserved_ids = {r["doc_family_id"] for r in reservations}
    compare("reserved_identity_count", len(reserved_ids), 1000)
    compare("inventory_reserved_count", inventory["reserved_family_count"], len(reserved_ids))

    signature_summaries = []
    reserved_signature_ids = set()
    old_signature_ids = set()
    for manifest in overlap["manifests"]:
        signatures = list(rows(root / manifest["path"]))
        ids = {r["family_id"] for r in signatures if r.get("family_id")}
        compare("signature_rows:" + manifest["path"], len(signatures), manifest["rows"])
        compare("signature_families:" + manifest["path"], sorted(ids), sorted(manifest["unique_family_ids"]))
        (reserved_signature_ids if manifest["path"] in SIGNATURES else old_signature_ids).update(ids)
        signature_summaries.append({k: v for k, v in manifest.items() if k != "unique_family_ids"})
    compare("four_reserved_manifest_paths", sorted(m["path"] for m in overlap["manifests"]
            if m["path"] in SIGNATURES), sorted(SIGNATURES))
    compare("reserved_signature_identity_union", sorted(reserved_signature_ids), sorted(reserved_ids))
    compare("all_signature_union_count", len(reserved_signature_ids | old_signature_ids), overlap["union_signature_families"])

    material = defaultdict(list)
    for line, row in enumerate(rows(run / "candidate-source-material.jsonl"), 1):
        material[row["family_id"]].append((line, row))
    compare("material_family_set", sorted(material), sorted(selected))
    inventory_by_family = {f["family_id"]: f for f in inventory["body_coverage_inventory"]}
    original = {f["id"]: f for f in inventory["original_administrative_rule_metadata"]}
    components = defaultdict(list)
    for family in families:
        components[family["copy_component"]].append({"family_id": family["family_id"], "split": family["split"]})
    crossing_components = [{"component": component, "members": members} for component, members in components.items()
                           if len({m["split"] for m in members}) > 1]
    crossing_edges = [edge for edge in allocation["copy_relationships"]
                      if edge["a"] in selected and edge["b"] in selected
                      and selected[edge["a"]]["split"] != selected[edge["b"]]["split"]]
    attribution_path = run / "source-overlap-attribution.json"
    attribution = {"path": str(attribution_path.relative_to(root)),
                   "state": "PENDING_INDEPENDENT_REVIEW", "required": True,
                   "binding": pin(attribution_path) if attribution_path.exists() else None,
                   "binding_note": "Parent-owned evidence. If absent at preparation, bind the later artifact hash to this frozen index before independent adjudication.",
                   "review_scope": "Family-specific below-threshold attribution within the pinned reserved/old-source signatures, not universal semantic closure."}
    attribution_by_family = defaultdict(list)
    source_attribution_units = []
    if attribution_path.exists():
        attribution_data = json.loads(attribution_path.read_text(encoding="utf-8"))
        for key, name in (("source_allocation", "candidate-source-allocation.json"),
                          ("source_screen", "source-overlap-screen.json"),
                          ("source_material", "candidate-source-material.jsonl")):
            compare("attribution_binding:" + key, attribution_data[key]["sha256"], inputs[name]["sha256"])
        for unit in attribution_data["units"]:
            fid = unit["family_id"]
            if fid not in selected:
                continue
            source_attribution_units.append(unit)
            compare("attribution_partition:" + unit["unit_id"], unit["split"], selected[fid]["split"], fid)
            by_reserved_family = defaultdict(list)
            for match in unit["matches"]:
                by_reserved_family[match["reserved_signature_metadata"]["family_id"]].append(match)
            summaries = []
            for reserved_fid, matches in sorted(by_reserved_family.items()):
                summaries.append({"reserved_family_id": reserved_fid, "matching_signature_rows": len(matches),
                    "max_shared_ngram8": max(m["shared_ngram8"] for m in matches),
                    "max_reserved_containment": max(m["reserved_containment"] for m in matches),
                    "max_candidate_containment": max(m["candidate_containment"] for m in matches),
                    "max_minhash_estimate": max(m["minhash_estimate"] for m in matches)})
            attribution_by_family[fid].append({
                **{k: unit[k] for k in ("unit_id", "text_sha256", "passage_pin", "source_scope",
                    "candidate_side_shared_ngrams", "shared_ngram8_total", "distinct_shared_ngram8")},
                "matching_signature_row_count": len(unit["matches"]),
                "matched_reserved_family_count": len(summaries), "matched_family_summaries": summaries,
                "exact_matching_rows": {"artifact": attribution["binding"], "unit_id": unit["unit_id"]},
                "use": "SOURCE_REVIEW_ONLY_NOT_AUTHORING; full-body contexts may include held articles."})
    def attribution_summary(units):
        identities = set()
        canonical = []
        for unit in units:
            matched = sorted({m["reserved_signature_metadata"]["family_id"] for m in unit["matches"]}
                             if "matches" in unit else
                             {m["reserved_family_id"] for m in unit["matched_family_summaries"]})
            identities.update(matched)
            canonical.append({**{k: unit[k] for k in ("unit_id", "text_sha256", "passage_pin",
                "candidate_side_shared_ngrams")}, "matched_family_ids": matched})
        return {"units": len(units),
                "unit_summed_candidate_gram_hashes": sum(len(u["candidate_side_shared_ngrams"]) for u in units),
                "candidate_occurrence_contexts": sum(len(g["occurrences"]) for u in units
                    for g in u["candidate_side_shared_ngrams"]),
                "matched_family_count": len(identities),
                "candidate_context_and_family_identity_sha256": digest(json.dumps(
                    sorted(canonical, key=lambda u: u["unit_id"]), ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode())}

    copied_attribution_units = [unit for units in attribution_by_family.values() for unit in units]
    attribution_completeness = {
        "full_proof_selected_units": attribution_summary(source_attribution_units),
        "packet_projection": attribution_summary(copied_attribution_units),
        "policy": "Deduplicate signature associations into matched-family summaries only; preserve every candidate-side gram hash, numeric/article context, occurrence and matched family ID without truncation or generic classification.",
        "full_proof_binding": attribution["binding"],
    }
    attribution_completeness["complete_projection_observed"] = compare(
        "attribution_projection_completeness", attribution_completeness["packet_projection"],
        attribution_completeness["full_proof_selected_units"])
    scope = {"selected_families": 50, "reserved_identity_families": len(reserved_ids),
             "reserved_text_free_manifest_count": 4,
             "reserved_signature_family_union": len(reserved_signature_ids),
             "old_source_fingerprint_families": len(old_signature_ids),
             "signature_inputs": signature_summaries,
             "scope": "These frozen allocation families, 1000 reserved identities, four text-free manifests, old-source fingerprints and observed selected copy components only."}
    execution_pins = [p for p in inventory["inputs"] if p["path"].startswith(EXECUTION)]
    history = {"known_local_8B": inventory["known_checkpoint_consumption_metadata"],
               "execution_metadata_pins": execution_pins,
               "known_local_train_families": ["admrul:41555", "admrul:91382"],
               "known_local_dev_families": ["admrul:43010"],
               "family_allocation_pin": pin(inventory["known_checkpoint_consumption_metadata"]["family_allocation_evidence"]),
               "old_declared_exposure_metadata": inventory["old_declared_exposure_metadata"],
               "foundation_base_pretraining_history": "UNKNOWN_NOT_CERTIFIED",
               "broader_original_corpus_consumption": "UNKNOWN_NOT_CERTIFIED",
               "legacy_train_label_is_checkpoint_proof": False, "historically_unseen_claim": False}
    dimension_pins = {
        "rights": [pin(provenance_path), inputs["source-inventory.json"]],
        "evidence": [inputs["candidate-source-allocation.json"], inputs["candidate-source-material.jsonl"]],
        "bounded_copy_review": [inputs["candidate-source-allocation.json"], inputs["source-overlap-screen.json"]]
            + ([attribution["binding"]] if attribution["binding"] else []),
        "protected_exclusions": [inputs["source-inventory.json"], pin(reservation_path)],
        "signature_review": [inputs["source-overlap-screen.json"]]
            + ([attribution["binding"]] if attribution["binding"] else []),
        "history_review": [inputs["source-inventory.json"], history["family_allocation_pin"], *execution_pins],
    }
    packet_values = []
    quote_count = 0
    body_count = 0
    for family in families:
        fid = family["family_id"]
        before = len(mismatches)
        raw_pin = pin(family["raw_source"]["path"])
        compare("raw_sha256", raw_pin["sha256"], family["raw_source"]["sha256"], fid)
        compare("raw_bytes", raw_pin["bytes"], family["raw_source"]["bytes"], fid)
        tree = ET.parse(root / family["raw_source"]["path"]).getroot()
        basic = {child.tag: child.text for child in tree.find("행정규칙기본정보")}
        nodes = tree.findall("./조문내용") or tree.findall("./조문/조문내용")
        texts = [normalized_body("".join(node.itertext())) for node in nodes]
        body = "\n\n".join(texts)
        for key, actual in {"body_sha256": digest(body.encode()), "body_chars": len(body),
                            "body_node_count": len(nodes), "version": basic.get("행정규칙일련번호"),
                            "family_id": "admrul:" + basic.get("행정규칙ID", ""),
                            "issuer": basic.get("소관부처명"), "title": basic.get("행정규칙명"),
                            "effective_date": basic.get("시행일자"), "listed_current": basic.get("현행여부")}.items():
            compare("raw_body_or_identity:" + key, actual, family[key], fid)
        body_count += 1
        source = original[family["doc_id"]]
        for key, original_key in (("family_id", "group"), ("source_url", "source"),
                                  ("license", "license"), ("legacy_cleaned_source_sha256", "sha256")):
            compare("original_metadata:" + key, family[key], source[original_key], fid)
        inventory_record = inventory_by_family[fid]
        for key in ("doc_id", "body_sha256", "body_chars", "version", "source_url", "raw_source"):
            compare("inventory:" + key, family[key], inventory_record[key], fid)
        compare("declared_license", family["license"], provenance["license"], fid)
        offsets = []
        offset = 0
        for text in texts:
            offsets.append((offset, offset + len(text)))
            offset += len(text) + 2
        frozen = material[fid]
        quote_checks = []
        compare("selected_material_count", len(frozen), len(family["passage_pins"]), fid)
        compare("article_coverage_count", len(frozen), family["article_coverage_count"], fid)
        compare("selected_quote_ordinals", [r["raw_body_node_ordinal"] for _, r in frozen],
                [p["raw_body_node_ordinal"] for p in family["passage_pins"]], fid)
        passage_by_ordinal = {p["raw_body_node_ordinal"]: p for p in family["passage_pins"]}
        selected_hold_findings = []
        articles = []
        for line, row in frozen:
            q_before = len(mismatches)
            ordinal = row["raw_body_node_ordinal"]
            passage = passage_by_ordinal.get(ordinal)
            compare(f"quote_{ordinal}:allocation_pin", {k: row.get(k) for k in passage} if passage else None, passage, fid)
            for key, expected in (("doc_id", family["doc_id"]), ("issuer", family["issuer"]),
                                  ("source_url", family["source_url"]), ("source_version", family["version"]),
                                  ("split", family["split"])):
                compare(f"quote_{ordinal}:" + key, row[key], expected, fid)
            valid_ordinal = 1 <= ordinal <= len(texts)
            compare(f"quote_{ordinal}:valid_node", valid_ordinal, True, fid)
            if valid_ordinal:
                compare(f"quote_{ordinal}:offsets", [row["char_start"], row["char_end"]], list(offsets[ordinal - 1]), fid)
                compare(f"quote_{ordinal}:raw_node_text_sha256", digest(texts[ordinal - 1].encode()), row["quote_sha256"], fid)
                # Avoid embedding mismatching text (including any misplaced partition text) in diagnostics.
                compare(f"quote_{ordinal}:raw_node_equals_material", texts[ordinal - 1] == row["text"], True, fid)
            compare(f"quote_{ordinal}:material_sha256", digest(row["text"].encode()), row["quote_sha256"], fid)
            compare(f"quote_{ordinal}:body_slice_equals_material", body[row["char_start"]:row["char_end"]] == row["text"], True, fid)
            holds = [name for name in inventory["old_statute_reference_holds"] if name and name in compact(row["text"])]
            if holds or SUBSTANTIVE_HOLD.search(row["text"]) or ASSET_HOLD.search(row["text"]):
                selected_hold_findings.append({"raw_body_node_ordinal": ordinal, "old_statute_matches": holds,
                    "lexical_topic_match": bool(SUBSTANTIVE_HOLD.search(row["text"])),
                    "asset_match": bool(ASSET_HOLD.search(row["text"]))})
            quote_checks.append({"material_record_ordinal": line, "raw_body_node_ordinal": ordinal,
                                 "quote_sha256": row["quote_sha256"], "concordant": len(mismatches) == q_before})
            # Emit only this family's official reconstructed node, never untrusted misplaced material text.
            articles.append({**{k: v for k, v in row.items() if k != "text"},
                             "text": texts[ordinal - 1] if valid_ordinal else None})
            quote_count += 1
        identity_matches = []
        for reserved in reservations:
            reasons = []
            if reserved["doc_family_id"] == fid:
                reasons.append("family_id")
            if reserved.get("url") and url_key(reserved["url"]) == url_key(family["source_url"]):
                reasons.append("normalized_source_url")
            if reserved.get("title") and compact(reserved["title"]) == compact(family["title"]):
                reasons.append("normalized_title_requires_review")
            if reasons:
                identity_matches.append({"reserved_family_id": reserved["doc_family_id"], "reasons": reasons})
        family_units = {key: value for key, value in overlap["units"].items() if key.startswith(fid + ":")}
        expected_units = {fid + ":full_body"} | {fid + ":article_" + str(p["raw_body_node_ordinal"]) for p in family["passage_pins"]}
        compare("signature_unit_coverage", sorted(family_units), sorted(expected_units), fid)
        packet = {"schema_version": 1, "state": "PENDING_INDEPENDENT_REVIEW", "admission_status": "NOT_ADMITTED",
            "family_id": fid, "split": family["split"], "source_only": True,
            "allocation_binding": {"file": inputs["candidate-source-allocation.json"], "family": family},
            "source_inventory_binding": inputs["source-inventory.json"],
            "source_material_binding": inputs["candidate-source-material.jsonl"],
            "rights_provenance": {"file": pin(provenance_path),
                **{k: provenance[k] for k in ("source", "license", "license_basis", "attribution", "changes")},
                "packet_projection_changes": inventory["body_projection"],
                "scope": "Official numbered body only; no annexes, tables, linked assets or private material. Declared Article 7 basis is not source-specific legal clearance.",
                "review_required": "Independent source-specific public rights and currentness fitness assessment."},
            "currentness": {"status": "OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED", "live_fetch": False,
                "effective_date": family["effective_date"], "listed_current": family["listed_current"],
                "list_pin": pin(family["list_path"]), "source_url": family["source_url"]},
            "source_concordance": {"analysis_kind": "OFFLINE_SOURCE_CONCORDANCE_NOT_TESTS",
                "projection_function": "prepare_expanded_source_inventory.normalized_body",
                "node_paths": ["./조문내용", "./조문/조문내용"],
                "join": "two newlines between normalized node texts", "offset_units": "Python Unicode code points; end exclusive",
                "observed_raw_pin": raw_pin, "reconstructed_body_sha256": digest(body.encode()),
                "reconstructed_body_chars": len(body), "reconstructed_node_count": len(nodes),
                "quote_checks": quote_checks, "findings": mismatches[before:]},
            "identity_reservation_comparison": {"reservation_pin": pin(reservation_path),
                "reserved_family_count": len(reserved_ids), "compared_fields": ["family_id", "normalized_source_url", "normalized_title"],
                "matches": identity_matches, "signature_family_identity_match": fid in reserved_signature_ids},
            "protected_exclusions": {"old_family_holds": inventory["old_family_exclusions"],
                "own_family_held": fid in inventory["old_family_exclusions"],
                "old_statute_reference_holds": inventory["old_statute_reference_holds"],
                "conservative_lexical_topic_pattern": SUBSTANTIVE_HOLD.pattern,
                "asset_hold_pattern": ASSET_HOLD.pattern, "non_substantive_heading_pattern": NON_SUBSTANTIVE.pattern,
                "historical_held_article_reason_counts": family["held_article_reason_counts"],
                "selected_article_hold_findings": selected_hold_findings,
                "semantic_exclusions": "PENDING_INDEPENDENT_REVIEW; lexical absence is not semantic clearance."},
            "bounded_copy_review": {"selected_component": family["copy_component"],
                "evidence_pins": dimension_pins["bounded_copy_review"],
                "selected_component_members": components[family["copy_component"]],
                "selected_cross_partition_component_count": len(crossing_components),
                "selected_cross_partition_edge_count": len(crossing_edges),
                "own_observed_edges": [e for e in allocation["copy_relationships"] if fid in (e["a"], e["b"])],
                "scope": scope, "semantic_scope_limitations": LIMITATIONS},
            "signature_review": {"screen_pin": inputs["source-overlap-screen.json"],
                "thresholds": overlap["thresholds"], "own_family_units": family_units,
                "below_threshold_attribution": {**attribution,
                    "own_family_units": attribution_by_family[fid],
                    "own_family_attributed_unit_count": len(attribution_by_family[fid]),
                    "own_family_projection_completeness": attribution_summary(attribution_by_family[fid]),
                    "zero_observed_is_clearance": False}},
            "history_review": {**history, "own_legacy_labels": family["legacy_labels"]},
            "requested_independent_dimensions": {name: {"verdict": "PENDING_INDEPENDENT_REVIEW",
                "reviewer_id": None, "reason": "No independent assessment supplied by this source analysis.",
                "evidence_pins": dimension_pins[name]} for name in DIMENSIONS},
            "question_capacity": {"status": "DEFERRED_TO_LATER_CASE_QA", "source_verdict_requested": False,
                "required_later_evidence": "Actual authored per-family cases plus independent case/semantic review reports; pin them when available. Article counts and source review do not prove question authorability."},
            "selected_own_family_articles": articles, "limitations": LIMITATIONS}
        packet_values.append(packet)

    # Global diagnostics are text-free; TRAIN packets never receive another family's body/quotes.
    for packet in packet_values:
        packet["input_integrity"] = {"recorded_file_pins_checked": len(checked_pins),
            "file_mismatch_count": sum(not p["concordant"] for p in checked_pins),
            "details": index_path.name}
    packet_index = []
    for path, packet in zip(targets, packet_values):
        save_new(path, packet)
        packet_index.append({"family_id": packet["family_id"], "split": packet["split"],
                             **pin(path), "selected_article_count": len(packet["selected_own_family_articles"]),
                             "source_concordance_mismatch_count": len(packet["source_concordance"]["findings"])})
    counts = {"packets": len(packet_index), "packets_by_split": dict(Counter(f["split"] for f in families)),
              "official_bodies_reconstructed": body_count, "selected_quotes_checked": quote_count,
              "recorded_file_pins_checked": len(checked_pins), "source_concordance_mismatches": len(mismatches),
              "selected_cross_partition_components": len(crossing_components),
              "selected_cross_partition_edges": len(crossing_edges),
              "reservation_metadata_matches": sum(len(p["identity_reservation_comparison"]["matches"]) for p in packet_values),
              "selected_article_hold_findings": sum(len(p["protected_exclusions"]["selected_article_hold_findings"]) for p in packet_values),
              "attribution_unit_summed_candidate_gram_hashes": attribution_completeness["packet_projection"]["unit_summed_candidate_gram_hashes"],
              "attribution_candidate_occurrence_contexts": attribution_completeness["packet_projection"]["candidate_occurrence_contexts"],
              "admitted_families": 0, "independent_verdicts": 0}
    index = {"schema_version": 1, "state": "PENDING_INDEPENDENT_REVIEW", "admission_status": "NOT_ADMITTED",
             "analysis_kind": "OFFLINE_SOURCE_CONCORDANCE_NOT_TESTS", "inputs": inputs,
             "implementation": pin(Path(__file__).resolve()), "counts": counts,
             "bounded_verified_scope": scope, "families": packet_index, "findings": mismatches,
             "recorded_pin_checks": checked_pins, "cross_partition_components": crossing_components,
             "cross_partition_edges": crossing_edges, "source_overlap_attribution": attribution,
             "attribution_projection_completeness": attribution_completeness,
             "question_capacity": {"status": "DEFERRED_TO_LATER_CASE_QA", "source_verdict_requested": False},
             "required_independent_dimensions": list(DIMENSIONS), "limitations": LIMITATIONS}
    save_new(index_path, index)
    return {"index": str(index_path.relative_to(root)), "counts": counts, "findings": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--packet-suffix", default="",
                        help="Immutable output namespace, e.g. v2; existing files are never replaced")
    args = parser.parse_args()
    root = args.root.resolve()
    run = args.run.resolve() if args.run.is_absolute() else root / args.run
    print(json.dumps(prepare(root, run, args.packet_suffix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
