"""Prepare pinned, NOT_ADMITTED grounded train/dev candidates; local CPU work only.

materialize emits a trainer-compatible candidate manifest, not training authority.
audit uses the production trainer's encoding path without loading any model weights.
Independent semantic, rights, family-owner and reservation reports remain required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.service.mois_graph import GraphStore
from trid.service.mois_graph_generation import build_generation_messages, generate_graph_reply

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BEHAVIORS = ("conditions", "exceptions", "normative_force", "information_boundary")
FAMILIES = {"train": {"admrul:41555", "admrul:91382"}, "dev": {"admrul:43010"}}
TASK = "mois_grounded_explanation"
EXCLUDED_PASSAGES = {
    "law-go-kr:admrul:2100000251966:0019",
    "law-go-kr:admrul:2100000251966:0020",
    "law-go-kr:admrul:2100000251966:0021",
}
LIMITATIONS = (
    "Structural, citation, hash and token checks are not semantic entailment or training-rights verification. "
    "Independent semantic review of every fresh target, rights/family-owner review, and source/candidate "
    "reservation overlap reports with explicit dispositions remain required. Historical passes are provenance only."
)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def verified(record, base=ROOT):
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    if file_sha(path) != record["sha256"] or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError(f"pinned file changed: {path}")
    return path


def load_contract(path, expected_sha256):
    if file_sha(path) != expected_sha256:
        raise ValueError("immutable preparation scope/contract bytes changed")
    contract = read_json(path)
    expected = {
        "schema": "trid-grounded-experiment-preparation-v1", "base_model": MODEL,
        "base_revision": REVISION, "adapter": "grounded_messages", "enable_thinking": False,
        "context_tokens": 3072, "answer_tokens_including_eos": 768,
        "train_rows": 24, "dev_rows": 8, "behavior_counts_per_split": {"train": 6, "dev": 2},
        "training_authorized": False, "aws_calls_authorized": False, "deployment_authorized": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"immutable experiment contract mismatch: {key}")
    if contract.get("behaviors") != list(BEHAVIORS):
        raise ValueError("immutable behavior contract changed")
    for split, families in FAMILIES.items():
        if set(contract[split + "_families"]) != families:
            raise ValueError("immutable family allocation changed")
    scope = read_json(verified(contract["scope"]))
    if (scope["base_model"], scope["base_revision"], scope["actual_training_requires_new_bounded_gpu_approval"]) != (MODEL, REVISION, True):
        raise ValueError("scope does not authorize this local preparation")
    for key in ("historical_source_allocation", "source_only_sidecar", "reservation_evidence", "tokenizer_acquisition"):
        verified(contract[key])
    return contract


def source_context(contract):
    historical = read_json(verified(contract["historical_source_allocation"]))
    history = {row["family_id"]: row for row in historical["families"]}
    # Deliberately project only source fields. Historical question/target/model output is never consumed.
    approved = {}
    for row in read_jsonl(verified(contract["source_only_sidecar"])):
        if row["family_id"] not in FAMILIES["train"] | FAMILIES["dev"]:
            continue
        for passage in row["passages"]:
            key = (row["doc_id"], passage["passage_id"])
            value = {k: passage[k] for k in ("quote_sha256", "char_start", "char_end")}
            if text_sha(passage["quote"]) != value["quote_sha256"]:
                raise ValueError("historical source-only quotation hash mismatch")
            if key in approved and approved[key] != value:
                raise ValueError("conflicting historical source quotation")
            approved[key] = value
    evidence = read_json(verified(contract["reservation_evidence"]))
    reservation = evidence["reservation"]
    reservation_path = verified({"path": reservation["path"], "sha256": reservation["recorded_sha256"]})
    reserved = read_jsonl(reservation_path)
    signatures = [pin(verified({"path": item["path"], "sha256": item["recorded_sha256"]}))
                  for item in evidence["signature_manifests"]]
    if len(signatures) != 4:
        raise ValueError("all four reservation signature manifests are required")
    metadata_dir = verified(contract["historical_source_allocation"]).parent
    exclusion_metadata = [pin(metadata_dir / name) for name in (
        "final24-domain-exclusion-keys.recovered.json",
        "final24-behavior-exclusion-patterns.recovered.json",
        "dev-domain-exclusion-metadata.jsonl", "dev-behavior-exclusion-metadata.jsonl",
        "evaluation-source-closure-v1/source-fingerprints.jsonl")]
    return history, approved, reserved, {"reservation": pin(reservation_path), "signatures": signatures,
                                         "protected_exclusion_metadata": exclusion_metadata,
                                         "protected_metadata_review": "pending_independent_report"}


def validate_allocation(contract, allocation, history, reserved):
    families = {item["family_id"]: item for item in allocation["families"]}
    if len(families) != len(allocation["families"]):
        raise ValueError("duplicate family allocation / train-dev source-family leakage")
    forbidden = set(contract["independent_acceptance_reserved_families"]) | {"admrul:68101"}
    forbidden.update(row["doc_family_id"] for row in reserved)
    urls = {str(row["url"]).removeprefix("https://").removeprefix("http://").rstrip("/").lower()
            for row in reserved if row.get("url")}
    if set(families) & forbidden:
        raise ValueError("reserved family cannot be admitted as a candidate")
    if set(families) != set().union(*FAMILIES.values()):
        raise ValueError("unallocated or missing experiment family")
    for fid, family in families.items():
        if family["split"] not in FAMILIES or fid not in FAMILIES[family["split"]]:
            raise ValueError("train-dev source-family leakage")
        original = history[fid]
        for field in ("group_key", "doc_id", "source_id", "version", "content_hash", "license", "rights", "source_record"):
            if family.get(field) != original[field]:
                raise ValueError(f"allocated source identity/pin changed: {fid} {field}")
        if not family["license"] or family["rights"].get("license") != family["license"]:
            raise ValueError("source license/rights assertion missing")
        source_url = (family.get("source_document") or {}).get("url")
        if source_url and source_url.removeprefix("https://").removeprefix("http://").rstrip("/").lower() in urls:
            raise ValueError("reserved source URL")
    for field in ("doc_id", "content_hash", "source_id"):
        if len({family[field] for family in families.values()}) != len(families):
            raise ValueError("source identity/content shared across allocated families")
    return families


def validate_authored_target(question, evidence_reply, messages, answer):
    """Exercise unchanged serving checks on supplied text, without inference."""
    def provided_target(rendered_messages):
        if rendered_messages != messages:
            raise ValueError("production validation prompt differs from candidate messages")
        return {"generated": answer, "finish_reason": "stop", "context_truncated": False,
                "truncated_output": False, "skipped_context": False}

    result = generate_graph_reply(
        question, evidence_reply, provided_target,
        generator={"kind": "provided_authored_target", "inference_performed": False},
    )
    generation = result.get("generation") or {}
    if generation.get("state") != "generated":
        raise ValueError(f"authored target rejected by production validator: {generation.get('reason')}")
    # The serving wrapper's model_used/generated fields describe its normal inference
    # path. Do not persist those claims for this supplied-target validation call.
    return {"validator": "generate_graph_reply", "input_kind": "provided_authored_target",
            "inference_performed": False, "semantic_verification": generation["semantic_verification"],
            "validation": generation["validation"]}


def prepare_candidates(contract, allocation, specs, store, history, approved, reserved):
    families = validate_allocation(contract, allocation, history, reserved)
    if store.metadata["release"] != allocation["graph_release"]:
        raise ValueError("graph release changed")
    if not specs or len({row["id"] for row in specs}) != len(specs):
        raise ValueError("empty candidate set or duplicate candidate id")
    counts = Counter((row["split"], row["behavior"]) for row in specs)
    expected = Counter({(split, behavior): contract["behavior_counts_per_split"][split]
                        for split in FAMILIES for behavior in BEHAVIORS})
    if counts != expected:
        raise ValueError("train/dev behavior counts differ from immutable contract")
    rows, sidecars = [], []
    seen_content = set()
    for spec in specs:
        family = families.get(spec["family_id"])
        if family is None or family["split"] != spec["split"] or family["doc_id"] != spec["doc_id"]:
            raise ValueError("candidate train-dev source-family leakage or document mismatch")
        if not isinstance(spec["id"], str) or not spec["id"].strip() or not isinstance(spec["question"], str) or not spec["question"].strip():
            raise ValueError("candidate id/question missing")
        source = store.get_document(spec["doc_id"])
        if source is None or any(source.get(k) != family[k] for k in ("content_hash", "license")):
            raise ValueError("allocated source hash/license changed")
        if family.get("source_document") != source:
            raise ValueError("allocated source document metadata changed")
        for field in ("source_id", "version"):
            if source.get(field) is not None and str(source[field]) != family[field]:
                raise ValueError("graph source identity changed")
        passages = {row["id"]: row for row in store.passages(spec["doc_id"])}
        pins = {row["passage_id"]: row for row in family["passage_pins"]}
        if len(pins) != len(family["passage_pins"]):
            raise ValueError("duplicate passage pin")
        pids = spec["passage_ids"]
        if (not pids or len(set(pids)) != len(pids)
                or set(pids) & (set(family["excluded_passage_ids"]) | EXCLUDED_PASSAGES)):
            raise ValueError("empty/duplicate/excluded passage selection")
        candidates, pointers = [], []
        for pid in pids:
            passage = passages.get(pid)
            if passage is None or pid not in pins or (spec["doc_id"], pid) not in approved:
                raise ValueError("passage lacks approved source-only pin")
            if any(passage.get(k) for k in ("value_hold", "private", "table_unverified")):
                raise ValueError("held/private/unverified passage cannot supply evidence")
            text = passage["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty evidence passage")
            actual = {"quote_sha256": text_sha(text), "char_start": passage["char_start"], "char_end": passage["char_end"]}
            if actual != approved[(spec["doc_id"], pid)] or any(pins[pid].get(k) != v for k, v in actual.items()):
                raise ValueError("whole passage source hash/offset changed")
            candidates.append({"doc_id": spec["doc_id"], "passage_id": pid, "text": text,
                               "prediction": {"answer": text, "abstains": False, "source_character_span": [0, len(text)]}})
            pointers.append({"passage_id": pid, **actual, "quote": text, "value_hold": False,
                             "private": False, "table_unverified": False, "whole_passage": True,
                             "review": passage.get("review")})
        reply = {"status": "answer", "answer_mode": "graph", "model_used": False,
                 "scope": "행정안전부 수록 버전의 제시된 근거만 사용하며 다른 기관에 대한 적용이나 현재 법적 유효성을 추론하지 않습니다.",
                 "warnings": ["수집 버전의 근거이며 현재 법적 유효성은 확인되지 않았습니다."],
                 "candidates": candidates, "graph": {"release": store.metadata["release"], "sources": [source], "paths": []}}
        messages = build_generation_messages(spec["question"], reply)
        packet = json.loads(messages[1]["content"])["source_data"]
        if len(packet["evidence"]) != len(pids):
            raise ValueError("production renderer dropped/merged selected evidence")
        answer = encoded(spec["target"])
        target_validation = validate_authored_target(spec["question"], reply, messages, answer)
        content_key = (encoded(messages), answer)
        if content_key in seen_content:
            raise ValueError("duplicate candidate messages/answer content, regardless of id or behavior")
        seen_content.add(content_key)
        provenance = {key: family[key] for key in ("family_id", "group_key", "doc_id", "source_id", "version", "content_hash", "license", "rights")}
        provenance.update(source_url=source.get("url"), passage_ids=pids, passage_pins=[pins[p] for p in pids],
                          target_sha256=text_sha(answer), messages_sha256=text_sha(encoded(messages)),
                          authored_target_validation=target_validation,
                          behavior=spec["behavior"], authorship="fresh source-grounded authoring; not historical targets",
                          review_status="pending_independent_semantic_review", reservation_clearance="pending_independent_reports",
                          admission_status="NOT_ADMITTED", development_exposure=contract["development_exposure"])
        rows.append({"id": spec["id"], "task": TASK, "split": spec["split"], "behavior": spec["behavior"],
                     "messages": messages, "answer": answer, "provenance": provenance})
        sidecars.append({"id": spec["id"], "split": spec["split"], **provenance, "question": spec["question"],
                         "target": spec["target"], "review_focus": spec["review_focus"], "source_document": source,
                         "passages": pointers, "evidence_reply": reply})
    if {row["provenance"]["family_id"] for row in rows} != set(families):
        raise ValueError("allocated family has no candidates")
    return rows, sidecars


def selected_source_texts(families):
    selected = {}
    expected = {family["doc_id"]: family for family in families.values()}
    for relative in sorted({family["source_record"] for family in families.values()}):
        with (ROOT / relative).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = row.get("id")
                if row.get("source_id") and row.get("version"):
                    identity = row["source_id"].rsplit(":", 1)[0] + ":" + str(row["version"])
                if identity not in expected:
                    continue
                family = expected[identity]
                if row.get("source_id"):
                    if row["source_id"] != family["source_id"] or str(row.get("version")) != family["version"] or row.get("id") not in (None, identity):
                        raise ValueError("original source identity conflict")
                elif row.get("group") != family["group_key"]:
                    raise ValueError("original source family conflict")
                if row.get("group") is not None and row["group"] != family["group_key"]:
                    raise ValueError("original source family conflict")
                if text_sha(row["text"]) != family["content_hash"]:
                    raise ValueError("original source content hash changed")
                selected[identity] = row["text"]
    if selected.keys() != expected.keys():
        raise ValueError("original allocated source missing")
    return selected


def overlap_exports(families, rows, store):
    full_texts = selected_source_texts(families)
    sources, parts, metadata = [], [], []
    for family in families.values():
        source = store.get_document(family["doc_id"])
        base = {"group": family["group_key"], "source": source.get("url"), "source_id": family["source_id"],
                "version": family["version"], "content_hash": family["content_hash"], "split": family["split"]}
        for identity in (family["source_id"], family["doc_id"]):
            metadata.append({**base, "id": identity, "kind": "allocated_source_metadata"})
        sources.append({**base, "id": family["doc_id"] + ":full-source", "kind": "original_source_exclusion_only",
                        "text": full_texts[family["doc_id"]], "usage": "EXCLUSION_ONLY_NOT_TRAINING"})
        passages = store.passages(family["doc_id"])
        for passage in passages:
            sources.append({**base, "id": passage["id"], "kind": "source_passage_exclusion_only", "text": passage["text"],
                            "value_hold": passage.get("value_hold", False), "usage": "EXCLUSION_ONLY_NOT_TRAINING"})
        sources.append({**base, "id": family["doc_id"], "kind": "source_passage_aggregate_exclusion_only",
                        "text": "\n".join(p["text"] for p in passages), "usage": "EXCLUSION_ONLY_NOT_TRAINING"})
    for row in rows:
        provenance = row["provenance"]
        base = {"group": provenance["family_id"], "source": provenance["source_url"], "candidate_id": row["id"], "split": row["split"]}
        user = json.loads(row["messages"][1]["content"])
        prose = "\n".join(p["text"] for p in json.loads(row["answer"])["paragraphs"])
        texts = {"question": user["question"], "answer_json": row["answer"], "answer_prose": prose,
                 "user_message": row["messages"][1]["content"]}
        quotes = [p.get("quote", p.get("object", "")) for p in user["source_data"]["evidence"]]
        texts.update({f"evidence_{i}": quote for i, quote in enumerate(quotes, 1)})
        texts["combined"] = "\n".join([user["question"], *quotes, prose])
        for name, text in texts.items():
            parts.append({**base, "id": row["id"] + ":" + name, "kind": "candidate_" + name, "text": text})
    return {"source-metadata-only.jsonl": metadata, "source-only.jsonl": sources, "candidate-text-parts.jsonl": parts}


def materialize(args):
    contract = load_contract(args.contract, args.contract_sha256)
    allocation, specs = read_json(args.allocation), read_jsonl(args.spec)
    history, approved, reserved, reservation_pins = source_context(contract)
    store = GraphStore.open(args.graph)
    rows, sidecars = prepare_candidates(contract, allocation, specs, store, history, approved, reserved)
    exports = overlap_exports({f["family_id"]: f for f in allocation["families"]}, rows, store)
    exports.update({"train-candidates.jsonl": [r for r in rows if r["split"] == "train"],
                    "dev-candidates.jsonl": [r for r in rows if r["split"] == "dev"], "review-sidecar.jsonl": sidecars})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in exports.items():
        (args.output_dir / name).write_text("".join(encoded(r) + "\n" for r in records), encoding="utf-8")
    inputs = {"contract": pin(args.contract), "allocation": pin(args.allocation), "spec": pin(args.spec),
              "graph_manifest": pin(args.graph / "manifest.json"), "materializer": pin(__file__),
              "generation_messages": pin(ROOT / "trid/service/mois_graph_generation.py"),
              "trainer": pin(ROOT / "scripts/train_open_weight_lora.py")}
    manifest = {"schema": "trid-grounded-native-experiment-candidates-v1", "state": "NOT_ADMITTED",
                "model_id": MODEL, "model_revision": REVISION, "family": "qwen3", "enable_thinking": False,
                "max_tokens": 3072, "max_answer_tokens": 768, "inputs": inputs,
                "graph": str(args.graph.resolve()), "reservation_inputs": reservation_pins,
                "development_exposure": contract["development_exposure"], "training_authorized": False,
                "independent_reports": [], "limitations": LIMITATIONS}
    for key, split in (("train", "train"), ("valid", "dev")):
        path = args.output_dir / f"{split}-candidates.jsonl"
        manifest[key] = [{"path": path.name, "sha256": file_sha(path), "adapter": "grounded_messages", "split": split, "epochs": 1}]
    manifest["artifacts"] = {name: pin(args.output_dir / name) for name in exports}
    manifest_path = args.output_dir / "native-train-manifest.json"
    manifest_path.write_text(encoded(manifest) + "\n", encoding="utf-8")
    receipt = {"state": "CANDIDATES_PENDING_SEMANTIC_AND_RESERVATION_ADMISSION", "manifest": pin(manifest_path),
               "rows": dict(Counter(row["split"] for row in rows)), "inputs": inputs,
               "independent_semantic_review": "pending", "reservation_admission": "pending", "rights_review": "pending",
               "training_authorized": False, "aws_calls_authorized": False, "deployment_authorized": False,
               "development_exposure": contract["development_exposure"], "limitations": LIMITATIONS,
               "overlap_scope": "Both splits exported separately. Full-source hits are not candidate admission decisions."}
    (args.output_dir / "candidate-receipt.json").write_text(encoded(receipt) + "\n", encoding="utf-8")
    return receipt


def verify_native_tokenizer(contract, directory):
    if directory.resolve() != (ROOT / contract["tokenizer_directory"]).resolve():
        raise ValueError("use the actual pinned native 8B tokenizer directory")
    acquisition = read_json(verified(contract["tokenizer_acquisition"]))
    if (acquisition["repo_id"], acquisition["revision"]) != (MODEL, REVISION):
        raise ValueError("native acquisition model/revision mismatch")
    records = {row["path"]: row for row in acquisition["files"]}
    for name in ("LICENSE", "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"):
        row = records[name]
        if row.get("body_downloaded") is not True:
            raise ValueError("native tokenizer metadata acquisition incomplete")
        verified(row, directory)
    if any(path.suffix in {".safetensors", ".bin", ".pt"} for path in directory.iterdir()):
        raise ValueError("tokenizer-only acquisition must not contain model weights")
    downloaded = {name for name, record in records.items() if record.get("body_downloaded") is True}
    if {path.name for path in directory.iterdir()} != downloaded:
        raise ValueError("native metadata directory contains unpinned/missing files")
    for name in downloaded:
        verified(records[name], directory)
    return acquisition


def audit_items(tok, entries, base, context, answer_tokens):
    from scripts import train_open_weight_lora as trainer
    loaded = [row for entry in entries for row in trainer.load_rows(entry, base)]
    items, dropped = trainer.build_items(tok, entries, base, "qwen3", context, answer_tokens, False)
    if dropped or len(items) != len(loaded) or not items:
        raise ValueError("audit dropped rows or empty split")
    measurements = []
    for row, item in zip(loaded, items):
        if trainer.encode(tok, row, "qwen3", context) != item:
            raise ValueError("production encode/build_items disagree")
        prompt = tok.apply_chat_template(trainer.messages_of(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tok(row["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if (item["ids"] != prompt_ids + answer_ids or item["labels"] != [-100] * len(prompt_ids) + answer_ids
                or not answer_ids or answer_ids[-1] != tok.eos_token_id):
            raise ValueError("full prompt masking/current answer plus EOS invariant failed")
        measurements.append({"id": row["id"], "task": row["task"], "prompt_tokens": len(prompt_ids),
                             "answer_tokens_including_eos": len(answer_ids), "total_tokens": len(item["ids"])})
    return items, measurements


def audit(args):
    manifest = read_json(args.manifest)
    contract = load_contract(verified(manifest["inputs"]["contract"]), manifest["inputs"]["contract"]["sha256"])
    if args.context != 3072 or args.answer_tokens != 768:
        raise ValueError("audit budgets must equal immutable experiment contract (3072/768)")
    for record in manifest["inputs"].values():
        verified(record)
    for record in manifest["artifacts"].values():
        verified(record)
    if (manifest.get("model_id"), manifest.get("model_revision"), manifest.get("enable_thinking"), manifest.get("training_authorized")) != (MODEL, REVISION, False, False):
        raise ValueError("candidate manifest model/scope changed")
    history, approved, reserved, reservation_pins = source_context(contract)
    if manifest["reservation_inputs"] != reservation_pins:
        raise ValueError("reservation input pins changed")
    store = GraphStore.open(Path(manifest["graph"]))
    allocation = read_json(verified(manifest["inputs"]["allocation"]))
    specs = read_jsonl(verified(manifest["inputs"]["spec"]))
    expected_rows, expected_sidecars = prepare_candidates(contract, allocation, specs, store, history, approved, reserved)
    if read_jsonl(verified(manifest["artifacts"]["review-sidecar.jsonl"])) != expected_sidecars:
        raise ValueError("review sidecar differs from pinned source/spec rendering")
    verify_native_tokenizer(contract, args.tokenizer)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    measurements, encoded_splits = [], {}
    for key, split in (("train", "train"), ("valid", "dev")):
        entries = manifest.get(key, [])
        if len(entries) != 1 or entries[0].get("split") != split or entries[0].get("adapter") != "grounded_messages" or entries[0].get("epochs") != 1:
            raise ValueError("manifest requires one nonempty pinned source_file per split")
        path = verified(entries[0], args.manifest.resolve().parent)
        actual = read_jsonl(path)
        if actual != [row for row in expected_rows if row["split"] == split]:
            raise ValueError("candidate rows/provenance differ from source-pinned materialization")
        items, stats = audit_items(tok, entries, args.manifest.resolve().parent, args.context, args.answer_tokens)
        encoded_splits[split] = items
        for row, stat in zip(actual, stats):
            measurements.append({**stat, "split": split, "behavior": row["behavior"], "family_id": row["provenance"]["family_id"]})
    groups = defaultdict(list)
    for measurement in measurements:
        for key in (measurement["split"], measurement["split"] + "/" + measurement["task"],
                    measurement["split"] + "/" + measurement["task"] + "/" + measurement["behavior"]):
            groups[key].append(measurement)
    summaries = {key: {"rows": len(values), **{field: {"min": min(v[field] for v in values),
                 "max": max(v[field] for v in values), "sum": sum(v[field] for v in values)}
                 for field in ("prompt_tokens", "answer_tokens_including_eos", "total_tokens")}} for key, values in sorted(groups.items())}
    report = {"state": "LOCAL_NATIVE_DATA_AUDIT_PASSED_NOT_ADMITTED", "manifest": pin(args.manifest),
              "tokenizer_acquisition": contract["tokenizer_acquisition"], "context_tokens": args.context,
              "answer_tokens_including_eos": args.answer_tokens, "rows": measurements, "token_summary": summaries,
              "dropped_rows": 0, "truncated_rows": 0, "full_prompt_masked": True, "current_answer_plus_eos_only": True,
              "device": "cpu", "model_weights_loaded": False, "training_authorized": False,
              "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                           for name in ("torch", "transformers", "tokenizers", "pyoxigraph")}},
              "development_exposure": contract["development_exposure"], "independent_reports": manifest.get("independent_reports", []),
              "semantic_review": "pending_separate_report", "reservation_admission": "pending_separate_reports", "limitations": LIMITATIONS}
    if args.command == "cpu-random-smoke":
        report["cpu_random_smoke"] = cpu_random_smoke(tok, encoded_splits)
        report["state"] = "CPU_RANDOM_PATH_SMOKE_PASSED_NOT_MODEL_PERFORMANCE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded(report) + "\n")
    return report


def cpu_random_smoke(tok, splits):
    """One real AdamW update on a tiny random Qwen3, using actual untruncated rows."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from scripts.train_open_weight_lora import batch_loss
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    config = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=3072,
                        tie_word_embeddings=True, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    item = max(splits["train"], key=lambda row: len(row["ids"]))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = batch_loss(model, [item], "cpu", pad, "qwen3")
    if not torch.isfinite(loss):
        raise ValueError("nonfinite random-model CPU loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients) or not any(torch.count_nonzero(g).item() for g in gradients):
        raise ValueError("invalid random-model CPU gradients")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        dev_loss = batch_loss(model, [max(splits["dev"], key=lambda row: len(row["ids"]))], "cpu", pad, "qwen3")
    if not torch.isfinite(dev_loss):
        raise ValueError("nonfinite random-model CPU validation loss")
    return {"device": "cpu", "pretrained_weights_loaded": False, "optimizer_steps": 1,
            "parameter_count": sum(p.numel() for p in model.parameters()), "train_row_tokens": len(item["ids"]),
            "random_train_loss": loss.item(), "random_dev_loss": dev_loss.item(),
            "meaning": "Encoding, production batch_loss, backward and AdamW path only; not 8B quality or performance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materializer = commands.add_parser("materialize")
    for name in ("contract", "allocation", "spec", "output-dir"):
        materializer.add_argument("--" + name, type=Path, required=True)
    materializer.add_argument("--contract-sha256", required=True)
    materializer.add_argument("--graph", type=Path, default=ROOT / "runs/mois-graph/v1")
    for name in ("audit", "cpu-random-smoke"):
        command = commands.add_parser(name)
        for option in ("manifest", "tokenizer", "output"):
            command.add_argument("--" + option, type=Path, required=True)
        command.add_argument("--context", type=int, default=3072)
        command.add_argument("--answer-tokens", type=int, default=768)
    args = parser.parse_args(argv)
    report = materialize(args) if args.command == "materialize" else audit(args)
    print(encoded(report))


if __name__ == "__main__":
    main()
