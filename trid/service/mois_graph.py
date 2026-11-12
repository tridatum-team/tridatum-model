"""Immutable, source-bound MOIS RDF releases; no model or implicit legal reasoning.

Builds consume only the admitted index, never original PDFs. Serving exposes fixed
read-only SPARQL queries, not a user-supplied SPARQL endpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterator
from urllib.parse import quote

from pyoxigraph import Literal, NamedNode, Quad, RdfFormat, Store

NS = "https://tridatum.co/ontology/mois/"
RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
SCHEMA_VERSION = 1
ONTOLOGY_VERSION = "mois-procedure-1"
MOIS = "행정안전부"
RULE_TYPES = {"ProcedureRule", "ApplicabilityRule", "ReferenceRule"}
PREDICATES = {"requires", "deadline", "appliesTo", "excludes", "prohibits",
              "exception", "authorizes", "references", "submittedTo"}
MAX_RECORD_CHARS = 32 * 1024 * 1024
PREFIX = f"PREFIX m: <{NS}> "


def _node(kind: str, value: str) -> NamedNode:
    return NamedNode(NS + kind + "/" + quote(value, safe=""))


def _p(name: str) -> NamedNode:
    return NamedNode(NS + name)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        while line := handle.readline(MAX_RECORD_CHARS + 1):
            if len(line) > MAX_RECORD_CHARS:
                raise ValueError(f"JSONL record exceeds import bound: {path}")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object: {path}")
                yield row


def _documents(path: Path) -> Iterator[dict]:
    """Decode a JSON array with bounded memory, including malformed-tail checks."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, eof, state = "", False, "start"
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = handle.read(65536)
                buffer += chunk
                eof = not chunk
                continue
            if state == "start":
                if not buffer.startswith("["):
                    raise ValueError("documents.json must be an array")
                buffer, state = buffer[1:], "first"
                continue
            if state in {"first", "item"}:
                if state == "first" and buffer.startswith("]"):
                    buffer, state = buffer[1:], "done"
                    continue
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError as error:
                    if eof or len(buffer) >= MAX_RECORD_CHARS:
                        raise ValueError("Invalid or oversized document record") from error
                    chunk = handle.read(65536)
                    buffer += chunk
                    eof = not chunk
                    continue
                if not isinstance(value, dict):
                    raise ValueError("Index document must be an object")
                buffer, state = buffer[end:], "separator"
                yield value
                continue
            if state == "separator":
                if buffer.startswith(","):
                    buffer, state = buffer[1:], "item"
                elif buffer.startswith("]"):
                    buffer, state = buffer[1:], "done"
                else:
                    raise ValueError("Invalid document array separator")
                continue
            if buffer:
                raise ValueError("Trailing data after document array")
            if eof:
                return


def _repo_root(index_dir: Path) -> Path | None:
    for directory in (index_dir, *index_dir.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return None


def _sources(index_dir: Path, selected: set[str], manifest: dict, hashes: dict):
    root = _repo_root(index_dir)
    normalized, supplemental = {}, {}
    if root is not None:
        for snapshot, binding in manifest.get("api_snapshots", {}).items():
            folder = root / snapshot
            snapshot_manifest = folder / "manifest.json"
            expected = binding.get("manifest_sha256")
            if not expected or _sha(snapshot_manifest) != expected:
                raise ValueError("Source snapshot manifest integrity mismatch")
            hashes[str(snapshot_manifest.relative_to(root))] = expected
            snapshot_data = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
            path = folder / "normalized" / "records-00000.jsonl"
            expected_normalized = next((entry["sha256"] for entry in snapshot_data.get("files", [])
                                        if entry.get("path") == "normalized/records-00000.jsonl"), None)
            if not expected_normalized or not path.is_file() or _sha(path) != expected_normalized:
                raise ValueError("Normalized source snapshot integrity mismatch")
            hashes[str(path.relative_to(root))] = expected_normalized
            for row in _jsonl(path):
                source_id, version = row.get("source_id"), row.get("version")
                if source_id and version:
                    doc_id = source_id.rsplit(":", 1)[0] + ":" + str(version)
                    if doc_id in selected:
                        normalized[doc_id] = {k: v for k, v in row.items() if k != "text"}
        path = root / "data/search-corpus-v4/documents.jsonl"
        if path.is_file():
            hashes[str(path.relative_to(root))] = _sha(path)
            for row in _jsonl(path):
                if row.get("id") in selected:
                    supplemental[row["id"]] = {k: v for k, v in row.items() if k != "text"}
    return normalized, supplemental


def _reviews(index_dir: Path, hashes: dict):
    root = _repo_root(index_dir)
    path = index_dir / "review.jsonl"
    if not path.exists() and root is not None:
        path = root / "data/search-corpus-v4/passages.jsonl"
        if not path.is_file():
            raise ValueError("Missing public passage review input")
    private, held, reviews = set(), set(), {}
    if path.is_file():
        hashes["passage_review"] = _sha(path)
        for row in _jsonl(path):
            review = row.get("value_review", {})
            pid = row.get("passage_id")
            if review and not isinstance(review, dict):
                raise ValueError("Malformed passage review")
            if review.get("privacy_review"):
                private.add(pid)
            if review.get("decision") == "hold_for_value_use":
                held.add(pid)
            if review:
                reviews[pid] = review
    return private, held, reviews


def _quad(subject, predicate: str, value):
    return Quad(subject, _p(predicate), value if isinstance(value, NamedNode) else Literal(value))


def _typed(subject, kind: str):
    return Quad(subject, RDF_TYPE, _p(kind))


def _record_quads(kind: str, identity: str, record: dict):
    node = _node(kind, identity)
    yield _typed(node, kind)
    yield _quad(node, "record", _json(record))


def _metadata(doc_id, meta, source, extra, document, scope, count, held_count):
    dates = source.get("dates", {})
    issuer = source.get("issuer", {})
    limitations = meta.get("limitations") or source.get("limitations") or []
    if not isinstance(limitations, list):
        limitations = [limitations]
    limitations = list(limitations)
    if not count:
        limitations.append("No admitted public body evidence; metadata inventory only.")
    if held_count:
        limitations.append("Value-held passages are not answer evidence.")
    if meta.get("table_check") == "unverified" or extra.get("table_check") == "unverified":
        limitations.append("Unverified tables do not establish monetary entitlement.")
    declared_hash = source.get("content_hash") or extra.get("text_sha256") or meta.get("content_hash")
    indexed_text = document.get("text", "")
    computed_hash = hashlib.sha256(indexed_text.encode()).hexdigest() if indexed_text else None
    content_hash = declared_hash or computed_hash
    return {
        "doc_id": doc_id, "title": meta.get("title") or document.get("title"),
        "institution": meta.get("institution"), "kind": meta.get("kind"),
        "url": meta.get("source") or source.get("location", {}).get("url") or extra.get("source_url"),
        "date": meta.get("date") or dates.get("published"), "date_type": meta.get("date_type"),
        "effective_date": dates.get("effective_from") or meta.get("effective_date"),
        "source_id": source.get("source_id") or meta.get("source_id"),
        "version": source.get("version") or meta.get("version") or extra.get("source_version"),
        "content_hash": content_hash, "indexed_hash": document.get("sha256"),
        "content_hash_basis": "normalized_source" if declared_hash else "indexed_text" if computed_hash else None,
        "raw_hash": extra.get("raw_sha256") or source.get("raw_sha256"),
        "retrieved_at": dates.get("retrieved_at") or meta.get("retrieved_at"),
        "currency": meta.get("currency") or source.get("currency"),
        "limitations": limitations, "passage_count": count, "scope": scope,
        "issuer_id": issuer.get("id") or meta.get("issuer_id"),
        "issuer_id_scheme": issuer.get("id_scheme") or meta.get("issuer_id_scheme"),
        "license": meta.get("license") or source.get("rights", {}).get("license") or extra.get("license"),
        "rights": source.get("rights"), "applies_to": meta.get("applies_to") or source.get("applies_to"),
        "date_kinds": source.get("date_kinds"), "table_check": meta.get("table_check") or extra.get("table_check"),
        "content_status": ("held_or_missing" if not count else "held_only" if held_count == count
                           else "indexed_text" if document.get("text") else "indexed_passages_only"),
        "metadata_provenance": {"index": "meta.json", "key": doc_id,
                                "snapshot": meta.get("metadata_snapshot_id") or meta.get("snapshot_id"),
                                "normalized_source": bool(source), "supplemental_source": bool(extra)},
    }


def _validate_fact(fact, metadata, passages, facts_by_id):
    required = {"id", "doc_id", "content_hash", "type", "subject", "predicate", "object", "topics", "qualifiers", "evidence"}
    if not required <= fact.keys():
        raise ValueError("Curated fact missing required fields")
    if any(not isinstance(fact[k], str) or not fact[k].strip() for k in ("id", "doc_id", "content_hash", "subject", "object")):
        raise ValueError("Curated fact requires nonempty identifiers and labels")
    if fact["type"] not in RULE_TYPES or fact["predicate"] not in PREDICATES:
        raise ValueError("Unknown rule type or predicate")
    if (fact["type"] == "ReferenceRule") != (fact["predicate"] == "references"):
        raise ValueError("Reference rule endpoint type mismatch")
    if not isinstance(fact["topics"], list) or any(not isinstance(t, str) for t in fact["topics"]):
        raise ValueError("Invalid fact topics")
    if not isinstance(fact["qualifiers"], dict):
        raise ValueError("Invalid fact qualifiers")
    for key, value in fact["qualifiers"].items():
        if not isinstance(key, str) or not (value is None or isinstance(value, (str, bool, int, float)) or
                                           isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError("Invalid qualifier value")
    doc = metadata.get(fact["doc_id"])
    if doc is None or fact["content_hash"] != doc["content_hash"]:
        raise ValueError("Fact document hash mismatch")
    evidence = fact["evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("Invalid fact evidence")
    passage = passages.get(evidence.get("passage_id"))
    if passage is None or passage["doc_id"] != fact["doc_id"] or passage["value_hold"]:
        raise ValueError("Fact evidence is absent, private, held, or in another document")
    start, end = evidence.get("start"), evidence.get("end")
    if type(start) is not int or type(end) is not int or not passage["char_start"] <= start < end <= passage["char_end"]:
        raise ValueError("Fact evidence offsets outside indexed passage")
    expected = passage["text"][start - passage["char_start"]:end - passage["char_start"]]
    if evidence.get("quote") != expected:
        raise ValueError("Fact quote differs from indexed evidence")
    target = fact.get("target_doc_id")
    if target is not None and (fact["type"] != "ReferenceRule" or target not in metadata):
        raise ValueError("Invalid reference document endpoint")
    qualified = fact["qualifiers"].get("exception_to")
    if qualified is not None:
        target_rule = facts_by_id.get(qualified) if isinstance(qualified, str) else None
        if (fact["predicate"] != "exception" or target_rule is None or qualified == fact["id"]
                or target_rule.get("doc_id") != fact["doc_id"]):
            raise ValueError("Invalid exception_to rule endpoint")


def _fact_quads(fact):
    node = _node("Rule", fact["id"])
    yield _typed(node, "Rule")
    yield _typed(node, fact["type"])
    yield _quad(node, "record", _json(fact))
    yield _quad(_node("DocumentVersion", fact["doc_id"]), "hasRule", node)
    subject, obj = _node("Entity", fact["id"] + ":subject"), _node("Entity", fact["id"] + ":object")
    for endpoint, label, kind in ((subject, fact["subject"], "Subject"), (obj, fact["object"], "RuleObject")):
        yield _typed(endpoint, "Entity")
        yield _typed(endpoint, kind)
        yield _quad(endpoint, "label", label)
    yield _quad(node, "subject", subject)
    yield _quad(node, "object", obj)
    yield _quad(node, "predicate", _p(fact["predicate"]))
    yield _quad(subject, fact["predicate"], obj)
    action = fact["qualifiers"].get("action")
    if isinstance(action, str) and action:
        action_node = _node("Action", fact["id"])
        yield _typed(action_node, "Action")
        yield _quad(action_node, "label", action)
        yield _quad(node, "action", action_node)
    for topic in fact["topics"]:
        topic_node = _node("Topic", topic)
        yield _typed(topic_node, "Topic")
        yield _quad(topic_node, "label", topic)
        yield _quad(node, "topic", topic_node)
    for key, value in fact["qualifiers"].items():
        qualifier = _node("Qualifier", fact["id"] + ":" + key)
        yield _typed(qualifier, "Qualifier")
        yield _quad(node, "qualifier", qualifier)
        yield _quad(qualifier, "key", key)
        for item in value if isinstance(value, list) else [value]:
            if item is None:
                yield _quad(qualifier, "unknownValue", True)
            else:
                yield _quad(qualifier, "value", item)
    evidence = fact["evidence"]
    evidence_node = _node("Evidence", fact["id"])
    yield _typed(evidence_node, "Evidence")
    yield _quad(node, "evidence", evidence_node)
    yield _quad(evidence_node, "passage", _node("Passage", evidence["passage_id"]))
    for key in ("start", "end", "quote"):
        yield _quad(evidence_node, key, evidence[key])
    yield _quad(evidence_node, "contentHash", fact["content_hash"])
    if fact["qualifiers"].get("exception_to") is not None:
        yield _quad(node, "qualifiesRule", _node("Rule", fact["qualifiers"]["exception_to"]))
    if fact["type"] == "ReferenceRule":
        target = fact.get("target_doc_id")
        endpoint = _node("DocumentVersion", target) if target else _node("UnresolvedReference", fact["id"])
        if not target:
            yield _typed(endpoint, "UnresolvedReference")
            yield _quad(endpoint, "label", fact["object"])
        yield _quad(node, "referenceTarget", endpoint)
        yield _quad(node, "resolved", bool(target))


def build_graph(index_dir: Path, output_dir: Path, facts_path: Path) -> dict:
    """Build once; a manifest is published only after every fact is validated."""
    index_dir, output_dir, facts_path = Path(index_dir), Path(output_dir), Path(facts_path)
    if output_dir.exists():
        raise FileExistsError(f"Immutable graph release already exists: {output_dir}")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    manifest_path = index_dir / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    facts = list(_jsonl(facts_path))
    if len({f.get("id") for f in facts}) != len(facts):
        raise ValueError("Duplicate curated fact ID")
    selected = {key for key, row in meta.items() if row.get("institution") == MOIS}
    references = {f["target_doc_id"] for f in facts if f.get("target_doc_id")}
    if not references <= meta.keys():
        raise ValueError("Referenced document is missing from indexed metadata")
    selected |= references
    if any(f.get("doc_id") not in selected for f in facts):
        raise ValueError("Fact source is outside MOIS and explicit reference inventory")
    hashes = {"facts.jsonl": _sha(facts_path), "meta.json": _sha(index_dir / "meta.json")}
    for name in ("manifest.json", "documents.json", "tables.json", "graph.json"):
        if (index_dir / name).is_file():
            hashes[name] = _sha(index_dir / name)
    ontology_path = Path(__file__).with_name("mois_ontology.ttl")
    hashes["ontology"] = _sha(ontology_path)
    normalized, supplemental = _sources(index_dir, selected, source_manifest, hashes)
    private, held, reviews = _reviews(index_dir, hashes)
    tables_path = index_dir / "tables.json"
    table_passages = set(json.loads(tables_path.read_text(encoding="utf-8"))) if tables_path.exists() else set()
    remaining_private = set(private)
    metadata, evidence_passages, seen, seen_passages = {}, {}, set(), set()
    counts = {"documents": 0, "mois_documents": 0, "reference_documents": 0, "passages": 0,
              "facts": len(facts), "references": sum(f.get("type") == "ReferenceRule" for f in facts),
              "private_passages": 0, "held_passages": 0, "missing_documents": 0}
    fact_passages = {f.get("evidence", {}).get("passage_id") for f in facts}
    output_dir.mkdir(parents=True, exist_ok=False)
    store = None
    try:
        store = Store(str(output_dir / "store"))
        store.load(path=str(ontology_path), format=RdfFormat.TURTLE)
        agency = _node("Agency", "행정기관코드:1741000")
        store.extend([_typed(agency, "Agency"), _quad(agency, "label", MOIS), _quad(agency, "alias", "행안부")])
        for document in _documents(index_dir / "documents.json"):
            doc_id = document.get("doc_id")
            for passage in document.get("passages", []):
                remaining_private.discard(passage.get("id"))
            if doc_id not in selected:
                continue
            if doc_id in seen:
                raise ValueError("Duplicate indexed document ID")
            seen.add(doc_id)
            admitted, held_count = [], 0
            public = "public" in document.get("acl", [])
            source, extra = normalized.get(doc_id, {}), supplemental.get(doc_id, {})
            if source.get("issuer", {}).get("name") not in (None, meta[doc_id].get("institution")):
                raise ValueError("Normalized source issuer differs from indexed metadata")
            public = public and not extra.get("usage_restriction") and not source.get("rights", {}).get("restrictions")
            text = document.get("text", "")
            declared_hash = source.get("content_hash") or extra.get("text_sha256") or meta[doc_id].get("content_hash")
            if text and declared_hash and hashlib.sha256(text.encode()).hexdigest() != declared_hash:
                raise ValueError("Normalized source content hash differs from indexed text")
            for original in document.get("passages", []):
                pid = original.get("id")
                if pid in seen_passages:
                    raise ValueError("Duplicate indexed passage ID")
                seen_passages.add(pid)
                inline_review = original.get("value_review") or original.get("review") or {}
                if not isinstance(inline_review, dict):
                    raise ValueError("Malformed indexed passage review")
                passage_review = {**inline_review, **reviews.get(pid, {})}
                if (not public or pid in private or original.get("private") or inline_review.get("privacy_review")
                        or "public" not in original.get("acl", ["public"])):
                    counts["private_passages"] += 1
                    continue
                start, end, body = original.get("char_start"), original.get("char_end"), original.get("text")
                if (original.get("doc_id") != doc_id or not isinstance(pid, str) or type(start) is not int or
                        type(end) is not int or start < 0 or not isinstance(body, str) or end - start != len(body)):
                    raise ValueError("Invalid indexed passage identity or offsets")
                if text and text[start:end] != body:
                    raise ValueError("Indexed passage differs from document text")
                if not body.strip():
                    continue
                # No reviewed row binding is imported here: indexed table text
                # is never silently promoted into an answerable numerical rule.
                table_unverified = pid in table_passages
                value_hold = (pid in held or bool(original.get("value_hold")) or table_unverified
                              or inline_review.get("decision") == "hold_for_value_use")
                held_count += int(value_hold)
                passage = {**original, "review": passage_review or None, "value_hold": value_hold,
                           "table_unverified": table_unverified, "private": False}
                admitted.append(passage)
                if pid in fact_passages:
                    evidence_passages[pid] = passage
            scope = "mois" if meta[doc_id].get("institution") == MOIS else "reference"
            record = _metadata(doc_id, meta[doc_id], source, extra, document, scope, len(admitted), held_count)
            metadata[doc_id] = record
            doc_node = _node("DocumentVersion", doc_id)
            issuer_key = (str(record["issuer_id_scheme"]) + ":" + str(record["issuer_id"])) if record["issuer_id"] else str(record["institution"])
            issuer = agency if scope == "mois" else _node("Agency", issuer_key)
            quads = list(_record_quads("DocumentVersion", doc_id, record))
            quads += [_quad(doc_node, "docId", doc_id), _quad(doc_node, "scope", scope),
                      _quad(doc_node, "issuedBy", issuer), _typed(issuer, "Agency"),
                      _quad(issuer, "label", record["institution"] or "unknown")]
            if record["content_hash"]:
                quads.append(_quad(doc_node, "contentHash", record["content_hash"]))
            if record["source_id"]:
                source_node = _node("SourceDocument", record["source_id"])
                quads += [_typed(source_node, "SourceDocument"),
                          _quad(source_node, "sourceId", record["source_id"]),
                          _quad(doc_node, "versionOf", source_node)]
            if record["version"]:
                quads.append(_quad(doc_node, "version", str(record["version"])))
            store.extend(quads)
            for passage in admitted:
                pn = _node("Passage", passage["id"])
                store.extend([*list(_record_quads("Passage", passage["id"], passage)),
                              _quad(doc_node, "hasPassage", pn), _quad(pn, "text", passage["text"]),
                              _quad(pn, "valueHold", passage["value_hold"]),
                              _quad(pn, "start", passage["char_start"]), _quad(pn, "end", passage["char_end"])])
            counts["documents"] += 1
            counts[scope + "_documents"] += 1
            counts["passages"] += len(admitted)
            counts["held_passages"] += held_count
        if remaining_private:
            raise ValueError("Passage review IDs do not match this index")
        if selected != seen:
            raise ValueError("Selected inventory metadata has no indexed document record")
        facts_by_id = {fact["id"]: fact for fact in facts}
        for fact in facts:
            _validate_fact(fact, metadata, evidence_passages, facts_by_id)
            store.extend(_fact_quads(fact))
        counts["missing_documents"] = sum(not d["passage_count"] for d in metadata.values())
        store.flush()
        store = None
        # Bind exact persisted database bytes; read_only never creates a writer.
        files = {str(p.relative_to(output_dir)): _sha(p) for p in sorted((output_dir / "store").rglob("*")) if p.is_file()}
        release = "mois-" + hashlib.sha256(_json(hashes).encode()).hexdigest()[:20]
        result = {"schema_version": SCHEMA_VERSION, "ontology_version": ONTOLOGY_VERSION,
                  "release": release, "engine": "pyoxigraph", "engine_version": "0.5.13",
                  "source_hashes": hashes, "database_files": files, "counts": counts,
                  "coverage": {**counts, "admitted_passages": counts["passages"] - counts["held_passages"]},
                  "source_notices": ["Indexed snapshot, not current legal validity.",
                                     f"{counts['mois_documents']} MOIS inventory entries are not necessarily complete regulations.",
                                     "Other-agency references retain their issuer and do not imply applicability.",
                                     "Passage adjacency is not a legal relation; only curated evidence creates rules."],
                  "source_snapshot": source_manifest.get("built_at_unix")}
        temporary = output_dir / "manifest.json.tmp"
        temporary.write_text(_json(result) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir / "manifest.json")
        return result
    except BaseException:
        store = None
        shutil.rmtree(output_dir)
        raise


class GraphStore:
    """Read-only graph projections; all evidence selection uses RDF joins."""

    def __init__(self, store: Store, metadata: dict):
        self._store, self.metadata = store, metadata

    @classmethod
    def open(cls, path: Path) -> "GraphStore":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("ontology_version") != ONTOLOGY_VERSION:
            raise ValueError("Unsupported MOIS graph schema/ontology")
        if manifest.get("engine") != "pyoxigraph" or not manifest.get("database_files"):
            raise ValueError("Invalid graph release manifest")
        for name, expected in manifest["database_files"].items():
            target = path / name
            if not target.resolve().is_relative_to((path / "store").resolve()) or not target.is_file() or _sha(target) != expected:
                raise ValueError("Graph release database integrity mismatch")
        return cls(Store.read_only(str(path / "store")), manifest)

    def _records(self, query: str, field="record") -> list[dict]:
        return [json.loads(row[field].value) for row in self._store.query(PREFIX + query)]

    def documents(self) -> list[dict]:
        return self._records("SELECT ?record WHERE { ?d a m:DocumentVersion; m:record ?record; m:docId ?id } ORDER BY ?id")

    def get_document(self, doc_id: str) -> dict | None:
        rows = self._records(f"SELECT ?record WHERE {{ {_node('DocumentVersion', doc_id)} a m:DocumentVersion; m:record ?record }}")
        return rows[0] if rows else None

    def facts(self, doc_id: str | None = None) -> list[dict]:
        document = str(_node("DocumentVersion", doc_id)) if doc_id is not None else "?d"
        return self._records(f"SELECT ?record WHERE {{ {document} a m:DocumentVersion; m:hasRule ?r . ?r a m:Rule; m:evidence/m:passage ?p; m:record ?record . {document} m:hasPassage ?p . ?p m:valueHold false }} ORDER BY ?r")

    def passages(self, doc_id: str) -> list[dict]:
        return self._records(f"SELECT ?record WHERE {{ {_node('DocumentVersion', doc_id)} m:hasPassage ?p . ?p a m:Passage; m:record ?record; m:start ?start }} ORDER BY ?start ?p")

    def references(self, doc_id: str) -> list[dict]:
        rows = self._store.query(PREFIX + f"SELECT ?record ?target WHERE {{ {_node('DocumentVersion', doc_id)} m:hasRule ?r . ?r a m:ReferenceRule; m:record ?record; m:referenceTarget ?t . OPTIONAL {{ ?t a m:DocumentVersion; m:record ?target }} }} ORDER BY ?r")
        result = []
        for row in rows:
            fact = json.loads(row["record"].value)
            target = json.loads(row["target"].value) if row["target"] is not None else None
            result.append({**fact, "resolved": target is not None, "target": target, "target_document": target,
                           "target_doc_id": target["doc_id"] if target else None,
                           "target_title": target["title"] if target else None,
                           "target_institution": target["institution"] if target else None})
        return result

    def search_passages(self, terms: list[str], doc_ids: list[str] | None = None, limit=12) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100")
        if len(terms) > 16 or any(not isinstance(t, str) or len(t) > 200 for t in terms):
            raise ValueError("Search accepts at most 16 terms of 200 characters")
        terms = list(dict.fromkeys(t.strip().lower() for t in terms if t.strip()))
        if not terms or doc_ids == []:
            return []
        if doc_ids is not None and (len(doc_ids) > 256 or any(not isinstance(d, str) for d in doc_ids)):
            raise ValueError("Invalid document scope")
        scope = ('?d m:scope "mois" .' if doc_ids is None else
                 "VALUES ?d { " + " ".join(str(_node("DocumentVersion", d)) for d in doc_ids) + " }")
        tests = [f"CONTAINS(LCASE(?text), {Literal(term)})" for term in terms]
        score = " + ".join(f"IF({test}, 1, 0)" for test in tests)
        query = f"SELECT ?record ?document ?score WHERE {{ {scope} ?d a m:DocumentVersion; m:record ?document; m:hasPassage ?p . ?p a m:Passage; m:record ?record; m:text ?text; m:valueHold false . BIND(({score}) AS ?score) FILTER(?score > 0) }} ORDER BY DESC(?score) ?d ?p LIMIT {limit}"
        result = []
        for row in self._store.query(PREFIX + query):
            result.append({**json.loads(row["record"].value), "document": json.loads(row["document"].value),
                           "score": int(row["score"].value)})
        return result
