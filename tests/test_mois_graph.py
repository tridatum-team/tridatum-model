"""Consumer-visible invariants against real persisted Oxigraph stores."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from pyoxigraph import Store
except ModuleNotFoundError as error:
    if error.name != "pyoxigraph":
        raise
    raise unittest.SkipTest("Install the graph extra to run persisted graph tests") from error

from trid.service.mois_graph import GraphStore, build_graph


def make_inputs(tmp_path: Path):
    index = tmp_path / "index"
    index.mkdir()
    text = "출장계획서는 출장 20일 전 제출한다. 토요일과 공휴일은 제외한다."
    docs = []
    meta = {}
    for doc_id, institution, body in (("mois:1", "행정안전부", text),
                                       ("ref:2", "인사혁신처", "공무원 여비 규정을 참조한다."),
                                       ("historic:3", "행정자치부", "과거 기관 문서"),
                                       ("mois:empty", "행정안전부", "")):
        meta[doc_id] = {"title": doc_id, "institution": institution, "kind": "행정규칙"}
        docs.append({"doc_id": doc_id, "title": doc_id, "acl": ["public"],
                     "sha256": hashlib.sha256(body.encode()).hexdigest(), "text": body,
                     "passages": ([{"id": doc_id + ":p", "doc_id": doc_id, "text": body,
                                    "char_start": 0, "char_end": len(body), "path": ["제1조"]}] if body else [])})
    fact = {"id": "deadline", "doc_id": "mois:1", "content_hash": docs[0]["sha256"],
            "type": "ProcedureRule", "subject": "출장계획서", "predicate": "deadline",
            "object": text, "topics": ["출장", "제출"],
            "qualifiers": {"trigger": "출장", "direction": "before", "days": 20,
                           "excluded_days": ["토요일", "공휴일"]},
            "evidence": {"passage_id": "mois:1:p", "start": 0, "end": len(text), "quote": text}}
    resolved = {**fact, "id": "reference", "type": "ReferenceRule", "predicate": "references",
                "object": "공무원 여비 규정", "target_doc_id": "ref:2"}
    unresolved = {**resolved, "id": "unresolved", "object": "위임전결규정"}
    unresolved.pop("target_doc_id")
    facts = [fact, resolved, unresolved]
    (index / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (index / "documents.json").write_text(json.dumps(docs, ensure_ascii=False))
    facts_path = tmp_path / "facts.jsonl"
    write_facts(facts_path, facts)
    return index, facts_path, docs, facts


def write_facts(path, facts):
    path.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts))


class MoisGraphTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def inputs(self, label=None):
        root = self.tmp_path
        if label is not None:
            root = root / label
            root.mkdir()
        return make_inputs(root)

    def test_persisted_release_traverses_rules_evidence_and_reference_issuers(self):
        index, facts_path, _, facts = self.inputs()
        release = self.tmp_path / "release"
        manifest = build_graph(index, release, facts_path)
        store = GraphStore.open(release)
        self.assertEqual(manifest["counts"]["mois_documents"], 2)
        self.assertEqual(manifest["counts"]["reference_documents"], 1)
        self.assertEqual({d["doc_id"] for d in store.documents()}, {"mois:1", "mois:empty", "ref:2"})
        self.assertIsNone(store.get_document("historic:3"))
        self.assertEqual(store.get_document("mois:empty")["content_status"], "held_or_missing")
        self.assertIsNone(store.get_document("mois:1")["retrieved_at"])
        self.assertEqual({f["id"] for f in store.facts("mois:1")}, {f["id"] for f in facts})
        self.assertEqual(store.facts("ref:2"), [])
        rule = next(f for f in store.facts() if f["id"] == "deadline")
        self.assertEqual(rule["qualifiers"]["excluded_days"], ["토요일", "공휴일"])
        self.assertEqual(rule["evidence"]["quote"], store.passages("mois:1")[0]["text"])
        refs = {f["id"]: f for f in store.references("mois:1")}
        self.assertIs(refs["reference"]["resolved"], True)
        self.assertEqual(refs["reference"]["target_institution"], "인사혁신처")
        self.assertEqual(refs["reference"]["target"]["scope"], "reference")
        self.assertIs(refs["unresolved"]["resolved"], False)
        self.assertIsNone(refs["unresolved"]["target"])
        # Read-only reopening does not change any sealed database bytes.
        self.assertEqual(GraphStore.open(release).metadata["release"], manifest["release"])
        with self.assertRaises(FileExistsError):
            build_graph(index, release, facts_path)

    def test_search_scope_and_user_text_cannot_become_sparql(self):
        index, facts_path, _, _ = self.inputs()
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        store = GraphStore.open(release)
        self.assertEqual(store.search_passages(["여비"]), [])
        self.assertEqual(store.search_passages(["여비"], ["ref:2"])[0]["document"]["institution"], "인사혁신처")
        self.assertEqual(store.search_passages(['") } UNION { ?s ?p ?o } #']), [])
        self.assertIsNone(store.get_document('x> . ?s ?p ?o . <x'))
        self.assertEqual(store.search_passages(["출장"], []), [])
        self.assertEqual(len(store.search_passages(["출장"], limit=1)), 1)
        with self.assertRaises(ValueError):
            store.search_passages(["출장"], limit=101)

    def test_invalid_fact_rejects_entire_publication(self):
        for mutation in ("hash", "quote", "offset", "endpoint", "type", "predicate", "duplicate"):
            with self.subTest(mutation=mutation):
                index, facts_path, _, facts = self.inputs(mutation)
                if mutation == "hash":
                    facts[0]["content_hash"] = "0" * 64
                elif mutation == "quote":
                    facts[0]["evidence"]["quote"] = "지원되지 않는 인용"
                elif mutation == "offset":
                    facts[0]["evidence"]["start"] = -1
                elif mutation == "endpoint":
                    facts[1]["target_doc_id"] = "missing"
                elif mutation == "type":
                    facts[0]["type"] = "InventedRule"
                elif mutation == "predicate":
                    facts[0]["predicate"] = "sameAs"
                else:
                    facts.append(dict(facts[0]))
                write_facts(facts_path, facts)
                release = facts_path.parent / "release"
                with self.assertRaises(ValueError):
                    build_graph(index, release, facts_path)
                self.assertFalse(release.exists())

    def test_review_protection_is_not_bypassed_by_curated_facts(self):
        reviews = ({"privacy_review": True}, {"decision": "hold_for_value_use"})
        for number, review in enumerate(reviews):
            with self.subTest(review=review):
                index, facts_path, _, _ = self.inputs(str(number))
                (index / "review.jsonl").write_text(
                    json.dumps({"passage_id": "mois:1:p", "value_review": review}) + "\n")
                release = facts_path.parent / "release"
                with self.assertRaisesRegex(ValueError, "absent, private, held"):
                    build_graph(index, release, facts_path)
                self.assertFalse(release.exists())

    def test_private_acl_and_value_hold_are_not_searchable(self):
        index, facts_path, docs, _ = self.inputs()
        write_facts(facts_path, [])
        docs[0]["acl"] = ["staff"]
        held = {"doc_id": "mois:held", "title": "held", "text": "보류 금액 100원", "acl": ["public"],
                "sha256": "unknown", "passages": [{"id": "mois:held:p", "doc_id": "mois:held",
                "text": "보류 금액 100원", "char_start": 0, "char_end": 10, "path": []}]}
        docs.append(held)
        meta = json.loads((index / "meta.json").read_text())
        meta["mois:held"] = {"title": "held", "institution": "행정안전부", "kind": "별표"}
        (index / "meta.json").write_text(json.dumps(meta))
        (index / "documents.json").write_text(json.dumps(docs))
        (index / "review.jsonl").write_text(json.dumps({
            "passage_id": "mois:held:p", "value_review": {"decision": "hold_for_value_use"}}))
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        store = GraphStore.open(release)
        self.assertEqual(store.passages("mois:1"), [])
        self.assertEqual(store.search_passages(["출장", "100"]), [])
        self.assertIs(store.passages("mois:held")[0]["value_hold"], True)

    def test_empty_document_text_preserves_admitted_attachment_passages(self):
        index, facts_path, docs, _ = self.inputs()
        docs[1]["text"] = ""
        (index / "documents.json").write_text(json.dumps(docs))
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        store = GraphStore.open(release)
        self.assertEqual(store.get_document("ref:2")["content_status"], "indexed_passages_only")
        self.assertEqual(store.passages("ref:2")[0]["text"], "공무원 여비 규정을 참조한다.")

    def test_review_index_identity_mismatch_fails_closed(self):
        index, facts_path, _, _ = self.inputs()
        (index / "review.jsonl").write_text(json.dumps({
            "passage_id": "not-indexed", "value_review": {"privacy_review": True}}))
        with self.assertRaisesRegex(ValueError, "review IDs"):
            build_graph(index, self.tmp_path / "release", facts_path)

    def test_corrupt_release_and_unsupported_schema_are_rejected(self):
        index, facts_path, _, _ = self.inputs()
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        path = release / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["schema_version"] = 999
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "schema"):
            GraphStore.open(release)
        manifest["schema_version"] = 1
        name = next(iter(manifest["database_files"]))
        manifest["database_files"][name] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "integrity"):
            GraphStore.open(release)

    def test_malformed_stream_does_not_publish_partial_graph(self):
        index, facts_path, _, _ = self.inputs()
        with (index / "documents.json").open("a") as handle:
            handle.write(" trailing garbage")
        release = self.tmp_path / "release"
        with self.assertRaisesRegex(ValueError, "Trailing data"):
            build_graph(index, release, facts_path)
        self.assertFalse(release.exists())

    def test_unverified_indexed_tables_are_inventory_not_answer_evidence(self):
        index, facts_path, _, _ = self.inputs()
        write_facts(facts_path, [])
        (index / "tables.json").write_text(json.dumps(["mois:1:p"]))
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        store = GraphStore.open(release)
        self.assertEqual(store.search_passages(["20일"]), [])
        self.assertIs(store.passages("mois:1")[0]["table_unverified"], True)

    def test_content_hash_is_bound_to_text_not_an_unverified_index_label(self):
        index, facts_path, docs, _ = self.inputs()
        docs[0]["text"] += " 추가 문장"
        (index / "documents.json").write_text(json.dumps(docs))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_graph(index, self.tmp_path / "release", facts_path)

    def test_qualified_rule_has_persisted_typed_evidence_path(self):
        index, facts_path, _, facts = self.inputs()
        facts.append({**facts[0], "id": "exception", "predicate": "exception",
                      "qualifiers": {"exception_to": "deadline", "provenance": "form footnote"}})
        write_facts(facts_path, facts)
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        graph = Store.read_only(str(release / "store"))
        rows = list(graph.query("""
            PREFIX m: <https://tridatum.co/ontology/mois/>
            SELECT ?quote ?days WHERE {
                ?exception a m:ProcedureRule; m:qualifiesRule ?rule; m:evidence/m:quote ?quote .
                ?rule a m:Rule; m:qualifier ?qualifier .
                ?qualifier m:key "days"; m:value ?days .
            }
        """))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quote"].value, facts[0]["evidence"]["quote"])
        self.assertEqual(rows[0]["days"].value, "20")
        self.assertTrue(rows[0]["days"].datatype.value.endswith("#integer"))

    def test_invalid_qualification_endpoint_rejects_publication(self):
        for target in ("missing", "exception", "foreign-rule"):
            with self.subTest(target=target):
                index, facts_path, docs, facts = self.inputs(target)
                foreign = {**facts[0], "id": "foreign-rule", "doc_id": "ref:2",
                           "content_hash": docs[1]["sha256"],
                           "evidence": {"passage_id": "ref:2:p", "start": 0,
                                        "end": len(docs[1]["text"]), "quote": docs[1]["text"]}}
                facts += [foreign, {**facts[0], "id": "exception", "predicate": "exception",
                                    "qualifiers": {"exception_to": target}}]
                write_facts(facts_path, facts)
                with self.assertRaisesRegex(ValueError, "exception_to"):
                    build_graph(index, facts_path.parent / "release", facts_path)

    def test_inventory_keeps_every_exact_mois_document_without_alias_expansion(self):
        index, facts_path, docs, _ = self.inputs()
        write_facts(facts_path, [])
        meta = json.loads((index / "meta.json").read_text())
        for number in range(141):
            doc_id = f"mois:inventory:{number}"
            meta[doc_id] = {"title": doc_id, "institution": "행정안전부", "kind": "서식"}
            docs.append({"doc_id": doc_id, "title": doc_id, "acl": ["public"],
                         "text": "", "sha256": None, "passages": []})
        (index / "meta.json").write_text(json.dumps(meta))
        (index / "documents.json").write_text(json.dumps(docs))
        release = self.tmp_path / "release"
        build_graph(index, release, facts_path)
        store = GraphStore.open(release)
        self.assertEqual(len(store.documents()), 143)
        self.assertEqual(store.metadata["counts"]["mois_documents"], 143)
        self.assertIsNone(store.get_document("historic:3"))

    def test_indexed_passage_protection_survives_without_sidecar(self):
        cases = (
            {"acl": ["staff"]},
            {"private": True},
            {"value_review": {"privacy_review": True}},
            {"review": {"decision": "hold_for_value_use"}},
        )
        for number, fields in enumerate(cases):
            with self.subTest(fields=fields):
                index, facts_path, docs, _ = self.inputs(str(number))
                docs[0]["passages"][0].update(fields)
                (index / "documents.json").write_text(json.dumps(docs))
                with self.assertRaisesRegex(ValueError, "absent, private, held"):
                    build_graph(index, facts_path.parent / "release", facts_path)


if __name__ == "__main__":
    unittest.main()
