"""Candidate safety gates against isolated persisted graphs; no private data or downloads."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from pyoxigraph import Store  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "pyoxigraph":
        raise
    raise unittest.SkipTest("Install the graph extra to run persisted graph tests") from error

from scripts import prepare_grounded_experiment as prep
from trid.service.mois_graph import GraphStore, build_graph


class CandidateSafetyTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        index = self.root / "index"
        index.mkdir()
        docs, metadata, families = [], {}, []
        for number, (fid, split) in enumerate((("admrul:41555", "train"), ("admrul:91382", "train"), ("admrul:43010", "dev")), 1):
            text = f"Synthetic rule {number}: the blue permit applies only after approval."
            doc_id, pid = f"synthetic:{number}", f"synthetic:{number}:p1"
            digest = prep.text_sha(text)
            docs.append({"doc_id": doc_id, "title": f"Synthetic {number}", "acl": ["public"], "sha256": digest,
                         "text": text, "passages": [{"id": pid, "doc_id": doc_id, "text": text,
                                                      "char_start": 0, "char_end": len(text), "path": ["Rule 1"]}]})
            metadata[doc_id] = {"title": f"Synthetic {number}", "institution": "행정안전부", "kind": "행정규칙", "license": "CC0"}
            families.append({"family_id": fid, "group_key": fid, "split": split, "doc_id": doc_id,
                             "source_id": f"synthetic-source:{number}", "version": str(number), "content_hash": digest,
                             "license": "CC0", "rights": {"license": "CC0"}, "source_record": "synthetic-only.jsonl",
                             "passage_pins": [{"passage_id": pid, "quote_sha256": digest, "char_start": 0, "char_end": len(text)}],
                             "excluded_passage_ids": []})
        (index / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        (index / "documents.json").write_text(json.dumps(docs), encoding="utf-8")
        facts = self.root / "facts.jsonl"
        facts.write_text("", encoding="utf-8")
        release = self.root / "graph"
        build_graph(index, release, facts)
        self.store = GraphStore.open(release)
        for family in families:
            family["source_document"] = self.store.get_document(family["doc_id"])
        self.allocation = {"graph_release": self.store.metadata["release"], "families": families}
        self.history = {row["family_id"]: deepcopy(row) for row in families}
        self.approved = {(row["doc_id"], pin["passage_id"]): {k: pin[k] for k in ("quote_sha256", "char_start", "char_end")}
                         for row in families for pin in row["passage_pins"]}
        self.contract = {"independent_acceptance_reserved_families": ["reserved:independent"],
                         "behavior_counts_per_split": {"train": 6, "dev": 2},
                         "development_exposure": "Synthetic development fixture; not an acceptance claim."}
        self.specs = []
        questions = {
            "conditions": "What prerequisite must the blue permit satisfy?",
            "exceptions": "Does this excerpt exempt the blue permit from approval?",
            "normative_force": "Is prior approval optional for the blue permit?",
            "information_boundary": "Does this excerpt identify who grants blue permit approval?",
        }
        for split, count in (("train", 6), ("dev", 2)):
            selected = [f for f in families if f["split"] == split]
            for behavior in prep.BEHAVIORS:
                for i in range(count):
                    family = selected[i % len(selected)]
                    self.specs.append({"id": f"{split}-{behavior}-{i}", "family_id": family["family_id"],
                                       "doc_id": family["doc_id"], "split": split, "behavior": behavior,
                                       "passage_ids": [family["passage_pins"][0]["passage_id"]],
                                       "question": f"{questions[behavior]} Consider application {i}.",
                                       "target": {"paragraphs": [{"text": "The permit requires prior approval.", "sources": ["E1"]}]},
                                       "review_focus": "Synthetic approval condition."})

    def prepare(self, reserved=()):
        return prep.prepare_candidates(self.contract, self.allocation, self.specs, self.store,
                                       self.history, self.approved, reserved)

    def test_explicit_contract_pin_rejects_tampering_but_allows_repinned_metadata(self):
        scope = self.root / "scope.json"
        scope.write_text(prep.encoded({"base_model": prep.MODEL, "base_revision": prep.REVISION,
                                      "actual_training_requires_new_bounded_gpu_approval": True}), encoding="utf-8")
        dependency = self.root / "dependency.json"
        dependency.write_text("{}", encoding="utf-8")
        contract = {"schema": "trid-grounded-experiment-preparation-v1", "base_model": prep.MODEL,
                    "base_revision": prep.REVISION, "adapter": "grounded_messages", "enable_thinking": False,
                    "context_tokens": 3072, "answer_tokens_including_eos": 768, "train_rows": 24, "dev_rows": 8,
                    "behavior_counts_per_split": {"train": 6, "dev": 2}, "behaviors": list(prep.BEHAVIORS),
                    "train_families": sorted(prep.FAMILIES["train"]), "dev_families": sorted(prep.FAMILIES["dev"]),
                    "training_authorized": False, "aws_calls_authorized": False, "deployment_authorized": False,
                    "scope": prep.pin(scope)}
        for name in ("historical_source_allocation", "source_only_sidecar", "reservation_evidence", "tokenizer_acquisition"):
            contract[name] = prep.pin(dependency)
        path = self.root / "contract.json"
        path.write_text(prep.encoded(contract), encoding="utf-8")
        initial_pin = prep.file_sha(path)
        self.assertEqual(prep.load_contract(path, initial_pin), contract)
        contract["metadata_note"] = "Synthetic metadata-only revision."
        path.write_text(prep.encoded(contract), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract bytes changed"):
            prep.load_contract(path, initial_pin)
        self.assertEqual(prep.load_contract(path, prep.file_sha(path)), contract)
        contract["base_model"] = "different-model"
        path.write_text(prep.encoded(contract), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            prep.load_contract(path, prep.file_sha(path))

    def test_disjoint_candidate_splits_preserve_exact_source_provenance(self):
        rows, sidecars = self.prepare()
        self.assertEqual(len(rows), 32)
        self.assertEqual(len(sidecars), 32)
        self.assertEqual(len({(prep.encoded(row["messages"]), row["answer"]) for row in rows}), 32)
        train = [r for r in rows if r["split"] == "train"]
        dev = [r for r in rows if r["split"] == "dev"]
        self.assertEqual((len(train), len(dev)), (24, 8))
        self.assertFalse({r["provenance"]["family_id"] for r in train} & {r["provenance"]["family_id"] for r in dev})
        for row, sidecar in zip(rows, sidecars):
            self.assertEqual(row["provenance"]["admission_status"], "NOT_ADMITTED")
            self.assertEqual(row["provenance"]["messages_sha256"], prep.text_sha(prep.encoded(row["messages"])))
            self.assertEqual(row["provenance"]["target_sha256"], prep.text_sha(row["answer"]))
            validation = row["provenance"]["authored_target_validation"]
            self.assertFalse(validation["inference_performed"])
            self.assertEqual(validation["semantic_verification"], "unverified")
            self.assertEqual(validation["validation"], {"citations": "passed", "numeric_values": "passed"})
            self.assertTrue(all(p["whole_passage"] for p in sidecar["passages"]))
            packet = json.loads(row["messages"][1]["content"])["source_data"]
            self.assertEqual(packet["evidence"][0]["quote"], sidecar["passages"][0]["quote"])

    def test_train_candidate_cannot_use_development_source_family(self):
        dev_family = self.allocation["families"][-1]
        self.specs[0].update(family_id=dev_family["family_id"], doc_id=dev_family["doc_id"],
                             passage_ids=[dev_family["passage_pins"][0]["passage_id"]])
        with self.assertRaisesRegex(ValueError, "source-family leakage"):
            self.prepare()

    def test_family_cannot_be_allocated_to_both_splits(self):
        family = deepcopy(self.allocation["families"][0])
        family["split"] = "dev"
        self.allocation["families"].append(family)
        with self.assertRaisesRegex(ValueError, "source-family leakage"):
            self.prepare()

    def test_reserved_family_is_rejected_before_candidate_publication(self):
        with self.assertRaisesRegex(ValueError, "reserved family"):
            self.prepare([{"doc_family_id": "admrul:41555"}])

    def test_tampered_source_pin_and_metadata_are_rejected(self):
        for field in ("content_hash", "license", "source_document", "passage_pins"):
            with self.subTest(field=field):
                original = deepcopy(self.allocation["families"][0][field])
                if field == "source_document":
                    self.allocation["families"][0][field]["title"] = "Changed"
                elif field == "passage_pins":
                    self.allocation["families"][0][field][0]["quote_sha256"] = "0" * 64
                else:
                    self.allocation["families"][0][field] = "changed"
                with self.assertRaises(ValueError):
                    self.prepare()
                self.allocation["families"][0][field] = original

    def test_excluded_passage_cannot_supply_target_evidence(self):
        family = self.allocation["families"][0]
        family["excluded_passage_ids"] = [family["passage_pins"][0]["passage_id"]]
        with self.assertRaisesRegex(ValueError, "excluded passage"):
            self.prepare()

    def test_fixed_scope_exclusion_cannot_be_removed_from_allocation(self):
        self.specs[0]["passage_ids"] = [sorted(prep.EXCLUDED_PASSAGES)[0]]
        for family in self.allocation["families"]:
            family["excluded_passage_ids"] = []
        with self.assertRaisesRegex(ValueError, "excluded passage"):
            self.prepare()

    def test_unknown_target_citation_is_rejected(self):
        self.specs[0]["target"]["paragraphs"][0]["sources"] = ["E2"]
        with self.assertRaisesRegex(ValueError, "invalid_citations"):
            self.prepare()

    def test_production_validator_rejects_unsupported_numeric_targets(self):
        for text, reason in (("Approval takes 99 hours.", "unsupported_numeric_value"),
                             ("두 시간 조건을 적용합니다.", "unsupported_written_numeric_value")):
            with self.subTest(reason=reason):
                self.specs[0]["target"]["paragraphs"][0]["text"] = text
                with self.assertRaisesRegex(ValueError, reason):
                    self.prepare()

    def test_duplicate_content_is_rejected_despite_new_id_or_behavior(self):
        for index in (1, 6):
            with self.subTest(index=index):
                original = deepcopy(self.specs[index])
                self.specs[index] = {**deepcopy(self.specs[0]), "id": original["id"],
                                     "behavior": original["behavior"]}
                with self.assertRaisesRegex(ValueError, "duplicate candidate messages/answer content"):
                    self.prepare()
                self.specs[index] = original

    def test_actual_grounded_input_overflow_fails_without_dropping_or_truncation(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
        backend = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
        template = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
                    "{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}")
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                     additional_special_tokens=["<|im_start|>"], chat_template=template)
        rows, _ = self.prepare()
        path = self.root / "candidates.jsonl"
        path.write_text(prep.encoded(rows[0]) + "\n", encoding="utf-8")
        entry = {"path": path.name, "sha256": prep.file_sha(path), "adapter": "grounded_messages", "split": "train"}
        items, measurements = prep.audit_items(tok, [entry], self.root, 20000, 2000)
        self.assertEqual(len(items), 1)
        total = measurements[0]["total_tokens"]
        exact, _ = prep.audit_items(tok, [entry], self.root, total, 2000)
        self.assertEqual(exact, items)
        with self.assertRaisesRegex(ValueError, "over_context_budget"):
            prep.audit_items(tok, [entry], self.root, total - 1, 2000)
        with self.assertRaisesRegex(ValueError, "over_answer_budget"):
            prep.audit_items(tok, [entry], self.root, total, measurements[0]["answer_tokens_including_eos"] - 1)


if __name__ == "__main__":
    unittest.main()
