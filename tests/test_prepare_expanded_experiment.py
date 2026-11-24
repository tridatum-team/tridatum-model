"""Expanded admission gates on deliberately synthetic cases, never historical artifacts.

No model, network, or actual experiment outputs are needed by these unit tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_expanded_experiment as prep
from scripts import train_open_weight_lora as trainer


GENERAL_BEHAVIORS = (
    "instruction_following", "summarization", "conditional_reasoning", "scope_negation",
    "information_boundary", "structured_transformation", "arithmetic_units", "conversation_context",
)
RUBRIC_FIELDS = ("required", "forbidden", "critical_errors", "accepted_variants")


def canonical_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def synthetic_contract():
    return {
        "schema": "trid-expanded-authoring-contract-v1",
        "general_behaviors": list(GENERAL_BEHAVIORS),
        "record_required": ["id", "split", "cohort", "behavior", "family_id", "messages",
                            "reference_answer", "rubric", "provenance"],
        "general_surface_partition": {"train": ["synthetic_workshop"], "evaluation": ["synthetic_museum"]},
    }


def synthetic_record(identifier="synthetic-train-0", split="train", cohort="general", behavior=None):
    return {
        "id": identifier,
        "split": split,
        "cohort": cohort,
        "behavior": behavior or GENERAL_BEHAVIORS[0],
        "family_id": f"synthetic-family:{identifier}",
        "messages": [{"role": "user", "content": f"가상 사례 {identifier}: 표식은 파랑입니다. 표식만 답하세요."}],
        "reference_answer": "파랑",
        "rubric": {
            "required": ["제시된 파랑 표식을 답한다."],
            "forbidden": ["표식을 빨강으로 바꾼다."],
            "critical_errors": ["제시되지 않은 다른 표식을 사실로 답한다."],
            "accepted_variants": ["파랑"],
        },
        "provenance": {
            "kind": "synthetic",
            "surface_domain": "synthetic_workshop" if split == "train" else "synthetic_museum",
            "creation_method": "Isolated synthetic unit-test fixture, not an authored experiment record.",
            "external_sources_used": False,
            "model_outputs_seen_during_authoring": False,
            "protected_records_accessed": False,
            "admission_status": "NOT_ADMITTED",
        },
        "admission_status": "NOT_ADMITTED",
    }


class CanonicalCaseHashTests(unittest.TestCase):
    def test_hash_uses_sorted_compact_utf8_json(self):
        record = {"한글": "파랑", "z": [2, {"b": False, "a": None}], "a": 1}
        self.assertEqual(prep.case_sha(record), canonical_sha(record))
        self.assertEqual(prep.case_sha(record), prep.case_sha(dict(reversed(list(record.items())))))

    def test_all_reviewed_record_fields_are_hash_bound(self):
        record = synthetic_record()
        original = prep.case_sha(record)
        for field, value in (("reference_answer", "빨강"), ("family_id", "synthetic-other-family"),
                             ("split", "evaluation"), ("behavior", GENERAL_BEHAVIORS[1])):
            with self.subTest(field=field):
                changed = deepcopy(record)
                changed[field] = value
                self.assertNotEqual(original, prep.case_sha(changed))
        for field in ("messages", "rubric", "provenance"):
            with self.subTest(field=field):
                changed = deepcopy(record)
                if field == "messages":
                    changed[field][0]["content"] += " 정정"
                elif field == "rubric":
                    changed[field]["required"].append("추가 조건")
                else:
                    changed[field]["creation_method"] += " changed"
                self.assertNotEqual(original, prep.case_sha(changed))

    def test_nonfinite_values_cannot_receive_a_review_hash(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prep.case_sha({"nested": [value]})


class RecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = synthetic_contract()
        self.record = synthetic_record()

    def test_valid_synthetic_record_is_accepted_without_mutation(self):
        before = deepcopy(self.record)
        prep.validate_record(self.record, self.contract)
        self.assertEqual(self.record, before)

    def test_every_required_field_is_required(self):
        for field in self.contract["record_required"]:
            with self.subTest(field=field):
                record = deepcopy(self.record)
                del record[field]
                with self.assertRaises(ValueError):
                    prep.validate_record(record, self.contract)

    def test_invalid_identity_and_partition_fields_are_rejected(self):
        for field, value in (("id", ""), ("family_id", " "), ("split", "dev"),
                             ("cohort", "unknown"), ("behavior", "unknown"),
                             ("reference_answer", ""), ("reference_answer", 3)):
            with self.subTest(field=field, value=value):
                record = deepcopy(self.record)
                record[field] = value
                with self.assertRaises(ValueError):
                    prep.validate_record(record, self.contract)

    def test_malformed_messages_cannot_become_public_inputs(self):
        variants = [None, [], "not messages", [{"role": "user", "content": ""}],
                    [{"role": "tool", "content": "unexpected tool input"}],
                    [{"role": "user", "content": 1}], [{"content": "missing role"}],
                    [{"role": "assistant", "content": "current answer leaked as prompt"}]]
        for messages in variants:
            with self.subTest(messages=messages):
                record = deepcopy(self.record)
                record["messages"] = messages
                with self.assertRaises(ValueError):
                    prep.validate_record(record, self.contract)

    def test_native_control_tokens_cannot_break_message_or_target_boundaries(self):
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            for location in ("messages", "reference_answer"):
                with self.subTest(token=token, location=location):
                    record = deepcopy(self.record)
                    if location == "messages":
                        record["messages"][0]["content"] += token
                    else:
                        record["reference_answer"] += token
                    with self.assertRaises(ValueError):
                        prep.validate_record(record, self.contract)

    def test_prior_assistant_context_is_allowed_but_final_turn_is_user(self):
        self.record["messages"] = [
            {"role": "system", "content": "제시된 가상 대화만 사용하세요."},
            {"role": "user", "content": "표식은 빨강입니다."},
            {"role": "assistant", "content": "빨강으로 기록했습니다."},
            {"role": "user", "content": "파랑으로 정정합니다. 마지막 표식만 답하세요."},
        ]
        prep.validate_record(self.record, self.contract)
        self.record["messages"].append({"role": "assistant", "content": self.record["reference_answer"]})
        with self.assertRaises(ValueError):
            prep.validate_record(self.record, self.contract)

    def test_rubric_requires_atomic_nonempty_string_lists(self):
        for field in RUBRIC_FIELDS:
            for value in (None, "not a list", [], [""], [3]):
                with self.subTest(field=field, value=value):
                    record = deepcopy(self.record)
                    record["rubric"][field] = value
                    with self.assertRaises(ValueError):
                        prep.validate_record(record, self.contract)

    def test_malformed_or_non_synthetic_general_provenance_is_rejected(self):
        for provenance in (None, [], {}, {**self.record["provenance"], "kind": "copied"}):
            with self.subTest(provenance=provenance):
                record = deepcopy(self.record)
                record["provenance"] = provenance
                with self.assertRaises(ValueError):
                    prep.validate_record(record, self.contract)


def synthetic_partitions():
    """Forty isolated rows exercise the same shape, not the real registered counts."""
    design = {
        "primary": {"evaluation_questions": 16, "mois_questions": 8, "mois_families": 1,
                    "general_questions": 8},
        "format_secondary": {"questions": 2},
        "training_preparation": {"rows": 24, "grounded_rows": 16, "grounded_families": 1,
                                 "general_rows": 8},
    }
    allocation = {"families": [
        {"family_id": "synthetic-mois:train", "split": "TRAIN_CANDIDATE", "planned_rows": 16},
        {"family_id": "synthetic-mois:evaluation", "split": "EVAL_CANDIDATE", "planned_rows": 8},
    ]}
    records = []
    for split, repeats in (("train", 2), ("evaluation", 1)):
        for behavior in prep.MOIS_BEHAVIORS:
            for number in range(repeats):
                record = synthetic_record(f"synthetic-{split}-mois-{behavior}-{number}",
                                          split, "mois", behavior)
                record["messages"].insert(0, {"role": "system", "content": "가상 제시문만 사용하세요."})
                record["family_id"] = f"synthetic-mois:{split}"
                record["provenance"]["family_id"] = record["family_id"]
                records.append(record)
        for behavior in GENERAL_BEHAVIORS:
            records.append(synthetic_record(f"synthetic-{split}-general-{behavior}", split,
                                            "general", behavior))
    return records, design, synthetic_contract(), allocation


class PartitionValidationTests(unittest.TestCase):
    def setUp(self):
        self.records, self.design, self.contract, self.allocation = synthetic_partitions()

    def validate(self):
        return prep.validate_partitions(self.records, self.design, self.contract, self.allocation)

    def test_exact_synthetic_registered_partitions_are_accepted_without_mutation(self):
        before = deepcopy((self.records, self.design, self.contract, self.allocation))
        self.validate()
        self.assertEqual((self.records, self.design, self.contract, self.allocation), before)

    def test_duplicate_id_is_rejected_even_with_distinct_public_messages(self):
        self.records[1]["id"] = self.records[0]["id"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_public_messages_are_rejected_despite_new_answer_and_id(self):
        self.records[1]["messages"] = deepcopy(self.records[0]["messages"])
        self.records[1]["reference_answer"] = "다른 답"
        with self.assertRaises(ValueError):
            self.validate()

    def test_identical_public_messages_cannot_cross_partitions(self):
        training = next(row for row in self.records if row["split"] == "train")
        evaluation = next(row for row in self.records if row["split"] == "evaluation")
        evaluation["messages"] = deepcopy(training["messages"])
        with self.assertRaises(ValueError):
            self.validate()

    def test_general_family_overlap_is_rejected_with_different_public_messages(self):
        training = next(row for row in self.records if row["split"] == "train" and row["cohort"] == "general")
        evaluation = next(row for row in self.records if row["split"] == "evaluation" and row["cohort"] == "general")
        evaluation["family_id"] = training["family_id"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_mois_family_cannot_be_reallocated_across_partitions(self):
        for record in self.records:
            if record["cohort"] == "mois" and record["split"] == "evaluation":
                record["family_id"] = "synthetic-mois:train"
                record["provenance"]["family_id"] = record["family_id"]
        self.allocation["families"][1]["family_id"] = "synthetic-mois:train"
        with self.assertRaises(ValueError):
            self.validate()

    def test_renamed_families_cannot_hide_shared_source_identity_across_splits(self):
        for field in ("group_key", "doc_id", "source_id", "body_sha256"):
            with self.subTest(field=field):
                allocation = deepcopy(self.allocation)
                for family in allocation["families"]:
                    family[field] = text_sha("synthetic shared body") if field == "body_sha256" else "synthetic-shared-source"
                with self.assertRaises(ValueError):
                    prep.validate_partitions(self.records, self.design, self.contract, allocation)

    def test_unallocated_mois_family_is_rejected(self):
        for record in self.records:
            if record["cohort"] == "mois" and record["split"] == "train":
                record["family_id"] = "synthetic-mois:unallocated"
                record["provenance"]["family_id"] = record["family_id"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_allocation_split_must_match_authored_case_split(self):
        self.allocation["families"][0]["split"] = "EVAL_CANDIDATE"
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_or_additional_rows_do_not_silently_shrink_registered_counts(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                records = deepcopy(self.records)
                if delta < 0:
                    records.pop()
                else:
                    records.append(synthetic_record("synthetic-additional", "evaluation"))
                with self.assertRaises(ValueError):
                    prep.validate_partitions(records, self.design, self.contract, self.allocation)

    def test_registered_cohort_and_family_counts_are_enforced(self):
        for section, key in (("primary", "mois_questions"), ("primary", "general_questions"),
                             ("primary", "mois_families"), ("training_preparation", "grounded_rows"),
                             ("training_preparation", "general_rows"),
                             ("training_preparation", "grounded_families")):
            with self.subTest(section=section, key=key):
                design = deepcopy(self.design)
                design[section][key] += 1
                with self.assertRaises(ValueError):
                    prep.validate_partitions(self.records, design, self.contract, self.allocation)

    def test_per_family_planned_rows_cannot_change(self):
        self.allocation["families"][0]["planned_rows"] -= 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_mois_behavior_coverage_is_required_in_each_family(self):
        record = next(row for row in self.records if row["cohort"] == "mois" and row["split"] == "evaluation")
        record["behavior"] = next(behavior for behavior in prep.MOIS_BEHAVIORS if behavior != record["behavior"])
        with self.assertRaises(ValueError):
            self.validate()

    def test_general_behavior_balance_is_required_in_each_partition(self):
        record = next(row for row in self.records if row["cohort"] == "general" and row["split"] == "train")
        record["behavior"] = GENERAL_BEHAVIORS[1]
        with self.assertRaises(ValueError):
            self.validate()

    def test_general_surface_domain_must_belong_to_its_partition(self):
        for domain in ("synthetic_museum", "unregistered_surface"):
            with self.subTest(domain=domain):
                records = deepcopy(self.records)
                record = next(row for row in records if row["cohort"] == "general" and row["split"] == "train")
                record["provenance"]["surface_domain"] = domain
                with self.assertRaises(ValueError):
                    prep.validate_partitions(records, self.design, self.contract, self.allocation)


class SyntheticCharacterTokenizer:
    """Deterministic character IDs for gate tests, not a model tokenizer substitute."""
    eos_token = "\x03"
    eos_token_id = ord(eos_token) + 10

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens is not False:
            raise AssertionError("Admission must not insert extra special tokens.")
        return {"input_ids": [ord(character) + 10 for character in text]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        if tokenize is not False or add_generation_prompt is not True or enable_thinking is not False:
            raise AssertionError("Admission must use the nonthinking native generation prompt.")
        return "".join(f"<{turn['role']}>\n{turn['content']}\n" for turn in messages) + (
            "<assistant>\n" + trainer.QWEN3_NO_THINK
        )


class EncodedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tok = SyntheticCharacterTokenizer()
        self.row = {
            "id": "synthetic-history-budget",
            "adapter": "grounded_messages",
            "split": "train",
            "task": "synthetic_admission",
            "file": "synthetic-only.jsonl",
            "messages": [
                {"role": "system", "content": "제시된 대화만 사용합니다."},
                {"role": "user", "content": "처음 표식은 빨강입니다."},
                {"role": "assistant", "content": "빨강으로 기록했습니다."},
                {"role": "user", "content": "파랑으로 정정합니다. 마지막 표식만 답하세요."},
            ],
            "answer": "파랑",
        }
        prompt = self.tok.apply_chat_template(self.row["messages"], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        self.prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        self.answer_ids = self.tok(self.row["answer"] + self.tok.eos_token,
                                   add_special_tokens=False)["input_ids"]
        self.total = len(self.prompt_ids) + len(self.answer_ids)

    def test_exact_context_and_answer_budgets_include_eos_and_preserve_history(self):
        before = deepcopy(self.row)
        measurements = prep.audit_encoded(self.tok, self.row, self.total, len(self.answer_ids), trainer)
        self.assertEqual(measurements, {
            "id": self.row["id"], "prompt_tokens": len(self.prompt_ids),
            "answer_tokens_including_eos": len(self.answer_ids), "total_tokens": self.total,
        })
        self.assertEqual(self.row, before)

    def test_one_token_context_overflow_is_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            prep.audit_encoded(self.tok, self.row, self.total - 1, len(self.answer_ids), trainer)

    def test_eos_is_charged_to_answer_budget(self):
        with self.assertRaises(ValueError):
            prep.audit_encoded(self.tok, self.row, self.total, len(self.answer_ids) - 1, trainer)

    def test_dropped_encoder_row_is_not_silently_admitted(self):
        dropping = SimpleNamespace(messages_of=trainer.messages_of, encode=lambda *args: None)
        with self.assertRaises(ValueError):
            prep.audit_encoded(self.tok, self.row, self.total, len(self.answer_ids), dropping)

    def test_prompt_target_and_eos_corruptions_are_rejected_independently(self):
        expected = {
            "ids": self.prompt_ids + self.answer_ids,
            "labels": [-100] * len(self.prompt_ids) + self.answer_ids,
            "task": self.row["task"],
            "file": self.row["file"],
        }
        corruptions = {}
        turn_lengths = [len(f"<{turn['role']}>\n{turn['content']}\n") for turn in self.row["messages"]]
        for name, index in (("system_supervised", 0),
                            ("prior_assistant_supervised", sum(turn_lengths[:2])),
                            ("current_user_supervised", sum(turn_lengths[:3]))):
            item = deepcopy(expected)
            item["labels"][index] = item["ids"][index]
            corruptions[name] = item
        item = deepcopy(expected)
        item["labels"][len(self.prompt_ids)] = -100
        corruptions["current_answer_partly_masked"] = item
        item = deepcopy(expected)
        item["labels"][-1] = -100
        corruptions["eos_masked"] = item
        item = deepcopy(expected)
        item["labels"][len(self.prompt_ids)] += 1
        corruptions["wrong_current_target"] = item
        item = deepcopy(expected)
        item["ids"][0] += 1
        corruptions["prompt_token_changed"] = item
        item = deepcopy(expected)
        item["ids"][-2] += 1
        corruptions["answer_token_changed"] = item
        item = deepcopy(expected)
        item["ids"].pop()
        item["labels"].pop()
        corruptions["eos_omitted"] = item
        item = deepcopy(expected)
        item["labels"].append(self.tok.eos_token_id)
        corruptions["unequal_ids_labels_lengths"] = item
        item = deepcopy(expected)
        item["ids"].append(self.tok.eos_token_id)
        item["labels"].append(self.tok.eos_token_id)
        corruptions["extra_supervised_eos"] = item
        for name, item in corruptions.items():
            with self.subTest(corruption=name):
                corrupt_trainer = SimpleNamespace(messages_of=trainer.messages_of,
                                                   encode=lambda *args, item=item: deepcopy(item))
                with self.assertRaises(ValueError):
                    prep.audit_encoded(self.tok, self.row, self.total + 10,
                                       len(self.answer_ids) + 10, corrupt_trainer)


def synthetic_review_packet(records):
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
    packet = {
        "schema": "trid-expanded-data-review-packet-v1",
        "source_file": {"path": "synthetic-only.jsonl", "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()},
        "cases": [{"id": record["id"], "case_sha256": canonical_sha(record), "record": deepcopy(record)}
                  for record in records],
    }
    return canonical_sha(packet), packet


def synthetic_review(record, packet_sha, reviewer, verdict="pass", **judgment_fields):
    return {
        "schema": "trid-expanded-data-review-v1",
        "reviewer_id": reviewer,
        "packet_sha256": packet_sha,
        "judgments": [{
            "id": record["id"], "case_sha256": canonical_sha(record), "verdict": verdict,
            "reason": "Isolated fixture judgment of the exact supplied case, not a real data review.",
            "issues": [] if verdict == "pass" else ["Synthetic material disagreement."],
            "suggested_fix": None if verdict == "pass" else "Resolve the synthetic disagreement.",
            **judgment_fields,
        }],
    }


class ReviewAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.record = synthetic_record()
        self.digest, packet = synthetic_review_packet([self.record])
        self.packets = {self.digest: packet}
        self.reports = [synthetic_review(self.record, self.digest, name)
                        for name in ("synthetic-reviewer:a", "synthetic-reviewer:b")]

    def validate(self):
        return prep.validate_reviews([self.record], self.packets, self.reports)

    def test_two_independent_exact_hash_passes_resolve_without_mutation(self):
        before = deepcopy((self.record, self.packets, self.reports))
        self.assertEqual(self.validate(), {
            "schema": "trid-expanded-semantic-review-resolution-v1",
            "cases": [{"id": self.record["id"], "case_sha256": canonical_sha(self.record),
                       "reviewer_ids": ["synthetic-reviewer:a", "synthetic-reviewer:b"], "verdict": "pass"}],
            "ignored_stale_judgments": [],
            "ignored_stale_count": 0,
        })
        self.assertEqual((self.record, self.packets, self.reports), before)

    def test_repeated_report_from_one_reviewer_is_not_independence(self):
        self.reports = [self.reports[0], deepcopy(self.reports[0])]
        with self.assertRaises(ValueError):
            self.validate()

    def test_one_pass_never_admits_a_case(self):
        self.reports.pop()
        with self.assertRaises(ValueError):
            self.validate()

    def test_author_cannot_supply_an_independent_review(self):
        self.record["provenance"]["author_id"] = "synthetic-reviewer:a"
        self.digest, packet = synthetic_review_packet([self.record])
        self.packets = {self.digest: packet}
        self.reports = [synthetic_review(self.record, self.digest, name)
                        for name in ("synthetic-reviewer:a", "synthetic-reviewer:b")]
        with self.assertRaises(ValueError):
            self.validate()

    def test_revised_case_cannot_reuse_stale_passes(self):
        self.record["reference_answer"] = "파란색"
        with self.assertRaises(ValueError):
            self.validate()

    def test_stale_conflicting_reviews_are_ignored_after_two_current_passes(self):
        self.reports[0]["judgments"][0]["verdict"] = "hold"
        self.reports.append(synthetic_review(self.record, self.digest, "synthetic-reviewer:a", "pass"))
        self.record["reference_answer"] = "파란색"
        digest, packet = synthetic_review_packet([self.record])
        self.packets[digest] = packet
        self.reports += [synthetic_review(self.record, digest, name)
                         for name in ("synthetic-reviewer:c", "synthetic-reviewer:d")]
        resolved = self.validate()["cases"][0]
        self.assertEqual(resolved["case_sha256"], canonical_sha(self.record))
        self.assertEqual(resolved["reviewer_ids"], ["synthetic-reviewer:c", "synthetic-reviewer:d"])

    def test_full_old_packet_and_changed_only_packet_reuse_only_exact_case_reviews(self):
        old_records = [deepcopy(self.record), synthetic_record("synthetic-unchanged-case")]
        old_digest, old_packet = synthetic_review_packet(old_records)
        reports = []
        old_reviewers = ("synthetic-reviewer:old-a", "synthetic-reviewer:old-b")
        new_reviewers = ("synthetic-reviewer:new-a", "synthetic-reviewer:new-b")
        for reviewer in old_reviewers:
            report = synthetic_review(old_records[0], old_digest, reviewer)
            report["judgments"] += synthetic_review(old_records[1], old_digest, reviewer)["judgments"]
            reports.append(report)
        records = deepcopy(old_records)
        records[0]["reference_answer"] = "파란색"
        new_digest, new_packet = synthetic_review_packet([records[0]])
        reports += [synthetic_review(records[0], new_digest, reviewer) for reviewer in new_reviewers]
        resolution = prep.validate_reviews(records, {old_digest: old_packet, new_digest: new_packet}, reports)
        cases = {case["id"]: case for case in resolution["cases"]}
        self.assertEqual(set(cases), {record["id"] for record in records})
        self.assertEqual(cases[records[0]["id"]]["reviewer_ids"], list(new_reviewers))
        self.assertEqual(cases[records[0]["id"]]["case_sha256"], canonical_sha(records[0]))
        self.assertEqual(cases[records[1]["id"]]["reviewer_ids"], list(old_reviewers))
        self.assertEqual(cases[records[1]["id"]]["case_sha256"], canonical_sha(old_records[1]))
        self.assertEqual(resolution["ignored_stale_count"], 2)
        self.assertCountEqual(resolution["ignored_stale_judgments"], [
            {"id": old_records[0]["id"], "case_sha256": canonical_sha(old_records[0]),
             "reviewer_id": reviewer, "packet_sha256": old_digest, "reason": "superseded_case_hash"}
            for reviewer in old_reviewers
        ])

    def test_reviews_for_cases_outside_current_dataset_are_explicitly_reported(self):
        removed = synthetic_record("synthetic-not-in-current-dataset")
        digest, packet = synthetic_review_packet([self.record, removed])
        reviewers = ("synthetic-reviewer:a", "synthetic-reviewer:b")
        reports = []
        for reviewer in reviewers:
            report = synthetic_review(self.record, digest, reviewer)
            report["judgments"] += synthetic_review(removed, digest, reviewer)["judgments"]
            reports.append(report)
        resolution = prep.validate_reviews([self.record], {digest: packet}, reports)
        self.assertEqual(len(resolution["cases"]), 1)
        self.assertEqual(resolution["cases"][0]["id"], self.record["id"])
        self.assertEqual(resolution["ignored_stale_count"], 2)
        self.assertCountEqual(resolution["ignored_stale_judgments"], [
            {"id": removed["id"], "case_sha256": canonical_sha(removed), "reviewer_id": reviewer,
             "packet_sha256": digest, "reason": "case_not_in_current_dataset"}
            for reviewer in reviewers
        ])

    def test_one_current_pass_cannot_be_combined_with_a_stale_pass(self):
        self.record["rubric"]["accepted_variants"].append("파란색")
        digest, packet = synthetic_review_packet([self.record])
        self.packets[digest] = packet
        self.reports.append(synthetic_review(self.record, digest, "synthetic-reviewer:c"))
        with self.assertRaises(ValueError):
            self.validate()

    def test_insufficient_current_reviews_still_publish_stale_review_diagnostics(self):
        old_case_sha = canonical_sha(self.record)
        self.record["reference_answer"] = "파란색"
        current_digest, packet = synthetic_review_packet([self.record])
        self.packets[current_digest] = packet
        self.reports.append(synthetic_review(self.record, current_digest, "synthetic-reviewer:c"))
        diagnostics = {}
        with self.assertRaises(ValueError):
            prep.validate_reviews([self.record], self.packets, self.reports, diagnostics=diagnostics)
        self.assertEqual(diagnostics["current_case_count"], 1)
        self.assertEqual(diagnostics["current_case_judgment_count"], 1)
        self.assertEqual(diagnostics["ignored_stale_count"], 2)
        self.assertCountEqual(diagnostics["ignored_stale_judgments"], [
            {"id": self.record["id"], "case_sha256": old_case_sha, "reviewer_id": reviewer,
             "packet_sha256": self.digest, "reason": "superseded_case_hash"}
            for reviewer in ("synthetic-reviewer:a", "synthetic-reviewer:b")
        ])

    def test_fix_or_hold_blocks_admission_even_with_two_other_passes(self):
        for verdict in ("fix", "hold"):
            with self.subTest(verdict=verdict):
                reports = self.reports + [synthetic_review(self.record, self.digest, "synthetic-reviewer:c", verdict)]
                with self.assertRaises(ValueError):
                    prep.validate_reviews([self.record], self.packets, reports)

    def test_third_reviewer_must_explicitly_adjudicate_all_conflicting_reviewers(self):
        self.reports[1] = synthetic_review(self.record, self.digest, "synthetic-reviewer:b", "fix")
        self.reports.append(synthetic_review(
            self.record, self.digest, "synthetic-reviewer:c",
            adjudicates=["synthetic-reviewer:a", "synthetic-reviewer:b"],
        ))
        expected = self.validate()
        self.reports.reverse()
        self.assertEqual(self.validate(), expected)
        self.reports[0]["judgments"][0]["adjudicates"] = ["synthetic-reviewer:a"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_unknown_packet_and_wrong_judgment_hash_are_rejected(self):
        for field in ("packet_sha256", "case_sha256"):
            with self.subTest(field=field):
                reports = deepcopy(self.reports)
                target = reports[0] if field == "packet_sha256" else reports[0]["judgments"][0]
                target[field] = "0" * 64
                with self.assertRaises(ValueError):
                    prep.validate_reviews([self.record], self.packets, reports)

    def test_packet_record_tampering_and_duplicate_case_ids_are_rejected(self):
        for corruption in ("record", "duplicate"):
            with self.subTest(corruption=corruption):
                packets = deepcopy(self.packets)
                cases = packets[self.digest]["cases"]
                if corruption == "record":
                    cases[0]["record"]["reference_answer"] = "빨강"
                else:
                    cases.append(deepcopy(cases[0]))
                with self.assertRaises(ValueError):
                    prep.validate_reviews([self.record], packets, self.reports)

    def test_judgments_need_substantive_reason_and_valid_verdict(self):
        for field, value in (("reason", ""), ("verdict", "approved"), ("issues", "not a list")):
            with self.subTest(field=field):
                reports = deepcopy(self.reports)
                reports[0]["judgments"][0][field] = value
                with self.assertRaises(ValueError):
                    prep.validate_reviews([self.record], self.packets, reports)
        del self.reports[0]["judgments"][0]["suggested_fix"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_judgment_in_one_report_is_rejected(self):
        self.reports[0]["judgments"].append(deepcopy(self.reports[0]["judgments"][0]))
        with self.assertRaises(ValueError):
            self.validate()

    def test_same_reviewer_cannot_submit_conflicting_judgments(self):
        self.reports.append(synthetic_review(self.record, self.digest, "synthetic-reviewer:a", "hold"))
        with self.assertRaises(ValueError):
            self.validate()


def synthetic_source_binding():
    quote = "제1조 가상 파랑 표식은 사전 승인을 받은 경우에만 사용한다."
    prefix, suffix = "가상 시험용 머리말\n\n", "\n\n가상 시험용 꼬리말"
    body = prefix + quote + suffix
    passage = {
        "article_id": "synthetic-article:1", "raw_body_node_ordinal": 1,
        "char_start": len(prefix), "char_end": len(prefix) + len(quote), "quote_sha256": text_sha(quote),
    }
    family = {
        "family_id": "synthetic-source:train", "doc_id": "synthetic-document:1", "version": "synthetic-v1",
        "body_sha256": text_sha(body), "passage_pins": [deepcopy(passage)],
        "title": "Synthetic source, not an actual administrative rule",
        "source_url": "https://example.invalid/synthetic-only", "issuer": "Synthetic fixture",
        "effective_date": "20260101",
    }
    packet = {key: family[key] for key in ("family_id", "doc_id", "version", "body_sha256")}
    packet["articles"] = [{**deepcopy(passage), "text": quote}]
    record = synthetic_record("synthetic-bound-source", cohort="mois", behavior="conditions")
    record["family_id"] = family["family_id"]
    record["messages"].insert(0, {"role": "system", "content": "가상 제시문만 사용하세요."})
    record["provenance"].update({
        key: family[key] for key in ("family_id", "doc_id", "version", "body_sha256")
    })
    record["provenance"]["passage_pins"] = [deepcopy(passage)]
    record["provenance"]["institution"] = family["issuer"]
    record["provenance"]["evidence_reply"] = {
        "status": "quoted",
        "scope": "제공된 수집본의 선택 원문만; 현행성은 재확인하지 않음",
        "warnings": ["다른 조항·외부 참조 내용은 제공되지 않은 한 확인하지 않음"],
        "graph": {"sources": [{
            "doc_id": family["doc_id"], "title": family["title"],
            "source_url": family["source_url"], "issuer": family["issuer"],
            "institution": family["issuer"], "effective_date": "2026-01-01",
            "currency": "OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED",
            "scope": "selected official numbered body articles only",
        }], "paths": []},
        "candidates": [{
            "doc_id": family["doc_id"], "passage_id": passage["article_id"], "text": quote,
            "prediction": {"answer": quote, "abstains": False, "source_character_span": [0, len(quote)]},
        }],
    }
    return record, family, packet, body


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.record, self.family, self.packet, self.body = synthetic_source_binding()

    def validate(self):
        return prep.validate_source_binding(self.record, self.family, self.packet, self.body)

    def test_exact_versioned_span_and_production_evidence_are_accepted_without_mutation(self):
        before = deepcopy((self.record, self.family, self.packet))
        self.assertEqual(self.validate(), [self.packet["articles"][0]["text"]])
        self.assertEqual((self.record, self.family, self.packet), before)

    def test_source_family_document_version_and_body_hash_must_match(self):
        for target in ("provenance", "packet"):
            for key in ("family_id", "doc_id", "version", "body_sha256"):
                with self.subTest(target=target, key=key):
                    record, family, packet, body = synthetic_source_binding()
                    destination = record["provenance"] if target == "provenance" else packet
                    destination[key] = "synthetic-other"
                    with self.assertRaises(ValueError):
                        prep.validate_source_binding(record, family, packet, body)

    def test_changed_source_body_is_rejected_even_if_selected_quote_is_unchanged(self):
        self.body += " tampered suffix"
        with self.assertRaises(ValueError):
            self.validate()

    def test_empty_missing_malformed_and_duplicate_passage_pins_are_rejected(self):
        pin = self.record["provenance"]["passage_pins"][0]
        for pins in (None, [], ["not a pin"], [{}], [pin, deepcopy(pin)]):
            with self.subTest(pins=pins):
                record = deepcopy(self.record)
                record["provenance"]["passage_pins"] = pins
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(record, self.family, self.packet, self.body)

    def test_each_pin_field_is_required_and_must_match_allocation(self):
        for key in prep.PIN_FIELDS:
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    record = deepcopy(self.record)
                    pin = record["provenance"]["passage_pins"][0]
                    if missing:
                        del pin[key]
                    else:
                        pin[key] = "not the allocated value"
                    with self.assertRaises(ValueError):
                        prep.validate_source_binding(record, self.family, self.packet, self.body)

    def test_source_packet_must_contain_one_unambiguous_matching_article(self):
        for articles in ([], self.packet["articles"] * 2):
            with self.subTest(articles=articles):
                packet = deepcopy(self.packet)
                packet["articles"] = articles
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(self.record, self.family, packet, self.body)

    def test_span_offsets_must_be_integer_nonempty_and_inside_body(self):
        for start, end in ((-1, 2), (2, 2), (3, 2), (0, len(self.body) + 1),
                           (True, 2), (0, False), (0.0, 2), (0, "2")):
            with self.subTest(start=start, end=end):
                record, family, packet, body = synthetic_source_binding()
                for pin in (record["provenance"]["passage_pins"][0], family["passage_pins"][0],
                            packet["articles"][0]):
                    pin.update(char_start=start, char_end=end)
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(record, family, packet, body)

    def test_matching_pin_copies_cannot_forge_quote_hash_or_packet_text(self):
        for corruption in ("hash", "text", "offset"):
            with self.subTest(corruption=corruption):
                record, family, packet, body = synthetic_source_binding()
                if corruption == "text":
                    packet["articles"][0]["text"] += " invented exception"
                else:
                    for pin in (record["provenance"]["passage_pins"][0], family["passage_pins"][0],
                                packet["articles"][0]):
                        if corruption == "hash":
                            pin["quote_sha256"] = "0" * 64
                        else:
                            pin["char_start"] += 1
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(record, family, packet, body)

    def test_evidence_reply_is_required_and_unbound_graph_facts_are_forbidden(self):
        for evidence in (None, {}, {"graph": {"paths": [{"synthetic_unbound_fact": True}]}}):
            with self.subTest(evidence=evidence):
                record = deepcopy(self.record)
                record["provenance"]["evidence_reply"] = evidence
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(record, self.family, self.packet, self.body)

    def test_evidence_source_identity_and_metadata_must_match(self):
        for key in ("doc_id", "title", "source_url", "issuer", "institution", "effective_date", "currency"):
            with self.subTest(key=key):
                record = deepcopy(self.record)
                record["provenance"]["evidence_reply"]["graph"]["sources"][0][key] = "synthetic-other"
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(record, self.family, self.packet, self.body)

    def test_provenance_institution_must_match_allocated_source_issuer(self):
        self.record["provenance"]["institution"] = "Synthetic other institution"
        with self.assertRaises(ValueError):
            self.validate()

    def test_evidence_cannot_add_drop_duplicate_or_rewrite_selected_passages(self):
        for corruption in ("missing", "duplicate", "doc_id", "text", "answer", "span", "abstains"):
            with self.subTest(corruption=corruption):
                record = deepcopy(self.record)
                candidates = record["provenance"]["evidence_reply"]["candidates"]
                candidate = candidates[0]
                if corruption == "missing":
                    candidates.clear()
                elif corruption == "duplicate":
                    candidates.append(deepcopy(candidate))
                elif corruption == "doc_id":
                    candidate["doc_id"] = "synthetic-other"
                elif corruption == "text":
                    candidate["text"] += " invented rule"
                elif corruption == "answer":
                    candidate["prediction"]["answer"] += " invented exception"
                elif corruption == "span":
                    candidate["prediction"]["source_character_span"][0] = 1
                else:
                    candidate["prediction"]["abstains"] = True
                with self.assertRaises(ValueError):
                    prep.validate_source_binding(record, self.family, self.packet, self.body)


class GroundedMessagesLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tok = SyntheticCharacterTokenizer()

    def load(self, messages):
        row = {
            "id": "synthetic-native-multiturn", "task": "synthetic_general_context", "split": "train",
            "messages": messages, "answer": "파랑",
        }
        payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        path = self.root / "synthetic-native.jsonl"
        path.write_bytes(payload)
        entry = {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                 "adapter": "grounded_messages"}
        return trainer.load_rows(entry, self.root)

    def test_native_single_turn_and_history_with_optional_system_are_preserved_exactly(self):
        history = [
            {"role": "user", "content": "표식은 빨강입니다."},
            {"role": "assistant", "content": "빨강으로 기록했습니다."},
            {"role": "user", "content": "파랑으로 정정합니다. 마지막 표식만 답하세요."},
        ]
        for with_system in (False, True):
            for with_history in (False, True):
                with self.subTest(with_system=with_system, with_history=with_history):
                    messages = deepcopy(history if with_history else history[-1:])
                    if with_system:
                        messages.insert(0, {"role": "system", "content": "가상 대화만 사용하세요."})
                    loaded, = self.load(messages)
                    self.assertEqual(trainer.messages_of(loaded), messages)
                    self.assertEqual(loaded["answer"], "파랑")
                    self.assertEqual(loaded["adapter"], "grounded_messages")
                    prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                    prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
                    answer_ids = self.tok("파랑" + self.tok.eos_token, add_special_tokens=False)["input_ids"]
                    encoded = trainer.encode(self.tok, loaded, "qwen3", len(prompt_ids) + len(answer_ids))
                    self.assertEqual(encoded["ids"], prompt_ids + answer_ids)
                    self.assertEqual(encoded["labels"], [-100] * len(prompt_ids) + answer_ids)
                    self.assertEqual(encoded["labels"][-1], self.tok.eos_token_id)

    def test_invalid_alternation_current_assistant_and_empty_text_are_rejected(self):
        user = {"role": "user", "content": "가상 질문"}
        assistant = {"role": "assistant", "content": "가상 과거 응답"}
        system = {"role": "system", "content": "가상 지시"}
        variants = [
            [], [system], [assistant], [user, assistant], [user, user],
            [system, user, user], [user, system, user], [user, assistant, assistant, user],
            [{"role": "user", "content": ""}], [{"role": "user", "content": "   "}],
            [{"role": "user", "content": None}], [{"role": "tool", "content": "unexpected"}],
            [{"role": "user", "content": "가상 질문", "reference_answer": "leak"}],
        ]
        for messages in variants:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.load(messages)


class EvaluationReserveTests(unittest.TestCase):
    def test_exact_full_generation_reserve_is_accepted(self):
        measurement = {"prompt_tokens": 3328, "answer_tokens_including_eos": 3, "total_tokens": 3331}
        prep.audit_evaluation_reserve(measurement, 4096, 768)

    def test_short_reference_cannot_hide_one_token_generation_reserve_overflow(self):
        measurement = {"prompt_tokens": 3329, "answer_tokens_including_eos": 3, "total_tokens": 3332}
        self.assertLess(measurement["total_tokens"], 4096)
        with self.assertRaises(ValueError):
            prep.audit_evaluation_reserve(measurement, 4096, 768)

    def test_generation_reserve_does_not_double_count_reference_answer(self):
        measurement = {"prompt_tokens": 3328, "answer_tokens_including_eos": 768, "total_tokens": 4096}
        prep.audit_evaluation_reserve(measurement, 4096, 768)


class TrainerRowTests(unittest.TestCase):
    def test_both_cohorts_use_verbatim_native_messages_without_private_metadata(self):
        for cohort in ("general", "mois"):
            with self.subTest(cohort=cohort):
                record = synthetic_record(cohort=cohort)
                record["messages"].insert(0, {"role": "system", "content": "가상 사용자 지정 지시"})
                before = deepcopy(record)
                row = prep.trainer_row(record)
                self.assertEqual(row["messages"], record["messages"])
                self.assertEqual(row["answer"], record["reference_answer"])
                self.assertEqual(row["id"], record["id"])
                self.assertEqual(row["split"], record["split"])
                self.assertEqual(set(row), {"id", "task", "split", "messages", "answer"})
                self.assertEqual(record, before)


class SourceReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        evidence = self.root / "synthetic-evidence.json"
        evidence.write_text('{"synthetic_unit_test_only":true}\n', encoding="utf-8")
        self.evidence_path = evidence
        self.evidence_pin = prep.pin(evidence, self.root)
        # Exercise real byte/hash verification inside an isolated temporary root.
        # Only the default repository root is redirected; verification is not bypassed.
        original_verified = prep.verified
        patcher = patch.object(prep, "verified", side_effect=lambda value: original_verified(value, self.root))
        self.verified = patcher.start()
        self.addCleanup(patcher.stop)
        source_record, _, _, _ = synthetic_source_binding()
        self.records = [source_record, synthetic_record("synthetic-general-receipt")]
        self.allocation_pin = {
            "path": "synthetic-allocation.json", "bytes": 17,
            "sha256": text_sha("synthetic allocation"),
        }
        self.source_inputs = [deepcopy(self.evidence_pin)]
        self.overlap_sha = text_sha("synthetic authored overlap screen")
        decision = {
            "verdict": "pass", "reviewer_id": "synthetic-source-reviewer",
            "reason": "Isolated unit-test decision, not an actual source clearance.",
            "evidence_pins": [deepcopy(self.evidence_pin)],
        }
        self.receipt = {
            "schema": "trid-expanded-source-admission-v1",
            "source_allocation": deepcopy(self.allocation_pin),
            "source_inputs": deepcopy(self.source_inputs),
            "authored_overlap_sha256": self.overlap_sha,
            "case_hashes": {record["id"]: canonical_sha(record) for record in self.records},
            "protected_case_review": deepcopy(decision),
            "historically_unseen": False,
            "checkpoint_consumption": "UNKNOWN_NOT_CERTIFIED",
            "currentness": "OFFICIAL_COLLECTED_SNAPSHOT_NOT_RECHECKED",
            "families": [{
                "family_id": source_record["family_id"],
                **{gate: deepcopy(decision) for gate in prep.SOURCE_GATES},
            }],
        }

    def validate(self):
        return prep.source_receipt(
            self.receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
        )

    def test_explicit_unknown_history_and_snapshot_currentness_are_accepted(self):
        before = deepcopy(self.receipt)
        self.assertEqual(self.validate(), before)
        self.assertEqual(self.receipt, before)
        self.assertEqual(self.verified.call_count, 1 + len(prep.SOURCE_GATES))
        for call in self.verified.call_args_list:
            self.assertEqual(call.args, (self.evidence_pin,))

    def test_absent_or_wrong_schema_receipt_leaves_candidates_unadmitted(self):
        for receipt in (None, {}, {**self.receipt, "schema": "not-a-source-admission"}):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                prep.source_receipt(
                    receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                )

    def test_allocation_source_input_and_current_overlap_pins_must_be_exact(self):
        for field in ("source_allocation", "source_inputs", "authored_overlap_sha256"):
            with self.subTest(field=field):
                receipt = deepcopy(self.receipt)
                if field == "source_allocation":
                    receipt[field]["sha256"] = "0" * 64
                elif field == "source_inputs":
                    receipt[field][0]["bytes"] += 1
                else:
                    receipt[field] = "0" * 64
                with self.assertRaises(ValueError):
                    prep.source_receipt(
                        receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                    )

    def test_every_current_case_hash_including_general_is_required(self):
        for record in self.records:
            for corruption in ("missing", "stale"):
                with self.subTest(cohort=record["cohort"], corruption=corruption):
                    receipt = deepcopy(self.receipt)
                    if corruption == "missing":
                        del receipt["case_hashes"][record["id"]]
                    else:
                        receipt["case_hashes"][record["id"]] = "0" * 64
                    with self.assertRaises(ValueError):
                        prep.source_receipt(
                            receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                        )
        self.receipt["case_hashes"]["synthetic-unregistered-case"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_general_case_invalidates_previously_matching_receipt(self):
        self.records[1]["rubric"]["required"].append("새로 추가한 가상 조건")
        with self.assertRaises(ValueError):
            self.validate()

    def test_protected_case_review_requires_pass_reviewer_reason_and_evidence(self):
        for field, value in (("verdict", "pending"), ("verdict", "hold"), ("reviewer_id", ""),
                             ("reason", " "), ("evidence_pins", [])):
            with self.subTest(field=field, value=value):
                receipt = deepcopy(self.receipt)
                receipt["protected_case_review"][field] = value
                with self.assertRaises(ValueError):
                    prep.source_receipt(
                        receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                    )
        del self.receipt["protected_case_review"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_protected_review_is_still_mandatory_for_general_only_records(self):
        records = [self.records[1]]
        receipt = deepcopy(self.receipt)
        receipt["case_hashes"] = {records[0]["id"]: canonical_sha(records[0])}
        receipt["families"] = []
        self.assertEqual(
            prep.source_receipt(receipt, self.allocation_pin, records, self.source_inputs, self.overlap_sha),
            receipt,
        )
        del receipt["protected_case_review"]
        with self.assertRaises(ValueError):
            prep.source_receipt(receipt, self.allocation_pin, records, self.source_inputs, self.overlap_sha)

    def test_each_family_gate_refuses_pending_or_incomplete_decisions(self):
        for gate in prep.SOURCE_GATES:
            for field, value in (("verdict", "pending"), ("verdict", "hold"), ("reviewer_id", ""),
                                 ("reason", ""), ("evidence_pins", [])):
                with self.subTest(gate=gate, field=field, value=value):
                    receipt = deepcopy(self.receipt)
                    receipt["families"][0][gate][field] = value
                    with self.assertRaises(ValueError):
                        prep.source_receipt(
                            receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                        )

    def test_source_family_coverage_cannot_be_partial_duplicate_or_extra(self):
        family = self.receipt["families"][0]
        for families in ([], [family, deepcopy(family)],
                         [{**deepcopy(family), "family_id": "synthetic-unregistered-source"}]):
            with self.subTest(families=families):
                receipt = deepcopy(self.receipt)
                receipt["families"] = families
                with self.assertRaises(ValueError):
                    prep.source_receipt(
                        receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                    )

    def test_unseen_and_certified_currentness_claims_are_not_admitted(self):
        for field, value in (("historically_unseen", True), ("historically_unseen", None),
                             ("checkpoint_consumption", "NOT_CONSUMED_CERTIFIED"),
                             ("currentness", "CURRENT_CERTIFIED")):
            with self.subTest(field=field, value=value):
                receipt = deepcopy(self.receipt)
                receipt[field] = value
                with self.assertRaises(ValueError):
                    prep.source_receipt(
                        receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                    )

    def test_protected_and_family_evidence_pins_are_really_verified(self):
        for location in ("protected_case_review", *prep.SOURCE_GATES):
            with self.subTest(location=location):
                receipt = deepcopy(self.receipt)
                decision = (receipt[location] if location == "protected_case_review"
                            else receipt["families"][0][location])
                decision["evidence_pins"][0]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    prep.source_receipt(
                        receipt, self.allocation_pin, self.records, self.source_inputs, self.overlap_sha,
                    )

    def test_evidence_file_changed_after_pin_is_rejected(self):
        self.evidence_path.write_text('{"synthetic_unit_test_only":false}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()


class ImmutableTokenizerMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "synthetic-model-layout"
        self.directory.mkdir()
        self.metadata_names = (
            "LICENSE", "config.json", "generation_config.json", "tokenizer_config.json",
            "tokenizer.json", "model.safetensors.index.json",
        )
        self.weight_names = tuple(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6))
        files = []
        for name in self.metadata_names:
            payload = json.dumps({"synthetic_unit_test_only": True, "filename": name}).encode("utf-8")
            (self.directory / name).write_bytes(payload)
            files.append({
                "path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "body_downloaded": True,
            })
        for name in self.weight_names:
            (self.directory / name).write_bytes(b"Synthetic opaque bytes; not model weights.")
            files.append({"path": name, "bytes": 999999, "body_downloaded": False})
        self.acquisition = {
            "schema": "trid-immutable-model-metadata-v1",
            "repo_id": "synthetic/unit-test-model", "revision": "synthetic-immutable-revision",
            "files": files,
        }
        self.design = {
            "model": {"repo_id": self.acquisition["repo_id"], "revision": self.acquisition["revision"]},
        }
        self.acquisition_path = self.root / "synthetic-original-acquisition.json"
        self.save_acquisition()
        original_pin = prep.pin
        patcher = patch.object(prep, "pin", side_effect=lambda path: original_pin(path, self.root))
        self.pin_spy = patcher.start()
        self.addCleanup(patcher.stop)

    def save_acquisition(self):
        payload = json.dumps(self.acquisition, ensure_ascii=False).encode("utf-8")
        self.acquisition_path.write_bytes(payload)
        self.acquisition_sha = hashlib.sha256(payload).hexdigest()

    def validate(self):
        return prep.verify_tokenizer_metadata(
            self.directory, self.acquisition_path, self.acquisition_sha, self.design,
        )

    def test_original_eleven_file_layout_accepts_native_metadata_without_merges_or_vocab(self):
        self.assertFalse((self.directory / "merges.txt").exists())
        self.assertFalse((self.directory / "vocab.json").exists())
        pins = self.validate()
        self.assertEqual({Path(pin["path"]).name for pin in pins}, set(self.metadata_names))
        self.assertEqual(len(pins), len(self.metadata_names))
        self.assertEqual(self.pin_spy.call_count, len(self.metadata_names))

    def test_tokenizer_admission_never_opens_or_hashes_weight_bodies(self):
        original_read_bytes = Path.read_bytes
        original_file_sha = prep.file_sha
        read_names = []

        def guarded_read_bytes(path):
            self.assertNotIn(path.name, self.weight_names, "Tokenizer admission opened a weight body.")
            read_names.append(path.name)
            return original_read_bytes(path)

        def guarded_file_sha(path):
            self.assertNotIn(Path(path).name, self.weight_names, "Tokenizer admission hashed a weight body.")
            return original_file_sha(path)

        with patch.object(Path, "read_bytes", guarded_read_bytes), patch.object(prep, "file_sha", guarded_file_sha):
            pins = self.validate()
        self.assertEqual(set(read_names), set(self.metadata_names))
        self.assertEqual(len(pins), len(self.metadata_names))

    def test_fresh_api_inventory_cannot_replace_original_acquisition_schema(self):
        self.acquisition["schema"] = "synthetic-fresh-api-inventory-v1"
        self.save_acquisition()
        with self.assertRaises(ValueError):
            self.validate()

    def test_acquisition_bytes_and_registered_model_revision_are_pinned(self):
        self.acquisition_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()
        self.save_acquisition()
        for key in ("repo_id", "revision"):
            with self.subTest(key=key):
                design = deepcopy(self.design)
                design["model"][key] = "synthetic-other"
                with self.assertRaises(ValueError):
                    prep.verify_tokenizer_metadata(
                        self.directory, self.acquisition_path, self.acquisition_sha, design,
                    )

    def test_required_native_metadata_cannot_be_missing_from_original_inventory(self):
        for name in ("LICENSE", "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"):
            with self.subTest(name=name):
                original = deepcopy(self.acquisition)
                self.acquisition["files"] = [row for row in self.acquisition["files"] if row["path"] != name]
                self.save_acquisition()
                with self.assertRaises(ValueError):
                    self.validate()
                self.acquisition = original
        self.save_acquisition()

    def test_native_metadata_must_have_verified_downloaded_body(self):
        for name in ("LICENSE", "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"):
            with self.subTest(name=name):
                original = deepcopy(self.acquisition)
                row = next(row for row in self.acquisition["files"] if row["path"] == name)
                row["body_downloaded"] = False
                self.save_acquisition()
                with self.assertRaises(ValueError):
                    self.validate()
                self.acquisition = original
        self.save_acquisition()

    def test_metadata_requires_original_sha256_not_only_an_api_blob_identifier(self):
        row = next(row for row in self.acquisition["files"] if row["path"] == "tokenizer_config.json")
        del row["sha256"]
        row["blob_id"] = "a" * 40
        self.save_acquisition()
        with self.assertRaises(ValueError):
            self.validate()

    def test_metadata_content_changes_are_rejected_even_at_same_size(self):
        path = self.directory / "tokenizer.json"
        payload = path.read_bytes()
        path.write_bytes(b"!" + payload[1:])
        with self.assertRaises(ValueError):
            self.validate()

    def test_metadata_size_changes_are_rejected(self):
        path = self.directory / "tokenizer_config.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_directory_entry_is_rejected_even_for_unread_weights(self):
        (self.directory / self.weight_names[0]).unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_unpinned_external_chat_template_cannot_override_native_template(self):
        (self.directory / "chat_template.jinja").write_text("Synthetic unexpected template", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_acquisition_file_names_are_rejected(self):
        self.acquisition["files"].append(deepcopy(self.acquisition["files"][0]))
        self.save_acquisition()
        with self.assertRaises(ValueError):
            self.validate()


class CompleteCaseAuditTests(unittest.TestCase):
    def setUp(self):
        self.records = [synthetic_record(f"synthetic-audit-{index}") for index in range(5)]

    def test_multiple_bad_cases_do_not_stop_later_case_callbacks(self):
        visited, blockers = [], []

        def check(record):
            visited.append(record["id"])
            if record["id"] == "synthetic-audit-1":
                raise ValueError("synthetic invalid source")
            if record["id"] == "synthetic-audit-3":
                raise TypeError("synthetic malformed field")
            return {"synthetic_measurement": record["id"]}

        audits = prep.audit_cases(self.records, "synthetic-source-gate", check, blockers)
        self.assertEqual(visited, [record["id"] for record in self.records])
        self.assertEqual(len(audits), len(self.records))
        self.assertEqual([audit["record_index"] for audit in audits], list(range(len(self.records))))
        self.assertEqual([audit["id"] for audit in audits], visited)
        self.assertEqual([audit["status"] for audit in audits], ["pass", "blocked", "pass", "blocked", "pass"])
        for index in (0, 2, 4):
            self.assertEqual(audits[index]["result"], {"synthetic_measurement": self.records[index]["id"]})
        self.assertEqual([(blocker["id"], blocker["record_index"]) for blocker in blockers],
                         [("synthetic-audit-1", 1), ("synthetic-audit-3", 3)])
        self.assertEqual([blocker["gate"] for blocker in blockers], ["synthetic-source-gate"] * 2)
        for index, reason in ((1, "synthetic invalid source"), (3, "synthetic malformed field")):
            self.assertIn(reason, audits[index]["reason"])
            blocker = next(blocker for blocker in blockers if blocker["record_index"] == index)
            self.assertIn(reason, blocker["reason"])
        # Successful siblings leave the admission blockers intact: no partial admission.
        self.assertTrue(blockers)

    def test_first_and_last_failures_still_preserve_full_denominator(self):
        visited, blockers = [], []

        def check(record):
            visited.append(record["id"])
            if record["id"] in ("synthetic-audit-0", "synthetic-audit-4"):
                raise ValueError("synthetic boundary-case failure")
            return None

        audits = prep.audit_cases(self.records, "synthetic-token-gate", check, blockers)
        self.assertEqual(visited, [record["id"] for record in self.records])
        self.assertEqual(len(audits), 5)
        self.assertEqual([blocker["record_index"] for blocker in blockers], [0, 4])
        self.assertEqual(sum(audit["status"] == "blocked" for audit in audits), 2)
        self.assertEqual(sum(audit["status"] == "pass" for audit in audits), 3)

    def test_successful_gate_does_not_clear_existing_admission_blockers(self):
        existing = {"gate": "synthetic-earlier-gate", "id": "synthetic-prior-case",
                    "record_index": 0, "reason": "synthetic earlier failure"}
        blockers = [deepcopy(existing)]
        audits = prep.audit_cases(self.records, "synthetic-later-gate", lambda record: "checked", blockers)
        self.assertEqual(blockers, [existing])
        self.assertEqual(len(audits), len(self.records))
        self.assertTrue(all(audit["status"] == "pass" for audit in audits))

    def test_repeated_gates_accumulate_attributed_failures_without_dropping_cases(self):
        blockers = []

        def reject(record):
            raise ValueError("synthetic independently blocked case")

        for gate in ("synthetic-record-gate", "synthetic-production-gate"):
            audits = prep.audit_cases(self.records, gate, reject, blockers)
            self.assertEqual(len(audits), len(self.records))
            self.assertTrue(all(audit["status"] == "blocked" for audit in audits))
        self.assertEqual(len(blockers), 2 * len(self.records))
        for gate in ("synthetic-record-gate", "synthetic-production-gate"):
            selected = [blocker for blocker in blockers if blocker["gate"] == gate]
            self.assertEqual([blocker["record_index"] for blocker in selected], list(range(len(self.records))))
            self.assertEqual([blocker["id"] for blocker in selected], [record["id"] for record in self.records])


class SymlinkRootPinTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.actual_root = self.base / "actual-root"
        self.actual_root.mkdir()
        self.alias_root = self.base / "alias-root"
        self.alias_root.symlink_to(self.actual_root, target_is_directory=True)
        self.payload = b"Synthetic symlink-root pin fixture.\n"
        self.actual_file = self.actual_root / "synthetic-source.json"
        self.actual_file.write_bytes(self.payload)

    def test_symlink_root_pin_round_trip_preserves_relative_identity(self):
        expected = {
            "path": self.actual_file.name, "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        for source in (Path(self.actual_file.name), self.actual_file, self.alias_root / self.actual_file.name):
            with self.subTest(source=source):
                pinned = prep.pin(source, self.alias_root)
                self.assertEqual(pinned, expected)
                self.assertTrue(prep.verified(pinned, self.alias_root).samefile(self.actual_file))
                self.assertTrue(prep.verified(pinned, self.actual_root).samefile(self.actual_file))

    def test_symlink_root_does_not_allow_a_child_symlink_to_escape_containment(self):
        outside = self.base / "synthetic-outside.json"
        outside.write_bytes(self.payload)
        escaped = self.actual_root / "escape.json"
        escaped.symlink_to(outside)
        with self.assertRaises(ValueError):
            prep.pin(self.alias_root / escaped.name, self.alias_root)
        with self.assertRaises(ValueError):
            prep.verified({
                "path": escaped.name, "bytes": len(self.payload),
                "sha256": hashlib.sha256(self.payload).hexdigest(),
            }, self.alias_root)


if __name__ == "__main__":
    unittest.main()
