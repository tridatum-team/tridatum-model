"""Consumer-visible graph answers, exercised against a real immutable RDF store."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import pyoxigraph
except ModuleNotFoundError as error:
    if error.name != "pyoxigraph":
        raise
    raise unittest.SkipTest("Install the graph extra to run persisted graph tests") from error

from trid.service.mois_graph import GraphStore, build_graph
from trid.service.mois_graph_answers import graph_reply


CORE = "fixture:mois:travel"
REFERENCE = "fixture:reference:service"
OTHER = "fixture:mois:inventory"
TITLE = "행정안전부 공무국외출장 운영지침"


def make_release(root, plan_days=20):
    root = Path(root)
    index = root / "index"
    index.mkdir()
    paragraphs = [
        "제2조 이 지침은 행정안전부 소속 공무원과 행정안전부 예산으로 공무수행을 위해 국외출장하는 다른 사람에게 적용한다. 다른 법령 또는 장관이 특별히 정한 경우는 제외한다.",
        "제3조 장관이 허가한다. 직급별 허가권 위임범위는 행정안전부 위임전결규정에 따른다.",
        f"제4조 출장계획서는 출장일 {plan_days}일 전까지 총괄 부서장에게 제출한다. 계획서 제출 7일 전까지 총괄 부서장과 사전협의한다. 두 기한은 토요일과 공휴일을 제외한다.",
        "제8조 직무상 이해관계자의 경비 부담은 금지한다. 다만 필요성이 인정되고 심사위원회 심사를 거쳐 장관이 허가한 경우는 예외로 한다.",
        "제11조 결과보고서는 귀국 후 30일 이내 제출한다. 출장 마친 날의 다음 날부터 기산하며 부서장 검토·결재 후 허가권자에게 제출하고 등록한다.",
        "제15조 다음 연도 연간계획은 매년 12월 20일까지 총괄 부서장에게 제출한다.",
        "제16조 그 밖의 사항은 국가공무원 복무규정을 참조한다.",
        "현지활동 연락망은 출국 전에 확인한다.",
        "별지 제7호서식 주석: 국가기밀보호 또는 보안유지가 필요한 경우 부서장이 구체적 사유를 밝혀 인사혁신처장에게 통보하면 시스템에 등록하지 않을 수 있다.",
        "미검증 표: 숙박비 999999원.",
    ]
    text = "\n".join(paragraphs)
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    passages = []
    start = 0
    for n, paragraph in enumerate(paragraphs):
        passages.append({"id": f"{CORE}:{n}", "doc_id": CORE, "text": paragraph, "char_start": start,
                         "char_end": start + len(paragraph), "path": [TITLE, f"구간 {n}"]})
        start += len(paragraph) + 1
    reference_text = "국가공무원은 복무규정의 적용 대상이다."
    documents = [
        {"doc_id": CORE, "title": TITLE, "sha256": content_hash, "acl": ["public"], "text": text, "passages": passages},
        {"doc_id": REFERENCE, "title": "국가공무원 복무규정", "sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
         "acl": ["public"], "text": reference_text, "passages": [
             {"id": REFERENCE + ":0", "doc_id": REFERENCE, "text": reference_text,
              "char_start": 0, "char_end": len(reference_text), "path": ["국가공무원 복무규정"]}]},
        {"doc_id": OTHER, "title": "행정안전부 자료목록", "sha256": hashlib.sha256(b"").hexdigest(),
         "acl": ["public"], "text": "", "passages": []},
    ]
    metadata = {
        CORE: {"title": TITLE, "institution": "행정안전부", "kind": "행정규칙", "date": "2020-03-20",
               "date_type": "발령일", "effective_date": "2020-03-20", "version": "fixture-version",
               "retrieved_at": "2026-01-01", "currency": "current_at_collection", "source_id": "fixture-source",
               "url": "https://example.test/travel"},
        REFERENCE: {"title": "국가공무원 복무규정", "institution": "인사혁신처", "kind": "법령"},
        OTHER: {"title": "행정안전부 자료목록", "institution": "행정안전부", "kind": "목록"},
    }
    facts = []

    def fact(key, n, subject, predicate, topics, qualifiers=None, rule_type="ProcedureRule", target=None):
        passage = passages[n]
        row = {"id": key, "doc_id": CORE, "content_hash": content_hash, "type": rule_type,
               "subject": subject, "predicate": predicate, "object": passage["text"], "topics": topics,
               "qualifiers": qualifiers or {}, "evidence": {
                   "passage_id": passage["id"], "start": passage["char_start"], "end": passage["char_end"],
                   "quote": passage["text"]}}
        if target:
            row["target_doc_id"] = target
        facts.append(row)

    fact("scope", 0, "적용범위", "appliesTo", ["적용", "대상", "제외", "예산"], rule_type="ApplicabilityRule")
    fact("permission", 1, "장관", "authorizes", ["허가", "허가권자"])
    fact("delegation", 1, "허가권 위임범위", "references", ["위임전결", "직급", "허가권"],
         {"resolution": "unresolved"}, rule_type="ReferenceRule")
    fact("plan", 2, "출장계획서 제출", "deadline", ["계획서", "제출", "기한"],
         {"trigger": "출장일", "direction": "before", "amount": plan_days, "excluded_days": ["토요일", "공휴일"]})
    facts[-1]["object"] = f"출장계획서는 출장일 {plan_days}일 전까지 총괄 부서장에게 제출한다. 토요일과 공휴일을 제외한다."
    fact("consult", 2, "사전협의", "deadline", ["사전협의", "출장계획서", "기한"],
         {"trigger": "계획서 제출", "direction": "before", "amount": 7, "excluded_days": ["토요일", "공휴일"]})
    facts[-1]["object"] = "계획서 제출 7일 전까지 총괄 부서장과 사전협의한다. 토요일과 공휴일을 제외한다."
    fact("expenses", 3, "이해관계자 경비", "prohibits", ["경비", "부담", "이해관계자", "예외"])
    fact("report", 4, "귀국 결과보고서", "deadline", ["귀국", "보고서", "제출", "기한"],
         {"trigger": "귀국", "direction": "after", "amount": 30, "count_start": "출장 마친 날의 다음 날"})
    fact("annual", 5, "연간계획", "deadline", ["연간계획", "제출", "기한"])
    fact("reference", 6, "그 밖의 사항", "references", ["참조규정"], rule_type="ReferenceRule", target=REFERENCE)
    fact("registration-exception", 8, "결과보고서 시스템 등록의 예외", "exception", ["보안유지", "국가기밀보호"],
         {"exception_to": "report", "article": "별지 제7호서식 주석"})
    for name, value in (("documents.json", documents), ("meta.json", metadata), ("tables.json", [passages[-1]["id"]])):
        (index / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    (index / "review.jsonl").write_text(json.dumps({"passage_id": passages[-1]["id"],
        "value_review": {"decision": "hold_for_value_use"}}) + "\n", encoding="utf-8")
    facts_path = root / "facts.jsonl"
    facts_path.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts), encoding="utf-8")
    release = root / "release"
    build_graph(index, release, facts_path)
    return GraphStore.open(release)


class GraphAnswersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.store = make_release(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.store = None
        cls.temp.cleanup()

    def ask(self, question, **kwargs):
        return graph_reply(self.store, question, **kwargs)

    def assertExactEvidence(self, answer):
        self.assertTrue(answer["candidates"])
        sources = {d["doc_id"] for d in answer["graph"]["sources"]}
        self.assertEqual(sources, {c["doc_id"] for c in answer["candidates"]})
        for number, candidate in enumerate(answer["candidates"], 1):
            start, end = candidate["prediction"]["source_character_span"]
            quote = candidate["text"][start:end]
            self.assertEqual(quote, candidate["prediction"]["answer"])
            passage = next(p for p in self.store.passages(candidate["doc_id"]) if p["id"] == candidate["passage_id"])
            self.assertEqual(candidate["page_text_start"], passage["char_start"])
            self.assertIn(f"[{number}]", answer["answer"])
            self.assertTrue(any(p.get("evidence", {}).get("quote") == quote for p in answer["graph"]["paths"]))

    def test_metadata_dates_are_not_body_quotes_or_current_validity(self):
        answer = self.ask(TITLE + " 발령일과 시행일")
        self.assertEqual(answer["graph"]["answer_kind"], "metadata")
        self.assertIn("2020-03-20", answer["answer"])
        self.assertIn("발령일", answer["answer"])
        self.assertIn("발령일: 2020-03-20", answer["answer"])
        self.assertNotIn("자료 기준일", answer["answer"])
        self.assertIn("보증하지 않습니다", answer["answer"])
        self.assertEqual(answer["candidates"], [])
        self.assertTrue(all(p["evidence_kind"] == "metadata" for p in answer["graph"]["paths"]))
        self.assertEqual(answer["graph"]["sources"][0]["version"], "fixture-version")

    def test_overview_contains_substantive_source_procedure(self):
        answer = self.ask("행안부 공무국외출장은 어떤 절차로 진행해?")
        self.assertEqual(answer["graph"]["answer_kind"], "overview")
        self.assertEqual({p.get("fact_id") for p in answer["graph"]["paths"]},
                         {"scope", "consult", "plan", "permission", "report", "registration-exception"})
        for word in ("사전협의", "장관", "결과보고서"):
            self.assertIn(word, answer["answer"])
        self.assertExactEvidence(answer)
        self.assertNotIn(OTHER, {d["doc_id"] for d in answer["graph"]["sources"]})
        self.assertNotIn(REFERENCE, {d["doc_id"] for d in answer["graph"]["sources"]})

    def test_two_deadlines_keep_distinct_triggers_and_calendar_exclusions(self):
        answer = self.ask("계획서 제출 기한과 사전협의 기한", pinned_doc=CORE)
        paths = {p["fact_id"]: p for p in answer["graph"]["paths"]}
        self.assertEqual(paths["plan"]["qualifiers"]["trigger"], "출장일")
        self.assertEqual(paths["consult"]["qualifiers"]["trigger"], "계획서 제출")
        self.assertIn("20일 전", answer["answer"])
        self.assertIn("7일 전", answer["answer"])
        self.assertIn("토요일과 공휴일을 제외", answer["answer"])
        self.assertExactEvidence(answer)

    def test_action_and_deadline_facets_do_not_pull_other_procedures(self):
        for question, expected in (
            ("행안부 국외출장 계획서는 언제까지 누구에게 제출해야 해?", {"plan"}),
            ("국외출장 계획서를 제출하기 전에 사전 협의는 언제 해야 해?", {"consult"}),
            ("계획서 제출 기한과 사전협의 기한을 둘 다 알려줘", {"plan", "consult"}),
        ):
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertEqual({p.get("fact_id") for p in answer["graph"]["paths"]}, expected)
                self.assertIn("검토된 근거 사실", answer["answer"])
                self.assertNotIn("30일", answer["answer"])
                self.assertNotIn("연간계획", answer["answer"])
                self.assertNotIn(answer["candidates"][0]["text"], answer["answer"])
                self.assertExactEvidence(answer)

    def test_discourse_prefixes_preserve_source_bound_followup_and_unknown_subject_guard(self):
        for scope in ({"pinned_doc": CORE}, {"context_doc": CORE}):
            for prefix in ("그럼", "그러면", "그런데", "그리고", "그래서", "그렇다면", "그러니까", "또한", "아울러", "이어서"):
                with self.subTest(scope=scope, prefix=prefix):
                    answer = self.ask(f"{prefix}, 사전협의는 언제 해야 해?", **scope)
                    self.assertEqual(answer["status"], "answer")
                    self.assertEqual(answer["graph"]["context_doc"], CORE)
                    self.assertEqual({p.get("fact_id") for p in answer["graph"]["paths"]}, {"consult"})
                    self.assertEqual({s["doc_id"] for s in answer["graph"]["sources"]}, {CORE})
                    self.assertIn("계획서 제출 7일 전", answer["answer"])
                    self.assertExactEvidence(answer)
                    unknown = self.ask(f"{prefix}, 해왕성 우주출장 사전협의는 언제 해야 해?", **scope)
                    self.assertEqual(unknown["graph"]["answer_kind"], "unknown")
                    self.assertEqual(unknown["candidates"], [])
                    self.assertEqual(unknown["graph"]["sources"], [])

    def test_return_report_preserves_next_day_start_without_invented_weekend_rule(self):
        answer = self.ask("귀국 결과보고서 기한", context_doc=CORE)
        self.assertIn("30일", answer["answer"])
        self.assertIn("다음 날", answer["answer"])
        self.assertNotIn("토요일", answer["answer"])
        self.assertNotIn("20일 전", answer["answer"])
        self.assertEqual(answer["graph"]["context_doc"], CORE)
        self.assertExactEvidence(answer)

    def test_population_and_expense_exceptions_are_not_removed(self):
        population = self.ask("직원이 아니어도 행안부 예산으로 가는 국외출장에 적용돼?", pinned_doc=CORE)
        self.assertIn("다른 사람", population["answer"])
        self.assertIn("특별히 정한 경우는 제외", population["answer"])
        expenses = self.ask("직무상 이해관계자가 경비를 부담해도 돼?", pinned_doc=CORE)
        for phrase in ("금지", "필요성이 인정", "심사위원회 심사", "장관이 허가", "예외"):
            self.assertIn(phrase, expenses["answer"])
        self.assertExactEvidence(expenses)

    def test_annual_plan_is_not_individual_trip_deadline(self):
        answer = self.ask("다음 연도 연간계획 제출 기한", pinned_doc=CORE)
        self.assertIn("12월 20일", answer["answer"])
        self.assertNotIn("출장일 20일 전", answer["answer"])

    def test_registration_exception_and_report_are_an_indivisible_qualified_bundle(self):
        for question in ("귀국 결과보고서 기한", "국가기밀보호 보안유지"):
            with self.subTest(question=question):
                answer = self.ask(question, pinned_doc=CORE)
                ids = {p.get("fact_id") for p in answer["graph"]["paths"]}
                self.assertTrue({"report", "registration-exception"} <= ids)
                self.assertIn("30일 이내 제출", answer["answer"])
                self.assertIn("별지 제7호서식 주석", answer["answer"])
                self.assertIn("부서장이 구체적 사유", answer["answer"])
                self.assertIn("인사혁신처장에게 통보", answer["answer"])
                self.assertTrue(any(p["predicate"] == "qualifiesRule" and p["object"] == "report"
                                    for p in answer["graph"]["paths"]))
                self.assertExactEvidence(answer)

    def test_unresolved_delegation_does_not_invent_approval_matrix(self):
        answer = self.ask("행정안전부 위임전결규정 직급별 허가권 위임범위")
        self.assertEqual(answer["graph"]["answer_kind"], "missing_reference")
        self.assertIn("본문은 이 그래프에 없습니다", answer["answer"])
        self.assertExactEvidence(answer)
        self.assertEqual({s["doc_id"] for s in answer["graph"]["sources"]}, {CORE})

    def test_pins_and_foreign_agencies_cannot_widen_scope(self):
        for question, kwargs in (
            ("계획서 기한", {"pinned_doc": "absent"}),
            ("국가보훈부 출장 계획서 기한", {}),
            ("행정자치부 출장 계획서 기한", {}),
            ("국가공무원 복무규정 적용", {"pinned_doc": CORE}),
            ("국내출장에도 이 기한을 적용해?", {"pinned_doc": CORE}),
        ):
            with self.subTest(question=question, kwargs=kwargs):
                answer = self.ask(question, **kwargs)
                self.assertEqual(answer["status"], "clarify")
                self.assertEqual(answer["candidates"], [])

    def test_explicit_reference_retains_issuer_and_missing_metadata(self):
        answer = self.ask("국가공무원 복무규정 시행일")
        self.assertEqual(answer["graph"]["sources"][0]["institution"], "인사혁신처")
        self.assertIn("미기록", answer["answer"])
        self.assertIsNone(answer["graph"]["sources"][0]["effective_date"])

    def test_unreviewed_amounts_and_unknown_travel_context_have_no_fake_answers(self):
        amount = self.ask("국외출장 숙박비 금액", pinned_doc=CORE)
        self.assertEqual(amount["status"], "review")
        self.assertNotIn("999999", json.dumps(amount, ensure_ascii=False))
        unknown = self.ask("해왕성 우주출장 식비 기준", pinned_doc=CORE)
        self.assertEqual(unknown["graph"]["answer_kind"], "unknown")
        self.assertEqual(unknown["candidates"], [])
        self.assertEqual(unknown["graph"]["sources"], [])

    def test_uncurated_admitted_passage_is_labeled_excerpt(self):
        answer = self.ask("현지활동 연락망", pinned_doc=CORE)
        self.assertEqual(answer["status"], "quoted")
        self.assertExactEvidence(answer)

    def test_total_deterministic_and_evidence_driven_not_question_answers(self):
        self.assertEqual(self.ask(""), self.ask(""))
        self.assertEqual(self.ask("계획서 기한"), self.ask("계획서 기한"))
        with tempfile.TemporaryDirectory() as directory:
            store = make_release(directory, plan_days=26)
            answer = graph_reply(store, "출장계획서 제출", pinned_doc=CORE)
            self.assertIn("출장일 26일 전", answer["answer"])
            self.assertNotIn("출장일 20일 전", answer["answer"])
            self.assertFalse(answer["model_used"])
            self.assertEqual(answer["answer_mode"], "graph")
            store = None
