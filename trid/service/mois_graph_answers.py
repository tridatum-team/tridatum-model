"""Deterministic, evidence-only answers over an immutable MOIS RDF release."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trid.service.mois_graph import GraphStore


_CURRENCY = "수집된 문서 버전의 근거입니다. 수집일·시행일·현행 표시는 현재 법적 유효성을 보증하지 않습니다."
_STOP = {
    "행정안전부", "행안부", "규정", "규정은", "규정의", "문서", "자료", "공무원", "공무", "출장",
    "공무국외출장", "국외출장", "해외출장", "국외", "해외", "어떻게", "언제", "언제까지", "무엇",
    "무슨", "어떤", "얼마나", "알려줘", "알려주세요", "설명", "설명해줘", "설명해주세요", "주세요",
    "알려", "대한", "대해", "있나요", "인가요", "하나요", "해야", "하나", "하나요", "뭐야", "뭔가요",
    "해", "줘", "그", "이", "것", "좀", "및", "그리고", "또", "각각", "차이", "기준", "확인",
    "둘", "다", "아니어", "아니어도", "정리", "간단히", "자세히", "순서", "차례", "개요", "요약",
    "내용", "절차", "진행", "운영", "방식", "필요", "전", "후", "내야",
    # Discourse connectives do not introduce a new business subject or widen source scope.
    "그럼", "그러면", "그런데", "그래서", "그렇다면", "그러니까", "또한", "아울러", "이어서",
}
# These are retrieval vocabulary, not answers: every displayed rule comes from a graph fact or passage.
_CONCEPTS = (
    (("적용", "대상", "공무직", "직원", "비공무원", "제외", "누구"), ("적용", "대상", "제외")),
    (("계획", "신청", "제출", "협의", "준비", "출국", "출발"), ("계획", "제출", "협의")),
    (("보고", "귀국", "복귀", "다녀온", "돌아온"), ("보고", "귀국")),
    (("심사", "위원회", "심의", "면제"), ("심사", "위원회", "심의")),
    (("승인", "허가", "결재"), ("승인", "허가")),
    (("마일리지",), ("마일리지",)),
    (("개인적", "사적"), ("사적", "마일리지")),
    (("경비", "부담", "여비", "비용", "식비", "숙박비", "일비"), ("경비", "부담", "여비", "지급")),
    (("기한", "며칠", "몇일", "언제까지", "토요일", "공휴일", "주말"), ("기한", "deadline", "기간")),
)


def _compact(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.lower())


def _terms(question: str) -> list[str]:
    terms = []
    for word in re.findall(r"[가-힣A-Za-z0-9]+", question):
        if word not in _STOP:
            word = re.sub(r"(?:에서는|으로는|에서|에게|까지|부터|으로|이란|에는|은|는|을|를|의|이|가|과|와|도|에|로)$", "", word)
        if len(word) > 1 and word not in _STOP and word not in terms:
            terms.append(word)
    return terms


def _held(passage: dict) -> bool:
    review = passage.get("value_review") or passage.get("review") or {}
    return bool(passage.get("value_hold") or passage.get("privacy_review")
                or (isinstance(review, dict) and (review.get("privacy_review")
                    or review.get("decision") == "hold_for_value_use")))


def _source(document: dict) -> dict:
    return {**document, "currency_notice": _CURRENCY}


def _response(store: GraphStore, kind: str, answer: str, *, documents=(), paths=(), candidates=(),
              context_doc=None, status=None, warnings=()) -> dict:
    used = {document["doc_id"]: _source(document) for document in documents}
    return {
        "status": status or ("answer" if kind in {"facts", "overview", "metadata"} else "not_found"),
        "answer": answer, "answer_mode": "graph", "model_used": False,
        "candidates": list(candidates), "checked_chunks": len(candidates),
        "warnings": list(dict.fromkeys([_CURRENCY, *warnings])),
        "scope": "행정안전부 공개 근거 그래프의 수록 버전만 사용했습니다. 다른 기관·대상에 대한 적용을 추론하지 않습니다.",
        "graph": {
            "release": store.metadata.get("release"), "sources": list(used.values()), "paths": list(paths),
            "coverage": {"kind": "indexed_snapshot", "counts": store.metadata.get("counts", {}),
                         "complete_institution_rules": False, "current_legal_validity": "unverified"},
            "answer_kind": kind, "context_doc": context_doc,
        },
    }


def _metadata_paths(document: dict, fields: list[str]) -> list[dict]:
    return [{"doc_id": document["doc_id"], "predicate": "metadata", "field": field,
             "value": document.get(field), "evidence_kind": "metadata"} for field in fields]


def _fact_topic(fact: dict) -> str:
    subject = _compact(str(fact.get("subject", "")))
    topical = _compact(" ".join([str(fact.get("subject", "")), *fact.get("topics", []),
                                str(fact.get("qualifiers", {}).get("action", ""))]))
    if fact.get("type") == "ReferenceRule":
        return "reference"
    if fact.get("type") == "ApplicabilityRule":
        return "applicability"
    if "연간" in topical:
        return "annual"
    if "반기" in topical:
        return "periodic"
    if "결과보고" in subject:
        return "report"
    if "마일리지" in topical:
        return "mileage"
    if "사전협의" in topical:
        return "consultation"
    if fact.get("predicate") == "authorizes":
        return "permission"
    if "심사" in subject or "사전심사" in topical:
        return "review"
    if any(word in subject for word in ("경비", "계약", "용역")):
        return "expense"
    if "계획서" in topical and fact.get("predicate") == "deadline":
        return "plan"
    return "other"


def _question_topics(question: str) -> set[str]:
    """Resolve the action first; deadline/recipient words are facets, not other actions."""
    q = _compact(question)
    topics = set()
    for words, topic in (
        (("연간", "다음연도"), "annual"), (("반기",), "periodic"),
        (("결과보고", "귀국보고", "다녀오", "돌아온"), "report"),
        (("마일리지",), "mileage"), (("적용", "공무직", "직원이", "비공무원"), "applicability"),
        (("이해관계자", "경비", "비용", "식비", "숙박비", "여비"), "expense"),
        (("심사", "위원회", "위원장"), "review"), (("허가권", "위임", "승인", "결재"), "permission"),
    ):
        if any(word in q for word in words):
            topics.add(topic)
    if "협의" in q:
        topics.add("consultation")
        if "제출기한" in q or ("제출" in q and any(word in q for word in ("둘", "각각", "모두", "차이"))):
            topics.add("plan")
    elif not topics and any(word in q for word in ("계획서", "제출", "신청")):
        topics.add("plan")
    return topics


def _fact_score(fact: dict, question: str, terms: list[str]) -> int:
    topic = _fact_topic(fact)
    wanted = _question_topics(question)
    topical = " ".join([str(fact.get("subject", "")), *fact.get("topics", []),
                        str(fact.get("qualifiers", {}).get("action", ""))])
    if wanted and topic not in wanted:
        # A reference is relevant only through its own named target or subject.
        if topic != "reference":
            return 0
        return sum(1 for term in terms if term in topical or term in str(fact.get("object", "")))
    if wanted:
        if topic in {"plan", "consultation", "report", "annual", "periodic"}:
            timing = any(word in question for word in ("기한", "언제", "며칠", "몇일", "제출", "등록"))
            if timing and fact.get("predicate") != "deadline":
                return 0
        if "이해관계자" in question and topic == "expense" and "이해관계" not in topical:
            return 0
        if topic == "mileage" and any(word in question for word in ("개인적", "사적")):
            if "사적" not in topical:
                return 0
        return 10 + sum(2 for term in terms if term in topical)
    specific = [term for term in terms if term not in {"기한", "제출", "허가", "진행", "절차"}]
    return sum(2 for term in specific if term in topical)


def _overview_facts(facts: list[dict]) -> list[dict]:
    """Present source-defined workflow stages, not every clause sharing a travel word."""
    ordered = sorted(facts, key=lambda f: (f["evidence"]["start"], f["id"]))
    result = []
    for topic in ("applicability", "consultation", "plan", "review", "permission", "report"):
        choices = [f for f in ordered if _fact_topic(f) == topic and f.get("type") != "ReferenceRule"]
        if topic in {"plan", "consultation", "report"}:
            choices = [f for f in choices if f.get("predicate") == "deadline"]
        if choices:
            result.append(choices[0])
    return result


def _unsupported_subjects(terms: list[str], facts: list[dict], documents: list[dict]) -> list[str]:
    """Reject ungrounded topic modifiers rather than matching only 'travel' or 'expenses'."""
    corpus = " ".join([d["title"] + " " + str(d.get("institution") or "") for d in documents] + [
        " ".join([str(f.get("subject", "")), str(f.get("object", "")),
                  f.get("evidence", {}).get("quote", ""), *f.get("topics", [])]) for f in facts
    ])
    vocabulary = [word for triggers, targets in _CONCEPTS for word in (*triggers, *targets)]
    unsupported = []
    for term in terms:
        if term in corpus or any(word in term for word in vocabulary):
            continue
        if re.search(r"(?:해|돼|되나|되나요|인가|있어|없어|하면|오면|되면|하는|하려면|해야|해도|해줘|주세요|나요|인가요)$", term):
            continue
        unsupported.append(term)
    return unsupported


def _candidate(document: dict, passage: dict, evidence: dict) -> dict | None:
    if _held(passage):
        return None
    text = passage.get("text", "")
    start = evidence.get("start", -1) - passage.get("char_start", 0)
    end = evidence.get("end", -1) - passage.get("char_start", 0)
    quote = evidence.get("quote", "")
    if not quote or not (0 <= start < end <= len(text)) or text[start:end] != quote:
        return None
    return {
        "doc_id": document["doc_id"], "passage_id": passage["id"], "text": text,
        "page": passage.get("page"), "page_text_start": passage.get("char_start", 0),
        "start": passage.get("char_start", 0), "path": passage.get("path", []), "evidence_kind": "quotation",
        "source_label": f"{document.get('institution') or '발행기관 미상'} · {document['title']}",
        "prediction": {"answer": quote, "abstains": False, "source_character_span": [start, end]},
    }


def _qualified_rules(selected: list[dict], available: list[dict]) -> list[dict]:
    """Keep a rule and graph-linked qualifications together in either retrieval direction."""
    chosen = {f["id"]: f for f in selected}
    by_id = {f["id"]: f for f in available}
    changed = True
    while changed:
        changed = False
        for fact in available:
            target = fact.get("qualifiers", {}).get("exception_to")
            if target and (fact["id"] in chosen or target in chosen):
                for linked in (fact, by_id.get(target)):
                    if linked is not None and linked["id"] not in chosen:
                        chosen[linked["id"]] = linked
                        changed = True
    return list(chosen.values())


def _quote_facts(store: GraphStore, facts: list[dict], documents: dict) -> tuple[list, list, list, list]:
    passages = {}
    lines, candidates, paths, used = [], [], [], []
    quote_numbers = {}
    for fact in facts:
        doc_id = fact["doc_id"]
        if doc_id not in passages:
            passages[doc_id] = {p["id"]: p for p in store.passages(doc_id)}
        evidence = fact.get("evidence", {})
        passage = passages[doc_id].get(evidence.get("passage_id"))
        if passage is None:
            continue
        candidate = _candidate(documents[doc_id], passage, evidence)
        if candidate is None:
            continue
        paths.append({key: fact[key] for key in ("doc_id", "id", "subject", "predicate", "object", "qualifiers",
                                                 "evidence", "target_doc_id") if key in fact})
        paths[-1]["fact_id"] = fact["id"]
        target = fact.get("qualifiers", {}).get("exception_to")
        if target:
            paths.append({"doc_id": doc_id, "fact_id": fact["id"], "predicate": "qualifiesRule",
                          "object": target, "evidence": evidence})
        used.append(documents[doc_id])
        quote_key = (doc_id, evidence["start"], evidence["end"])
        if quote_key not in quote_numbers:
            candidates.append(candidate)
            quote_numbers[quote_key] = len(candidates)
        citation = quote_numbers[quote_key]
        provenance = fact.get("qualifiers", {}).get("article") or " / ".join(passage.get("path", []))
        lines.append(f"{fact.get('subject') or documents[doc_id]['title']} ({provenance}) — 검토된 근거 사실: {fact['object']} [{citation}]")
    return lines, candidates, paths, used


def graph_reply(store: GraphStore, question: str, *, pinned_doc: str | None = None,
                context_doc: str | None = None) -> dict:
    """Return a graph response for every question; never fall through to model memory.

    Explicit pins and follow-up context restrict retrieval. A linked reference is
    answerable only when its title or document ID is explicitly selected.
    """
    question = question.strip()
    documents = {d["doc_id"]: d for d in store.documents()}
    selected_id = pinned_doc or context_doc
    if selected_id and selected_id not in documents:
        return _response(store, "scope", "선택한 문서는 이 그래프에 없습니다. 문서 선택을 해제하거나 수록 문서를 선택해 주세요.", status="clarify")
    if not question:
        return _response(store, "clarification", "확인할 규정명이나 질문을 입력해 주세요.", context_doc=selected_id, status="clarify")
    compact = _compact(question.replace("행안부", "행정안전부"))
    named = [d for d in documents.values() if _compact(d["title"]) and _compact(d["title"]) in compact]
    if selected_id and named and all(d["doc_id"] != selected_id for d in named):
        return _response(store, "scope", "선택한 문서와 질문에서 지정한 문서가 다릅니다. 기존 문서 범위를 넓히지 않았습니다. 문서 선택을 변경해 주세요.",
                         context_doc=selected_id, status="clarify")
    if not selected_id and named:
        longest = max(len(_compact(d["title"])) for d in named)
        named = [d for d in named if len(_compact(d["title"])) == longest]
    selected = [documents[selected_id]] if selected_id else named
    agency_words = set(re.findall(r"[가-힣]{2,}(?:위원회|부|청|처)(?=의|에서|은|는|\s|[?？.,]|$)", question))
    agency_words -= {"행정안전부", "행안부", "제출부", "첨부", "여부", "심사위원회"}
    allowed_agencies = {d.get("institution") for d in selected}
    foreign = agency_words - allowed_agencies
    if foreign:
        return _response(store, "scope", f"{', '.join(sorted(foreign))}의 규정으로 답할 근거 범위를 선택하지 않았습니다. 행정안전부 규정을 다른 기관에 적용하지 않습니다. 수록된 참조 문서는 정확한 문서명으로 지정해 주세요.",
                         context_doc=selected_id, status="clarify")
    if not selected:
        selected = [d for d in documents.values() if d.get("scope") == "mois"]
    terms = _terms(question)
    metadata_fields = []
    for triggers, fields in (
        (("시행",), ("effective_date",)), (("제정", "발령", "개정", "날짜", "기준일"), ("date", "date_type", "version")),
        (("수집",), ("retrieved_at",)), (("최신", "현행", "유효", "현재"), ("currency", "retrieved_at", "effective_date")),
        (("발행", "소관", "출처", "원문 URL", "원문 링크", "링크"), ("institution", "url", "source_id")),
    ):
        if any(trigger in question for trigger in triggers):
            metadata_fields.extend(fields)
    overview = any(word in question for word in ("개요", "요약", "어떤 내용", "무슨 내용", "무엇을", "전체 내용"))
    overview = overview or (any(word in question for word in ("설명해", "알려줘", "알려주세요"))
                            and not any(any(word in question for word in triggers) for triggers, _ in _CONCEPTS)
                            and bool(named))
    procedure = any(word in question for word in ("절차", "어떻게", "준비해야"))
    all_facts = [fact for d in selected for fact in store.facts(d["doc_id"])]
    ranked = sorted(((_fact_score(f, question, terms), f) for f in all_facts), key=lambda pair: (-pair[0], pair[1]["id"]))
    # A title-only question is an overview, not an invented answer to a missing question.
    if named and not metadata_fields and (not terms or any(compact == _compact(d["title"]) for d in named)):
        overview = True
    if len(selected) > 1:
        scores = {}
        for score, fact in ranked:
            scores[fact["doc_id"]] = max(scores.get(fact["doc_id"], 0), score)
        for d in selected:
            title = _compact(d["title"])
            score = sum(len(term) for term in terms if _compact(term) in title)
            if any(word in compact for word in ("국외출장", "해외출장")) and "국외출장" in title:
                score += 10
            scores[d["doc_id"]] = scores.get(d["doc_id"], 0) + score
        highest = max(scores.values(), default=0)
        winners = [d for d in selected if highest > 0 and scores.get(d["doc_id"], 0) == highest]
        if len(winners) > 1:
            curated_ids = {f["doc_id"] for f in all_facts}
            curated = [d for d in winners if d["doc_id"] in curated_ids]
            if len(curated) == 1:
                winners = curated
        if len(winners) == 1:
            selected = winners
            ranked = [(score, f) for score, f in ranked if f["doc_id"] == winners[0]["doc_id"]]
        elif metadata_fields or (procedure and not overview):
            return _response(store, "clarification", "어느 문서의 내용인지 지정해 주세요. 기관 전체의 자료를 하나의 규정처럼 답하지 않습니다.", status="clarify")
    single = selected[0] if len(selected) == 1 else None
    context = single["doc_id"] if single else selected_id
    if single and "국내" in question and "국외" in single["title"]:
        return _response(store, "scope", f"「{single['title']}」은 국외출장 문서입니다. 국내출장의 지급액·절차 근거로 대체할 수 없습니다.",
                         documents=[single], paths=_metadata_paths(single, ["title"]), context_doc=context, status="clarify")
    # References establish a citation, not the content or applicability of the cited regulation.
    missing = [r for d in selected for r in store.references(d["doc_id"])
               if not r.get("resolved") and any(t in str(r.get("object", "")) or
                                                any(t in topic for topic in r.get("topics", [])) for t in terms)]
    if missing:
        lines, candidates, paths, used = _quote_facts(store, missing, documents)
        return _response(store, "missing_reference", "참조 관계는 확인되지만 해당 참조 규정 본문은 이 그래프에 없습니다. 위임·전결권자나 세부 기준을 확정할 수 없습니다.\n\n" + "\n\n".join(lines),
                         documents=used, paths=paths, candidates=candidates, context_doc=context)
    if single and metadata_fields:
        fields = list(dict.fromkeys(metadata_fields))
        labels = {"date": single.get("date_type") or "자료 기준일", "date_type": "날짜 종류", "version": "버전", "effective_date": "시행일",
                  "retrieved_at": "수집일", "currency": "수집 시 현행성 표기", "institution": "발행기관", "url": "원문 URL", "source_id": "원문 식별자"}
        lines = [f"{labels[field]}: {single.get(field) or '미기록·확인 불가'}" for field in fields if field != "date_type"]
        return _response(store, "metadata", f"「{single['title']}」의 출처 메타데이터입니다(본문 인용 아님).\n" + "\n".join(lines) + "\n" + _CURRENCY,
                         documents=[single], paths=_metadata_paths(single, fields), context_doc=context)
    monetary = (any(word in question for word in ("금액", "정액", "상한액", "몇 원", "몇원"))
                or ("얼마" in question and "얼마나" not in question and "얼마 전" not in question)
                or ("계산" in question and any(word in question for word in ("여비", "식비", "숙박비", "일비", "비용"))))
    if monetary:
        refs = [f for score, f in ranked if score > 0 and f.get("type") == "ReferenceRule"]
        lines, candidates, paths, used = _quote_facts(store, refs, documents)
        answer = "이 질문의 지급액·상한·계산 결과를 확정할 검증된 금액 근거가 없습니다. 직급·국가·등급 등 적용 조건과 검증된 표 없이 금액을 계산하지 않습니다."
        if lines:
            answer += "\n아래는 참조 관계에 관해 검토된 사실일 뿐 금액 또는 참조 판본의 적용을 확정한 근거는 아닙니다.\n\n" + "\n\n".join(lines)
        return _response(store, "table_boundary", answer, documents=used, paths=paths, candidates=candidates,
                         context_doc=context, status="review")
    unsupported = _unsupported_subjects(terms, [f for _, f in ranked], selected)
    if unsupported:
        supporting = store.search_passages(unsupported, doc_ids=[d["doc_id"] for d in selected], limit=8)
        if not any(not _held(hit) and all(t in hit.get("text", "") for t in unsupported) for hit in supporting):
            return _response(store, "unknown", "질문의 대상·상황에 맞는 검증된 근거를 확인하지 못했습니다. '출장'이나 '비용' 같은 일부 단어가 같다는 이유로 다른 대상의 규정을 적용하지 않습니다.",
                             context_doc=context)
        ranked = []
    broad_procedure = procedure and not _question_topics(question)
    if overview or broad_procedure:
        facts = _overview_facts([f for _, f in ranked])
    else:
        facts = [f for score, f in ranked if score > 0 and f.get("type") != "ReferenceRule"]
        # Within one action, prefer the requested facet over adjacent form or administration rules.
        best = {}
        for score, fact in ranked:
            topic = _fact_topic(fact)
            best[topic] = max(best.get(topic, 0), score)
        facts = [f for f in facts if _fact_score(f, question, terms) == best[_fact_topic(f)]]
    facts = _qualified_rules(facts, [f for f in all_facts if f["doc_id"] in {d["doc_id"] for d in selected}])
    lines, candidates, paths, used = _quote_facts(store, facts, documents)
    if lines:
        heading = "수록된 규정의 주요 절차" if overview or broad_procedure else "질문 관련 절차·조건"
        return _response(store, "overview" if overview or broad_procedure else "facts", heading + "입니다. 아래는 출처에 근거해 검토한 사실이며 원문 그대로의 인용이 아닙니다. 번호별 정확한 원문은 근거 구간에 표시했습니다. 개별 사안의 적용 판정은 아닙니다.\n\n" + "\n\n".join(lines),
                         documents=used, paths=paths, candidates=candidates, context_doc=context)
    # Uncurated topics can still return admitted exact excerpts, never a generated legal conclusion.
    search_terms = [t for t in terms if len(t) >= 2]
    hits = store.search_passages(search_terms, doc_ids=[d["doc_id"] for d in selected], limit=8) if search_terms else []
    candidates, paths, used, lines = [], [], [], []
    for hit in hits:
        if _held(hit) or not all(t in hit.get("text", "") or t in documents[hit["doc_id"]]["title"] for t in search_terms):
            continue
        evidence = {"passage_id": hit["id"], "start": hit["char_start"], "end": hit["char_end"], "quote": hit["text"]}
        candidate = _candidate(documents[hit["doc_id"]], hit, evidence)
        if candidate:
            candidates.append(candidate)
            paths.append({"doc_id": hit["doc_id"], "predicate": "hasPassage", "evidence": evidence})
            used.append(documents[hit["doc_id"]])
            lines.append(f"[{len(candidates)}] 원문 인용:\n“{hit['text']}”")
    if candidates:
        return _response(store, "excerpts", "관련 원문 구간입니다. 정형화된 절차 사실로 검증된 답이나 적용 판단은 아닙니다.\n\n" + "\n\n".join(lines),
                         documents=used, paths=paths, candidates=candidates, context_doc=context, status="quoted")
    return _response(store, "unknown", "이 그래프의 수록·검증된 근거에서 질문에 답할 내용을 확인하지 못했습니다. 규정이 없다는 뜻은 아니며, 기관 전체나 현재 법령을 모두 확인한 결과도 아닙니다.", context_doc=context)
