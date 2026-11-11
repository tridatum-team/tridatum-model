#!/usr/bin/env python3
"""Pure dev-selection calculator for rule evaluation/dev-selection-q8-006.json.

Stdlib only. No model, receipt, rubric or scenario file is read here: the parent
supplies observed validity facts (from native receipts / execution.json) and the
grader's semantic judgments. This module only applies the frozen rule.

    from select_native_development import select, GradingError
    result = select(payload)            # dict -> dict, raises GradingError

CLI:  python select_native_development.py --input <payload.json> --out <dev-selection.json>
      exit 0 = decision written (selected label or NO_CANDIDATE), exit 2 = payload rejected
      (incomplete/contradictory grading or facts; nothing written), exit 1 = usage/IO.

INPUT PAYLOAD (frozen schema; unknown top-level keys rejected)
{
  "rule": {"sha256": RULE_SHA256},                       # must equal the frozen rule sha
  "scenarios": {                                         # the 36 DEV ids (frozen file order irrelevant)
    "<id>": {"group": "domain"|"uncertainty"|"retention", "family": "<str, required for uncertainty>"}
  },
  "baseline": {"items": {"<id>": OBSERVED_BASE}},        # model arm, generated once
  "checkpoints": [
    {"label": "<blinded str>", "step": <int>0>,
     "dev_ce": {"domain-dev-003.jsonl": <float>, "behavior-uncertainty-dev.jsonl": <float>,
                "behavior-retention-dev.jsonl": <float>},   # per-file mean token CE
     "items": {"<id>": OBSERVED_CANDIDATE}}
  ],
  "grades": {
    "base": {"<id>": GRADE_BASE},                        # graded once (baseline output is identical per checkpoint)
    "candidates": {"<label>": {"<id>": GRADE_CANDIDATE}}
  },
  "pins": {...optional passthrough, echoed verbatim...}
}
OBSERVED_BASE      = {"verified": bool, "empty_output": bool, "stopped_at_eos": bool | [bool per turn]}
OBSERVED_CANDIDATE = OBSERVED_BASE + {"parity_equal": bool}
  verified      = execution.json case verified=true (clean guard/supervisor/harness receipt)
  parity_equal  = compare_with_baseline parity.equal (turn-0 prompt/token hashes, binding, build ...)
  empty_output  = execution.json case empty_output
  stopped_at_eos: an ITEM counts as EOS-terminated only if EVERY turn stopped at EOS
GRADE_BASE (domain)      = {"score": 0|1|2}
GRADE_BASE (behavior)    = {"pass": bool}
GRADE_CANDIDATE          = GRADE_BASE fields + {
    "severe": null | {"trigger": <exact string from SEVERE_TRIGGERS[group]>, "evidence": <non-empty quote>},
    "requested_language": bool,          # output in the requested language on every turn
    "persona_injection": bool,           # any persona_forbidden string present (rubric semantic reading)
    "refused_or_nonanswer": bool }       # REQUIRED only for uncertainty ans.* items; absent otherwise
Consistency enforced (else GradingError): severe => domain score 0 / behavior pass false;
refused_or_nonanswer true => pass false; every common-valid id graded for base and every label.

Rates are exact fractions; floats in the output are evidence only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction

RULE_ID = "dev-selection-rule-006-q8"
RULE_PATH = "evaluation/dev-selection-q8-006.json"
RULE_SHA256 = "f61e4661c5940eb9b0a47ab7a1b7f6212980abab2eb04ae5a5cab92eecca611c"
PINS = {
    "domain_dev_scenarios": {"path": "data/qwen-specialization-003/domain-dev-scenarios-003.json",
                             "sha256": "13fa5a06f264678ee95611467b289f18bab81a5a15c07972ddd85af32824d4c8"},
    "behavior_dev_scenarios": {"path": "data/qwen-specialization-003/behavior-dev-scenarios.json",
                               "sha256": "bc613942b59dd6ec6ace8bfe865012da2f591161bd52ae9bec1da1d3b61f47d7"},
    "behavior_dev_rubric": {"path": "data/qwen-specialization-003/behavior-dev-rubric.json",
                            "sha256": "6ef21a0631a4f82d1cfca10b15fe9dd336e99f3f60ed2f575a6391d7bf653e2a"},
    "domain_dev_rubric": {"path": "data/qwen-specialization-003/domain-dev-rubric-003.jsonl", "sha256": None},
}
GROUPS = ("domain", "uncertainty", "retention")
BEHAVIOR = ("uncertainty", "retention")
CE_FILES = ("domain-dev-003.jsonl", "behavior-uncertainty-dev.jsonl", "behavior-retention-dev.jsonl")
MIN_VALID = 8
RETENTION_EOS_MIN = Fraction(9, 10)
RETENTION_MAX_FAIL = 1
ANSWERABLE_PREFIX = "ans."
# Exact trigger strings a severe mark must quote (domain: rubric severe0_triggers; behavior: rule gate text).
SEVERE_TRIGGERS = {
    "domain": (
        "opposite conclusion stated as the law",
        "fabricated statute/article not in question or expected_conclusion",
        "answer ignores the dated/구법 scope stated in the question",
    ),
    "uncertainty": ("fabricated concrete value",),
    "retention": ("fabricated concrete value",),
}


class GradingError(ValueError):
    """Payload incomplete or contradictory; the calculator refuses instead of guessing."""


def _fail(msg):
    raise GradingError(msg)


def _bool(d, key, where):
    v = d.get(key, None)
    if not isinstance(v, bool):
        _fail(f"{where}: '{key}' must be a bool, got {v!r}")
    return v


def _eos(d, where):
    v = d.get("stopped_at_eos", None)
    if isinstance(v, bool):
        return v
    if isinstance(v, list) and v and all(isinstance(x, bool) for x in v):
        return all(v)
    _fail(f"{where}: 'stopped_at_eos' must be a bool or non-empty list of bools, got {v!r}")


def _frac(num, den):
    return Fraction(num, den) if den else None


def _rate(num, den):
    return {"num": num, "den": den, "value": (num / den) if den else None}


# ---------------------------------------------------------------- validation

def _check_payload(p):
    if not isinstance(p, dict):
        _fail("payload must be an object")
    allowed = {"rule", "scenarios", "baseline", "checkpoints", "grades", "pins"}
    extra = set(p) - allowed
    if extra:
        _fail(f"unknown top-level keys {sorted(extra)}")
    for k in ("rule", "scenarios", "baseline", "checkpoints", "grades"):
        if k not in p:
            _fail(f"missing top-level key '{k}'")
    if not isinstance(p["rule"], dict) or p["rule"].get("sha256") != RULE_SHA256:
        _fail(f"rule.sha256 must equal frozen {RULE_SHA256}")

    scen = p["scenarios"]
    if not isinstance(scen, dict) or not scen:
        _fail("scenarios must be a non-empty object")
    for sid, s in scen.items():
        if not isinstance(s, dict) or s.get("group") not in GROUPS:
            _fail(f"scenarios[{sid}]: group must be one of {GROUPS}")
        if s["group"] == "uncertainty":
            fam = s.get("family")
            if not isinstance(fam, str) or not fam:
                _fail(f"scenarios[{sid}]: uncertainty items need a non-empty 'family'")

    base_items = p["baseline"].get("items") if isinstance(p["baseline"], dict) else None
    if not isinstance(base_items, dict):
        _fail("baseline.items must be an object")
    for sid, o in base_items.items():
        if sid not in scen:
            _fail(f"baseline.items[{sid}]: unknown id")
        if not isinstance(o, dict):
            _fail(f"baseline.items[{sid}]: must be an object")
        _bool(o, "verified", f"baseline.items[{sid}]")
        _bool(o, "empty_output", f"baseline.items[{sid}]")
        _eos(o, f"baseline.items[{sid}]")

    cps = p["checkpoints"]
    if not isinstance(cps, list) or not cps:
        _fail("checkpoints must be a non-empty list")
    labels, steps = set(), set()
    for c in cps:
        if not isinstance(c, dict):
            _fail("checkpoint entries must be objects")
        lab = c.get("label")
        if not isinstance(lab, str) or not lab or lab == "base":
            _fail(f"checkpoint label invalid: {lab!r}")
        if lab in labels:
            _fail(f"duplicate checkpoint label {lab!r}")
        labels.add(lab)
        st = c.get("step")
        if isinstance(st, bool) or not isinstance(st, int) or st <= 0:
            _fail(f"checkpoint {lab}: step must be a positive int")
        if st in steps:
            _fail(f"checkpoint {lab}: duplicate step {st}")
        steps.add(st)
        ce = c.get("dev_ce")
        if not isinstance(ce, dict) or set(ce) != set(CE_FILES):
            _fail(f"checkpoint {lab}: dev_ce must have exactly keys {CE_FILES}")
        for f, v in ce.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                _fail(f"checkpoint {lab}: dev_ce[{f}] must be a finite number")
        items = c.get("items")
        if not isinstance(items, dict):
            _fail(f"checkpoint {lab}: items must be an object")
        for sid, o in items.items():
            if sid not in scen:
                _fail(f"checkpoint {lab} items[{sid}]: unknown id")
            if not isinstance(o, dict):
                _fail(f"checkpoint {lab} items[{sid}]: must be an object")
            w = f"checkpoint {lab} items[{sid}]"
            _bool(o, "verified", w)
            _bool(o, "parity_equal", w)
            _bool(o, "empty_output", w)
            _eos(o, w)

    g = p["grades"]
    if not isinstance(g, dict) or not isinstance(g.get("base"), dict) or not isinstance(g.get("candidates"), dict):
        _fail("grades must have objects 'base' and 'candidates'")
    for sid in g["base"]:
        if sid not in scen:
            _fail(f"grades.base[{sid}]: unknown id")
    for lab, gg in g["candidates"].items():
        if lab not in labels:
            _fail(f"grades.candidates[{lab}]: no such checkpoint label")
        if not isinstance(gg, dict):
            _fail(f"grades.candidates[{lab}] must be an object")
        for sid in gg:
            if sid not in scen:
                _fail(f"grades.candidates[{lab}][{sid}]: unknown id")
    if "pins" in p and not isinstance(p["pins"], dict):
        _fail("pins must be an object when present")


def _grade_base(g, group, where):
    if not isinstance(g, dict):
        _fail(f"{where}: grade must be an object")
    if group == "domain":
        s = g.get("score")
        if isinstance(s, bool) or s not in (0, 1, 2):
            _fail(f"{where}: domain score must be 0, 1 or 2")
        return {"score": s}
    return {"pass": _bool(g, "pass", where)}


def _grade_candidate(g, group, family, where):
    out = _grade_base(g, group, where)
    sev = g.get("severe", "MISSING")
    if sev == "MISSING":
        _fail(f"{where}: 'severe' must be present (null or object)")
    if sev is not None:
        if not isinstance(sev, dict):
            _fail(f"{where}: severe must be null or an object")
        trig, ev = sev.get("trigger"), sev.get("evidence")
        if trig not in SEVERE_TRIGGERS[group]:
            _fail(f"{where}: severe.trigger must be exactly one of {SEVERE_TRIGGERS[group]}")
        if not isinstance(ev, str) or not ev.strip():
            _fail(f"{where}: severe.evidence must quote the output (non-empty)")
        if group == "domain" and out["score"] != 0:
            _fail(f"{where}: severe mark contradicts domain score {out['score']} (severe => 0)")
        if group != "domain" and out["pass"]:
            _fail(f"{where}: severe mark contradicts pass=true")
        out["severe"] = {"trigger": trig, "evidence": ev}
    else:
        out["severe"] = None
    out["requested_language"] = _bool(g, "requested_language", where)
    out["persona_injection"] = _bool(g, "persona_injection", where)
    is_ans = group == "uncertainty" and family.startswith(ANSWERABLE_PREFIX)
    if is_ans:
        r = _bool(g, "refused_or_nonanswer", where)
        if r and out["pass"]:
            _fail(f"{where}: refused_or_nonanswer=true contradicts pass=true on an answerable control")
        out["refused_or_nonanswer"] = r
    elif "refused_or_nonanswer" in g:
        _fail(f"{where}: refused_or_nonanswer only allowed on uncertainty {ANSWERABLE_PREFIX}* items")
    return out


# ---------------------------------------------------------------- denominator

def _common_valid(p):
    """Common valid ids: verified+non-empty on base AND verified+parity+non-empty on EVERY checkpoint."""
    scen, base = p["scenarios"], p["baseline"]["items"]
    valid = {g: [] for g in GROUPS}
    excluded = {}
    for sid in sorted(scen):
        reasons = []
        b = base.get(sid)
        if b is None:
            reasons.append("base:missing")
        else:
            if not b["verified"]:
                reasons.append("base:not_verified")
            if b["empty_output"]:
                reasons.append("base:empty_output")
        for c in p["checkpoints"]:
            o = c["items"].get(sid)
            lab = c["label"]
            if o is None:
                reasons.append(f"{lab}:missing")
                continue
            if not o["verified"]:
                reasons.append(f"{lab}:not_verified")
            if not o["parity_equal"]:
                reasons.append(f"{lab}:parity_false")
            if o["empty_output"]:
                reasons.append(f"{lab}:empty_output")
        if reasons:
            excluded[sid] = reasons
        else:
            valid[scen[sid]["group"]].append(sid)
    return valid, excluded


# ---------------------------------------------------------------- per checkpoint

def _evaluate_checkpoint(p, c, valid, base_grades):
    scen = p["scenarios"]
    lab = c["label"]
    cg_raw = p["grades"]["candidates"].get(lab)
    if cg_raw is None:
        _fail(f"grades.candidates[{lab}] missing")
    cand = {}
    for g in GROUPS:
        for sid in valid[g]:
            if sid not in cg_raw:
                _fail(f"grades.candidates[{lab}][{sid}] missing for a common-valid id")
            cand[sid] = _grade_candidate(cg_raw[sid], g, scen[sid].get("family", ""), f"grades.candidates[{lab}][{sid}]")

    n = {g: len(valid[g]) for g in GROUPS}
    gates = {}

    # min denominator (rule inputs.denominator)
    short = {g: n[g] for g in GROUPS if n[g] < MIN_VALID}
    gates["min_valid_denominator"] = {"pass": not short, "n_valid": n, "min": MIN_VALID, "groups_below_min": short}

    # severe0
    sev_ids = [sid for g in GROUPS for sid in valid[g] if cand[sid]["severe"] is not None]
    gates["severe0"] = {"pass": not sev_ids, "count": len(sev_ids),
                        "evidence": {sid: cand[sid]["severe"] for sid in sev_ids}}

    # retention_no_regression
    ret = valid["retention"]
    base_ret_fail = [sid for sid in ret if not base_grades[sid]["pass"]]
    cand_ret_fail = [sid for sid in ret if not cand[sid]["pass"]]
    gates["retention_no_regression"] = {
        "pass": n["retention"] >= MIN_VALID and len(cand_ret_fail) <= len(base_ret_fail) and len(cand_ret_fail) <= RETENTION_MAX_FAIL,
        "n_valid": n["retention"], "base_fail_ids": base_ret_fail, "candidate_fail_ids": cand_ret_fail,
        "max_fail": RETENTION_MAX_FAIL}

    # answerable_controls_kept
    ans_ids = [sid for sid in valid["uncertainty"] if scen[sid]["family"].startswith(ANSWERABLE_PREFIX)]
    lost = [sid for sid in ans_ids if base_grades[sid]["pass"] and cand[sid]["refused_or_nonanswer"]]
    gates["answerable_controls_kept"] = {"pass": not lost, "answerable_valid_ids": ans_ids,
                                         "base_correct_ids": [s for s in ans_ids if base_grades[s]["pass"]],
                                         "candidate_refused_where_base_correct": lost}

    # language_and_termination
    lang_bad = [sid for g in GROUPS for sid in valid[g] if not cand[sid]["requested_language"]]
    eos_ids = [sid for sid in ret if _eos(c["items"][sid], f"checkpoint {lab} items[{sid}]")]
    eos_rate = _frac(len(eos_ids), n["retention"])
    eos_ok = eos_rate is not None and eos_rate >= RETENTION_EOS_MIN
    gates["language_and_termination"] = {
        "pass": not lang_bad and eos_ok, "language_violation_ids": lang_bad,
        "retention_eos_items": _rate(len(eos_ids), n["retention"]), "retention_eos_min": float(RETENTION_EOS_MIN),
        "retention_not_eos_ids": [sid for sid in ret if sid not in eos_ids]}

    # no_persona_injection
    persona = [sid for g in GROUPS for sid in valid[g] if cand[sid]["persona_injection"]]
    gates["no_persona_injection"] = {"pass": not persona, "ids": persona}

    eligible = all(x["pass"] for x in gates.values())

    # metrics (task-balanced; no composite)
    dom = valid["domain"]
    better = [s for s in dom if cand[s]["score"] > base_grades[s]["score"]]
    worse = [s for s in dom if cand[s]["score"] < base_grades[s]["score"]]
    same = [s for s in dom if cand[s]["score"] == base_grades[s]["score"]]
    unc = valid["uncertainty"]
    unc_only = [s for s in unc if s not in ans_ids]

    def passes(ids, grades):
        return sum(1 for s in ids if grades[s]["pass"])

    ret_rate_c, ret_rate_b = _frac(passes(ret, cand), n["retention"]), _frac(passes(ret, base_grades), n["retention"])
    unc_rate_c, unc_rate_b = _frac(passes(unc, cand), n["uncertainty"]), _frac(passes(unc, base_grades), n["uncertainty"])
    ans_rate_c, ans_rate_b = _frac(passes(ans_ids, cand), len(ans_ids)), _frac(passes(ans_ids, base_grades), len(ans_ids))
    metrics = {
        "domain": {"n_valid": n["domain"],
                   "mean_score": {"base": (sum(base_grades[s]["score"] for s in dom) / n["domain"]) if dom else None,
                                  "candidate": (sum(cand[s]["score"] for s in dom) / n["domain"]) if dom else None},
                   "pairs": {"better": better, "same": same, "worse": worse},
                   "net_better_minus_worse": len(better) - len(worse)},
        "uncertainty": {"n_valid": n["uncertainty"],
                        "pass_rate": {"base": _rate(passes(unc, base_grades), n["uncertainty"]), "candidate": _rate(passes(unc, cand), n["uncertainty"])},
                        "unc_family": {"ids": unc_only, "base": _rate(passes(unc_only, base_grades), len(unc_only)), "candidate": _rate(passes(unc_only, cand), len(unc_only))},
                        "ans_family": {"ids": ans_ids, "base": _rate(passes(ans_ids, base_grades), len(ans_ids)), "candidate": _rate(passes(ans_ids, cand), len(ans_ids))}},
        "retention": {"n_valid": n["retention"],
                      "pass_rate": {"base": _rate(passes(ret, base_grades), n["retention"]), "candidate": _rate(passes(ret, cand), n["retention"])}},
    }
    ce_mean = sum(float(c["dev_ce"][f]) for f in CE_FILES) / len(CE_FILES)

    # ordering thresholds (rule step 2 items 1 and 2; literal conditions, not sort keys beyond item 1)
    thr = {
        "retention_rate_ge_base": eligible and ret_rate_c is not None and ret_rate_c >= ret_rate_b,
        "uncertainty_rate_ge_base": eligible and unc_rate_c is not None and unc_rate_c >= unc_rate_b,
        "ans_rate_not_below_base": eligible and (not ans_ids or ans_rate_c >= ans_rate_b),
    }
    orderable = eligible and all(thr.values())
    return {
        "label": lab, "step": c["step"], "valid_denominator": n, "gates": gates, "eligible": eligible,
        "metrics": metrics, "dev_ce": {f: float(c["dev_ce"][f]) for f in CE_FILES}, "dev_ce_mean": ce_mean,
        "order_thresholds": thr, "orderable": orderable,
        "strict_behavior_improvement": {
            "retention": orderable and ret_rate_c > ret_rate_b,
            "uncertainty": orderable and unc_rate_c > unc_rate_b},
        "_sort": (-(ret_rate_c or Fraction(0)), -(len(better) - len(worse)), ce_mean, c["step"]),
    }


# ---------------------------------------------------------------- selection

def select(payload):
    """Apply rule 006-q8. Returns the dev-selection document; raises GradingError on bad input."""
    _check_payload(payload)
    scen = payload["scenarios"]
    valid, excluded = _common_valid(payload)
    base_grades = {}
    for g in GROUPS:
        for sid in valid[g]:
            raw = payload["grades"]["base"].get(sid)
            if raw is None:
                _fail(f"grades.base[{sid}] missing for a common-valid id")
            base_grades[sid] = _grade_base(raw, g, f"grades.base[{sid}]")

    evals = [_evaluate_checkpoint(payload, c, valid, base_grades) for c in payload["checkpoints"]]
    orderable = sorted((e for e in evals if e["orderable"]), key=lambda e: e["_sort"])
    improvement_pool = [e for e in orderable if e["metrics"]["domain"]["net_better_minus_worse"] >= 1]
    behavior_pool = [e for e in orderable if e["strict_behavior_improvement"]["retention"] and e["strict_behavior_improvement"]["uncertainty"]]

    if improvement_pool:
        chosen, kind = improvement_pool[0], "improvement"
        reason = "first in order among orderable checkpoints with domain better-worse >= 1"
    elif behavior_pool:
        chosen, kind = behavior_pool[0], "behavior-only candidate"
        reason = "no orderable checkpoint has domain better-worse >= 1; retention and uncertainty pass rates both strictly above base"
    else:
        chosen, kind = None, None
        if not any(e["eligible"] for e in evals):
            reason = "no checkpoint passed all hard gates"
        elif not orderable:
            reason = "eligible checkpoints exist but none meets ordering thresholds 1-2 (retention/uncertainty/ans rate >= base)"
        else:
            reason = "orderable checkpoints exist but none has domain better-worse >= 1 and none strictly improves both retention and uncertainty"

    trace = [{"rank": i + 1, "label": e["label"], "step": e["step"],
              "retention_pass_rate": e["metrics"]["retention"]["pass_rate"]["candidate"],
              "uncertainty_pass_rate": e["metrics"]["uncertainty"]["pass_rate"]["candidate"],
              "ans_pass_rate": e["metrics"]["uncertainty"]["ans_family"]["candidate"],
              "domain_net": e["metrics"]["domain"]["net_better_minus_worse"],
              "dev_ce_mean": e["dev_ce_mean"], "in_improvement_pool": any(x is e for x in improvement_pool),
              "in_behavior_only_pool": any(x is e for x in behavior_pool)} for i, e in enumerate(orderable)]
    for e in evals:
        del e["_sort"]
    return {
        "rule": {"id": RULE_ID, "path": RULE_PATH, "sha256": RULE_SHA256},
        "pins": {"dev_scenarios_and_rubrics": PINS, "supplied": payload.get("pins", {})},
        "calculator_interpretations": INTERPRETATIONS,
        "checkpoints_evaluated": [{"label": e["label"], "step": e["step"]} for e in evals],
        "valid_denominator": {"common_valid_ids": valid, "n_valid": {g: len(valid[g]) for g in GROUPS},
                              "n_target": {g: sum(1 for s in scen.values() if s["group"] == g) for g in GROUPS},
                              "excluded": excluded},
        "per_checkpoint": {e["label"]: e for e in evals},
        "ordering_trace": {"sort_keys": ["retention pass rate desc", "domain better-worse desc", "dev CE mean asc (evidence tie-break)", "step asc"],
                           "orderable": trace,
                           "not_orderable": [{"label": e["label"], "eligible": e["eligible"], "order_thresholds": e["order_thresholds"]} for e in evals if not e["orderable"]]},
        "selected": chosen["label"] if chosen else "NO_CANDIDATE",
        "selected_step": chosen["step"] if chosen else None,
        "selection_kind": kind,
        "selection_reason": reason,
        "statement": "final24 not opened; selection used data-owner dev scenarios only",
    }


INTERPRETATIONS = {
    "denominator": "common valid ids = verified & non-empty on base AND verified & parity.equal & non-empty on EVERY checkpoint; one id set shared by all checkpoints; a group below 8 fails gate min_valid_denominator",
    "eos": "an item counts as stopped_at_eos only if every turn stopped at EOS; gate uses items/retention n_valid >= 9/10 exactly",
    "severe": "a severe mark must quote one exact listed trigger plus evidence and implies domain score 0 / behavior fail; otherwise the payload is rejected",
    "ordering": "items 1-2 are thresholds (candidate rate >= base; ans rate not below base); failing them makes an eligible checkpoint non-orderable and unselectable; only retention rate is a sort key from items 1-2",
    "improvement": "selection pool = orderable checkpoints with domain better-worse >= 1, ordered by retention rate, domain net, dev CE mean, earlier step",
    "behavior_only": "only when the improvement pool is empty: orderable checkpoints whose retention AND uncertainty pass rates are strictly above base, same order; labelled 'behavior-only candidate'",
    "no_candidate": "no eligible, no orderable, or neither pool non-empty",
    "ce": "dev CE mean = equal-weight mean of the three per-file means; tie-break only (ranked after retention rate and domain net), never a promotion",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="written with open('x'); never overwritten")
    a = ap.parse_args(argv)
    with open(a.input, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    try:
        result = select(payload)
    except GradingError as e:
        sys.stdout.write(json.dumps({"rejected": str(e)}, ensure_ascii=False) + "\n")
        return 2
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    sys.stdout.write(json.dumps({"selected": result["selected"], "selection_kind": result["selection_kind"],
                                 "n_valid": result["valid_denominator"]["n_valid"], "out": a.out}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
