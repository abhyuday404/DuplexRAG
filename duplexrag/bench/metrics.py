"""Metrics for the six evaluation gates of the Theme 4 guide plus retrieval / answer quality.

G1 Reproducibility      - checked by the container smoke test (scripts/smoke_test.sh), not here
G2 Early retrieval      - share of retrieval-eligible turns whose first retrieval starts before the
                          utterance ends (target >= 80%), plus false-trigger rate on no-retrieval turns
G3 Multi-intent         - share of compound turns where >= 2 distinct gold sub-intents are isolated by
                          distinct sub-queries (bipartite matching on retrieved evidence; target >= 70%)
G4 Factual grounding    - share of answer claims supported by their cited chunk; fabricated IDs must be 0
                          (target >= 85%)
G5 Session refinement   - refinement turns that bump the answer version, keep session state, and fetch
                          only the delta (no re-search of earlier intents)
G6 Telemetry            - share of turns whose trace contains every required event (target 100%)
"""
from __future__ import annotations

import re
import statistics

from ..telemetry import trace_coverage

EXPECTED_KIND = {"single": "retrieval", "compound": "retrieval", "unanswerable": "retrieval",
                 "refinement": "refinement", "presentation": "presentation", "chitchat": "chitchat"}


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return re.sub(r"\s+", " ", s)


def _pct(xs: list[float]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 1)


def _evidence(rec: dict, k: int, only_current: bool = True) -> list[list[str]]:
    """Top-k evidence labels per (current-version) intent."""
    out = []
    for it in rec.get("intents", []):
        if it.get("status", "active") != "active":
            continue
        if only_current and rec.get("kind") == "refinement" and it.get("version") != rec.get("version"):
            continue
        out.append([h["label"] for h in it.get("evidence", [])[:k]])
    return out


def _matching(adj: list[set[int]]) -> int:
    """Maximum bipartite matching size (our intents -> gold intents)."""
    match: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match or augment(match[v], seen):
                match[v] = u
                return True
        return False

    return sum(augment(u, set()) for u in range(len(adj)))


def evaluate(records: list[dict], gold_sessions: list[dict], events: list[dict] | None = None) -> dict:
    gold = {(s["session_id"], t["turn_id"]): t for s in gold_sessions for t in s["turns"]}
    by_sess: dict[str, list[dict]] = {}
    for r in records:
        by_sess.setdefault(r["session_id"], []).append(r)

    kind_ok, retrieval_ok = [], []
    confusion: dict[str, dict[str, int]] = {}
    early, early_strict, leads, false_trig = [], [], [], []
    g3_pass, g3_frac, g3_detected = [], [], []
    hit3, recall5, keyfacts = [], [], []
    claims = supported = fabricated = 0
    cite_prec = []
    unans_flagged, false_flag = [], []
    g5 = []
    g5_detail = []
    ttft, post_ms, cost_usd, cpu_ms, tokens = [], [], [], [], []
    reuse = []

    for sid, recs in by_sess.items():
        prev = None
        for r in recs:
            g = gold[(sid, r["turn_id"])]
            exp = EXPECTED_KIND[g["type"]]
            kind_ok.append(r["kind"] == exp)
            confusion.setdefault(exp, {}).setdefault(r["kind"], 0)
            confusion[exp][r["kind"]] += 1
            retrieval_ok.append(bool(r["retrieved"]) == bool(g["retrieval_required"]))
            # ---------------- G2
            if g["retrieval_required"]:
                early.append(bool(r["early"]))
                last_chunk_t = r["chunks"][-1][0] if r.get("chunks") else r["t_end"]
                fr = r["latency"].get("first_retrieval_s")
                early_strict.append(fr is not None and fr < last_chunk_t)
                if r["early"] and r["latency"].get("retrieval_lead_ms") is not None:
                    leads.append(r["latency"]["retrieval_lead_ms"])
            else:
                false_trig.append(bool(r["retrieved"]))
            # ---------------- retrieval quality
            ans_intents = [it for it in g["intents"] if it["answerable"] and it["gold"]]
            if g["retrieval_required"] and ans_intents:
                ev3 = _evidence(r, 3)
                ev5 = _evidence(r, 5)
                u3 = {l for e in ev3 for l in e}
                u5 = {l for e in ev5 for l in e}
                for it in ans_intents:
                    hit3.append(bool(u3 & set(it["gold"])))
                    recall5.append(len(u5 & set(it["gold"])) / len(it["gold"]))
                    for kf in it.get("key_facts", []):
                        keyfacts.append(_norm(kf) in _norm(r["text"]))
                # G3: compound = >= 2 gold intents in the turn
                if g["type"] == "compound" or len(g["intents"]) >= 2:
                    adj = [{j for j, it in enumerate(ans_intents) if set(e) & set(it["gold"])} for e in ev3]
                    m = _matching(adj)
                    need = min(2, len(ans_intents))
                    g3_detected.append(len(ev3) >= 2)
                    g3_pass.append(m >= need and need >= 1)
                    g3_frac.append(m / len(ans_intents))
                gold_labels = {l for it in ans_intents for l in it["gold"]}
                cites = r.get("citations", [])
                if r["kind"] == "retrieval" and cites:      # refinements also carry retained citations
                    cite_prec.append(len(set(cites) & gold_labels) / len(set(cites)))
            # ---------------- uncertainty
            if g["retrieval_required"]:
                has_unc = bool(r.get("uncertainty"))
                if any(not it["answerable"] for it in g["intents"]):
                    unans_flagged.append(has_unc)
                elif g["intents"]:
                    false_flag.append(has_unc)
            # ---------------- G4
            if r["kind"] in ("retrieval", "refinement"):
                gr = r["grounding"]
                claims += gr["claims"]
                supported += gr["supported"]
                fabricated += len(gr.get("fabricated_ids", []))
            # ---------------- G5
            if g["type"] == "refinement":
                d = r.get("diff") or {}
                prev_v = prev["version"] if prev else 0
                prev_q = {q["text"] for q in (prev or {}).get("sub_queries", [])}
                versioned = r["kind"] == "refinement" and r["version"] == prev_v + 1 and d.get("parent_version") == prev_v
                continuity = versioned and d.get("mode") == "refine" and bool(d.get("retained") or d.get("retired"))
                delta_only = r["kind"] == "refinement" and len(r.get("sub_queries", [])) <= 4 and not any(
                    q["text"] in prev_q for q in r.get("sub_queries", []))
                ok = versioned and continuity and delta_only
                g5.append(ok)
                g5_detail.append({"turn": f"{sid}/{r['turn_id']}", "versioned": versioned, "continuity": continuity,
                                  "delta_only": delta_only})
            # ---------------- latency / cost
            if r["kind"] in ("retrieval", "refinement") and r["latency"].get("ttft_ms") is not None:
                ttft.append(r["latency"]["ttft_ms"])
                post_ms.append(r["latency"].get("post_utterance_retrieval_ms", 0.0))
            cost_usd.append(r["cost"]["usd"])
            cpu_ms.append(r["cost"]["cpu_ms"])
            tokens.append(r["cost"].get("embed_tokens", 0) + r["cost"].get("rerank_tokens", 0))
            prev = r if r["kind"] in ("retrieval", "refinement") else prev

    out = {
        "turns": len(records),
        "gate_accuracy_pct": _pct(kind_ok),
        "retrieval_decision_accuracy_pct": _pct(retrieval_ok),
        "turn_kind_confusion": confusion,
        "G2_early_retrieval_pct": _pct(early),
        "G2_early_before_last_chunk_pct": _pct(early_strict),
        "G2_false_trigger_pct": _pct(false_trig),
        "retrieval_lead_ms_median": _q(leads, 0.5),
        "G3_multi_intent_pct": _pct(g3_pass),
        "G3_compound_turns": len(g3_pass),
        "G3_mean_intent_isolation_pct": _pct(g3_frac),
        "G3_detected_2plus_pct": _pct(g3_detected),
        "retrieval_hit_at_3_pct": _pct(hit3),
        "retrieval_recall_at_5_pct": _pct(recall5),
        "gold_intents": len(hit3),
        "key_fact_recall_pct": _pct(keyfacts),
        "citation_precision_pct": _pct(cite_prec),
        "G4_claim_support_pct": round(100 * supported / claims, 1) if claims else None,
        "G4_claims": claims,
        "G4_fabricated_ids": fabricated,
        "unanswerable_flagged_pct": _pct(unans_flagged),
        "false_uncertainty_pct": _pct(false_flag),
        "G5_refinement_pass_pct": _pct(g5),
        "G5_refinement_turns": len(g5),
        "G5_detail": g5_detail,
        "ttft_ms_p50": _q(ttft, 0.5),
        "ttft_ms_p95": _q(ttft, 0.95),
        "ttft_ms_mean": round(statistics.mean(ttft), 1) if ttft else None,
        "post_utterance_retrieval_ms_mean": round(statistics.mean(post_ms), 1) if post_ms else None,
        "cost_usd_per_turn_mean": round(statistics.mean(cost_usd), 8) if cost_usd else None,
        "cpu_ms_per_turn_mean": round(statistics.mean(cpu_ms), 1) if cpu_ms else None,
        "model_tokens_per_turn_mean": round(statistics.mean(tokens), 1) if tokens else None,
    }
    if events is not None:
        tids = [(e["session_id"], e["turn_id"]) for e in events if e["event"] == "turn_started"]
        retr = {(e["session_id"], e["turn_id"]) for e in events if e["event"] == "retrieval_started"}
        cov = trace_coverage(events, tids, retr)
        out["G6_trace_coverage_pct"] = round(100 * cov["coverage"], 1)
        out["G6_missing"] = cov["missing"]
    return out
