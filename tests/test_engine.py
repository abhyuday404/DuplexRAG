"""End-to-end replay tests on the demo corpus (loads the two ONNX models once, ~20 s)."""
import json
from pathlib import Path

import pytest

from duplexrag.engine import DuplexEngine
from duplexrag.stream import replay_session
from duplexrag.telemetry import REQUIRED_PER_TURN

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    return DuplexEngine(log=lambda *a: None)


@pytest.fixture(scope="module")
def workshop(engine):
    sc = [json.loads(l) for l in open(ROOT / "data" / "demo" / "scenarios.jsonl")]
    session = next(s for s in sc if s["session_id"] == "demo-workshop")
    return replay_session(engine, session)


def test_compound_request_is_decomposed_and_retrieved_early(workshop):
    recs, _ = workshop
    t1 = recs[0]
    assert t1["kind"] == "retrieval"
    assert t1["early"], "retrieval must start before the utterance ends"
    assert len([i for i in t1["intents"] if i["status"] == "active"]) >= 3
    assert any(c.startswith("Doc_03") for c in t1["citations"])


def test_every_claim_is_grounded_and_ids_are_real(workshop, engine):
    recs, _ = workshop
    for r in recs:
        g = r["grounding"]
        assert g["fabricated_ids"] == []
        assert g["support_rate"] >= 0.85
        for label in r["citations"]:
            assert label in engine.index.by_label


def test_missing_evidence_is_flagged(workshop):
    recs, _ = workshop
    msgs = " ".join(u["message"] for u in recs[0]["uncertainty"])
    assert "Hinjewadi" in msgs and "could not be verified" in msgs


def test_late_detail_refines_instead_of_restarting(workshop):
    recs, _ = workshop
    t1, t2 = recs[0], recs[1]
    assert t2["kind"] == "refinement"
    assert t2["version"] == t1["version"] + 1
    assert t2["diff"]["mode"] == "refine" and t2["diff"]["retained"]
    assert len(t2["sub_queries"]) <= 2          # delta only, not a full re-search


def test_presentation_turn_suppresses_retrieval(workshop):
    recs, _ = workshop
    t3 = recs[2]
    assert t3["kind"] == "presentation" and not t3["retrieved"]
    assert t3["text"].count("\n- ") + t3["text"].startswith("- ") == 2
    assert set(t3["citations"]) <= set(recs[1]["citations"])   # no new citations invented


def test_trace_has_full_coverage(workshop):
    _, events = workshop
    for tid in ("t1", "t2", "t3"):
        kinds = {e["event"] for e in events if e["turn_id"] == tid}
        assert set(REQUIRED_PER_TURN) <= kinds
    done = [e for e in events if e["event"] == "turn_completed"]
    assert all("cost" in e and "latency" in e for e in done)
