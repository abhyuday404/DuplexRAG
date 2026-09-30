"""Structured, append-only telemetry (JSON Lines). Schema: schemas/telemetry.schema.json.

Every event carries both the *stream clock* (seconds since session start, the
same clock as transcript timestamps) and the wall clock, so traces from the
real-time server and from the discrete-event replay are directly comparable.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

EVENT_TYPES = (
    "session_started", "turn_started", "chunk_received", "controller_decision", "decomposition",
    "retrieval_started", "retrieval_completed", "evidence_fused", "retrieval_suppressed", "answer_started",
    "first_token", "answer_version", "grounding_check", "uncertainty_flag", "turn_completed", "session_ended",
)

REQUIRED_PER_TURN = ("turn_started", "chunk_received", "controller_decision", "answer_version", "grounding_check",
                     "turn_completed")


class Tracer:
    def __init__(self, session_id: str, sink_path: str | None = None, listener=None):
        self.session_id = session_id
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._sink = open(sink_path, "a", encoding="utf-8") if sink_path else None
        self.listener = listener
        self._seq = 0

    def emit(self, event: str, t: float, turn_id: str | None = None, **payload) -> dict:
        assert event in EVENT_TYPES, event
        with self._lock:
            self._seq += 1
            rec = {"seq": self._seq, "event": event, "session_id": self.session_id, "turn_id": turn_id,
                   "t_stream": round(float(t), 4), "t_wall": round(time.time(), 4), **payload}
            self.events.append(rec)
            if self._sink:
                self._sink.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")
                self._sink.flush()
        if self.listener:
            self.listener(rec)
        return rec

    def turn_events(self, turn_id: str) -> list[dict]:
        return [e for e in self.events if e.get("turn_id") == turn_id]

    def close(self) -> None:
        if self._sink:
            self._sink.close()
            self._sink = None


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:  # pragma: no cover
        pass
    return str(o)


def trace_coverage(events: list[dict], turn_ids: list[tuple[str, str]], retrieval_turns: set) -> dict:
    """G6: fraction of turns whose trace contains every required event type (+ retrieval events
    for turns that retrieved) with timestamps, versions and cost fields. Turns are keyed by
    (session_id, turn_id)."""
    ok = 0
    missing: dict[str, list[str]] = {}
    by_turn: dict[tuple, list[dict]] = {}
    for e in events:
        by_turn.setdefault((e.get("session_id"), e.get("turn_id")), []).append(e)
    for tid in turn_ids:
        evs = by_turn.get(tid, [])
        kinds = {e["event"] for e in evs}
        need = list(REQUIRED_PER_TURN)
        if tid in retrieval_turns:
            need += ["retrieval_started", "retrieval_completed"]
        miss = [k for k in need if k not in kinds]
        done = next((e for e in evs if e["event"] == "turn_completed"), None)
        if done is None or "cost" not in done or "latency" not in done:
            miss.append("turn_completed.cost/latency")
        if not all("t_stream" in e and "t_wall" in e for e in evs):
            miss.append("timestamps")
        if miss:
            missing["/".join(map(str, tid))] = miss
        else:
            ok += 1
    return {"coverage": ok / max(1, len(turn_ids)), "turns": len(turn_ids), "missing": missing}


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
