"""Incremental transcript simulator + replay driver.

Voice input is simulated from transcripts (allowed by the brief): each utterance is
emitted as ASR-style partial chunks of 2-4 words at a conversational speaking rate
(2.7 words/s, about 160 wpm) with a 350 ms end-of-utterance silence, deterministically
seeded per turn. Turns may also carry explicit ``chunks: [{"t": 0.0, "text": ...}]``.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

WORDS_PER_SEC = 2.7
ENDPOINT_SILENCE = 0.35


@dataclass
class TimedChunk:
    t: float
    text: str


def chunk_utterance(utterance: str, seed: str, wps: float = WORDS_PER_SEC) -> tuple[list[TimedChunk], float]:
    rng = random.Random(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    words = utterance.split()
    chunks, i, t = [], 0, 0.0
    while i < len(words):
        n = rng.choice((2, 3, 3, 4))
        part = words[i:i + n]
        i += len(part)
        t = i / wps * rng.uniform(0.92, 1.08)
        chunks.append(TimedChunk(round(t, 3), " ".join(part)))
    # keep time monotonic after jitter
    for k in range(1, len(chunks)):
        if chunks[k].t <= chunks[k - 1].t:
            chunks[k].t = round(chunks[k - 1].t + 0.05, 3)
    end = round((chunks[-1].t if chunks else 0.0) + ENDPOINT_SILENCE, 3)
    return chunks, end


def turn_chunks(turn: dict, session_id: str) -> tuple[list[TimedChunk], float]:
    if turn.get("chunks"):
        ch = [TimedChunk(float(c["t"]), c["text"]) for c in turn["chunks"]]
        end = float(turn.get("end_t", ch[-1].t + ENDPOINT_SILENCE))
        return ch, end
    return chunk_utterance(turn["utterance"], f"{session_id}/{turn['turn_id']}")


def replay_session(engine, session: dict, *, trace_path: str | None = None, on_turn=None
                   ) -> tuple[list[dict], list[dict]]:
    """Replay one benchmark/demo session through the engine on the simulated stream clock.
    Returns (turn records, telemetry events)."""
    sh = engine.new_session(session.get("session_id"), realtime=False, trace_path=trace_path)
    offset = 0.5
    out = []
    for turn in session["turns"]:
        chunks, end = turn_chunks(turn, sh.session.session_id)
        base = max(offset, sh.executor.free_at + 0.25)
        engine.start_turn(sh, turn["turn_id"], base)
        for c in chunks:
            engine.on_chunk(sh, c.text, base + c.t)
        res = engine.end_turn(sh, base + end)
        rec = {"session_id": sh.session.session_id, "turn_id": turn["turn_id"], "t_start": base,
               "t_end": base + end, "chunks": [(round(base + c.t, 3), c.text) for c in chunks], **res.to_dict()}
        out.append(rec)
        if on_turn:
            on_turn(rec)
        offset = base + end + 2.5 + len(res.text.split()) / 3.0     # time the assistant spends speaking
    engine.end_session(sh, offset)
    return out, sh.tracer.events
