"""Ephemeral, session-scoped memory. Nothing here outlives a session or is written to disk
(no cross-session profiles). Answers are versioned so late details refine instead of restart."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .retrieve import Hit, SubQuery


@dataclass
class Claim:
    cid: str
    intent_id: str
    text: str
    citations: list[str]
    version: int
    status: str = "active"          # active | retired
    support: float | None = None

    def to_dict(self) -> dict:
        return {"cid": self.cid, "intent_id": self.intent_id, "text": self.text, "citations": self.citations,
                "version": self.version, "status": self.status, "support": self.support}


@dataclass
class Intent:
    iid: str
    label: str
    query: SubQuery
    evidence: list[Hit] = field(default_factory=list)
    version: int = 1
    status: str = "active"          # active | retired
    uncertain: str | None = None    # reason when the corpus lacks evidence

    def to_dict(self) -> dict:
        return {"iid": self.iid, "label": self.label, "query": self.query.to_dict(),
                "evidence": [h.to_dict() for h in self.evidence], "version": self.version, "status": self.status,
                "uncertain": self.uncertain}


@dataclass
class AnswerState:
    version: int = 0
    turn_id: str | None = None
    intents: dict[str, Intent] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    uncertainty: list[dict] = field(default_factory=list)
    text: str = ""
    view: str = "full"               # full | bullets:N | short (presentation-only transforms)

    def active_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "active"]

    def citations(self) -> list[str]:
        return list(dict.fromkeys(l for c in self.active_claims() for l in c.citations))

    def snapshot(self) -> dict:
        return {"version": self.version, "turn_id": self.turn_id, "text": self.text, "view": self.view,
                "citations": self.citations(), "claims": [c.to_dict() for c in self.claims],
                "intents": {k: v.to_dict() for k, v in self.intents.items()}, "uncertainty": self.uncertainty}


@dataclass
class Session:
    session_id: str
    started: float = field(default_factory=time.time)
    answer: AnswerState = field(default_factory=AnswerState)
    history: list[dict] = field(default_factory=list)      # answer snapshots per version
    ctx: dict = field(default_factory=dict)                # inherited entities/topic (session only)
    turns: list[dict] = field(default_factory=list)
    intent_seq: int = 0
    claim_seq: int = 0

    def next_intent_id(self) -> str:
        self.intent_seq += 1
        return f"i{self.intent_seq}"

    def next_claim_id(self) -> str:
        self.claim_seq += 1
        return f"c{self.claim_seq}"

    @property
    def has_answer(self) -> bool:
        return self.answer.version > 0 and bool(self.answer.text)

    def merge_ctx(self, new: dict) -> None:
        """Newer entities first; older ones kept so a refinement does not erase the topic."""
        for key in ("proper", "topic"):
            merged = list(dict.fromkeys(list(new.get(key, [])) + list(self.ctx.get(key, []))))
            self.ctx[key] = merged[:5]
        if new.get("people"):
            self.ctx["people"] = new["people"]
        self.ctx["surface"] = (list(new.get("surface", [])) + list(self.ctx.get("surface", [])))[:8]
