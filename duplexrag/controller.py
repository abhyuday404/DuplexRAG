"""Retrieval controller: decides, for every incoming transcript chunk, whether to
WAIT, RETRIEVE (provisional / multi-intent / refinement) or SUPPRESS retrieval.

It has two layers:

* a *turn gate* that classifies what the utterance-so-far is: a new information
  request, a refinement of the current answer (late-arriving constraint), a
  presentation-only request ("repeat that in two bullets") or chit-chat.
  Three interchangeable implementations: ``rule`` (linguistic cues), ``model``
  (a softmax-regression classifier over bge-small embeddings + 14 cue features,
  trained in about 1 second on ``data/controller/train.jsonl``) and ``hybrid``
  (high-precision rules first, model otherwise).
* a *stability detector* (in ``engine.py``) that only lets semantically complete
  clauses - or an in-progress clause that already carries enough salient,
  non-dangling content - reach the retriever, which prevents thrashing on every token.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decompose import Decomposer, normalize_spoken, spoken_count
from .text import content_tokens

LABELS = ("retrieval", "refinement", "presentation", "chitchat")

PRESENTATION_RE = re.compile(
    r"\b(repeat|say (that|it) again|read (that|it) back|come again|(i )?missed (that|the last)|didn'?t catch|"
    r"shorter|shorten|brief(er|ly)?|gist|tl;?dr|bullet|bullets|bullet points|key points|main points|"
    r"summari[sz]e (that|it|this|the (last|previous|above))|sum (that|it) up|rephrase|reword|simplify|simpler|"
    r"in short|recap|one[- ]line(r)?|one sentence|short version|quick version|shorter version|the last (part|bit)|"
    r"as a list|as a table|in plain english|condense)\b", re.I)
DEIXIS_RE = re.compile(r"\b(that|it|this|your (last )?answer|the (last|previous) (answer|part|bit)|the above|"
                       r"what you (just )?said)\b", re.I)
CHITCHAT_RE = re.compile(
    r"^\W*((ok(ay)?|cool|great|perfect|awesome|nice|thanks|thank you( so much| very much)?|cheers|got it|"
    r"understood|alright|all right|sounds good|that'?s (really )?(helpful|great|perfect|all|it|clear|useful)|"
    r"hi|hello|hey( there)?|good (morning|afternoon|evening)|bye|goodbye|see you|no that'?s all|nothing else|"
    r"that'?s everything|brilliant|makes sense|sure|yes|yeah|no|nope|appreciate it|much appreciated|"
    r"that helps|helpful|wonderful|fantastic|excellent|good|fine|so|um|uh|well|for that|really|so much)"
    r"[\s,.!]*)+$", re.I)
GRATITUDE_RE = re.compile(r"\b(thanks|thank you|cheers|appreciate (it|that|the help)|much appreciated|that'?s all|"
                          r"that helps|got it|perfect|brilliant)\b", re.I)
REFINE_CUE_RE = re.compile(
    r"\b(wait|actually|hang on|hold on|by the way|btw|one (more )?thing|forgot|should'?ve (said|mentioned)|"
    r"should have (said|mentioned)|i should (mention|say|add)|full disclosure|to be clear|realistically|what if|"
    r"and if|instead|make (it|that)|change (it|that)|scratch that|correction|turns out|update|just (heard|found out|"
    r"checked|looked|realised|realized|noticed)|looked at the \w+ again|i guess|now that)\b", re.I)
HYPOTHETICAL_RE = re.compile(r"\b(what if|and if|does (that|it) change|would (that|it) change|does that affect|"
                             r"what happens if|in that case|if we|if i|is that (gonna|going to) be a problem|"
                             r"does that matter|is that ok(ay)?|still (apply|allowed|ok))\b", re.I)
STATEMENT_START_RE = re.compile(r"^\W*((oh|wait|actually|and|so|hmm|also|but|by the way|one thing)[,\s]+)*"
                                r"(it's|it is|it'll|it was|we're|we are|we'll|i'm|i am|i'd|i got|i've|i was|the \w+ "
                                r"(is|was|are|were|got|has|had|includes?)|they're|there('s| will be| are)|make it|"
                                r"change it|we (got|had|have|need|moved|changed)|my \w+)", re.I)
QUESTION_RE = re.compile(r"^\W*((oh|and|so|also|okay|ok|hey|hi|um|well)[,\s]+)*(what|what's|how|which|when|where|"
                         r"who|why|is|are|do|does|can|could|should|will|would)\b", re.I)


@dataclass
class GateResult:
    label: str
    confidence: float
    reason: str
    source: str

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3), "reason": self.reason,
                "source": self.source}


def cue_features(text: str, has_prev: bool, n_content: int, n_proper: int, topic_overlap: float) -> np.ndarray:
    t = normalize_spoken(text)
    words = t.split()
    f = [
        1.0,
        float(has_prev),
        min(len(words), 40) / 20.0,
        min(n_content, 8) / 4.0,
        min(n_proper, 4) / 2.0,
        float(bool(PRESENTATION_RE.search(t))) * float(has_prev),
        float(bool(DEIXIS_RE.search(t))),
        float(bool(CHITCHAT_RE.match(t))),
        float(bool(REFINE_CUE_RE.search(text))) * float(has_prev),
        float(bool(HYPOTHETICAL_RE.search(t))) * float(has_prev),
        float(bool(STATEMENT_START_RE.search(t))),
        float(bool(QUESTION_RE.search(t)) or t.endswith("?")),
        float(topic_overlap) * float(has_prev),
        float(spoken_count(t) is not None),
    ]
    return np.array(f, dtype=np.float32)


class RuleGate:
    name = "rule"

    def __init__(self, decomposer: Decomposer):
        self.d = decomposer

    def classify(self, text: str, has_prev: bool, final: bool, prev_topic: set[str] | None = None) -> GateResult:
        t = normalize_spoken(text)
        clauses = self.d.segment(text, final=final)
        content = [x for c in clauses for x in c.content]
        proper = [x for c in clauses for x in c.proper]
        prev = prev_topic or set()
        new_terms = [x for x in dict.fromkeys(content + proper) if x not in prev]
        if has_prev and PRESENTATION_RE.search(t):
            fmt_words = {"bullet", "point", "short", "version", "line", "sentence", "list", "tabl", "part", "bit",
                         "miss", "catch", "last", "key", "main", "put", "give", "make", "turn", "format", "send",
                         "write", "read", "quick", "simple", "plain", "english", "form"}
            new_real = [x for x in new_terms if x not in fmt_words and not x.isdigit()]
            if len(new_real) <= 1 or (spoken_count(t) and len(new_real) <= 2):
                return GateResult("presentation", 0.9, "presentation cue referring to the previous answer", "rule")
        if (CHITCHAT_RE.match(t) or (GRATITUDE_RE.search(t) and len(new_terms) <= 1)) and len(new_terms) <= 1 \
                and not QUESTION_RE.search(t):
            return GateResult("chitchat", 0.9 if final else 0.6, "acknowledgement / small talk", "rule")
        if not content and not proper:
            if final:
                return GateResult("chitchat", 0.9, "no retrievable content", "rule")
            return GateResult("undecided", 0.5, "waiting for content", "rule")
        if has_prev:
            kinds = [c.kind for c in clauses if c.kind != "filler"]
            only_context = bool(kinds) and all(k == "context" for k in kinds)
            head = " ".join(t.split()[:12])
            # a follow-up *question* about something new is a fresh request, not a refinement
            new_question = any(c.kind == "request" and _WH_OR_AUX.search(c.text) and
                               len([x for x in c.content + c.proper if x not in prev]) >= 2 for c in clauses)
            if HYPOTHETICAL_RE.search(t) and not new_question:
                return GateResult("refinement", 0.8, "hypothetical change to the current scenario", "rule")
            if only_context:
                return GateResult("refinement", 0.85, "late-arriving constraint (statement, no new question)", "rule")
            if REFINE_CUE_RE.search(head) and not new_question:
                return GateResult("refinement", 0.8, "refinement cue + constraint", "rule")
        return GateResult("retrieval", 0.8, f"{len(content) + len(proper)} salient terms", "rule")


_WH_OR_AUX = re.compile(r"^(what|what's|how|which|when|where|who|can|could|do|does|is|are|should|will|would)\b|\?$",
                        re.I)


class ModelGate:
    """Softmax regression over [bge-small embedding (384) ; cue features (14)]."""
    name = "model"

    def __init__(self, decomposer: Decomposer, models, settings, train_path: str | None = None):
        self.d = decomposer
        self.models = models
        root = Path(settings.corpus_dir).parents[0]
        self.train_path = Path(train_path or root / "controller" / "train.jsonl")
        self.cache = Path(settings.index_dir) / "controller_gate.npz"
        self.W = None
        self._load_or_train()

    def _features(self, text: str, has_prev: bool, final: bool = True, prev_topic: set[str] | None = None):
        clauses = self.d.segment(text, final=final)
        content = [x for c in clauses for x in c.content]
        proper = [x for c in clauses for x in c.proper]
        overlap = (len(set(content) & prev_topic) / max(1, len(set(content)))) if prev_topic else 0.0
        return cue_features(text, has_prev, len(content), len(proper), overlap)

    def _load_or_train(self) -> None:
        raw = self.train_path.read_bytes()
        from .index import INDEX_VERSION
        digest = hashlib.sha256(raw + str(INDEX_VERSION).encode()).hexdigest()[:16]
        if self.cache.exists():
            z = np.load(self.cache, allow_pickle=False)
            if str(z["digest"]) == digest:
                self.W = z["W"]
                return
        rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        texts = [r["text"] for r in rows]
        emb = self.models.embed_queries(texts)
        feats = np.stack([self._features(r["text"], bool(r.get("prev")), True,
                                         set(content_tokens(r.get("prev_topic", "")))) for r in rows])
        X = np.hstack([emb, feats * 2.0])
        y = np.array([LABELS.index(r["label"]) for r in rows])
        self.W = _train_softmax(X, y, len(LABELS))
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, W=self.W, digest=np.array(digest))

    def probs(self, text: str, has_prev: bool, final: bool, prev_topic: set[str] | None = None) -> np.ndarray:
        emb = self.models.embed_queries([normalize_spoken(text) or text])[0]
        x = np.concatenate([emb, self._features(text, has_prev, final, prev_topic) * 2.0])
        z = x @ self.W
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        if not has_prev:     # refinement / presentation need a previous answer
            p[1] = p[2] = 0.0
            p = p / p.sum()
        return p

    def classify(self, text: str, has_prev: bool, final: bool, prev_topic: set[str] | None = None) -> GateResult:
        clauses = self.d.segment(text, final=final)
        if not any(c.content or c.proper for c in clauses) and not final and not PRESENTATION_RE.search(text):
            return GateResult("undecided", 0.5, "waiting for content", "model")
        p = self.probs(text, has_prev, final, prev_topic)
        k = int(p.argmax())
        return GateResult(LABELS[k], float(p[k]), "p=" + ",".join(f"{l[:4]}:{v:.2f}" for l, v in zip(LABELS, p)),
                          "model")


class HybridGate:
    """High-precision rules decide the easy cases; the classifier arbitrates the rest."""
    name = "hybrid"

    def __init__(self, rule: RuleGate, model: ModelGate):
        self.rule = rule
        self.model = model

    def classify(self, text: str, has_prev: bool, final: bool, prev_topic: set[str] | None = None) -> GateResult:
        r = self.rule.classify(text, has_prev, final, prev_topic)
        if r.label == "undecided" or r.confidence >= 0.85:
            return r
        if not has_prev:
            return r
        m = self.model.classify(text, has_prev, final, prev_topic)
        if m.label == r.label or m.confidence >= 0.75:
            return GateResult(m.label, m.confidence, f"rule={r.label}; model {m.reason}", "hybrid")
        return GateResult(r.label, r.confidence, f"rule={r.label} (model unsure: {m.reason})", "hybrid")


def _train_softmax(X: np.ndarray, y: np.ndarray, k: int, epochs: int = 600, lr: float = 0.5,
                   l2: float = 1e-3) -> np.ndarray:
    n, d = X.shape
    W = np.zeros((d, k), dtype=np.float64)
    Y = np.eye(k)[y]
    # class-balanced weights
    counts = np.bincount(y, minlength=k).astype(float)
    cw = (n / (k * np.maximum(counts, 1)))[y][:, None]
    for _ in range(epochs):
        Z = X @ W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = X.T @ ((P - Y) * cw) / n + l2 * W
        W -= lr * G
    return W.astype(np.float32)


def make_gate(kind: str, decomposer: Decomposer, models, settings):
    rule = RuleGate(decomposer)
    if kind == "rule":
        return rule
    model = ModelGate(decomposer, models, settings)
    if kind == "model":
        return model
    return HybridGate(rule, model)
