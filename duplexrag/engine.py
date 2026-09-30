"""DuplexRAG engine: event-driven streaming RAG over incremental transcript chunks.

    chunk -> [1] controller (turn gate + stability) -> [2] decomposer -> [3] hybrid retrieval,
    RRF fusion, cross-encoder rerank  (all *while the user is still speaking*)
    utterance end -> reuse speculative results, fetch only what is new -> [4] session-aware
    synthesis (versioned answer, delta refinement, presentation transforms), grounding check,
    uncertainty flags -> streamed answer + citations + telemetry

The same code runs against two clocks:
  * ``RealtimeExecutor`` - background worker thread, wall clock (web demo / live mic);
  * ``SimExecutor`` - discrete-event replay: every retrieval job really runs (its CPU time is
    measured) and is placed on a single virtual worker at stream time, which gives exact,
    reproducible TTFT / early-retrieval numbers for benchmark replays.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .config import Settings, load_settings
from .controller import GateResult, make_gate
from .decompose import Decomposer, ends_dangling, normalize_spoken, spoken_count
from .grounding import verify
from .index import HybridIndex
from .models import Models
from .retrieve import Hit, Retriever, SubQuery, fuse_across_queries
from .session import AnswerState, Intent, Session
from .synthesize import Composer
from .telemetry import Tracer, new_id
from .text import content_tokens

REPLACE_CUE = re.compile(r"\b(instead|now|moved|changed|change it|change that|not|rather|switch(ed)?|no longer|"
                         r"actually no|scratch that|correction|make it)\b", re.I)


# ============================================================================ executors
@dataclass
class Job:
    t_dispatch: float
    t_start: float = 0.0
    t_end: float = 0.0
    compute_ms: float = 0.0
    result: object = None
    _future: object = None

    def wait(self):
        if self._future is not None:
            self.result = self._future.result()
            self._future = None
        return self.result


class SimExecutor:
    """Single virtual worker. Jobs execute immediately (so results are real) but their start /
    end are placed on the stream clock: start = max(dispatch, worker_free), end = start + cpu."""

    realtime = False

    def __init__(self):
        self.free_at = 0.0

    def submit(self, fn, t_dispatch: float) -> Job:
        t0 = time.perf_counter()
        res = fn()
        dur = time.perf_counter() - t0
        start = max(t_dispatch, self.free_at)
        job = Job(t_dispatch, start, start + dur, dur * 1000, res)
        self.free_at = job.t_end
        return job

    def charge(self, t: float, seconds: float) -> float:
        """Account for synchronous work done at stream time t; returns completion time."""
        start = max(t, self.free_at)
        self.free_at = start + seconds
        return self.free_at


class RealtimeExecutor:
    realtime = True

    def __init__(self, t0: float | None = None):
        self.t0 = t0 if t0 is not None else time.monotonic()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def submit(self, fn, t_dispatch: float) -> Job:
        job = Job(t_dispatch)

        def run():
            job.t_start = self.now()
            t0 = time.perf_counter()
            r = fn()
            job.compute_ms = (time.perf_counter() - t0) * 1000
            job.t_end = self.now()
            return r

        job._future = self.pool.submit(run)
        return job

    def charge(self, t: float, seconds: float) -> float:
        return self.now()


# ============================================================================ turn state
@dataclass
class Dispatch:
    query: SubQuery
    job: Job
    vec: np.ndarray
    used: bool = False


@dataclass
class TurnState:
    turn_id: str
    t_start: float
    text: str = ""
    chunks: list = field(default_factory=list)
    gate: GateResult | None = None
    dispatches: list[Dispatch] = field(default_factory=list)
    provisional_keys: set = field(default_factory=set)
    decisions: int = 0
    first_retrieval_t: float | None = None
    stats_before: dict = field(default_factory=dict)
    controller_ms: float = 0.0


@dataclass
class TurnResult:
    turn_id: str
    kind: str                       # retrieval | refinement | presentation | chitchat
    text: str
    version: int
    citations: list[str]
    intents: list[dict]
    sub_queries: list[dict]
    uncertainty: list[dict]
    grounding: dict
    latency: dict
    cost: dict
    retrieved: bool
    early: bool
    diff: dict | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SessionHandle:
    def __init__(self, engine: "DuplexEngine", session: Session, tracer: Tracer, executor):
        self.engine = engine
        self.session = session
        self.tracer = tracer
        self.executor = executor
        self.turn: TurnState | None = None
        self.lock = threading.RLock()

    def now(self) -> float:
        return self.executor.now() if self.executor.realtime else 0.0


# ============================================================================ engine
class DuplexEngine:
    def __init__(self, settings: Settings | None = None, *, decompose: bool = True, speculative: bool = True,
                 log=print, **overrides):
        self.s = (settings or load_settings()).with_overrides(**{k: v for k, v in overrides.items() if v is not None})
        self.decompose_enabled = decompose
        self.speculative = speculative
        self.models = Models.get(self.s)
        self.index = HybridIndex.load_or_build(self.s, self.models, log=log)
        self.retriever = Retriever(self.index, self.models, self.s)
        self.decomposer = Decomposer(self.index, self.s)
        self.composer = Composer(self.index, self.models, self.s)
        self.gate = make_gate(self.s.controller, self.decomposer, self.models, self.s)
        self.llm = None
        if self.s.synthesis == "llm":
            from .llm import LLMSynthesizer
            self.llm = LLMSynthesizer(self.s)

    # ------------------------------------------------------------------ sessions
    def new_session(self, session_id: str | None = None, *, realtime: bool = False, trace_path: str | None = None,
                    listener=None) -> SessionHandle:
        sid = session_id or new_id("sess")
        tracer = Tracer(sid, trace_path, listener)
        ex = RealtimeExecutor() if realtime else SimExecutor()
        sh = SessionHandle(self, Session(sid), tracer, ex)
        tracer.emit("session_started", 0.0, None, config=self.config_summary())
        return sh

    def config_summary(self) -> dict:
        return {"controller": self.s.controller, "retrieval": self.s.retrieval_mode, "rerank": self.s.rerank,
                "decompose": self.decompose_enabled, "speculative": self.speculative, "synthesis": self.s.synthesis,
                "embed_model": self.s.embed_model, "rerank_model": self.s.rerank_model,
                "corpus_chunks": len(self.index.chunks)}

    def end_session(self, sh: SessionHandle, t: float) -> None:
        sh.tracer.emit("session_ended", t, None, versions=sh.session.answer.version)
        sh.tracer.close()

    # ------------------------------------------------------------------ helpers
    def _prev_topic(self, session: Session) -> set[str]:
        terms = set(session.ctx.get("topic", [])) | set(session.ctx.get("proper", []))
        for i in session.answer.intents.values():
            terms |= set(content_tokens(i.query.text))
        return terms

    def _classify(self, sh: SessionHandle, text: str, final: bool) -> GateResult:
        return self.gate.classify(text, sh.session.has_answer, final, self._prev_topic(sh.session))

    def _key(self, q: SubQuery) -> str:
        return " ".join(sorted(set(content_tokens(q.text))))

    def _hard_terms(self, text: str) -> set[str]:
        """Entities and numbers: constraints a reused speculative search must already contain."""
        _, proper = self.decomposer.salient_terms(text)
        return set(proper) | {t for t in content_tokens(text) if t.isdigit()}

    def _dispatch(self, sh: SessionHandle, queries: list[SubQuery], t: float, trigger: str) -> None:
        turn = sh.turn
        for q in queries:
            q.trigger = trigger
        vecs = self.models.embed_queries([q.text for q in queries])
        job = sh.executor.submit(lambda qs=list(queries): self.retriever.retrieve(qs), t)
        for q, v in zip(queries, vecs):
            turn.dispatches.append(Dispatch(q, job, v))
            sh.tracer.emit("retrieval_started", t, turn.turn_id, qid=q.qid, query=q.text, keywords=q.keywords,
                           trigger=trigger, origin=q.origin)
        if turn.first_retrieval_t is None:
            turn.first_retrieval_t = t

    def _new_queries(self, sh: SessionHandle, cands: list[SubQuery]) -> list[SubQuery]:
        """Drop candidates already dispatched this turn (same terms or near-identical meaning)."""
        turn = sh.turn
        if not cands:
            return []
        keys = {self._key(d.query) for d in turn.dispatches}
        vecs = self.models.embed_queries([c.text for c in cands])
        out = []
        for c, v in zip(cands, vecs):
            if self._key(c) in keys:
                continue
            hard = self._hard_terms(c.text)
            if any(float(d.vec @ v) >= 0.93 and hard <= self._hard_terms(d.query.text) for d in turn.dispatches):
                continue
            keys.add(self._key(c))
            out.append(c)
        return out

    # ------------------------------------------------------------------ streaming input
    def start_turn(self, sh: SessionHandle, turn_id: str, t: float) -> None:
        with sh.lock:
            sh.turn = TurnState(turn_id, t, stats_before=self.models.stats.snapshot())
            sh.tracer.emit("turn_started", t, turn_id)

    def on_chunk(self, sh: SessionHandle, text: str, t: float, *, cumulative: bool = False) -> dict:
        """Feed one transcript chunk (new words, or the full partial transcript if cumulative)."""
        with sh.lock:
            turn = sh.turn
            c0 = time.perf_counter()
            turn.text = text if cumulative else (turn.text + " " + text).strip()
            turn.chunks.append((t, text))
            sh.tracer.emit("chunk_received", t, turn.turn_id, text=text, partial=turn.text)
            gate = self._classify(sh, turn.text, final=False)
            turn.gate = gate
            action, reason, dispatched = "wait", gate.reason, []
            if gate.label in ("presentation", "chitchat"):
                action = "suppress"
            elif not self.speculative and gate.label in ("retrieval", "refinement"):
                reason = "speculative retrieval disabled (ablation): waiting for utterance end"
            elif gate.label == "retrieval" and self.decompose_enabled:
                clauses = self.decomposer.segment(turn.text, final=False)
                cands = self.decomposer.subqueries(clauses, sh.session.ctx, include_incomplete=False,
                                                   id_prefix=f"{turn.turn_id}q{len(turn.dispatches)}_")
                trigger = "multi_intent" if len([c for c in clauses if c.kind == "request"]) > 1 else "provisional"
                # an in-progress clause may fire a provisional search once it is semantically stable
                tail = clauses[-1] if clauses else None
                if tail is not None and not tail.complete and tail.kind in ("request", "context") and \
                        not ends_dangling(tail.text) and (len(tail.content) + 2 * len(tail.proper) >= 4):
                    prov = self.decomposer.subqueries(clauses, sh.session.ctx, include_incomplete=True,
                                                      id_prefix=f"{turn.turn_id}p{len(turn.dispatches)}_")
                    prov = [q for q in prov if q.origin == tail.text or tail.text in q.origin]
                    sig = frozenset(tail.content + tail.proper)
                    if prov and not any(len(sig - k) < 2 for k in turn.provisional_keys):
                        turn.provisional_keys.add(sig)
                        cands += prov
                        trigger = "provisional" if not turn.dispatches else trigger
                new = self._new_queries(sh, cands)
                if new:
                    action, dispatched = "retrieve", new
                    reason = f"{len(new)} new stable intent(s)"
                    self._dispatch(sh, new, t, trigger)
                else:
                    reason = "no new stable intent yet" if gate.label == "retrieval" else reason
            elif gate.label == "refinement" and sh.session.has_answer and self.decompose_enabled:
                # speculative *delta* retrieval: fire as soon as a constraint clause is complete
                deltas, _ = self._refinement_deltas(sh, turn.text, final=False)
                new = self._new_queries(sh, [q for _, _, q in deltas])
                if new:
                    action, dispatched = "retrieve", new
                    reason = f"refinement: {len(new)} delta quer{'y' if len(new) == 1 else 'ies'}"
                    self._dispatch(sh, new, t, "refinement")
                else:
                    reason = "refinement detected; waiting for a complete constraint"
            turn.controller_ms += (time.perf_counter() - c0) * 1000
            turn.decisions += 1
            ev = sh.tracer.emit("controller_decision", t, turn.turn_id, action=action, gate=gate.to_dict(),
                                reason=reason, queries=[q.text for q in dispatched], final=False)
            return ev

    # ------------------------------------------------------------------ utterance end
    def end_turn(self, sh: SessionHandle, t_end: float) -> TurnResult:
        with sh.lock:
            turn = sh.turn
            s = sh.session
            c0 = time.perf_counter()
            gate = self._classify(sh, turn.text, final=True)
            if not self.decompose_enabled and gate.label == "refinement":
                gate = GateResult("retrieval", gate.confidence, "decomposition disabled", gate.source)
            sh.tracer.emit("controller_decision", t_end, turn.turn_id, action="finalize", gate=gate.to_dict(),
                           reason=gate.reason, final=True)
            if gate.label in ("presentation", "chitchat"):
                return self._finish_suppressed(sh, gate, t_end, c0)
            if gate.label == "refinement" and s.has_answer:
                return self._finish_refinement(sh, gate, t_end, c0)
            return self._finish_retrieval(sh, gate, t_end, c0)

    # .................................................................. retrieval turn
    def _final_queries(self, sh: SessionHandle) -> tuple[list[SubQuery], list]:
        turn = sh.turn
        clauses = self.decomposer.segment(turn.text, final=True)
        if not self.decompose_enabled:
            q = SubQuery(f"{turn.turn_id}q1", normalize_spoken(turn.text), origin=turn.text, anchor=True)
            q.keywords = " ".join(content_tokens(q.text))
            return [q], clauses
        qs = self.decomposer.subqueries(clauses, sh.session.ctx, id_prefix=f"{turn.turn_id}q")
        if not qs:   # nothing request-like: fall back to the whole normalised utterance
            q = SubQuery(f"{turn.turn_id}q1", normalize_spoken(turn.text), origin=turn.text, anchor=True)
            q.keywords = " ".join(content_tokens(q.text))
            qs = [q]
        return self._merge_fragments(qs), clauses

    def _merge_fragments(self, qs: list[SubQuery]) -> list[SubQuery]:
        """Over-fragmentation guard: merge near-identical sub-queries; cap the fan-out."""
        if len(qs) <= 1:
            return qs
        vecs = self.models.embed_queries([q.text for q in qs])
        keep: list[int] = []
        for i in range(len(qs)):
            dup = next((k for k in keep if float(vecs[k] @ vecs[i]) >= self.s.dedup_cosine), None)
            if dup is None:
                keep.append(i)
            else:
                qs[dup].origin += " | " + qs[i].origin
        out = [qs[k] for k in keep]
        while len(out) > self.s.max_subqueries:
            # merge the least specific query into its nearest neighbour
            spec = [len(content_tokens(q.origin)) for q in out]
            j = int(np.argmin(spec))
            v = self.models.embed_queries([q.text for q in out])
            sims = v @ v[j]
            sims[j] = -1
            k = int(np.argmax(sims))
            out[k].text += " " + out[j].origin
            out[k].origin += " | " + out[j].origin
            out.pop(j)
        return out

    def _resolve(self, sh: SessionHandle, finals: list[SubQuery], t_end: float) -> tuple[dict, list[Job], int]:
        """Map final sub-queries onto speculative results; retrieve only what is missing."""
        turn = sh.turn
        results: dict[str, list[Hit]] = {}
        jobs: list[Job] = []
        missing: list[SubQuery] = []
        reused = 0
        if turn.dispatches:
            fvecs = self.models.embed_queries([q.text for q in finals])
        for i, q in enumerate(finals):
            match = None
            hard = self._hard_terms(q.text)
            for d in turn.dispatches:
                # reuse a speculative result only if it already knew every entity / number of the final
                # query: a late "...oh and it's in Bengaluru" must invalidate a Bengaluru-less search
                if self._key(d.query) == self._key(q) or (float(d.vec @ fvecs[i]) >= 0.93
                                                          and hard <= self._hard_terms(d.query.text)):
                    match = d
                    break
            if match is not None:
                res, _ = match.job.wait()
                results[q.qid] = [Hit(**{**h.__dict__, "qid": q.qid}) for h in res[match.query.qid]]
                match.used = True
                jobs.append(match.job)
                reused += 1
            else:
                missing.append(q)
        if missing:
            job = sh.executor.submit(lambda qs=list(missing): self.retriever.retrieve(qs), t_end)
            for q in missing:
                q.trigger = "final"
                sh.tracer.emit("retrieval_started", t_end, turn.turn_id, qid=q.qid, query=q.text,
                               keywords=q.keywords, trigger="final", origin=q.origin)
            res, _ = job.wait()
            for q in missing:
                results[q.qid] = res[q.qid]
            jobs.append(job)
            if turn.first_retrieval_t is None:
                turn.first_retrieval_t = t_end
        for d in turn.dispatches:
            d.job.wait()
        return results, jobs, reused

    def _finish_retrieval(self, sh: SessionHandle, gate: GateResult, t_end: float, c0: float) -> TurnResult:
        turn, s = sh.turn, sh.session
        finals, clauses = self._final_queries(sh)
        sh.tracer.emit("decomposition", t_end, turn.turn_id, clauses=[c.to_dict() for c in clauses],
                       sub_queries=[q.to_dict() for q in finals])
        results, jobs, reused = self._resolve(sh, finals, t_end)
        self._emit_retrieval_done(sh, jobs, results, finals)
        fused = fuse_across_queries(results, self.s.rrf_k, self.s.evidence_per_query)
        sh.tracer.emit("evidence_fused", t_end, turn.turn_id, chunks=[h.label for h in fused[:12]],
                       per_query={qid: [h.label for h in hs[:self.s.evidence_per_query]] for qid, hs in results.items()})
        # new answer (new topic) -> fresh answer state, version continues the session lineage
        prev_version = s.answer.version
        if s.has_answer:
            s.history.append(s.answer.snapshot())
        ans = AnswerState(version=prev_version + 1, turn_id=turn.turn_id)
        for q in finals:
            iid = s.next_intent_id()
            ans.intents[iid] = Intent(iid, self.decomposer.label(q.origin.split(" | ")[0]), q,
                                      results.get(q.qid, []), ans.version)
        seen: dict[str, str] = {}        # claim text -> intent that already states it
        for it in list(ans.intents.values()):
            claims = self.composer.compose_intent(s, it, ans.version)
            fresh = [c for c in claims if c.text not in seen]
            if claims and not fresh:
                # every claim duplicates an earlier intent: it is the same need, fold it in
                owner = ans.intents[seen[claims[0].text]]
                owner.label = f"{owner.label} / {it.label}" if it.label.lower() not in owner.label.lower() \
                    else owner.label
                owner.query.origin += " | " + it.query.origin
                del ans.intents[it.iid]
                continue
            for c in fresh:
                seen[c.text] = it.iid
            ans.claims += fresh
        self._coverage(ans)
        ans.text = self.composer.render(ans)
        s.answer = ans
        s.merge_ctx(self.decomposer.session_context(clauses))
        diff = {"parent_version": prev_version, "mode": "new_answer",
                "added": [c.cid for c in ans.active_claims()], "retired": [], "retained": []}
        return self._complete(sh, "retrieval", gate, t_end, c0, jobs, finals, diff, reused)

    def _coverage(self, ans: AnswerState) -> None:
        intents = [i for i in ans.intents.values() if i.status == "active"]
        if len(intents) < 2:
            return
        anchor = next((i for i in intents if i.query.anchor), intents[0])
        anchor_claims = [c for c in ans.active_claims() if c.intent_id == anchor.iid]
        flags = self.composer.coverage_gaps(anchor, [i for i in intents if i is not anchor], anchor_claims)
        known = {(u["intent_id"], u.get("doc_id")) for u in ans.uncertainty}
        ans.uncertainty += [f for f in flags if (f["intent_id"], f.get("doc_id")) not in known]

    # .................................................................. refinement turn
    def _refinement_deltas(self, sh: SessionHandle, text: str, final: bool
                           ) -> tuple[list[tuple[str, Intent | None, SubQuery]], list]:
        """Turn a late-arriving detail into *delta* sub-queries against the current answer.

        A detail *replaces* part of an existing intent only when it restates the same constraint
        (a new head-count, or an explicit change: "moved to Germany now", "not a birth"); otherwise
        it *adds* a delta intent and every earlier claim is retained ("the standard rule still
        applies"). Only complete clauses are used unless ``final``.
        """
        turn, s = sh.turn, sh.session
        ans = s.answer
        clauses = [c for c in self.decomposer.segment(text, final=final) if c.kind != "filler"]
        usable = [c for c in clauses if c.complete and (c.content or c.proper or c.people)]
        anchor_terms = list(dict.fromkeys(list(s.ctx.get("proper", []))[:3] + list(s.ctx.get("topic", []))[:3]))
        active = [i for i in ans.intents.values() if i.status == "active"]
        ivecs = self.models.embed_queries([i.query.text for i in active]) if active else np.zeros((0, 384))
        replace = bool(REPLACE_CUE.search(text))
        new_proper = {p for c in clauses for p in c.proper}
        deltas: list[tuple[str, Intent | None, SubQuery]] = []
        surface = {}
        for w in re.findall(r"[A-Za-z][A-Za-z0-9-]*", text):
            surface.setdefault(content_tokens(w)[0] if content_tokens(w) else w.lower(), w)
        for k, c in enumerate(usable):
            body = self.decomposer.clean_request(c.text)
            # the delta's own informative terms (entities + content words), in surface form
            terms = [surface.get(t, t) for t in list(c.proper) + [t for t in c.content if t not in c.proper]]
            probe0 = f"{body} {' '.join(t for t in anchor_terms if t not in content_tokens(body))}".strip()
            pv = self.models.embed_queries([probe0])[0]
            sims = ivecs @ pv if len(active) else np.array([])
            nearest = active[int(np.argmax(sims))] if len(sims) else None
            target = None
            if c.people and active:
                target = next((i for i in active if i.query.constraints.get("people")), None)
            elif replace and nearest is not None and float(sims.max()) >= 0.70:
                target = nearest
            if target is not None:
                q = SubQuery(f"{turn.turn_id}r{k + 1}", target.query.text, origin=c.text, trigger="refinement",
                             anchor=target.query.anchor)
                if c.people:
                    q.constraints["people"] = c.people
                else:
                    q.text += " " + " ".join(t for t in terms if not t.isdigit())
                deltas.append(("modify", target, q))
            else:
                # additive detail: ask it *in the context of* the closest existing intent
                ctx_q = nearest.query.text if nearest is not None and float(sims.max()) >= 0.35 else \
                    " ".join(t for t in anchor_terms if t not in content_tokens(body))
                deltas.append(("add", None, SubQuery(f"{turn.turn_id}r{k + 1}", f"{body} {ctx_q}".strip(),
                                                     origin=c.text, trigger="refinement")))
        # merge deltas aimed at the same intent
        merged: dict[str, tuple[str, Intent | None, SubQuery]] = {}
        for mode, target, q in deltas:
            key = target.iid if target is not None else self._key(q)
            if key in merged and target is not None:
                mq = merged[key][2]
                extra = q.text[len(target.query.text):].strip()
                if extra:
                    mq.text += " " + extra
                mq.constraints.update(q.constraints)
                mq.origin += " | " + q.origin
            else:
                merged[key] = (mode, target, q)
        out = list(merged.values())[: self.s.max_subqueries]
        for mode, target, q in out:
            if mode == "modify":
                q.constraints = {**target.query.constraints, **q.constraints}
                if replace and new_proper:   # "it's going to be Germany now" replaces "Japan"
                    for p in set(content_tokens(target.query.text)):
                        if (p in self.index.proper_terms or p in s.ctx.get("proper", [])) and p not in new_proper:
                            q.text = re.sub(rf"\b{re.escape(p)}\w*\b", "", q.text, flags=re.I)
                    q.text = re.sub(r"\b(to|in|at|for)\s+(?=(to|in|at|for)\b|$)", "", q.text)
                n = q.constraints.get("people")
                if n:                        # rebuild the head-count facet
                    q.text = re.sub(r"room or venue capacity for \d+ people:\s*|capacity for \d+ people", "", q.text)
                    q.text = re.sub(r"\b(about |around )?\d+\s+(people|attendees|persons|guests)\b", "", q.text)
                    q.text = f"room or venue capacity for {n} people: {q.text}"
            q.text = re.sub(r"\s+", " ", q.text).strip()
            q.keywords = " ".join(dict.fromkeys(content_tokens(q.text)))
        return out, clauses

    def _finish_refinement(self, sh: SessionHandle, gate: GateResult, t_end: float, c0: float) -> TurnResult:
        turn, s = sh.turn, sh.session
        ans = s.answer
        s.history.append(ans.snapshot())
        constraint = re.sub(r"^((oh|wait|actually|and|so|hmm|also|okay|ok|well|one thing|no|i forgot|"
                            r"i should mention)[,.\s]+)+", "", normalize_spoken(turn.text), flags=re.I)
        deltas, clauses = self._refinement_deltas(sh, turn.text, final=True)
        if not deltas:   # nothing retrievable in the refinement: fall back to a normal retrieval turn
            s.history.pop()
            return self._finish_retrieval(sh, gate, t_end, c0)
        queries = [q for _, _, q in deltas]
        sh.tracer.emit("decomposition", t_end, turn.turn_id, clauses=[c.to_dict() for c in clauses],
                       sub_queries=[q.to_dict() for q in queries], scope="delta",
                       targets=[{"mode": m, "intent": (t.iid if t else None)} for m, t, _ in deltas])
        results, jobs, reused = self._resolve(sh, queries, t_end)
        self._emit_retrieval_done(sh, jobs, results, queries)
        prev_version = ans.version
        ans.version += 1
        ans.turn_id = turn.turn_id
        added, retired, delta_ids = [], [], []
        for mode, target, q in deltas:
            if mode == "modify":
                old = [c for c in ans.active_claims() if c.intent_id == target.iid]
                label = target.label
                if q.constraints.get("people"):
                    label = re.sub(r"\b\d+(?=\s+(people|attendees|persons|guests))", str(q.constraints["people"]),
                                   label)
                newi = Intent(target.iid, label, q, results.get(q.qid, []), ans.version)
                new_claims = self.composer.compose_intent(s, newi, ans.version)
                if new_claims or newi.uncertain:
                    for c in old:
                        c.status = "retired"
                        retired.append(c.cid)
                    ans.intents[target.iid] = newi
                    ans.claims += new_claims
                    added += [c.cid for c in new_claims]
                    delta_ids.append(target.iid)
            else:
                iid = s.next_intent_id()
                it = Intent(iid, self.decomposer.label(q.origin.split(" | ")[0]), q, results.get(q.qid, []),
                            ans.version)
                new_claims = self.composer.compose_intent(s, it, ans.version)
                ans.intents[iid] = it
                ans.claims += new_claims
                added += [c.cid for c in new_claims]
                delta_ids.append(iid)
        ans.uncertainty = [u for u in ans.uncertainty if u["intent_id"] not in delta_ids]
        self._coverage(ans)
        ans.text = self.composer.render(ans, delta_ids=delta_ids, constraint=constraint)
        s.merge_ctx(self.decomposer.session_context(clauses))
        retained = [c.cid for c in ans.active_claims() if c.cid not in added]
        diff = {"parent_version": prev_version, "mode": "refine", "added": added, "retired": retired,
                "retained": retained, "delta_intents": delta_ids,
                "targets": [{"mode": m, "intent": (t.iid if t else None), "query": q.text} for m, t, q in deltas]}
        return self._complete(sh, "refinement", gate, t_end, c0, jobs, queries, diff, reused)

    # .................................................................. presentation / chit-chat
    def _finish_suppressed(self, sh: SessionHandle, gate: GateResult, t_end: float, c0: float) -> TurnResult:
        turn, s = sh.turn, sh.session
        sh.tracer.emit("retrieval_suppressed", t_end, turn.turn_id, reason=gate.label, detail=gate.reason)
        diff = None
        if gate.label == "presentation" and s.has_answer:
            t = normalize_spoken(turn.text).lower()
            n = spoken_count(t)
            style = "bullets" if (n or re.search(r"bullet|points|list", t)) else \
                    "repeat" if re.search(r"\b(repeat|again|read that back)\b", t) and not \
                    re.search(r"short|brief|gist", t) else "short"
            text = self.composer.present(s.answer, style, n)
            s.answer.view = f"{style}:{n}" if n else style
            diff = {"parent_version": s.answer.version, "mode": "presentation", "style": style, "count": n,
                    "added": [], "retired": [], "retained": [c.cid for c in s.answer.active_claims()]}
        else:
            text = ""
        return self._complete(sh, gate.label, gate, t_end, c0, [], [], diff, 0, text_override=text)

    # .................................................................. common completion
    def _emit_retrieval_done(self, sh: SessionHandle, jobs: list[Job], results: dict, queries: list[SubQuery]):
        turn = sh.turn
        seen = set()
        for d in turn.dispatches:
            if id(d.job) in seen:
                continue
            seen.add(id(d.job))
            qids = [x.query.qid for x in turn.dispatches if x.job is d.job]
            res, st = d.job.wait()
            sh.tracer.emit("retrieval_completed", d.job.t_end, turn.turn_id, qids=qids, trigger=d.query.trigger,
                           t_dispatch=d.job.t_dispatch, t_start=d.job.t_start, compute_ms=round(d.job.compute_ms, 2),
                           used=any(x.used for x in turn.dispatches if x.job is d.job),
                           top={q: [h.label for h in res[q][:3]] for q in qids if q in res}, stats=st)
        for j in jobs:
            if id(j) in seen:
                continue
            seen.add(id(j))
            res, st = j.wait()
            sh.tracer.emit("retrieval_completed", j.t_end, turn.turn_id, qids=list(res.keys()), trigger="final",
                           t_dispatch=j.t_dispatch, t_start=j.t_start, compute_ms=round(j.compute_ms, 2), used=True,
                           top={q: [h.label for h in hs[:3]] for q, hs in res.items()}, stats=st)

    def _complete(self, sh: SessionHandle, kind: str, gate: GateResult, t_end: float, c0: float, jobs: list[Job],
                  queries: list[SubQuery], diff: dict | None, reused: int, text_override: str | None = None
                  ) -> TurnResult:
        turn, s = sh.turn, sh.session
        ans = s.answer
        text = ans.text if text_override is None else text_override
        tail_s = time.perf_counter() - c0          # final decomposition + synthesis CPU after utterance end
        retr_s = sum(j.compute_ms for j in jobs if j.t_dispatch >= t_end) / 1000
        synth_s = max(0.0, tail_s - retr_s)
        t_ready = max([t_end] + [j.t_end for j in jobs])
        if sh.executor.realtime:
            t_first = sh.executor.now()
        else:
            t_first = sh.executor.charge(t_ready, synth_s)
        grounding = {"claims": 0, "supported": 0, "support_rate": 1.0, "fabricated_ids": [], "unretrieved_ids": []}
        if kind in ("retrieval", "refinement", "presentation") and ans.version:
            retrieved = {h.label for i in ans.intents.values() for h in i.evidence}
            claims = [c.to_dict() for c in ans.active_claims()]
            grounding = verify(claims, self.index, retrieved)
            for r in grounding["results"]:
                for c in ans.claims:
                    if c.cid == r["cid"]:
                        c.support = r["score"]
        if text:
            sh.tracer.emit("answer_started", t_ready, turn.turn_id, kind=kind)
            sh.tracer.emit("first_token", t_first, turn.turn_id, ttft_ms=round((t_first - t_end) * 1000, 2),
                           token=text.split()[0] if text.split() else "")
        version = ans.version
        sh.tracer.emit("answer_version", t_first, turn.turn_id, version=version, kind=kind, text=text,
                       citations=ans.citations() if kind != "chitchat" else [], diff=diff,
                       view=ans.view if kind == "presentation" else "full")
        sh.tracer.emit("grounding_check", t_first, turn.turn_id, claims=grounding["claims"],
                       supported=grounding["supported"], support_rate=round(grounding["support_rate"], 4),
                       fabricated_ids=grounding["fabricated_ids"], unretrieved_ids=grounding["unretrieved_ids"])
        unc = ans.uncertainty if kind in ("retrieval", "refinement") else []
        unc_all = unc + [{"intent_id": i.iid, "message": f"{i.label}: {i.uncertain}"} for i in ans.intents.values()
                         if i.uncertain and kind in ("retrieval", "refinement") and i.status == "active"]
        for u in unc_all:
            sh.tracer.emit("uncertainty_flag", t_first, turn.turn_id, **u)
        after = self.models.stats.snapshot()
        before = turn.stats_before
        tokens = {k: after[k] - before.get(k, 0) for k in ("embed_tokens", "rerank_tokens")}
        model_ms = (after["embed_ms"] - before.get("embed_ms", 0)) + (after["rerank_ms"] - before.get("rerank_ms", 0))
        cpu_ms = model_ms + turn.controller_ms + synth_s * 1000
        usd = cpu_ms / 3.6e6 * self.s.cpu_usd_per_vcpu_hour * self.s.threads
        n_tokens = len(text.split())
        wasted = sum(1 for d in turn.dispatches if not d.used)
        latency = {
            "ttft_ms": round((t_first - t_end) * 1000, 2) if text else None,
            "first_retrieval_s": turn.first_retrieval_t,
            "retrieval_lead_ms": round((t_end - turn.first_retrieval_t) * 1000, 1)
            if turn.first_retrieval_t is not None else None,
            "post_utterance_retrieval_ms": round(retr_s * 1000, 2),
            "synthesis_ms": round(synth_s * 1000, 2),
            "controller_ms_total": round(turn.controller_ms, 2),
            "controller_decisions": turn.decisions,
        }
        cost = {**tokens, "cpu_ms": round(cpu_ms, 2), "usd": round(usd, 8), "llm_tokens_in": 0,
                "llm_tokens_out": 0, "answer_words": n_tokens}
        retrieved_any = bool(turn.dispatches) or bool(jobs)
        early = turn.first_retrieval_t is not None and turn.first_retrieval_t < t_end
        sh.tracer.emit("turn_completed", t_first, turn.turn_id, kind=kind, latency=latency, cost=cost,
                       retrieved=retrieved_any, early_retrieval=early, speculative={"dispatched": len(
                           turn.dispatches), "reused": reused, "wasted": wasted}, gate=gate.to_dict())
        s.turns.append({"turn_id": turn.turn_id, "kind": kind, "text": turn.text})
        intents = [i.to_dict() for i in ans.intents.values()] if kind in ("retrieval", "refinement") else []
        return TurnResult(turn.turn_id, kind, text, version, ans.citations() if kind != "chitchat" else [],
                          intents, [q.to_dict() for q in queries], unc_all, grounding, latency, cost, retrieved_any,
                          early, diff)
