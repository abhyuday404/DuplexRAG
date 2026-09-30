"""Conventional turn-based RAG baseline (what the brief calls the "static, batch-oriented" cycle).

* waits for the end of the utterance, then retrieves once with the whole utterance as the query;
* dense-only top-k (no BM25, no fusion, no reranker), no decomposition;
* no session memory: every turn - including late details and "repeat that" - is a fresh search;
* same extractive composer as DuplexRAG so answer-quality differences come from retrieval/control.
"""
from __future__ import annotations

import time

from ..decompose import normalize_spoken
from ..grounding import verify
from ..retrieve import Hit, SubQuery
from ..session import AnswerState, Intent, Session
from ..stream import turn_chunks
from ..text import content_tokens


class BaselineRAG:
    def __init__(self, engine, top_k: int = 5):
        self.e = engine            # reuse loaded models / index / composer
        self.k = top_k

    def _search(self, text: str) -> tuple[list[Hit], float, dict]:
        t0 = time.perf_counter()
        before = self.e.models.stats.snapshot()
        qv = self.e.models.embed_queries([text])[0]
        ranked = self.e.index.dense_rank(qv, self.k)
        hits = [Hit(i, self.e.index.chunks[i].label, float(s) * 10, 0.0, None, r + 1, "q1")
                for r, (i, s) in enumerate(ranked)]
        after = self.e.models.stats.snapshot()
        return hits, time.perf_counter() - t0, {k: after[k] - before[k] for k in after}

    def replay_session(self, session: dict) -> list[dict]:
        s = Session(session.get("session_id", "baseline"))
        out, offset = [], 0.5
        for turn in session["turns"]:
            chunks, end = turn_chunks(turn, s.session_id)
            t_end = offset + end
            text = normalize_spoken(" ".join(c.text for c in chunks))
            c0 = time.perf_counter()
            hits, search_s, st = self._search(text)
            q = SubQuery(f"{turn['turn_id']}q1", text, keywords=" ".join(content_tokens(text)), origin=text,
                         trigger="final", anchor=True)
            it = Intent("i1", "Answer", q, hits, s.answer.version + 1)
            ans = AnswerState(version=s.answer.version + 1, turn_id=turn["turn_id"], intents={"i1": it})
            # the baseline has no uncertainty gating beyond the composer's own checks
            ans.claims = self.e.composer.compose_intent(s, it, ans.version)
            ans.text = self.e.composer.render(ans)
            s.answer = ans
            total_s = time.perf_counter() - c0
            grounding = verify([c.to_dict() for c in ans.active_claims()], self.e.index, {h.label for h in hits})
            cpu_ms = st["embed_ms"] + st["rerank_ms"] + (total_s - search_s) * 1000
            out.append({
                "session_id": s.session_id, "turn_id": turn["turn_id"], "t_start": offset, "t_end": t_end,
                "kind": "retrieval", "text": ans.text, "version": ans.version, "citations": ans.citations(),
                "intents": [it.to_dict()], "sub_queries": [q.to_dict()],
                "uncertainty": ([{"intent_id": "i1", "message": it.uncertain}] if it.uncertain else []),
                "grounding": grounding, "retrieved": True, "early": False, "diff": None,
                "latency": {"ttft_ms": round(total_s * 1000, 2), "first_retrieval_s": t_end,
                            "retrieval_lead_ms": 0.0, "post_utterance_retrieval_ms": round(search_s * 1000, 2),
                            "synthesis_ms": round((total_s - search_s) * 1000, 2)},
                "cost": {"embed_tokens": st["embed_tokens"], "rerank_tokens": st["rerank_tokens"],
                         "cpu_ms": round(cpu_ms, 2),
                         "usd": round(cpu_ms / 3.6e6 * self.e.s.cpu_usd_per_vcpu_hour * self.e.s.threads, 8)},
            })
            offset = t_end + 2.5 + len(ans.text.split()) / 3.0
        return out
