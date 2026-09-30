"""Multi-query hybrid retrieval: BM25 + dense -> RRF -> cross-encoder rerank -> cross-query fusion.

All sub-queries of a turn are processed as ONE batch (one embedding call, one
cross-encoder call), which is how "parallel retrieval" is realised cheaply on CPU.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .index import HybridIndex


@dataclass
class SubQuery:
    qid: str
    text: str                  # natural-language sub-query (dense + reranker)
    keywords: str = ""         # keyword form (BM25); defaults to text
    origin: str = ""           # clause the sub-query came from
    trigger: str = "multi_intent"   # provisional | multi_intent | refinement | final
    anchor: bool = False       # True for the clause that sets the topic/entities
    constraints: dict = field(default_factory=dict)
    alt: str = ""              # context-free variant for vague clauses (context-hedged retrieval)

    def to_dict(self) -> dict:
        return {"qid": self.qid, "text": self.text, "keywords": self.keywords or self.text,
                "origin": self.origin, "trigger": self.trigger, "anchor": self.anchor,
                "constraints": self.constraints, "alt": self.alt}


@dataclass
class Hit:
    idx: int
    label: str
    score: float               # cross-encoder logit (or fused score when rerank is off)
    rrf: float
    bm25_rank: int | None
    dense_rank: int | None
    qid: str
    sent_scores: list | None = None   # cross-encoder logit per sentence unit (top chunks only)

    def to_dict(self) -> dict:
        return {"label": self.label, "score": round(self.score, 3), "rrf": round(self.rrf, 4),
                "bm25_rank": self.bm25_rank, "dense_rank": self.dense_rank, "qid": self.qid}


class Retriever:
    def __init__(self, index: HybridIndex, models, settings):
        self.index = index
        self.models = models
        self.s = settings
        self._cache: OrderedDict[tuple, list[Hit]] = OrderedDict()
        # superseded docs whose replacement exists in the corpus get demoted
        self._demote = np.array([
            1.0 if (c.status == "superseded" and (c.superseded_by in index.doc_ids or not c.superseded_by)) else 0.0
            for c in index.chunks], dtype=np.float32)

    def _key(self, q: SubQuery) -> tuple:
        return (q.text.lower().strip(), (q.keywords or q.text).lower().strip(), q.alt.lower().strip(),
                self.s.retrieval_mode, self.s.rerank)

    def _pool(self, text: str, keywords: str, qvec, n: int, pool: dict) -> None:
        if self.s.retrieval_mode in ("hybrid", "bm25"):
            for r, (i, _) in enumerate(self.index.bm25_rank(keywords or text, n)):
                p = pool.setdefault(i, {"rrf": 0.0, "bm25_rank": None, "dense_rank": None})
                p["rrf"] += 1.0 / (self.s.rrf_k + r + 1)
                p["bm25_rank"] = min(r + 1, p["bm25_rank"] or 10 ** 6)
        if self.s.retrieval_mode in ("hybrid", "dense") and qvec is not None:
            for r, (i, _) in enumerate(self.index.dense_rank(qvec, n)):
                p = pool.setdefault(i, {"rrf": 0.0, "bm25_rank": None, "dense_rank": None})
                p["rrf"] += 1.0 / (self.s.rrf_k + r + 1)
                p["dense_rank"] = min(r + 1, p["dense_rank"] or 10 ** 6)

    def retrieve(self, queries: list[SubQuery]) -> tuple[dict[str, list[Hit]], dict]:
        """Returns ({qid: hits}, stats)."""
        t0 = time.perf_counter()
        out: dict[str, list[Hit]] = {}
        todo = []
        for q in queries:
            k = self._key(q)
            if k in self._cache:
                out[q.qid] = [Hit(**{**h.__dict__, "qid": q.qid}) for h in self._cache[k]]
            else:
                todo.append(q)
        cache_hits = len(queries) - len(todo)
        n = self.s.first_stage_n
        cand: dict[str, dict[int, dict]] = {}
        if todo:
            use_dense = self.s.retrieval_mode in ("hybrid", "dense")
            texts = [q.text for q in todo] + [q.alt for q in todo if q.alt]
            vecs = self.models.embed_queries(texts) if use_dense else None
            alt_i = len(todo)
            for qi, q in enumerate(todo):
                pool: dict[int, dict] = {}
                self._pool(q.text, q.keywords, vecs[qi] if vecs is not None else None, n, pool)
                if q.alt:     # context-hedged: also pool candidates for the context-free variant
                    self._pool(q.alt, q.alt, vecs[alt_i] if vecs is not None else None, n, pool)
                    alt_i += 1
                cand[q.qid] = pool
            # one batched cross-encoder call over every (sub-query [variant], candidate) pair
            pairs, owners = [], []
            for q in todo:
                top = sorted(cand[q.qid].items(), key=lambda kv: -kv[1]["rrf"])[: self.s.rerank_candidates]
                cand[q.qid] = dict(top)
                if self.s.rerank:
                    for i, _ in top:
                        for v, text in enumerate([q.text] + ([q.alt] if q.alt else [])):
                            pairs.append((text, self.index.chunks[i].index_text()))
                            owners.append((q.qid, i, v))
            scores = self.models.rerank_pairs(pairs) if pairs else np.zeros(0)
            logits: dict[tuple[str, int], float] = {}
            for (qid, i, _), sc in zip(owners, scores):
                logits[(qid, i)] = max(logits.get((qid, i), -1e9), float(sc))
            spairs, sowners = [], []
            for q in todo:
                hits = []
                for i, meta in cand[q.qid].items():
                    if self.s.rerank:
                        score = logits[(q.qid, i)] - self.s.superseded_penalty * self._demote[i]
                    else:
                        score = meta["rrf"] * 100 - self.s.superseded_penalty * self._demote[i] * 0.2
                    hits.append(Hit(i, self.index.chunks[i].label, score, meta["rrf"],
                                    meta["bm25_rank"], meta["dense_rank"], q.qid))
                hits.sort(key=lambda h: -h.score)
                hits = hits[: max(self.s.evidence_per_query * 2, 8)]
                out[q.qid] = hits
                # second stage: score the sentence units of the top chunks (answer selection)
                if self.s.rerank:
                    for h in hits[: self.s.sentence_chunks]:
                        ch = self.index.chunks[h.idx]
                        for si, sent in enumerate(ch.sentences):
                            spairs.append((q.alt or q.text, f"{ch.short_title} - {ch.section_title}: {sent}"))
                            sowners.append((h, si))
            if spairs:
                sscores = self.models.rerank_pairs(spairs)
                for (h, si), sc in zip(sowners, sscores):
                    if h.sent_scores is None:
                        h.sent_scores = [None] * len(self.index.chunks[h.idx].sentences)
                    h.sent_scores[si] = float(sc)
            for q in todo:
                self._cache[self._key(q)] = out[q.qid]
                if len(self._cache) > 2048:
                    self._cache.popitem(last=False)
        stats = {"latency_ms": round((time.perf_counter() - t0) * 1000, 2), "queries": len(queries),
                 "cache_hits": cache_hits, "pairs_reranked": sum(len(v) for v in cand.values()) if self.s.rerank else 0}
        return out, stats


def fuse_across_queries(results: dict[str, list[Hit]], k: int = 60, per_query: int = 4) -> list[Hit]:
    """Reciprocal-rank fusion across sub-queries with de-duplication.

    A chunk retrieved by several sub-queries is kept once (attributed to the
    sub-query where it ranked best) and its fused score accumulates, so evidence
    that serves multiple intents rises to the top.
    """
    fused: dict[int, Hit] = {}
    acc: dict[int, float] = {}
    for qid, hits in results.items():
        for r, h in enumerate(hits[:per_query]):
            acc[h.idx] = acc.get(h.idx, 0.0) + 1.0 / (k + r + 1)
            if h.idx not in fused or h.score > fused[h.idx].score:
                fused[h.idx] = h
    return sorted(fused.values(), key=lambda h: -acc[h.idx])
