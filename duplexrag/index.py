"""Hybrid index: Okapi BM25 (sparse) + bge-small (dense) over section chunks.

The index also derives a *corpus vocabulary* (IDF per stemmed term and a set of
proper-noun terms). The retrieval controller uses it to judge whether a partial
utterance already contains retrievable content - knowledge that comes from the
corpus itself, never from hard-coded query lists.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .corpus import Chunk, corpus_fingerprint, load_corpus
from .text import STOPWORDS, content_tokens, raw_tokens, stem

INDEX_VERSION = 5


def sentence_index_text(chunk: Chunk, sentence: str) -> str:
    return f"{chunk.short_title} - {chunk.section_title}: {sentence}"


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(q_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            idx = np.fromiter((p[0] for p in posting), dtype=np.int64)
            tf = np.fromiter((p[1] for p in posting), dtype=np.float32)
            s[idx] += idf * tf * (self.k1 + 1) / (tf + norm[idx])
        return s


class HybridIndex:
    def __init__(self, chunks: list[Chunk], dense: np.ndarray, fingerprint: str, build_ms: float = 0.0,
                 sent_vecs: np.ndarray | None = None):
        self.chunks = chunks
        self.dense = dense
        # sentence-level vectors (answer composition scores sentences without a model call)
        self.sent_owner: list[tuple[int, int]] = [(ci, si) for ci, c in enumerate(chunks) for si in range(len(c.sentences))]
        self.sent_offset = np.cumsum([0] + [len(c.sentences) for c in chunks])
        self.sent_vecs = sent_vecs
        self.fingerprint = fingerprint
        self.build_ms = build_ms
        self.by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        self.by_label = {c.label: i for i, c in enumerate(chunks)}
        self.bm25 = BM25([content_tokens(c.index_text()) for c in chunks])
        self.doc_ids = {c.doc_id for c in chunks}
        self._build_vocab()

    # ------------------------------------------------------------------ vocabulary
    def _build_vocab(self) -> None:
        self.idf = dict(self.bm25.idf)
        max_idf = max(self.idf.values()) if self.idf else 1.0
        self.max_idf = max_idf
        proper = Counter()
        lower = Counter()
        for c in self.chunks:
            text = c.text
            # capitalised words in running text that follow a lower-case word (not sentence starts)
            for m in re.finditer(r"(?<=[a-z,] )([A-Z][a-zA-Z]{2,})\b", text):
                proper[m.group(1).lower()] += 1
            for t in raw_tokens(text):
                lower[t] += 1
        # a proper noun is a capitalised word that (almost) never appears lower-cased
        self.proper_terms = {stem(w) for w, n in proper.items()
                             if w not in STOPWORDS and n >= 1 and lower[w] - n <= max(1, n // 4)}

    def salience(self, token_stem: str) -> float:
        """0..1 informativeness of a stemmed token w.r.t. this corpus (0 = unknown)."""
        idf = self.idf.get(token_stem)
        if idf is None:
            return 0.0
        return idf / self.max_idf

    # ------------------------------------------------------------------ search
    def bm25_rank(self, query: str, n: int) -> list[tuple[int, float]]:
        s = self.bm25.scores(content_tokens(query))
        if not s.any():
            return []
        top = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    def dense_rank(self, qvec: np.ndarray, n: int) -> list[tuple[int, float]]:
        sims = self.dense @ qvec
        top = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in top]

    # ------------------------------------------------------------------ persistence
    @classmethod
    def build(cls, corpus_dir: str, models) -> "HybridIndex":
        t0 = time.perf_counter()
        chunks = load_corpus(corpus_dir)
        if not chunks:
            raise SystemExit(f"No documents found in {corpus_dir}")
        dense = models.embed_passages([c.index_text() for c in chunks])
        sent_vecs = models.embed_passages([sentence_index_text(c, s) for c in chunks for s in c.sentences])
        return cls(chunks, dense, corpus_fingerprint(corpus_dir), (time.perf_counter() - t0) * 1000, sent_vecs)

    def sentence_vecs(self, chunk_idx: int) -> np.ndarray:
        a, b = self.sent_offset[chunk_idx], self.sent_offset[chunk_idx + 1]
        return self.sent_vecs[a:b]

    def save(self, index_dir: str, embed_model: str) -> None:
        p = Path(index_dir)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "dense.npy", self.dense)
        np.save(p / "sentences.npy", self.sent_vecs)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (p / "manifest.json").write_text(json.dumps({
            "version": INDEX_VERSION, "fingerprint": self.fingerprint, "embed_model": embed_model,
            "chunks": len(self.chunks), "build_ms": round(self.build_ms, 1),
        }, indent=2))

    @classmethod
    def load_or_build(cls, settings, models, log=print) -> "HybridIndex":
        p = Path(settings.index_dir)
        fp = corpus_fingerprint(settings.corpus_dir)
        man = p / "manifest.json"
        if man.exists():
            meta = json.loads(man.read_text())
            if (meta.get("fingerprint") == fp and meta.get("version") == INDEX_VERSION
                    and meta.get("embed_model") == settings.embed_model):
                chunks = []
                for line in open(p / "chunks.jsonl", encoding="utf-8"):
                    d = json.loads(line)
                    d.pop("label", None)
                    chunks.append(Chunk(**d))
                return cls(chunks, np.load(p / "dense.npy"), fp, meta.get("build_ms", 0.0),
                           np.load(p / "sentences.npy"))
        log(f"[index] building index for {settings.corpus_dir} ...")
        idx = cls.build(settings.corpus_dir, models)
        idx.save(settings.index_dir, settings.embed_model)
        log(f"[index] {len(idx.chunks)} chunks indexed in {idx.build_ms:.0f} ms")
        return idx
