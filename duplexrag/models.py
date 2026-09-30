"""CPU-only ONNX models (fastembed): a 33M bi-encoder and a 22M cross-encoder."""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ModelStats:
    """Accumulates compute time and token counts for the cost model."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.embed_tokens = 0
        self.rerank_tokens = 0
        self.embed_ms = 0.0
        self.rerank_ms = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            return {"embed_tokens": self.embed_tokens, "rerank_tokens": self.rerank_tokens,
                    "embed_ms": round(self.embed_ms, 3), "rerank_ms": round(self.rerank_ms, 3)}


class Models:
    _instances: dict = {}

    def __init__(self, embed_model: str, rerank_model: str, cache_dir: str, threads: int = 4):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model
        self.embedder = TextEmbedding(embed_model, cache_dir=cache_dir, threads=threads, cuda=False)
        self.reranker = TextCrossEncoder(rerank_model, cache_dir=cache_dir, threads=threads, cuda=False)
        self.stats = ModelStats()
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()   # ONNX sessions are shared; serialise to avoid oversubscription

    @classmethod
    def get(cls, settings) -> "Models":
        key = (settings.embed_model, settings.rerank_model, settings.model_dir)
        if key not in cls._instances:
            cls._instances[key] = cls(settings.embed_model, settings.rerank_model, settings.model_dir,
                                      settings.threads)
        return cls._instances[key]

    def _count(self, model, texts) -> int:
        try:
            return int(sum(len(e.ids) for e in model.model.tokenizer.encode_batch(list(texts))))
        except Exception:  # pragma: no cover - tokenizer API drift
            return int(sum(len(t.split()) * 1.3 for t in texts))

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        with self._lock:
            t0 = time.perf_counter()
            vecs = np.array(list(self.embedder.passage_embed(texts, batch_size=batch_size)), dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000
        with self.stats.lock:
            self.stats.embed_ms += dt
            self.stats.embed_tokens += self._count(self.embedder, texts)
        return _normalize(vecs)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Batched query embedding with an LRU cache (partial utterances repeat a lot)."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._qcache]
        if missing:
            with self._lock:
                t0 = time.perf_counter()
                vecs = _normalize(np.array(list(self.embedder.query_embed(missing)), dtype=np.float32))
                dt = (time.perf_counter() - t0) * 1000
            with self.stats.lock:
                self.stats.embed_ms += dt
                self.stats.embed_tokens += self._count(self.embedder, missing)
            for t, v in zip(missing, vecs):
                self._qcache[t] = v
                if len(self._qcache) > 4096:
                    self._qcache.popitem(last=False)
        return np.stack([self._qcache[t] for t in texts]) if texts else np.zeros((0, 384), np.float32)

    def rerank_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            t0 = time.perf_counter()
            scores = np.array(list(self.reranker.rerank_pairs(pairs, batch_size=64)), dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000
        with self.stats.lock:
            self.stats.rerank_ms += dt
            self.stats.rerank_tokens += self._count(self.reranker, [q + " " + d for q, d in pairs])
        return scores


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n
