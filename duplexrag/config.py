"""Runtime configuration. Every knob can be overridden with a DUPLEXRAG_* env var."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default):
    raw = os.environ.get(f"DUPLEXRAG_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    corpus_dir: str = str(ROOT / "data" / "corpus")
    index_dir: str = str(ROOT / "index")
    model_dir: str = str(ROOT / "models")
    runs_dir: str = str(ROOT / "runs")
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    threads: int = 4

    # retrieval
    retrieval_mode: str = "hybrid"          # hybrid | dense | bm25
    first_stage_n: int = 30                 # candidates per retriever per sub-query
    rrf_k: int = 60
    rerank: bool = True
    rerank_candidates: int = 10             # fused candidates sent to the cross-encoder
    hedge_candidates: int = 5               # of those, how many are also scored with the context-free variant
    sentence_chunks: int = 3                # top chunks whose sentences are scored for answer selection
    sentences_per_chunk: int = 8
    evidence_per_query: int = 4
    superseded_penalty: float = 4.0         # logit penalty for documents marked superseded

    # controller
    controller: str = "hybrid"              # rule | model | hybrid
    min_salient_terms: int = 2
    max_subqueries: int = 4
    dedup_cosine: float = 0.90              # sub-queries closer than this are merged

    # synthesis / grounding
    synthesis: str = "extractive"           # extractive | llm
    evidence_threshold: float = -99.0       # absolute-score "not found" disabled: unreliable for vague speech
    aspect_threshold: float = -1.0          # min logit for an entity's own section to cover an aspect
    max_sentences_per_intent: int = 3
    llm_base_url: str = "http://localhost:11434/v1"   # any OpenAI-compatible endpoint
    llm_model: str = "qwen2.5:3b-instruct"
    llm_api_key_env: str = "DUPLEXRAG_LLM_API_KEY"
    llm_usd_per_mtok_in: float = 0.0
    llm_usd_per_mtok_out: float = 0.0

    # cost model: on-demand price of one vCPU-hour (c7i.large = USD 0.0893 / 2 vCPU)
    cpu_usd_per_vcpu_hour: float = 0.0447

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)


def load_settings(**overrides) -> Settings:
    base = Settings()
    env = {f.name: _env(f.name, getattr(base, f.name)) for f in fields(Settings)}
    env.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**env)
