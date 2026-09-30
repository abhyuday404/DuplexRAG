"""Benchmark runner: replays a split under one or more configurations and writes results.

    python -m duplexrag.bench.run --split test                 # DuplexRAG + baseline
    python -m duplexrag.bench.run --split test --ablations     # + all ablations
    python -m duplexrag.bench.run --split dev --configs duplexrag,no_rerank

Outputs per config: results/<split>/<config>/{turns.jsonl, trace.jsonl, metrics.json}
and a combined results/<split>/summary.json + summary.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import ROOT, load_settings
from ..engine import DuplexEngine
from ..stream import replay_session
from ..telemetry import write_jsonl
from .baseline import BaselineRAG
from .metrics import evaluate

CONFIGS: dict[str, dict] = {
    "duplexrag": {},
    "baseline": {"baseline": True},
    "dense_only": {"retrieval_mode": "dense"},
    "bm25_only": {"retrieval_mode": "bm25"},
    "no_rerank": {"rerank": False},
    "rule_controller": {"controller": "rule"},
    "model_controller": {"controller": "model"},
    "no_decomposition": {"decompose": False},
    "no_speculation": {"speculative": False},
}
DEFAULT = ["duplexrag", "baseline"]
ABLATIONS = ["dense_only", "bm25_only", "no_rerank", "rule_controller", "model_controller", "no_decomposition",
             "no_speculation"]

SUMMARY_KEYS = [
    ("gate_accuracy_pct", "Turn-type accuracy %"),
    ("G2_early_retrieval_pct", "G2 early retrieval %"),
    ("G2_false_trigger_pct", "G2 false triggers %"),
    ("G3_multi_intent_pct", "G3 multi-intent %"),
    ("retrieval_hit_at_3_pct", "Hit@3 %"),
    ("retrieval_recall_at_5_pct", "Recall@5 %"),
    ("key_fact_recall_pct", "Key-fact recall %"),
    ("G4_claim_support_pct", "G4 claim support %"),
    ("G4_fabricated_ids", "Fabricated IDs"),
    ("unanswerable_flagged_pct", "Unanswerable flagged %"),
    ("false_uncertainty_pct", "False uncertainty %"),
    ("G5_refinement_pass_pct", "G5 refinement %"),
    ("G6_trace_coverage_pct", "G6 trace coverage %"),
    ("ttft_ms_p50", "TTFT p50 ms"),
    ("ttft_ms_p95", "TTFT p95 ms"),
    ("cost_usd_per_turn_mean", "USD / turn"),
]


def load_split(split: str) -> list[dict]:
    path = Path(split) if split.endswith(".jsonl") else ROOT / "data" / "benchmark" / f"{split}.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run_config(name: str, sessions: list[dict], out_dir: Path, engine_cache: dict) -> dict:
    cfg = dict(CONFIGS[name])
    t0 = time.time()
    records, events = [], []
    if cfg.pop("baseline", False):
        eng = engine_cache.setdefault("duplexrag", DuplexEngine(load_settings(), log=lambda *a: None))
        base = BaselineRAG(eng)
        for s in sessions:
            records += base.replay_session(s)
        events = None
    else:
        decompose = cfg.pop("decompose", True)
        speculative = cfg.pop("speculative", True)
        eng = DuplexEngine(load_settings(**cfg), decompose=decompose, speculative=speculative, log=lambda *a: None)
        engine_cache.setdefault(name, eng)
        for s in sessions:
            recs, evs = replay_session(eng, s)
            records += recs
            events += evs
    metrics = evaluate(records, sessions, events)
    metrics["wall_s"] = round(time.time() - t0, 1)
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "turns.jsonl", records)
    if events is not None:
        write_jsonl(d / "trace.jsonl", events)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def summary_md(results: dict[str, dict], split: str) -> str:
    names = list(results)
    lines = [f"# Benchmark summary - `{split}` split", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in SUMMARY_KEYS:
        row = []
        for n in names:
            v = results[n].get(key)
            row.append("-" if v is None else (f"{v:.6f}" if key == "cost_usd_per_turn_mean" else str(v)))
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--configs", default=None, help="comma-separated config names")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    names = a.configs.split(",") if a.configs else DEFAULT + (ABLATIONS if a.ablations else [])
    sessions = load_split(a.split)
    split_name = Path(a.split).stem
    out_dir = Path(a.out) / split_name
    results = {}
    cache: dict = {}
    for n in names:
        print(f"[bench] {split_name}: running {n} ...", flush=True)
        m = run_config(n, sessions, out_dir, cache)
        results[n] = m
        print("   " + ", ".join(f"{lbl}={m.get(k)}" for k, lbl in SUMMARY_KEYS if m.get(k) is not None), flush=True)
    prev = {}
    sfile = out_dir / "summary.json"
    if sfile.exists():
        prev = json.loads(sfile.read_text())
    prev.update({k: {kk: vv for kk, vv in v.items() if kk not in ("G5_detail", "G6_missing")}
                 for k, v in results.items()})
    sfile.write_text(json.dumps(prev, indent=2))
    order = [n for n in DEFAULT + ABLATIONS if n in prev]
    (out_dir / "summary.md").write_text(summary_md({n: prev[n] for n in order}, split_name))
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
