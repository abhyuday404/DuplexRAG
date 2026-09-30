"""Generate docs/BENCHMARK.md tables from results/*/summary.json (+ the hand-written analysis in
docs/benchmark_analysis.md, appended verbatim)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = {}
for split in ("test", "dev", "devb", "devc"):
    p = ROOT / "results" / split / "summary.json"
    if p.exists():
        RES[split] = json.loads(p.read_text())
V1 = {}
for cfg in ("duplexrag", "baseline"):
    p = ROOT / "results" / "archive" / f"devc_heldout_run_engine_v1_{cfg}" / "metrics.json"
    if p.exists():
        V1[cfg] = json.loads(p.read_text())
SPLITS = {}
for split in ("test", "dev", "devb", "devc"):
    p = ROOT / "data" / "benchmark" / f"{split}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p) if l.strip()]
        SPLITS[split] = (len(rows), sum(len(r["turns"]) for r in rows))


def f(v, fmt="{:.1f}"):
    if v is None:
        return "-"
    if isinstance(v, float):
        return fmt.format(v)
    return str(v)


def get(split, cfg, key):
    src = V1 if split == "v1" else RES.get(split, {})
    return src.get(cfg, {}).get(key)


def gate_table() -> str:
    t = RES.get("test", {}).get("duplexrag", {})
    b = RES.get("test", {}).get("baseline", {})

    def verdict(ok):
        return "**pass**" if ok else "fail"

    rows = [
        ("G1 Reproducibility", "pass/fail", "one-command `docker compose up` (app + automated replay); "
         "`scripts/smoke_test.sh` passes from a clean `git archive` with the frozen lockfile", "**pass**¹", "-"),
        ("G2 Early retrieval", ">= 80% of eligible turns", f"{f(t.get('G2_early_retrieval_pct'))}% early, "
         f"{f(t.get('G2_false_trigger_pct'))}% false triggers", verdict((t.get("G2_early_retrieval_pct") or 0) >= 80),
         f"{f(b.get('G2_early_retrieval_pct'))}%"),
        ("G3 Multi-intent", ">= 70% of compound turns", f"{f(t.get('G3_multi_intent_pct'))}%",
         verdict((t.get("G3_multi_intent_pct") or 0) >= 70), f"{f(b.get('G3_multi_intent_pct'))}%"),
        ("G4 Factual grounding", ">= 85% claim support, 0 fabricated IDs",
         f"{f(t.get('G4_claim_support_pct'))}% of {t.get('G4_claims', '-')} claims, {t.get('G4_fabricated_ids', '-')} fabricated",
         verdict((t.get("G4_claim_support_pct") or 0) >= 85 and t.get("G4_fabricated_ids") == 0),
         f"{f(b.get('G4_claim_support_pct'))}%"),
        ("G5 Session refinement", "state continuity", f"{f(t.get('G5_refinement_pass_pct'))}% of refinement turns versioned, "
         "state kept, delta-only", verdict((t.get("G5_refinement_pass_pct") or 0) >= 80), f"{f(b.get('G5_refinement_pass_pct'))}%"),
        ("G6 Telemetry", "100% trace coverage", f"{f(t.get('G6_trace_coverage_pct'))}%",
         verdict(t.get("G6_trace_coverage_pct") == 100.0), "-"),
    ]
    out = ["| Gate | Target | DuplexRAG (held-out test) | Verdict | Baseline |", "|---|---|---|---|---|"]
    out += [f"| {a} | {b_} | {c} | {d} | {e} |" for a, b_, c, d, e in rows]
    return "\n".join(out)


METRICS = [
    ("gate_accuracy_pct", "Turn-type accuracy %"), ("retrieval_decision_accuracy_pct", "Retrieve / suppress decision accuracy %"),
    ("G2_early_retrieval_pct", "G2 early retrieval %"), ("G2_early_before_last_chunk_pct", "... started before the last chunk %"),
    ("retrieval_lead_ms_median", "Median retrieval head start (ms before end of speech)"),
    ("G2_false_trigger_pct", "G2 false triggers on no-retrieval turns %"), ("G3_multi_intent_pct", "G3 multi-intent %"),
    ("G3_mean_intent_isolation_pct", "Gold intents isolated per compound turn %"),
    ("retrieval_hit_at_3_pct", "Retrieval hit@3 (per gold intent) %"), ("retrieval_recall_at_5_pct", "Retrieval recall@5 %"),
    ("key_fact_recall_pct", "Key-fact recall in answer %"), ("citation_precision_pct", "Citation precision %"),
    ("G4_claim_support_pct", "G4 claim support %"), ("G4_fabricated_ids", "Fabricated citation IDs"),
    ("unanswerable_flagged_pct", "Unanswerable needs flagged %"), ("false_uncertainty_pct", "False 'not found' flags %"),
    ("entity_coverage_notes_pct", "Turns with entity x aspect coverage notes %"),
    ("G5_refinement_pass_pct", "G5 refinement pass %"), ("G6_trace_coverage_pct", "G6 trace coverage %"),
    ("ttft_ms_p50", "TTFT p50 (ms after end of speech)"), ("ttft_ms_p95", "TTFT p95 (ms)"),
    ("post_utterance_retrieval_ms_mean", "Retrieval CPU after end of speech, mean (ms)"),
    ("cpu_ms_per_turn_mean", "CPU per turn, mean (ms)"), ("model_tokens_per_turn_mean", "Model tokens per turn"),
    ("cost_usd_per_turn_mean", "Cost per turn (USD)"),
]


def main_table(split: str, cfgs: list[str]) -> str:
    out = ["| Metric | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    for key, label in METRICS:
        vals = []
        for c in cfgs:
            v = get(split, c, key)
            vals.append(f(v, "{:.6f}") if key == "cost_usd_per_turn_mean" else f(v))
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def ablation_table() -> str:
    t = RES.get("test", {})
    cfgs = [c for c in ["duplexrag", "dense_only", "bm25_only", "no_rerank", "rule_controller", "model_controller",
                        "no_decomposition", "no_speculation", "baseline"] if c in t]
    keys = [("gate_accuracy_pct", "Turn type"), ("G2_early_retrieval_pct", "G2"), ("G2_false_trigger_pct", "False trig."),
            ("G3_multi_intent_pct", "G3"), ("retrieval_hit_at_3_pct", "Hit@3"), ("key_fact_recall_pct", "Key facts"),
            ("G5_refinement_pass_pct", "G5"), ("ttft_ms_p50", "TTFT p50 ms"), ("ttft_ms_p95", "TTFT p95 ms"),
            ("cpu_ms_per_turn_mean", "CPU ms/turn")]
    out = ["| Configuration | " + " | ".join(l for _, l in keys) + " |", "|---|" + "---|" * len(keys)]
    for c in cfgs:
        out.append(f"| `{c}` | " + " | ".join(f(t[c].get(k)) for k, _ in keys) + " |")
    return "\n".join(out)


def splits_table() -> str:
    names = {"dev": "dev - used for all tuning", "devb": "dev-b - first held-out, then used for diagnosis",
             "devc": "dev-c - second held-out (engine v1), then used for diagnosis", "test": "**test - final held-out**"}
    out = ["| Split | Sessions | Turns | Role |", "|---|---|---|---|"]
    for s in ("dev", "devb", "devc", "test"):
        if s in SPLITS:
            out.append(f"| `{s}.jsonl` | {SPLITS[s][0]} | {SPLITS[s][1]} | {names[s]} |")
    return "\n".join(out)


def all_splits_table() -> str:
    keys = [("gate_accuracy_pct", "Turn type"), ("G2_early_retrieval_pct", "G2"), ("G2_false_trigger_pct", "False trig."),
            ("G3_multi_intent_pct", "G3"), ("retrieval_hit_at_3_pct", "Hit@3"), ("key_fact_recall_pct", "Key facts"),
            ("G4_claim_support_pct", "G4"), ("false_uncertainty_pct", "False 'not found'"),
            ("G5_refinement_pass_pct", "G5"), ("ttft_ms_p50", "TTFT p50")]
    out = ["| Split / engine | " + " | ".join(l for _, l in keys) + " |", "|---|" + "---|" * len(keys)]
    for label, split, cfg in [("dev, engine v2", "dev", "duplexrag"), ("dev-b, engine v2", "devb", "duplexrag"),
                              ("dev-c, engine v1 (its held-out run)", "v1", "duplexrag"),
                              ("dev-c, engine v2", "devc", "duplexrag"), ("**test, engine v2 (final)**", "test", "duplexrag"),
                              ("test, baseline", "test", "baseline")]:
        if (split == "v1" and V1) or split in RES:
            out.append(f"| {label} | " + " | ".join(f(get(split, cfg, k)) for k, _ in keys) + " |")
    return "\n".join(out)


def main() -> None:
    analysis = (ROOT / "docs" / "benchmark_analysis.md")
    doc = f"""# Benchmark and evaluation report

All numbers are produced by `duplexrag bench` (discrete-event streaming replay, see docs/ARCHITECTURE.md §2) and can
be regenerated with `docker compose up` or `uv run duplexrag bench --split test --ablations`. Per-turn records,
telemetry traces and metrics for every configuration are in `results/<split>/<config>/`.

Hardware: laptop CPU (AMD Ryzen 5 4600H, 6 cores / 12 threads), 4 ONNX threads, no GPU; a web browser was running
during the measurements, so latency figures are conservative.

## Data and protocol

{splits_table()}

Every split was written by a separate AI agent that could read only the corpus (never the code, results or the
other splits). Utterances are spoken-style transcripts with gold sections, key facts and turn types; they are
streamed as ASR-like chunks of 2-4 words at 2.7 words/s with a 350 ms end-of-speech silence. The engine was tuned on
`dev` only. `devb` was our first held-out set; its results exposed generalisation bugs, so it became a diagnosis set
and a new held-out set was written after the fixes. That set (`devc`) was run once with the frozen engine ("engine
v1"; results archived in `results/archive/`) - it again revealed controller and uncertainty problems, so it too became
a diagnosis set, the engine was fixed and frozen again ("engine v2"), and the final `test` split was written
afterwards. We report every held-out run, including the ones that led to fixes.

## Evaluation gates (final held-out test)

{gate_table()}

¹ The Docker daemon was not available on the build machine, so G1 was verified with the identical steps outside a
container: fresh `git archive` checkout → `uv sync --frozen` → model download → `duplexrag index` → streaming replay →
schema validation of the trace (`scripts/smoke_test.sh`).

## DuplexRAG vs the turn-based baseline (held-out test)

The baseline is a conventional turn-based RAG pipeline: it waits for the end of speech, retrieves once with the whole
utterance as the query (dense only, top 5), has no reranker, no decomposition and no session memory, and composes the
answer with the same extractive composer.

{main_table("test", [c for c in ["duplexrag", "baseline"] if c in RES.get("test", {})])}

## Ablations (held-out test)

{ablation_table()}

* `dense_only` / `bm25_only`: one retriever instead of hybrid RRF (the cross-encoder still reranks).
* `no_rerank`: fused RRF order without the cross-encoder (and without cross-encoder sentence scoring).
* `rule_controller` / `model_controller`: the turn gate uses only the rules or only the trained classifier
  (default `duplexrag` = hybrid).
* `no_decomposition`: one query per utterance (same retrieval stack).
* `no_speculation`: identical pipeline, but retrieval waits for the end of speech - isolates the latency effect of
  streaming.

## All splits

{all_splits_table()}

"""
    if analysis.exists():
        doc += analysis.read_text()
    metrics_doc = ROOT / "docs" / "benchmark_metrics.md"
    if metrics_doc.exists():
        doc += "\n" + metrics_doc.read_text()
    (ROOT / "docs" / "BENCHMARK.md").write_text(doc)
    print("wrote docs/BENCHMARK.md")
    # README summary block
    t = RES.get("test", {})
    if "duplexrag" in t:
        d, b = t["duplexrag"], t.get("baseline", {})
        ns = t.get("no_speculation", {})
        n = SPLITS.get("test", (0, 0))
        rows = [
            ("Retrieval starts before the user stops talking (G2, target >= 80%)", "G2_early_retrieval_pct", "%"),
            ("False retrieval on 'repeat that' / 'thanks' turns", "G2_false_trigger_pct", "%"),
            ("Compound requests split into >= 2 correct intents (G3, target >= 70%)", "G3_multi_intent_pct", "%"),
            ("Claims supported by their citation (G4, target >= 85%)", "G4_claim_support_pct", "%"),
            ("Fabricated citation IDs (G4, target 0)", "G4_fabricated_ids", ""),
            ("Refinements versioned + delta-only (G5)", "G5_refinement_pass_pct", "%"),
            ("Telemetry trace coverage (G6, target 100%)", "G6_trace_coverage_pct", "%"),
            ("Gold section in top-3 evidence", "retrieval_hit_at_3_pct", "%"),
            ("Key facts present in the answer", "key_fact_recall_pct", "%"),
            ("Time to first token after end of speech, p50", "ttft_ms_p50", " ms"),
            ("Compute cost per turn", "cost_usd_per_turn_mean", ""),
        ]
        lines = [f"Held-out test split ({n[0]} sessions, {n[1]} turns, written independently after the engine was "
                 "frozen); full report with ablations and failure analysis in [docs/BENCHMARK.md](docs/BENCHMARK.md).",
                 "", "| | DuplexRAG | Turn-based baseline |", "|---|---|---|"]
        for label, key, unit in rows:
            def fmt(v):
                if v is None:
                    return "-"
                if key == "cost_usd_per_turn_mean":
                    return f"${v:.6f}"
                return f"{v:.0f}{unit}" if isinstance(v, float) else f"{v}{unit}"
            lines.append(f"| {label} | {fmt(d.get(key))} | {fmt(b.get(key))} |")
        if ns.get("ttft_ms_p50") is not None:
            lines.append(f"| Same stack without speculative retrieval: TTFT p50 | {ns['ttft_ms_p50']:.0f} ms | |")
        block = "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
        readme = ROOT / "README.md"
        txt = readme.read_text()
        s, e = txt.index("<!-- RESULTS:START -->"), txt.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
        readme.write_text(txt[:s] + block + txt[e:])
        print("updated README results")


if __name__ == "__main__":
    main()
