"""Render the figures used by the README, docs and the submission deck (docs/img/*.png).

    python scripts/make_figures.py            # architecture + timeline (+ results if results/test exists)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, PURPLE, BLUE, GREEN, AMBER, LINE = "#14142B", "#63637E", "#6D28D9", "#2F80ED", "#1E9E6A", "#D98E04", "#D9D4F0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": LINE})


def box(ax, x, y, w, h, title, lines, color=PURPLE, fill="#FFFFFF"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=color, lw=2))
    ax.text(x + 0.12, y + h - 0.2, title, color=color, fontsize=12.5, fontweight="bold", va="top")
    for k, ln in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.62 - k * 0.33, ln, color=INK, fontsize=10, va="top")


def arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, color=color, lw=lw,
                                 linestyle=ls))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 7.4), dpi=160)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(0.2, 7.2, "Incoming ASR stream (one natural request, several hidden questions)", fontsize=12,
            fontweight="bold", color=INK, va="top")
    chunks = [("0.0 s", "I need to plan a customer", "workshop in"), ("0.8 s", "Pune for 30 people,", "and I need"),
              ("1.6 s", "the cancellation policy and", "the catering options."), ("2.1 s", "[end of speech]", "")]
    for k, (t, a, b) in enumerate(chunks):
        x = 0.2 + k * 3.95
        ax.add_patch(FancyBboxPatch((x, 5.95), 3.7, 0.85, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F4F1FD",
                                    ec=LINE, lw=1.2))
        ax.text(x + 0.12, 6.7, t, fontsize=9, color=MUTED, va="top")
        ax.text(x + 0.12, 6.45, a, fontsize=9.5, color=INK, va="top")
        ax.text(x + 0.12, 6.2, b, fontsize=9.5, color=INK, va="top")
    # speculative band between the stream and the stages
    ax.add_patch(FancyBboxPatch((4.15, 5.35), 7.65, 0.38, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#E9F2FE",
                                ec=BLUE, lw=1.4))
    ax.text(7.97, 5.54, "speculative retrieval during speech - reused at the end unless a late entity invalidates it",
            fontsize=9.4, color=BLUE, ha="center", va="center")
    stages = [
        ("1  Retrieval controller", ["turn gate: retrieval / refinement /", "presentation / chit-chat",
                                     "stability gate per clause:", "WAIT · RETRIEVE · SUPPRESS"]),
        ("2  Multi-intent decomposer", ["self-repair, spoken numbers", "request / context / modifier",
                                        "context carry-over, anaphora", "fragment merging"]),
        ("3  Retrieval & fusion", ["BM25 + bge-small dense", "RRF fusion (k=60)", "MiniLM cross-encoder rerank",
                                   "cross-query RRF + dedup"]),
        ("4  Session-aware synthesis", ["versioned answer (v1 -> v2)", "delta refine / presentation",
                                        "extractive cited claims", "grounding check + uncertainty"]),
    ]
    xs = [0.2, 4.15, 8.1, 12.05]
    for (t, lines), x in zip(stages, xs):
        box(ax, x, 2.75, 3.7, 2.4, t, lines)
    for x in xs[:-1]:
        arrow(ax, x + 3.72, 3.95, x + 3.93, 3.95, color=PURPLE, lw=2.2)
    arrow(ax, 2.05, 5.93, 2.05, 5.18, color=PURPLE, lw=2.2)
    # session memory below synthesis, feeding back to controller / decomposer
    box(ax, 8.1, 1.05, 3.7, 1.25, "Session memory", ["ephemeral: versions, claims, entities"], color=GREEN,
        fill="#F0FAF5")
    arrow(ax, 11.82, 2.1, 12.4, 2.73, color=GREEN, style="<|-|>", lw=1.8)
    arrow(ax, 8.08, 1.7, 6.0, 2.73, color=GREEN, style="-|>", ls="--")
    ax.text(5.55, 1.72, "entities / topic\nfor follow-ups", fontsize=9, color=GREEN, ha="center")
    # output
    arrow(ax, 13.9, 2.73, 13.9, 1.95, color=PURPLE, lw=2.2)
    ax.text(13.9, 1.85, "streamed answer\n+ [Doc_ID §N] citations\n+ uncertainty flags", fontsize=10.5, color=INK,
            fontweight="bold", ha="center", va="top")
    # telemetry
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 15.55, 0.55, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#FFF8E8",
                                ec=AMBER, lw=1.4))
    ax.text(7.97, 0.47, "5  Telemetry (JSONL): chunks · decisions · sub-queries · retrieval timing · versions · "
            "grounding · TTFT · cost per turn", fontsize=10, color="#8A5A00", ha="center", va="center")
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def timeline() -> None:
    from duplexrag.engine import DuplexEngine
    from duplexrag.stream import replay_session

    sc = [json.loads(l) for l in open(ROOT / "data" / "demo" / "scenarios.jsonl")]
    session = next(s for s in sc if s["session_id"] == "demo-workshop")
    one = {"session_id": "fig", "turns": session["turns"][:1]}
    rows = {}
    for name, spec in (("DuplexRAG (speculative)", True), ("Same stack, retrieve at end of speech", False)):
        eng = DuplexEngine(speculative=spec, log=lambda *a: None)
        recs, evs = replay_session(eng, one)
        rows[name] = (recs[0], evs)
    rec, evs = rows["DuplexRAG (speculative)"]
    t0 = rec["t_start"]
    fig, ax = plt.subplots(figsize=(16, 5.2), dpi=160)
    end = rec["t_end"] - t0
    # chunks (staggered on two rows so labels never collide)
    starts = [0.0] + [c[0] - t0 for c in rec["chunks"]]
    for k, (t, txt) in enumerate(rec["chunks"]):
        t = t - t0
        prev = starts[k]
        y = 2.95 if k % 2 == 0 else 2.5
        ax.add_patch(FancyBboxPatch((prev + 0.02, y), max(0.05, t - prev - 0.06), 0.36,
                                    boxstyle="round,pad=0.01,rounding_size=0.05", fc="#F4F1FD", ec=LINE))
        room = (starts[k + 2] if k + 2 < len(starts) else t + 2.0) - prev - 0.1   # until the next box on this row
        cap = max(4, int(room * 14))
        label = txt if len(txt) <= cap else txt[: cap - 1].rstrip() + "…"
        ax.text(prev + 0.06, y + 0.18, label, fontsize=8.3, color=INK, va="center")
    ax.text(-0.15, 2.9, "speech", ha="right", va="center", fontsize=10.5, color=MUTED)
    colors = {"provisional": BLUE, "multi_intent": GREEN, "final": AMBER, "refinement": PURPLE}
    for y, (name, (r, ev)) in zip((1.75, 0.8), rows.items()):
        ax.text(-0.15, y + 0.2, name.split(" (")[0] if y > 1 else "no speculation", ha="right", va="center", fontsize=10.5,
                color=MUTED)
        done = [e for e in ev if e["event"] == "retrieval_completed"]
        for k, e in enumerate(done):
            s, f = e["t_dispatch"] - t0, e["t_stream"] - t0
            ax.add_patch(FancyBboxPatch((s, y + (k % 2) * 0.22), max(f - s, 0.04), 0.18, boxstyle="square,pad=0",
                                        fc=colors.get(e["trigger"], MUTED), ec="none", alpha=0.9))
        ft = next(e for e in ev if e["event"] == "first_token")["t_stream"] - t0
        ax.plot([ft, ft], [y - 0.1, y + 0.55], color=INK, lw=2)
        ax.text(ft + 0.08, y + 0.5, f"first token  (+{(ft - end) * 1000:.0f} ms)", fontsize=9.5, color=INK, va="bottom")
    ax.axvline(end, color="#C0392B", lw=1.6, ls="--")
    ax.text(end + 0.05, 3.5, "end of speech", color="#C0392B", fontsize=10)
    for k, (lab, c) in enumerate([("provisional search", BLUE), ("multi-intent search", GREEN), ("search after speech", AMBER)]):
        ax.add_patch(FancyBboxPatch((0.2 + k * 3.2, 3.7), 0.35, 0.16, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(0.65 + k * 3.2, 3.78, lab, fontsize=9.5, va="center", color=INK)
    ax.set_xlim(-0.2, max(end + 4.5, 14))
    ax.set_ylim(0.5, 4.0)
    ax.set_yticks([])
    ax.set_xlabel("stream time (s) - guide Example 1 utterance, simulated ASR at 2.7 words/s")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "timeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def results(split: str = "test") -> None:
    path = ROOT / "results" / split / "summary.json"
    if not path.exists():
        print("no results yet:", path)
        return
    res = json.loads(path.read_text())
    a, b = res.get("duplexrag"), res.get("baseline")
    if not a or not b:
        return
    keys = [("G2_early_retrieval_pct", "Early retrieval\n(G2)"), ("gate_accuracy_pct", "Turn-type\naccuracy"),
            ("G3_multi_intent_pct", "Multi-intent\n(G3)"), ("retrieval_hit_at_3_pct", "Hit@3"),
            ("key_fact_recall_pct", "Key-fact\nrecall"), ("G4_claim_support_pct", "Claims\ngrounded (G4)"),
            ("G5_refinement_pass_pct", "Refinement\n(G5)")]
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=160)
    xs = range(len(keys))
    va = [a.get(k) or 0 for k, _ in keys]
    vb = [b.get(k) or 0 for k, _ in keys]
    ax.bar([x - 0.2 for x in xs], va, 0.38, color=PURPLE, label="DuplexRAG")
    ax.bar([x + 0.2 for x in xs], vb, 0.38, color="#B9B6CF", label="Turn-based baseline")
    for x, v in zip(xs, va):
        ax.text(x - 0.2, v + 1.5, f"{v:.0f}", ha="center", fontsize=10, color=PURPLE, fontweight="bold")
    for x, v in zip(xs, vb):
        ax.text(x + 0.2, v + 1.5, f"{v:.0f}", ha="center", fontsize=10, color=MUTED)
    ax.set_xticks(list(xs), [l for _, l in keys])
    ax.set_ylim(0, 112)
    ax.set_ylabel("% (held-out test split)")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["architecture", "timeline", "results"]
    if "architecture" in what:
        architecture()
    if "timeline" in what:
        timeline()
    if "results" in what:
        results()
    print("figures ->", OUT)
