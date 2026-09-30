"""Calibrate the evidence / uncertainty thresholds on the DEV split only.

Replays dev sessions with the evidence threshold disabled and records, for every
answered intent, its best cross-encoder score, whether its evidence hits the turn's
gold sections, and whether the turn is unanswerable. Prints the threshold that best
separates answerable from unanswerable intents.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplexrag.config import load_settings  # noqa: E402
from duplexrag.engine import DuplexEngine  # noqa: E402
from duplexrag.stream import replay_session  # noqa: E402
from duplexrag.text import content_tokens  # noqa: E402
from duplexrag.decompose import GENERIC  # noqa: E402


def main() -> None:
    eng = DuplexEngine(load_settings(evidence_threshold=-99.0), log=lambda *a: None)
    rows = []
    for line in open(ROOT / "data" / "benchmark" / "dev.jsonl"):
        sess = json.loads(line)
        gold = {t["turn_id"]: t for t in sess["turns"]}
        for r in replay_session(eng, sess)[0]:
            g = gold[r["turn_id"]]
            if r["kind"] not in ("retrieval", "refinement"):
                continue
            gold_labels = {x for it in g["intents"] for x in it["gold"]}
            unans = g["intents"] and all(not it["answerable"] for it in g["intents"])
            for it in r["intents"]:
                if it["version"] != r["version"]:
                    continue
                ev = it["evidence"]
                best = ev[0]["score"] if ev else -99
                hit = any(h["label"] in gold_labels for h in ev[:3])
                own = [t for t in content_tokens(it["query"]["origin"]) if t not in GENERIC and not t.isdigit()]
                unknown = [t for t in own if len(t) >= 4 and eng.index.salience(t) == 0.0]
                ev_text = set()
                for h in ev[:3]:
                    ev_text |= set(content_tokens(eng.index.chunks[eng.index.by_label[h["label"]]].index_text()))
                known = [t for t in own if eng.index.salience(t) > 0]
                cov = sum(t in ev_text for t in own) / max(1, len(own))
                focus = max(known, key=eng.index.salience) if known else None
                rows.append({"sid": r["session_id"], "tid": r["turn_id"], "label": it["label"], "best": best,
                             "hit": hit, "unans": bool(unans), "unknown": unknown, "cov": cov,
                             "focus": focus, "focus_in": focus in ev_text if focus else None})
    for r in sorted(rows, key=lambda r: r["best"]):
        tag = "UNANS" if r["unans"] else ("hit" if r["hit"] else "MISS")
        print(f"{r['best']:7.2f} {tag:5s} cov={r['cov']:.2f} focus={r['focus']}:{r['focus_in']} {r['sid']} "
              f"{r['tid']} {r['label'][:40]:40s} unknown={r['unknown']}")
    pos = [r["best"] for r in rows if not r["unans"] and r["hit"]]
    neg = [r["best"] for r in rows if r["unans"]]
    best_t, best_acc = None, -1
    for t in [x / 2 for x in range(-12, 12)]:
        acc = sum(p >= t for p in pos) + sum(n < t for n in neg)
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"answerable-with-gold-hit n={len(pos)}  unanswerable n={len(neg)}  best threshold={best_t} "
          f"correct={best_acc}/{len(pos) + len(neg)}")


if __name__ == "__main__":
    main()
