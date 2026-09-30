"""Print per-turn failures (missed key facts, unflagged unanswerables, false flags, gold misses)
for a finished benchmark run: python scripts/error_analysis.py results/dev/duplexrag dev"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"(?<=\d),(?=\d)", "", s.lower()))


def main(run_dir: str, split: str) -> None:
    gold = {}
    for line in open(ROOT / "data" / "benchmark" / f"{split}.jsonl"):
        s = json.loads(line)
        for t in s["turns"]:
            gold[(s["session_id"], t["turn_id"])] = t
    for line in open(Path(run_dir) / "turns.jsonl"):
        r = json.loads(line)
        g = gold[(r["session_id"], r["turn_id"])]
        if not g["retrieval_required"]:
            continue
        problems = []
        missing = [kf for it in g["intents"] if it["answerable"] for kf in it.get("key_facts", [])
                   if norm(kf) not in norm(r["text"])]
        if missing:
            problems.append(f"missing key facts {missing}")
        unans = [it["intent"] for it in g["intents"] if not it["answerable"]]
        if unans and not r["uncertainty"]:
            problems.append(f"unanswerable not flagged: {unans}")
        if not unans and r["uncertainty"]:
            problems.append(f"false uncertainty: {[u['message'] for u in r['uncertainty']]}")
        ev = {h["label"] for it in r["intents"] if it.get("version") == r["version"] or r["kind"] != "refinement"
              for h in it["evidence"][:3]}
        for it in g["intents"]:
            if it["answerable"] and it["gold"] and not (ev & set(it["gold"])):
                problems.append(f"gold miss for '{it['intent']}' gold={it['gold']}")
        if problems:
            print(f"=== {r['session_id']} {r['turn_id']} [{g['type']}] kind={r['kind']}")
            print("   U:", g["utterance"])
            for q in r["sub_queries"]:
                print("   Q:", q["text"])
            for p in problems:
                print("   !", p)
            print("   A:", r["text"].replace("\n", "\n      "))


if __name__ == "__main__":
    main(*sys.argv[1:])
