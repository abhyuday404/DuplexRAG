"""Print clause segmentation and sub-queries for a benchmark split (use on the dev split only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duplexrag.config import load_settings  # noqa: E402
from duplexrag.decompose import Decomposer, normalize_spoken  # noqa: E402
from duplexrag.index import HybridIndex  # noqa: E402
from duplexrag.models import Models  # noqa: E402


def main(split: str = "dev") -> None:
    s = load_settings()
    idx = HybridIndex.load_or_build(s, Models.get(s))
    d = Decomposer(idx, s)
    for line in open(Path(s.corpus_dir).parents[0] / "benchmark" / f"{split}.jsonl"):
        sess = json.loads(line)
        sctx = None
        for t in sess["turns"]:
            cl = d.segment(t["utterance"], final=True)
            if t["type"] in ("presentation", "chitchat"):
                continue
            qs = d.subqueries(cl, sctx)
            print(f"{sess['session_id']} {t['turn_id']} {t['type'][:5]} gold={len(t['intents'])} got={len(qs)} :: "
                  f"{normalize_spoken(t['utterance'])}")
            for c in cl:
                print(f"     [{c.kind[:3]}] {c.text}  c={c.content} p={c.proper}")
            for q in qs:
                print(f"       -> {q.text}")
            sctx = d.session_context(cl)


if __name__ == "__main__":
    main(*sys.argv[1:])
