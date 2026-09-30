"""Validate the demo corpus: format, sequential § sections and engineered hard cases."""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duplexrag.corpus import load_corpus, parse_front_matter  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
FOOD = re.compile(r"\b(cater\w*|food|meal\w*|lunch|breakfast|dinner|snack\w*|refreshment\w*|tea|coffee|buffet)\b", re.I)


def main() -> int:
    errors: list[str] = []
    files = sorted(CORPUS.glob("Doc_*.md"))
    for f in files:
        raw = f.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        if not f.name.startswith(meta.get("doc_id", "?") + "_"):
            errors.append(f"{f.name}: doc_id {meta.get('doc_id')} does not match file name")
        for key in ("title", "category", "effective_date", "status"):
            if key not in meta:
                errors.append(f"{f.name}: missing front-matter key {key}")
        secs = [int(s) for s in re.findall(r"^## §(\d+) ", body, re.M)]
        if secs != list(range(1, len(secs) + 1)) or len(secs) < 3:
            errors.append(f"{f.name}: sections not sequential: {secs}")
        bad = {ch for ch in raw if ord(ch) > 127 and ch != "§"}
        if bad:
            errors.append(f"{f.name}: non-ASCII characters {bad}")
    for doc in ("Doc_04", "Doc_10"):
        f = next(CORPUS.glob(f"{doc}_*.md"))
        hits = FOOD.findall(f.read_text(encoding="utf-8"))
        if hits:
            errors.append(f"{doc} must not mention catering/food, found {hits}")
    meta23, _ = parse_front_matter(next(CORPUS.glob("Doc_23_*.md")).read_text(encoding="utf-8"))
    if meta23.get("status") != "superseded" or meta23.get("superseded_by") != "Doc_15":
        errors.append("Doc_23 must be superseded_by Doc_15")

    chunks = load_corpus(CORPUS)
    per_doc = defaultdict(int)
    for c in chunks:
        per_doc[c.doc_id] += 1
    words = [len(c.text.split()) for c in chunks]
    print(f"documents={len(files)} chunks={len(chunks)} sentences={sum(len(c.sentences) for c in chunks)} "
          f"words/chunk min={min(words)} mean={sum(words) / len(words):.0f} max={max(words)}")
    if len(files) != 46:
        errors.append(f"expected 46 documents, found {len(files)}")
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
