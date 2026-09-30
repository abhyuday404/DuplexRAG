"""Claim-level grounding verification.

A claim is *supported* when every citation resolves to a real corpus chunk that was
actually retrieved for this answer, every number in the claim appears in the cited
text, and most of the claim's content words appear there too. Citations that do not
exist in the corpus are counted as *fabricated* (the automated gate requires zero).
"""
from __future__ import annotations

import re

from .text import content_tokens, numbers_in

CITE_RE = re.compile(r"\[((?:Doc_[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*§\s*\d+(?:\s*[,;]\s*[A-Za-z0-9_]+\s*§\s*\d+)*)\]")
LABEL_RE = re.compile(r"([A-Za-z0-9_]+)\s*§\s*(\d+)")


def extract_citations(text: str) -> list[str]:
    out = []
    for m in CITE_RE.finditer(text):
        for d, s in LABEL_RE.findall(m.group(1)):
            out.append(f"{d} §{s}")
    return out


def strip_citations(text: str) -> str:
    return CITE_RE.sub("", text).strip()


def support_score(claim: str, cited_texts: list[str]) -> float:
    if not cited_texts:
        return 0.0
    body = strip_citations(claim)
    source = " ".join(cited_texts)
    nums = numbers_in(body)
    if nums and not nums <= numbers_in(source):
        return 0.0
    ctoks = set(content_tokens(body))
    if not ctoks:
        return 1.0
    stoks = set(content_tokens(source))
    return len(ctoks & stoks) / len(ctoks)


def verify(claims: list[dict], index, retrieved: set[str], threshold: float = 0.7) -> dict:
    """claims: [{"text", "citations"}]. Returns per-claim results and aggregate rates."""
    results = []
    fabricated, unretrieved = [], []
    for c in claims:
        cites = c.get("citations") or extract_citations(c["text"])
        texts = []
        for lab in cites:
            i = index.by_label.get(lab)
            if i is None:
                fabricated.append(lab)
                continue
            if retrieved and lab not in retrieved:
                unretrieved.append(lab)
            ch = index.chunks[i]
            texts.append(ch.index_text())
        score = support_score(c["text"], texts)
        results.append({"cid": c.get("cid"), "supported": bool(cites) and score >= threshold and not
                        any(l not in index.by_label for l in cites), "score": round(score, 3), "citations": cites})
    n = len(results)
    return {
        "claims": n,
        "supported": sum(r["supported"] for r in results),
        "support_rate": (sum(r["supported"] for r in results) / n) if n else 1.0,
        "fabricated_ids": fabricated,
        "unretrieved_ids": unretrieved,
        "results": results,
    }
