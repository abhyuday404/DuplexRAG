"""Grounded answer composition.

Default mode is *extractive*: every claim is a corpus sentence (bullet or sentence
unit) selected per intent and cited with its ``Doc §Section`` label, so answers are
grounded by construction, cost nothing to generate and start streaming within
milliseconds. An optional LLM mode (any OpenAI-compatible endpoint) rewrites the
selected evidence and is post-verified claim by claim (see ``llm.py``).

Uncertainty is explicit, never silent:
  * an intent whose best evidence is weak (cross-encoder logit below threshold), or
    whose focus term does not exist in the corpus, or that asks for a quantity the
    evidence does not contain -> "could not be found" flag;
  * entity x aspect coverage: when an aspect intent (e.g. catering) is answered from
    per-entity documents (venue fact sheets), every sibling entity surfaced by the
    anchor intent is checked; entities without evidence get a "could not be verified"
    flag instead of being silently dropped.
"""
from __future__ import annotations

import re

import numpy as np

from .decompose import GENERIC
from .retrieve import Hit
from .session import AnswerState, Claim, Intent, Session
from .text import STOPWORDS, content_tokens, numbers_in, raw_tokens, stem

_XREF_RE = re.compile(r"\b(see|is in|are in|set out in|described in|listed in|per|under|follow(s)?)\s+(the\s+)?"
                      r"([A-Z][\w ]+\()?Doc_\d+", re.I)
QUANTITY_Q = re.compile(r"\b(how much|how many|how long|how early|how late|how soon|how quickly|how often|"
                        r"what'?s the (\w+ )?(rate|cost|price|fee|cap|limit|budget|amount|allowance|per diem)|"
                        r"what (is|are) the (\w+ )?(rate|cost|price|fee|cap|limit|budget|amount|allowance)s?|"
                        r"how far|how big|what time)\b", re.I)
DURATION_Q = re.compile(r"\b(how long|how early|how late|how soon|how quickly|how often|deadline|by when|"
                        r"how many (days|weeks|months|hours|nights|years))\b", re.I)
MONEY_Q = re.compile(r"\b(fee|fees|cost|costs|price|prices|rate|rates|charge|charges|budget|cap|caps|limit|limits|"
                     r"allowance|per diem|spend|spending|amount|pay|paid|reimburse\w*|stipend|deductible)\b", re.I)
CURRENCY_RE = re.compile(r"\b(INR|USD|EUR|GBP|SGD|JPY|Rs\.?)\s?\d|\d+(\.\d+)?\s?%|\bper cent\b", re.I)
DURATION_RE = re.compile(r"\d+\s*(-\s*\d+\s*)?(calendar |working |business )?(day|days|week|weeks|month|months|hour|"
                         r"hours|minute|minutes|year|years|night|nights)\b", re.I)
TITLE_GENERIC = {"policy", "rule", "guide", "guideline", "sheet", "fact", "venue", "process", "requirement",
                 "document", "overview", "support"}


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


class Composer:
    def __init__(self, index, models, settings):
        self.index = index
        self.models = models
        self.s = settings

    # ------------------------------------------------------------------ evidence
    def usable(self, hits: list[Hit]) -> list[Hit]:
        good = [h for h in hits if h.score >= self.s.evidence_threshold]
        # prefer current documents: drop superseded chunks when a current chunk is available
        cur = [h for h in good if self.index.chunks[h.idx].status != "superseded"]
        return (cur or good)[: self.s.evidence_per_query]

    @staticmethod
    def expected_answer_type(question: str) -> str | None:
        """What kind of figure a quantity question needs: currency | duration | number."""
        if not QUANTITY_Q.search(question):
            return None
        if DURATION_Q.search(question):
            return "duration"
        if MONEY_Q.search(question):
            return "currency"
        return "number"

    @staticmethod
    def satisfies(kind: str | None, text: str) -> bool:
        if kind is None:
            return True
        if kind == "currency":
            return bool(CURRENCY_RE.search(text))
        if kind == "duration":
            return bool(DURATION_RE.search(text))
        return bool(numbers_in(text))

    def entity_attribute_gap(self, intent: Intent) -> str | None:
        """'Is there parking at the Sector 62 centre?' -> if the entity's own document never mentions
        the asked-about attribute (while the evidence *is* about that entity), say so explicitly."""
        origin = intent.query.origin
        named = {stem(w.lower()) for w in re.findall(r"\b[A-Z][A-Za-z0-9-]+", origin)} | \
                {t for t in content_tokens(origin) if t in self.index.proper_terms}
        named -= {"i"}
        docs: dict[str, set[str]] = {}
        for c in self.index.chunks:
            title = set(content_tokens(c.short_title))
            if named & title:
                docs.setdefault(c.doc_id, title)
        if not docs or len(docs) > 2:
            return None
        ev_docs = {self.index.chunks[h.idx].doc_id for h in intent.evidence[:3]}
        if not (ev_docs & set(docs)):
            return None
        title_terms = set().union(*docs.values())
        aspect = [t for t in content_tokens(origin) if t not in title_terms and t not in GENERIC and
                  not t.isdigit() and t not in named and self.index.salience(t) > 0]
        if not aspect:
            return None
        text = set()
        for c in self.index.chunks:
            if c.doc_id in docs:
                text |= set(content_tokens(c.index_text()))
        if any(t in text for t in aspect):
            return None
        name = next(c.short_title for c in self.index.chunks if c.doc_id in docs)
        return f"{name} does not mention {' / '.join(aspect[:2])}"

    def uncertainty_reason(self, intent: Intent, selected_text: str) -> str | None:
        hits = intent.evidence
        best = hits[0].score if hits else -99.0
        if best < self.s.evidence_threshold:
            return "no sufficiently relevant evidence in the corpus"
        gap = self.entity_attribute_gap(intent)
        if gap:
            return gap
        kind = self.expected_answer_type(intent.query.origin)
        if not self.satisfies(kind, selected_text):
            return f"the evidence does not state the requested {'amount' if kind == 'currency' else kind}"
        return None

    # ------------------------------------------------------------------ claims
    def _subject_tokens(self, chunk) -> set[str]:
        return {t for t in content_tokens(chunk.short_title) if t not in TITLE_GENERIC}

    @staticmethod
    def _is_bullet(chunk, sentence: str) -> bool:
        return bool(re.search(r"^\s*([-*+]|\d+[.)])\s+" + re.escape(sentence[:25]), chunk.text, re.M))

    def claim_text(self, chunk, sentence: str, with_subject: bool = True) -> str:
        """Make a corpus unit stand on its own: add the entity and, for bare bullet
        fragments, the section heading they sit under (both are part of the cited chunk)."""
        s = sentence.strip().rstrip(";")
        s = re.sub(r"\s*\((see|per) Doc_\d+[^)]*\)", "", s)
        stoks = set(content_tokens(s))
        parts = []

        def key_term(title: str) -> str | None:
            toks = [t for t in content_tokens(title) if t not in TITLE_GENERIC and not t.isdigit()]
            return max(toks, key=self.index.salience) if toks else None

        subj = key_term(chunk.short_title)
        if with_subject and subj and subj not in stoks:
            parts.append(chunk.short_title)
        sec = key_term(chunk.section_title)
        if self._is_bullet(chunk, s) and sec and sec not in stoks and len(s.split()) < 16 and \
                (not parts or sec not in set(content_tokens(chunk.short_title))):
            parts.append(chunk.section_title)
        if parts:
            s = f"{', '.join(parts)}: {s[0].upper() + s[1:] if s else s}"
        if not re.search(r"[.!?]$", s):
            s += "."
        return s

    def select_sentences(self, intent: Intent, evidence: list[Hit]) -> list[tuple[int, int, float]]:
        """Return [(chunk_idx, sentence_idx, score)] for the claims of one intent."""
        if not evidence:
            return []
        qvec = self.models.embed_queries([intent.query.text])[0]
        qtok = {t for t in content_tokens(intent.query.text) if t not in GENERIC}
        qnums = numbers_in(intent.query.text)
        need = intent.query.constraints.get("people")
        cands = []
        top = evidence[0].score
        for rank, h in enumerate(evidence):
            ch = self.index.chunks[h.idx]
            if not ch.sentences:
                continue
            sv = self.index.sentence_vecs(h.idx)
            sims = sv @ qvec
            chunk_bonus = 0.25 * _sigmoid(h.score - top + 2.0)
            for si, sent in enumerate(ch.sentences):
                stoks = set(content_tokens(sent)) | self._subject_tokens(ch)
                lex = len(qtok & stoks) / max(1, len(qtok))
                score = float(sims[si]) + 0.30 * lex + chunk_bonus
                nums = numbers_in(sent)
                if nums:
                    score += 0.06      # specific (figures, limits, days) beats generic
                if qnums and qnums & nums:
                    score += 0.05
                if need and nums:
                    # head-count constraint: prefer capacities that can actually host the group
                    vals = [float(n) for n in nums if n.replace(".", "").isdigit()]
                    if any(need <= v <= need * 8 for v in vals):
                        score += 0.15
                    elif vals and all(v < need for v in vals):
                        score -= 0.10
                if _XREF_RE.search(sent) and len(sent.split()) < 18:
                    score -= 0.25
                if si == 0 and len(ch.sentences) > 2 and not nums:
                    score -= 0.08      # section openers restate the subject; prefer specifics
                cands.append((h.idx, si, score, ch.doc_id))
        if not cands:
            return []
        cands.sort(key=lambda c: -c[2])
        best = cands[0][2]
        docs_close = []
        for h in evidence:
            d = self.index.chunks[h.idx].doc_id
            if h.score >= top - 3.0 and h.score >= self.s.aspect_threshold and d not in docs_close:
                docs_close.append(d)
        enumerative = len(docs_close) >= 2
        limit = self.s.max_sentences_per_intent + (1 if enumerative else 0)
        chosen: list[tuple[int, int, float]] = []
        per_chunk: dict[int, int] = {}

        def redundant(ci: int, si: int) -> bool:
            v = self.index.sentence_vecs(ci)[si]
            return any(float(self.index.sentence_vecs(c)[s] @ v) > 0.92 for c, s, _ in chosen)

        if enumerative:   # one best sentence from each close entity document, nothing else
            for d in docs_close[:limit]:
                for ci, si, sc, dd in cands:
                    if dd == d and sc >= best - 0.25 and not redundant(ci, si):
                        chosen.append((ci, si, sc))
                        per_chunk[ci] = per_chunk.get(ci, 0) + 1
                        break
            return chosen
        strong = {h.idx for h in evidence if h.score >= top - 2.0}
        best_chunk = cands[0][0]
        for ci, si, sc, _ in cands:
            if len(chosen) >= limit:
                break
            floor = best - (0.18 if ci == best_chunk else 0.12)
            if ci not in strong or sc < floor or per_chunk.get(ci, 0) >= 2 or \
                    any(ci == c and si == s for c, s, _ in chosen):
                continue
            if redundant(ci, si):
                continue
            chosen.append((ci, si, sc))
            per_chunk[ci] = per_chunk.get(ci, 0) + 1
        return chosen

    def compose_intent(self, session: Session, intent: Intent, version: int) -> list[Claim]:
        evidence = self.usable(intent.evidence)
        picks = self.select_sentences(intent, evidence)
        # keep document order stable and name each entity once
        order = {ci: k for k, (ci, _, _) in enumerate(picks)}
        picks.sort(key=lambda p: (order[p[0]], p[1]))
        # repair: a quantity question must be answered with the figure if the evidence has one
        kind = self.expected_answer_type(intent.query.origin)
        chosen_text = " ".join(self.index.chunks[ci].sentences[si] for ci, si, _ in picks)
        if picks and not self.satisfies(kind, chosen_text):
            qvec = self.models.embed_queries([intent.query.text])[0]
            best = None
            for h in evidence[:3]:
                sv = self.index.sentence_vecs(h.idx)
                for si, sent in enumerate(self.index.chunks[h.idx].sentences):
                    if self.satisfies(kind, sent):
                        sc = float(sv[si] @ qvec) + 0.05 * (h.score >= evidence[0].score - 1)
                        if best is None or sc > best[2]:
                            best = (h.idx, si, sc)
            if best is not None:
                picks = [best] + picks[: max(1, len(picks) - 1)]
        claims = []
        last_doc = None
        for ci, si, _ in picks:
            ch = self.index.chunks[ci]
            text = self.claim_text(ch, ch.sentences[si], with_subject=ch.doc_id != last_doc)
            last_doc = ch.doc_id
            claims.append(Claim(session.next_claim_id(), intent.iid, text, [ch.label], version))
        intent.uncertain = self.uncertainty_reason(intent, " ".join(c.text for c in claims))
        if intent.uncertain:
            claims = []
        return claims

    # ------------------------------------------------------------------ entity x aspect coverage
    def _section_signature(self, doc_id: str) -> set[str]:
        return {stem(w) for c in self.index.chunks if c.doc_id == doc_id
                for w in raw_tokens(c.section_title) if w not in STOPWORDS}

    def coverage_gaps(self, anchor: Intent, aspects: list[Intent], anchor_claims: list[Claim]) -> list[dict]:
        entity_docs = list(dict.fromkeys(l.split(" §")[0] for c in anchor_claims for l in c.citations))
        flags = []
        for asp in aspects:
            if asp.uncertain or asp is anchor:
                continue
            asp_docs = {self.index.chunks[h.idx].doc_id for h in self.usable(asp.evidence)}
            covered = [d for d in entity_docs if d in asp_docs]
            if not covered:
                continue           # aspect answered by general policy docs, not per entity
            sig = set().union(*(self._section_signature(d) for d in covered))
            for d in entity_docs:
                if d in covered:
                    continue
                dsig = self._section_signature(d)
                if not sig or len(sig & dsig) / len(sig | dsig) < 0.3:
                    continue       # not a sibling of the per-entity documents (e.g. a directory)
                # targeted check: does ANY section of this entity cover the aspect?
                idxs = [i for i, c in enumerate(self.index.chunks) if c.doc_id == d]
                scores = self.models.rerank_pairs([(asp.query.text, self.index.chunks[i].index_text()) for i in idxs])
                best = float(scores.max()) if len(scores) else -99
                if best < self.s.aspect_threshold:
                    title = self.index.chunks[idxs[0]].short_title
                    flags.append({"intent_id": asp.iid, "doc_id": d, "best_score": round(best, 2),
                                  "message": f"{asp.label} for {title} could not be verified from the corpus."})
        return flags

    # ------------------------------------------------------------------ rendering
    @staticmethod
    def _fmt_claim(c: Claim) -> str:
        return f"{c.text} [{', '.join(c.citations)}]"

    def render(self, answer: AnswerState, delta_ids: list[str] | None = None, constraint: str = "") -> str:
        intents = [i for i in answer.intents.values() if i.status == "active"]
        by_intent = {i.iid: [c for c in answer.active_claims() if c.intent_id == i.iid] for i in intents}
        multi = len(intents) > 1

        def line(i: Intent) -> str:
            cl = by_intent[i.iid]
            if i.uncertain or not cl:
                return f"{i.label}: I could not find this in the knowledge base ({i.uncertain or 'no evidence'})."
            body = " ".join(self._fmt_claim(c) for c in cl)
            return f"{i.label}: {body}" if multi or delta_ids else body

        lines: list[str] = []
        if delta_ids:
            retained = [i for i in intents if i.iid not in delta_ids]
            for i in retained:
                cl = by_intent[i.iid]
                if cl:
                    lines.append(f"Still applies - {i.label}: {self._fmt_claim(cl[0])}")
            head = f"Update ({constraint.strip().rstrip('.')})" if constraint else "Update"
            lines.append(head + ":")
            lines += [line(i) for i in intents if i.iid in delta_ids]
        else:
            lines += [line(i) for i in intents]
        if answer.uncertainty:
            lines.append("Not verified: " + " ".join(u["message"] for u in answer.uncertainty))
        return "\n".join(lines)

    def present(self, answer: AnswerState, style: str, count: int | None) -> str:
        """Presentation-only transform of the current answer: no retrieval, same citations."""
        intents = [i for i in answer.intents.values() if i.status == "active"]
        firsts = []
        for i in intents:
            cl = [c for c in answer.active_claims() if c.intent_id == i.iid]
            if cl:
                firsts.append((i, cl))
        if style == "bullets":
            n = max(1, min(count or 3, 6))
            if not firsts:
                return answer.text
            if len(firsts) >= n:        # group intents into n bullets, keep each intent's key claim
                groups = [firsts[k::n] for k in range(n)]
                groups = [sorted(g, key=lambda x: intents.index(x[0])) for g in groups if g]
                bullets = [" ".join(self._fmt_claim(cl[0]) for _, cl in g) for g in groups]
            else:                       # fewer intents than bullets: spread each intent's claims
                pool = [c for _, cl in firsts for c in cl]
                bullets = [self._fmt_claim(c) for c in pool[:n]]
            text = "\n".join(f"- {b}" for b in bullets)
        elif style == "short":
            text = " ".join(self._fmt_claim(cl[0]) for _, cl in firsts) or answer.text
        else:  # repeat
            text = answer.text
        if answer.uncertainty and style != "repeat":
            text += "\nNot verified: " + " ".join(u["message"] for u in answer.uncertainty)
        return text
