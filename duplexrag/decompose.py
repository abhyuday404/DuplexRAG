"""Spoken-utterance normalisation, clause segmentation and multi-intent decomposition.

Pipeline (all rule-based, corpus-aware, O(words)):

1. ``normalize_spoken`` - drop fillers, turn spoken numbers into digits
   ("a hundred and ten" -> "110") and apply self-repairs
   ("the hotel, sorry, the flight class" -> "the flight class").
2. ``segment`` - split at punctuation and coordinators into clauses and label
   each one: request (an information need), context (sets entities/constraints:
   "it's in Bengaluru"), modifier (attaches to the previous request) or filler.
   Only clauses followed by a boundary are *complete*; the tail clause of a
   partial utterance is still in progress.
3. ``Decomposer.subqueries`` - turn request clauses into search-ready
   sub-queries, carrying shared context (entities/constraints mentioned anywhere
   in the utterance or earlier in the session) into each one, and merging
   near-duplicates so the retriever is not flooded (over-fragmentation guard).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieve import SubQuery
from .text import COMMA_JOIN, DANGLING, STOPWORDS, WORD_NUMBERS, content_tokens, raw_tokens, stem

FILLERS = [
    r"\bum+\b", r"\buh+\b", r"\berm+\b", r"\bhmm+\b", r"\bmm+\b", r"\byou know\b", r"\bi mean\b(?!,)",
    r"\bkind of\b", r"\bsort of\b", r"\bbasically\b", r"\bliterally\b", r"\blike,", r",\s*like\b",
    r"\bso yeah\b", r"\bokay so\b", r"\bjust\b", r"\bactually\b(?!,? no)", r"\breally\b", r"\bpretty much\b",
]
_FILLER_RE = re.compile("|".join(FILLERS), re.I)

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_NUM_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|" \
            r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|" \
            r"thousand|lakh|lakhs)"
_NUM_SEQ_RE = re.compile(rf"\b(?:a\s+)?{_NUM_WORD}(?:(?:\s+|-)(?:and\s+)?{_NUM_WORD})*\b", re.I)

REPAIR_MARKERS = r"(?:no sorry|no wait|no no|sorry|i mean|or rather|scratch that|no)"
_REPAIR_RE = re.compile(rf"(^|[,.;?!])([^,.;?!]*?)\s*,\s*{REPAIR_MARKERS}\s*,\s*", re.I)

COORD = r"(?:oh and also|oh and|and also|and then|and what about|as well as|along with|and|plus)"
_SPLIT_RE = re.compile(rf"([,;?!.]+)\s*|\s+\b({COORD})\b\s+", re.I)
_LEAD_RE = re.compile(r"^((so|and|oh|okay|ok|hey|hi|hello|well|also|but|hmm|right|alright|now|then|plus|wait|"
                      r"yeah|yes|great|cool|thanks|thank you|quick question|one thing|one more thing)\b[,\s]*)+", re.I)
_PREP_START = re.compile(r"^(for|at|in|on|to|from|with|within|about|regarding|during|after|before|like|maybe|"
                         r"around|roughly|approximately|more like|by|until|than|because|since|cause|so that|"
                         r"as long as|unless|\d+)\b", re.I)
_WH_ANYWHERE = re.compile(r"\b(what|what's|how|which|when|where|who|whether)\b", re.I)

QUESTION_START = re.compile(
    r"^(what|what's|whats|how|which|when|when's|where|where's|who|who's|whom|whose|why|is|are|am|was|were|do|does|"
    r"did|can|could|should|shall|will|would|may|might|must|any|anything|if)\b", re.I)
REQUEST_CUE = re.compile(
    r"\b(need|needs|want|wanted|looking for|find|tell me|know|check|summari[sz]e|explain|plan|book|recommend|"
    r"suggest|help me|options|allowed|eligible|entitled|how much|how many|how long|how early|how late|what happens|"
    r"any idea|wondering)\b", re.I)
CONTEXT_START = re.compile(
    r"^((it's|it is|it'll be|it will be|it was|its|it gets|it keeps|it has|we're|we are|we'll be|we've|we have|"
    r"i'm|i am|i've|i have|i'll be|i'd be|i got|i was|this is|that's|there'll be|there will be|turns out|"
    r"by the way|btw|heading|travell?ing|flying|going to)\b|(my|our|the|this|that) [\w' -]{1,30}?"
    r"(\b(is|was|are|were|has|had|got|will|includes?|keeps?|dies|died|expires?|runs?)\b|'s\b)"
    r"|(the )?(trip|event|workshop|booking|flight|meeting|training|venue|receipt|rate|hotel|offsite|launch|team|claim|"
    r"stay|laptop|phone)\b)", re.I)
PREAMBLE_RE = re.compile(
    r"^((so|and|also|oh|okay|ok|hey|hi|hello|well|right|quick question|one more thing|one thing|hmm|wait|"
    r"alright|now)[,\s]+)*((i|we)\s+(also\s+)?(need|want|would like|'d like|wanted)\s+(to\s+)?(know|check|find out|"
    r"understand|get|see|ask)?\s*(about|whether|if)?\s*|can you\s+(please\s+)?(tell me|summari[sz]e|check|"
    r"explain|find|give me|let me know)?\s*(about)?\s*|could you\s+(please\s+)?(tell me|summari[sz]e|check|"
    r"explain|find|give me)?\s*|please\s+|tell me\s+(about\s+)?|do you know\s+|i was wondering\s+(if|whether)?\s*|"
    r"any idea\s+)?", re.I)
PEOPLE_RE = re.compile(r"\b(\d{1,5})\s*(?:\+\s*)?(people|persons|attendees|participants|guests|pax|folks|"
                       r"delegates|heads|employees|members|seats|of us)\b", re.I)
ANAPHORA_RE = re.compile(r"\b(it|its|it's|they|them|their|there|that|those|these|this|same|such|ones|him|her|"
                         r"what about|how about)\b|\b(the|that|this|new|same|which|other) one\b", re.I)
_EXISTENTIAL_RE = re.compile(r"\b(is|are|was|were|will)\s+there\b|\bthere\s+(is|are|was|were|'s|will)\b", re.I)
GENERIC = frozenset(stem(w) for w in """
thanks thank helpful great perfect cool awesome nice good fine sure right maybe stuff question quick help idea lot
bit time today tomorrow tonight anyway pretty okay alright thing things wonderful brilliant appreciate much bye
hello hey morning evening happen happens sorry wait next week month year day days everyone someone something
think supposed try first mention forgot realistically end whatever around roughly approximately people
run running look looking planning go going come coming take taking make get got keep keeps need want use have do
bring send ask start say see hand tell know person last moved move moving changed change mention place stuff
""".split())


def _words_to_number(phrase: str) -> int | None:
    toks = [t for t in re.split(r"[\s-]+", phrase.lower()) if t and t not in ("and", "a")]
    total, current = 0, 0
    for t in toks:
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == "hundred":
            current = max(current, 1) * 100
        elif t == "thousand":
            total += max(current, 1) * 1000
            current = 0
        elif t in ("lakh", "lakhs"):
            total += max(current, 1) * 100000
            current = 0
        else:
            return None
    return total + current


def _replace_numbers(text: str) -> str:
    def rep(m: re.Match) -> str:
        phrase = m.group(0)
        if phrase.strip().lower() in ("one", "a", "a one"):
            return phrase
        n = _words_to_number(phrase)
        return str(n) if n is not None else phrase
    return _NUM_SEQ_RE.sub(rep, text)


def _apply_repairs(text: str) -> str:
    """Self-repair: 'X, sorry, Y' drops the reparandum X.

    * repair starts with a function word ('what flight class') -> cut back to the same word
      ('what hotel'), or drop the whole segment if that word is absent;
    * repair starts with a content word ('Hyderabad') -> drop only the words after the
      last function word ('in Mumbai' -> 'in').
    """
    for _ in range(3):
        m = _REPAIR_RE.search(text)
        if not m:
            break
        lead, before, after = m.group(1), m.group(2), text[m.end():]
        words = before.split()
        first_after = raw_tokens(after)[0] if raw_tokens(after) else ""
        cut = 0
        if first_after in STOPWORDS or first_after in DANGLING:
            for i in range(len(words) - 1, -1, -1):
                if words[i].lower() == first_after:
                    cut = i
                    break
        else:
            for i in range(len(words) - 1, -1, -1):
                if words[i].lower() in DANGLING:
                    cut = i + 1
                    break
        kept = " ".join(words[:cut])
        text = text[:m.start()] + lead + (" " + kept if kept else "") + " " + after
    return text


ALIASES = [  # common spoken variants of place / country names (general knowledge, not corpus facts)
    (r"\b(the )?U\.?S\.?A?\b|\bthe States\b|\bAmerica\b", "USA"), (r"\bU\.?K\.?\b|\bBritain\b|\bEngland\b", "UK"),
    (r"\bBangalore\b", "Bengaluru"), (r"\bBombay\b", "Mumbai"), (r"\bMadras\b", "Chennai"),
    (r"\bGurgaon\b", "Gurugram"), (r"\bPoona\b", "Pune"), (r"\bwi-?fi\b", "Wi-Fi"),
]


def normalize_spoken(text: str) -> str:
    t = " " + text.strip() + " "
    for pat, rep in ALIASES:
        t = re.sub(pat, rep, t)
    t = _apply_repairs(t)
    t = _FILLER_RE.sub(" ", t)
    t = _replace_numbers(t)
    t = re.sub(r"\s+([,.;?!])", r"\1", t)
    t = re.sub(r"([,;])\s*[,;]+", r"\1", t)
    t = re.sub(r"^\s*[,;]\s*", "", t)
    # a comma right after a function word or pronoun contraction is a pause, not a boundary
    t = re.sub(r"\b(" + "|".join(sorted(COMMA_JOIN, key=len, reverse=True)) + r")\s*,\s+", r"\1 ", t, flags=re.I)
    t = re.sub(r"\b(\w+['\u2019](re|m|s|ve|ll|d))\s*,\s+", r"\1 ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Clause:
    text: str
    kind: str                   # request | context | modifier | filler
    complete: bool
    joiner: str = ""            # boundary that preceded this clause
    content: list[str] = field(default_factory=list)   # salient stemmed terms
    proper: list[str] = field(default_factory=list)    # proper-noun terms (stemmed, lower)
    people: int | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "complete": self.complete, "content": self.content,
                "proper": self.proper, "people": self.people}


class Decomposer:
    def __init__(self, index, settings):
        self.index = index
        self.s = settings

    # ------------------------------------------------------------------ term analysis
    def salient_terms(self, text: str) -> tuple[list[str], list[str]]:
        """Return (content_terms, proper_terms) judged against the corpus vocabulary."""
        proper, content = [], []
        # proper nouns: capitalised mid-utterance words, or words the corpus only uses capitalised
        for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9-]*)(?:['’](?:s|re|ll|d|ve|m))?\b", text):
            w = m.group(1)
            s = stem(w.lower())
            acronym = w.isupper() and len(w) >= 2 and w not in ("OK", "AM", "PM")
            if not acronym and (w.lower() in STOPWORDS or s in GENERIC or len(w) < 3):
                continue
            capitalised_mid = (w[0].isupper() and m.start() > 0 and w != "I"
                               and not re.search(r"(^|[.?!,;]\s*|\b(so|and|oh|okay|hey|hi|well)\s+)$",
                                                 text[:m.start()], re.I))
            if acronym or s in self.index.proper_terms or capitalised_mid:
                if s not in proper:
                    proper.append(s)
        for tok in content_tokens(text):
            if tok in GENERIC or tok in proper:
                continue
            if tok.isdigit() or self.index.salience(tok) >= 0.30:
                if tok not in content:
                    content.append(tok)
        return content, proper

    def _is_request(self, text: str) -> bool:
        return bool(QUESTION_START.search(text) or REQUEST_CUE.search(text) or text.rstrip().endswith("?"))

    # ------------------------------------------------------------------ segmentation
    def segment(self, raw: str, final: bool) -> list[Clause]:
        text = normalize_spoken(raw)
        pieces: list[tuple[str, str]] = []     # (joiner, clause text)
        pos, joiner = 0, ""
        for m in _SPLIT_RE.finditer(text):
            seg = text[pos:m.start()].strip()
            if seg:
                pieces.append((joiner, seg + ("?" if m.group(1) and "?" in m.group(1) else "")))
            joiner = (m.group(1) or m.group(2) or "").strip().lower()
            pos = m.end()
        tail = text[pos:].strip()
        tail_complete = final
        if tail:
            pieces.append((joiner, tail))
        elif pieces:
            tail_complete = True        # text ended exactly on a boundary
        clauses: list[Clause] = []
        for i, (j, seg) in enumerate(pieces):
            complete = i < len(pieces) - 1 or tail_complete
            lead = _LEAD_RE.match(seg)
            if lead and lead.end() < len(seg):
                if re.search(r"\b(and|also|plus)\b", lead.group(0), re.I):
                    j = j or "and"
                seg = seg[lead.end():].strip()
            elif lead:
                seg = ""
            content, proper = self.salient_terms(seg)
            pm = PEOPLE_RE.search(seg)
            people = int(pm.group(1)) if pm else None
            question = bool(QUESTION_START.search(seg) or seg.rstrip().endswith("?"))
            has_req = self._is_request(seg)
            has_prev_request = any(c.kind == "request" for c in clauses)
            if not content and not proper and people is None:
                kind = "qempty" if (question or has_req) and seg else "filler"
            elif CONTEXT_START.search(seg) and not QUESTION_START.search(seg) and not (
                    _WH_ANYWHERE.search(seg) and seg.rstrip().endswith("?")):
                kind = "context"
            elif _PREP_START.search(seg) and not _WH_ANYWHERE.search(seg) and not REQUEST_CUE.search(seg):
                kind = "modifier" if has_prev_request else "context"
            elif has_req:
                kind = "request"
            elif has_prev_request and j in (",", "and", "plus", "as well as", "along with", "and also", "oh and",
                                            "oh and also", "and what about", ";"):
                kind = "request"            # list item / ellipsis: "I need X, Y and Z"
            elif not has_prev_request:
                kind = "request"
            else:
                kind = "modifier"
            clauses.append(Clause(seg, kind, complete, j, content, proper, people))
        merged: list[Clause] = []
        for c in clauses:
            if c.kind == "modifier" and merged:
                prev = next((m for m in reversed(merged) if m.kind == "request"), None)
                if prev is not None:
                    _absorb(prev, c)
                    continue
                c.kind = "context"
            if c.kind == "qempty":
                # "... lost my card in Singapore, what am I supposed to do?" -> the previous clause is the need
                prev = next((m for m in reversed(merged) if m.kind in ("context", "request")), None)
                if prev is not None:
                    _absorb(prev, c)
                    prev.kind = "request"
                continue
            merged.append(c)
        return merged

    # ------------------------------------------------------------------ sub-queries
    @staticmethod
    def clean_request(text: str) -> str:
        t = PREAMBLE_RE.sub("", text.strip()).strip(" ,?.")
        t = re.sub(r"^((i|we)\s+)?((also|still)\s+)?((need|want|have|'d like|would like|am going|are going)\s+to\s+)?"
                   r"(plan|organi[sz]e|arrange|host|hold|run|set up|put together|book|find|get)\s+", "", t, flags=re.I)
        t = re.sub(r"^(the|a|an|any|whatever|and|some)\s+", "", t, flags=re.I)
        t = re.sub(r"\b(about|around|roughly|approximately|like)\s+(\d)", r"\2", t, flags=re.I)
        t = re.sub(r"\s+(there are|there is|we need|we have|you have|out there)$", "", t, flags=re.I)
        return t.strip() or text.strip(" ,?.")

    def label(self, text: str) -> str:
        """Short human label for an intent, e.g. 'How much paternity leave do I get' -> 'Paternity leave'."""
        t = self.clean_request(text)
        wh = r"(what's|whats|what is|what are|what|which|how much|how many|how long|how early|how late|how often|" \
             r"how quickly|how soon|how|when's|when is|when|where's|where|who's|who|whether|is there|are there)"
        m = re.search(rf"\b{wh}\b", t, flags=re.I)
        if m and m.start() > 0 and len(t[:m.start()].split()) <= 6:
            t = t[m.start():]                       # "for that same trip what's the cap" -> "what's the cap"
        t = re.sub(rf"^{wh}\s+((the|a|an|any|my|our)\s+)?", "", t, flags=re.I)
        t = re.sub(r"^(do|does|did|can|could|should|would|will|is|are|am)\s+(i|we|they|you|it)\s+", "", t, flags=re.I)
        short = re.sub(r"\s+\b(do|does|did|can|could|should|would|will|am|is|are|have|has)\s+(i|we|you|they|it|he|she)"
                       r"\b.*$", "", t, flags=re.I)
        short = re.sub(r"\s+\b(i|we|you|they)\s+(have to|need to|must|should|can|get|am|are|'re|'m)\b.*$", "", short,
                       flags=re.I)
        if len(short.split()) >= 1 and short != t:
            t = short
        t = re.sub(r"\b(by when|by|for|of|to|in|at|on|with)$", "", t.strip(), flags=re.I).strip()
        words = t.split()
        if len(words) > 7:
            words = words[:7]
            while words and words[-1].lower() in DANGLING:
                words.pop()
        t = " ".join(words)
        return t[:1].upper() + t[1:] if t else text

    def context_terms(self, clauses: list[Clause]) -> dict:
        """Shared context for the utterance: entities from context clauses and the anchor
        (first request) clause, topic terms from context clauses, the anchor's own topic
        terms (for anaphoric follow-up clauses) and any head-count constraint."""
        requests = [c for c in clauses if c.kind == "request"]
        anchor = requests[0] if requests else None
        proper: list[str] = []
        for c in clauses:
            if c.kind == "context" or c is anchor:
                proper += [p for p in c.proper if p not in proper]
        people = next((c.people for c in clauses if c.people), None)

        def top(c: Clause, k: int = 3) -> list[str]:
            ranked = sorted((t for t in c.content if not t.isdigit() and t not in c.proper),
                            key=lambda t: -self.index.salience(t))
            return ranked[:k]

        topic: list[str] = []
        for c in clauses:
            if c.kind == "context":
                topic += [t for t in top(c, 2) if t not in topic]
        return {"proper": proper, "topic": topic, "anchor_topic": top(anchor) if anchor else [], "people": people,
                "anchor": anchor, "all_proper": [p for c in clauses for p in c.proper]}

    def subqueries(self, clauses: list[Clause], session_ctx: dict | None = None, *, include_incomplete: bool = False,
                   id_prefix: str = "q", surface_text: str = "") -> list[SubQuery]:
        ctx = self.context_terms(clauses)
        surf = _surface_map(surface_text or " ".join(c.text for c in clauses))
        if session_ctx:
            surf = {**_surface_map(" ".join(session_ctx.get("surface", []))), **surf}
        requests = [c for c in clauses if c.kind == "request" and (c.complete or include_incomplete)]
        requests = [c for c in requests if len(c.content) + len(c.proper) + (c.people is not None) >= 1]
        # a follow-up turn with no local entities/topic whose anchor is anaphoric ("those lab machines",
        # "that same trip") inherits the session's entities for every sub-query of the turn
        anchor = ctx["anchor"]
        session_topic: list[str] = []
        if (session_ctx and anchor is not None and not ctx["all_proper"] and not ctx["topic"]
                and _is_anaphoric(self.clean_request(anchor.text))):
            ctx["proper"] = list(session_ctx.get("proper", []))[:3]
            session_topic = list(session_ctx.get("topic", []))[:3]
        out: list[SubQuery] = []
        for i, c in enumerate(requests):
            body = self.clean_request(c.text)
            have = set(content_tokens(body)) | set(c.proper)
            extra: list[str] = []
            anaphoric = _is_anaphoric(body)
            vague = len(c.content) < 3          # few content words of its own -> needs shared context
            # 1) entities named in context clauses / the anchor clause apply to every sub-query
            extra += [p for p in ctx["proper"] if p not in have]
            # 2) topic terms of this utterance's context clauses ("I'm at director level") always apply
            extra += [t for t in ctx["topic"] if t not in have]
            # session topic terms only help clauses that carry (almost) no content of their own
            if len(c.content) < 1:
                extra += [t for t in session_topic if t not in have]
            # 3) "how often do we have to change it" -> resolve against the anchor clause
            if c is not ctx["anchor"] and not c.proper and anaphoric:
                extra += [t for t in ctx["anchor_topic"] if t not in have]
            extra = list(dict.fromkeys(extra))[:6]
            n = c.people or ctx["people"]
            facet = ""
            suffix = ""
            if n and (c.people or re.search(r"\b(venue|room|hall|fit|hold|capacity|seat|accommodat|space|host)",
                                            body, re.I)):
                # a head-count implies a capacity need; lead with it (cross-encoders weight query onsets)
                facet = f"room or venue capacity for {n} people: "
                body = re.sub(rf"\bfor\s+{n}\s+\w+", "", body).strip()
            elif n:
                suffix = f" for an event with {n} people"     # thresholds often depend on group size
            text = facet + body + suffix + ("" if not extra else " " + " ".join(surf.get(t, t) for t in extra))
            kw = " ".join(dict.fromkeys(content_tokens(text)))
            out.append(SubQuery(qid=f"{id_prefix}{i + 1}", text=text.strip(), keywords=kw, origin=c.text,
                                anchor=(c is ctx["anchor"]), constraints={"people": n} if n else {}))
        return out

    def session_context(self, clauses: list[Clause]) -> dict:
        """What a later turn may inherit (entities + topic terms), kept in session memory only."""
        ctx = self.context_terms(clauses)
        proper = list(dict.fromkeys(ctx["proper"]))[:4]
        topic = list(dict.fromkeys(ctx["anchor_topic"] + ctx["topic"]))[:4]
        return {"proper": proper, "topic": topic, "people": ctx["people"],
                "surface": [c.text for c in clauses]}


def _surface_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for w in re.findall(r"[A-Za-z][A-Za-z0-9-]*", text):
        out.setdefault(stem(w.lower()), w)
    return out


def _absorb(prev: Clause, c: Clause) -> None:
    prev.text = f"{prev.text} {c.text}"
    prev.content += [t for t in c.content if t not in prev.content]
    prev.proper += [t for t in c.proper if t not in prev.proper]
    prev.people = prev.people or c.people
    prev.complete = c.complete


def _is_anaphoric(text: str) -> bool:
    return bool(ANAPHORA_RE.search(_EXISTENTIAL_RE.sub(" ", text)))


def ends_dangling(text: str) -> bool:
    toks = raw_tokens(text)
    return not toks or toks[-1] in DANGLING or bool(re.search(r"[-,]\s*$", text))


def spoken_count(text: str) -> int | None:
    m = re.search(r"\b(\d+|" + "|".join(WORD_NUMBERS) + r")\s+(bullet|bullets|points|point|lines|line|sentences|"
                  r"sentence|items|things|parts)\b", text.lower())
    if not m:
        return None
    v = m.group(1)
    return int(v) if v.isdigit() else WORD_NUMBERS.get(v)
