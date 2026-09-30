"""Lightweight, dependency-free text utilities shared by every stage.

Everything here is deliberately rule-based and corpus-agnostic: no query,
prompt or answer from any benchmark is embedded in this module.
"""
from __future__ import annotations

import re
from functools import lru_cache

STOPWORDS = frozenset(
    """
a about above after again against all also am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's
with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
please tell know need needs want wants like would could get give let us okay ok um uh hmm well just really
actually also basically kind sort thing things something anything lot much many one also yeah yes hey hi
""".split()
)

# Words that make a partial utterance "dangle" when they are the last token:
# the speaker has clearly not finished the thought yet.
DANGLING = frozenset(
    """
a an the and or but of in on at to for with about from by into onto over under between my our your their his
her its this that these those some any need want like know tell get find check what which who whose how when
where is are was were be been do does did can could would should will shall i we you they he she it also plus
as than if whether because so then while about regarding around per via including especially without within
before after until till across through during against among toward towards upon near take
""".split()
)

# a comma after one of these is a pause inside a phrase ("the rule for, a normal trip"), not a clause boundary
COMMA_JOIN = DANGLING - frozenset("that this these those it need want know tell get find check".split()) | frozenset(
    """
am i
""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=65536)
def stem(tok: str) -> str:
    """Tiny suffix stripper (a pragmatic subset of Porter step 1)."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    base = tok
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "es" and not re.search(r"(s|x|z|ch|sh)es$", tok):
                continue                                  # "images" -> "image", but "taxes" -> "tax"
            if suf == "s" and tok.endswith(("ss", "us", "is")):
                continue
            base = tok[: -len(suf)] + rep
            if suf in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            break
    if len(base) > 4 and base.endswith("e"):
        base = base[:-1]                                  # "charge"/"charged" -> "charg"
    return base


_CONTRACTION_RE = re.compile(r"(?<=[a-z])['\u2019](s|re|ll|d|ve|m)\b")


def raw_tokens(text: str) -> list[str]:
    text = text.lower().replace("§", " section ").replace("n't", " not").replace("n\u2019t", " not")
    text = _CONTRACTION_RE.sub("", text)
    toks = []
    for t in _TOKEN_RE.findall(text):
        # "2,00,000" / "1,000" -> "200000" / "1000" so numbers match however they are written
        if re.fullmatch(r"[0-9][0-9,]*[0-9]", t):
            t = t.replace(",", "")
        toks.append(t)
    return toks


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed tokens (numbers kept)."""
    return [stem(t) for t in raw_tokens(text) if t not in STOPWORDS]


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        n = m.group(0).rstrip(",").replace(",", "")
        if n:
            out.add(n)
    return out


def split_sentences(text: str) -> list[str]:
    """Split a section body into answer-sized units: bullets and sentences."""
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        # drop markdown table header rows (the row right above a |---| separator)
        lines = [ln for k, ln in enumerate(lines)
                 if not (ln.startswith("|") and k + 1 < len(lines) and re.fullmatch(r"\|?[\s:|-]+\|?", lines[k + 1]))]
        para: list[str] = []
        for ln in lines:
            if re.match(r"^([-*+]|\d+[.)])\s+", ln):
                if para:
                    units.extend(_SENT_SPLIT_RE.split(" ".join(para)))
                    para = []
                units.append(re.sub(r"^([-*+]|\d+[.)])\s+", "", ln))
            elif ln.startswith("|"):
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if cells and not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    units.append(" - ".join(c for c in cells if c))
            else:
                para.append(ln)
        if para:
            units.extend(_SENT_SPLIT_RE.split(" ".join(para)))
    return [normalize_space(u) for u in units if len(u.split()) >= 3]


# Small, domain-agnostic spoken-to-written synonym table (general workplace English, not corpus facts).
# Used only to expand BM25 keywords and lexical checks; dense retrieval handles the rest.
_SYN = {
    "food": "cater meal", "meal": "cater", "lunch": "meal cater", "snack": "cater", "drink": "cater",
    "uber": "cab taxi", "ola": "cab taxi", "taxi": "cab", "lyft": "cab taxi",
    "wifi": "internet", "internet": "wifi",
    "resign": "notice resignation", "quit": "notice resignation",
    "vacation": "leave", "holiday": "leave", "pto": "leave", "sick": "leave",
    "boss": "manager", "salary": "pay", "refund": "reimburs", "money": "reimburs",
    "computer": "laptop", "mobile": "phone", "cellphone": "phone", "kid": "child", "baby": "child",
    "stay": "hotel accommodation", "room": "hotel accommodation",
}
SYNONYMS = {stem(k): [stem(x) for x in v.split()] for k, v in _SYN.items()}


def expand_terms(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        for s in SYNONYMS.get(t, []):
            if s not in out:
                out.append(s)
    return out


WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "single": 1, "couple": 2, "pair": 2,
}
