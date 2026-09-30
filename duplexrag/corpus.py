"""Corpus loading and section-aware chunking.

Supported inputs (any mix, recursively):
  * Markdown / text with optional YAML-ish front matter and ``## §N Title`` headings
    (falls back to ``#``/``##`` headings, then to ~180-word windows)
  * PDF (one or more sections per page)
  * JSONL with ``{"doc_id", "title", "text"[, "section"]}`` records

Every chunk carries a stable citation label ``Doc_ID §N`` so that answers can be
traced back to verifiable corpus locations.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .text import normalize_space, split_sentences

SECTION_RE = re.compile(r"^##\s*§\s*(\d+)\s*[.:-]?\s*(.*)$", re.M)
HEADING_RE = re.compile(r"^#{2,3}\s+(.*)$", re.M)


@dataclass
class Chunk:
    chunk_id: str            # "Doc_03§2" - internal key
    doc_id: str              # "Doc_03"
    section: str             # "2"
    section_title: str
    doc_title: str
    text: str                # section body
    sentences: list[str] = field(default_factory=list)
    status: str = "current"  # current | superseded
    superseded_by: str | None = None
    category: str | None = None
    effective_date: str | None = None
    source: str = ""

    @property
    def label(self) -> str:
        return f"{self.doc_id} §{self.section}"

    @property
    def short_title(self) -> str:
        return re.split(r"\s+-\s+|:\s+", self.doc_title)[0].strip()

    def index_text(self) -> str:
        """Contextualised text used for indexing (title + section title + body)."""
        return f"{self.doc_title}. {self.section_title}. {self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d


def parse_front_matter(raw: str) -> tuple[dict, str]:
    meta: dict = {}
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            raw = raw[end + 4:]
    return meta, raw


def _windows(body: str, size: int = 180) -> list[tuple[str, str]]:
    words = body.split()
    out = []
    for i in range(0, len(words), size):
        out.append((f"Part {i // size + 1}", " ".join(words[i:i + size])))
    return out


def _split_sections(body: str) -> list[tuple[str, str, str]]:
    """Return [(section_no, section_title, text)]."""
    marks = list(SECTION_RE.finditer(body))
    if marks:
        out = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
            out.append((m.group(1), m.group(2).strip(), body[m.end():end].strip()))
        return out
    heads = list(HEADING_RE.finditer(body))
    if heads:
        out = []
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
            out.append((str(i + 1), m.group(1).strip(), body[m.end():end].strip()))
        return out
    return [(str(i + 1), t, x) for i, (t, x) in enumerate(_windows(body))]


def _doc_id_from_path(path: Path) -> str:
    m = re.match(r"(Doc_\d+)", path.stem)
    return m.group(1) if m else re.sub(r"[^A-Za-z0-9_]+", "_", path.stem)[:40]


def load_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    doc_id = meta.get("doc_id") or _doc_id_from_path(path)
    title = meta.get("title")
    if not title:
        m = re.search(r"^#\s+(.*)$", body, re.M)
        title = m.group(1).strip() if m else path.stem
    body = re.sub(r"^#\s+.*$", "", body, count=1, flags=re.M)
    chunks = []
    for sec, sec_title, text in _split_sections(body):
        text = text.strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc_id}§{sec}", doc_id=doc_id, section=sec, section_title=sec_title,
            doc_title=title, text=text, sentences=split_sentences(text),
            status=meta.get("status", "current"), superseded_by=meta.get("superseded_by"),
            category=meta.get("category"), effective_date=meta.get("effective_date"), source=path.name,
        ))
    return chunks


def load_pdf(path: Path) -> list[Chunk]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    doc_id = _doc_id_from_path(path)
    chunks = []
    n = 0
    for pno, page in enumerate(reader.pages, 1):
        text = normalize_space(page.extract_text() or "")
        for title, part in _windows(text):
            n += 1
            chunks.append(Chunk(
                chunk_id=f"{doc_id}§{n}", doc_id=doc_id, section=str(n), section_title=f"Page {pno} {title}",
                doc_title=path.stem, text=part, sentences=split_sentences(part), source=path.name,
            ))
    return chunks


def load_jsonl(path: Path) -> list[Chunk]:
    chunks = []
    counters: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        doc_id = str(rec.get("doc_id") or rec.get("id"))
        counters[doc_id] = counters.get(doc_id, 0) + 1
        sec = str(rec.get("section") or counters[doc_id])
        text = rec.get("text", "")
        chunks.append(Chunk(
            chunk_id=f"{doc_id}§{sec}", doc_id=doc_id, section=sec,
            section_title=rec.get("section_title", ""), doc_title=rec.get("title", doc_id), text=text,
            sentences=split_sentences(text), status=rec.get("status", "current"),
            superseded_by=rec.get("superseded_by"), source=path.name,
        ))
    return chunks


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.name.lower() == "readme.md":
            continue
        suffix = path.suffix.lower()
        if suffix in (".md", ".markdown", ".txt"):
            chunks.extend(load_markdown(path))
        elif suffix == ".pdf":
            chunks.extend(load_pdf(path))
        elif suffix == ".jsonl":
            chunks.extend(load_jsonl(path))
    seen = set()
    for c in chunks:
        if c.chunk_id in seen:
            raise ValueError(f"duplicate chunk id {c.chunk_id} ({c.source})")
        seen.add(c.chunk_id)
    return chunks


def corpus_fingerprint(corpus_dir: str | Path) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(corpus_dir).rglob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]
