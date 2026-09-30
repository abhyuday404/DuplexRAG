"""Optional abstractive synthesis through any OpenAI-compatible chat endpoint (Ollama, llama.cpp
server, vLLM, Groq, OpenAI ...), enabled with DUPLEXRAG_SYNTHESIS=llm.

The LLM only *rephrases* the evidence the extractive composer already selected; it never sees the
rest of the corpus. Its output is post-verified sentence by sentence with the same grounding
checker used for G4: a sentence survives only if every citation it carries exists, was retrieved
for this answer, and supports the sentence (numbers must match, content words must overlap).
On any error, timeout or empty result the extractive answer is returned unchanged.
"""
from __future__ import annotations

import json
import os
import re

import httpx

from .grounding import extract_citations, verify

SYSTEM = (
    "You are a voice assistant answering from company documents. Rewrite the EVIDENCE into a short, "
    "natural spoken answer. Rules: use only facts stated in the evidence; never add numbers, names or "
    "conditions that are not in it; end every sentence with the citation(s) of the evidence it uses, "
    "copied exactly, e.g. [Doc_03 §2]; if an item is marked NOT FOUND, say plainly that it could not be "
    "verified. No preamble."
)


class LLMSynthesizer:
    def __init__(self, settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        key = os.environ.get(settings.llm_api_key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(base_url=settings.llm_base_url.rstrip("/"), headers=headers, timeout=20.0,
                                   transport=transport)

    def _prompt(self, answer) -> str:
        lines = []
        for it in answer.intents.values():
            if it.status != "active":
                continue
            lines.append(f"## {it.label}")
            claims = [c for c in answer.active_claims() if c.intent_id == it.iid]
            if it.uncertain or not claims:
                lines.append(f"NOT FOUND: {it.uncertain or 'no evidence'}")
            for c in claims:
                lines.append(f"{c.text} [{', '.join(c.citations)}]")
        for u in answer.uncertainty:
            lines.append(f"NOT FOUND: {u['message']}")
        return "EVIDENCE:\n" + "\n".join(lines)

    def stream(self, answer):
        """Yield text deltas from the endpoint (server-sent events)."""
        body = {"model": self.s.llm_model, "stream": True, "temperature": 0.1, "max_tokens": 400,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": self._prompt(answer)}]}
        with self.client.stream("POST", "/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def rewrite(self, answer, index) -> tuple[str, dict]:
        """Return (verified text, usage). Falls back to the extractive text on failure."""
        prompt = self._prompt(answer)
        usage = {"llm_tokens_in": len(prompt) // 4, "llm_tokens_out": 0, "dropped_sentences": 0, "fallback": False}
        try:
            text = "".join(self.stream(answer)).strip()
        except Exception as exc:  # network / endpoint errors -> grounded extractive answer
            usage.update(fallback=True, error=str(exc)[:200])
            return answer.text, usage
        usage["llm_tokens_out"] = len(text) // 4
        retrieved = {h.label for i in answer.intents.values() for h in i.evidence}
        kept = []
        for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", text):
            if not sent.strip():
                continue
            cites = extract_citations(sent)
            if not cites:
                if re.search(r"could not be verified|not (be )?found|no information", sent, re.I):
                    kept.append(sent)          # explicit uncertainty needs no citation
                else:
                    usage["dropped_sentences"] += 1
                continue
            res = verify([{"text": sent, "citations": cites}], index, retrieved)
            if res["supported"] == 1 and not res["fabricated_ids"] and not res["unretrieved_ids"]:
                kept.append(sent)
            else:
                usage["dropped_sentences"] += 1
        if not kept:
            usage["fallback"] = True
            return answer.text, usage
        return " ".join(kept), usage
