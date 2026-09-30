"""The optional LLM path must keep only grounded sentences (mocked OpenAI-compatible endpoint)."""
import json

import httpx
import pytest

from duplexrag.config import load_settings
from duplexrag.engine import DuplexEngine
from duplexrag.llm import LLMSynthesizer
from duplexrag.stream import replay_session


def _sse(text: str) -> bytes:
    chunks = [text[i:i + 12] for i in range(0, len(text), 12)]
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks] + ["data: [DONE]"]
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture(scope="module")
def answer_and_engine():
    eng = DuplexEngine(log=lambda *a: None)
    recs, _ = replay_session(eng, {"session_id": "llm", "turns": [
        {"turn_id": "t1", "utterance": "What is the hotel cap per night in Pune?"}]})
    # rebuild the session answer by replaying once more through a handle
    sh = eng.new_session("llm2")
    eng.start_turn(sh, "t1", 0.0)
    eng.on_chunk(sh, "What is the hotel cap per night in Pune?", 1.0)
    eng.end_turn(sh, 1.4)
    return sh.session.answer, eng


def test_llm_output_is_verified(answer_and_engine):
    answer, eng = answer_and_engine
    good = "The Pune hotel cap is INR 6,500 per night. [Doc_18 §2]"
    fabricated = "Pune hotels include free breakfast. [Doc_99 §1]"
    wrong_number = "The Pune cap is INR 9,999. [Doc_18 §2]"
    body = _sse(f"{good} {fabricated} {wrong_number}")
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body,
                                                               headers={"content-type": "text/event-stream"}))
    llm = LLMSynthesizer(load_settings(synthesis="llm"), transport=transport)
    text, usage = llm.rewrite(answer, eng.index)
    assert "6,500" in text
    assert "Doc_99" not in text and "9,999" not in text
    assert usage["dropped_sentences"] == 2 and not usage["fallback"]


def test_llm_failure_falls_back_to_extractive(answer_and_engine):
    answer, eng = answer_and_engine
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    llm = LLMSynthesizer(load_settings(synthesis="llm"), transport=transport)
    text, usage = llm.rewrite(answer, eng.index)
    assert text == answer.text and usage["fallback"]
