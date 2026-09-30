"""Real-time demo server (FastAPI + WebSocket).

The browser streams transcript chunks (scenario replay, live microphone via the Web
Speech API, or typed text) and receives every telemetry event as it happens, followed
by the streamed answer. One engine instance serves all connections; each WebSocket is
its own ephemeral session (nothing persists after the socket closes).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, load_settings
from .engine import DuplexEngine
from .stream import turn_chunks

WEB = ROOT / "web"
SCENARIOS = ROOT / "data" / "demo" / "scenarios.jsonl"

app = FastAPI(title="DuplexRAG", version="1.0.0")
_engine: DuplexEngine | None = None


def engine() -> DuplexEngine:
    global _engine
    if _engine is None:
        _engine = DuplexEngine(load_settings())
    return _engine


@app.on_event("startup")
async def _warm() -> None:
    await asyncio.get_running_loop().run_in_executor(None, engine)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/api/health")
def health():
    e = engine()
    return {"status": "ok", "chunks": len(e.index.chunks), "config": e.config_summary()}


@app.get("/api/scenarios")
def scenarios():
    if not SCENARIOS.exists():
        return []
    return [json.loads(l) for l in SCENARIOS.read_text(encoding="utf-8").splitlines() if l.strip()]


@app.get("/api/chunk")
def chunk(label: str):
    """Citation traceability: resolve 'Doc_03 §2' to its source section."""
    e = engine()
    i = e.index.by_label.get(label)
    if i is None:
        raise HTTPException(404, f"unknown citation {label}")
    c = e.index.chunks[i]
    return {"label": c.label, "doc_title": c.doc_title, "section_title": c.section_title, "text": c.text,
            "status": c.status, "superseded_by": c.superseded_by, "source": c.source}


@app.get("/api/corpus")
def corpus():
    e = engine()
    docs: dict[str, dict] = {}
    for c in e.index.chunks:
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.doc_title, "status": c.status, "sections": 0})
        d["sections"] += 1
    return list(docs.values())


class Conn:
    """One WebSocket = one ephemeral session."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sh = None
        self.turn_no = 0
        self.lock = asyncio.Lock()

    def listener(self, ev: dict) -> None:           # called from engine threads
        self.loop.call_soon_threadsafe(self.queue.put_nowait, {"type": "event", "event": ev})

    async def pump(self) -> None:
        while True:
            msg = await self.queue.get()
            await self.ws.send_text(json.dumps(msg, default=str))

    def new_session(self) -> dict:
        self.sh = engine().new_session(realtime=True, listener=self.listener)
        self.turn_no = 0
        return {"type": "session", "session_id": self.sh.session.session_id, "config": engine().config_summary()}

    async def run_blocking(self, fn, *a, **kw):
        return await self.loop.run_in_executor(None, lambda: fn(*a, **kw))

    async def start_turn(self) -> str:
        if self.sh is None:
            await self.queue.put(self.new_session())
        self.turn_no += 1
        tid = f"t{self.turn_no}"
        await self.run_blocking(engine().start_turn, self.sh, tid, self.sh.executor.now())
        return tid

    async def chunk(self, text: str, cumulative: bool) -> None:
        await self.run_blocking(engine().on_chunk, self.sh, text, self.sh.executor.now(), cumulative=cumulative)

    async def end_turn(self, word_delay: float = 0.035) -> None:
        res = await self.run_blocking(engine().end_turn, self.sh, self.sh.executor.now())
        # stream the answer the way a TTS front-end would consume it
        words = res.text.split(" ") if res.text else []
        for k, w in enumerate(words):
            await self.queue.put({"type": "token", "turn_id": res.turn_id, "text": w + (" " if k < len(words) - 1
                                                                                       else "")})
            await asyncio.sleep(word_delay)
        await self.queue.put({"type": "answer", "turn": res.to_dict()})

    async def replay(self, scenario: dict, speed: float) -> None:
        for turn in scenario["turns"]:
            await self.start_turn()
            chunks, end = turn_chunks(turn, scenario.get("session_id", "demo"))
            t0 = time.monotonic()
            for c in chunks:
                delay = c.t / speed - (time.monotonic() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                await self.queue.put({"type": "user_chunk", "text": c.text, "t": c.t})
                await self.chunk(c.text, cumulative=False)
            delay = end / speed - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            await self.end_turn()
            await asyncio.sleep(float(turn.get("pause_after", 2.5)) / max(speed, 0.25))
        await self.queue.put({"type": "replay_done", "scenario": scenario.get("session_id")})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn = Conn(ws)
    pump = asyncio.create_task(conn.pump())
    replay_task: asyncio.Task | None = None
    try:
        await conn.queue.put(conn.new_session())
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "new_session":
                if replay_task:
                    replay_task.cancel()
                await conn.queue.put(conn.new_session())
            elif kind == "start_turn":
                tid = await conn.start_turn()
                await conn.queue.put({"type": "turn", "turn_id": tid})
            elif kind == "chunk":
                await conn.chunk(msg.get("text", ""), bool(msg.get("cumulative")))
            elif kind == "end_turn":
                await conn.end_turn()
            elif kind == "replay":
                sc = next((s for s in scenarios() if s.get("session_id") == msg.get("scenario")), None)
                if sc is None:
                    await conn.queue.put({"type": "error", "message": "unknown scenario"})
                    continue
                if replay_task:
                    replay_task.cancel()
                await conn.queue.put(conn.new_session())
                replay_task = asyncio.create_task(conn.replay(sc, float(msg.get("speed", 1.0))))
            elif kind == "trace":
                await conn.queue.put({"type": "trace", "events": conn.sh.tracer.events if conn.sh else []})
    except WebSocketDisconnect:
        pass
    finally:
        if replay_task:
            replay_task.cancel()
        pump.cancel()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
