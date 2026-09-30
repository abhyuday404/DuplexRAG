"""DuplexRAG command line.

    duplexrag index                       build / refresh the hybrid index for data/corpus
    duplexrag serve [--port 8000]         real-time web demo (WebSocket streaming)
    duplexrag replay FILE.jsonl           replay streaming sessions, print turns, write a JSONL trace
    duplexrag ask "an utterance"          stream one utterance through the engine and print the answer
    duplexrag bench [--split test] [--ablations]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, load_settings


def cmd_index(a) -> None:
    from .engine import DuplexEngine
    e = DuplexEngine(load_settings(corpus_dir=a.corpus) if a.corpus else load_settings())
    docs = {c.doc_id for c in e.index.chunks}
    print(f"indexed {len(docs)} documents / {len(e.index.chunks)} chunks "
          f"({sum(len(c.sentences) for c in e.index.chunks)} sentence units)")


def _print_turn(r: dict) -> None:
    lat = r["latency"]
    print(f"\n[{r['session_id']} {r['turn_id']}] kind={r['kind']} v{r['version']} early={r['early']} "
          f"ttft={lat.get('ttft_ms')}ms lead={lat.get('retrieval_lead_ms')}ms cost=${r['cost']['usd']:.6f}")
    for q in r["sub_queries"]:
        print(f"   sub-query [{q['trigger']}] {q['text']}")
    print("   " + (r["text"] or "(no answer text - retrieval suppressed)").replace("\n", "\n   "))


def cmd_replay(a) -> None:
    from .engine import DuplexEngine
    from .stream import replay_session
    from .telemetry import write_jsonl
    e = DuplexEngine(load_settings(), log=lambda *x: None)
    sessions = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    events, records = [], []
    for s in sessions:
        recs, evs = replay_session(e, s)
        records += recs
        events += evs
        for r in recs:
            _print_turn(r)
    out = Path(a.out)
    write_jsonl(out / "trace.jsonl", events)
    write_jsonl(out / "turns.jsonl", records)
    print(f"\nwrote {len(events)} telemetry events -> {out / 'trace.jsonl'}")


def cmd_ask(a) -> None:
    from .engine import DuplexEngine
    from .stream import replay_session
    e = DuplexEngine(load_settings(), log=lambda *x: None)
    turns = [{"turn_id": f"t{i + 1}", "utterance": u} for i, u in enumerate(a.utterances)]
    recs, _ = replay_session(e, {"session_id": "cli", "turns": turns})
    for r in recs:
        _print_turn(r)


def cmd_serve(a) -> None:
    from .server import main
    main(a.host, a.port)


def cmd_bench(a) -> None:
    from .bench.run import main
    argv = ["--split", a.split, "--out", a.out]
    if a.ablations:
        argv.append("--ablations")
    if a.configs:
        argv += ["--configs", a.configs]
    main(argv)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="duplexrag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index")
    s.add_argument("--corpus", default=None)
    s.set_defaults(fn=cmd_index)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("replay")
    s.add_argument("file")
    s.add_argument("--out", default=str(ROOT / "runs" / "replay"))
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("ask")
    s.add_argument("utterances", nargs="+", help="one or more utterances = consecutive turns of one session")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("bench")
    s.add_argument("--split", default="test")
    s.add_argument("--configs", default=None)
    s.add_argument("--ablations", action="store_true")
    s.add_argument("--out", default=str(ROOT / "results"))
    s.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
