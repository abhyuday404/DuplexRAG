"""Validate a telemetry trace (JSONL) against schemas/telemetry.schema.json and report G6 coverage.

Dependency-free: checks the schema's required fields (base + per-event conditional blocks) and
enum values, then computes per-turn trace coverage. Exit code 1 on any violation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplexrag.telemetry import trace_coverage  # noqa: E402


def main(path: str) -> int:
    schema = json.loads((ROOT / "schemas" / "telemetry.schema.json").read_text())
    base_req = schema["required"]
    events_enum = set(schema["properties"]["event"]["enum"])
    cond = {}
    for block in schema["allOf"]:
        ev = block["if"]["properties"]["event"]["const"]
        cond[ev] = block["then"]
    errors = 0
    events = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for e in events:
        miss = [k for k in base_req if k not in e]
        if e.get("event") not in events_enum:
            miss.append(f"event={e.get('event')}")
        then = cond.get(e.get("event"))
        if then:
            miss += [k for k in then.get("required", []) if k not in e]
            for k, spec in then.get("properties", {}).items():
                if "enum" in spec and k in e and e[k] not in spec["enum"]:
                    miss.append(f"{k}={e[k]}")
        if miss:
            errors += 1
            print("INVALID", e.get("event"), e.get("turn_id"), miss)
    tids = [(e["session_id"], e["turn_id"]) for e in events if e["event"] == "turn_started"]
    retr = {(e["session_id"], e["turn_id"]) for e in events if e["event"] == "retrieval_started"}
    cov = trace_coverage(events, tids, retr)
    print(f"events={len(events)} turns={cov['turns']} schema_violations={errors} "
          f"G6_trace_coverage={100 * cov['coverage']:.1f}%")
    for k, v in cov["missing"].items():
        print("MISSING", k, v)
    return 1 if errors or cov["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
