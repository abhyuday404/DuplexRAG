#!/usr/bin/env bash
# G1 reproducibility smoke test: build the index, replay the demo sessions through the streaming
# engine, validate the telemetry trace against the schema, run the unit/e2e tests.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

echo "== corpus validation";      $PY scripts/validate_corpus.py
echo "== index";                  $PY -m duplexrag index
echo "== streaming replay";       $PY -m duplexrag replay data/demo/scenarios.jsonl --out runs/smoke > runs_smoke.log
tail -n 3 runs_smoke.log
echo "== telemetry schema / G6";  $PY scripts/check_trace.py runs/smoke/trace.jsonl
if $PY -c "import pytest" 2>/dev/null; then
  echo "== tests";                $PY -m pytest -q
fi
rm -f runs_smoke.log
echo "SMOKE TEST PASSED"
