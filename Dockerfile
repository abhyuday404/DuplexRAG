# DuplexRAG - CPU-only streaming live RAG. Models and index are baked in, so the container runs offline.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DUPLEXRAG_THREADS=4

WORKDIR /app

# 1) pinned dependencies from the lockfile (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir "uv==0.12.19" \
 && uv sync --frozen --no-dev --no-install-project

# 2) application code, corpus, benchmark sets and demo scenarios
COPY duplexrag ./duplexrag
COPY web ./web
COPY data ./data
COPY scripts ./scripts
COPY schemas ./schemas
RUN uv sync --frozen --no-dev

# 3) download the two ONNX models (bge-small 67 MB, MiniLM-L6 cross-encoder 91 MB) and build the index
RUN uv run --no-sync python -m duplexrag index

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "python", "-m", "duplexrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
