"""Runtime settings, all overridable through environment variables."""
from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Total wall-clock budget for one question, in seconds. Keep it below the
# harness per-item limit (MC3 used 30 s) so the answer file is always written.
TIME_BUDGET_S = _float("WA_TIME_BUDGET_S", 24.0)

# LLM backend: "transformers" (in-process) or "openai" (vLLM / any
# OpenAI-compatible server started by the container entrypoint).
LLM_BACKEND = _env("WA_LLM_BACKEND", "transformers")
LLM_MODEL = _env("WA_LLM_MODEL", "Qwen/Qwen3-1.7B")
LLM_BASE_URL = _env("WA_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MAX_CONTEXT_CHARS = _int("WA_LLM_CONTEXT_CHARS", 9000)

# Web access.
USER_AGENT = _env(
    "WA_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0 Safari/537.36",
)
API_USER_AGENT = _env("WA_API_USER_AGENT", "amd-match4-webagent/0.1 (https://github.com/StuartYo; contact@stuartyo.com) python-httpx")
FETCH_TIMEOUT_S = _float("WA_FETCH_TIMEOUT_S", 6.0)
MAX_PAGES = _int("WA_MAX_PAGES", 6)
MAX_FOLLOW = _int("WA_MAX_FOLLOW", 3)
IO_WORKERS = _int("WA_IO_WORKERS", 4)

# Hard wall-clock cap for one app.py process (0 = budget + 6 s).
HARD_DEADLINE_S = _float("WA_HARD_DEADLINE_S", 0.0)
# How long app.py waits for the resident server while the container is still loading weights.
SERVER_WAIT_S = _float("WA_SERVER_WAIT_S", 540.0)

# Local server that keeps the model loaded between per-question processes.
SERVER_HOST = _env("WA_SERVER_HOST", "127.0.0.1")
SERVER_PORT = _int("WA_SERVER_PORT", 8765)

# Output directory used by the harness.
OUTPUT_DIR = _env("WA_OUTPUT_DIR", "/app/output")

# Below this confidence the answer is replaced by an empty string. 0 means
# "always answer when there is a grounded candidate".
ABSTAIN_BELOW = _float("WA_ABSTAIN_BELOW", 0.0)

# Optional on-disk cache for search results and pages. Development only: it
# keeps repeated evaluation runs from hammering search engines. Never set it
# in the challenge image.
CACHE_DIR = _env("WA_CACHE_DIR", "")
