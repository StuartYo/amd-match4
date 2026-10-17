"""Long-running process that keeps the model loaded.

The harness starts a new `python3 /app/app.py ...` process for every question,
so loading the model there would cost the whole time budget each time. This
server is started once by the container CMD; app.py forwards questions to it.

The HTTP port opens immediately; /health says ok=false while weights load, so a
question that arrives early can wait instead of loading a second copy of the model.
If the model cannot be loaded at all, the server still answers (deterministic
extractors only, no model reading) rather than leaving every question empty.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import agent, config
from .llm import get_llm, llm_error

READY_FILE = Path(os.environ.get("WA_READY_FILE", "/tmp/wa_ready"))
READY = threading.Event()
EMPTY = {"answer": "", "text": "", "sources": [], "citations": [], "confidence": 0.0}
POOL = ThreadPoolExecutor(max_workers=4)


def run_agent(q: str, budget: float) -> dict:
    res = agent.answer(q, budget)
    out = res.to_output()
    out["_trace"] = res.trace
    out["_usage"] = res.usage.as_dict()
    return out


def handle_question(q: str, budget: float) -> dict:
    """Answer within budget seconds no matter what the agent does."""
    t0 = time.time()
    if not READY.wait(timeout=max(0.0, budget - 2)):
        return EMPTY | {"_error": "model still loading"}
    left = max(2.0, budget - (time.time() - t0))
    fut = POOL.submit(run_agent, q, left)
    try:
        return fut.result(timeout=left + 1.0)
    except FutTimeout:
        return EMPTY | {"_error": "agent exceeded budget"}
    except Exception as e:
        return EMPTY | {"_error": f"{type(e).__name__}: {e}"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep stdout clean
        sys.stderr.write("[server] " + (fmt % args) + "\n")

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": READY.is_set(), "loading": not READY.is_set(),
                             "model": config.LLM_MODEL, "model_error": llm_error()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/answer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            q = str(req.get("question", ""))
            budget = float(req.get("budget", config.TIME_BUDGET_S))
        except Exception as e:
            self._send(200, EMPTY | {"_error": f"bad request: {type(e).__name__}"})
            return
        self._send(200, handle_question(q, budget))


def load_model() -> None:
    t0 = time.time()
    try:
        get_llm()  # load weights before accepting questions
        # Warm up once so the first graded question does not pay CUDA/ROCm init.
        get_llm().chat([{"role": "user", "content": "Say OK."}], max_tokens=4)
    except Exception as e:
        sys.stderr.write(f"[server] model unavailable, running without it: {type(e).__name__}: {e}\n")
    READY.set()
    try:
        READY_FILE.write_text("ready")
    except OSError:
        pass
    sys.stderr.write(f"[server] ready in {time.time() - t0:.1f}s on {config.SERVER_HOST}:{config.SERVER_PORT}\n")


def main() -> None:
    srv = ThreadingHTTPServer((config.SERVER_HOST, config.SERVER_PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=load_model, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
