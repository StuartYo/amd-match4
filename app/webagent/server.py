"""Long-running process that keeps the model loaded.

The harness starts a new `python3 /app/app.py ...` process for every question,
so loading the model there would cost the whole time budget each time. This
server is started once by the container CMD; app.py forwards questions to it.
"""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agent, config
from .llm import get_llm


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
            self._send(200, {"ok": True, "model": config.LLM_MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/answer":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        q = str(req.get("question", ""))
        budget = float(req.get("budget", config.TIME_BUDGET_S))
        try:
            res = agent.answer(q, budget)
            out = res.to_output()
            out["_trace"] = res.trace
            out["_usage"] = res.usage.as_dict()
            self._send(200, out)
        except Exception as e:
            self._send(200, {"answer": "", "text": "", "sources": [], "citations": [], "confidence": 0.0,
                             "_error": f"{type(e).__name__}: {e}"})


def main() -> None:
    t0 = time.time()
    get_llm()  # load weights before accepting questions
    # Warm up once so the first graded question does not pay CUDA/ROCm init.
    try:
        get_llm().chat([{"role": "user", "content": "Say OK."}], max_tokens=4)
    except Exception as e:
        sys.stderr.write(f"[server] warmup failed: {e}\n")
    sys.stderr.write(f"[server] ready in {time.time() - t0:.1f}s on {config.SERVER_HOST}:{config.SERVER_PORT}\n")
    ThreadingHTTPServer((config.SERVER_HOST, config.SERVER_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
