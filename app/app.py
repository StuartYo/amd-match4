"""Thin per-question entry point called by the challenge harness.

Accepted forms (the official Match 4 contract was not published when this was
written, so the common shapes from earlier matches are all supported):

  python3 /app/app.py --query-id query_01 --query "..."      -> /app/output/query_01_output.json
  python3 /app/app.py --question "..." [--output-dir DIR]     -> DIR/answer_output.json
  python3 /app/app.py --input /app/input/q_01.json            -> /app/output/q_01_output.json
  python3 /app/app.py --input-dir /app/input                  -> one file per input

A valid JSON answer file is written first, so a crash or timeout still leaves
a well-formed (empty) answer rather than a missing file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from webagent import config  # noqa: E402

QUESTION_KEYS = ("question", "query", "prompt", "request", "task", "text", "instruction")
EMPTY = {"answer": "", "text": "", "sources": [], "citations": [], "confidence": 0.0}


def read_question(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for k in QUESTION_KEYS:
                    if isinstance(obj.get(k), str):
                        return obj[k]
            if isinstance(obj, str):
                return obj
        except json.JSONDecodeError:
            pass
    return raw


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: v for k, v in obj.items() if not k.startswith("_")}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def ask_server(question: str, budget: float) -> dict | None:
    url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.3)
    else:
        return None
    req = urllib.request.Request(url + "/answer", data=json.dumps({"question": question, "budget": budget}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=budget + 5).read())
    except Exception:
        return None


def solve(question: str, out_path: Path, debug: bool) -> dict:
    write(out_path, EMPTY)
    budget = config.TIME_BUDGET_S
    res = ask_server(question, budget)
    if res is None:  # no server (local runs): answer in-process
        from webagent import agent
        r = agent.answer(question, budget)
        res = r.to_output() | {"_trace": r.trace, "_usage": r.usage.as_dict()}
    write(out_path, res)
    if debug:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    ap.add_argument("--question")
    ap.add_argument("--input")
    ap.add_argument("--input-dir")
    ap.add_argument("--output-dir", default=config.OUTPUT_DIR)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir)
    if a.query is not None or a.question is not None:
        qid = a.query_id or "answer"
        solve(a.query if a.query is not None else a.question, out / f"{qid}_output.json", a.debug)
    elif a.input:
        p = Path(a.input)
        solve(read_question(p), out / f"{p.stem}_output.json", a.debug)
    elif a.input_dir:
        for p in sorted(Path(a.input_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in (".json", ".txt"):
                solve(read_question(p), out / f"{p.stem}_output.json", a.debug)
    else:
        ap.error("give --query, --question, --input or --input-dir")


if __name__ == "__main__":
    main()
