"""Thin per-question entry point called by the challenge harness.

Accepted forms (the official Match 4 contract was not published when this was
written, so the common shapes from earlier matches are all supported):

  python3 /app/app.py --query-id query_01 --query "..."      -> /app/output/query_01_output.json
  python3 /app/app.py --question "..." [--output-dir DIR]     -> DIR/answer_output.json
  python3 /app/app.py --input /app/input/q_01.json            -> /app/output/q_01_output.json
  python3 /app/app.py --input-dir /app/input                  -> one file per input
  python3 /app/app.py "question text"                         -> positional question
  echo "question" | python3 /app/app.py                       -> question on stdin
  QUERY / QUERY_ID environment variables                      -> same as --query / --query-id

A JSON input that is a list of objects produces one output file per item
(`id` / `query_id` names the file).

Safety rules (these hold for every input form):
  * a valid JSON answer file is written first, so a crash or timeout still leaves
    a well-formed (empty) answer rather than a missing file;
  * a hard wall-clock deadline (WA_HARD_DEADLINE_S, default budget + 3 s) ends the
    process even if a network call or the model hangs;
  * a request to the resident server that times out is NOT retried in-process
    (that would double the time spent);
  * the exit code is always 0 once the placeholder exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from webagent import config  # noqa: E402

QUESTION_KEYS = ("question", "query", "prompt", "request", "task", "text", "instruction")
ID_KEYS = ("query_id", "id", "qid", "question_id")
READY_FILE = Path(os.environ.get("WA_READY_FILE", "/tmp/wa_ready"))
STARTING_FILE = Path(os.environ.get("WA_STARTING_FILE", "/tmp/wa_starting"))
EMPTY = {"answer": "", "text": "", "sources": [], "citations": [], "confidence": 0.0}


def _question_of(obj) -> str | None:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in QUESTION_KEYS:
            if isinstance(obj.get(k), str):
                return obj[k]
    return None


def read_items(path: Path) -> list[tuple[str | None, str]]:
    """Return [(id or None, question)] from a .json/.txt input file."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if path.suffix.lower() == ".json" or raw[:1] in "[{\"":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return [(None, raw)]
        if isinstance(obj, list):
            items = []
            for n, it in enumerate(obj, 1):
                q = _question_of(it)
                if q is not None:
                    qid = next((str(it[k]) for k in ID_KEYS if isinstance(it, dict) and k in it), f"{path.stem}_{n}")
                    items.append((qid, q))
            return items
        q = _question_of(obj)
        if q is not None:
            qid = next((str(obj[k]) for k in ID_KEYS if isinstance(obj, dict) and k in obj), None)
            return [(qid, q)]
    return [(None, raw)]


def read_question(path: Path) -> str:
    items = read_items(path)
    return items[0][1] if items else ""


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80] or "answer"


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: v for k, v in obj.items() if not k.startswith("_")}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def server_state() -> str:
    """'ready' (server started and warmed up), 'starting', or 'absent' (local run)."""
    if READY_FILE.exists():
        return "ready"
    if STARTING_FILE.exists():
        return "starting"
    return "absent"


def wait_for_server(url: str, deadline: float) -> bool:
    """Poll /health until it answers or the deadline passes."""
    while time.time() < deadline:
        try:
            if json.loads(urllib.request.urlopen(url + "/health", timeout=1).read()).get("ok"):
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def ask_server(question: str, budget: float, t_start: float) -> tuple[dict | None, bool]:
    """Returns (answer, server_was_expected). server_was_expected=True means the
    container's resident server exists, so a failure must not trigger a second,
    in-process model load."""
    url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"
    state = server_state()
    # While the container is still loading weights, wait for it (the harness gives the
    # container its own start-up time; a question that arrives early just waits).
    wait = config.SERVER_WAIT_S if state == "starting" else 3.0
    if not wait_for_server(url, time.time() + wait):
        return None, state != "absent"
    left = max(3.0, budget - (time.time() - t_start))
    req = urllib.request.Request(url + "/answer", data=json.dumps({"question": question, "budget": left}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=left + 2).read()), True
    except Exception:
        return None, True


def solve(question: str, out_path: Path, debug: bool) -> dict:
    t_start = time.time()
    write(out_path, EMPTY)
    budget = config.TIME_BUDGET_S
    hard = config.HARD_DEADLINE_S if config.HARD_DEADLINE_S > 0 else budget + 3

    box: dict = {"res": EMPTY}

    def work() -> None:
        try:
            res, expected = ask_server(question, budget, t_start)
            if res is None and not expected:  # no server (local runs): answer in-process
                from webagent import agent
                r = agent.answer(question, max(5.0, budget - (time.time() - t_start)))
                res = r.to_output() | {"_trace": r.trace, "_usage": r.usage.as_dict()}
            if res is not None:
                box["res"] = res
        except Exception as e:  # never lose the placeholder
            box["res"] = EMPTY | {"_error": f"{type(e).__name__}: {e}"}

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(hard)
    if th.is_alive():
        box["res"] = EMPTY | {"_error": f"hard deadline {hard:.0f}s reached"}
    res = box["res"]
    write(out_path, res)
    if debug:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def named_items(p: Path, forced_id: str | None) -> list[tuple[str | None, str]]:
    """One question in the file -> named after the file (or --query-id); several -> named by item id."""
    items = read_items(p)
    if len(items) == 1:
        return [(forced_id or p.stem, items[0][1])]
    return items


def run_items(items: list[tuple[str | None, str]], out: Path, debug: bool, default_id: str) -> None:
    for qid, q in items:
        solve(q, out / f"{safe_name(qid or default_id)}_output.json", debug)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question_pos", nargs="?", help="question text (positional form)")
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    ap.add_argument("--question")
    ap.add_argument("--input")
    ap.add_argument("--input-dir")
    ap.add_argument("--output-dir", default=config.OUTPUT_DIR)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--index", nargs="?", const="", help="accepted for harnesses that index a corpus first; nothing to index")
    ap.add_argument("--corpus")
    a, _unknown = ap.parse_known_args()  # tolerate extra flags from the harness
    if a.index is not None and a.query is None and a.question is None:
        return  # an "index the corpus" call (as in Match 3): a web agent has nothing to index
    out = Path(a.output_dir)
    q = a.query if a.query is not None else a.question if a.question is not None else a.question_pos
    if q is None and os.environ.get("QUERY"):
        q = os.environ["QUERY"]
    qid = a.query_id or os.environ.get("QUERY_ID") or "answer"
    if q is not None:
        run_items([(qid, q)], out, a.debug, qid)
    elif a.input:
        p = Path(a.input)
        run_items(named_items(p, a.query_id), out, a.debug, p.stem)
    elif a.input_dir:
        for p in sorted(Path(a.input_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in (".json", ".txt"):
                run_items(named_items(p, None), out, a.debug, p.stem)
    elif not sys.stdin.isatty():
        txt = sys.stdin.read().strip()
        if not txt:
            ap.error("give --query, --question, --input, --input-dir, a positional question or stdin")
        run_items([(qid, txt)], out, a.debug, qid)
    else:
        ap.error("give --query, --question, --input, --input-dir, a positional question or stdin")


if __name__ == "__main__":
    main()
