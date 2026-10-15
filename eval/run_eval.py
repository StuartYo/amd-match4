"""Evaluate the agent and the no-web control on the same questions.

    python eval/run_eval.py --set dev --mode both
    python eval/run_eval.py --set holdout --mode both

Scoring mirrors the grader used in Matches 2 and 3: upper-case, drop
whitespace and "- . · _", then exact match against any accepted answer.
Phone answers are also scored on digits only ("lenient"), because the exact
phone format Match 4 expects has not been published.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from webagent import agent  # noqa: E402
from webagent.extract import grade_norm, phone_digits  # noqa: E402


def low_priority() -> None:
    if os.name == "nt":
        BELOW_NORMAL = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    else:
        os.nice(10)


def correct(pred: str, answers: list[str], qtype: str) -> tuple[bool, bool]:
    strict = any(grade_norm(pred) == grade_norm(a) for a in answers) if pred else False
    lenient = strict
    if not lenient and qtype == "phone" and pred:
        pd = phone_digits(pred)[-10:]
        lenient = any(len(phone_digits(a)) >= 10 and phone_digits(a)[-10:] == pd for a in answers)
    if not lenient and pred and qtype == "number":
        lenient = any(grade_norm(pred).replace(",", "") == grade_norm(a).replace(",", "") for a in answers)
    return strict, lenient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "holdout", "holdout2"])
    ap.add_argument("--mode", default="both", choices=["agent", "baseline", "both"])
    ap.add_argument("--only")
    a = ap.parse_args()
    low_priority()
    import torch
    torch.set_num_threads(4)

    qs = [json.loads(l) for l in (ROOT / "eval" / f"questions_{a.set}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.only:
        qs = [q for q in qs if q["id"] in a.only.split(",")]
    modes = ["agent", "baseline"] if a.mode == "both" else [a.mode]
    rows, summary = [], {}
    for mode in modes:
        s_ok = l_ok = 0
        secs, toks = [], 0
        for q in qs:
            t0 = time.time()
            try:
                r = agent.answer(q["question"]) if mode == "agent" else agent.baseline_answer(q["question"])
                pred, srcs, conf, trace, usage = r.answer, r.sources, r.confidence, r.trace, r.usage.as_dict()
            except Exception as e:
                pred, srcs, conf, trace, usage = "", [], 0.0, {"error": repr(e)}, {}
            dt = time.time() - t0
            s, l = correct(pred, q["answers"], q["type"])
            s_ok += s
            l_ok += l
            secs.append(dt)
            toks += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            rows.append({"mode": mode, "id": q["id"], "pred": pred, "answers": q["answers"], "strict": s, "lenient": l,
                         "seconds": round(dt, 1), "confidence": conf, "sources": srcs, "usage": usage, "trace": trace})
            if mode == "agent":
                time.sleep(2)  # be polite to search engines between questions
            print(f"[{mode}] {q['id']} {'OK ' if l else 'BAD'} {dt:5.1f}s pred={pred!r} want={q['answers'][0]!r}", flush=True)
        summary[mode] = {"n": len(qs), "strict": s_ok, "lenient": l_ok,
                         "mean_s": round(sum(secs) / max(1, len(secs)), 1), "max_s": round(max(secs or [0]), 1),
                         "tokens": toks}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "eval" / "results" / f"{a.set}_{stamp}.json"
    out.write_text(json.dumps({"set": a.set, "summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
