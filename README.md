# Grounded Web Retriever (AMD AI League, Match 4)

Ask a question in plain English; get back one exact value and the web page it came from.
The answer is only returned if it actually appears on a page the agent fetched, so the model
cannot answer from stale memory.

```
$ python3 /app/app.py --query-id query_01 --query "What is the latest LTS version of Node.js?"
/app/output/query_01_output.json
{"answer": "24.22.0", "sources": ["https://endoflife.date/nodejs"], "confidence": 0.93, ...}
```

## Why an agent and not just a model

The same model with no web access (the control condition in `eval/`) answers from what it
learned during training. Software versions, phone numbers and "who is the current CEO"
questions go stale, and small models also invent plausible phone numbers. Results on the
same questions are in [Evaluation](#evaluation).

## How it works

```
question
  -> LLM plan: answer type, entity, 2 search queries          (1 model call)
  -> in parallel: keyless search (Brave, DuckDuckGo, Seznam; Bing as fallback)
                  Wikidata lookup of the entity's official website
  -> "latest version" questions: endoflife.date, PyPI, npm, GitHub releases
  -> fetch top pages (official domain first), follow contact/support/release links one hop
  -> candidates: typed extractors (phone, email, version, date, price, number),
                 tel:/mailto: links, JSON-LD, and an LLM read of the best page excerpts (1 model call)
  -> program-side checks: the value must appear on a fetched page; score by source type,
     official domain, context words, and how many independent domains agree
  -> {"answer", "sources", "confidence"}
```

Design rules:

- The model plans and reads; code fetches, checks and scores. Confidence is computed, not
  generated.
- An answer that no fetched page contains is thrown away, however confident the model is.
- Two model calls per question, hard deadlines on every network step, and a placeholder
  answer file written first, so a slow site can never leave a question without a file.
- No API keys anywhere: the image is public.

## Container contract

| Item | Value |
|---|---|
| Base image | `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` (final stage, not squashed) |
| Code | `/app/app.py` (thin client) + `/app/webagent/` |
| Start-up | `entrypoint.sh` downloads the model weights, then `python3 -m webagent.server` keeps the model resident |
| Per question | `python3 /app/app.py --query-id ID --query "..."` (also `--question`, `--input FILE`, `--input-dir DIR`) |
| Output | `/app/output/<id>_output.json` with `answer`, `sources`, `confidence` (plus `text`/`citations` aliases) |
| Model | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) via transformers on ROCm |
| torch | pinned to the base image's ROCm build; the build fails if `torch.version.hip` is missing |

The official Match 4 input/output contract had not been published when this was written;
`app.py` accepts the shapes used in Matches 2 and 3 and is a few lines to adapt.

## Reliability rules

- `app.py` writes a valid empty answer file before doing anything, and ends the process at a
  hard deadline (`WA_HARD_DEADLINE_S`, default budget + 3 s) even if a network call or the
  model hangs.
- The resident server opens its port at once; `/health` reports `ok: false` while weights load.
  A question that arrives early waits (up to `WA_SERVER_WAIT_S`) instead of loading a second
  copy of the model. A request to the server that fails is never retried in-process.
- If the model cannot be loaded, the server still answers with the deterministic extractors
  (typed regexes, `tel:`/JSON-LD, package registries, a rule-based entity guess) and marks the
  failure in `/health`. The grounding check and the confidence score do not depend on the model.
- The weight download retries four times before the server starts.
- Input forms accepted: `--query-id/--query`, `--question`, a positional question, stdin,
  `QUERY`/`QUERY_ID`, `--input FILE` (one object or a list), `--input-dir DIR`. Unknown flags are ignored.
- Abstention: by default the agent answers whenever a grounded candidate exists
  (`WA_ABSTAIN_BELOW=0`). On the 32 agent answers in `eval/results/`, all four answers below
  confidence 0.5 were wrong and none of the correct ones were below 0.75, so a threshold of 0.5
  is available if the spec turns out to penalise wrong answers more than empty ones.

## Evaluation

`eval/questions_dev.jsonl` (16) and `eval/questions_holdout.jsonl` (8): software versions,
official phone numbers and current office holders, with answers checked by hand on official
pages on 2026-10-15. Scoring uses the normalization from Matches 2 and 3 (upper-case, remove
whitespace and `- . · _`, exact match). The control is the same model with no web access.

See `eval/RESULTS.md` for the numbers and `eval/results/` for every run with full traces.

```
python eval/run_eval.py --set dev --mode both
python eval/run_eval.py --set holdout --mode both
python -m pytest -q tests          # offline tests, no network, no model
```

## Limits

- Keyless search engines throttle bursts of automated queries; the agent spreads queries over
  several engines and drops unrelated results, but a fully blocked network path still means
  no answer.
- Expected answer formats (for example phone number style) follow the earlier matches until
  the Match 4 spec says otherwise.
- Ground truth for "latest" questions changes over time; the date each answer was checked is
  stored with the question.

## License

MIT
