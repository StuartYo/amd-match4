# Evaluation results (2026-10-15)

Same questions, same model, two conditions:

- **agent**: the full pipeline in `app/webagent/`
- **control**: the same model, no web access, asked for the value directly (`agent.baseline_answer`)

Local runs used `Qwen/Qwen3-1.7B` on a laptop RTX 4050 shared with other jobs. The challenge
image uses `Qwen/Qwen3-4B-Instruct-2507` on an AMD GPU; those runs are still to do, so the
timings below are an upper bound, not the timings on the evaluation hardware.

Scoring: Match 2/3 grader normalization (upper-case, remove whitespace and `- . · _`), exact
match against the accepted answers in the question file.

| Set | Questions | When the set was frozen | Agent | Control | File |
|---|---|---|---|---|---|
| dev | 16 | used for tuning | 15/16 | 0/16 | `dev_20261015-111013.json` (agent), `dev_20261015-105143.json` (control) |
| holdout | 8 | before the first run | **5/8** | **0/8** | `holdout_20261015-111257.json` |
| holdout2 | 8 | written and committed before its first run (commit "holdout2 set (frozen before run)") | **6/8** | **0/8** | `holdout2_20261015-111942.json` |

The holdout set was re-run after fixes (8/8, `holdout_20261015-111639.json`), but that number
is not a clean measurement because the fixes were made after looking at its failures. The
clean figures are the bold ones: **11/16 for the agent, 0/16 for the control** on questions
not used for tuning.

## What the control gets wrong

Every control answer was wrong. Typical errors: outdated versions (`requests 3.1.0`,
`Node.js 18.14`), invented phone numbers (`123-456-7890` for Grand Canyon), and naming the
company instead of its CEO. The model's knowledge stops at its training date, and a small
model does not refuse.

## Remaining agent failures (clean sets)

| Question | Agent answer | Why |
|---|---|---|
| Latest Ollama release | 0.32.6 | no canonical-repo lookup yet; fixed afterwards |
| USPTO general phone | (empty) | official homepage not fetched; fixed afterwards |
| Latest vLLM release | 13.0 | picked a CUDA version from an install page; fixed afterwards (GitHub canonical repo) |
| Latest Docker Compose | 26.1 | Docker Engine version on the same pages; fixed afterwards (two-word repo names) |
| Yosemite general questions phone | 877-444-6777 | picked the reservations line; qualifier words ("general questions") not weighted enough |

## Timing and tokens (local laptop)

Agent: mean 12–16 s per question, two model calls, about 1,900 tokens per question.
The first question of each local run also pays model loading; in the container the model is
loaded once at start-up by `webagent.server`.

## Known measurement risks

- Search engines throttled this machine after repeated runs (DuckDuckGo returned HTTP 202,
  Brave HTTP 429). The agent falls back across engines; dev runs used a local cache
  (`WA_CACHE_DIR`) so repeated tuning runs saw the same pages. Holdout runs did not use the cache.
- Ground truth for "latest" questions was checked on 2026-10-15 and will drift.
