#!/usr/bin/env bash
# Container CMD: fetch weights once (with retries), then keep the model resident in a local server.
# /tmp/wa_starting tells app.py that a server is on its way, so an early question waits for it
# instead of loading a second copy of the model.
set -u
rm -f /tmp/wa_ready
touch /tmp/wa_starting
python3 - <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
m = os.environ.get("WA_LLM_MODEL", "")
if m and not os.path.isdir(m):
    t = time.time()
    for attempt in range(1, 5):
        try:
            path = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
            print(f"[entrypoint] weights ready in {time.time()-t:.0f}s at {path}", file=sys.stderr)
            break
        except Exception as e:  # network blip: retry, the 10 minute start-up budget is generous
            print(f"[entrypoint] download attempt {attempt} failed: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5 * attempt)
PY
cd /app && exec python3 -m webagent.server
