#!/usr/bin/env bash
# Container CMD: fetch weights once, then keep the model resident in a local server.
set -u
python3 - <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
m = os.environ.get("WA_LLM_MODEL", "")
if m and not os.path.isdir(m):
    t = time.time()
    path = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
    print(f"[entrypoint] weights ready in {time.time()-t:.0f}s at {path}", file=sys.stderr)
PY
cd /app && exec python3 -m webagent.server
