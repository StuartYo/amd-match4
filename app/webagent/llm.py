"""Small LLM wrapper with two backends and a token counter.

Both backends expose the same `chat(messages, max_tokens) -> str` call. The
transformers backend runs in-process (CUDA locally, ROCm in the challenge
image); the OpenAI-compatible backend talks to a vLLM server started by the
container entrypoint.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 2),
        }


class BaseLLM:
    name = "base"

    def chat(self, messages: list[dict], max_tokens: int = 256, usage: Usage | None = None, max_time: float | None = None) -> str:
        raise NotImplementedError


class TransformersLLM(BaseLLM):
    name = "transformers"

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        self._torch = torch
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    def chat(self, messages, max_tokens=256, usage=None, max_time=None):
        torch = self._torch
        try:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with self._lock, torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.eos_token_id,
                max_time=max_time,
            )
        gen = out[0][ids["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        if usage is not None:
            usage.add(Usage(1, int(ids["input_ids"].shape[1]), int(gen.shape[0]), time.time() - t0))
        return strip_think(text)


class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def __init__(self, base_url: str, model: str):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.client = httpx.Client(timeout=60)

    def chat(self, messages, max_tokens=256, usage=None, max_time=None):
        t0 = time.time()
        body = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self.client.post(self.base_url + "/chat/completions", json=body, timeout=max_time or 60)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        if usage is not None:
            u = data.get("usage") or {}
            usage.add(Usage(1, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), time.time() - t0))
        return strip_think(text)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict | None:
    """Return the first JSON object found in a model reply, or None."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    return None


_LLM: BaseLLM | None = None
_LLM_LOCK = threading.Lock()


def get_llm() -> BaseLLM:
    global _LLM
    with _LLM_LOCK:
        if _LLM is None:
            if config.LLM_BACKEND == "openai":
                _LLM = OpenAICompatLLM(config.LLM_BASE_URL, config.LLM_MODEL)
            else:
                _LLM = TransformersLLM(config.LLM_MODEL)
        return _LLM


def set_llm(llm: BaseLLM | None) -> None:
    """Inject a backend (used by tests with a fake model)."""
    global _LLM
    _LLM = llm
