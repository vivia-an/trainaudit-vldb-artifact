"""DeepSeek LLMClient for the S5 catalog ablation.

Implements the same `LLMClient` protocol as trainaudit's stub clients
(`__call__(system, user, *, max_tokens) -> str`), so it drops into
`propose_hypotheses(..., llm_client=...)` unchanged.

The API key is read from $DEEPSEEK_API_KEY — never hardcoded, so the
run configuration can be committed.
"""
from __future__ import annotations

import os
import time

import requests

API_URL = "https://api.deepseek.com/chat/completions"

# Pin the concrete model. The `deepseek-chat` alias currently resolves to
# deepseek-v4-flash; naming it explicitly keeps the two arms provably on
# the same model even if the alias moves.
MODEL = "deepseek-v4-flash"
TEMPERATURE = 1.0

# deepseek-v4-flash is a reasoning model: `reasoning_content` is billed
# against max_tokens before any `content` is emitted. At 1024 the thinking
# pass consumed the whole budget and content came back empty, so every
# hypothesis silently parsed to zero. Keep this generous so truncation is
# never a confound. Shared by both arms, so it cannot bias the comparison.
MAX_TOKENS = 32768


class DeepSeekClient:
    """Both ablation arms share one instance: identical model, temperature
    and max_tokens. The only difference between arms is the system prompt
    that `propose_hypotheses` selects via `use_catalog`."""

    def __init__(self, *, model: str = MODEL, temperature: float = TEMPERATURE,
                 seed: int | None = None, max_retries: int = 5):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.api_key = os.environ["DEEPSEEK_API_KEY"]
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, system: str, user: str, *,
                 max_tokens: int = MAX_TOKENS) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    API_URL, json=body, timeout=120,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(2 ** attempt)
                    continue
                blob = resp.json()
                usage = blob.get("usage", {})
                self.n_calls += 1
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return blob["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek call failed after "
                           f"{self.max_retries} attempts: {last_err}")
