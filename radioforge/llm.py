from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class LLMConfig:
    model: str
    temperature: float = 0.2
    timeout_s: int = 120


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Uses only Python stdlib so this method can run in a clean baseline env.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.openai.com/v1"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.available:
            raise RuntimeError("No OPENAI_API_KEY/DEEPSEEK_API_KEY available")
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]
