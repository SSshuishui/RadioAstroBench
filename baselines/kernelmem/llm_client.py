from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    api_key: str
    api_base: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_s: int = 180
    max_retries: int = 3

    @classmethod
    def from_env(
        cls,
        *,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_s: int = 180,
        max_retries: int = 3,
    ) -> "LLMConfig":
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("Missing API key. Set LLM_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
        base = api_base or os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_API_BASE")
        if not base:
            base = "https://api.openai.com/v1"
        mdl = model or os.environ.get("LLM_MODEL") or "gpt-4.1"
        return cls(key, base.rstrip("/"), mdl, temperature, max_tokens, timeout_s, max_retries)


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        last_error = None
        for attempt in range(self.config.max_retries):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
            except Exception as e:  # noqa: BLE001
                last_error = repr(e)
            if attempt + 1 < self.config.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM request failed: {last_error}")

    def complete_text(self, messages: List[Dict[str, str]]) -> str:
        data = self.chat(messages)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e
