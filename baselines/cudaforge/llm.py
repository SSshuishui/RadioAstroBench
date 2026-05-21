from __future__ import annotations
import csv
import datetime as dt
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any


def _env_key(server_type: str) -> Optional[str]:
    if server_type == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if server_type == "openai":
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")


def _base_url(server_type: str, server_address: str, server_port: int) -> str:
    if server_type == "deepseek":
        return os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").rstrip("/")
    if server_type == "openai":
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if server_type in {"vllm", "sglang"}:
        return f"http://{server_address}:{server_port}/v1".rstrip("/")
    return os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")


def _log_usage(log_path: Optional[Path], round_idx: int, call_type: str, usage: Dict[str, Any]):
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "round_idx", "call_type", "input_tokens", "output_tokens", "total_tokens"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "round_idx": round_idx,
            "call_type": call_type,
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0,
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        })


def query_server(
    prompt: str,
    system_prompt: str,
    *,
    server_type: str,
    model_name: str,
    server_address: str = "localhost",
    server_port: int = 8000,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 8192,
    log_path: Optional[Path] = None,
    call_type: str = "unknown",
    round_idx: int = -1,
    timeout_s: int = 600,
) -> str:
    key = _env_key(server_type)
    if not key:
        raise RuntimeError("Missing API key. Set LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY.")
    base = _base_url(server_type, server_address, server_port)
    url = base + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _log_usage(log_path, round_idx, call_type, data.get("usage", {}) or {})
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f"LLM request failed: {last}")
