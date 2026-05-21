from __future__ import annotations
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, List

_CODE_FENCE_OPEN_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]+)?\s*\n?")


def extract_code_block(text: str) -> str:
    text = text or ""
    m = _CODE_FENCE_OPEN_RE.search(text)
    if not m:
        return text.strip() + "\n"
    start = m.end()
    m2 = re.search(r"```", text[start:])
    end = start + m2.start() if m2 else len(text)
    return text[start:end].strip() + "\n"


def extract_json(raw: str) -> Any:
    raw = str(raw or "")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"raw": raw[-4000:]}


def extract_cuda_kernel_names(py_path: Path) -> List[str]:
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    pats = [
        re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
        re.compile(r"__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    ]
    out, seen = [], set()
    for pat in pats:
        for name in pat.findall(src):
            if name not in seen:
                seen.add(name); out.append(name)
    return out


def save_kernel_code(code: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"kernel_{stamp}.py"
    path.write_text(code, encoding="utf-8")
    return path


def task_slug(path: str | Path) -> str:
    s = str(Path(path).with_suffix(""))
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", s.replace("/", "__").replace("\\", "__")).strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
