from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from baselines.llm_direct.evaluator import evaluate_candidate


FORBIDDEN_PATTERNS = [
    r"\bdef\s+get_inputs\b",
    r"\bclass\s+Model\s*\(",
    r"\bTASK_ID\b\s*=",
    r"\bSUPPORTED_SCALES\b\s*=",
]


SYSTEM_PROMPT = """You are an expert CUDA/PyTorch extension optimizer.
You will receive one self-contained benchmark task file. The task already defines Model, get_inputs, and correctness checks.
Your job is to append a replacement candidate implementation by defining class ModelNew only.

Hard rules:
- Return only Python/CUDA code. Do not use Markdown fences.
- Do not modify get_inputs, Model, task metadata, fixture loading, or output comparison logic.
- Preserve the ModelNew.forward input signature and output semantics of Model.forward.
- Do not wrap, inherit from, instantiate, or delegate to the baseline Model.
- Do not call the original get_extension() function.
- Do not call original extension functions through ext.<baseline_function>(...).
- Do not copy the original CUDA kernel and only rename functions or variables. A renamed clone is invalid.
- If you define a new extension, use a separate loader name such as get_optimized_extension().
- The candidate must make a material implementation change, such as changed launch geometry, tiling, memory access pattern, caching strategy, fusion, vectorization, or algorithmic decomposition.
- You may define helper CUDA/C++ source strings, helper functions, and a separate extension cache.
- Do not require new inputs. Any preprocessing must happen inside ModelNew.forward and is included in timing.
- Prefer correctness over risky optimization, but do not return a trivial identity wrapper.
"""


def repo_root_from_this_file() -> Path:
    # baselines/llm_direct/run_llm_direct.py -> repo root
    return Path(__file__).resolve().parents[2]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_task_name(task_path: Path, repo_root: Path) -> str:
    try:
        rel = task_path.resolve().relative_to(repo_root.resolve())
    except Exception:
        rel = task_path
    s = str(rel).replace(os.sep, "__")
    if s.endswith(".py"):
        s = s[:-3]
    return s


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def strip_code_fences(text: str) -> str:
    t = text.strip()
    # If the model returns one or more fenced blocks, concatenate python/code blocks.
    blocks = re.findall(r"```(?:python|py|cuda|cpp|c\+\+)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks).strip()
    return t


def _extract_original_ext_function_names(original_source: str) -> list[str]:
    """Find baseline extension functions called as ext.<name>(...) in the original task."""
    return sorted(set(re.findall(r"\bext\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", original_source)))


def _extract_cuda_source_blocks(source: str) -> list[str]:
    """Extract likely CUDA source string blocks from a task or snippet."""
    patterns = [
        r"""(?:CUDA_SRC(?:_[A-Za-z0-9_]+)?|cuda_src)\s*=\s*r?\"\"\"(.*?)\"\"\"""",
        r"""(?:CUDA_SRC(?:_[A-Za-z0-9_]+)?|cuda_src)\s*=\s*r?\'\'\'(.*?)\'\'\'""",
    ]
    blocks: list[str] = []
    for pat in patterns:
        blocks.extend(re.findall(pat, source, flags=re.DOTALL))
    return blocks


def _normalize_cuda_for_clone_check(src: str) -> str:
    """Normalize CUDA text so simple renaming still looks similar."""
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"\b(opt|optimized|candidate|new|fast)\b", "", src, flags=re.IGNORECASE)
    src = re.sub(r"(_opt|_optimized|_candidate|_new|_fast)\b", "", src, flags=re.IGNORECASE)
    src = re.sub(r"\s+", "", src)
    return src.lower()


def _max_cuda_clone_ratio(snippet: str, original_source: str) -> float:
    cand_blocks = [b for b in _extract_cuda_source_blocks(snippet) if len(b) >= 800]
    orig_blocks = [b for b in _extract_cuda_source_blocks(original_source) if len(b) >= 800]
    best = 0.0
    for cb in cand_blocks:
        nc = _normalize_cuda_for_clone_check(cb)
        if not nc:
            continue
        for ob in orig_blocks:
            no = _normalize_cuda_for_clone_check(ob)
            if not no:
                continue
            ratio = difflib.SequenceMatcher(None, nc, no).ratio()
            best = max(best, ratio)
    return best


def validate_snippet(snippet: str, original_source: str) -> tuple[bool, str]:
    if "class ModelNew" not in snippet:
        return False, "candidate snippet must define class ModelNew"

    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, snippet):
            return False, f"candidate snippet modifies forbidden object matching pattern: {pat}"

    banned_delegation_patterns = [
        r"\bget_extension\s*\(",
        r"\bclass\s+ModelNew\s*\(\s*Model\s*\)",
        r"\bModel\s*\(",
        r"\bsuper\s*\(\s*\)\.forward\s*\(",
        r"\boriginal_model\b",
        r"\bbase_model\b",
        r"\bbaseline_model\b",
    ]
    for pat in banned_delegation_patterns:
        if re.search(pat, snippet):
            return False, f"degenerate candidate delegates to baseline or wraps identity pattern: {pat}"

    # Reject direct calls to original extension symbols, e.g. ext.recon_xxx(...).
    for fn in _extract_original_ext_function_names(original_source):
        if re.search(rf"\bext\.{re.escape(fn)}\s*\(", snippet):
            return False, f"degenerate candidate calls original extension function: ext.{fn}"

    # Reject renamed clones of the baseline CUDA source.
    clone_ratio = _max_cuda_clone_ratio(snippet, original_source)
    if clone_ratio >= 0.92:
        return False, (
            "degenerate candidate appears to be a renamed copy of the baseline CUDA kernel "
            f"(cuda_clone_ratio={clone_ratio:.4f})"
        )

    return True, "ok"


def append_candidate(original_source: str, snippet: str, *, attempt: int) -> str:
    return (
        original_source.rstrip()
        + "\n\n\n# ============================================================\n"
        + f"# LLM candidate appended by baselines.llm_direct, attempt={attempt}\n"
        + "# ============================================================\n"
        + snippet.strip()
        + "\n"
    )


def make_initial_messages(task_path: Path, original_source: str, scale: str, fixture_profile: Optional[str]) -> list[dict[str, str]]:
    fixture_note = (
        f"The evaluation may use fixture_profile={fixture_profile}. Inputs are fixed by get_inputs/run_bench; do not change them."
        if fixture_profile
        else "The evaluation may use synthetic inputs from get_inputs. Do not change get_inputs."
    )
    user = f"""Optimize this CUDA benchmark task by appending a ModelNew implementation.

Task path: {task_path}
Scale: {scale}
{fixture_note}

Return only the appended candidate code.

Additional rejection rules:
- Do not call get_extension().
- Do not call original ext.<function>(...) baseline extension symbols.
- Do not inherit ModelNew from Model.
- Do not copy the original CUDA_SRC and merely rename functions.
- A valid candidate must make a material implementation change.

The original task source is below:

{original_source}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def make_repair_messages(
    task_path: Path,
    original_source: str,
    previous_snippet: str,
    eval_result: dict[str, Any],
    scale: str,
    fixture_profile: Optional[str],
) -> list[dict[str, str]]:
    stdout = str(eval_result.get("stdout", ""))[-12000:]
    stderr = str(eval_result.get("stderr", ""))[-12000:]
    bench = eval_result.get("bench_result")
    bench_text = json.dumps(bench, ensure_ascii=False, indent=2)[:12000]
    user = f"""The previous ModelNew candidate failed or was not accepted. Repair it.

Task path: {task_path}
Scale: {scale}
Fixture profile: {fixture_profile}

Previous candidate snippet:
{previous_snippet}

Evaluation bench_result:
{bench_text}

STDOUT tail:
{stdout}

STDERR tail:
{stderr}

Return only the corrected appended candidate code defining class ModelNew. Do not modify get_inputs or Model.

Additional rejection rules:
- Do not call get_extension().
- Do not call original ext.<function>(...) baseline extension symbols.
- Do not inherit ModelNew from Model.
- Do not copy the original CUDA_SRC and merely rename functions.
- A valid candidate must make a material implementation change.

Original task source:
{original_source}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class ChatClient:
    def __init__(self, *, api_key: str, api_base: str, model: str, timeout_s: int = 120):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        url = self.api_base + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {e.code}: {body[:2000]}") from e
        obj = json.loads(body)
        return obj["choices"][0]["message"]["content"]


def get_speedup(bench_result: Any) -> float:
    """Use only candidate timing for LLM-direct ranking.

    run_bench may also report identity_* fields for harness self-checks. Those fields are
    not valid optimization results for LLM-direct and must not be used for ranking.
    """
    if not isinstance(bench_result, dict):
        return float("-inf")
    v = bench_result.get("candidate_speedup")
    if isinstance(v, (int, float)):
        return float(v)
    base = bench_result.get("baseline_ms")
    cand = bench_result.get("candidate_ms")
    if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand > 0:
        return float(base) / float(cand)
    return float("-inf")


def optimize_one_task(args: argparse.Namespace, repo_root: Path, task_path: Path, client: Optional[ChatClient]) -> dict[str, Any]:
    original_source = read_text(task_path)
    run_root = Path(args.out_dir) if args.out_dir else repo_root / "baselines" / "llm_direct" / "runs" / now_stamp()
    task_run_dir = run_root / safe_task_name(task_path, repo_root)
    task_run_dir.mkdir(parents=True, exist_ok=True)
    write_text(task_run_dir / "original_task.py", original_source)

    attempts: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    previous_snippet = ""
    previous_eval: Optional[dict[str, Any]] = None

    for attempt in range(args.max_iters):
        attempt_dir = task_run_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        if args.mock:
            raw = "class ModelNew(Model):\n    pass\n"
        else:
            assert client is not None
            if attempt == 0 or previous_eval is None:
                messages = make_initial_messages(task_path, original_source, args.scale, args.fixture_profile)
            else:
                messages = make_repair_messages(
                    task_path,
                    original_source,
                    previous_snippet,
                    previous_eval,
                    args.scale,
                    args.fixture_profile,
                )
            raw = client.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)

        write_text(attempt_dir / "llm_raw.txt", raw)
        snippet = strip_code_fences(raw)
        ok_snip, reason = validate_snippet(snippet, original_source)
        write_text(attempt_dir / "candidate_snippet.py", snippet)

        if not ok_snip:
            attempt_result = {
                "attempt": attempt,
                "ok": False,
                "returncode": None,
                "error": reason,
                "candidate_snippet": str(attempt_dir / "candidate_snippet.py"),
            }
            attempts.append(attempt_result)
            previous_snippet = snippet
            previous_eval = {"stderr": reason, "bench_result": {"error": reason}}
            write_text(attempt_dir / "result.json", json.dumps(attempt_result, ensure_ascii=False, indent=2))
            continue

        candidate_source = append_candidate(original_source, snippet, attempt=attempt)
        candidate_task = attempt_dir / "candidate_task.py"
        write_text(candidate_task, candidate_source)

        eval_result = evaluate_candidate(
            repo_root=repo_root,
            task_path=candidate_task,
            original_task_path=task_path,
            scale=args.scale,
            segment_profile=args.segment_profile,
            fixture=args.fixture,
            fixture_profile=args.fixture_profile,
            warmup=args.warmup,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
            cuda_visible_devices=args.cuda_visible_devices,
            json_path=attempt_dir / "bench_result.json",
        )
        write_text(attempt_dir / "stdout.txt", str(eval_result.get("stdout", "")))
        write_text(attempt_dir / "stderr.txt", str(eval_result.get("stderr", "")))

        attempt_result = {
            "attempt": attempt,
            "ok": bool(eval_result.get("ok")),
            "returncode": eval_result.get("returncode"),
            "candidate_task": str(candidate_task.relative_to(repo_root)) if candidate_task.is_absolute() else str(candidate_task),
            "candidate_snippet": str((attempt_dir / "candidate_snippet.py").relative_to(repo_root)) if (attempt_dir / "candidate_snippet.py").is_absolute() else str(attempt_dir / "candidate_snippet.py"),
            "bench_result": eval_result.get("bench_result"),
        }
        attempts.append(attempt_result)
        write_text(attempt_dir / "result.json", json.dumps(attempt_result, ensure_ascii=False, indent=2))

        if attempt_result["ok"]:
            if best is None or get_speedup(attempt_result.get("bench_result")) > get_speedup(best.get("bench_result")):
                best = attempt_result
            if not args.continue_after_success:
                break

        previous_snippet = snippet
        previous_eval = eval_result

    summary = {
        "task": str(task_path.relative_to(repo_root)) if task_path.is_absolute() else str(task_path),
        "scale": args.scale,
        "segment_profile": args.segment_profile,
        "fixture": args.fixture,
        "fixture_profile": args.fixture_profile,
        "attempts": attempts,
        "best": best,
    }
    write_text(task_run_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def collect_tasks(repo_root: Path, task_arg: Optional[str], level_arg: str) -> list[Path]:
    if task_arg:
        return [Path(task_arg) if Path(task_arg).is_absolute() else repo_root / task_arg]

    if level_arg in ("", "all"):
        levels = ["level1", "level2", "level3"]
    else:
        levels = [level_arg]

    tasks: list[Path] = []
    for lv in levels:
        d = repo_root / "radio_bench" / lv
        tasks.extend(sorted(d.glob("*.py")))
    return tasks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-direct CUDA optimizer baseline for radio_astronomy_cuda_bench tasks")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=str, default=None, help="Task file path, e.g. radio_bench/level2/03-ws-recon-representative.py")
    g.add_argument("--level", type=str, default="", choices=["", "all", "level1", "level2", "level3"], help="Run all tasks in one level or all levels")

    p.add_argument("--scale", type=str, default="smoke")
    p.add_argument("--segment-profile", type=str, default="all10")
    p.add_argument("--fixture", type=str, default=None, help="Explicit fixture path for one task")
    p.add_argument("--fixture-profile", type=str, default=None, help="Real-data fixture profile, e.g. nside512_day1_10m_ring")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--continue-after-success", action="store_true")
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument("--cuda-visible-devices", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)

    p.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "deepseek-v4-pro"))
    p.add_argument("--api-base", type=str, default=os.environ.get("LLM_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.1")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", "12000")))
    p.add_argument("--mock", action="store_true", help="Do not call LLM; use identity ModelNew for harness testing")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = repo_root_from_this_file()

    os.environ.setdefault("PYTHONPATH", str(repo_root))
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("RKB_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")
    os.environ.setdefault("CUDAHOSTCXX", "/usr/bin/g++")
    os.environ.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")

    tasks = collect_tasks(repo_root, args.task, args.level)
    if not tasks:
        print("No tasks found.", file=sys.stderr)
        return 2

    client: Optional[ChatClient] = None
    if not args.mock:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("Missing LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY. Use --mock for harness testing.", file=sys.stderr)
            return 2
        client = ChatClient(api_key=api_key, api_base=args.api_base, model=args.model)

    batch: list[dict[str, Any]] = []
    failed = 0
    for task in tasks:
        print("\n" + "=" * 72)
        print(f"[LLM DIRECT] task={task.relative_to(repo_root) if task.is_absolute() else task}")
        print(f"scale={args.scale} fixture_profile={args.fixture_profile} mock={args.mock}")
        print("=" * 72)
        t0 = time.time()
        try:
            summary = optimize_one_task(args, repo_root, task, client)
            best_ok = bool(summary.get("best"))
            if not best_ok:
                failed += 1
            batch.append(summary)
            dt = time.time() - t0
            print(f"[DONE] ok={best_ok} elapsed={dt:.1f}s")
            if summary.get("best"):
                br = summary["best"].get("bench_result") or {}
                print(f"       speedup={get_speedup(br):.4g}")
        except Exception as e:
            failed += 1
            err = {"task": str(task), "error": repr(e)}
            batch.append(err)
            print(f"[ERROR] {task}: {e}", file=sys.stderr)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = repo_root / "baselines" / "llm_direct" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / f"batch_summary_{now_stamp()}.json", json.dumps(batch, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print(f"LLM direct batch summary: total={len(tasks)}, failed={failed}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
