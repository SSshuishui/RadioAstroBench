from __future__ import annotations

import argparse
import json
from pathlib import Path

from vendor.compile_and_run import compare_and_bench


def main() -> None:
    p = argparse.ArgumentParser("KernelBench evaluator used by baselines/cudaforge")
    p.add_argument("--ref", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--json", required=True)
    args = p.parse_args()
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = compare_and_bench(
            ref_py=Path(args.ref),
            test_py=Path(args.candidate),
            warmup=args.warmup,
            repeat=args.repeat,
            tol=args.tol,
            log_dir=out.parent / "debug",
        )
        ref_avg = result["ref_latency_ms"]["avg"]
        test_avg = result["test_latency_ms"]["avg"]
        result["ok"] = True
        result["speedup"] = ref_avg / test_avg if test_avg > 0 else None
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        out.write_text(json.dumps({"ok": False, "error": str(e)}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
