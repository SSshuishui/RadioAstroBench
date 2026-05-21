from __future__ import annotations
import json
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import Any, Optional

DEFAULT_SYSTEM_PROMPT = """You are a senior CUDA-kernel optimisation specialist. Generate high-quality, compilable, runnable Python code that builds and launches hand-written CUDA kernels. Return code only in a python code block."""

GPU_SPEC = {
    "RTX 4090": {
        "GPU Architecture": "Ada Lovelace / SM 8.9",
        "SMs": "128",
        "CUDA cores": "16384",
        "Memory": "24GB GDDR6X",
        "Memory bandwidth": "~1008 GB/s",
        "Shared memory": "up to 99KB per SM depending on configuration",
        "Warp size": "32",
    },
    "NVIDIA GeForce RTX 4090": {
        "GPU Architecture": "Ada Lovelace / SM 8.9",
        "SMs": "128",
        "CUDA cores": "16384",
        "Memory": "24GB GDDR6X",
        "Memory bandwidth": "~1008 GB/s",
        "Warp size": "32",
    },
}

FEW_BASE = """import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n\ndef get_inputs():\n    return [torch.randn(1024, device='cuda')]\n\ndef get_init_inputs():\n    return []\n"""

FEW_NEW = """import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nsource = r'''\n#include <torch/extension.h>\n#include <cuda_runtime.h>\n__global__ void relu_kernel(const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i] > 0 ? x[i] : 0; }\ntorch::Tensor relu_forward(torch::Tensor x){ auto y=torch::empty_like(x); int n=x.numel(); relu_kernel<<<(n+255)/256,256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n); return y; }\n'''\ncpp_src = 'torch::Tensor relu_forward(torch::Tensor x);'\next = load_inline(name='relu_ext', cpp_sources=cpp_src, cuda_sources=source, functions=['relu_forward'], extra_cuda_cflags=['-O3'])\n\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return ext.relu_forward(x)\n"""


def _gpu_block(gpu_name: str) -> str:
    info = GPU_SPEC.get(gpu_name) or GPU_SPEC.get("RTX 4090") or {}
    arch = info.get("GPU Architecture", "Unknown")
    items = "\n".join(f"- {k}: {v}" for k, v in info.items() if k != "GPU Architecture")
    return f"GPU Name: {gpu_name}\nArchitecture: {arch}\n{items}"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def build_seed_prompt(arch_path: Path, gpu_name: str, dataset: str) -> str:
    src = _read(arch_path)
    extra = "" if dataset == "kernelbench" else dedent("""
    RADIO_BENCH RULES:
    - This task is an existing CUDA scientific benchmark. Preserve the physical/geometry semantics.
    - Do not remove occultation/visibility checks, half-symmetry behavior, or output shape/dtype semantics.
    - You may redefine ModelNew and add new CUDA extension code; do not change Model/get_inputs/TASK_ID.
    """).strip()
    return dedent(f"""
    # Target GPU
    {_gpu_block(gpu_name)}

    # Task
    Generate hand-written CUDA kernels that optimize the provided benchmark task. The reference class is Model; your output must define ModelNew with the same public inputs/outputs. You may fuse operations and change internal kernel implementation, but must preserve correctness.

    {extra}

    OUTPUT RULES:
    1. Return only complete Python code for ModelNew and helper extension code.
    2. The code must be self-contained when appended to the task file or used as a candidate module.
    3. Do not include testing code or prose.
    4. Do not specify -arch, -gencode, compute_XX, or sm_XX flags.

    Few-shot original:
    ```python
    {FEW_BASE}
    ```
    Few-shot optimized:
    ```python
    {FEW_NEW}
    ```

    Target benchmark file:
    ```python
    {src}
    ```
    """).strip()


def build_correctness_prompts(error_log: str, arch_path: Path, cuda_code: str) -> tuple[str, str]:
    sys = "You are a CUDA correctness debugging expert. Return a compact JSON diagnosis."
    prompt = dedent(f"""
    The candidate failed correctness/compilation/runtime checks.

    Error log:
    ```text
    {error_log[-8000:]}
    ```

    Task source:
    ```python
    {_read(arch_path)[-12000:]}
    ```

    Candidate code:
    ```python
    {cuda_code[-12000:]}
    ```

    Return JSON with keys: bottleneck, root_cause, fix_plan.
    """).strip()
    return sys, prompt


def build_error_prompt(old_code: str, error_log: str, problem: Any, gpu_name: str) -> str:
    return dedent(f"""
    # Target GPU
    {_gpu_block(gpu_name)}

    Repair the previous CUDA candidate. Preserve API and semantics.

    Problem analysis:
    ```json
    {json.dumps(problem, ensure_ascii=False, indent=2)}
    ```

    Error log:
    ```text
    {error_log[-8000:]}
    ```

    Previous candidate:
    ```python
    {old_code[-16000:]}
    ```

    Return only corrected Python code defining ModelNew and any helper extension code. No prose.
    """).strip()


def build_judger_optimization_prompts(arch_path: Path, gpu_name: str, ncu_metrics_block: str, cuda_code: str) -> tuple[str, str]:
    sys = "You are a CUDA optimization judge. Produce JSON strategy only."
    prompt = dedent(f"""
    Analyze this candidate and GPU profiling metrics. Identify the main bottleneck and a concrete optimization plan.

    GPU:
    {_gpu_block(gpu_name)}

    Task:
    ```python
    {_read(arch_path)[-12000:]}
    ```

    Candidate:
    ```python
    {cuda_code[-14000:]}
    ```

    NCU / runtime metrics:
    ```text
    {ncu_metrics_block[-12000:]}
    ```

    Return JSON with keys: bottleneck, optimisation method, modification plan, expected impact, risk.
    """).strip()
    return sys, prompt


def build_optimization_prompt(arch_path: Path, gpu_name: str, optimization_suggestion: Any, history_block: str = "") -> str:
    arch_src = _read(arch_path)
    return dedent(f"""
    # Target GPU
    {_gpu_block(gpu_name)}

    You are a CUDA-kernel optimization specialist. Apply the strategy below to produce a faster ModelNew while preserving correctness.

    Optimization strategy:
    ```json
    {json.dumps(optimization_suggestion, ensure_ascii=False, indent=2)}
    ```

    History:
    {history_block or '(None)'}

    Current candidate/task file:
    ```python
    {arch_src[-18000:]}
    ```

    OUTPUT RULES:
    - Return only Python code defining ModelNew and helper CUDA extension code.
    - Preserve public API, shapes, dtypes, and semantics.
    - Do not include testing code or prose.
    - Do not specify CUDA architecture flags.
    """).strip()
