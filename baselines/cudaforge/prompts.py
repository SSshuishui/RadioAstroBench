from __future__ import annotations
import json
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import Any, Optional

DEFAULT_SYSTEM_PROMPT = """You are an expert CUDA/PyTorch extension optimizer.
You will receive one self-contained benchmark task file. The task already defines Model, get_inputs, and correctness checks.
Your job is to append a replacement candidate implementation by defining class ModelNew only.

Hard rules:
- Return only Python/CUDA code. Do not include prose.
- Do not modify get_inputs, Model, task metadata, fixture loading, or output comparison logic.
- Preserve the ModelNew.forward input signature and output semantics of Model.forward.
- Do not wrap, inherit from, instantiate, or delegate to the baseline Model.
- Do not call the original get_extension() function.
- If you define a new extension, use a separate loader name such as get_optimized_extension().
- Do not require new inputs. Any preprocessing must happen inside ModelNew.forward and is included in timing.
- Prefer correctness over risky optimization, but do not return a trivial identity wrapper.
"""

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


def build_seed_prompt(arch_path: Path, gpu_name: str, dataset: str, *, scale: str = "smoke", fixture_profile: Optional[str] = None) -> str:
    src = _read(arch_path)
    fixture_note = f"The evaluation may use fixture_profile={fixture_profile}; do not change fixture loading or get_inputs." if fixture_profile else "The evaluation may use synthetic inputs from get_inputs; do not change get_inputs."
    extra = "" if dataset == "kernelbench" else dedent(f"""
    RADIO_BENCH RULES:
    - This task is an existing CUDA scientific benchmark. Preserve the physical/geometry semantics.
    - Evaluation scale: {scale}. {fixture_note}
    - Do not remove occultation/visibility checks, half-symmetry behavior, or output shape/dtype semantics.
    - You may redefine ModelNew and add new CUDA extension code; do not change Model/get_inputs/TASK_ID/SUPPORTED_SCALES.
    - Do not inherit ModelNew from Model and do not instantiate/delegate to Model.
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
    5. Do not call get_extension() or baseline Model.

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

    Return only corrected Python code defining ModelNew and any helper extension code. No prose. Do not call get_extension() or baseline Model.
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
    - Do not call get_extension() or baseline Model.
    """).strip()
