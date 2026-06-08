# radioforge: RadioForge-SEMS

`radioforge` is a new agentic CUDA-kernel optimization baseline for `radio_bench`.

The method is **RadioForge-SEMS**: **Skill-conditioned Epilogue-Memory Search**.
It combines:

1. **CODA-style epilogue fusion idea**: explicitly asks the generator to fuse cheap elementwise/reduction epilogues into the main kernel instead of writing intermediate tensors.
2. **KernelMem-style memory**: stores successful candidates, failed compile/runtime traces, and reusable implementation patterns per task/operator signature.
3. **KDA-style staged workflow**: analyze → plan → generate → verify → repair → promote.
4. **4090-only specialization**: emits code and compile flags for `sm_89` only. Hopper/CuTeDSL-only paths are deliberately disabled.

The code is designed to be dropped into your repository root as:

```text
radioforge/
  run_radioforge.py
  radioforge/*.py
  prompts/*.md
  knowledge/*.md
  scripts/*.sh
```

It does **not** depend on the old `baselines/` directory. If you want to clear baselines, run `scripts/clear_baselines_keep_radioforge.sh` from your repo root.

## Quick smoke test

```bash
cd /path/to/radio_astronomy_bench
bash radioforge/scripts/run_radioforge_radio_smoke.sh
```

Default task:

```text
radio_bench/level1/05-ws-build-nm1.py
```

## Main command

```bash
python radioforge/run_radioforge.py \
  --tasks 'radio_bench/level1/*.py' \
  --out-dir radioforge/runs/$(date +%Y%m%d_%H%M%S)_radio \
  --rounds 3 \
  --arch 8.9 \
  --model deepseek-coder \
  --temperature 0.2
```

The runner supports OpenAI-compatible APIs through environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com/v1   # or your proxy
```

If no API key is available, it falls back to a safe reference-copy candidate so the pipeline remains debuggable.

## Outputs

For each task, the runner writes:

```text
radioforge/runs/.../
  candidates/<task-stem>/round_*.py
  logs/<task-stem>/round_*.json
  memory/kernel_memory.jsonl
  summary.json
```

## Method summary

For each input problem, RadioForge-SEMS builds an operator signature using AST/text heuristics, retrieves reusable implementation hints, then generates multiple candidates. Each candidate is evaluated. Compile/runtime/correctness feedback is fed back into the next repair prompt. The best correct candidate is promoted by speedup.

The intended novelty compared with `cudaforge`/`kernelmem` is not just “more iterations”, but a **structured optimization prior**:

- CODA-inspired epilogue-fusion checklist;
- explicit operator-signature memory instead of only task-level memory;
- repair prompts constrained by 4090/SM89;
- optional novelty guard to avoid regenerating almost identical code.


## Layout / run location

Put `radioforge/` and the three `run_radioforge_*.sh` files at the **repo root** of `radio_astronomy_bench`, the same directory that contains `radio_bench/`. The scripts export `PYTHONPATH="$PWD:${PYTHONPATH:-}"`, matching the KernelMem scripts after you uncommented that line.

`seed_ref` is disabled by default because some radio_bench reference tasks depend on helper functions such as `get_extension()` that are provided only in the original benchmark context. The normal RadioForge search starts from LLM candidates. Use `EXTRA_ARGS="--seed-reference"` only when debugging pure PyTorch tasks.
