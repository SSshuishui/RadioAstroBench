# KernelMem baseline adapter

This directory contains the KernelMem-style baseline adapted to `radio_bench`, based on the uploaded `KernelMem-main.zip` and aligned with the latest runnable CudaForge radio harness in this repo.

It keeps the KernelMem baseline ingredients:

- long-term memory from `memorybank/` and KernelMem prompt/memory files;
- short-term memory from previous local runs;
- LLM generation of an appended candidate `ModelNew`;
- compile/runtime/correctness feedback repair loop;
- optional NCU profiling feedback for `radio_bench` tasks;
- unified support for `radio_bench/` and `kernelbench/`.

## RadioBench integration

The radio evaluator uses:

```text
python -m radio_astronomy_cuda_bench.run_bench
```

and reads the current RadioBench fields:

```text
candidate_ms
candidate_speedup
candidate_correctness
```

It also supports real-data fixture runs:

```text
--fixture
--fixture-profile
--require-fixture
```

## Smoke run

From the repository root:

```bash
export LLM_API_KEY="..."
bash run_kernelmem_radio_smoke.sh
```

For a harness-only check on the default `05-ws-build-nm1` smoke task:

```bash
bash run_kernelmem_radio_mock.sh
```

## Real fixture run

```bash
export LLM_API_KEY="..."
export RKB_RADIO_ASTRO_DATA_ROOT="$PWD/radio_astro_data"
bash run_kernelmem_radio_real.sh nside512_day1_10m_ring
```

Optional NCU feedback:

```bash
ENABLE_NCU=1 bash run_kernelmem_radio_real.sh nside512_day1_10m_ring
```

## Single task run

```bash
export LLM_API_KEY="..."
bash run_kernelmem_radio_one.sh radio_bench/level1/05-ws-build-nm1.py
```

or directly:

```bash
python -m baselines.kernelmem.run_kernelmem \
  --bench radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3 \
  --continue-after-success
```

## Outputs

Runs are written to:

```text
baselines/kernelmem/runs/<timestamp>/
```

Each task/round stores:

```text
prompt.txt
llm_raw.txt
candidate_snippet.py
candidate_task.py
bench_result.json
stdout.txt
stderr.txt
result.json
```

The top-level `summary.json` includes:

```text
avg_speedup
accuracy
num_tasks
tasks
```
