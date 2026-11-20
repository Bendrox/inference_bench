# vllm-bench

A benchmarking harness for **vLLM** inference performance on an RTX 5090 (32 GB).
Each run starts a vLLM server from a composed config, drives it with `vllm bench serve`, repeats the measurement N times and computes the run-to-run noise.

## Setup

```bash
uv sync
```

## Running a benchmark

### Option 1 : a single run

One run = start the vLLM server → N measurements → summary.

**Example 1 : I want to measure the default model with 16 users at the same time, and check that responses stay fast (first token under 1 s, then under 50 ms per token):**

```bash
uv run python bench/main.py --expe-name my_test --output_dir conc_16 --max-conc 16 --goodput ttft:1000 tpot:50
```

**Example 2 : I want to test long prompts with short answers, with 32k context, 64 users at once and only 2 repetitions:**

```bash
uv run python bench/main.py --expe-name my_long_short --output_dir conc_64 \
  --dataset configs/dataset/long-short.yaml \
  --engine configs/engine/ctx32k-seqs256-batch8k.yaml \
  --max-conc 64 --nb-runs 2
```

Results land in `results/<os>/<expe-name>/<output_dir>/`.

### Option 2 : a ready-made sweep

Scripts in `experiments/` chain several runs (list them with `make available-benchmarks`):

```bash
bash experiments/sweep_gpumem.sh                # gpu-memory-utilization 0.55 → 0.9
bash experiments/sweep_concurrency_noslo.sh     # concurrency 1 → 512
bash experiments/sweep_concurrency_slo.sh       # same, with goodput (TTFT ≤ 1 s, TPOT ≤ 50 ms)
bash experiments/sweep_powerlimit_concur.sh     # concurrency SLO sweep × GPU power limit (needs sudo)
bash experiments/workload_profiles.sh           # workload profiles × concurrency (32k context)
bash experiments/workload_profiles_long-long.sh # long-long profile (lighter sweep)
```

Each script name is also its experiment name, i.e. its folder in `results/<os>/`.

## Project structure

```
bench/main.py            # entry point: one run = config → vLLM server → N bench runs → summary
src/inference_bench/     # Python package (installed by `uv sync`)
├── config.py            # composes the YAML configs, writes run.yaml
├── hardware.py          # GPU / OS description recorded in run.yaml
├── server.py            # vLLM server lifecycle (start, health check, stop)
├── loadgen.py           # load generator (`vllm bench serve`)
├── metrics.py           # aggregates run*.json into synthese.json
└── analysis/            # plots, one module per experiment family
configs/                 # composable configs (see below)
experiments/             # one script per experiment
results/<os>/<expe>/     # versioned results (light files only)
hardware/                # machine setup (MSI Afterburner / undervolt profiles)
```

## Composable configs


| Folder | Content | Example |
|---|---|---|
| `models/` | Model and quantization | `Qwen3-8B-FP8-ckpt.yaml` |
| `kv_cache/` | KV cache dtype, prefix caching | `fp8-nocache.yaml` |
| `engine/` | Context length, max sequences, batch size | `ctx8k-seqs256-batch8k.yaml` |
| `client/` | Benchmark-side concurrency and request count | `conc256-n3000.yaml` |
| `dataset/` | Prompts: ShareGPT or random input/output lengths | `short-long.yaml` |

## Results

Results are split by environment, detected automatically: `results/wsl/` (Windows + WSL) or `results/linux/` (native). 

An existing run is never overwritten (results take hours to produce and are versioned, so a re-run with the same `--expe-name` / `--output_dir` stops with an error instead of erasing them; the check is on the folder name, not on the parameters: two runs with exactly the same parameters but different folder names both run and are kept side by side, which costs a second run but gives a repeatability check; only a run interrupted before writing `synthese.json` can be re-run in the same folder). 

Plots read the current OS folder; use `BENCH_OS=wsl` to plot the other one.

```
results/<os>/<expe-name>/<output_dir>/
├── run.yaml          # full snapshot: server config, client, GPU, OS, date
├── server.log        # vLLM log (not versioned)
├── run{1..N}.json    # raw vllm bench serve results (not versioned)
├── bench_run{i}.txt  # console output
└── synthese.json     # median / min / max / spread % of the 6 key metrics
```

Aggregated metrics: output throughput, total throughput, TTFT (median, p99), TPOT (median, p99).


## Plots

```bash
uv run python -m inference_bench.analysis.concurrency sweep_concurrency_slo
uv run python -m inference_bench.analysis.concurrency sweep_powerlimit_concur/sweep_concurrency_slo_400w
uv run python -m inference_bench.analysis.gpu_memory sweep_gpumem
uv run python -m inference_bench.analysis.workload            # all workload_* profiles
make plots                                                     # all of the above for the current OS
```

Each accepts `--output <file.png>`; by default the figure is written next to the results.

## Tools (`Makefile`)

| Target | Purpose |
|---|---|
| `make help` (or `make`) | List all available targets |
| `make available-benchmarks` | List the ready-made sweeps in `experiments/` |
| `make plots` | Regenerate all plots for the current OS |
| `make nvidia-followup` | GPU monitoring (utilization, memory, power) every second |
| `make get-models` | Models in the HF cache |
| `make server-log` | Key lines from a `server.log` (KV cache, max concurrency…) |

## Planned to do:

- Quantization BF16 → FP8 / AWQ / GPTQ / NVFP4: speed vs quality.
- Agentic workload profiling.
- Pareto curve: throughput vs latency.
- Interpretation of the results.
- Multi-GPU benchmark: tensor (TP), data (DP) and pipeline (PP) parallelism.

Research notes live in `doc/` (not versioned).
