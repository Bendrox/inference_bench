# vllm-bench

A benchmarking harness for **vLLM** inference performance on an RTX 5090 (32 GB).
Each run starts a vLLM server from a composed config, drives it with `vllm bench serve`, repeats the measurement N times and computes the run-to-run noise.

## Setup

```bash
uv sync
```

## Running a benchmark

Ready-made sweeps in `bench_sh/`:

```bash
bash bench_sh/gpumem.sh            # gpu-memory-utilization 0.55 → 0.9
bash bench_sh/max_conc_noslo.sh    # concurrency 1 → 512
bash bench_sh/max_conc_slo.sh      # same, with goodput (TTFT ≤ 1 s, TPOT ≤ 50 ms)
bash bench_sh/wrkld_profile.sh     # workload profiles × concurrency (32k context)
bash bench_sh/wrkld_profile_ll.sh  # long-long profile (lighter sweep)
```


Useful options:

| Option | Purpose | Default |
|---|---|---|
| `--expe-name` | Experiment name (required) | — |
| `--output_dir` | Run subfolder | `expe_1` |
| `--model` / `--kv_cache` / `--engine` / `--client` / `--dataset` | Config files to compose | see `bench/config.py` |
| `--max-conc` | Max concurrency (overrides the client file) | client file |
| `--num-prompts` / `--prompts-per-conc` | Number of requests (fixed, or N × concurrency) | client file |
| `--gpu-mem` | `gpu-memory-utilization` | `0.9` |
| `--nb-runs` | Repetitions to measure noise | `3` |
| `--num-warmups` | Warmup requests | max concurrency |
| `--goodput` | SLO in ms, e.g. `ttft:1000 tpot:50` | none |

## Composable configs

The server config is merged in this order (later files override earlier ones):

```
configs/base.yaml → configs/models/*.yaml → configs/kv_cache/*.yaml → configs/engine/*.yaml
```

| Folder | Content | Example |
|---|---|---|
| `models/` | Model and quantization | `Qwen3-8B-FP8-ckpt.yaml` |
| `kv_cache/` | KV cache dtype, prefix caching | `fp8-nocache.yaml` |
| `engine/` | Context length, max sequences, batch size | `ctx8k-seqs256-batch8k.yaml` |
| `client/` | Benchmark-side concurrency and request count | `conc256-n3000.yaml` |
| `dataset/` | Prompts: ShareGPT or random input/output lengths | `short-long.yaml` |

## Results

```
results/<expe-name>/<output_dir>/
├── run.yaml          # full snapshot: server config, client, GPU, OS, date
├── server.log        # vLLM log (not versioned)
├── run{1..N}.json    # raw vllm bench serve results (not versioned)
├── bench_run{i}.txt  # console output
└── synthese.json     # median / min / max / spread % of the 6 key metrics
```

Aggregated metrics: output throughput, total throughput, TTFT (median, p99), TPOT (median, p99).


## Tools (`makefile`)

| Target | Purpose |
|---|---|
| `make nvidia-followup` | GPU monitoring (utilization, memory, power) every second |
| `make get-models` | Models in the HF cache |
| `make server-log` | Key lines from a `server.log` (KV cache, max concurrency…) |

## Progress

All runs so far were made on Windows + WSL Ubuntu; next ones run on native Ubuntu (see the `os` section of each `run.yaml`).

Planned:
- Quantization BF16 → FP8 / AWQ / GPTQ / NVFP4: speed vs quality.
- Agentic workload profiling.
- Pareto curve: throughput vs latency.
- GPU power limit sweep and its impact.
- Interpretation of the results.
- Native Linux benchmark, compared with Windows + WSL.
- Multi-GPU benchmark: tensor (TP), data (DP) and pipeline (PP) parallelism.

Research notes live in `doc/` (not versioned).
