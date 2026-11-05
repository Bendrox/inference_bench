# vllm-bench

A benchmarking harness for **vLLM** inference performance on an RTX 5090 (32 GB).
Each run starts a vLLM server from a composed config, drives it with `vllm bench serve`, repeats the measurement N times and computes the run-to-run noise.

## Setup

```bash
uv sync
make is-cuda-available-device   # check the GPU is visible

# ShareGPT dataset (not versioned)
mkdir -p data
hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
  ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir data
```

## Running a benchmark

```bash
uv run python bench/main.py --expe-name my_expe --output_dir run_1
```

Useful options:

| Option | Purpose | Default |
|---|---|---|
| `--expe-name` | Experiment name (required) | — |
| `--output_dir` | Run subfolder | `expe_1` |
| `--model` / `--kv_cache` / `--engine` / `--client` | Config files to compose | see `bench/config.py` |
| `--max-conc` | Max concurrency (overrides the client file) | client file |
| `--num-prompts` / `--prompts-per-conc` | Number of requests (fixed, or N × concurrency) | client file |
| `--gpu-mem` | `gpu-memory-utilization` | `0.9` |
| `--nb-runs` | Repetitions to measure noise | `3` |

Ready-made sweeps in `bench_sh/`:

```bash
bash bench_sh/run_bench_max_conc.sh   # concurrency 1 → 512
bash bench_sh/run_bench_gpumem.sh     # gpu-memory-utilization
```

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

## Results

```
results/<expe-name>/<output_dir>/
├── run.yaml          # full snapshot: server config, client, GPU, date
├── server.log        # vLLM log
├── run{1..N}.json    # raw vllm bench serve results
├── bench_run{i}.txt  # console output
└── synthese.json     # median / min / max / spread % of the 6 key metrics
```

Aggregated metrics: output throughput, total throughput, TTFT (median, p99), TPOT (median, p99).

```bash
uv run python bench/summarize.py results/<expe>/<run>   # re-aggregate a folder
uv run python bench/plot.py <expe>                      # gpu-mem sweep plots
```

## Tools (`makefile`)

| Target | Purpose |
|---|---|
| `make nvidia-followup` | GPU monitoring (utilization, memory, power) every second |
| `make get-models` | Models in the HF cache |
| `make server-log` | Key lines from a `server.log` (KV cache, max concurrency…) |

## Planned experiments

- Quantization BF16 → FP8 / AWQ / GPTQ / NVFP4: speed vs quality.
- Workload profiling on a fixed model: short/short, long prompt, long output, RAG, agentic.

Research notes live in `doc/` (not versioned).
