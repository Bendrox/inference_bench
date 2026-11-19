#!/usr/bin/env bash
# Effet de gpu-memory-utilization (taille du KV cache) sur débit et latences.
# Graphe : uv run python -m inference_bench.analysis.gpu_memory sweep_gpumem
for g in 0.55 0.6 0.7 0.8 0.90; do
  uv run python bench/main.py --gpu-mem $g --output_dir gpu_$g --expe-name sweep_gpumem || echo "ÉCHEC gpu=$g"
done
