#!/usr/bin/env bash
# Sweep de concurrence sans SLO : débit brut et latences selon max-concurrency.
# --prompts-per-conc : nb de requêtes proportionnel à la concurrence, sinon trop long et inutile.
# Graphe : uv run python -m inference_bench.analysis.concurrency sweep_concurrency_noslo
for c in 1 2 4 8 16 32 64 96 128 192 256 512; do
  uv run python bench/main.py --expe-name sweep_concurrency_noslo --output_dir conc_$c \
    --max-conc $c --prompts-per-conc 10 --gpu-mem 0.9 \
    --engine configs/engine/ctx8k-seqs512-batch8k.yaml || echo "ÉCHEC conc=$c"
done
