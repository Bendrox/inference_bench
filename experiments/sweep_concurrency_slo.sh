#!/usr/bin/env bash
# Sweep de concurrence avec goodput : débit des requêtes qui respectent le SLO.
# SLO : TTFT <= 1000 ms et TPOT <= 50 ms (par requête).
# --prompts-per-conc : nb de requêtes proportionnel à la concurrence, sinon trop long et inutile.
# EXPE est surchargeable (utilisé par sweep_powerlimit_concur.sh).
# Graphe : uv run python -m inference_bench.analysis.concurrency sweep_concurrency_slo
EXPE=${EXPE:-sweep_concurrency_slo}
for c in 1 2 4 8 16 32 64 96 128 192 256 512; do
  uv run inference-bench --expe-name "$EXPE" --output_dir conc_$c \
    --max-conc $c --prompts-per-conc 10 --gpu-mem 0.9 \
    --engine configs/engine/ctx8k-seqs512-batch8k.yaml \
    --goodput ttft:1000 tpot:50 || echo "ÉCHEC conc=$c"
done
