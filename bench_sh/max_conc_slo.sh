# Sweep de concurrence avec goodput : débit des requêtes qui respectent le SLO.
# SLO : TTFT <= 1000 ms et TPOT <= 50 ms (par requête).
# --prompts-per-conc : nb de requêtes proportionnel à la concurrence, sinon trop long et inutile.
for c in 1 2 4 8 16 32 64 96 128 192 256 512; do
  uv run python bench/main.py --expe-name sweep_concurrency_slo --output_dir conc_$c \
    --max-conc $c --prompts-per-conc 10 --gpu-mem 0.9 \
    --engine configs/engine/ctx8k-seqs512-batch8k.yaml \
    --goodput ttft:1000 tpot:50 || echo "ÉCHEC conc=$c"
done
