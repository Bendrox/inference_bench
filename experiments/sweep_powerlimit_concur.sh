#!/usr/bin/env bash
# Sweep power limit × concurrence (Linux natif) : rejoue sweep_concurrency_slo.sh
# pour chaque power limit GPU. Nécessite sudo pour nvidia-smi -pl.
# Résultats : results/linux/sweep_powerlimit_concur/sweep_concurrency_slo_<W>w/conc_<N>/
# Graphe d'une limite : uv run python -m inference_bench.analysis.concurrency sweep_powerlimit_concur/sweep_concurrency_slo_400w
set -e
LIMITS=${LIMITS:-"400 448 510 575"}
HERE=$(dirname "$0")
for w in $LIMITS; do
  sudo nvidia-smi -pl "$w"
  EXPE="sweep_powerlimit_concur/sweep_concurrency_slo_${w}w" bash "$HERE/sweep_concurrency_slo.sh"
done
sudo nvidia-smi -pl "$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)"
