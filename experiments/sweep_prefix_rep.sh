#!/usr/bin/env bash
# Bench agentique prefix_repetition (doc 08 et 09) : TTFT et taux de hit selon la longueur
# du préfixe et le nombre d'agents, avec et sans prefix caching.
# num_prefixes = concurrence : chaque requête en vol appartient à un agent différent.
# --vary-seed : préfixes différents à chaque run, sinon le run 2 profite du cache du run 1.
# Surchargeable : EXPE, PREFIX_LENS (ex. PREFIX_LENS="8192" pour une 1re passe), CONCS.
# Prérequis conseillé : make obs-up (taux de hit et KV cache, clés srv_* de synthese.json).
# Graphe : uv run python -m inference_bench.analysis.prefix_rep sweep_prefix_rep
EXPE=${EXPE:-sweep_prefix_rep}
PREFIX_LENS=${PREFIX_LENS:-"2048 8192 16384"}
CONCS=${CONCS:-"1 2 4 8 16 32 64"}
for cache in fp8-cache fp8-nocache; do
  for p in $PREFIX_LENS; do
    for c in $CONCS; do
      uv run python bench/main.py --expe-name "$EXPE" --output_dir "${cache}_p${p}_conc_${c}" \
        --dataset configs/dataset/prefix-rep.yaml \
        --dataset-set prefix-repetition-prefix-len=$p prefix-repetition-num-prefixes=$c \
        --kv_cache configs/kv_cache/$cache.yaml \
        --engine configs/engine/ctx32k-seqs256-batch8k.yaml \
        --max-conc $c --prompts-per-conc 10 --vary-seed \
        || echo "ÉCHEC cache=$cache p=$p conc=$c"
    done
  done
done
