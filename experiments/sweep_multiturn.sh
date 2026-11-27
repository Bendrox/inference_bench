#!/usr/bin/env bash
# Bench multi-tours : sessions dont le contexte grandit à chaque tour (historique réel renvoyé),
# avec et sans prefix caching, selon le nombre de sessions simultanées.
# Générateur : inference_bench/multiturn.py (pas `vllm bench serve`, qui ne gère pas les sessions).
#
# Pièges couverts ici :
# - ctx32k : à 30 tours x (256 + 256) tokens + 2048 de system, une session atteint ~17k tokens ;
#   avec ctx8k elle s'arrêterait au tour ~11 et on ne verrait pas la croissance.
# - --vary-seed : conversations différentes à chaque run, sinon le run 2 tape dans le cache du run 1.
# - CONC vide => max-concurrency = sessions (toutes actives) : à 64 sessions, éviction + préemptions
#   + batch saturé, effets mélangés. CONC=8 => charge GPU constante d'un point à l'autre, des
#   sessions attendent et leurs blocs sont évictables (cas agentique réel). Évictable != évincé : le
#   cache ne craque que si sessions x contexte > KV cache (~48 sessions de 30 tours ici), mais la
#   hausse de TTFT vient alors uniquement de l'éviction. Cerner la falaise : SESSIONS="16 32 48 64 96".
#   THINK>0 (pause outil entre deux tours) a le même effet que CONC < sessions.
# - Sans cache, le TTFT croît avec le contexte : un gros SESSIONS x TURNS en nocache est long.
# - Modèle par défaut Qwen3.5-9B = hybride : blocs de cache de 1056 tokens (voir server.log), le hit
#   par tour monte en escalier. Pour un modèle à attention pure : --model configs/models/Qwen3-8B-FP8-ckpt.yaml
# Surchargeable : EXPE, SESSIONS, CONC, TURNS, THINK, NB_RUNS.
# Prérequis conseillé : make obs-up (taux de hit serveur et pic KV cache, clés srv_*).
# Graphe : uv run python -m inference_bench.analysis.multiturn sweep_multiturn
EXPE=${EXPE:-sweep_multiturn}
SESSIONS=${SESSIONS:-"4 8 16 32 64"}
CONC=${CONC:-}
TURNS=${TURNS:-30}
THINK=${THINK:-0}
NB_RUNS=${NB_RUNS:-2}
for cache in fp8-cache fp8-nocache; do
  for s in $SESSIONS; do
    c=${CONC:-$s}
    uv run inference-bench --expe-name "$EXPE" --output_dir "${cache}_s${s}_conc_${c}" \
      --dataset configs/dataset/multiturn.yaml \
      --dataset-set multiturn-sessions=$s multiturn-turns=$TURNS multiturn-think-time=$THINK \
      --kv_cache configs/kv_cache/$cache.yaml \
      --engine configs/engine/ctx32k-seqs256-batch8k.yaml \
      --max-conc "$c" --vary-seed --nb-runs "$NB_RUNS" \
      || echo "ÉCHEC cache=$cache sessions=$s conc=$c"
  done
done
