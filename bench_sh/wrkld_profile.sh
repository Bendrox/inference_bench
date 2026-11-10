# Workload profiling : profil × concurrence, setup figé 
# Phase 1 : bash bench_sh/wrkld_profile.sh
# Phase 2 : PROFILS="real-court-court real-court-long real-long-court" bash bench_sh/wrkld_profile.sh
PROFILS=${PROFILS:-"short-short short-long long-short"}
ENGINE=configs/engine/ctx32k-seqs256-batch8k.yaml
SLO="ttft:1000 tpot:50"

for p in $PROFILS; do
  for c in 1 4 16 64 128 256; do
    # if [ test ]; then A; else B; fi : exécute A si le test est vrai, sinon B ; "fi" ferme le if.
    #   [ $c -le 4 ] : compare deux entiers, -le = "less or equal" (<=). Les espaces après "[" et avant "]" sont obligatoires.
    #   N="..." : affecte le texte à la variable N (pas d'espace autour du "=").
    #   ";" sépare des commandes écrites sur une même ligne.
    if [ $c -le 4 ]; then N="--num-prompts 30"; else N="--prompts-per-conc 10"; fi

    # "\" en fin de ligne : la commande continue sur la ligne suivante (rien après, pas même un espace).
    # $c, $p, $ENGINE : remplacés par la valeur de la variable.
    # $N sans guillemets : bash découpe la valeur sur les espaces -> deux arguments ("--num-prompts" puis "30").
    #   Avec "$N", ce serait un seul argument "--num-prompts 30". Même chose pour $SLO.
    # A || B : exécute B seulement si A échoue (code de retour non nul).
    uv run python bench/main.py --expe-name wrkld_$p --output_dir conc_$c \
      --dataset configs/dataset/$p.yaml --engine $ENGINE \
      --max-conc $c $N --gpu-mem 0.9 \
      --goodput $SLO || echo "ÉCHEC $p conc=$c"
  done
done
