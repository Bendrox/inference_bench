#!/usr/bin/env bash
# ↑ "Shebang" : indique au sys que ce fichier doit être exécuté par bash.
# => C'est ce qui permet de le lancer avec ./run_bensh.sh

# run_bensh.sh : lance UN benchmark complet (serveur vLLM + N runs + résumé)

# Usage   : ./run_bensh.sh <modele> <dataset> <kv_cache> <engine> <client>
# Exemple : ./run_bensh.sh Qwen3.5-9B-FP8-self sharegpt fp8-nocache ctx8k-seqs256-batch8k conc256-n1000
# Chaque argument est le nom d'un fichier .yaml (sans l'extension) dans configs/ :
#   modele   -> configs/models/<modele>.yaml
#   kv_cache -> configs/kv_cache/<kv_cache>.yaml
#   engine   -> configs/engine/<engine>.yaml   (options de "vllm serve")
#   client   -> configs/client/<client>.yaml   (options de "vllm bench serve")
# Résultats dans : results/<dataset>/<modele>_<kv_cache>_<engine>_<client>/

# Réglages modifiables au lancement (facultatif) :
# on les écrit juste devant la commande, et ils remplacent la valeur du fichier
# pour CE lancement seulement (les fichiers .yaml ne sont jamais modifiés).
#
#   Variable      Sert à                          Sans la variable, on utilise
#   MAX_CONC      requêtes simultanées            max-concurrency de configs/client/<client>.yaml
#   NUM_PROMPTS   nombre de requêtes par run      num-prompts de configs/client/<client>.yaml
#   NB_RUNS       nombre de runs successifs       3
#   NUM_WARMUPS   requêtes d'échauffement         la valeur de MAX_CONC
#
# Ex. : MAX_CONC=64 NB_RUNS=1 ./run_bensh.sh Qwen3.5-9B-FP8-self sharegpt fp8-nocache ctx8k-seqs256-batch8k conc256-n1000

set -euo pipefail
# set -e          = le script s'arrête dès qu'une commande échoue.
# set -u          = utiliser une variable jamais définie est une erreur.
# set -o pipefail = dans "a | b", l'échec de "a" compte aussi (pas seulement celui de "b").

# VARIABLES
if [ $# -ne 5 ]; then
  echo "Usage : $0 <modele> <dataset> <kv_cache> <engine> <client>"
  echo "Ex.   : $0 Qwen3.5-9B-FP8-self sharegpt fp8-nocache ctx8k-seqs256-batch8k conc256-n1000"
  exit 1
fi
# $# = nombre d'arguments tapés. S'il n'y en a pas exactement 5, on affiche l'usage et on sort.

MODELE=$1; DATA=$2; KV_CACHE=$3; ENGINE=$4; CLIENT=$5
# $1 = 1er mot tapé après ./run_bensh.sh, $2 = le 2e, etc.

DOSSIER="results/$DATA/${MODELE}_${KV_CACHE}_${ENGINE}_${CLIENT}"
CLIENT_FILE="configs/client/$CLIENT.yaml"
SERVER_FILES=(configs/base.yaml "configs/models/$MODELE.yaml"
              "configs/kv_cache/$KV_CACHE.yaml" "configs/engine/$ENGINE.yaml")
# SERVER_FILES est une liste (tableau) : les 4 fichiers qui forment la config du SERVEUR.
# On l'écrit une seule fois ; "${SERVER_FILES[@]}" la redonne en entier.
# Le fichier client est à part : ses options n'existent pas pour "vllm serve".

# ÉTAPE 1 : vérifier les fichiers, préparer le dossier, assembler la config
for f in "${SERVER_FILES[@]}" "$CLIENT_FILE"; do
  [ -f "$f" ] || { echo "ERREUR : fichier introuvable : $f"; exit 1; }
done
# [ -f "$f" ] = vrai si le fichier existe. "|| { ... }" = sinon, message d'erreur et arrêt.

mkdir -p "$DOSSIER"
# -p = crée le dossier et ses parents s'ils manquent, sans erreur s'il existe déjà.

awk 1 "${SERVER_FILES[@]}" > "$DOSSIER/serve.yaml"
# awk 1 = recopie les fichiers l'un après l'autre en ajoutant un retour à la ligne
#         final si besoin (nos yaml n'en ont pas, "cat" collerait deux lignes).
# >      = écrit le résultat dans serve.yaml au lieu de l'afficher.
# => serve.yaml est EXACTEMENT ce que reçoit "vllm serve --config".

cp "$CLIENT_FILE" "$DOSSIER/client.yaml"
# On garde une copie des paramètres de charge pour retrouver comment le run a été fait.

NOM_HF=$(awk '/^model:/{print $2}' "$DOSSIER/serve.yaml")
# $( ... ) = exécute la commande et met son résultat dans NOM_HF.
# awk '/^model:/{print $2}' = sur la ligne qui COMMENCE par "model:", affiche le 2e mot
#                             -> "Qwen/Qwen3.5-9B". Le bench en a besoin pour viser le bon modèle.

MAX_CONC=${MAX_CONC:-$(awk '/^max-concurrency:/{print $2}' "$CLIENT_FILE")}
NUM_PROMPTS=${NUM_PROMPTS:-$(awk '/^num-prompts:/{print $2}' "$CLIENT_FILE")}
NB_RUNS=${NB_RUNS:-3}
NUM_WARMUPS=${NUM_WARMUPS:-$MAX_CONC}
# ${VAR:-valeur} = utilise VAR si elle est définie, sinon "valeur".
# => une variable d'environnement (MAX_CONC=64 ./run_bensh.sh ...) prend le dessus sur le fichier.
# NB_RUNS     = nombre de passages successifs : sert à mesurer le bruit entre runs.
# NUM_WARMUPS = requêtes d'échauffement envoyées avant chaque mesure (non comptées).

case "$DATA" in
  sharegpt) DATA_FILE="data/ShareGPT_V3_unfiltered_cleaned_split.json" ;;
  *) echo "ERREUR : dataset '$DATA' inconnu (attendu : sharegpt)"; exit 1 ;;
esac
# case = choisit selon la valeur de DATA. Pour ajouter un dataset : une ligne "nom) FICHIER ;;".

# ÉTAPE 2 : démarrer le serveur vLLM en arrière-plan
echo "Démarrage du serveur ($DATA / $MODELE / $KV_CACHE / $ENGINE / $CLIENT)..."

vllm serve --config "$DOSSIER/serve.yaml" > "$DOSSIER/server.log" 2>&1 &
# > server.log = les messages normaux vont dans server.log.
# 2>&1         = les messages d'erreur vont au même endroit (2 = erreurs, 1 = normal).
# &            = lance EN ARRIÈRE-PLAN : le script continue sans attendre la fin du serveur
#                (sinon il resterait bloqué ici pour toujours).

PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
# $! = numéro de processus (PID) de la dernière commande lancée avec &.
# trap EXIT = exécute cette commande quand le script se termine, quelle qu'en soit la raison
#             (erreur, Ctrl+C...) : le serveur n'est jamais laissé orphelin.

# ÉTAPE 3 : attendre que le serveur soit prêt (ou qu'il plante)
until curl -s localhost:8000/health > /dev/null; do
  kill -0 $PID 2>/dev/null || { echo "ERREUR : serveur planté. Voir $DOSSIER/server.log"; exit 1; }
  sleep 5
done
# until ... do = répète le bloc JUSQU'À ce que la commande réussisse.
# curl -s .../health = interroge le serveur ; il répond quand il est prêt (-s = silencieux).
# kill -0 $PID       = ne tue rien : vérifie seulement que le processus existe encore.
#                      S'il n'existe plus (mémoire insuffisante, mauvaise option...), on
#                      s'arrête au lieu d'attendre indéfiniment.
# sleep 5            = pause de 5 secondes entre deux essais.

echo "Serveur prêt. Benchmark en cours..."

# ÉTAPE 4 : lancer le benchmark, NB_RUNS fois de suite sur le même serveur
for i in $(seq 1 "$NB_RUNS"); do
  echo "--- Run $i/$NB_RUNS ---"
  vllm bench serve \
    --model "$NOM_HF" \
    --dataset-name "$DATA" \
    --dataset-path "$DATA_FILE" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONC" \
    --num-warmups "$NUM_WARMUPS" \
    --temperature 0 \
    --ignore-eos \
    --save-result --result-dir "$DOSSIER" --result-filename "run${i}.json" \
    | tee "$DOSSIER/bench_run${i}.txt"
done
# Les runs se SUIVENT (pas en parallèle) : l'objectif est de mesurer le bruit d'une mesure
# à l'autre, pour savoir si un écart entre deux configs est réel.
# \                 = la commande continue à la ligne suivante.
# --num-warmups     = requêtes d'échauffement (compilation, graphes CUDA) non comptées.
# --temperature 0   = réponses déterministes (sinon le serveur choisit son défaut).
# --ignore-eos      = chaque requête génère la longueur prévue par le dataset :
#                     même quantité de travail à chaque run.
# --result-filename = un fichier JSON par run (run1.json, run2.json...).
# | tee             = affiche le résumé à l'écran ET le copie dans bench_runN.txt.

# ÉTAPE 5 : arrêter le serveur
kill $PID
# Envoie au serveur le signal d'arrêt (= Ctrl+C).

wait $PID 2>/dev/null || true
# Attend que le serveur soit VRAIMENT arrêté, donc que la mémoire du GPU soit libérée.
# Indispensable si tu enchaînes plusieurs tests : le suivant ne doit pas démarrer
# tant que le GPU est occupé.
# "|| true" : après un kill, wait renvoie un code d'erreur normal, qui arrêterait
# le script à cause de "set -e".

# ÉTAPE 6 : afficher les infos mémoire clés du log
echo ""
echo "===== Infos mémoire (server.log) ====="
grep -iE "KV cache size|maximum concurrency" "$DOSSIER/server.log" || true
# grep -i = ignore majuscules/minuscules ; -E permet le "|" qui veut dire "OU".
# Affiche les lignes du log sur la taille du KV cache OU la concurrence maximale.
# "|| true" : si rien n'est trouvé, grep renvoie une erreur qui ne doit pas tout arrêter.

echo ""
echo "Terminé. Tout est dans $DOSSIER/"
