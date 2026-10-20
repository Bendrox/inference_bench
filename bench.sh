#!/usr/bin/env bash
# ↑ "Shebang" : indique au système que ce fichier doit être exécuté par bash.
# => C'est ce qui permet de le lancer avec ./bench.sh

# bench.sh : lance UN benchmark complet 

# Usage   : ./bench.sh <modele> <variante>
# Exemple : ./bench.sh Qwen3-8B-FP8 fp8-kv
# Résultats  dans : results/Qwen3-8B-FP8_fp8-kv/

# VARIABLES

MODELE=$1
# $1 1er mot tapé après ./bench.sh
# => "./bench.sh Qwen3-8B-FP8 fp8-kv", MODELE vaut "Qwen3-8B-FP8"

VARIANTE=$2
# $2 = le 2e mot tapé. Ici VARIANTE vaut "fp8-kv"

DOSSIER=results/${MODELE}_${VARIANTE}

# ÉTAPE 1 : préparer le dossier et assembler la config

mkdir -p $DOSSIER
# -p    = crée dossier et les dossiers parents s'ils manquent (ici "results/"),
#         et ne fait pas d'erreur si le dossier existe déjà.

awk 1 configs/base.yaml configs/models/$MODELE.yaml configs/variants/$VARIANTE.yaml > $DOSSIER/config.yaml
# cat = affiche le contenu de fichiers, l'un après l'autre.
# >   = au lieu d'afficher à l'écran, écrit le tout dans un fichier.
# => Résultat : config.yaml contient base + modèle + variante collés ensemble.

NOM_HF=$(grep '^model:' $DOSSIER/config.yaml | awk '{print $2}')
# $( ... ) = exécute la commande entre parenthèses et met son résultat dans NOM_HF.
# grep '^model:' = garde uniquement la ligne qui COMMENCE (^) par "model:"
#                  -> "model: Qwen/Qwen3-8B-FP8"
# |              = "pipe" | : envoie le résultat de grep à la commande suivante.
# awk '{print $2}' = découpe la ligne aux espaces et garde le 2e morceau
#                  -> "Qwen/Qwen3-8B-FP8"
# Le benchmark a besoin de ce nom pour envoyer ses requêtes au bon modèle.


# ÉTAPE 2 : démarrer le serveur vLLM en arrière-plan
echo "Démarrage du serveur vllm avec ($MODELE / $VARIANTE)..."

vllm serve --config $DOSSIER/config.yaml > $DOSSIER/server.log 2>&1 &
# vllm serve --config ... = lance le serveur avec notre config assemblée.
# > $DOSSIER/server.log   = les messages normaux vont dans server.log.
# 2>&1                    = les messages d'erreur vont au même endroit
#                           (2 = canal des erreurs, 1 = canal normal).
# &                       = lance la commande EN ARRIÈRE-PLAN : le script
#                           continue sans attendre que le serveur s'arrête
#                           (sinon il resterait bloqué ici pour toujours).

PID=$!
# $! = le numéro de processus (PID) de la dernière commande lancée avec &.
# On le garde pour pouvoir surveiller puis arrêter le serveur plus tard.

# ÉTAPE 3 : attendre que le serveur soit prêt
# On interroge le serveur toutes les 5 secondes jusqu'à ce qu'il réponde.
 
until curl -s localhost:8000/health > /dev/null; do
# until ... do = répète le bloc JUSQU'À ce que la commande réussisse.
# curl -s localhost:8000/health = interroge l'adresse /health du serveur ;
#                                 elle répond "OK" quand le serveur est prêt.
#                                 -s = silencieux (pas de barre de progression).
# > /dev/null = jette la réponse : on veut juste savoir SI ça répond.

  if ! kill -0 $PID 2>/dev/null; then
  # kill -0 $PID = ne tue rien ! Vérifie seulement que le processus existe encore.
  # !            = inverse le test : "si le processus N'existe PLUS..."
  # 2>/dev/null  = cache le message d'erreur de kill.
  # Utile : si le serveur plante (mémoire insuffisante, mauvaise option...), sans ce test on attendrait indéfiniment.

    echo "ERREUR : le serveur a planté. Regarde $DOSSIER/server.log"
    exit 1
    # exit 1 = arrête le script. Le "1" signale une erreur (0 = succès).
  fi
  # fi = fin du "if" (c'est "if" à l'envers).
 
  sleep 5
  # Attend 5 secondes avant de réessayer.
done
# done = fin de la boucle "until".
 
echo "Serveur prêt. Benchmark en cours..."


# ÉTAPE 4 : lancer le benchmark

# Le "\" en fin de ligne signifie : "la commande continue à la ligne suivante".
 # | tee                = affiche le résumé à l'écran ET le copie dans bench.txt
vllm bench serve \
  --model $NOM_HF \
  --dataset-name sharegpt \
  --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --max-concurrency 128 \
  --save-result --result-dir $DOSSIER \
  | tee $DOSSIER/bench.txt
 
# ÉTAPE 5 : arrêter le serveur
 
kill $PID
# Envoie au serveur le signal d'arrêt (= Ctrl+C).
 
wait $PID 2>/dev/null
# Attend que le serveur soit VRAIMENT arrêté, et donc que la mémoire du GPU
# soit libérée. Indispensable si tu enchaînes plusieurs tests :
# le suivant ne doit pas démarrer tant que le GPU est encore occupé.
 
 
# ÉTAPE 6 : afficher les infos mémoire clés du log
 
echo "" 
echo "===== Infos mémoire (server.log) ====="
 
grep -iE "KV cache size|maximum concurrency" $DOSSIER/server.log
# grep = cherche des lignes dans un fichier.
# -i   = ignore majuscules/minuscules.
# -E   = permet le "|" qui veut dire "OU".
# Affiche donc les lignes du log qui parlent de la taille du KV cache
# OU de la concurrence maximale : les deux chiffres à noter dans ton carnet.
 
echo ""
echo "Terminé. Tout est dans $DOSSIER/"
 
