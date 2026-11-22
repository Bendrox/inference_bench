import subprocess
import time
from pathlib import Path

# Lignes affichées par `vllm bench serve` : elles délimitent la partie mesurée (hors warmup).
# Le warmup renvoie N fois la même requête (presque 100 % de hits) : il ne doit pas compter.
DEBUT_MESURE = "Starting main benchmark run"
FIN_MESURE = "Serving Benchmark Result"


def profil_to_args(profil: dict) -> list[str]:
    """{"random-input-len": 256, "ignore-eos": True} -> ["--random-input-len", "256", "--ignore-eos"]"""
    args = []
    for cle, val in profil.items():
        if val is True:                 # option sans valeur (flag) : présente
            args.append(f"--{cle}")
        elif val is False:              # flag désactivé : absent de la commande
            continue
        else:
            args += [f"--{cle}", str(val)]
    return args

def run_bench(i: int, dossier: Path, nom_hf: str, client: dict) -> tuple[float | None, float | None]:
    """Lance un run et retourne (t_start, t_end) de la partie mesurée, hors warmup.
    None si la ligne attendue n'est pas apparue (run planté)."""
    cmd= ["vllm", "bench", "serve",
          "--model", nom_hf,
          *profil_to_args(client["profil"]), 
          "--num-prompts", str(client["num_prompts"]),
          "--max-concurrency", str(client["max_conc"]),
          "--num-warmups", str(client["num_warmups"]),
          "--temperature", "0", # "--ignore-eos",
          "--save-result", "--result-dir",
          str(dossier),
          "--result-filename",
          f"run{i}.json",
          "--save-detailed",     # garde ttft/tpot par requête : goodput recalculable pour tout SLO
          ]
    if client["goodput"]:
        cmd += ["--goodput", *client["goodput"]]
    # Le serveur reste allumé pendant les N runs : avec la même seed, le run 2 retrouverait
    # les préfixes du run 1 déjà en cache (prefix_repetition). Désactivé par défaut.
    if client.get("vary_seed"):
        cmd += ["--seed", str(i)]
    with subprocess.Popen(                             # Ouvre un gestionnaire de contexte pour lancer le processus et garantir sa fermeture
        cmd,                                           # Fournit la liste des arguments de la commande système à exécuter
        stdout=subprocess.PIPE,                        # Crée un tuyau (pipe) pour intercepter la sortie standard (les logs du processus)
        stderr=subprocess.STDOUT,                      # Redirige et fusionne les erreurs (stderr) avec la sortie standard (stdout)
        text=True                                      # Configure les flux pour manipuler directement du texte (str) plutôt que des octets
    ) as p, open(dossier / f"bench_run{i}.txt", "w") as out: # Ouvre simultanément le fichier de log en écriture ("w") et nomme les variables 'p' et 'out'
        assert p.stdout is not None
        t_start = t_end = None
        for ligne in p.stdout:                         # Parcourt en temps réel chaque ligne générée par le processus
            if t_start is None and DEBUT_MESURE in ligne:
                t_start = time.time()                  # fin du warmup : début de la fenêtre de mesure
            elif t_start is not None and t_end is None and FIN_MESURE in ligne:
                t_end = time.time()                    # bandeau des résultats : fin de la fenêtre
            print(ligne, end="")                       # Affiche la ligne instantanément dans le terminal sans saut de ligne additionnel
            out.write(ligne)                           # Écrit simultanément cette même ligne dans le fichier de log
    return t_start, t_end
