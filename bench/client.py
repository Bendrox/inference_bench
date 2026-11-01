import subprocess
from pathlib import Path

from config import DATASETS

def run_bench(i:int, dossier: Path, nom_hf: str, dataset: str, client: dict) -> None:
    cmd= ["vllm, bench", "serve",
          "--model", nom_hf,
          "--dataset-name", dataset,
          "--dataset-path", DATASETS[dataset], 
          "--num-prompts", str(client["num_prompts"]),
          "--num-concurrency", str(client["max_conc"]),
          "--num-warmups", str(client["num_warmups"]),
          "temperature", "0", "--ignore-eos",
          "save-result", "--result-dir", 
          str(dossier),
          "--result-filename",
          f"run{i}.json"
          ]
    with subprocess.Popen(                             # Ouvre un gestionnaire de contexte pour lancer le processus et garantir sa fermeture
        cmd,                                           # Fournit la liste des arguments de la commande système à exécuter
        stdout=subprocess.PIPE,                        # Crée un tuyau (pipe) pour intercepter la sortie standard (les logs du processus)
        stderr=subprocess.STDOUT,                      # Redirige et fusionne les erreurs (stderr) avec la sortie standard (stdout)
        text=True                                      # Configure les flux pour manipuler directement du texte (str) plutôt que des octets
    ) as p, open(dossier / f"bench_run{i}.txt", "w") as out: # Ouvre simultanément le fichier de log en écriture ("w") et nomme les variables 'p' et 'out'
        assert p.stdout is not None
        for ligne in p.stdout:                         # Parcourt en temps réel chaque ligne générée par le processus
            print(ligne, end="")                       # Affiche la ligne instantanément dans le terminal sans saut de ligne additionnel
            out.write(ligne)                           # Écrit simultanément cette même ligne dans le fichier de log