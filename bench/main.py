import socket
import sys
import tempfile
from pathlib import Path

from client import run_bench
from config import PROJECT_ROOT, assemble_yaml, load_client, read_arg, write_run_yaml
from server import start_server, stop_server, wait_ready
from summarize import print_synthese, summarize

def main():
    args= read_arg()
    
    dossier= PROJECT_ROOT / "results" / args.expe_name /args.output_dir 
    dossier.mkdir(parents=True, exist_ok=True)
    
    # vllm serve exige un fichier --config : on le génère dans un dossier temporaire
    # (supprimé à la sortie du `with`). La trace de l'expérience est dans run.yaml.
    with tempfile.TemporaryDirectory() as tmp:
        args.output_file= Path(tmp) / "serve.yaml"
        config= assemble_yaml(args)
        client = load_client(args)
        write_run_yaml(dossier, args, config, client)

        # Si un serveur (ex. orphelin d'un run interrompu) écoute déjà sur le port,
        # wait_ready le prendrait pour le nôtre et on mesurerait la mauvaise config.
        port = config.get("port", 8000)
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                sys.exit(f"ERREUR : le port {port} est déjà utilisé (serveur vLLM orphelin ?). "
                         f"Voir : ss -ltnp | grep :{port}")

        proc= start_server(args.output_file, dossier / "server.log")

        try:
            wait_ready(proc, port)
            for i in range(1, args.nb_runs +1):
                print(f"-- Run {i}/{args.nb_runs} --")
                run_bench(i, dossier, config["model"],
                          client)
        finally:
            stop_server(proc)

    print_synthese(summarize(dossier))
        
if __name__ == "__main__":
    main()
            