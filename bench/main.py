import json
import socket
import sys
import tempfile
from pathlib import Path

from inference_bench import observability
from inference_bench.config import PROJECT_ROOT, assemble_yaml, load_client, read_arg, write_run_yaml
from inference_bench.hardware import os_tag
from inference_bench.loadgen import run_bench
from inference_bench.metrics import print_synthese, summarize
from inference_bench.server import start_server, stop_server, wait_ready

def main(argv: list[str] | None = None):
    args= read_arg(argv)
    
    dossier= PROJECT_ROOT / "results" / os_tag() / args.expe_name / args.output_dir
    if (dossier / "synthese.json").exists():
        sys.exit(f"ERREUR : {dossier} contient déjà des résultats, on n'écrase pas. "
                 f"Change --expe-name / --output_dir ou supprime le dossier.")
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
                t_start, t_end = run_bench(i, dossier, config["model"], client)
                if t_start is None or t_end is None:
                    print(f"Run {i} : fenêtre de mesure introuvable, métriques serveur ignorées")
                    continue
                # Métriques serveur (Prometheus) sur la fenêtre du run ; None si Prometheus est absent
                srv = observability.collect(t_start, t_end)
                if srv is not None:
                    (dossier / f"server_metrics{i}.json").write_text(
                        json.dumps(srv, indent=4), encoding="utf-8")
                    observability.annotate(t_start, t_end, f"{args.expe_name}/{args.output_dir} run {i}",
                                           [args.expe_name, args.output_dir])
        finally:
            stop_server(proc)

    print_synthese(summarize(dossier))
        
if __name__ == "__main__":
    main()
            