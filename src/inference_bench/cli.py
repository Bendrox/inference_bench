"""Point d'entrée du harness : une expérience = config -> serveur vLLM -> N runs -> synthèse.

Usage :  uv run inference-bench --expe-name <nom> [options]   (commande déclarée dans pyproject.toml)
         uv run python -m inference_bench.cli --expe-name <nom> [options]
"""
import sys
import tempfile
from pathlib import Path

from inference_bench import observability
from inference_bench.config import (PROJECT_ROOT, assemble_yaml, ecrire_yaml, load_client, read_arg,
                                    write_run_yaml)
from inference_bench.hardware import os_tag
from inference_bench.loadgen import run_bench
from inference_bench.metrics import print_synthese, summarize
from inference_bench.server import port_libre, start_server, stop_server, wait_ready


def main(argv: list[str] | None = None):
    args = read_arg(argv)

    # 1. Dossier de résultats : on n'écrase jamais une expérience terminée
    dossier = PROJECT_ROOT / "results" / os_tag() / args.expe_name / args.output_dir
    if (dossier / "synthese.json").exists():
        sys.exit(f"ERREUR : {dossier} contient déjà des résultats, on n'écrase pas. "
                 f"Change --expe-name / --output_dir ou supprime le dossier.")

    # 2. Configs serveur et client
    config = assemble_yaml(args)
    client = load_client(args)
    port = config.get("port", 8000)
    if not port_libre(port):
        sys.exit(f"ERREUR : le port {port} est déjà utilisé (serveur vLLM orphelin ?). "
                 f"Voir : ss -ltnp | grep :{port}")

    dossier.mkdir(parents=True, exist_ok=True)
    write_run_yaml(dossier, args, config, client)

    # 3. Serveur + N runs. vllm serve exige un fichier --config : on l'écrit dans un dossier
    #    temporaire (supprimé à la sortie du `with`), la trace de l'expérience est dans run.yaml.
    with tempfile.TemporaryDirectory() as tmp:
        serve_yaml = Path(tmp) / "serve.yaml"
        ecrire_yaml(serve_yaml, config)
        proc = start_server(serve_yaml, dossier / "server.log")
        try:
            wait_ready(proc, port)
            for i in range(1, args.nb_runs + 1):
                print(f"-- Run {i}/{args.nb_runs} --")
                t_start, t_end = run_bench(i, dossier, config["model"], client)
                if t_start is None or t_end is None:
                    print(f"Run {i} : fenêtre de mesure introuvable, métriques serveur ignorées")
                    continue
                observability.enregistrer(dossier, i, t_start, t_end, [args.expe_name, args.output_dir])
        finally:
            stop_server(proc)

    # 4. Synthèse des N runs
    print_synthese(summarize(dossier))


if __name__ == "__main__":
    main()
