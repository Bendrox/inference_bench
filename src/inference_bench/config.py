"""Configuration d'une expérience : configs YAML composées (serveur) + paramètres client.

- côté serveur : base.yaml + modèle + kv_cache + engine, fusionnés en une config `vllm serve` ;
- côté client : fichier client + profil dataset + options de la ligne de commande ;
- run.yaml : la trace complète de l'expérience (sources, machine, config résolue).
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .hardware import get_gpu_info, get_os_info

# src/inference_bench/config.py -> racine du repo (le package est installé en mode éditable)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = PROJECT_ROOT / "configs" / "base.yaml"


def lire_yaml(chemin: Path) -> dict:
    """Lit un YAML ; quitte avec un message clair si le fichier est absent ou invalide."""
    if not chemin.is_file():
        sys.exit(f"ERREUR : fichier introuvable : {chemin}")
    try:
        return yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        sys.exit(f"ERREUR : yaml invalide dans {chemin} : {exc}")


def ecrire_yaml(chemin: Path, data: dict) -> None:
    """Écrit `data` en YAML lisible (ordre des clés conservé, accents gardés)."""
    chemin.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True,
                                     default_flow_style=False), encoding="utf-8")


def assemble_yaml(args) -> dict:
    """Fusionne les configs serveur, par priorité croissante : un fichier écrase le précédent."""
    fusion = {}
    for chemin in [BASE_FILE, PROJECT_ROOT / args.model,
                   PROJECT_ROOT / args.kv_cache, PROJECT_ROOT / args.engine]:
        fusion.update(lire_yaml(chemin))
    if args.gpu_mem:
        fusion["gpu-memory-utilization"] = args.gpu_mem
    return fusion


def load_dataset(args) -> dict:
    """Lit le YAML du profil (chaque clé est un argument de `vllm bench serve`),
    puis applique les surcharges --dataset-set CLE=VAL."""
    profil = lire_yaml(PROJECT_ROOT / args.dataset)
    for paire in args.dataset_set or []:
        if "=" not in paire:
            sys.exit(f"ERREUR : --dataset-set attend CLE=VAL, reçu : {paire}")
        cle, val = paire.split("=", 1)
        profil[cle] = yaml.safe_load(val)   # "8192" -> 8192, "true" -> True
    return profil


def nb_prompts(args, client: dict, max_conc: int) -> int:
    """Priorité : --num-prompts, puis --prompts-per-conc, puis le fichier client."""
    if args.num_prompts:
        return args.num_prompts
    if args.prompts_per_conc:
        return min(client["num-prompts"], max(100, args.prompts_per_conc * max_conc))
    return client["num-prompts"]


def load_client(args) -> dict:
    client = lire_yaml(PROJECT_ROOT / args.client)
    max_conc = args.max_conc or client["max-concurrency"]
    return {
        "max_conc": max_conc,
        "num_prompts": nb_prompts(args, client, max_conc),
        "num_warmups": args.num_warmups or max_conc,
        "goodput": args.goodput,       # ex. ["ttft:1000", "tpot:50"], ou None
        "profil": load_dataset(args),
        "vary_seed": args.vary_seed,   # seed client = n° du run (évite un cache chaud entre runs)
    }


def write_run_yaml(dossier: Path, args, config: dict, client: dict) -> None:
    """Snapshot complet de l'expérience dans <dossier>/run.yaml (par sections).
    `config` = config vLLM fusionnée (celle passée à `vllm serve --config`), `client` = valeurs résolues de load_client."""
    ecrire_yaml(dossier / "run.yaml", {
        "experiment": {"name": args.expe_name, "output_dir": args.output_dir,
                       "date": datetime.now().isoformat(timespec="seconds"),
                       "nb_runs": args.nb_runs},
        "sources": {"model": args.model, "kv_cache": args.kv_cache,
                    "engine": args.engine, "client": args.client, "dataset": args.dataset},
        "os": get_os_info(),
        "gpu": get_gpu_info(),
        "serve": config,
        "client": client,
    })


def read_arg(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Lance un benchmark vLLM à partir de configs composées")
    parser.add_argument("--model", default="configs/models/Qwen3.5-9B-FP8-self.yaml")
    parser.add_argument("--dataset", default="configs/dataset/sharegpt.yaml")
    parser.add_argument("--kv_cache", default="configs/kv_cache/fp8-nocache.yaml")
    parser.add_argument("--engine", default="configs/engine/ctx8k-seqs256-batch8k.yaml")
    parser.add_argument("--client", default="configs/client/conc256-n3000.yaml")
    parser.add_argument("--max-conc", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--nb-runs", type=int, default=3)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="expe_1")
    parser.add_argument("--gpu-mem", type=float, default=None)
    parser.add_argument("--expe-name", type=str, default=None, required=True)
    parser.add_argument("--prompts-per-conc", type=int, default=None,
                        help="nb de requêtes = N x concurrence (borné entre 100 et la valeur du fichier client)")
    parser.add_argument("--goodput", nargs="+", default=None,
                        help="SLO en ms pour le goodput, ex. --goodput ttft:1000 tpot:50")
    parser.add_argument("--dataset-set", nargs="+", default=None, metavar="CLE=VAL",
                        help="surcharge le profil dataset, ex. --dataset-set prefix-repetition-prefix-len=8192")
    parser.add_argument("--vary-seed", action="store_true",
                        help="seed client différente à chaque run (évite un cache chaud entre runs)")
    return parser.parse_args(argv)
