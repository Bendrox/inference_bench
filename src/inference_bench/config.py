import argparse
from datetime import datetime
from pathlib import Path

import yaml

from .hardware import get_gpu_info, get_os_info

# src/inference_bench/config.py -> racine du repo (le package est installé en mode éditable)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH= PROJECT_ROOT / "configs"
BASE_FILE=CONFIG_PATH / "base.yaml"

def load_dataset(args):
    """Lit le YAML du profil : chaque clé est un argument de `vllm bench serve`."""
    with open(PROJECT_ROOT / args.dataset, encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_server_files_path(args) -> list[Path]:
    "Donne ordre de prio croissante : un fichier bas écrase le précédent"
    return [BASE_FILE, PROJECT_ROOT / args.model , 
            PROJECT_ROOT / args.kv_cache,
            PROJECT_ROOT / args.engine]

def assemble_yaml(args):
    import sys
    fusion = {}
    paths=get_server_files_path(args)
    for path in paths:

        if not path.is_file():
            sys.exit(f"Erreur: fichier introuvable: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                    contenu = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            sys.exit(f"ERREUR : yaml invalide dans {path} : {exc}")
        fusion.update(contenu)
    
    if args.gpu_mem:
        fusion["gpu-memory-utilization"]= args.gpu_mem
        
    sortie=Path(args.output_file)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with open(sortie, "w", encoding="utf-8") as f :
        yaml.safe_dump(fusion, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Fusion réussie ! Le fichier a été enregistré sous : {sortie}")
    return fusion

def nb_prompts(args, client: dict, max_conc: int) -> int:
    """Priorité : --num-prompts, puis --prompts-per-conc, puis le fichier client."""
    if args.num_prompts:
        return args.num_prompts
    if args.prompts_per_conc:
        return min(client["num-prompts"], max(100, args.prompts_per_conc * max_conc))
    return client["num-prompts"]

def load_client(args) -> dict:
    with open(PROJECT_ROOT / args.client, encoding="utf-8") as f:
        client= yaml.safe_load(f)
    max_conc = args.max_conc or client["max-concurrency"]
    return {
        "max_conc":max_conc,
        "num_prompts": nb_prompts(args, client, max_conc),
        "num_warmups": args.num_warmups or max_conc,
        "goodput": args.goodput,   # ex. ["ttft:1000", "tpot:50"], ou None
        "profil": load_dataset(args),
        
    }
    
def write_run_yaml(dossier: Path, args, config: dict, client: dict) -> None:
    """Snapshot complet de l'expérience dans <dossier>/run.yaml (par sections).
    `config` = config vLLM fusionnée (celle passée à `vllm serve --config`), `client` = valeurs résolues de load_client."""
    run = {
        "experiment": {"name": args.expe_name, "output_dir": args.output_dir,
                       "date": datetime.now().isoformat(timespec="seconds"),
                       "nb_runs": args.nb_runs},
        "sources": {"model": args.model, "kv_cache": args.kv_cache,
                    "engine": args.engine, "client": args.client, "dataset": args.dataset},
        "os": get_os_info(),
        "gpu": get_gpu_info(),
        "serve": config,
        "client": client,
    }
    with open(dossier / "run.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def read_arg(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Lance un benchmark vLLM à partir de configs composées")
    parser.add_argument("--output-file", default= "global_config.yaml")
    parser.add_argument("--model", default= "configs/models/Qwen3.5-9B-FP8-self.yaml")
    parser.add_argument("--dataset", default="configs/dataset/sharegpt.yaml")
    parser.add_argument("--kv_cache", default= "configs/kv_cache/fp8-nocache.yaml")
    parser.add_argument("--engine", default= "configs/engine/ctx8k-seqs256-batch8k.yaml")
    parser.add_argument("--client", default= "configs/client/conc256-n3000.yaml")
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
    return parser.parse_args(argv)
    

if __name__ == "__main__":
    args = read_arg()
    config = assemble_yaml(args)
    client = load_client(args)
    print("modèle :", config["model"])
    print("client :", client)