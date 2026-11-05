import yaml
from pathlib import Path
import argparse

PROJECT_ROOT=Path(__file__).resolve().parent.parent
CONFIG_PATH= PROJECT_ROOT / "configs"
BASE_FILE=CONFIG_PATH / "base.yaml"
DATASETS = {"sharegpt": "data/ShareGPT_V3_unfiltered_cleaned_split.json"}

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


def load_client(args) -> dict:
    with open(PROJECT_ROOT / args.client, encoding="utf-8") as f:
        client= yaml.safe_load(f)
    max_conc = args.max_conc or client["max-concurrency"]
    return {
        "max_conc":max_conc,
        "num_prompts": args.num_prompts or client["num-prompts"],
        "num_warmups": args.num_warmups or max_conc,
    }
    
def read_arg():
    parser = argparse.ArgumentParser(description="Assemble les configs pour un benchmark")
    parser.add_argument("--output-file", default= "global_config.yaml")
    parser.add_argument("--model", default= "configs/models/Qwen3.5-9B-FP8-self.yaml")
    parser.add_argument("--dataset", default= "sharegpt", choices=DATASETS)
    parser.add_argument("--kv_cache", default= "configs/kv_cache/fp8-nocache.yaml")
    parser.add_argument("--engine", default= "configs/engine/ctx8k-seqs256-batch8k.yaml")
    parser.add_argument("--client", default= "configs/client/conc256-n1000.yaml")
    parser.add_argument("--max-conc", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--nb-runs", type=int, default=3)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="expe_1")
    parser.add_argument("--gpu-mem", type=float, default=None)
    parser.add_argument("--expe-name", type=str, default="study", required=True)
    return parser.parse_args()
    
if __name__ == "__main__":
    args=read_arg()
    assemble_yaml(args)
    
if __name__ == "__main__":
    args = read_arg()
    config = assemble_yaml(args)
    client = load_client(args)
    print("modèle :", config["model"])
    print("client :", client)