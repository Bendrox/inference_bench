from pathlib import Path

from client import run_bench
from config import PROJECT_ROOT, assemble_yaml, load_client, read_arg
from server import start_server, stop_server, wait_ready

def main():
    args= read_arg()
    
    nom = "_".join(Path(p).stem for p in (args.model, args.kv_cache, args.engine, args.client))
    dossier= PROJECT_ROOT / "results" /args.dataset / nom
    dossier.mkdir(parents=True, exist_ok=True)
    
    args.output_file= dossier / "serve.yaml"
    config= assemble_yaml(args)
    
    client = load_client(args)
    
    proc= start_server(dossier/ "serve.yaml", dossier / "server.log")
    
    try: 
        wait_ready(proc, config.get("port", 8000))
        for i in range(1, args.nb_runs +1):
            print(f"-- Run {i}/{args.nb_runs} --")
            run_bench(i, dossier, config["model"],
                      args.dataset,
                      client)
    finally:
        stop_server(proc)
        
if __name__ == "__main__":
    main()
            