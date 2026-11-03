"""
Objective: measure the noise and the variance of results between runs.
Script to agregate the results of n benchmarks results. it aggregate 6 metrics: 
- output_throughput
- total_token_throughput
- median_ttft_ms
- p99_ttft_ms
- median_tpot_ms
- p99_tpot_ms
For every metric, it will compute mediane, min, max and diff between min and max.
"""


import json, glob, statistics, sys
from pathlib import Path
fichiers = sorted(glob.glob(sys.argv[1] + "/run*.json"))
cles = ["output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"]
txt=[]

fichiers = sorted(glob.glob(sys.argv[1] + "/run*.json"))
if not fichiers:
    sys.exit(f"Aucun run*.json trouvé dans : {sys.argv[1]}")

synthese = {}
for cle in cles:
    vals = [json.load(open(f))[cle] for f in fichiers]
    med = statistics.median(vals)
    synthese[cle] = {"mediane": med, "min": min(vals), "max": max(vals),
                     "ecart_pct": 100 * (max(vals) - min(vals)) / med}

with open(Path(sys.argv[1]) / "synthese.json", "w", encoding="utf-8") as f:
    json.dump(synthese, f, ensure_ascii=False, indent=4)



# uv run python summarize.py "$DOSSIER"
# uv run python bench/summarize.py results/sharegpt/Qwen3.5-9B-FP8-self_fp8-nocache_ctx8k-seqs256-batch8k_conc256-n1000
