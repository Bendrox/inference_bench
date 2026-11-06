"""
Objective: measure the noise and the variance of results between runs.
Script to agregate the results of n benchmarks results. it aggregate 6 metrics: 
- output_throughput
- total_token_throughput
- median_ttft_ms
- p99_ttft_ms
- median_tpot_ms
- p99_tpot_ms
For every metric, it will compute mediane, min, max and diff between (max-min) / median x100.
"""


import json, statistics, sys
from pathlib import Path

CLES = ["output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"]
# Agrégées seulement si présentes : request_goodput vaut None quand le run est lancé sans --goodput
CLES_OPTIONNELLES = ["request_throughput", "request_goodput"]


def summarize(dossier: Path) -> dict:
    """Agrège les run*.json de `dossier`, écrit synthese.json et le retourne"""
    dossier = Path(dossier)
    fichiers = sorted(dossier.glob("run*.json"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun run*.json trouvé dans : {dossier}")

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in fichiers]
    synthese = {}
    for cle in CLES + CLES_OPTIONNELLES:
        vals = [r.get(cle) for r in runs]
        if None in vals:              # métrique optionnelle absente : on l'ignore
            continue
        med = statistics.median(vals)
        ecart = 100 * (max(vals) - min(vals)) / med if med else None   # goodput peut valoir 0
        synthese[cle] = {"mediane": med, "min": min(vals), "max": max(vals), "ecart_pct": ecart}

    with open(dossier / "synthese.json", "w", encoding="utf-8") as f:
        json.dump(synthese, f, ensure_ascii=False, indent=4)
    return synthese


def print_synthese(synthese: dict) -> None:
    print(f"{'métrique':<26}{'médiane':>12}{'min':>12}{'max':>12}{'écart %':>9}")
    for cle, s in synthese.items():
        ecart = "-" if s["ecart_pct"] is None else f"{s['ecart_pct']:.1f}"
        print(f"{cle:<26}{s['mediane']:>12.2f}{s['min']:>12.2f}{s['max']:>12.2f}{ecart:>9}")

if __name__ == "__main__":          # l'usage en ligne de commande continue de marcher
    if len(sys.argv) != 2:
        sys.exit("Usage : python bench/summarize.py <dossier_de_résultats>")
    print_synthese(summarize(Path(sys.argv[1])))



# uv run python bench/summarize.py results/sharegpt/Qwen3.5-9B-FP8-self_fp8-nocache_ctx8k-seqs256-batch8k_conc256-n1000
