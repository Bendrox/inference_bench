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
# cached_tokens_ratio : seulement dans les runs multiturn (taux de hit vu par le client)
CLES_OPTIONNELLES = ["request_throughput", "request_goodput", "cached_tokens_ratio"]
# Métriques serveur (server_metrics<i>.json, écrits par observability.collect), préfixées srv_
CLES_SERVEUR = ["prefix_cache_hit_rate", "kv_cache_usage_max", "requests_waiting_max",
                "preemptions", "gpu_power_avg_w", "energy_j", "joules_per_output_token"]


def _stats(vals: list[float]) -> dict:
    """Médiane, min, max et écart relatif (max - min) / médiane en %."""
    med = statistics.median(vals)
    ecart = 100 * (max(vals) - min(vals)) / med if med else None   # goodput peut valoir 0
    return {"mediane": med, "min": min(vals), "max": max(vals), "ecart_pct": ecart}


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
        synthese[cle] = _stats(vals)

    # Métriques serveur : seulement si chaque run a son server_metrics<i>.json
    serveurs = sorted(dossier.glob("server_metrics*.json"))
    if len(serveurs) == len(runs):
        srv = [json.loads(f.read_text(encoding="utf-8")) for f in serveurs]
        for s, r in zip(srv, runs):   # même tri : run1.json <-> server_metrics1.json
            # joules/token : seul endroit qui voit l'énergie (serveur) et les tokens (client)
            if s.get("energy_j") is not None and r.get("total_output_tokens"):
                s["joules_per_output_token"] = s["energy_j"] / r["total_output_tokens"]
        for cle in CLES_SERVEUR:
            vals = [s.get(cle) for s in srv]
            if None not in vals:      # ex. taux de hit None sans prefix caching
                synthese[f"srv_{cle}"] = _stats(vals)

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
        sys.exit("Usage : uv run python -m inference_bench.metrics <dossier_de_résultats>")
    print_synthese(summarize(Path(sys.argv[1])))

