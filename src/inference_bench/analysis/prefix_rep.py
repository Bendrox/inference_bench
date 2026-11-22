"""
prefix_rep.py : bench agentique prefix_repetition (doc 08 et 09).

Entrée : `results/<os>/<expe>/<cache>_p<prefix_len>_conc_<N>/synthese.json`

Sortie : `results/<os>/<expe>/prefix_rep.png` (ou --output), 3 graphiques :
  1. TTFT médian vs concurrence, une courbe par (cache, prefix_len) ;
  2. taux de hit mesuré côté serveur vs concurrence (cache activé) ;
  3. pic d'usage du KV cache vs concurrence : la falaise d'éviction.
Les graphiques 2 et 3 n'apparaissent que si les runs ont été lancés avec Prometheus
(clés srv_* de synthese.json).

Usage :  uv run python -m inference_bench.analysis.prefix_rep [nom_expe] [--output fichier.png]
         (défaut : "sweep_prefix_rep")
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .common import RESULTS, axe_concurrence, errorbar, sauver

MOTIF = re.compile(r"(?P<cache>.+)_p(?P<p>\d+)_conc_(?P<c>\d+)$")   # ex. fp8-cache_p8192_conc_16


def charger_grille(expe: Path) -> dict[tuple[str, int], dict[int, dict]]:
    """{(cache, prefix_len): {conc: synthese}}, triés. Quitte si aucun dossier ne correspond."""
    grille: dict = defaultdict(dict)
    for f in expe.glob("*/synthese.json"):
        m = MOTIF.match(f.parent.name)
        if m:
            grille[(m["cache"], int(m["p"]))][int(m["c"])] = json.loads(f.read_text(encoding="utf-8"))
    if not grille:
        sys.exit(f"Aucun <cache>_p<P>_conc_<N>/synthese.json dans {expe}")
    return {k: dict(sorted(v.items())) for k, v in sorted(grille.items())}


def avec_cache(cache: str) -> bool:
    return "nocache" not in cache


def tracer(ax, grille: dict, cle: str, titre: str, ylabel: str, garder=lambda cache: True) -> None:
    """Une courbe médiane + barres min/max par (cache, prefix_len), en fonction de la concurrence."""
    concs: set[int] = set()
    for (cache, p), data in grille.items():
        if not garder(cache):
            continue
        points = {c: s[cle] for c, s in data.items() if cle in s}
        if points:
            errorbar(ax, points, label=f"{cache} p={p}", capsize=3)
            concs |= set(points)
    if not concs:
        ax.text(0.5, 0.5, f"pas de données {cle}\n(runs lancés sans Prometheus ?)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    else:
        axe_concurrence(ax, concs)
        ax.legend(fontsize=7)
    ax.set_title(titre)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bench prefix repetition : TTFT, taux de hit, KV cache")
    parser.add_argument("expe", nargs="?", default="sweep_prefix_rep")
    parser.add_argument("--output", type=Path, default=None, help="défaut : results/<os>/<expe>/prefix_rep.png")
    args = parser.parse_args(argv)

    expe = RESULTS / args.expe
    grille = charger_grille(expe)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer(axes[0], grille, "median_ttft_ms", "TTFT médian", "ms")
    axes[0].set_yscale("log")          # cache vs sans cache : plusieurs ordres de grandeur
    tracer(axes[1], grille, "srv_prefix_cache_hit_rate", "Taux de hit mesuré",
           "tokens en cache / tokens demandés", avec_cache)
    tracer(axes[2], grille, "srv_kv_cache_usage_max", "Pic d'usage du KV cache",
           "fraction (0 à 1)", avec_cache)
    sauver(fig, args.output or expe / "prefix_rep.png", f"Prefix repetition : {args.expe}")


if __name__ == "__main__":
    main()
