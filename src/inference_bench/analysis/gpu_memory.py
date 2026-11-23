"""
sweep_gpumem.py : trace l'effet de `gpu-memory-utilization` sur les métriques du benchmark.

Chaque lancement produit (voir cli.py et metrics.py) :
    results/<expe>/gpu_<valeur>/synthese.json

Ce script lit tous ces `synthese.json` d'une expérience, puis trace une courbe
par métrique en fonction de la valeur de gpu-memory-utilization.

Entrée: `results/<expe>/gpu_*/synthese.json`

Sortie:
`results/<expe>/gpu_sweep.png` (ou --output) : grille 2x3, un graphique par métrique de `metrics.CLES`.
  - le point est la médiane des runs ;
  - la barre d'erreur va du min au max des runs (= le bruit de mesure).
Si l'écart entre deux points est du même ordre que la barre d'erreur, la
différence n'est pas significative.

Usage :  uv run python -m inference_bench.analysis.gpu_memory [nom_expe] [--output fichier.png]
         (défaut : "sweep_gpumem")

"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..metrics import CLES   # liste des métriques, partagée avec metrics.py
from .common import RESULTS, charger, errorbar, sauver

PREFIXE = "gpu_"                       # préfixe des dossiers : gpu_0.7, gpu_0.8, ...


def tracer_metrique(ax, data: dict[float, dict], cle: str) -> None:
    """Trace une métrique sur l'axe `ax` : médiane + barre d'erreur min/max."""
    errorbar(ax, {gpu: s[cle] for gpu, s in data.items()}, capsize=4)
    ax.set_title(cle)
    ax.set_xlabel("gpu-memory-utilization")
    ax.set_xticks(list(data))          # une graduation par valeur testée
    ax.grid(alpha=0.3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Effet de gpu-memory-utilization")
    parser.add_argument("expe", nargs="?", default="sweep_gpumem")
    parser.add_argument("--output", type=Path, default=None, help="défaut : results/<os>/<expe>/gpu_sweep.png")
    args = parser.parse_args(argv)
    dossier = RESULTS / args.expe

    # La valeur de gpu-memory-utilization est lue dans le nom du dossier (gpu_0.7 -> 0.7) :
    # c'est pourquoi le balayage utilise `--output_dir gpu_<valeur>`.
    data = charger(dossier, PREFIXE, float)

    # 6 métriques -> grille 2 lignes x 3 colonnes. Si CLES change de taille,
    # adapter la grille (zip s'arrête à la plus courte des deux séquences).
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cle in zip(axes.flat, CLES):
        tracer_metrique(ax, data, cle)

    sauver(fig, args.output or dossier / "gpu_sweep.png", "Effet de gpu-memory-utilization "
                        "(point = médiane, barre = min/max sur les runs)")


if __name__ == "__main__":
    main()
