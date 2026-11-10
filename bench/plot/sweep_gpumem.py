"""
sweep_gpumem.py : trace l'effet de `gpu-memory-utilization` sur les métriques du benchmark.

Chaque lancement produit (voir main.py et summarize.py) :
    results/<expe>/gpu_<valeur>/synthese.json

Ce script lit tous ces `synthese.json` d'une expérience, puis trace une courbe
par métrique en fonction de la valeur de gpu-memory-utilization.

Entrée: `results/<expe>/gpu_*/synthese.json`

Sortie:
`results/<expe>/gpu_sweep.png` : grille 2x3, un graphique par métrique de `summarize.CLES`.
  - le point est la médiane des runs ;
  - la barre d'erreur va du min au max des runs (= le bruit de mesure).
Si l'écart entre deux points est du même ordre que la barre d'erreur, la
différence n'est pas significative.

Usage :  uv run python bench/plot/sweep_gpumem.py [nom_expe]     (défaut : "study_gpumem")

"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from common import RESULTS, charger, errorbar, sauver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # bench/, pour importer summarize
from summarize import CLES   # liste des métriques, partagée avec summarize.py

EXPE = sys.argv[1] if len(sys.argv) > 1 else "study_gpumem"
DOSSIER = RESULTS / EXPE
PREFIXE = "gpu_"                       # préfixe des dossiers : gpu_0.7, gpu_0.8, ...
SORTIE = DOSSIER / "gpu_sweep.png"


def tracer_metrique(ax, data: dict[float, dict], cle: str) -> None:
    """Trace une métrique sur l'axe `ax` : médiane + barre d'erreur min/max."""
    errorbar(ax, {gpu: s[cle] for gpu, s in data.items()}, capsize=4)
    ax.set_title(cle)
    ax.set_xlabel("gpu-memory-utilization")
    ax.set_xticks(list(data))          # une graduation par valeur testée
    ax.grid(alpha=0.3)


def main() -> None:
    # La valeur de gpu-memory-utilization est lue dans le nom du dossier (gpu_0.7 -> 0.7) :
    # c'est pourquoi le balayage utilise `--output_dir gpu_<valeur>`.
    data = charger(DOSSIER, PREFIXE, float)

    # 6 métriques -> grille 2 lignes x 3 colonnes. Si CLES change de taille,
    # adapter la grille (zip s'arrête à la plus courte des deux séquences).
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cle in zip(axes.flat, CLES):
        tracer_metrique(ax, data, cle)

    sauver(fig, SORTIE, "Effet de gpu-memory-utilization "
                        "(point = médiane, barre = min/max sur les runs)")


if __name__ == "__main__":
    main()
