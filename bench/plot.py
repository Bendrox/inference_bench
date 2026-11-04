"""
plot.py : trace l'effet de `gpu-memory-utilization` sur les métriques du benchmark.

Chaque lancement produit (voir main.py et summarize.py) :
    results/gpu_<valeur>/<nom_de_la_config>/synthese.json

Ce script lit tous ces `synthese.json`, puis trace une courbe par métrique en
fonction de la valeur de gpu-memory-utilization.

Entrée: `synthese.json` 

Sortie: 
`results/gpu_sweep.png` : grille 2x3, un graphique par métrique de `summarize.CLES`.
  - le point est la médiane des runs ;
  - la barre d'erreur va du min au max des runs (= le bruit de mesure).
Si l'écart entre deux points est du même ordre que la barre d'erreur, la
différence n'est pas significative.

Usage :  uv run python bench/plot.py

Limites:
- Un seul dossier de config par `gpu_<valeur>` est attendu. S'il y en a plusieurs,
  le dernier (ordre alphabétique) écrase les autres.
- Une valeur pour laquelle vLLM a planté n'a pas de `synthese.json` : elle
  n'apparaît tout simplement pas sur le graphique.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from summarize import CLES   # liste des métriques, partagée avec summarize.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
PREFIXE = "gpu_"                       # préfixe des dossiers : gpu_0.7, gpu_0.8, ...
SORTIE = RESULTS / "gpu_sweep.png"


def charger() -> dict[float, dict]:
    """Charge les synthèses de tous les dossiers `results/gpu_<valeur>/*/`.

    La valeur de gpu-memory-utilization est déduite du nom du dossier parent
    (`gpu_0.7` -> 0.7) : c'est pourquoi le balayage utilise `--output_dir gpu_<valeur>`.

    Retourne {valeur_gpu: synthese}, où `synthese` est le contenu de synthese.json.
    """
    data = {}
    for f in sorted(RESULTS.glob(f"{PREFIXE}*/*/synthese.json")):
        dossier_gpu = f.parent.parent.name                       # ex. "gpu_0.7"
        gpu = float(dossier_gpu.removeprefix(PREFIXE))           # ex. 0.7
        data[gpu] = json.loads(f.read_text(encoding="utf-8"))
    return data


def tracer_metrique(ax, data: dict[float, dict], cle: str) -> None:
    """Trace une métrique sur l'axe `ax` : médiane + barre d'erreur min/max.

    `yerr` de matplotlib attend les longueurs *relatives* à la médiane
    (vers le bas, vers le haut), pas les valeurs absolues min et max.
    """
    xs = sorted(data)
    med = [data[x][cle]["mediane"] for x in xs]
    vers_bas = [data[x][cle]["mediane"] - data[x][cle]["min"] for x in xs]
    vers_haut = [data[x][cle]["max"] - data[x][cle]["mediane"] for x in xs]

    ax.errorbar(xs, med, yerr=[vers_bas, vers_haut], marker="o", capsize=4)
    ax.set_title(cle)
    ax.set_xlabel("gpu-memory-utilization")
    ax.set_xticks(xs)                  # une graduation par valeur testée
    ax.grid(alpha=0.3)


def main() -> None:
    data = charger()
    if not data:
        sys.exit(f"Aucun synthese.json trouvé dans {RESULTS}/{PREFIXE}*/*/")

    # 6 métriques -> grille 2 lignes x 3 colonnes. Si CLES change de taille,
    # adapter la grille (zip s'arrête à la plus courte des deux séquences).
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cle in zip(axes.flat, CLES):
        tracer_metrique(ax, data, cle)

    fig.suptitle("Effet de gpu-memory-utilization "
                 "(point = médiane, barre = min/max sur les runs)")
    fig.tight_layout()
    fig.savefig(SORTIE, dpi=150)
    print(f"Graphique enregistré : {SORTIE}")


if __name__ == "__main__":
    main()
