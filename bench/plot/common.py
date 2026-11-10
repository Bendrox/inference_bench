"""
common.py : fonctions partagées par les scripts de bench/plot/.

Convention des résultats : `results/<expe>/<prefixe><valeur>/synthese.json`
(ex. conc_16, gpu_0.7), la valeur balayée étant lue dans le nom du dossier.

Import depuis un script de bench/plot/ :  from common import RESULTS, charger, ...
(python ajoute le dossier du script lancé à sys.path, aucun hack nécessaire).
"""
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "results"


def charger(expe: Path, prefixe: str, type_val=int) -> dict:
    """Retourne {valeur: synthese}, trié par valeur. Quitte si aucun synthese.json.

    `type_val` convertit la valeur lue dans le nom du dossier : int pour conc_16, float pour gpu_0.7.
    """
    data = {}
    for f in expe.glob(f"{prefixe}*/synthese.json"):
        valeur = type_val(f.parent.name.removeprefix(prefixe))
        data[valeur] = json.loads(f.read_text(encoding="utf-8"))
    if not data:
        sys.exit(f"Aucun synthese.json trouvé dans {expe}/{prefixe}*/")
    return dict(sorted(data.items()))


def lire_slo(expe: Path) -> dict[str, float]:
    """Lit le SLO dans un run.yaml : ["ttft:1000", "tpot:50"] -> {"ttft": 1000, "tpot": 50}."""
    run_yaml = next(expe.glob("*/run.yaml"), None)
    if run_yaml is None:
        return {}
    goodput = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))["client"].get("goodput") or []
    return {cle: float(val) for cle, val in (g.split(":") for g in goodput)}


def errorbar(ax, points: dict, **kwargs) -> None:
    """Trace points = {x: {"mediane", "min", "max"}} : médiane + barre d'erreur min/max.

    `yerr` de matplotlib attend les longueurs *relatives* à la médiane (vers le bas, vers le haut).
    `kwargs` est transmis à ax.errorbar (label, capsize, ...).
    """
    xs = list(points)
    med = [p["mediane"] for p in points.values()]
    bas = [p["mediane"] - p["min"] for p in points.values()]
    haut = [p["max"] - p["mediane"] for p in points.values()]
    ax.errorbar(xs, med, yerr=[bas, haut], marker="o", **kwargs)


def axe_concurrence(ax, concs) -> None:
    """Axe x en log2, une graduation par concurrence testée."""
    concs = sorted(concs)
    ax.set_xscale("log", base=2)
    ax.set_xticks(concs, [str(c) for c in concs])
    ax.set_xlabel("max-concurrency")


def sauver(fig, sortie: Path, titre: str) -> None:
    """Ajoute le titre général, enregistre la figure et affiche le chemin."""
    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    print(f"Graphique enregistré : {sortie}")
