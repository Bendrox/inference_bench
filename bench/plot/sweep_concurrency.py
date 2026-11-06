"""
sweep_concurrency.py : compromis débit / latence et goodput selon la concurrence.

Entrée : `results/<expe>/conc_<N>/synthese.json` (+ `run.yaml` pour lire le SLO)

Sortie : `results/<expe>/conc_sweep.png`, 3 graphiques :
  1. débit de sortie (x) vs TTFT p99 (y), un point par concurrence ;
  2. débit de sortie (x) vs TPOT p99 (y) ;
  3. débit total vs goodput (req/s) selon la concurrence.
     L'écart entre les deux = requêtes hors SLO ; le max du goodput = meilleure concurrence.
Les seuils SLO (lus dans run.yaml, option --goodput) sont tracés en pointillés.
Le graphique 3 n'affiche le goodput que si les runs ont été lancés avec --goodput.

Usage :  uv run python bench/plot/sweep_concurrency.py [nom_expe]   (défaut : "sweep_concurrency_goodput")
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPE = sys.argv[1] if len(sys.argv) > 1 else "sweep_concurrency_goodput"
RESULTS = PROJECT_ROOT / "results" / EXPE
PREFIXE = "conc_"                      # dossiers : conc_1, conc_2, ...
SORTIE = RESULTS / "conc_sweep.png"


def charger() -> dict[int, dict]:
    """Retourne {concurrence: synthese}, la concurrence étant lue dans le nom du dossier."""
    data = {}
    for f in RESULTS.glob(f"{PREFIXE}*/synthese.json"):
        conc = int(f.parent.name.removeprefix(PREFIXE))
        data[conc] = json.loads(f.read_text(encoding="utf-8"))
    return dict(sorted(data.items()))


def lire_slo() -> dict[str, float]:
    """Lit le SLO dans le run.yaml d'un des dossiers : ["ttft:1000", "tpot:50"] -> {"ttft": 1000, "tpot": 50}."""
    run_yaml = next(RESULTS.glob(f"{PREFIXE}*/run.yaml"), None)
    if run_yaml is None:
        return {}
    goodput = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))["client"].get("goodput") or []
    return {cle: float(val) for cle, val in (g.split(":") for g in goodput)}


def tracer_latence(ax, data: dict[int, dict], cle: str, seuil: float | None) -> None:
    """Débit de sortie (x) vs latence `cle` (y), chaque point annoté avec sa concurrence."""
    xs = [s["output_throughput"]["mediane"] for s in data.values()]
    ys = [s[cle]["mediane"] for s in data.values()]
    ax.plot(xs, ys, marker="o")
    for conc, x, y in zip(data, xs, ys):
        ax.annotate(str(conc), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    if seuil is not None:
        ax.axhline(seuil, color="red", linestyle="--", label=f"SLO {seuil:.0f} ms")
        ax.legend()
    ax.set_yscale("log")               # la latence explose à saturation : échelle log lisible
    ax.set_xlabel("output_throughput (tok/s)")
    ax.set_ylabel(f"{cle} (ms)")
    ax.set_title(f"Débit vs {cle}")
    ax.grid(alpha=0.3)


def tracer_goodput(ax, data: dict[int, dict]) -> None:
    """Débit total et goodput (req/s) en fonction de la concurrence."""
    concs = list(data)
    for cle in ["request_throughput", "request_goodput"]:
        if all(cle in s for s in data.values()):
            ax.plot(concs, [s[cle]["mediane"] for s in data.values()], marker="o", label=cle)
    ax.set_xscale("log", base=2)
    ax.set_xticks(concs, [str(c) for c in concs])
    ax.set_xlabel("max-concurrency")
    ax.set_ylabel("req/s")
    ax.set_title("Débit total vs goodput")
    ax.legend()
    ax.grid(alpha=0.3)


def main() -> None:
    data = charger()
    if not data:
        sys.exit(f"Aucun synthese.json trouvé dans {RESULTS}/{PREFIXE}*/")
    slo = lire_slo()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_latence(axes[0], data, "p99_ttft_ms", slo.get("ttft"))
    tracer_latence(axes[1], data, "p99_tpot_ms", slo.get("tpot"))
    tracer_goodput(axes[2], data)

    fig.suptitle(f"Sweep de concurrence : {EXPE} (point = médiane des runs)")
    fig.tight_layout()
    fig.savefig(SORTIE, dpi=150)
    print(f"Graphique enregistré : {SORTIE}")


if __name__ == "__main__":
    main()
