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
import sys

import matplotlib.pyplot as plt

from common import RESULTS, axe_concurrence, charger, lire_slo, sauver

EXPE = sys.argv[1] if len(sys.argv) > 1 else "sweep_concurrency_goodput"
DOSSIER = RESULTS / EXPE
PREFIXE = "conc_"                      # dossiers : conc_1, conc_2, ...
SORTIE = DOSSIER / "conc_sweep.png"


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
    for cle in ["request_throughput", "request_goodput"]:
        if all(cle in s for s in data.values()):
            ax.plot(list(data), [s[cle]["mediane"] for s in data.values()], marker="o", label=cle)
    axe_concurrence(ax, data)
    ax.set_ylabel("req/s")
    ax.set_title("Débit total vs goodput")
    ax.legend()
    ax.grid(alpha=0.3)


def main() -> None:
    data = charger(DOSSIER, PREFIXE)
    slo = lire_slo(DOSSIER)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_latence(axes[0], data, "p99_ttft_ms", slo.get("ttft"))
    tracer_latence(axes[1], data, "p99_tpot_ms", slo.get("tpot"))
    tracer_goodput(axes[2], data)

    sauver(fig, SORTIE, f"Sweep de concurrence : {EXPE} (point = médiane des runs)")


if __name__ == "__main__":
    main()
