"""
workload_profile.py : compare les profils de charge (short/long input × output) selon la concurrence.

Un seul script pour tous les benchs wrkld : chaque expérience `results/wrkld_<profil>/` devient
une courbe, superposée aux autres sur les mêmes graphiques.

Entrée : `results/wrkld_<profil>/conc_<N>/synthese.json` (+ `run.yaml` pour lire le SLO)

Sortie : `results/wrkld_profiles.png`, grille 2x2, x = max-concurrency :
  1. débit de sortie (tok/s)  -> où chaque profil plafonne ;
  2. TTFT p99 (ms)            -> sensible au prefill (prompts longs) ;
  3. TPOT p99 (ms)            -> sensible au decode (sorties longues, gros KV cache) ;
  4. % de requêtes dans le SLO (goodput / débit) -> concurrence max tenable par profil.
Point = médiane des runs, barre = min/max. Seuils SLO en pointillés rouges.

Usage :  uv run python bench/plot/workload_profile.py [profil ...]
         (défaut : tous les dossiers results/wrkld_*, ex. short-short long-long)
"""
import sys

import matplotlib.pyplot as plt

from common import RESULTS, axe_concurrence, charger, errorbar, lire_slo, sauver

PREFIXE_EXPE = "wrkld_"                # dossiers : wrkld_short-short, wrkld_long-long, ...
PREFIXE_CONC = "conc_"                 # sous-dossiers : conc_1, conc_4, ...
SORTIE = RESULTS / "wrkld_profiles.png"


def pct_slo(s: dict) -> dict | None:
    """% de requêtes dans le SLO = goodput / débit.

    Le ratio est fait sur médiane, min et max séparément : rien ne garantit que ratio(min) < ratio(médiane),
    donc la barre d'erreur prend le min et le max des trois ratios.
    """
    if "request_goodput" not in s:
        return None
    ratios = {k: 100 * s["request_goodput"][k] / s["request_throughput"][k] for k in ("mediane", "min", "max")}
    return {"mediane": ratios["mediane"], "min": min(ratios.values()), "max": max(ratios.values())}


def main() -> None:
    profils = sys.argv[1:] or sorted(p.name.removeprefix(PREFIXE_EXPE) for p in RESULTS.glob(f"{PREFIXE_EXPE}*"))
    dossiers = [RESULTS / f"{PREFIXE_EXPE}{p}" for p in profils]
    dossiers = [d for d in dossiers if any(d.glob(f"{PREFIXE_CONC}*/synthese.json"))]   # ignore les profils vides
    if not dossiers:
        sys.exit(f"Aucun synthese.json trouvé dans {RESULTS}/{PREFIXE_EXPE}*/{PREFIXE_CONC}*/")
    donnees = {d.name.removeprefix(PREFIXE_EXPE): charger(d, PREFIXE_CONC) for d in dossiers}
    slo = lire_slo(dossiers[0])

    # (titre, unité, fonction d'extraction, seuil SLO, échelle log en y)
    graphiques = [
        ("output_throughput", "tok/s", lambda s: s["output_throughput"], None, False),
        ("p99_ttft_ms", "ms", lambda s: s["p99_ttft_ms"], slo.get("ttft"), True),
        ("p99_tpot_ms", "ms", lambda s: s["p99_tpot_ms"], slo.get("tpot"), True),
        ("requêtes dans le SLO", "%", pct_slo, None, False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    toutes_concs = {c for d in donnees.values() for c in d}
    for ax, (titre, unite, extraire, seuil, log_y) in zip(axes.flat, graphiques):
        for profil, data in donnees.items():
            points = {c: extraire(s) for c, s in data.items()}
            points = {c: p for c, p in points.items() if p is not None}
            if points:
                errorbar(ax, points, capsize=3, label=profil)
        if seuil is not None:
            ax.axhline(seuil, color="red", linestyle="--", label=f"SLO {seuil:.0f} ms")
        if log_y:
            ax.set_yscale("log")       # la latence explose à saturation : échelle log lisible
        axe_concurrence(ax, toutes_concs)
        ax.set_ylabel(f"{titre} ({unite})")
        ax.set_title(titre)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    sauver(fig, SORTIE, "Workload profiling : profils de charge selon la concurrence "
                        "(point = médiane, barre = min/max sur les runs)")


if __name__ == "__main__":
    main()
