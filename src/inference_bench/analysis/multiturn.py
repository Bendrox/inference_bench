"""
multiturn.py : bench sessions multi-tours (contexte qui grandit), cf. experiments/sweep_multiturn.sh.

Entrée : `results/<os>/<expe>/<cache>_s<sessions>_conc_<N>/run*.json` (détail par tour)
         et `synthese.json` (agrégats des N runs)

Sortie : `results/<os>/<expe>/multiturn.png` (ou --output), 4 graphiques :
  1. TTFT médian par tour : plat avec cache, linéaire sans ; une remontée = éviction ;
  2. part du prompt servie par le cache, par tour (cached_tokens / prompt_tokens) ;
  3. TTFT médian (tous tours) vs nombre de sessions ;
  4. pic d'usage du KV cache vs nombre de sessions (Prometheus, clés srv_*).
Les tours sont regroupés sur tous les runs d'un dossier.

Usage :  uv run python -m inference_bench.analysis.multiturn [nom_expe] [--output fichier.png]
         (défaut : "sweep_multiturn")
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .common import RESULTS, errorbar, sauver

MOTIF = re.compile(r"(?P<cache>.+)_s(?P<s>\d+)_conc_(?P<c>\d+)$")   # ex. fp8-cache_s16_conc_16


def charger(expe: Path) -> dict[tuple[str, int, int], dict]:
    """{(cache, sessions, conc): {"tours": [requêtes...], "synthese": {...}}}, trié."""
    data = {}
    for d in expe.iterdir():
        m = MOTIF.match(d.name) if d.is_dir() else None
        if not m or not (d / "synthese.json").exists():
            continue
        tours = []
        for f in sorted(d.glob("run*.json")):
            tours += [r for r in json.loads(f.read_text(encoding="utf-8")).get("requests", [])
                      if "error" not in r and r.get("ttft") is not None]
        data[(m["cache"], int(m["s"]), int(m["c"]))] = {
            "tours": tours, "synthese": json.loads((d / "synthese.json").read_text(encoding="utf-8"))}
    if not data:
        sys.exit(f"Aucun <cache>_s<S>_conc_<N>/synthese.json dans {expe}")
    return dict(sorted(data.items()))


def par_tour(tours: list[dict], valeur) -> tuple[list[int], list[float]]:
    """Médiane de valeur(r) pour chaque indice de tour (None ignorés)."""
    groupes = defaultdict(list)
    for r in tours:
        v = valeur(r)
        if v is not None:
            groupes[r["turn"]].append(v)
    xs = sorted(groupes)
    return xs, [statistics.median(groupes[x]) for x in xs]


def ratio_cache(r: dict) -> float | None:
    if r.get("cached_tokens") is None or not r.get("prompt_tokens"):
        return None
    return r["cached_tokens"] / r["prompt_tokens"]


def style(cache: str) -> str:
    return "--" if "nocache" in cache else "-"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bench multi-tours : TTFT et hit par tour, KV cache")
    parser.add_argument("expe", nargs="?", default="sweep_multiturn")
    parser.add_argument("--output", type=Path, default=None, help="défaut : results/<os>/<expe>/multiturn.png")
    args = parser.parse_args(argv)

    expe = RESULTS / args.expe
    data = charger(expe)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    a_ttft, a_hit, a_sess, a_kv = axes.flat

    for (cache, s, c), d in data.items():
        label = f"{cache} s={s} c={c}"
        xs, ys = par_tour(d["tours"], lambda r: r["ttft"] * 1000)
        a_ttft.plot(xs, ys, style(cache), marker=".", label=label)
        if "nocache" not in cache:
            xs, ys = par_tour(d["tours"], ratio_cache)
            if xs:
                a_hit.plot(xs, ys, marker=".", label=label)
    a_ttft.set(title="TTFT médian par tour", xlabel="tour", ylabel="ms", yscale="log")
    a_hit.set(title="Part du prompt servie par le cache", xlabel="tour",
              ylabel="cached_tokens / prompt_tokens", ylim=(0, 1.05))
    if not a_hit.lines:
        a_hit.text(0.5, 0.5, "pas de cached_tokens\n(serveur sans --enable-prompt-tokens-details ?)",
                   ha="center", va="center", transform=a_hit.transAxes, color="gray")

    # Graphes 3 et 4 : une courbe par (cache, conc fixe ou "conc = sessions")
    courbes = defaultdict(dict)
    for (cache, s, c), d in data.items():
        courbes[(cache, "=s" if c == s else str(c))][s] = d["synthese"]
    for (cache, c), pts in courbes.items():
        for ax, cle in ((a_sess, "median_ttft_ms"), (a_kv, "srv_kv_cache_usage_max")):
            points = {s: syn[cle] for s, syn in sorted(pts.items()) if cle in syn}
            if points:
                errorbar(ax, points, label=f"{cache} conc{c}", capsize=3, linestyle=style(cache))
    a_sess.set(title="TTFT médian (tous tours)", ylabel="ms", yscale="log")
    a_kv.set(title="Pic d'usage du KV cache", ylabel="fraction (0 à 1)")
    if not a_kv.lines:
        a_kv.text(0.5, 0.5, "pas de srv_kv_cache_usage_max\n(runs lancés sans Prometheus ?)",
                  ha="center", va="center", transform=a_kv.transAxes, color="gray")
    sessions = sorted({s for _, s, _ in data})
    for ax in (a_sess, a_kv):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sessions, [str(s) for s in sessions])
        ax.set_xlabel("sessions")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7)
    sauver(fig, args.output or expe / "multiturn.png", f"Sessions multi-tours : {args.expe}")


if __name__ == "__main__":
    main()
