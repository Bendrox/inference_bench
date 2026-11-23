"""Métriques serveur d'un run, lues dans Prometheus (vLLM /metrics + exporter GPU).

Prometheus enregistre en continu, sans connaître les runs. Ce module découpe la
fenêtre [t_start, t_end] d'un run et en tire quelques chiffres, écrits dans results/.
Si Prometheus est absent, collect() retourne None : le bench continue sans.
"""
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROM_URL = "http://127.0.0.1:9090"
GRAFANA_URL = "http://127.0.0.1:3000"
SCRAPE_S = 1  # = scrape_interval de observability/prometheus.yml

# {w} = durée du run (ex. "143s"). Requêtes évaluées à t_end : la fenêtre couvre le run.
REQUETES = {
    "prefix_cache_hit_rate": "sum(increase(vllm:prefix_cache_hits_total[{w}]))"
                             " / sum(increase(vllm:prefix_cache_queries_total[{w}]))",
    "kv_cache_usage_max": "max(max_over_time(vllm:kv_cache_usage_perc[{w}]))",
    "requests_running_max": "max(max_over_time(vllm:num_requests_running[{w}]))",
    "requests_waiting_max": "max(max_over_time(vllm:num_requests_waiting[{w}]))",
    "preemptions": "sum(increase(vllm:num_preemptions_total[{w}]))",
    "gpu_power_avg_w": "avg(avg_over_time(nvidia_smi_power_draw_watts[{w}]))",
    "gpu_mem_used_max_gb": "max(max_over_time(nvidia_smi_memory_used_bytes[{w}])) / 1e9",
}


def prometheus_ok() -> bool:
    try:
        urllib.request.urlopen(f"{PROM_URL}/-/ready", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def query(expr: str, at: float) -> float | None:
    """Valeur scalaire d'une requête PromQL à l'instant `at`, None si vide ou NaN."""
    url = f"{PROM_URL}/api/v1/query?" + urllib.parse.urlencode({"query": expr, "time": at})
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            resultat = json.load(r)["data"]["result"]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None
    if not resultat:
        return None
    val = float(resultat[0]["value"][1])
    return None if math.isnan(val) or math.isinf(val) else val


def collect(t_start: float, t_end: float) -> dict | None:
    if not prometheus_ok():
        print("Prometheus absent : métriques serveur non collectées")
        return None
    time.sleep(3 * SCRAPE_S)  # laisse Prometheus enregistrer les derniers scrapes
    duree = t_end - t_start
    w = f"{max(1, round(duree))}s"
    out = {"t_start": t_start, "t_end": t_end, "duration_s": duree}
    for cle, expr in REQUETES.items():
        out[cle] = query(expr.format(w=w), t_end)
    if out["gpu_power_avg_w"] is not None:
        out["energy_j"] = out["gpu_power_avg_w"] * duree   # puissance moyenne × durée
    return out


def annotate(t_start: float, t_end: float, texte: str, tags: list[str]) -> None:
    """Annotation Grafana sur la durée du run (zone visible sur tous les panneaux)."""
    corps = json.dumps({"time": int(t_start * 1000), "timeEnd": int(t_end * 1000),
                        "tags": ["bench", *tags],   # "bench" : tag filtré par le dashboard vllm.json
                        "text": texte}).encode()
    req = urllib.request.Request(f"{GRAFANA_URL}/api/annotations", data=corps,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=2)
    except (urllib.error.URLError, OSError):
        pass  # Grafana absent : pas grave


def enregistrer(dossier: Path, i: int, t_start: float, t_end: float, tags: list[str]) -> None:
    """Métriques serveur du run i -> <dossier>/server_metrics<i>.json, plus une annotation Grafana.
    Ne fait rien si Prometheus est absent."""
    srv = collect(t_start, t_end)
    if srv is None:
        return
    (dossier / f"server_metrics{i}.json").write_text(json.dumps(srv, indent=4), encoding="utf-8")
    annotate(t_start, t_end, f"{'/'.join(tags)} run {i}", tags)
