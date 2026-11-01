import json, glob, statistics, sys
from pathlib import Path
fichiers = sorted(glob.glob(sys.argv[1] + "/run*.json"))
cles = ["output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"]
txt=[]

fichiers = sorted(glob.glob(sys.argv[1] + "/run*.json"))
if not fichiers:
    sys.exit(f"Aucun run*.json trouvé dans : {sys.argv[1]}")

synthese = {}
for cle in cles:
    vals = [json.load(open(f))[cle] for f in fichiers]
    med = statistics.median(vals)
    synthese[cle] = {"mediane": med, "min": min(vals), "max": max(vals),
                     "ecart_pct": 100 * (max(vals) - min(vals)) / med}

with open(Path(sys.argv[1]) / "synthese.json", "w", encoding="utf-8") as f:
    json.dump(synthese, f, ensure_ascii=False, indent=4)



# uv run python summarize.py "$DOSSIER"
# Comment l'interpréter : si l'écart entre min et max du débit dépasse quelques pour cent, la mesure est bruitée. 
# Dans ce cas, augmente num-prompts (par exemple 3000 dans client/) avant de conclure quoi que ce soit entre deux configs. 
# Le P99 est naturellement plus instable que la médiane.