import json, glob, statistics, sys

fichiers = sorted(glob.glob(sys.argv[1] + "/run*.json"))
cles = ["output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"]
txt=[]
for cle in cles:
    vals = [json.load(open(f))[cle] for f in fichiers]
    res=f"{cle:25} médiane={statistics.median(vals):10.2f}  min={min(vals):10.2f}  max={max(vals):10.2f}"
    print(res)
    txt.append(res)

with open("results/sharegpt/synthèse.json", "w", encoding="utf-8") as f:
    json.dump(txt, f, ensure_ascii=False, indent=4)


# uv run python summarize.py "$DOSSIER"
# Comment l'interpréter : si l'écart entre min et max du débit dépasse quelques pour cent, la mesure est bruitée. 
# Dans ce cas, augmente num-prompts (par exemple 3000 dans client/) avant de conclure quoi que ce soit entre deux configs. 
# Le P99 est naturellement plus instable que la médiane.