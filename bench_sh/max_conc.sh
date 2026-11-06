for c in 1 2 4 8 16 32 64 96 128 192 256 512; do
  uv run python bench/main.py --expe-name sweep_concurrency --output_dir conc_$c \
    --max-conc $c --prompts-per-conc 10 --gpu-mem 0.9 \
    --engine configs/engine/ctx8k-seqs512-batch8k.yaml || echo "ÉCHEC conc=$c"
done
# needed to handle number of prompt per max concurency otherwise its too long and not necessary