for g in 0.55 0.6 0.7 0.8 0.95; do
  uv run python bench/main.py --gpu-mem $g --output_dir gpu_$g --expe-name "study_gpumem" || true
done