for g in 0.90; do
  uv run python bench/main.py --gpu-mem $g --output_dir gpu_$g --expe-name "study_gpumem" || true
done