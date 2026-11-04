for g in 0.55 0.95; do
  uv run python bench/main.py --gpu-mem $g --output_dir gpu_$g || true
done