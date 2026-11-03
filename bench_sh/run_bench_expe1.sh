for g in 0.5 0.6 0.7 0.8 0.9; do
  uv run python bench/main.py --gpu-mem $g --output_dir gpu_$g || true
done