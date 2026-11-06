for d in  ; do
  uv run python bench/main.py \
  --gpu-mem 0.9 \
  --output_dir data_$g \ 
  --expe-name "study_gpumem" || true
done