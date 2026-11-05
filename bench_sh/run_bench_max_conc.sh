for c in 8 16 24 48 126 256 512 ; do 
  uv run python bench/main.py --gpu-mem 0.9 --output_dir conc_$c --expe-name "study_concur" --max-conc $c || true
done