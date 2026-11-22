.PHONY: help is-cuda-available-device nvidia-followup get-models get-models-qwen server-log available-benchmarks plots obs-up obs-down obs-check

help:
	@grep -E "^[a-z-]+:" Makefile | cut -d: -f1

is-cuda-available-device:
	uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

nvidia-followup:
	nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1

get-models:
	hf cache ls --sort size:desc | grep '^model/' 

get-models-qwen:
	hf cache ls --sort size:desc | grep '^model/Qwen' 

server-log:
	grep -iE "non-default args|model loading took|KV cache|maximum concurrency|startup complete" server.log

available-benchmarks:
	@ls experiments/*.sh

# Régénère les graphes de l'OS courant (BENCH_OS=wsl pour l'autre) ; une expérience absente est ignorée
plots:
	-uv run python -m inference_bench.analysis.concurrency sweep_concurrency_slo
	-uv run python -m inference_bench.analysis.concurrency sweep_concurrency_noslo
	-uv run python -m inference_bench.analysis.gpu_memory sweep_gpumem
	-uv run python -m inference_bench.analysis.workload
	-uv run python -m inference_bench.analysis.prefix_rep sweep_prefix_rep

obs-up:
	docker compose -f observability/docker-compose.yml up -d

obs-down:
	docker compose -f observability/docker-compose.yml down

obs-check:
	@curl -s localhost:9090/api/v1/targets | python3 -c "import json,sys; [print(t['labels']['job'], t['health']) for t in json.load(sys.stdin)['data']['activeTargets']]"
