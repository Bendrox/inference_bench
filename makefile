.IPHONY : main.py

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