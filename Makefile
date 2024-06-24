slurm-bash:
	srun --pty --gres=gpu:ampere --container-image=mam10eks/reneuir-tinybert:0.0.1 --mem=60G --cpus-per-task=4 bash
