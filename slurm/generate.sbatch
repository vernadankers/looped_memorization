#!/bin/bash
#SBATCH --job-name=ouro-generate
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/generate_%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ouro-vllm

MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models'][$SLURM_ARRAY_TASK_ID])")

mkdir -p results slurm_logs

python -m generate.run_vllm_generation --model "$MODEL"
