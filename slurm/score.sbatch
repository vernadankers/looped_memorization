#!/bin/bash
#SBATCH --job-name=ouro-score
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_logs/score_%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ouro-vllm

for i in 0 1 2 3; do
    FULL_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models'][$i])")
    MODEL="${FULL_MODEL##*/}"
    python -m score.score_extraction --model "$MODEL"
done
