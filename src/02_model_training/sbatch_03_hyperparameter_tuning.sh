#!/bin/bash
# submit_tuning_jobs.sh
# ---------------------
# Submits one SLURM job per model for hyperparameter tuning, each running
# 03_hyperparameter_tuning.py with a model name as argument.
#
# Usage:
#   bash submit_tuning_jobs.sh

MODELS=(
    "random_forest"
    "xgboost"
    "extra_trees"
)

SCRIPT="/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/src/02_model_training/03_hyperparameter_tuning.py"
PYTHON="python"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do
    JOB_NAME="tuning_${MODEL}"
    echo "Submitting job for: $MODEL"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --account=gvca
#SBATCH --output=${LOG_DIR}/${MODEL}_%j.out
#SBATCH --error=${LOG_DIR}/${MODEL}_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
ml miniforge
source activate forestFireGT
$PYTHON $SCRIPT $MODEL
EOF

done

echo "All jobs submitted. Total: ${#MODELS[@]} jobs."
