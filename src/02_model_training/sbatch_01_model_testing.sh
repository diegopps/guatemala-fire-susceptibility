#!/bin/bash
# submit_model_jobs.sh
# --------------------
# Submits one SLURM job per model, each running model_testing.py
# with a different model name as argument.
#
# Usage:
#   bash submit_model_jobs.sh

MODELS=(
    "logistic_regression"
    "decision_tree"
    "random_forest"
    "gradient_boosting"
    "xgboost"
    "extra_trees"
    "knn"
    "naive_bayes"
)

SCRIPT="/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/src/02_model_training/01_model_testing.py"
PYTHON="python"

for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for: $MODEL"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=model_${MODEL}
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=logs/${MODEL}_%j.out
#SBATCH --error=logs/${MODEL}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

ml miniforge
source activate forestFireGT

$PYTHON $SCRIPT $MODEL
EOF
done

echo "All jobs submitted."
