#!/bin/bash
# =============================================================================
# submit_future_classification_jobs.sh
# -----------------------------------------------------------------------------
# Submits one SLURM job per model for 01_future_classification.py.
# Each job runs all scenarios and years for a single model independently.
#
# Usage:
#     bash submit_future_classification_jobs.sh
#
# Run from the project root:
#     /home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT
# =============================================================================

PROJECT_DIR="/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
SCRIPT="$PROJECT_DIR/src/03_future_classification/01_future_classification.py"
LOG_DIR="$PROJECT_DIR/logs"
CONDA_ENV="forestFireGT"

mkdir -p "$LOG_DIR"

MODELS=(
    "random_forest"
    "xgboost"
    "extra_trees"
)

for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for model: $MODEL"

    sbatch \
        --job-name="classify_${MODEL}" \
        --partition=standard \
        --qos=short \
        --account=gvca \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=08:00:00 \
        --output="${LOG_DIR}/classify_${MODEL}_%j.log" \
        --wrap="conda run -n $CONDA_ENV python $SCRIPT $MODEL"

    echo "  Submitted: classify_${MODEL}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"