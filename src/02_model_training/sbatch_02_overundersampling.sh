#!/bin/bash
# submit_sampling_jobs.sh
# -----------------------
# Submits one SLURM job per model + sampler combination, each running
# 02_overunder_sampling.py with a model name and sampler name as arguments.
#
# Usage:
#   bash submit_sampling_jobs.sh

MODELS=(
    "random_forest"
    "xgboost"
    "extra_trees"
)

SAMPLERS=(
    "ros"
    "smote"
    "svmsmote"
    "adasyn"
    "rus"
    "nearmiss"
    "tomek"
    "ncl"
)

SCRIPT="/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/src/02_model_training/02_overundersampling.py"
PYTHON="python"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do
    for SAMPLER in "${SAMPLERS[@]}"; do
        JOB_NAME="sampling_${MODEL}_${SAMPLER}"
        echo "Submitting job for: $MODEL + $SAMPLER"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=${LOG_DIR}/${MODEL}_${SAMPLER}_%j.out
#SBATCH --error=${LOG_DIR}/${MODEL}_${SAMPLER}_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

ml miniforge
source activate forestFireGT
$PYTHON $SCRIPT $MODEL $SAMPLER
EOF
    done
done

echo "All jobs submitted. Total: $((${#MODELS[@]} * ${#SAMPLERS[@]})) jobs."