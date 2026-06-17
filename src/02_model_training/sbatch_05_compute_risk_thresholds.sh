#!/bin/bash
# sbatch_05_compute_risk_thresholds.sh
# ------------------------------------
# Submits SLURM job for computing the frozen prediction-rate risk thresholds.
#
# Usage:
#   bash sbatch_05_compute_risk_thresholds.sh

SCRIPT="/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/src/02_model_training/05_compute_risk_thresholds.py"
PYTHON="python"

echo "Submitting job for '05_compute_risk_thresholds.py'"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=risk_thresholds
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=logs/risk_thresholds_%j.out
#SBATCH --error=logs/risk_thresholds_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

ml miniforge
source activate forestFireGT

$PYTHON $SCRIPT
EOF
