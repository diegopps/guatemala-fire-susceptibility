#!/bin/bash
# sbatch_05_compute_jenks.sh
# --------------------
# Submits SLURM job for computing JENKS breaks
#
# Usage:
#   bash sbatch_05_compute_jenks.sh

SCRIPT="/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/src/02_model_training/05_compute_jenks.py"
PYTHON="python"

echo "Submitting job for '05_compute_jenks.py'"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=jenks_compute
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=logs/jenks_%j.out
#SBATCH --error=logs/jenks_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

ml miniforge
source activate forestFireGT

$PYTHON $SCRIPT
EOF
