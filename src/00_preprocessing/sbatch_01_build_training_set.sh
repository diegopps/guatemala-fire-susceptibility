#!/bin/bash
# sbatch_01_build_training_set.sh
# -------------------------------
# Submits 01_build_training_set.py as a SLURM job.
#
# Requires the monthly per-GCM climate (sbatch_00_aggregate_monthly_gcm.sh). Pass that
# job's ID to start only after it has finished successfully.
#
# Usage (from any directory):
#   bash src/00_preprocessing/sbatch_01_build_training_set.sh [after_job_id]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

DEPENDENCY=""
[ -n "$1" ] && DEPENDENCY="--dependency=afterok:$1"

sbatch $DEPENDENCY <<EOF
#!/bin/bash
#SBATCH --job-name=build_training_set
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/build_training_set_%j.out
#SBATCH --error=$LOG_DIR/build_training_set_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

ml miniforge
source activate forestFireGT

python $SCRIPT_DIR/01_build_training_set.py
EOF
