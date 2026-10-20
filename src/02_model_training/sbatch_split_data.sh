#!/bin/bash
# sbatch_split_data.sh
# --------------------
# Submits 00_split_data.py (split index and model matrices) as a SLURM job.
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_split_data.sh [--force] [--after JOB_ID]
#   --force  recompute output that already exists
#   --after  start only after job JOB_ID has finished successfully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

FORCE=""; DEPENDENCY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force" ;;
        --after) DEPENDENCY="--dependency=afterok:$2"; shift ;;
        *) sed -n '2,9p' "$0"; exit 1 ;;
    esac
    shift
done

sbatch $DEPENDENCY <<EOF
#!/bin/bash
#SBATCH --job-name=split_data
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/split_data_%j.out
#SBATCH --error=$LOG_DIR/split_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3

ml miniforge
source activate forestFireGT

python $SCRIPT_DIR/00_split_data.py $FORCE
EOF
