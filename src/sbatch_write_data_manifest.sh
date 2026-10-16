#!/bin/bash
# sbatch_write_data_manifest.sh
# -----------------------------
# Submits write_data_manifest.py as a SLURM job. Pass a job ID to start only after
# that job (e.g. a data-writing job) has finished successfully.
#
# Usage:
#   bash src/sbatch_write_data_manifest.sh write|verify [after_job_id]

if [ "$1" != "write" ] && [ "$1" != "verify" ]; then
    sed -n '2,8p' "$0"; exit 1
fi
MODE=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

DEPENDENCY=""
[ -n "$2" ] && DEPENDENCY="--dependency=afterok:$2"

sbatch $DEPENDENCY <<EOF
#!/bin/bash
#SBATCH --job-name=data_manifest_${MODE}
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/data_manifest_${MODE}_%j.out
#SBATCH --error=$LOG_DIR/data_manifest_${MODE}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

ml miniforge
source activate forestFireGT

python $SCRIPT_DIR/write_data_manifest.py $MODE
EOF
