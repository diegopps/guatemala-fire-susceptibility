#!/bin/bash
# sbatch_extrapolation.sh
# -----------------------
# Submits the extrapolation tables (02_extrapolation.py: the climate predictors of the projected
# periods against the training envelope) as a SLURM job.
#
# Usage (from any directory; submit from a shell without the environment activated):
#   bash src/03_future_classification/sbatch_extrapolation.sh [options]
#   options:
#     --trial   first GCM only, output to <results folder>_trial
#     --after   JOB_ID: start only after that job has finished successfully (several:
#               ID1:ID2:...)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Resources: "memory (GB) CPUs time (min)", CPUs = memory / 5.5 GB rounded up. 150 GCM-years per
# GCM, 5 NetCDF files each. Measured on the trial (2026-11-14, job 3253739: 1 GCM, 2 processes):
# 1.0 min including the envelope of the training set, 3.5 GB peak.
declare -A RESOURCES=(
    [trial]="8 2 15"
    [full]="8 2 30"
)

usage() { sed -n '2,12p' "$0"; exit 1; }

TRIAL=""; AFTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --trial) TRIAL="--trial" ;;
        --after) [ -n "$2" ] || usage; AFTER="--dependency=afterok:$2"; shift ;;
        *) usage ;;
    esac
    shift
done

RUN=${TRIAL:+trial}; RUN=${RUN:-full}
read -r MEM CPUS MINUTES <<< "${RESOURCES[$RUN]}"
NAME="future_extrapolation${TRIAL:+_trial}"

ID=$(sbatch --parsable $AFTER <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/${NAME}_%j.out
#SBATCH --error=$LOG_DIR/${NAME}_%j.err
#SBATCH --time=$MINUTES
#SBATCH --mem=${MEM}G
#SBATCH --cpus-per-task=$CPUS

ml miniforge
source activate forestFireGT

# Numerical libraries use only the allocated CPUs; the script runs one process per CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python $SCRIPT_DIR/02_extrapolation.py $TRIAL
EOF
)
echo "extrapolation $RUN (${MEM} GB, $CPUS CPUs, $MINUTES min): $ID" >&2
[ -n "$ID" ] || exit 1
