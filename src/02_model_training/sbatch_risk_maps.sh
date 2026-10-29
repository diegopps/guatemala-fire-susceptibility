#!/bin/bash
# sbatch_risk_maps.sh
# -------------------
# Submits the fire-susceptibility classes of the final models as SLURM jobs
# (05_risk_maps.py, then 05_risk_maps_plot.py).
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_risk_maps.sh compute [options]
#     susceptibility per cell, frozen class breaks, classes and model agreement (tables)
#   bash src/02_model_training/sbatch_risk_maps.sh plot [options]
#     the maps, from the tables of compute
#   bash src/02_model_training/sbatch_risk_maps.sh all [options]
#     compute, then plot (chained)
#   options:
#     --after   JOB_ID: start only after that job has finished successfully (several:
#               ID1:ID2:...), e.g. the predict jobs of sbatch_final_models.sh
#     --exclude NODES: never run on these nodes (comma-separated), e.g. a node that fails
#               every job at once

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Resources: "memory (GB) CPUs time (min)". compute reads the predictions of one model at a
# time (85M rows, float32) with the cell and month of each row; a few GB at most.
declare -A RESOURCES=(
    [compute]="8 2 30"
    [plot]="8 2 30"
)
declare -A SCRIPTS=(
    [compute]="05_risk_maps.py"
    [plot]="05_risk_maps_plot.py"
)

usage() { sed -n '2,19p' "$0"; exit 1; }

# submit <job name> <memory GB> <CPUs> <time (min)> <extra sbatch options> <commands>; prints the job ID
submit() {
    local name=$1 mem=$2 cpus=$3 minutes=$4 opts=$5 commands=$6
    sbatch --parsable $EXCLUDE $opts <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/${name}_%j.out
#SBATCH --error=$LOG_DIR/${name}_%j.err
#SBATCH --time=$minutes
#SBATCH --mem=${mem}G
#SBATCH --cpus-per-task=$cpus

ml miniforge
source activate forestFireGT

# Numerical libraries use only the allocated CPUs.
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK

$commands
EOF
}

dependency() { [ -n "$1" ] && echo "--dependency=afterok:$1"; }

submit_command() {   # <command> <after>; prints the job ID
    local command=$1 after=$2 id mem cpus minutes
    read -r mem cpus minutes <<< "${RESOURCES[$command]}"
    id=$(submit "risk_maps_${command}" "$mem" "$cpus" "$minutes" "$(dependency "$after")" \
         "python $SCRIPT_DIR/${SCRIPTS[$command]}")
    echo "$command (${mem} GB, $cpus CPUs, $minutes min): $id" >&2
    [ -n "$id" ] || exit 1
    echo "$id"
}

ARGS=()
AFTER=""; EXCLUDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --after) AFTER="$2"; shift ;;
        --exclude) [ -n "$2" ] || usage; EXCLUDE="--exclude=$2"; shift ;;
        --*) usage ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
[ ${#ARGS[@]} -eq 1 ] || usage

# A failed submission stops the chain, so that no job starts without its dependency.
case "${ARGS[0]}" in
    compute|plot) submit_command "${ARGS[0]}" "$AFTER" >/dev/null || exit 1 ;;
    all) id=$(submit_command compute "$AFTER") || exit 1
         submit_command plot "$id" >/dev/null || exit 1 ;;
    *) usage ;;
esac
