#!/bin/bash
# sbatch_final_models.sh
# ----------------------
# Submits the final models as SLURM jobs (04_final_models.py), one job per model and command.
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_final_models.sh fit <model|all> [options]
#     fits and saves the model, predicts every test and validation row, commits the threshold
#   bash src/02_model_training/sbatch_final_models.sh check <model|all> [options]
#     compares the saved test predictions with those of steps 01-03 (fits nothing)
#   bash src/02_model_training/sbatch_final_models.sh importance <model|all> [options]
#     permutation importance on the validation subset (needs the saved model)
#   bash src/02_model_training/sbatch_final_models.sh all [options]
#     fit of every model, then its check and importance (chained)
#   options:
#     --force   recompute output that already exists
#     --after   JOB_ID: start only after that job has finished successfully
#     --exclude NODES: never run on these nodes (comma-separated), e.g. a node that fails
#               every job at once

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/04_final_models.py"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(extra_trees xgboost random_forest)      # model_training.models in config.toml

# Resources per model and command: "memory (GB) CPUs time (min)", CPUs = memory / 5.5 GB
# rounded up. Set from the refits of step 03 (2026-10-24 to 2026-10-28; peak memory, run time):
# extra trees + SMOTE 79.3 GB, 19.5 min (16 threads); XGBoost + class weights 11.1 GB,
# 3.6 min; random forest + RUS 5.4 GB, 1.9 min. fit adds the uncompressed dump of the model
# (extra trees about 20 GB) and the prediction of the validation rows. importance holds the
# model and two copies of the validation predictors (about 1 GB) and predicts the 8.5M
# validation rows 91 times (extra trees about 1 h, XGBoost about 25 min, random forest about
# 55 min, from the prediction times of step 03).
declare -A RESOURCES=(
    [fit,extra_trees]="88 16 60"         [fit,xgboost]="16 3 30"         [fit,random_forest]="8 2 30"
    [importance,extra_trees]="32 6 120"  [importance,xgboost]="8 2 60"   [importance,random_forest]="12 3 120"
    [check,extra_trees]="4 1 15"         [check,xgboost]="4 1 15"        [check,random_forest]="4 1 15"
)

usage() { sed -n '2,19p' "$0"; exit 1; }

# submit <job name> <memory GB> <CPUs> <time (min)> <extra sbatch options> <commands>; prints the job ID
submit() {
    local name=$1 mem=$2 cpus=$3 minutes=$4 opts=$5 commands=$6 qos=short
    [ "$minutes" -gt 1440 ] && qos=medium
    sbatch --parsable $EXCLUDE $opts <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=standard
#SBATCH --qos=$qos
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

submit_command() {   # <command> <model> <after>; prints the job ID
    local command=$1 model=$2 after=$3 args id mem cpus minutes
    read -r mem cpus minutes <<< "${RESOURCES[$command,$model]}"
    case "$command" in
        fit)        args="fit $model $FORCE" ;;
        check)      args="fit $model --check" ;;
        importance) args="importance $model $FORCE" ;;
    esac
    id=$(submit "final_${command}_${model}" "$mem" "$cpus" "$minutes" "$(dependency "$after")" \
         "python $SCRIPT $args")
    echo "$command $model (${mem} GB, $cpus CPUs, $minutes min): $id" >&2
    [ -n "$id" ] || exit 1
    echo "$id"
}

pick() {   # pick <value|all> <allowed ...>: prints the selection, or nothing if not allowed
    local value=$1; shift
    if [ "$value" = "all" ]; then echo "$@"; return; fi
    for v in "$@"; do [ "$v" = "$value" ] && echo "$v" && return; done
}

ARGS=()
FORCE=""; AFTER=""; EXCLUDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force" ;;
        --after) AFTER="$2"; shift ;;
        --exclude) [ -n "$2" ] || usage; EXCLUDE="--exclude=$2"; shift ;;
        --*) usage ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

COMMAND=${ARGS[0]}
case "$COMMAND" in
    all) [ ${#ARGS[@]} -eq 1 ] || usage ;;
    fit|check|importance) [ ${#ARGS[@]} -eq 2 ] || usage
                          RUN_MODELS=($(pick "${ARGS[1]}" "${MODELS[@]}"))
                          [ ${#RUN_MODELS[@]} -gt 0 ] || usage ;;
    *) usage ;;
esac

# A failed submission stops the chain, so that no job starts without its dependency.
case "$COMMAND" in
    fit|check|importance)
        for model in "${RUN_MODELS[@]}"; do
            submit_command "$COMMAND" "$model" "$AFTER" >/dev/null || exit 1
        done ;;
    all)
        for model in "${MODELS[@]}"; do
            fit_id=$(submit_command fit "$model" "$AFTER") || exit 1
            submit_command check "$model" "$fit_id" >/dev/null || exit 1
            submit_command importance "$model" "$fit_id" >/dev/null || exit 1
        done ;;
esac
