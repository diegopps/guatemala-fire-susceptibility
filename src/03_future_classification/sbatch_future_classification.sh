#!/bin/bash
# sbatch_future_classification.sh
# -------------------------------
# Submits the predictions of the final models for every GCM and scenario
# (01_future_classification.py; one array per model, one task per GCM and scenario, read from
# config.toml), their tables (_evaluate.py) and figures (_plot.py) as SLURM jobs.
#
# Usage (from any directory; submit from a shell without the environment activated):
#   bash src/03_future_classification/sbatch_future_classification.sh predict <model|all> [options]
#     fire probability of every cell-month, one array per model
#   bash src/03_future_classification/sbatch_future_classification.sh evaluate [options]
#     the tables of the figures, from all predictions
#   bash src/03_future_classification/sbatch_future_classification.sh plot [options]
#     the figures, from the tables
#   options:
#     --force   recompute output that already exists
#     --after   JOB_ID: start only after that job has finished successfully (several:
#               ID1:ID2:...)
#     --exclude NODES: never run on these nodes (comma-separated), e.g. a node that fails
#               every job at once
#     --task I  submit only the array task I

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(extra_trees xgboost random_forest)      # model_training.models in config.toml

# One task per (GCM, scenario) pair
SCENARIOS=$(python3 -c "import tomllib; c = tomllib.load(open('$PROJECT_DIR/config.toml','rb'))['scenarios']; print(' '.join([c['historical']] + c['ssp']))")
GCMS=$(python3 -c "import tomllib; print(' '.join(tomllib.load(open('$PROJECT_DIR/config.toml','rb'))['gcms']['models']))")
TASKS=()
for g in $GCMS; do for s in $SCENARIOS; do TASKS+=("$g:$s"); done; done
N=${#TASKS[@]}

# Resources: "memory (GB) CPUs time (min) tasks at once", CPUs = memory / 5.5 GB rounded up.
# A task has 85M (historical) or 136M rows (SSPs) and reads 270 or 430 NetCDF files. Measured on
# one historical task (2026-10-30, jobs 2791770-2791772; the SSPs take 1.6 times as long and 200
# MB more): extra trees 3.1 min on 9 CPUs, 37.2 GB peak (mostly loading the model); XGBoost 2.7
# min on 2 CPUs, 0.9 GB; random forest 4.8 min on 3 CPUs, 1.3 GB. Reading: 0.4 min.
declare -A RESOURCES=(
    [extra_trees]="42 8 20 3"   [xgboost]="3 2 15 3"   [random_forest]="3 3 20 3"
)
# Evaluate and plot: "memory (GB) CPUs time (min)". Measured (2026-10-30, jobs 2792338-2792339):
# evaluate 1.9 min, 1.0 GB in the process (4.0 GB with the memory-mapped predictions, which the
# system can release); plot 1.2 min, 0.9 GB.
declare -A SINGLE_RESOURCES=(
    [evaluate]="4 1 15"
    [plot]="2 1 10"
)
declare -A SCRIPTS=(
    [evaluate]="01_future_classification_evaluate.py"
    [plot]="01_future_classification_plot.py"
)

usage() { sed -n '2,21p' "$0"; exit 1; }

# submit <job name> <memory GB> <CPUs> <time (min)> <extra sbatch options> <commands>; prints the job ID
submit() {
    local name=$1 mem=$2 cpus=$3 minutes=$4 opts=$5 commands=$6 log=%j
    case "$opts" in *--array*) log=%A_%a ;; esac
    sbatch --parsable $EXCLUDE $opts <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/${name}_${log}.out
#SBATCH --error=$LOG_DIR/${name}_${log}.err
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

array() {   # array <tasks at once>: the tasks to submit
    if [ -n "$TASK" ]; then echo "--array=$TASK"; else echo "--array=0-$((N - 1))%$1"; fi
}

submit_predict() {   # <model> <after>
    local model=$1 after=$2 id mem cpus minutes at_once
    read -r mem cpus minutes at_once <<< "${RESOURCES[$model]}"
    id=$(submit "future_${model}" "$mem" "$cpus" "$minutes" "$(array "$at_once") $(dependency "$after")" \
         "TASKS=(${TASKS[*]})
TASK=\${TASKS[\$SLURM_ARRAY_TASK_ID]}
python $SCRIPT_DIR/01_future_classification.py $model \${TASK%%:*} \${TASK##*:} $FORCE")
    echo "predict $model (${mem} GB, $cpus CPUs, $minutes min, ${TASK:-$N tasks, $at_once at once}): $id" >&2
    [ -n "$id" ] || exit 1
}

submit_single() {   # <command> <after>
    local command=$1 after=$2 id mem cpus minutes
    read -r mem cpus minutes <<< "${SINGLE_RESOURCES[$command]}"
    id=$(submit "future_${command}" "$mem" "$cpus" "$minutes" "$(dependency "$after")" \
         "python $SCRIPT_DIR/${SCRIPTS[$command]}")
    echo "$command (${mem} GB, $cpus CPUs, $minutes min): $id" >&2
    [ -n "$id" ] || exit 1
}

pick() {   # pick <value|all> <allowed ...>: prints the selection, or nothing if not allowed
    local value=$1; shift
    if [ "$value" = "all" ]; then echo "$@"; return; fi
    for v in "$@"; do [ "$v" = "$value" ] && echo "$v" && return; done
}

ARGS=()
FORCE=""; AFTER=""; EXCLUDE=""; TASK=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force" ;;
        --after) AFTER="$2"; shift ;;
        --exclude) [ -n "$2" ] || usage; EXCLUDE="--exclude=$2"; shift ;;
        --task) [ -n "$2" ] || usage; TASK="$2"; shift ;;
        --*) usage ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

case "${ARGS[0]}" in
    predict)
        [ ${#ARGS[@]} -eq 2 ] || usage
        RUN_MODELS=($(pick "${ARGS[1]}" "${MODELS[@]}"))
        [ ${#RUN_MODELS[@]} -gt 0 ] || usage
        for model in "${RUN_MODELS[@]}"; do
            submit_predict "$model" "$AFTER" || exit 1
        done ;;
    evaluate|plot)
        [ ${#ARGS[@]} -eq 1 ] || usage
        submit_single "${ARGS[0]}" "$AFTER" || exit 1 ;;
    *) usage ;;
esac
