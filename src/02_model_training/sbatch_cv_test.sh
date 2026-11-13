#!/bin/bash
# sbatch_cv_test.sh
# -----------------
# Submits the cross-validation test of the final configurations as SLURM jobs
# (04b_cv_test.py, then 04b_cv_test_evaluate.py).
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_cv_test.sh folds [options]
#     combined model matrices, cell-month table, folds of every scheme
#   bash src/02_model_training/sbatch_cv_test.sh fit <model|all> <scheme|all> [--folds A-B] [options]
#     one job array per model and scheme, one task per fold, one task at a time
#     (schemes: loyo, blocks); --folds A-B: only these folds, e.g. 0
#   bash src/02_model_training/sbatch_cv_test.sh baseline <scheme|validation|all> [options]
#   bash src/02_model_training/sbatch_cv_test.sh autocorrelation [options]
#   bash src/02_model_training/sbatch_cv_test.sh evaluate [options]
#   bash src/02_model_training/sbatch_cv_test.sh all [options]
#     folds, then the fits (the two schemes of a model one after the other) and the
#     baselines, the autocorrelation, then the evaluation (chained)
#   options:
#     --force   recompute output that already exists
#     --after   JOB_ID: start only after that job has finished successfully (several:
#               ID1:ID2:...)
#     --exclude NODES: never run on these nodes (comma-separated), e.g. a node that fails
#               every job at once

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/04b_cv_test.py"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(extra_trees xgboost random_forest)      # model_training.models in config.toml
SCHEMES=(loyo blocks)                           # model_training.cv_test.schemes
declare -A N_FOLDS=( [loyo]=6 [blocks]=5 )      # training.years; model_training.cv_test.block_folds

# Resources: "memory (GB) CPUs time (min)", CPUs = memory / 5.5 GB rounded up. Measured on the
# runs of 2026-11-12 (the same code, 85.3M rows of all cell-months; peak memory, run time per
# fold fit): extra trees + SMOTE 95-109 GB, 23-27 min (16 threads, 134-142M rows after SMOTE);
# XGBoost + class weights 14.3-15.1 GB, 3.7-4.8 min (4 threads); random forest + RUS
# 7.0-7.8 GB, 1.2-1.9 min (3 threads). The CPUs of XGBoost and random forest are those of
# these runs, so the threads are the same (XGBoost's histogram sums can depend on them in
# the last bits). folds: combining the three subsets (4.8 GB predictors, memory-mapped) and
# the cell-month table checked against the split index, 9.1 GB, 1 min (2026-11-13). Baselines (KDE on 131,568 cells, a few thousand
# fire cells per fold) and the evaluation (AP, ROC-AUC, top-k on 9.5M cell-months per
# method): about 2 GB, a few minutes per scheme; 9 min for all schemes and methods.
declare -A FIT_RESOURCES=( [extra_trees]="120 22 45" [xgboost]="18 4 20" [random_forest]="9 3 15" )
FOLDS_RESOURCES="12 3 20"
BASELINE_RESOURCES="4 1 20"
AUTOCORRELATION_RESOURCES="4 1 10"
EVALUATE_RESOURCES="6 2 30"

usage() { sed -n '2,24p' "$0"; exit 1; }

# submit <job name> <memory GB> <CPUs> <time (min)> <extra sbatch options> <commands>; prints the job ID
submit() {
    local name=$1 mem=$2 cpus=$3 minutes=$4 opts=$5 commands=$6 qos=short log
    [ "$minutes" -gt 1440 ] && qos=medium
    [[ "$opts" == *--array* ]] && log="${name}_%A_%a" || log="${name}_%j"
    sbatch --parsable $EXCLUDE $opts <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=standard
#SBATCH --qos=$qos
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/$log.out
#SBATCH --error=$LOG_DIR/$log.err
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

submit_simple() {   # <job name> <resources> <after> <commands>; prints the job ID
    local name=$1 resources=$2 after=$3 commands=$4 id mem cpus minutes
    read -r mem cpus minutes <<< "$resources"
    id=$(submit "$name" "$mem" "$cpus" "$minutes" "$(dependency "$after")" "$commands")
    echo "$name (${mem} GB, $cpus CPUs, $minutes min): $id" >&2
    [ -n "$id" ] || exit 1
    echo "$id"
}

submit_fit() {   # <model> <scheme> <after>; prints the job ID
    local model=$1 scheme=$2 after=$3 id mem cpus minutes range
    read -r mem cpus minutes <<< "${FIT_RESOURCES[$model]}"
    range=${FOLDS:-0-$(( ${N_FOLDS[$scheme]} - 1 ))}
    id=$(submit "cv_test_fit_${model}_${scheme}" "$mem" "$cpus" "$minutes" \
         "$(dependency "$after") --array=${range}%1" \
         "python $SCRIPT fit $model $scheme \$SLURM_ARRAY_TASK_ID $FORCE")
    echo "fit $model $scheme folds $range (${mem} GB, $cpus CPUs, $minutes min, one task at a time): $id" >&2
    [ -n "$id" ] || exit 1
    echo "$id"
}

submit_baseline() {   # <scheme|validation> <after>; prints the job ID
    submit_simple "cv_test_baseline_$1" "$BASELINE_RESOURCES" "$2" "python $SCRIPT baseline $1 $FORCE"
}

pick() {   # pick <value|all> <allowed ...>: prints the selection, or nothing if not allowed
    local value=$1; shift
    if [ "$value" = "all" ]; then echo "$@"; return; fi
    for v in "$@"; do [ "$v" = "$value" ] && echo "$v" && return; done
}

ARGS=()
FORCE=""; AFTER=""; EXCLUDE=""; FOLDS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force" ;;
        --after) AFTER="$2"; shift ;;
        --exclude) [ -n "$2" ] || usage; EXCLUDE="--exclude=$2"; shift ;;
        --folds) [ -n "$2" ] || usage; FOLDS="$2"; shift ;;
        --*) usage ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

COMMAND=${ARGS[0]}
case "$COMMAND" in
    folds|autocorrelation|evaluate|all) [ ${#ARGS[@]} -eq 1 ] || usage ;;
    fit) [ ${#ARGS[@]} -eq 3 ] || usage
         RUN_MODELS=($(pick "${ARGS[1]}" "${MODELS[@]}"))
         RUN_SCHEMES=($(pick "${ARGS[2]}" "${SCHEMES[@]}"))
         [ ${#RUN_MODELS[@]} -gt 0 ] && [ ${#RUN_SCHEMES[@]} -gt 0 ] || usage ;;
    baseline) [ ${#ARGS[@]} -eq 2 ] || usage
              RUN_SCHEMES=($(pick "${ARGS[1]}" "${SCHEMES[@]}" validation))
              [ ${#RUN_SCHEMES[@]} -gt 0 ] || usage ;;
    *) usage ;;
esac
[ -n "$FOLDS" ] && [ "$COMMAND" != fit ] && usage

# A failed submission stops the chain, so that no job starts without its dependency.
case "$COMMAND" in
    folds)
        submit_simple cv_test_folds "$FOLDS_RESOURCES" "$AFTER" "python $SCRIPT folds $FORCE" >/dev/null || exit 1 ;;
    fit)
        for model in "${RUN_MODELS[@]}"; do
            for scheme in "${RUN_SCHEMES[@]}"; do
                submit_fit "$model" "$scheme" "$AFTER" >/dev/null || exit 1
            done
        done ;;
    baseline)
        for scheme in "${RUN_SCHEMES[@]}"; do
            submit_baseline "$scheme" "$AFTER" >/dev/null || exit 1
        done ;;
    autocorrelation)
        submit_simple cv_test_autocorrelation "$AUTOCORRELATION_RESOURCES" "$AFTER" \
            "python $SCRIPT autocorrelation $FORCE" >/dev/null || exit 1 ;;
    evaluate)
        submit_simple cv_test_evaluate "$EVALUATE_RESOURCES" "$AFTER" \
            "python $SCRIPT_DIR/04b_cv_test_evaluate.py" >/dev/null || exit 1 ;;
    all)
        folds_id=$(submit_simple cv_test_folds "$FOLDS_RESOURCES" "$AFTER" "python $SCRIPT folds $FORCE") || exit 1
        IDS=()
        for model in "${MODELS[@]}"; do      # at most one fit per model at a time
            after=$folds_id
            for scheme in "${SCHEMES[@]}"; do
                after=$(submit_fit "$model" "$scheme" "$after") || exit 1
            done
            IDS+=("$after")
        done
        for scheme in "${SCHEMES[@]}" validation; do
            id=$(submit_baseline "$scheme" "$folds_id") || exit 1
            IDS+=("$id")
        done
        id=$(submit_simple cv_test_autocorrelation "$AUTOCORRELATION_RESOURCES" "$AFTER" \
             "python $SCRIPT autocorrelation $FORCE") || exit 1
        IDS+=("$id")
        submit_simple cv_test_evaluate "$EVALUATE_RESOURCES" "$(IFS=:; echo "${IDS[*]}")" \
            "python $SCRIPT_DIR/04b_cv_test_evaluate.py" >/dev/null || exit 1 ;;
esac
