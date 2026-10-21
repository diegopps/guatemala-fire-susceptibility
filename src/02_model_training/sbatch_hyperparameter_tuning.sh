#!/bin/bash
# sbatch_hyperparameter_tuning.sh
# -------------------------------
# Submits the hyperparameter tuning as SLURM jobs (03_hyperparameter_tuning.py and
# 03_hyperparameter_tuning_evaluate.py).
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_hyperparameter_tuning.sh prepare [options]
#   bash src/02_model_training/sbatch_hyperparameter_tuning.sh cv <model|all> [options]
#     one job array per model and number of trees, one task per configuration and fold
#     (--trial: one job per model running all its fits)
#   bash src/02_model_training/sbatch_hyperparameter_tuning.sh refit <model|all> [options]
#     once every fit of the model is scored (its resources depend on the selected
#     configuration); with "all" the evaluation is submitted too, after the refits
#   bash src/02_model_training/sbatch_hyperparameter_tuning.sh evaluate [options]
#   bash src/02_model_training/sbatch_hyperparameter_tuning.sh all [options]
#     prepare, then cv of all models; with --trial also refit and evaluate (all chained)
#   options:
#     --trial   1% of the cell-months, small jobs, separate output
#     --force   recompute output that already exists
#     --after   JOB_ID: start only after that job has finished successfully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/03_hyperparameter_tuning.py"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(extra_trees xgboost random_forest)      # model_training.models in config.toml

# Training size of each imbalance correction (model_training.samplers): large (about twice
# the training cell-months: ros, smote, adasyn), full (about all of them: class_weights,
# tomek, ncr) or small (rus).
declare -A SIZE=( [ros]=large [smote]=large [adasyn]=large
                  [class_weights]=full [tomek]=full [ncr]=full [rus]=small )

# Resources of one CV fit (2/3 of the training cell-months, at most 16 threads), per model
# and training size, as a function of the number of trees n:
#   memory (GB) = MEM_BASE + MEM_TREES * n / 100     (data and working arrays + trees)
#   time limit (min) = MIN_TREES * n / 100, at least 30
# Estimated from the step-02 runs (100 trees on the whole train subset, scaled to 2/3; time
# with a margin for the slower configurations, e.g. max_features 0.5). Measured for the
# selected setups: extra trees + SMOTE 84 GB, 17.8 min; XGBoost + class weights 10.5 GB,
# 1.4 min; random forest + RUS 5.3 GB, seconds. PLACEHOLDERS until checked with the trial
# runs (cv JSON files: fit_minutes, tree_nodes, peak_memory_gb).
declare -A MEM_BASE=(  [extra_trees,large]=24 [xgboost,large]=20 [random_forest,large]=32
                       [extra_trees,full]=16  [xgboost,full]=10  [random_forest,full]=16
                       [extra_trees,small]=8  [xgboost,small]=8  [random_forest,small]=8 )
declare -A MEM_TREES=( [extra_trees,large]=32 [xgboost,large]=1  [random_forest,large]=60
                       [extra_trees,full]=12  [xgboost,full]=1   [random_forest,full]=20
                       [extra_trees,small]=1  [xgboost,small]=1  [random_forest,small]=1 )
declare -A MIN_TREES=( [extra_trees,large]=40 [xgboost,large]=30 [random_forest,large]=180
                       [extra_trees,full]=30  [xgboost,full]=15  [random_forest,full]=90
                       [extra_trees,small]=10 [xgboost,small]=10 [random_forest,small]=10 )
CV_AT_ONCE=4      # PLACEHOLDER: tasks of one array (model and number of trees) running at the same time
REFIT_SCALE=3/2   # refit: whole train subset, 3/2 of a CV fit (memory and time)

usage() { sed -n '2,21p' "$0"; exit 1; }

# submit <job name> <memory GB> <CPUs> <time (min)> <extra sbatch options> <commands>; prints the job ID
submit() {
    local name=$1 mem=$2 cpus=$3 minutes=$4 opts=$5 commands=$6 log qos=short
    [[ "$opts" == *--array* ]] && log="${name}_%A_%a" || log="${name}_%j"
    [ "$minutes" -gt 1440 ] && qos=medium
    sbatch --parsable $opts <<EOF
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

# CPUs in proportion to memory (about 5.5 GB per core on the standard nodes, which SLURM
# allocates anyway); the scripts use at most 16 threads per fit.
cpus_for() { local c=$(( ($1 * 2 + 10) / 11 )); [ "$c" -lt 1 ] && c=1; echo "$c"; }

# Resources of one fit of a model with a training size and n trees, times a factor: "mem cpus minutes"
resources() {
    local key="$1,$2" n=$3 scale=${4:-1/1} mem minutes
    mem=$(( (${MEM_BASE[$key]} + ${MEM_TREES[$key]} * n / 100) * ${scale%/*} / ${scale#*/} ))
    minutes=$(( ${MIN_TREES[$key]} * n / 100 * ${scale%/*} / ${scale#*/} ))
    [ "$minutes" -lt 30 ] && minutes=30
    echo "$mem $(cpus_for "$mem") $minutes"
}

# The project environment, for the helper calls of 03_hyperparameter_tuning.py (no model data)
activate() { ml miniforge >/dev/null 2>&1; source activate forestFireGT || exit 1; }

dependency() { [ -n "$1" ] && echo "--dependency=afterok:$1"; }

submit_prepare() {   # <after>
    local id
    if [ -n "$TRIAL" ]; then
        id=$(submit tuning_prepare_trial 16 3 60 "$(dependency "$1")" "python $SCRIPT prepare --trial $FORCE")
    else
        # Resampling of 3 folds per sampler: seconds for ros/smote/adasyn/rus, about 5-10 min
        # per fold for tomek/ncr (step 02: 3-9 min on all training cell-months, 11 GB with the fit)
        id=$(submit tuning_prepare 16 3 120 "$(dependency "$1")" "python $SCRIPT prepare $FORCE")
    fi
    echo "prepare: $id" >&2
    [ -n "$id" ] || exit 1
    echo "$id"
}

submit_cv() {   # <model> <after>; prints the job IDs, separated by ":"
    local model=$1 after=$2 out sampler folds size n ids res ID_LIST=()
    if [ -n "$TRIAL" ]; then
        # All 303 fits of the model on 1% of the cell-months, one after the other (step-02
        # trials: 0.6 min per 100-tree fit of extra trees + SMOTE on 3 CPUs; generous limits)
        ID_LIST+=($(submit "tuning_cv_${model}_trial" 44 8 1440 "$(dependency "$after")" \
                    "python $SCRIPT cv $model all all --trial $FORCE"))
        echo "cv $model (trial): ${ID_LIST[0]}" >&2
    else
        out=$(python "$SCRIPT" trees "$model") || exit 1
        sampler=$(awk '$1 == "sampler" {print $2}' <<< "$out")
        folds=$(awk '$1 == "folds" {print $2}' <<< "$out")
        size=${SIZE[$sampler]}
        while read -r n ids; do
            [[ "$n" =~ ^[0-9]+$ ]] || continue
            ids=($ids)
            read -r mem cpus minutes <<< "$(resources "$model" "$size" "$n")"
            run="C=(${ids[*]}); i=\$SLURM_ARRAY_TASK_ID; python $SCRIPT cv $model \${C[\$((i / $folds))]} \$((i % $folds)) $FORCE"
            res=$(submit "tuning_cv_${model}_${n}trees" "$mem" "$cpus" "$minutes" \
                  "$(dependency "$after") --array=0-$(( ${#ids[@]} * folds - 1 ))%$CV_AT_ONCE" "$run")
            echo "cv $model ($sampler, $n trees: ${#ids[@]} configurations x $folds folds, ${mem} GB, $cpus CPUs, $minutes min): $res" >&2
            [ -n "$res" ] || exit 1
            ID_LIST+=("$res")
        done <<< "$out"
    fi
    [ ${#ID_LIST[@]} -gt 0 ] && [ -n "${ID_LIST[0]}" ] || exit 1
    (IFS=:; echo "${ID_LIST[*]}")
}

submit_refit() {   # <model> <after>
    local model=$1 after=$2 out sampler n id
    if [ -n "$TRIAL" ]; then
        # Refit on 1% of the cell-months and prediction of 1% of the test cell-months
        id=$(submit "tuning_refit_${model}_trial" 16 3 120 "$(dependency "$after")" \
             "python $SCRIPT refit $model --trial $FORCE")
        echo "refit $model (trial): $id" >&2
    else
        out=$(python "$SCRIPT" trees "$model") || exit 1
        sampler=$(awk '$1 == "sampler" {print $2}' <<< "$out")
        n=$(python "$SCRIPT" trees "$model" --best) || exit 1
        read -r mem cpus minutes <<< "$(resources "$model" "${SIZE[$sampler]}" "$n" "$REFIT_SCALE")"
        id=$(submit "tuning_refit_${model}" "$mem" "$cpus" "$minutes" "$(dependency "$after")" \
             "python $SCRIPT refit $model $FORCE")
        echo "refit $model ($sampler, $n trees, ${mem} GB, $cpus CPUs, $minutes min): $id" >&2
    fi
    [ -n "$id" ] || exit 1
    echo "$id"
}

submit_evaluate() {   # <after>
    local id
    id=$(submit "tuning_evaluate${TRIAL:+_trial}" 16 3 60 "$(dependency "$1")" \
         "python $SCRIPT_DIR/03_hyperparameter_tuning_evaluate.py $TRIAL")
    echo "evaluate: $id" >&2
}

pick() {   # pick <value|all> <allowed ...>: prints the selection, or nothing if not allowed
    local value=$1; shift
    if [ "$value" = "all" ]; then echo "$@"; return; fi
    for v in "$@"; do [ "$v" = "$value" ] && echo "$v" && return; done
}

ARGS=()
TRIAL=""; FORCE=""; AFTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --trial) TRIAL="--trial" ;;
        --force) FORCE="--force" ;;
        --after) AFTER="$2"; shift ;;
        --*) usage ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

COMMAND=${ARGS[0]}
case "$COMMAND" in
    prepare|evaluate|all) [ ${#ARGS[@]} -eq 1 ] || usage ;;
    cv|refit) [ ${#ARGS[@]} -eq 2 ] || usage
              RUN_MODELS=($(pick "${ARGS[1]}" "${MODELS[@]}"))
              [ ${#RUN_MODELS[@]} -gt 0 ] || usage ;;
    *) usage ;;
esac
activate

# Every model needs its imbalance correction (model_training.samplers) before anything is submitted.
if [ "$COMMAND" != "evaluate" ]; then
    for model in "${MODELS[@]}"; do python "$SCRIPT" trees "$model" >/dev/null || exit 1; done
fi

# A failed submission stops the chain, so that no job starts without its dependency.
IDS=()
case "$COMMAND" in
    prepare)  submit_prepare "$AFTER" >/dev/null ;;
    evaluate) submit_evaluate "$AFTER" ;;
    cv)       for model in "${RUN_MODELS[@]}"; do submit_cv "$model" "$AFTER" >/dev/null || exit 1; done ;;
    refit)    for model in "${RUN_MODELS[@]}"; do
                  id=$(submit_refit "$model" "$AFTER") || exit 1
                  IDS+=("$id")
              done
              [ "${ARGS[1]}" = "all" ] && submit_evaluate "$(IFS=:; echo "${IDS[*]}")" ;;
    all)      prep=$(submit_prepare "$AFTER") || exit 1
              for model in "${MODELS[@]}"; do
                  cv_ids=$(submit_cv "$model" "$prep") || exit 1
                  if [ -n "$TRIAL" ]; then
                      id=$(submit_refit "$model" "$cv_ids") || exit 1
                      IDS+=("$id")
                  fi
              done
              if [ -n "$TRIAL" ]; then
                  submit_evaluate "$(IFS=:; echo "${IDS[*]}")"
              else
                  echo "Submit the refits once every fit is scored: bash $0 refit all"
              fi ;;
esac
