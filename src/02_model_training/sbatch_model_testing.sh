#!/bin/bash
# sbatch_model_testing.sh
# -----------------------
# Submits the model testing as SLURM jobs: 01_model_testing.py, one job per model (k-NN: a
# job array over blocks of test rows), and 01_model_testing_evaluate.py.
# CPUs are requested in proportion to memory (about 5.5 GB per core on the standard nodes,
# which SLURM allocates anyway); the scripts use only the CPUs of their allocation.
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_model_testing.sh <model|all|evaluate> [options]
#     <model>   one of the models below
#     all       every model; without --trial, the evaluation is submitted too and starts
#               when every model has finished successfully
#     evaluate  only the evaluation
#   options:
#     --trial   1% of the rows, small jobs, separate output (models only)
#     --force   recompute output that already exists (models only)
#     --after   JOB_ID: start only after that job has finished successfully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(logistic_regression decision_tree random_forest gradient_boosting xgboost extra_trees knn naive_bayes)

# model -> memory, CPUs, time limit, QOS of the full run (scaled from the 1% trial runs;
# refine from the resource line at the end of each log)
declare -A MEM=(  [logistic_regression]=32G [decision_tree]=32G [random_forest]=128G
                  [gradient_boosting]=16G [xgboost]=32G [extra_trees]=128G
                  [knn]=32G [naive_bayes]=24G )
declare -A CPUS=( [logistic_regression]=6 [decision_tree]=6 [random_forest]=23
                  [gradient_boosting]=3 [xgboost]=6 [extra_trees]=23
                  [knn]=6 [naive_bayes]=4 )
declare -A TIME=( [logistic_regression]=04:00:00 [decision_tree]=06:00:00 [random_forest]=06:00:00
                  [gradient_boosting]=2-00:00:00 [xgboost]=02:00:00 [extra_trees]=06:00:00
                  [knn]=12:00:00 [naive_bayes]=01:00:00 )
declare -A QOS=(  [gradient_boosting]=medium )
KNN_PARTS=20        # blocks of test rows for k-NN
KNN_AT_ONCE=4       # array tasks running at the same time

usage() { sed -n '2,18p' "$0"; exit 1; }

# submit <job name> <memory> <CPUs> <time> <QOS> <extra sbatch options> <script and arguments>
# Prints the job ID.
submit() {
    local name=$1 mem=$2 cpus=$3 time=$4 qos=$5 opts=$6 cmd=$7 log
    [[ "$opts" == *--array* ]] && log="${name}_%A_%a" || log="${name}_%j"
    sbatch --parsable $opts <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=standard
#SBATCH --qos=$qos
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/$log.out
#SBATCH --error=$LOG_DIR/$log.err
#SBATCH --time=$time
#SBATCH --mem=$mem
#SBATCH --cpus-per-task=$cpus

ml miniforge
source activate forestFireGT

# Numerical libraries use only the allocated CPUs.
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK

python $SCRIPT_DIR/$cmd
EOF
}

submit_evaluate() {   # <extra sbatch options>
    echo "evaluate: $(submit model_testing_evaluate 16G 3 01:00:00 short "$1" \
                      "01_model_testing_evaluate.py")"
}

TARGET="$1"; shift
TRIAL=""; FORCE=""; DEPENDENCY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --trial) TRIAL="--trial" ;;
        --force) FORCE="--force" ;;
        --after) DEPENDENCY="--dependency=afterok:$2"; shift ;;
        *) usage ;;
    esac
    shift
done

if [ "$TARGET" = "evaluate" ]; then
    submit_evaluate "$DEPENDENCY"
    exit 0
elif [ "$TARGET" = "all" ]; then
    RUN=("${MODELS[@]}")
elif [ -n "$TARGET" ] && [ -n "${MEM[$TARGET]}" ]; then
    RUN=("$TARGET")
else
    usage
fi

IDS=()
for m in "${RUN[@]}"; do
    name="model_testing_$m"
    if [ -n "$TRIAL" ]; then
        id=$(submit "${name}_trial" 16G 3 02:00:00 short "$DEPENDENCY" "01_model_testing.py $m --trial $FORCE")
    elif [ "$m" = "knn" ]; then
        id=$(submit "$name" "${MEM[$m]}" "${CPUS[$m]}" "${TIME[$m]}" short \
             "$DEPENDENCY --array=1-${KNN_PARTS}%${KNN_AT_ONCE}" \
             "01_model_testing.py $m --part "'$SLURM_ARRAY_TASK_ID'" --parts $KNN_PARTS $FORCE")
    else
        id=$(submit "$name" "${MEM[$m]}" "${CPUS[$m]}" "${TIME[$m]}" "${QOS[$m]:-short}" \
             "$DEPENDENCY" "01_model_testing.py $m $FORCE")
    fi
    echo "$m: $id"
    IDS+=("$id")
done

if [ "$TARGET" = "all" ] && [ -z "$TRIAL" ]; then
    submit_evaluate "--dependency=afterok:$(IFS=:; echo "${IDS[*]}")"
fi
