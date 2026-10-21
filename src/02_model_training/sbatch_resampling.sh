#!/bin/bash
# sbatch_resampling.sh
# --------------------
# Submits the resampling step as SLURM jobs: 02_resampling.py as job arrays per model over
# the samplers of similar training size, and 02_resampling_evaluate.py.
# CPUs are requested in proportion to memory (about 5.5 GB per core on the standard nodes,
# which SLURM allocates anyway); the scripts use only the CPUs of their allocation.
#
# Usage (from any directory):
#   bash src/02_model_training/sbatch_resampling.sh <model|all> <sampler|all> [options]
#   bash src/02_model_training/sbatch_resampling.sh evaluate [--after JOB_ID]
#     with "all all" (and no --trial) the evaluation is submitted too and starts when
#     every fit has finished successfully
#   options:
#     --trial   1% of the data, small jobs, separate output
#     --force   recompute output that already exists
#     --after   JOB_ID: start only after that job has finished successfully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=(extra_trees xgboost random_forest)      # model_training.models in config.toml
SAMPLERS=(ros smote adasyn rus tomek ncr)       # resampling.SAMPLERS

# Training size of each sampler: large (about twice the train subset: ros, smote, adasyn),
# medium (about the train subset: tomek, ncr) or small (rus).
declare -A SIZE=( [ros]=large [smote]=large [adasyn]=large [tomek]=medium [ncr]=medium [rus]=small )

# Memory, CPUs, time limit per model and training size. 02_resampling.py uses at most 16
# threads, so the memory covers the trees plus about 2 GB (large) or 1 GB (medium) per
# thread; memory for the random-forest and extra-trees trees on the large sets is on the
# safe side, since the 1% trial runs did not show their size.
declare -A MEM=(  [extra_trees,large]=192G [xgboost,large]=64G [random_forest,large]=360G
                  [extra_trees,medium]=96G [xgboost,medium]=32G [random_forest,medium]=96G
                  [extra_trees,small]=16G [xgboost,small]=16G [random_forest,small]=16G )
declare -A CPUS=( [extra_trees,large]=16 [xgboost,large]=12 [random_forest,large]=16
                  [extra_trees,medium]=16 [xgboost,medium]=6 [random_forest,medium]=16
                  [extra_trees,small]=3 [xgboost,small]=3 [random_forest,small]=3 )
declare -A TIME=( [extra_trees,large]=12:00:00 [xgboost,large]=04:00:00 [random_forest,large]=1-00:00:00
                  [extra_trees,medium]=06:00:00 [xgboost,medium]=02:00:00 [random_forest,medium]=06:00:00
                  [extra_trees,small]=01:00:00 [xgboost,small]=01:00:00 [random_forest,small]=01:00:00 )
declare -A AT_ONCE=( [large]=1 [medium]=2 [small]=1 )   # array tasks of one model running at the same time

usage() { sed -n '2,17p' "$0"; exit 1; }

# submit <job name> <memory> <CPUs> <time> <extra sbatch options> <commands>; prints the job ID
submit() {
    local name=$1 mem=$2 cpus=$3 time=$4 opts=$5 commands=$6 log
    [[ "$opts" == *--array* ]] && log="${name}_%A_%a" || log="${name}_%j"
    sbatch --parsable $opts <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=standard
#SBATCH --qos=short
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

$commands
EOF
}

submit_evaluate() {   # <dependency option>
    echo "evaluate: $(submit resampling_evaluate 16G 3 01:00:00 "$1" \
                      "python $SCRIPT_DIR/02_resampling_evaluate.py")"
}

pick() {   # pick <value|all> <allowed ...>: prints the selection, or nothing if not allowed
    local value=$1; shift
    if [ "$value" = "all" ]; then echo "$@"; return; fi
    for v in "$@"; do [ "$v" = "$value" ] && echo "$v" && return; done
}

ARGS=()
TRIAL=""; FORCE=""; DEPENDENCY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --trial) TRIAL="--trial" ;;
        --force) FORCE="--force" ;;
        --after) DEPENDENCY="--dependency=afterok:$2"; shift ;;
        --*) usage ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

if [ "${ARGS[0]}" = "evaluate" ] && [ ${#ARGS[@]} -eq 1 ]; then
    submit_evaluate "$DEPENDENCY"
    exit 0
fi
[ ${#ARGS[@]} -eq 2 ] || usage
RUN_MODELS=($(pick "${ARGS[0]}" "${MODELS[@]}"))
RUN_SAMPLERS=($(pick "${ARGS[1]}" "${SAMPLERS[@]}"))
[ ${#RUN_MODELS[@]} -gt 0 ] && [ ${#RUN_SAMPLERS[@]} -gt 0 ] || usage

IDS=()
for model in "${RUN_MODELS[@]}"; do
    for size in large medium small; do
        group=()
        for s in "${RUN_SAMPLERS[@]}"; do [ "${SIZE[$s]}" = "$size" ] && group+=("$s"); done
        [ ${#group[@]} -eq 0 ] && continue
        run="S=(${group[*]}); python $SCRIPT_DIR/02_resampling.py $model \${S[\$SLURM_ARRAY_TASK_ID]}"
        if [ -n "$TRIAL" ]; then
            id=$(submit "resampling_${model}_trial" 16G 3 02:00:00 \
                 "$DEPENDENCY --array=0-$((${#group[@]} - 1))" "$run --trial $FORCE")
        else
            id=$(submit "resampling_${model}_$size" "${MEM[$model,$size]}" "${CPUS[$model,$size]}" \
                 "${TIME[$model,$size]}" "$DEPENDENCY --array=0-$((${#group[@]} - 1))%${AT_ONCE[$size]}" \
                 "$run $FORCE")
        fi
        echo "$model (${group[*]}): $id"
        IDS+=("$id")
    done
done

if [ "${ARGS[0]}" = "all" ] && [ "${ARGS[1]}" = "all" ] && [ -z "$TRIAL" ]; then
    submit_evaluate "--dependency=afterok:$(IFS=:; echo "${IDS[*]}")"
fi
