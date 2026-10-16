#!/bin/bash
# sbatch_data_exploration.sh
# --------------------------
# Submits the data-exploration plotters (stage 1), one SLURM job each.
# predictormaps, trends and correlation reuse their saved tables
# (results/01_data_exploration/) unless --recompute is given, so re-plotting is fast;
# firemap is quick and always recomputes.
#
# Usage (from any directory):
#   bash src/01_data_exploration/sbatch_data_exploration.sh [--recompute] [plotter ...]
#   plotters: firemap  predictormaps  trends  correlation   (default: all)
#   e.g.  bash src/01_data_exploration/sbatch_data_exploration.sh predictormaps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# plotter -> script, memory, CPUs, time limit
declare -A SCRIPT=( [firemap]=01_fire_records_map.py   [predictormaps]=02_predictor_maps.py
                    [trends]=03_climate_trends.py      [correlation]=04_correlation.py )
declare -A MEM=(    [firemap]=8G   [predictormaps]=16G [trends]=32G [correlation]=64G )
declare -A CPUS=(   [firemap]=1    [predictormaps]=1   [trends]=16  [correlation]=1 )
declare -A TIME=(   [firemap]=00:20:00 [predictormaps]=01:00:00 [trends]=02:00:00 [correlation]=01:00:00 )
ALL=(firemap predictormaps trends correlation)

RECOMPUTE=""
PLOTTERS=()
for arg in "$@"; do
    if [ "$arg" = "--recompute" ]; then
        RECOMPUTE="--recompute"
    elif [ -n "${SCRIPT[$arg]}" ]; then
        PLOTTERS+=("$arg")
    else
        sed -n '2,13p' "$0"; exit 1
    fi
done
[ ${#PLOTTERS[@]} -eq 0 ] && PLOTTERS=("${ALL[@]}")

for p in "${PLOTTERS[@]}"; do
    ARGS=""
    [ "$p" != "firemap" ] && ARGS="$RECOMPUTE"
    echo -n "$p: "
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dataexpl_$p
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=$LOG_DIR/dataexpl_${p}_%j.out
#SBATCH --error=$LOG_DIR/dataexpl_${p}_%j.err
#SBATCH --time=${TIME[$p]}
#SBATCH --mem=${MEM[$p]}
#SBATCH --cpus-per-task=${CPUS[$p]}

ml miniforge
source activate forestFireGT

python $SCRIPT_DIR/${SCRIPT[$p]} $ARGS
EOF
done
