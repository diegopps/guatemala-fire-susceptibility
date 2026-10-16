#!/bin/bash
# sbatch_00_aggregate_monthly_gcm.sh
# ----------------------------------
# Submits 00_aggregate_monthly_gcm.py as a SLURM array job, one task per GCM and
# scenario (GCMs, scenarios and year ranges are read from config.toml).
#
# Usage:
#   bash src/00_preprocessing/sbatch_00_aggregate_monthly_gcm.sh
#       all GCMs x all scenarios (historical + SSPs), all years
#   bash src/00_preprocessing/sbatch_00_aggregate_monthly_gcm.sh <scenario> <first_year> <last_year>
#       all GCMs, one scenario, a year range (e.g. ssp370 2017 2024)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

if [ $# -eq 0 ]; then
    SCENARIOS=$(python3 -c "import tomllib; c = tomllib.load(open('$PROJECT_DIR/config.toml','rb'))['scenarios']; print(' '.join([c['historical']] + c['ssp']))")
    YEARS=""
elif [ $# -eq 3 ]; then
    SCENARIOS=$1; YEARS="$2 $3"
else
    sed -n '2,12p' "$0"; exit 1
fi
GCMS=$(python3 -c "import tomllib; print(' '.join(tomllib.load(open('$PROJECT_DIR/config.toml','rb'))['gcms']['models']))")

# One task per (GCM, scenario) pair
TASKS=()
for g in $GCMS; do for s in $SCENARIOS; do TASKS+=("$g:$s"); done; done
N=${#TASKS[@]}

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=monthly_gcm
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --array=0-$((N - 1))
#SBATCH --output=$LOG_DIR/monthly_gcm_%A_%a.out
#SBATCH --error=$LOG_DIR/monthly_gcm_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

ml miniforge
source activate forestFireGT

TASKS=(${TASKS[@]})
TASK=\${TASKS[\$SLURM_ARRAY_TASK_ID]}
python $SCRIPT_DIR/00_aggregate_monthly_gcm.py \${TASK%%:*} \${TASK##*:} $YEARS
EOF
echo "Tasks: $N (GCM x scenario)"
