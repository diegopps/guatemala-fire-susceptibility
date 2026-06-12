
SCRIPT="$PROJECT_DIR/src/03_future_classification/01_plot_future_classification.py"
PYTHON="python"

echo "Submitting job for 'plot future classif'"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=plot_futclassif
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=gvca
#SBATCH --output=logs/plot_futclassif_%j.out
#SBATCH --error=logs/plot_futclassif_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

ml miniforge
source activate forestFireGT

$PYTHON $SCRIPT
EOF
