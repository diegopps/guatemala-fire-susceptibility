#!/bin/bash
#SBATCH --job-name=FFGT-plots
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=gvca
#SBATCH --output=slurm_log/FFGT-%j.out
#SBATCH --error=slurm_log/FFGT-%j.err

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================"

ml miniforge
source activate forestFireGT

python3 timevars_visualization.py

echo ""
echo "============================================"
echo "End:    $(date)"
echo "Exit:   $?"
echo "============================================"
