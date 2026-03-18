#!/bin/bash
#SBATCH --job-name=learning_curves
#SBATCH --account=def-anatole   # <-- CHANGE THIS to your allocation
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=learning_curves_%j.out
#SBATCH --error=learning_curves_%j.err

# =============================================================================
# Learning Curves on Cedar (Compute Canada)
# =============================================================================
#
# BEFORE FIRST USE:
#   1. Edit --account above to your allocation (e.g., def-avlilienfeld)
#   2. Copy this directory to Cedar:
#        scp -r rose_uber/ cedar:~/projects/def-YOUR_ACCOUNT/$USER/adaptive/
#   3. Submit:
#        cd ~/projects/def-YOUR_ACCOUNT/$USER/adaptive/rose_uber/
#        sbatch run_learning_curves.sh
#   4. Monitor:
#        squeue -u $USER
#        tail -f learning_curves_*.out
#   5. After completion, copy results back:
#        scp cedar:~/projects/def-YOUR_ACCOUNT/$USER/adaptive/rose_uber/learning_curves_results.npz .
#   6. Plot locally:
#        python learning_curves_plot.py
# =============================================================================

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Load modules
module load python/3.11

# Create virtualenv if it doesn't exist
VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies (uses Compute Canada wheels)
pip install --no-index --upgrade pip
pip install --no-index numpy scipy matplotlib scikit-learn pandas

# Run computation
# n_jobs=1: memory-safe, each GridSearchCV runs sequentially
# The 4 CPUs are used by numpy/scipy BLAS internally
cd $SLURM_SUBMIT_DIR
python learning_curves_compute.py \
    --n-seeds 5 \
    --n-jobs 1 \
    --output learning_curves_results.npz

echo "Job finished: $(date)"
