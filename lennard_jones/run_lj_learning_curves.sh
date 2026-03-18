#!/bin/bash
#SBATCH --job-name=lj_learning_curves
#SBATCH --account=def-anatole
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=lj_learning_curves_%j.out
#SBATCH --error=lj_learning_curves_%j.err

# =============================================================================
# LJ Learning Curves on Cedar (Compute Canada)
# =============================================================================
#
# USAGE:
#   1. Copy this directory to Cedar:
#        scp -r lennard_jones/ cedar:~/projects/def-anatole/$USER/adaptive/
#   2. Submit:
#        cd ~/projects/def-anatole/$USER/adaptive/lennard_jones/
#        sbatch run_lj_learning_curves.sh
#   3. Monitor:
#        squeue -u $USER
#        tail -f lj_learning_curves_*.out
#   4. After completion, copy results back:
#        scp cedar:~/projects/def-anatole/$USER/adaptive/lennard_jones/lj_learning_curves_results.npz .
#   5. Plot locally:
#        python lj_learning_curves_plot.py
# =============================================================================

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module load python/3.11

VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy scipy matplotlib scikit-learn pandas

cd $SLURM_SUBMIT_DIR
python lj_learning_curves_compute.py \
    --n-seeds 5 \
    --n-jobs 1 \
    --output lj_learning_curves_results.npz

echo "Job finished: $(date)"
