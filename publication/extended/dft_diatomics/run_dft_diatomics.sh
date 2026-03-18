#!/bin/bash
#SBATCH --job-name=dft_diatomics
#SBATCH --account=def-anatole
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# DFT binding curve scan for 30 diatomic molecules
# PBE/def2-SVP, 25 r-points per molecule, r_max=5*r_eq
# UKS for all molecules (handles spin symmetry breaking at dissociation)
# Expected runtime: ~3-6 hours for 30 diatomics

echo "=== DFT Diatomic Scan ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Load python module (same as setup)
module load python/3.11

# Activate the PERSISTENT virtualenv (created by setup_and_run.sh on login node)
VENV_DIR=$HOME/scratch/dft_diatomics_venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtualenv not found at $VENV_DIR"
    echo "Run 'bash setup_and_run.sh' on the login node first!"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Work in the job directory
WORK_DIR=$HOME/scratch/dft_diatomics
cd "$WORK_DIR"

# Make sure data/figures directories exist
mkdir -p data figures

echo "Python: $(which python)"
echo "PySCF version: $(python -c 'import pyscf; print(pyscf.__version__)')"
echo ""

# Run the scan (25 points, wider range, UKS for all)
python diatomic_scan.py \
    --basis def2-svp \
    --level pbe \
    --n-points 25 \
    --output data/diatomic_curves.npz

echo ""
echo "=== Scan complete ==="
echo "End: $(date)"

# Run the adaptive vs direct comparison on the DFT data
echo ""
echo "=== Running ML comparison ==="
MPLBACKEND=Agg python diatomic_adaptive_vs_direct.py --data data/diatomic_curves.npz

echo ""
echo "=== All done ==="
echo "End: $(date)"
