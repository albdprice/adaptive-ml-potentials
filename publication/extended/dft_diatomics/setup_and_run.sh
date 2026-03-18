#!/bin/bash
# Setup script for Nibi/Alliance Canada cluster
# RUN THIS ON THE LOGIN NODE (needs internet for pip install pyscf)
#
# Usage:
#   cd ~/scratch/dft_diatomics
#   bash setup_and_run.sh
#
# Then submit:
#   sbatch run_dft_diatomics.sh

set -e

echo "=== Setting up DFT diatomics environment on $(hostname) ==="

# Load modules
module load python/3.11

# Create virtualenv in scratch (persistent across jobs)
VENV_DIR=$HOME/scratch/dft_diatomics_venv

if [ -d "$VENV_DIR" ]; then
    echo "Removing old virtualenv..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtualenv at $VENV_DIR..."
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install dependencies
# numpy/scipy/sklearn come from the scipy-stack module at runtime,
# but PySCF needs its own numpy at install time
echo ""
echo "Installing PySCF and dependencies..."
pip install --upgrade pip
pip install numpy scipy
pip install pyscf
pip install scikit-learn matplotlib

# Verify
echo ""
echo "=== Verifying installation ==="
python -c "
import pyscf; print(f'PySCF:       {pyscf.__version__}')
import numpy; print(f'NumPy:       {numpy.__version__}')
import scipy; print(f'SciPy:       {scipy.__version__}')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import matplotlib; print(f'Matplotlib:  {matplotlib.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo "Virtualenv: $VENV_DIR"
echo "Python: $(which python)"
echo ""
echo "Now submit the job with:"
echo "  sbatch run_dft_diatomics.sh"
