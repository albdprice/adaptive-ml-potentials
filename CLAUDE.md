# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring **Adaptive Homogeneity Constraints for Machine Learning Interatomic Potentials**, a collaboration with Anatole von Lilienfeld (U of T / Vector Institute).

**Core Hypothesis**: Learning adaptive parameters (like exchange mixing alpha in aPBE0) is simpler than learning energy directly. For interatomic potentials, this means learning Rose/UBER parameters (E_c, r_e, l) instead of V(r).

## Current Status (Feb 2025)

**Completed work:**
- Rose/UBER synthetic experiments (Morse potentials): 2.3x extrapolation advantage
- Lennard-Jones synthetic experiments (fixed grid): 42x extrapolation advantage
- Learning curves on Cedar cluster for both systems (Ridge, 5 seeds, 10 sizes)
- Extended analyses: polynomial features, force prediction, DFT diatomics
- DFT diatomics validated on 30 real molecules (PBE/def2-SVP): LOO 1.15x, LGO 1.74x
- Methods section, paper outline, and all figures drafted

**Archived**: k_eff method in `morse_exploration/` (abandoned due to singularity issues)

## Repository Structure

```
adaptive_paper_anatole/
├── README.md
├── CLAUDE.md                    # This file
├── homogeneity_research_plan.tex
├── other_notes.tex
│
├── rose_uber/                   # Rose/UBER approach (Morse potentials)
│   ├── rose_uber_experiment.py  # 5-experiment demonstration
│   ├── refined_ml_demo.py       # 6-panel ML comparison
│   ├── generate_tables.py       # Numerical tables
│   ├── virial_plot_demo.py      # Virial plots
│   ├── learning_curves_compute.py + _plot.py  # Cluster + local
│   └── run_learning_curves.sh   # SLURM for Cedar
│
├── lennard_jones/               # LJ parallel experiments
│   ├── lj_experiment.py         # 5-experiment demonstration
│   ├── lj_refined_ml_demo.py    # 6-panel ML comparison
│   ├── lj_generate_tables.py    # Numerical tables
│   ├── lj_learning_curves_compute.py + _plot.py
│   └── run_lj_learning_curves.sh
│
├── publication/                 # Publication-ready materials
│   ├── methods.md               # Detailed methods section
│   ├── key_findings_and_notes.md
│   ├── data/                    # Learning curve .npz results
│   ├── figures/                 # Rose + LJ publication figures
│   ├── tables/                  # Formatted numerical tables
│   └── extended/
│       ├── polynomial_features/ # Degree-6 poly can't beat physics
│       ├── force_prediction/    # Analytic forces from adaptive
│       └── dft_diatomics/       # Real DFT: 30 diatomics
│           ├── diatomic_scan.py
│           ├── diatomic_adaptive_vs_direct.py
│           ├── run_dft_diatomics.sh
│           └── data/diatomic_curves.npz
│
└── morse_exploration/           # ARCHIVED - k_eff approach
```

## Key Concepts

### Euler's Homogeneous Function Theorem

For homogeneous functions of degree d: `r * dV/dr = d * V`

- **Homogeneous** (C6, Coulomb): d is constant, Virial plot is linear
- **Non-homogeneous** (Morse, LJ, DFT): Virial plot forms loops

### Rose/UBER Equation

Universal binding energy relation:
```
E(a*) = -E_c * (1 + a*) * exp(-a*)
where a* = (r - r_e) / l
```

Parameters: E_c (cohesive energy), r_e (equilibrium distance), l (length scale)

### ML Approach

**Direct**: descriptor -> V(r)
**Adaptive**: descriptor -> (E_c, r_e, l) -> V(r) via physics

## Running Experiments

```bash
export MPLBACKEND=Agg

# Rose/UBER
cd rose_uber
python rose_uber_experiment.py
python refined_ml_demo.py
python generate_tables.py

# Lennard-Jones
cd lennard_jones
python lj_experiment.py
python lj_refined_ml_demo.py

# DFT diatomics
cd publication/extended/dft_diatomics
python diatomic_adaptive_vs_direct.py  # demo mode
python diatomic_adaptive_vs_direct.py --data data/diatomic_curves.npz  # real DFT
```

## Critical Methodological Notes

- **Ridge, not KRR**: KRR with RBF kernel cannot extrapolate (kernel -> 0). Ridge is required.
- **Fixed r-grid for LJ**: sigma-scaled grid trivializes the problem. Fixed grid preserves nonlinear sigma dependence.
- **UKS for DFT diatomics**: RKS fails at dissociation for singlets. Always use UKS.
- **Scaled coordinates for DFT**: r_scaled = r / r_cov_sum for common grid across molecules.

## Connection to aPBE0

| aPBE0 (Khan, Price et al. 2025) | This Work |
|--------------------------------|-----------|
| Learn alpha (exchange mixing) | Learn (E_c, r_e, l) |
| alpha is bounded [0,1] | Parameters are bounded |
| E = (1-alpha)E_PBE + alpha*E_HF | V = E_c * f(a*) |

Same principle: bounded, smooth parameters + physics equation beats direct learning.

## Dependencies

```
numpy, scipy, matplotlib, sklearn, pandas
pyscf (for DFT diatomics only)
```

## Compute Canada (Cedar)

- Account: `def-anatole`
- `module load python/3.11`
- `pip install --no-index` (uses CC wheelhouse)
- `$SLURM_TMPDIR` for virtualenvs
