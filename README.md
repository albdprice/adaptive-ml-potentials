# Adaptive Machine Learning of Universal Scaling Parameters for Interatomic Potentials

**A. J. A. Price** and **O. A. von Lilienfeld**

University of Toronto / Acceleration Consortium

## Summary

This work applies the adaptive parameter learning strategy — first established for density functionals with aPBE0 (Khan, Price et al., *Sci. Adv.* 2025) — to interatomic potentials. Rather than fitting V(r) directly, I train ML models to predict the universal scaling parameters of known analytical forms (Rose/UBER or Lennard-Jones), then reconstruct the full potential via the physics equation.

The key insight is dimensionality reduction: a 50-point potential energy curve is reduced to 2–3 physically meaningful parameters that are bounded and smooth. The hard nonlinearities ((σ/r)¹² for LJ, exp(-a*) for Rose) are handled exactly by the equation rather than learned approximately.

### Results

| System | Extrapolation Advantage |
|:---|:---|
| Morse (synthetic, Rose/UBER) | 2.3× |
| Lennard-Jones (synthetic, fixed grid) | 42× |
| DFT diatomics (30 molecules, PBE/def2-SVP) | 1.74× (leave-group-out) |
| Forces (DFT diatomics) | 3.35× (leave-group-out) |

A comprehensive MLP baseline sweep (56 architectures, 7 widths × 4 activations) confirms the advantage stems from the physics equation, not the model class — adaptive Ridge regression outperforms the best direct MLP (SiLU activation).

## Repository Layout

- `rose_uber/` — Rose/UBER experiments on Morse potentials
- `lennard_jones/` — Parallel LJ experiments with fixed radial grid
- `publication/` — Manuscript (v2), methods, figures, and extended analyses:
  - `extended/mlp_baseline/` — 56-config MLP sweep
  - `extended/dft_diatomics/` — Real DFT validation (PySCF, 30 diatomics, UKS)
  - `extended/force_prediction/` — Analytical forces at no additional cost
  - `extended/delta_learning/` — Delta-learning extension (2.7× further improvement for Rose)
  - `extended/position_dependent/` — Stiffness and screening function analysis
- `morse_exploration/` — Archived: k_eff approach (abandoned, singularity issues)
- `anatole_expanded/` — Exploratory scripts from discussions with Anatole

## Running

```bash
export MPLBACKEND=Agg
pip install numpy scipy matplotlib scikit-learn pandas

cd rose_uber && python rose_uber_experiment.py
cd ../lennard_jones && python lj_experiment.py
cd ../publication/extended/dft_diatomics && python diatomic_adaptive_vs_direct.py --data data/diatomic_curves.npz
```

Learning curves are computationally expensive and should be run on a cluster (SLURM scripts included).

## Methodological Notes

- **Ridge, not KRR**: RBF kernels decay to zero outside the training range, making KRR unable to extrapolate. Ridge is the correct baseline for extrapolation comparisons.
- **Fixed r-grid for LJ**: A σ-scaled grid trivializes the problem. Fixed grid r = linspace(2.5, 12.0, 50) preserves the nonlinear σ dependence.
- **UKS for DFT diatomics**: RKS fails at dissociation for singlet states.
- **Scaled coordinates**: r_scaled = r / r_cov_sum for a common grid across different molecules.

## Connection to aPBE0

The same principle operates in both works: aPBE0 learns α (exchange mixing) rather than the total energy; here I learn (E_c, r_e, l) or (ε, σ) rather than V(r). Bounded, physically meaningful parameters plus exact physics equations consistently outperform direct learning.

## Dependencies

numpy, scipy, matplotlib, scikit-learn, pandas. PySCF for the DFT diatomic calculations only.
