# Key Findings and Notes for Paper Writing

## Summary of Quantitative Results

### Rose/UBER (Morse potentials)

**Extrapolation learning curves** (Ridge regression, 5 seeds):
- Direct: MAE decays as N^(-0.30), plateaus at ~0.14 eV
- Adaptive: MAE decays as N^(-0.57), reaches ~0.06 eV at N=160
- **Adaptive is 2.3x better at N=160** (extrapolation regime)
- At small N (10-15), both approaches are comparable
- Crossover occurs around N=20

**Interpolation learning curves**:
- Direct wins for interpolation: ~0.009 eV vs adaptive ~0.022 eV at N=160
- This is because the Rose fit introduces a systematic approximation error (~0.02 eV floor)
- The Morse curve and Rose curve are not identical -- the fitting has residual error
- **Key point for paper**: Adaptive's advantage is specifically for extrapolation

**Data efficiency** (fixed seed=42, single run):
| N_train | Direct R^2 | Adaptive R^2 | MSE Improvement |
|---------|-----------|-------------|-----------------|
| 10      | 0.969     | 0.983       | 43.5%           |
| 40      | 0.983     | 0.997       | 84.3%           |
| 80      | 0.983     | 0.999       | 94.2%           |
| 160     | 0.985     | 0.999       | 95.0%           |

**Interpolation vs extrapolation** (N=100):
| Regime                  | Direct R^2 | Adaptive R^2 |
|------------------------|-----------|-------------|
| Interpolation (center) | 0.9992    | 0.9993      |
| Mild extrapolation     | 0.9930    | 0.9992      |
| Strong extrapolation   | 0.9880    | 0.9991      |

**Noise robustness** (N=100, extrapolation):
- Direct R^2 drops from 0.982 (0% noise) to 0.973 (20% noise)
- Adaptive R^2 drops from 0.999 (0% noise) to 0.984 (20% noise)
- Both maintain high R^2, but adaptive degrades less

**Parameter prediction** (extrapolation):
- E_c: R^2 = 0.998
- r_e: R^2 = 1.000
- l:   R^2 = 0.986


### Lennard-Jones

**Extrapolation learning curves** (Ridge, 5 seeds, fixed r-grid):
- Direct: MAE ~8-9 eV, essentially flat (N^(-0.03))-- no learning at all!
- Adaptive: MAE decays as N^(-0.94), from 2.5 eV (N=10) to 0.22 eV (N=160)
- **Adaptive is 42x better at N=160** (extrapolation regime)
- Direct cannot learn because LJ depends nonlinearly on sigma at fixed r-points, and Ridge is linear

**Interpolation learning curves** (fixed r-grid):
- Direct: MAE ~0.02 eV, flat (N^(-0.04))
- Adaptive: MAE decays as N^(-0.98), reaches 0.0003 eV at N=160
- **Adaptive is 63x better at N=160** even for interpolation
- This is because even within the training range, the nonlinear sigma dependence is hard for Ridge

**Data efficiency** (fixed seed=42, single run):
| N_train | Direct R^2 | Adaptive R^2 | MSE Improvement |
|---------|-----------|-------------|-----------------|
| 10      | 0.105     | 0.916       | 90.6%           |
| 40      | 0.108     | 0.991       | 99.0%           |
| 80      | 0.118     | 0.998       | 99.8%           |
| 160     | 0.110     | 0.999       | 99.9%           |

**Interpolation vs extrapolation** (N=100):
| Regime                  | Direct R^2 | Adaptive R^2 |
|------------------------|-----------|-------------|
| Interpolation (center) | 0.931     | 1.000       |
| Mild extrapolation     | 0.575     | 1.000       |
| Strong extrapolation   | 0.211     | 0.999       |

**Noise robustness** (N=100, extrapolation):
- Direct R^2 stuck at ~0.10 regardless of noise (already failing)
- Adaptive R^2 stays at 0.998+ for all noise levels up to 20%

**Parameter prediction** (extrapolation):
- epsilon: R^2 = 1.000 (MAE = 0.0003 eV)
- sigma:   R^2 = 0.998 (MAE = 0.011 A)


## Key Messages for the Paper

### 1. The central finding
Learning universal scaling parameters + physics equations beats direct energy learning for extrapolation. This is the same principle as aPBE0 (learning alpha instead of E_xc).

### 2. Why it works -- the physics argument
Non-homogeneous potentials (Morse, LJ) have material-specific parameters that modulate a universal functional form. By learning parameters instead of energies, we:
- Reduce the dimensionality of the ML target (2-3 numbers vs 50-dimensional curve)
- Exploit the universal functional form for physically correct extrapolation
- Maintain correct asymptotic behavior (repulsion, decay) by construction

### 3. Complementary evidence from two systems
- **Rose/UBER**: More subtle advantage (2.3x). The Rose fitting step introduces approximation error, so adaptive has a floor. But the extrapolation advantage is clear and grows with N.
- **Lennard-Jones**: Dramatic advantage (42x). The nonlinear sigma dependence at fixed r-points makes direct learning fundamentally limited with a linear model. Adaptive bypasses this entirely.

### 4. The r-grid matters (important methodological point)
For LJ, using a sigma-scaled grid (r proportional to sigma) trivializes the problem for direct learning by making sigma/r constant. A fixed r-grid is the physically correct choice -- in real simulations, atomic distances don't rescale with potential parameters.

### 5. Ridge vs KRR (important negative result)
KRR with RBF kernel fundamentally cannot extrapolate: the kernel decays to zero outside the training range, so predictions collapse to the training mean. We verified this on Cedar (both approaches gave ~3.2 eV MAE, ratio 1.0x, no improvement with N). Ridge is the honest choice for demonstrating extrapolation.

### 6. When does adaptive NOT help?
- **Interpolation with Morse/Rose**: Direct slightly wins because the Rose fitting introduces systematic error (~0.02 eV floor). If the physics equation were exact, adaptive would also win here.
- **Trivially linear problems**: If the descriptor-to-curve mapping is already linear (e.g., sigma-scaled LJ grid), any linear model can learn it directly.

### 7. Connection to aPBE0
| Aspect     | aPBE0            | Rose/UBER              | Lennard-Jones        |
|-----------|-----------------|----------------------|---------------------|
| Parameter | alpha (mixing)   | (E_c, r_e, l)         | (epsilon, sigma)     |
| Equation  | E=(1-a)E_PBE+aE_HF | Rose/UBER equation  | LJ equation          |
| Bounded?  | Yes [0,1]        | Yes (positive, finite) | Yes (positive, finite) |
| Smooth?   | Yes              | Yes (linear in d)      | Yes (linear in d)     |


## Figure Inventory (publication/figures/)

### Rose/UBER figures
| File | Description | Use |
|------|-------------|-----|
| `fig2_universal_collapse.png` | Raw Morse curves -> Rose collapse | Fig 1 or 2: motivation |
| `fig3_parameter_smoothness.png` | Rose params vary smoothly with descriptors | Fig 2 or 3: smoothness |
| `fig4_ml_comparison.png` | Direct vs adaptive example predictions | Fig 3: qualitative |
| `fig_refined_ml_demo.png` | 6-panel: data eff, interp/extrap, noise, params | Main results figure |
| `fig_learning_curves_combined.png` | 3-panel learning curves with power laws | Main LC figure |
| `fig_virial_plots.png` | Virial plots: homogeneous vs non-homogeneous | Theory figure |

### Lennard-Jones figures
| File | Description | Use |
|------|-------------|-----|
| `fig2_lj_universal_collapse.png` | Raw LJ curves -> V/eps vs r/sigma collapse | Parallel to Rose |
| `fig3_lj_parameter_smoothness.png` | LJ params vary smoothly with descriptors | Parallel to Rose |
| `fig4_lj_ml_comparison.png` | Direct vs adaptive (zoomed to show wells) | Qualitative |
| `fig_lj_refined_ml_demo.png` | 6-panel: all experiments | Main results figure |
| `fig_lj_learning_curves_combined.png` | 3-panel learning curves | Main LC figure |
| `fig_lj_virial.png` | LJ virial plot shows loops | Theory figure |


### Extended analysis figures (publication/extended/)
| File | Description | Use |
|------|-------------|-----|
| `polynomial_features/figures/fig_poly_features_lj.png` | 4-panel: poly degree sweep, learning curves, example | SI: model capacity argument |
| `force_prediction/figures/fig_force_analysis.png` | 6-panel: energy + force for LJ and Rose | SI or main: force prediction |
| `force_prediction/figures/fig_force_grid_sensitivity.png` | Force MAE vs grid spacing | SI: grid independence |
| `dft_diatomics/figures/fig_diatomic_comparison.png` | 6-panel: LOO + LGO on diatomics | SI or main: real systems |


## Suggested Paper Outline

1. **Introduction**
   - ML potentials are data-hungry and extrapolate poorly
   - aPBE0 showed that learning parameters instead of energies works
   - We apply the same principle to interatomic potentials via universal scaling

2. **Theory**
   - Euler's theorem, homogeneity, virial plots
   - Non-homogeneous potentials and universal scaling (Rose, LJ)
   - Adaptive parameter learning hypothesis

3. **Methods**
   - See methods.md (ready to go)

4. **Results**
   - A. Universal collapse demonstration (Rose + LJ)
   - B. Data efficiency (learning curves)
   - C. Extrapolation vs interpolation
   - D. Noise robustness
   - E. Parameter interpretability

5. **Discussion**
   - Why adaptive works (dimensionality, smoothness, physics)
   - When it doesn't (interpolation with approximate physics)
   - Connection to aPBE0
   - Toward real systems (SOAP/ACE descriptors, DFT data)

6. **Conclusion**


## Extended Analyses (publication/extended/)

### 1. Polynomial Features (Model Capacity vs Physics)

**Key question**: Is the adaptive advantage about physics or model capacity?

**Results** (LJ, extrapolation, N=100):
| Poly Degree | Direct MAE [eV] | # Features | vs Adaptive (0.39 eV) |
|:-----------:|:---------------:|:----------:|:---------------------:|
| 1 (linear)  | 9.63            | 2          | 24.8x worse           |
| 2           | 8.79            | 5          | 22.6x worse           |
| 3           | 7.42            | 9          | 19.1x worse           |
| 4           | 5.68            | 14         | 14.6x worse           |
| 5           | 3.24            | 20         | 8.3x worse            |
| 6           | 0.67            | 27         | 1.7x worse            |

**At N=5000**: Direct (degree 4) still at 3.77 eV MAE vs adaptive at 0.006 eV.

**For interpolation**: Degree 5-6 polynomials match adaptive, confirming the advantage is specifically about extrapolation physics.

**Key message**: Even 27 polynomial features cannot match 2 linear features + physics equation for extrapolation. The advantage is the correct functional form, not model capacity.

### 2. Force Prediction (Free Analytic Forces)

Adaptive provides analytic forces (F = -dV/dr with predicted parameters); direct requires numerical differentiation.

**LJ Results** (N=100, extrapolation):
| Metric    | Direct         | Adaptive        |
|-----------|---------------|-----------------|
| Force MAE | 7.83 eV/A     | 0.32 eV/A       |
| Force R2  | 0.109         | 0.998           |

**Grid spacing sensitivity**: Adaptive force accuracy is grid-independent (analytic derivative), while direct depends on dr (numerical diff error + prediction error). This is a "free" bonus of the adaptive approach.

**Note**: For Rose/UBER, the force comparison is confounded by Rose != Morse (systematic fitting error dominates in derivatives). The clean force demonstration is LJ where the physics equation is exact.

### 3. DFT Diatomics (Real Systems Validation)

**Real DFT** (30 diatomics, UKS-PBE/def2-SVP, 25 r-points, r_max=5*r_eq):
- 30/30 molecules converged, 28 used (H2 excluded: Rose RMSE=1.12 eV; N2 excluded: RMSE=2.09 eV)
- 14 atomic pair descriptors: Z, EN, r_cov, IE for each atom + sums/diffs
- Common grid: r_scaled = r / r_cov_sum (scaled overlap width = 3.11)
- Ridge regression with StandardScaler, alpha=1.0

**LOO (interpolation, 28 molecules):**
- Direct MAE = 0.451 eV, Adaptive MAE = 0.393 eV (**Adaptive wins 1.15x**)
- Adaptive wins on 18/28 molecules

**LGO (extrapolation, train rows 1-2, test row 3+):**
- Direct MAE = 0.821 eV, Adaptive MAE = 0.471 eV (**Adaptive wins 1.74x**)
- Adaptive wins on 11/13 test molecules
- Standouts: KH 13.8x, NaH 4.4x, PH 3.3x

**Rose fit quality vs experiment:**
- D_e: MAE = 0.31 eV (PBE overbinding expected)
- r_e: MAE = 0.053 A

**To reproduce**: `sbatch run_dft_diatomics.sh` on Cedar (12h), or `python diatomic_adaptive_vs_direct.py --data data/diatomic_curves.npz` locally with existing data.


## Data Files (publication/data/)

| File | Description | Generated by |
|------|-------------|-------------|
| `rose_uber_learning_curves.npz` | Ridge LC results, 5 seeds, 10 train sizes | `rose_uber/learning_curves_compute.py` |
| `lj_learning_curves.npz` | Ridge LC results (fixed grid), 5 seeds, 10 sizes | `lennard_jones/lj_learning_curves_compute.py` |


## Reproduction Commands

```bash
# Rose/UBER
cd rose_uber/
MPLBACKEND=Agg python rose_uber_experiment.py     # Figs 1-5
MPLBACKEND=Agg python refined_ml_demo.py           # 6-panel figure
MPLBACKEND=Agg python generate_tables.py           # Tables 1-4
MPLBACKEND=Agg python virial_plot_demo.py          # Virial plots
python learning_curves_compute.py --n-seeds 5      # Learning curves data
MPLBACKEND=Agg python learning_curves_plot.py      # Learning curves figures

# Lennard-Jones
cd lennard_jones/
MPLBACKEND=Agg python lj_experiment.py             # Figs 1-5
MPLBACKEND=Agg python lj_refined_ml_demo.py        # 6-panel figure
MPLBACKEND=Agg python lj_generate_tables.py        # Tables 1-4
MPLBACKEND=Agg python lj_virial_plot.py            # Virial plots
python lj_learning_curves_compute.py --n-seeds 5   # Learning curves data
MPLBACKEND=Agg python lj_learning_curves_plot.py   # Learning curves figures
```

For cluster (Cedar):
```bash
cd rose_uber/ && sbatch run_learning_curves.sh
cd lennard_jones/ && sbatch run_lj_learning_curves.sh
```

# Extended analyses
```bash
cd publication/extended/polynomial_features/
MPLBACKEND=Agg python poly_features_lj.py

cd publication/extended/force_prediction/
MPLBACKEND=Agg python force_analysis.py

cd publication/extended/dft_diatomics/
pip install pyscf                                # if not installed
python diatomic_scan.py                          # compute DFT curves (~hours)
MPLBACKEND=Agg python diatomic_adaptive_vs_direct.py  # demo mode (no DFT needed)
MPLBACKEND=Agg python diatomic_adaptive_vs_direct.py --data data/diatomic_curves.npz  # with DFT data
```
