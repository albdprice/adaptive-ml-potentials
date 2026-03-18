# Lennard-Jones Approach for Adaptive Homogeneity

This directory contains the **Lennard-Jones** demonstration of adaptive parameter learning, paralleling the Rose/UBER approach for Morse potentials.

## Theoretical Background

### Lennard-Jones Potential

```
V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
```

Parameters:
- **epsilon**: well depth (energy scale, eV)
- **sigma**: zero-crossing distance (length scale, Angstrom)

### Universal Reduced Form

```
V*(r*) = 4 * [(1/r*)^12 - (1/r*)^6]
```
where `r* = r/sigma` and `V* = V/epsilon`.

All LJ potentials collapse to ONE universal curve when scaled by (epsilon, sigma).

### Why This Works (ML Perspective)

Instead of learning V(r) directly (50-dimensional curve), learn 2 parameters:
- **epsilon**: well depth
- **sigma**: atomic/molecular size

The LJ equation then provides:
- Correct r^-12 repulsion and r^-6 attraction
- Proper extrapolation beyond training data
- Physical interpretability

## Files

### Core Scripts

| File | Description |
|------|-------------|
| `lj_experiment.py` | Main 5-experiment demonstration |
| `lj_refined_ml_demo.py` | Comprehensive ML comparison (4 experiments) |
| `lj_generate_tables.py` | Numerical results tables |
| `lj_virial_plot.py` | Virial plots (homogeneous vs non-homogeneous) |

### Learning Curves (for cluster)

| File | Description |
|------|-------------|
| `lj_learning_curves_compute.py` | Computation (KRR + CV), saves .npz |
| `lj_learning_curves_plot.py` | Plotting from .npz results |
| `run_lj_learning_curves.sh` | SLURM script for Cedar |

### Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1_lj_potentials.png` | LJ potentials with varying (epsilon, sigma) |
| `fig2_lj_universal_collapse.png` | Universal collapse when scaled |
| `fig3_lj_parameter_smoothness.png` | Parameters vary smoothly |
| `fig4_lj_ml_comparison.png` | ML extrapolation comparison |
| `fig5_lj_apbe0_analogy.png` | Connection to aPBE0 |
| `fig_lj_refined_ml_demo.png` | 6-panel comprehensive ML results |
| `fig_lj_virial.png` | Virial plots (Coulomb/dispersion vs LJ) |
| `fig_lj_learning_curves.png` | MAE learning curves |
| `fig_lj_learning_curves_combined.png` | 3-panel combined learning curves |

## Running the Experiments

```bash
export MPLBACKEND=Agg

# Main demonstration
python lj_experiment.py

# Comprehensive ML comparison
python lj_refined_ml_demo.py

# Numerical tables
python lj_generate_tables.py

# Virial plots
python lj_virial_plot.py

# Learning curves (local quick test)
python lj_learning_curves_compute.py --n-seeds 1

# Learning curves (cluster)
sbatch run_lj_learning_curves.sh
```

## Connection to Other Approaches

| Approach | Adaptive Parameters | Physics Equation |
|----------|-------------------|------------------|
| aPBE0 | alpha (exchange mixing) | E = (1-alpha)E_PBE + alpha*E_HF |
| Rose/UBER | (E_c, r_e, l) | E = E_c * (1 - (1+a*)exp(-a*)) |
| **LJ** | **(epsilon, sigma)** | **V = 4eps[(sig/r)^12 - (sig/r)^6]** |

Same principle: bounded, smooth parameters + physics equation beats direct learning.

## Dependencies

```
numpy, scipy, matplotlib, scikit-learn, pandas
```
