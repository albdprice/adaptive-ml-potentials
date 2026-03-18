# Rose/UBER Approach for Adaptive Homogeneity

This directory contains the **current active approach** using the Rose/UBER equation for adaptive parameter learning.

## Theoretical Background

### The Problem: Non-Homogeneous Potentials

Real interatomic potentials (Morse, LJ, DFT binding curves) are **not homogeneous**. Euler's theorem `r·F = d·V` doesn't hold with a constant d. The Virial plot (r·F vs V) forms **loops**, not straight lines.

### The Solution: Rose/UBER Universal Scaling

The Rose equation provides a universal binding energy relation:
```
E(a*) = -E_c · (1 + a*) · exp(-a*)
```
where `a* = (r - r_e) / l` is the reduced coordinate.

**Key insight:** All binding curves collapse to ONE universal shape when scaled by (E_c, r_e, l).

### Why This Works (ML Perspective)

Instead of learning V(r) directly (high-dimensional, complex), learn 3 parameters:
- **E_c**: Cohesive/binding energy
- **r_e**: Equilibrium distance
- **l**: Length scale (related to bulk modulus)

The physics equation then provides:
- Correct asymptotic behavior (repulsion at short range, decay at long range)
- Proper extrapolation beyond training data
- Physical interpretability

## Files in This Directory

### Core Scripts

| File | Description |
|------|-------------|
| `rose_uber_experiment.py` | Main 5-experiment demonstration |
| `refined_ml_demo.py` | Comprehensive ML comparison with 4 tests |
| `generate_tables.py` | Generate detailed numerical tables |
| `ml_comparison_demo.py` | Multi-system ML comparison |
| `complete_framework_demo.py` | Euler's theorem visualization |
| `virial_plot_demo.py` | Virial plots showing homo vs non-homo |
| `other_potentials_demo.py` | C6, LJ comparisons |
| `adaptive_parameters_demo.py` | Earlier parameter demo (archived) |

### Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1_rose_fits_morse.png` | Rose equation fits Morse potentials |
| `fig2_universal_collapse.png` | All curves collapse to one universal shape |
| `fig3_parameter_smoothness.png` | Parameters vary smoothly with physics |
| `fig4_ml_comparison.png` | ML extrapolation comparison |
| `fig5_apbe0_analogy.png` | Connection to aPBE0 |
| `fig_refined_ml_demo.png` | Comprehensive 6-panel ML results |
| `fig_complete_framework.png` | Euler's theorem: homo vs non-homo |
| `fig_virial_plots.png` | Virial plots for different potentials |
| `fig_c6_virial.png` | C6 dispersion linear scaling |

### Data/Reference Files

| File | Description |
|------|-------------|
| `sciadv.adt7769-6.pdf` | aPBE0 paper (Khan, Price et al. 2025) |
| `4EC002CF-...jpeg` | Anatole's chalkboard notes |

## Key Results

### Table 1: Data Efficiency (Extrapolation)

Training: De ~ 1.5-3.5 eV | Testing: De ~ 5.5-8.5 eV

| N_train | Direct R² | Adaptive R² | MSE Improvement |
|---------|-----------|-------------|-----------------|
| 10 | 0.9693 | 0.9826 | +43.5% |
| 20 | 0.9767 | 0.9942 | +75.0% |
| 40 | 0.9829 | 0.9973 | +84.3% |
| 80 | 0.9830 | 0.9990 | +94.2% |
| 160 | 0.9845 | 0.9992 | +95.0% |

### Table 2: Interpolation vs Extrapolation

Training: d1 ~ 1.0-2.5

| Test Regime | Direct R² | Adaptive R² |
|-------------|-----------|-------------|
| Interpolation (center) | 0.9992 | 0.9993 |
| Mild extrapolation | 0.9930 | 0.9992 |
| Strong extrapolation | 0.9880 | 0.9991 |

### Table 3: Noise Robustness

| Noise Level | Direct R² | Adaptive R² |
|-------------|-----------|-------------|
| 0% | 0.9818 | 0.9990 |
| 5% | 0.9800 | 0.9987 |
| 10% | 0.9799 | 0.9989 |
| 15% | 0.9753 | 0.9965 |
| 20% | 0.9731 | 0.9838 |

### Table 4: Parameter Prediction Quality

| Parameter | R² Score | Physical Meaning |
|-----------|----------|------------------|
| E_c | 0.9976 | Binding/cohesive energy |
| r_e | 0.9999 | Equilibrium distance |
| l | 0.9857 | Length scale (stiffness) |

## Running the Experiments

```bash
# Set matplotlib backend for non-interactive use
export MPLBACKEND=Agg

# Run main demonstration
python rose_uber_experiment.py

# Run comprehensive ML comparison
python refined_ml_demo.py

# Generate detailed tables
python generate_tables.py

# Visualize Euler's theorem
python complete_framework_demo.py
python virial_plot_demo.py
```

## Key Conclusions

1. **Homogeneous potentials** (C6, Coulomb): Euler's theorem holds, d is constant, both ML approaches work equally well.

2. **Non-homogeneous potentials** (Morse, LJ, DFT): Virial plot forms loops. Adaptive approach wins on:
   - Extrapolation (90%+ improvement)
   - Data efficiency (~50% less data needed)
   - Noise robustness
   - Interpretability

3. **aPBE0 Connection**: Same principle - learn bounded, smooth parameters instead of raw energy. Physics equation handles the complexity.

## Next Steps

1. ~~Test on real DFT binding curves~~ -- DONE (see `publication/extended/dft_diatomics/`)
2. Use proper atomic descriptors (SOAP, ACE) instead of simple pair features
3. Extend to multi-element / polyatomic systems
4. Write paper with Anatole
