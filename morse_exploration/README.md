# Morse Exploration (Archived Approach)

**Status:** ARCHIVED - See `rose_uber/` for current active approach

## Overview

This directory contains the initial exploration using the **k_eff approach** for Morse potentials. This approach was later superseded by the Rose/UBER approach due to numerical issues with singularities.

## The k_eff Concept

Defined the "effective homogeneity degree":
```
k_eff(r) = r·V'(r) / V(r)
```

For homogeneous functions, k_eff = d (constant). For Morse, k_eff varies with r.

### Analytical Result for Morse

For Morse potential in reduced coordinates ρ = r/r_e:
```
k_eff(ρ; α) = 2α·ρ·exp(-α(ρ-1)) / (1 - exp(-α(ρ-1)))
```

where α = a·r_e is the dimensionless stiffness parameter.

**Key finding:** k_eff depends only on (ρ, α), NOT on D_e! This is dimensionality reduction from 3 parameters to 1.

## Why This Approach Was Abandoned

1. **Singularity at ρ=1**: k_eff has a simple pole at the equilibrium distance where V(r_e) = 0. This causes numerical instabilities.

2. **Integration recovery fails**: Recovering V(r) from k_eff via integration:
   ```
   V(r) = V(r_0) · exp(∫ k_eff(x)/x dx)
   ```
   accumulates large errors near the singularity.

3. **ML comparison unclear**: The figures generated were confusing and didn't clearly demonstrate the ML advantage.

## Files in This Directory

| File | Description |
|------|-------------|
| `explore_adaptive_homogeneity.ipynb` | Initial Jupyter exploration |
| `demo_for_anatole.ipynb` | Demo notebook (incomplete) |
| `adaptive_homogeneity_experiment.py` | Python script with 4 experiments |
| `train_set.csv`, `test_set.csv` | Generated datasets |
| `fig_*.png` | Various figures (mostly deprecated) |

## Key Experiments Attempted

1. **Euler Violation**: Showed Morse is non-homogeneous (Virial loops)
2. **Corresponding States**: Showed k_eff collapses in reduced coordinates
3. **Alpha Dependence**: k_eff depends only on α, not D_e
4. **ML Comparison**: Attempted but results were unclear

## Lessons Learned

1. The k_eff concept is mathematically elegant but numerically problematic
2. The Rose/UBER equation avoids the singularity issues
3. Rose parameters (E_c, r_e, l) are easier to learn than k_eff(r)
4. Universal scaling exists in both formulations

## Moving Forward

The **Rose/UBER approach** in `../rose_uber/` addresses these issues:
- No singularities (Rose equation is well-behaved everywhere)
- Clear physical parameters (E_c, r_e, l)
- Strong ML results (90%+ improvement on extrapolation)

See `../rose_uber/README.md` for the current active approach.
