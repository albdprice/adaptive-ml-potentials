# Position-Dependent Adaptive Parameters: Methods and Results

## Concept

The global adaptive approach (Rose/UBER) represents a binding curve through 3 scalar parameters (E_c, r_e, l) combined via a nonlinear physics equation. Here we extend this to **position-dependent** parameters: at each grid point r_j, we define local parameters that vary along the binding curve.

## The b=0 Parabola Decomposition

Any Morse potential in the unshifted convention (V(r_e) = 0, V(∞) = D_e) can be exactly decomposed as:

    V(r) = a(r) · (r - r_e)²

where the position-dependent curvature is:

    a(r) = D_e · α² · f(u)²
    f(u) = (1 - exp(-u)) / u
    u = α · (r - r_e)

The function f(u) is universal across all Morse curves — curve-specific information enters only through the scaling factors D_e·α² (amplitude) and α (coordinate stretching). At equilibrium, a(r_e) = D_e·α² (the harmonic force constant).

### Why b=0?

The form a·(r-r_e)² with no linear or constant term:
- Is a **monomial** (homogeneous of degree 2 in displacement), connecting to Euler's theorem
- Has no singularities (unlike the vertex form a·(r-r₀)²+b which diverges at the inflection point where V''=0)
- Requires knowing r_e, which adds a nonlinear reconstruction step

### The sqrt(a) insight

The square root of the curvature, √a(r) = √D_e · α · |f(u)|, is smoother than a(r) itself (smaller gradients on the repulsive wall). Predicting √a and squaring introduces an additional nonlinearity in the reconstruction:

    V(r_j) = [√a_j]² · (r_j - r_e)²

## ML Experiment

### Data generation
- **Descriptors**: (d1, d2) ∈ [0.5, 1.5]² for training
- **Morse parameters**: D_e = 1 + 2·d1, α = 0.8 + 0.6·d2, r_e = 1.5 + 0.5·d1
- **Grid**: Fixed r ∈ [1.0, 8.0] Å, 50 points
- **Extrapolation test**: d1 ∈ [2.5, 4.0] (larger D_e, larger r_e)
- **Interpolation test**: d1 ∈ [0.7, 1.3], d2 ∈ [0.6, 1.4]
- 100 training, 50 test samples

### Methods compared
1. **Direct**: Ridge regression, descriptor → V(r_j) at 50 grid points
2. **Adaptive a(r)**: Ridge → r_e (1 scalar) + a(r_j) (50 values), reconstruct V = a·(r-r_e)²
3. **Adaptive √a(r)**: Ridge → r_e + √a(r_j), reconstruct V = (√a)²·(r-r_e)²

All use Ridge regression (α=1.0) with StandardScaler normalization.

### Results

| Regime         | Direct MAE | Adaptive a(r) | Ratio | Adaptive √a | Ratio |
|----------------|-----------|---------------|-------|-------------|-------|
| Extrapolation  | 181.8 eV  | 155.1 eV      | 1.17× | 132.6 eV    | 1.37× |
| Interpolation  | 0.395 eV  | 0.203 eV      | 1.94× | 0.088 eV    | 4.46× |

### Key findings
- The b=0 decomposition provides a genuine ML advantage through the nonlinear (r-r_e)² reconstruction step
- √a outperforms a in both regimes, confirming the value of the smoother target
- Interpolation advantage (4.5×) is stronger than extrapolation (1.4×)
- r_e is predicted accurately (MAE = 0.011 Å extrap, 0.0007 Å interp) since it is linear in d1

### Comparison to global Rose
The global Rose approach (3 parameters + nonlinear equation) gives ~2.3× extrapolation advantage on similar Morse data. The position-dependent approach has 51 parameters (r_e + 50 a-values) with weaker physics constraints, resulting in a smaller extrapolation advantage but more flexibility. The approaches are complementary: global Rose constrains the entire curve shape with 3 scalars, while position-dependent a(r) allows local flexibility.

## Files
- `visualize_local_parabolas.py` — Step 0: visualize a(r), b(r), c(r) Taylor expansion parameters
- `explore_b0_parabola.py` — Step 0b: b=0 decomposition, universal function f(u)
- `position_dependent_experiment.py` — Step 1a: abc parabola ML (shows linear equivalence for Ridge)
- `b0_ml_experiment.py` — Step 1b: b=0 adaptive ML (the main result)
- `lj_for_morse_experiment.py` — Step 2: LJ as approximate physics for Morse (negative result)

## Step 2: LJ as Approximate Physics for Morse

### Motivation
Anatole suggested testing whether the adaptive approach works with an **imperfect** physics model: fit LJ parameters (ε, σ) to Morse reference curves, then learn ε, σ from descriptors. Since LJ ≠ Morse, there is an irreducible approximation error.

### Setup
- Same Morse curves as Step 1 (same descriptor ranges, grid, train/test splits)
- For each Morse curve, fit LJ parameters (ε, σ) by nonlinear least squares
- Direct: descriptor → V_Morse(r) on grid
- LJ Adaptive: descriptor → (ε, σ) → V_LJ(r) = 4ε[(σ/r)¹² - (σ/r)⁶]

### Results: Negative

| Regime         | Direct MAE | LJ Adaptive MAE | Ratio  | LJ Best-Fit Limit |
|----------------|-----------|-----------------|--------|-------------------|
| Extrapolation  | 181.8 eV  | 4050.3 eV       | 0.04×  | 108.3 eV          |
| Interpolation  | 0.395 eV  | 14.4 eV         | 0.03×  | 0.976 eV          |

LJ adaptive is catastrophically **worse** than direct in both regimes.

### Why LJ fails as approximate physics for Morse
1. **Different repulsive walls**: LJ uses (σ/r)¹² (power law), Morse uses exp(-α·r) (exponential). These diverge in fundamentally different ways.
2. **Error amplification**: The 12th power in LJ amplifies small σ errors catastrophically. A 1 Å error in σ changes V by thousands of eV on the repulsive wall.
3. **Physics floor too high**: Even with perfect LJ parameter prediction, the best-fit LJ approximation error (0.98 eV interp, 108 eV extrap) is already worse than direct Ridge prediction.

### Key lesson
The adaptive approach requires **good enough** approximate physics. Rose works for Morse (same exponential family, physics floor near zero). LJ fails for Morse (wrong functional form, high physics floor). The quality of the physics model is essential — nonlinearity alone is not sufficient.
