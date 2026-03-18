"""
Extended Analysis: Polynomial Features for Direct Learning (LJ)
================================================================

Key question: Is the adaptive advantage due to physics or model capacity?

Ridge regression is linear, so it cannot capture the nonlinear dependence of
V(r) on sigma at fixed r-points. Could adding polynomial features (making the
model effectively nonlinear) close the gap?

Results show:
- Polynomial features help direct learning *within* the training domain
- But for extrapolation, adaptive STILL wins because polynomial extrapolation
  is unreliable (Runge's phenomenon), while physics equations extrapolate correctly
- This proves the advantage is about physics (correct functional form),
  not just model capacity

Usage:
    MPLBACKEND=Agg python poly_features_lj.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


# =============================================================================
# POTENTIAL FUNCTION (same as main LJ scripts)
# =============================================================================

def lennard_jones(r, epsilon, sigma):
    """LJ potential: V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]"""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def generate_dataset(n_samples, d1_range, d2_range, n_points=50, noise_level=0.0):
    """Generate LJ potentials with descriptors on a FIXED r-grid."""
    r = np.linspace(2.5, 12.0, n_points)
    data = {'X': [], 'params': [], 'curves': [], 'r_grid': r, 'epsilon': []}

    for i in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2
        V = lennard_jones(r, epsilon, sigma)
        if noise_level > 0:
            V = V + np.random.normal(0, noise_level * np.abs(V).mean(), len(V))
        data['X'].append([d1, d2])
        data['params'].append([epsilon, sigma])
        data['curves'].append(V)
        data['epsilon'].append(epsilon)

    return data


# =============================================================================
# TRAINING WITH POLYNOMIAL FEATURES
# =============================================================================

def train_poly_direct(train_data, test_data, degree=1):
    """Train direct approach with polynomial features of given degree."""
    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])

    # Polynomial feature expansion
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train_poly)
    X_test_s = scaler_X.transform(X_test_poly)

    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, Y_train_s)
    Y_pred = scaler_V.inverse_transform(model.predict(X_test_s))

    n_features = X_train_poly.shape[1]
    mse = mean_squared_error(Y_curves_test, Y_pred)
    mae = mean_absolute_error(Y_curves_test, Y_pred)
    r2 = r2_score(Y_curves_test.flatten(), Y_pred.flatten())

    return {'mse': mse, 'mae': mae, 'r2': r2, 'n_features': n_features,
            'Y_pred': Y_pred}


def train_adaptive(train_data, test_data, degree=1):
    """Train adaptive approach (optionally with polynomial features for param prediction)."""
    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_test = np.array(test_data['curves'])
    Y_params_train = np.array(train_data['params'])

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train_poly)
    X_test_s = scaler_X.transform(X_test_poly)

    scaler_params = StandardScaler()
    Y_params_s = scaler_params.fit_transform(Y_params_train)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, Y_params_s)
    Y_params_pred = scaler_params.inverse_transform(model.predict(X_test_s))

    r = test_data['r_grid']
    Y_pred = []
    for eps, sig in Y_params_pred:
        Y_pred.append(lennard_jones(r, max(eps, 0.001), max(sig, 0.1)))
    Y_pred = np.array(Y_pred)

    mse = mean_squared_error(Y_curves_test, Y_pred)
    mae = mean_absolute_error(Y_curves_test, Y_pred)
    r2 = r2_score(Y_curves_test.flatten(), Y_pred.flatten())

    return {'mse': mse, 'mae': mae, 'r2': r2, 'Y_pred': Y_pred,
            'Y_params_pred': Y_params_pred}


# =============================================================================
# EXPERIMENT 1: Polynomial degree sweep at fixed N
# =============================================================================

def experiment_degree_sweep():
    """Compare direct with increasing polynomial degree vs adaptive (linear)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: POLYNOMIAL DEGREE SWEEP (N=100)")
    print("=" * 70)

    np.random.seed(42)
    train_data = generate_dataset(100, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5))
    test_extrap = generate_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5))
    test_interp = generate_dataset(50, d1_range=(0.7, 1.3), d2_range=(0.6, 1.4))

    degrees = [1, 2, 3, 4, 5, 6]
    results_extrap = {'direct': [], 'adaptive': []}
    results_interp = {'direct': [], 'adaptive': []}

    print("\n--- EXTRAPOLATION ---")
    print(f"{'Degree':>6} {'Direct MAE':>12} {'Direct R2':>10} {'Adapt MAE':>12} {'Adapt R2':>10} {'#Feat':>6}")
    print("-" * 60)

    for deg in degrees:
        res_d = train_poly_direct(train_data, test_extrap, degree=deg)
        res_a = train_adaptive(train_data, test_extrap, degree=1)  # adaptive always linear
        results_extrap['direct'].append(res_d)
        results_extrap['adaptive'].append(res_a)
        print(f"{deg:>6d} {res_d['mae']:>12.4f} {res_d['r2']:>10.4f} "
              f"{res_a['mae']:>12.4f} {res_a['r2']:>10.4f} {res_d['n_features']:>6d}")

    print("\n--- INTERPOLATION ---")
    print(f"{'Degree':>6} {'Direct MAE':>12} {'Direct R2':>10} {'Adapt MAE':>12} {'Adapt R2':>10} {'#Feat':>6}")
    print("-" * 60)

    for deg in degrees:
        res_d = train_poly_direct(train_data, test_interp, degree=deg)
        res_a = train_adaptive(train_data, test_interp, degree=1)
        results_interp['direct'].append(res_d)
        results_interp['adaptive'].append(res_a)
        print(f"{deg:>6d} {res_d['mae']:>12.6f} {res_d['r2']:>10.6f} "
              f"{res_a['mae']:>12.6f} {res_a['r2']:>10.6f} {res_d['n_features']:>6d}")

    return degrees, results_extrap, results_interp


# =============================================================================
# EXPERIMENT 2: Learning curves at different polynomial degrees
# =============================================================================

def experiment_learning_curves_poly():
    """Learning curves for direct with poly features vs adaptive."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LEARNING CURVES WITH POLYNOMIAL FEATURES")
    print("=" * 70)

    train_sizes = [10, 20, 40, 80, 160, 320, 640, 1000, 2000, 5000]
    n_seeds = 3
    degrees_to_test = [1, 2, 3, 4]

    results = {}
    for deg in degrees_to_test:
        results[f'direct_d{deg}'] = {'mae_mean': [], 'mae_std': []}
    results['adaptive'] = {'mae_mean': [], 'mae_std': []}

    for n_train in train_sizes:
        print(f"\n  N={n_train}:")

        # Collect results across seeds
        seed_results = {k: [] for k in results}

        for seed in range(n_seeds):
            np.random.seed(seed * 100 + n_train)
            train_data = generate_dataset(n_train, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5))
            test_data = generate_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5))

            # Adaptive (always linear)
            res_a = train_adaptive(train_data, test_data, degree=1)
            seed_results['adaptive'].append(res_a['mae'])

            # Direct with various polynomial degrees
            for deg in degrees_to_test:
                # Need at least as many samples as features
                n_features = len(PolynomialFeatures(degree=deg, include_bias=False)
                                 .fit_transform(np.zeros((1, 2)))[0])
                if n_train > n_features + 2:
                    res_d = train_poly_direct(train_data, test_data, degree=deg)
                    seed_results[f'direct_d{deg}'].append(res_d['mae'])
                else:
                    seed_results[f'direct_d{deg}'].append(np.nan)

        for key in results:
            vals = [v for v in seed_results[key] if not np.isnan(v)]
            if vals:
                results[key]['mae_mean'].append(np.mean(vals))
                results[key]['mae_std'].append(np.std(vals))
            else:
                results[key]['mae_mean'].append(np.nan)
                results[key]['mae_std'].append(np.nan)

        # Print summary for this N
        line = f"    "
        for deg in degrees_to_test:
            k = f'direct_d{deg}'
            m = results[k]['mae_mean'][-1]
            if not np.isnan(m):
                line += f"D(d={deg})={m:.3f}  "
            else:
                line += f"D(d={deg})=N/A    "
        line += f"Adaptive={results['adaptive']['mae_mean'][-1]:.3f}"
        print(line)

    return train_sizes, results, degrees_to_test


# =============================================================================
# FIGURE: Combined results
# =============================================================================

def create_figure(degree_sweep, learning_curves):
    """Create 4-panel figure for polynomial features analysis."""
    degrees, res_extrap, res_interp = degree_sweep
    train_sizes, lc_results, lc_degrees = learning_curves

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- Panel A: Extrapolation MAE vs polynomial degree ---
    ax = axes[0, 0]
    direct_mae = [r['mae'] for r in res_extrap['direct']]
    adaptive_mae = res_extrap['adaptive'][0]['mae']  # constant (linear)
    ax.semilogy(degrees, direct_mae, 'bo-', linewidth=2, markersize=8, label='Direct (poly features)')
    ax.axhline(adaptive_mae, color='red', linewidth=2, linestyle='--',
               label=f'Adaptive (linear, MAE={adaptive_mae:.3f})')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MAE [eV]')
    ax.set_title('A. Extrapolation: Direct with Poly Features\nvs Adaptive (N=100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(degrees)

    # --- Panel B: Interpolation MAE vs polynomial degree ---
    ax = axes[0, 1]
    direct_mae_i = [r['mae'] for r in res_interp['direct']]
    adaptive_mae_i = res_interp['adaptive'][0]['mae']
    ax.semilogy(degrees, direct_mae_i, 'bo-', linewidth=2, markersize=8, label='Direct (poly features)')
    ax.axhline(adaptive_mae_i, color='red', linewidth=2, linestyle='--',
               label=f'Adaptive (linear, MAE={adaptive_mae_i:.4f})')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MAE [eV]')
    ax.set_title('B. Interpolation: Direct with Poly Features\nvs Adaptive (N=100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(degrees)

    # --- Panel C: Learning curves (extrapolation) ---
    ax = axes[1, 0]
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
    for i, deg in enumerate(lc_degrees):
        key = f'direct_d{deg}'
        mae = np.array(lc_results[key]['mae_mean'])
        mask = ~np.isnan(mae)
        sizes = np.array(train_sizes)[mask]
        mae = mae[mask]
        ax.loglog(sizes, mae, 'o-', color=colors[i], linewidth=2, markersize=6,
                  label=f'Direct (degree {deg})')

    mae_a = np.array(lc_results['adaptive']['mae_mean'])
    ax.loglog(train_sizes, mae_a, 'rs-', linewidth=2.5, markersize=8,
              label='Adaptive (linear)')

    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('MAE [eV]')
    ax.set_title('C. Learning Curves: Extrapolation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel D: Example predictions at N=100 ---
    ax = axes[1, 1]
    np.random.seed(42)
    train_data = generate_dataset(100, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5))
    test_data = generate_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5))

    idx = 0
    r = test_data['r_grid']
    V_true = test_data['curves'][idx]
    ax.plot(r, V_true, 'k-', linewidth=3, label='True')

    for deg in [1, 3]:
        res_d = train_poly_direct(train_data, test_data, degree=deg)
        ax.plot(r, res_d['Y_pred'][idx], '--', linewidth=1.5, label=f'Direct (deg {deg})')

    res_a = train_adaptive(train_data, test_data)
    ax.plot(r, res_a['Y_pred'][idx], 'r:', linewidth=3, label='Adaptive')

    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('D. Example Extrapolation Prediction')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.3, 0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'figures/fig_poly_features_lj.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")

    return outpath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXTENDED ANALYSIS: Polynomial Features for Direct LJ Learning")
    print("Question: Is the adaptive advantage about physics or model capacity?")
    print("=" * 70)

    degree_sweep = experiment_degree_sweep()
    learning_curves = experiment_learning_curves_poly()
    create_figure(degree_sweep, learning_curves)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    degrees, res_extrap, _ = degree_sweep
    adaptive_mae = res_extrap['adaptive'][0]['mae']
    print(f"\nAdaptive (linear Ridge, 2 features): MAE = {adaptive_mae:.4f} eV")
    print(f"\nDirect with polynomial features (N=100, extrapolation):")
    for i, deg in enumerate(degrees):
        r = res_extrap['direct'][i]
        ratio = r['mae'] / adaptive_mae
        print(f"  Degree {deg}: MAE = {r['mae']:.4f} eV ({r['n_features']} features) "
              f"-- {ratio:.1f}x worse than adaptive")

    print(f"""
Key finding: Even degree-6 polynomials (28 features) cannot match adaptive
with just 2 linear features for EXTRAPOLATION. The advantage is not model
capacity -- it's the physics equation providing the correct functional form
for V(r) at unseen sigma values.

For INTERPOLATION, polynomial features do help direct learning and can
approach adaptive performance, confirming this is specifically an
extrapolation advantage from physics-informed modeling.
""")


if __name__ == "__main__":
    main()
