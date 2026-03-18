"""
Delta Learning for Lennard-Jones System
========================================

For LJ, the physics equation is EXACT (no fitting approximation).
Delta learning should show minimal improvement over pure adaptive,
since the residual V_true - V_adaptive is due only to parameter
prediction error, not systematic equation error.

This contrasts with Rose/UBER where Rose != Morse gives ~0.02 eV systematic error.

Usage:
    MPLBACKEND=Agg python delta_learning.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


# =============================================================================
# POTENTIAL
# =============================================================================

def lj_V(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)

    r = np.linspace(2.5, 12.0, n_points)
    data = {'X': [], 'params': [], 'curves': [], 'r_grid': r}

    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2
        V = lj_V(r, epsilon, sigma)
        data['X'].append([d1, d2])
        data['params'].append([epsilon, sigma])
        data['curves'].append(V)

    data['X'] = np.array(data['X'])
    data['params'] = np.array(data['params'])
    data['curves'] = np.array(data['curves'])
    return data


# =============================================================================
# THREE-WAY COMPARISON
# =============================================================================

def train_and_evaluate(train_data, test_data):
    X_train, X_test = train_data['X'], test_data['X']
    V_train, V_test = train_data['curves'], test_data['curves']
    params_train = train_data['params']
    r = test_data['r_grid']

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # 1. DIRECT
    scaler_V = StandardScaler()
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, scaler_V.fit_transform(V_train))
    V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    # 2. ADAPTIVE
    scaler_p = StandardScaler()
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, scaler_p.fit_transform(params_train))
    params_pred = scaler_p.inverse_transform(model_adaptive.predict(X_test_s))

    V_pred_adaptive = np.zeros_like(V_test)
    for i, (eps, sig) in enumerate(params_pred):
        V_pred_adaptive[i] = lj_V(r, max(eps, 0.001), max(sig, 0.1))

    # 3. DELTA
    # Adaptive baseline on training data
    params_train_pred = scaler_p.inverse_transform(model_adaptive.predict(X_train_s))
    V_baseline_train = np.zeros_like(V_train)
    for i, (eps, sig) in enumerate(params_train_pred):
        V_baseline_train[i] = lj_V(r, max(eps, 0.001), max(sig, 0.1))

    delta_train = V_train - V_baseline_train
    scaler_d = StandardScaler()
    model_delta = Ridge(alpha=1.0)
    model_delta.fit(X_train_s, scaler_d.fit_transform(delta_train))
    delta_pred = scaler_d.inverse_transform(model_delta.predict(X_test_s))
    V_pred_delta = V_pred_adaptive + delta_pred

    return {
        'V_pred_direct': V_pred_direct,
        'V_pred_adaptive': V_pred_adaptive,
        'V_pred_delta': V_pred_delta,
        'mae_direct': mean_absolute_error(V_test, V_pred_direct),
        'mae_adaptive': mean_absolute_error(V_test, V_pred_adaptive),
        'mae_delta': mean_absolute_error(V_test, V_pred_delta),
        'r2_direct': r2_score(V_test.flatten(), V_pred_direct.flatten()),
        'r2_adaptive': r2_score(V_test.flatten(), V_pred_adaptive.flatten()),
        'r2_delta': r2_score(V_test.flatten(), V_pred_delta.flatten()),
    }


# =============================================================================
# LEARNING CURVES
# =============================================================================

def learning_curves(n_sizes, n_seeds=5):
    results = {k: [] for k in ['n_train',
               'direct_extrap', 'adaptive_extrap', 'delta_extrap',
               'direct_interp', 'adaptive_interp', 'delta_interp']}

    for N in n_sizes:
        ext_d, ext_a, ext_del = [], [], []
        int_d, int_a, int_del = [], [], []

        for seed in range(n_seeds):
            train = generate_dataset(N, (0.5, 1.5), (0.5, 1.5), seed=seed*1000+N)
            test_ext = generate_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=9999+seed)
            test_int = generate_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=8888+seed)

            if len(train['X']) < 3:
                continue

            res_ext = train_and_evaluate(train, test_ext)
            res_int = train_and_evaluate(train, test_int)

            ext_d.append(res_ext['mae_direct'])
            ext_a.append(res_ext['mae_adaptive'])
            ext_del.append(res_ext['mae_delta'])
            int_d.append(res_int['mae_direct'])
            int_a.append(res_int['mae_adaptive'])
            int_del.append(res_int['mae_delta'])

        results['n_train'].append(N)
        results['direct_extrap'].append((np.mean(ext_d), np.std(ext_d)))
        results['adaptive_extrap'].append((np.mean(ext_a), np.std(ext_a)))
        results['delta_extrap'].append((np.mean(ext_del), np.std(ext_del)))
        results['direct_interp'].append((np.mean(int_d), np.std(int_d)))
        results['adaptive_interp'].append((np.mean(int_a), np.std(int_a)))
        results['delta_interp'].append((np.mean(int_del), np.std(int_del)))

        print(f"  N={N:4d}: Extrap MAE -- Direct={np.mean(ext_d):.4f}, "
              f"Adaptive={np.mean(ext_a):.4f}, Delta={np.mean(ext_del):.4f}")

    return results


# =============================================================================
# FIGURE
# =============================================================================

def create_figure(lc_results, example_results, r_grid, V_test):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = {'direct': '#888888', 'adaptive': 'steelblue', 'delta': '#E07020'}

    # Panel A: Learning curves (extrapolation)
    ax = axes[0, 0]
    N = lc_results['n_train']
    for key, label, color in [('direct_extrap', 'Direct', colors['direct']),
                               ('adaptive_extrap', 'Adaptive', colors['adaptive']),
                               ('delta_extrap', 'Delta', colors['delta'])]:
        means = [x[0] for x in lc_results[key]]
        stds = [x[1] for x in lc_results[key]]
        ax.errorbar(N, means, yerr=stds, fmt='o-', color=color, label=label,
                    linewidth=2, capsize=3, markersize=5)
    ax.set_xlabel('Training set size N')
    ax.set_ylabel('Extrapolation MAE [eV]')
    ax.set_title('A. Learning Curves (Extrapolation)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel B: Learning curves (interpolation)
    ax = axes[0, 1]
    for key, label, color in [('direct_interp', 'Direct', colors['direct']),
                               ('adaptive_interp', 'Adaptive', colors['adaptive']),
                               ('delta_interp', 'Delta', colors['delta'])]:
        means = [x[0] for x in lc_results[key]]
        stds = [x[1] for x in lc_results[key]]
        ax.errorbar(N, means, yerr=stds, fmt='o-', color=color, label=label,
                    linewidth=2, capsize=3, markersize=5)
    ax.set_xlabel('Training set size N')
    ax.set_ylabel('Interpolation MAE [eV]')
    ax.set_title('B. Learning Curves (Interpolation)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel C: Improvement ratios
    ax = axes[0, 2]
    ratio_adapt = [d[0]/max(a[0], 1e-10) for d, a in
                   zip(lc_results['direct_extrap'], lc_results['adaptive_extrap'])]
    ratio_delta = [d[0]/max(dl[0], 1e-10) for d, dl in
                   zip(lc_results['direct_extrap'], lc_results['delta_extrap'])]
    ax.plot(N, ratio_adapt, 'o-', color=colors['adaptive'], linewidth=2,
            markersize=6, label='Direct/Adaptive')
    ax.plot(N, ratio_delta, 's-', color=colors['delta'], linewidth=2,
            markersize=6, label='Direct/Delta')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training set size N')
    ax.set_ylabel('MAE Ratio')
    ax.set_title('C. Extrapolation Advantage')
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Panel D: Example extrapolation
    ax = axes[1, 0]
    idx = 0
    ax.plot(r_grid, V_test[idx], 'k-', linewidth=3, label='True')
    ax.plot(r_grid, example_results['V_pred_direct'][idx], '--',
            color=colors['direct'], linewidth=2, label='Direct')
    ax.plot(r_grid, example_results['V_pred_adaptive'][idx], ':',
            color=colors['adaptive'], linewidth=2.5, label='Adaptive')
    ax.plot(r_grid, example_results['V_pred_delta'][idx], '-.',
            color=colors['delta'], linewidth=2.5, label='Delta')
    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('D. Example Extrapolation')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.3, 0.5)
    ax.grid(True, alpha=0.3)

    # Panel E: Example interpolation
    ax = axes[1, 1]
    ax.plot(r_grid, V_test[idx], 'k-', linewidth=3, label='True')
    ax.plot(r_grid, example_results['V_pred_adaptive'][idx], ':',
            color=colors['adaptive'], linewidth=2.5, label='Adaptive')
    ax.plot(r_grid, example_results['V_pred_delta'][idx], '-.',
            color=colors['delta'], linewidth=2.5, label='Delta')
    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('E. Adaptive vs Delta (Detail)')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.3, 0.5)
    ax.grid(True, alpha=0.3)

    # Panel F: Summary
    ax = axes[1, 2]
    labels = ['Extrap.', 'Interp.']
    x = np.arange(len(labels))
    width = 0.25
    ext_vals = [lc_results['direct_extrap'][-1][0],
                lc_results['adaptive_extrap'][-1][0],
                lc_results['delta_extrap'][-1][0]]
    int_vals = [lc_results['direct_interp'][-1][0],
                lc_results['adaptive_interp'][-1][0],
                lc_results['delta_interp'][-1][0]]
    ax.bar(x - width, [ext_vals[0], int_vals[0]], width,
           label='Direct', color=colors['direct'], alpha=0.8)
    ax.bar(x, [ext_vals[1], int_vals[1]], width,
           label='Adaptive', color=colors['adaptive'], alpha=0.8)
    ax.bar(x + width, [ext_vals[2], int_vals[2]], width,
           label='Delta', color=colors['delta'], alpha=0.8)
    ax.set_ylabel('MAE [eV]')
    ax.set_title(f'F. Summary (N={lc_results["n_train"][-1]})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = 'fig_lj_delta_learning.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DELTA LEARNING: Lennard-Jones System")
    print("V_delta = V_lj(epsilon, sigma) + delta_correction(descriptors)")
    print("=" * 70)
    print("\nNote: For LJ the physics equation is EXACT, so delta learning")
    print("should show minimal improvement over pure adaptive.\n")

    print("Learning curves (5 seeds each)...")
    n_sizes = [10, 20, 40, 60, 80, 100, 120, 160]
    lc_results = learning_curves(n_sizes, n_seeds=5)

    print("\nExample predictions (N=100, extrapolation)...")
    np.random.seed(42)
    train = generate_dataset(100, (0.5, 1.5), (0.5, 1.5), seed=42)
    test_ext = generate_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=9999)

    example_results = train_and_evaluate(train, test_ext)
    print(f"  Direct:   MAE = {example_results['mae_direct']:.4f} eV, "
          f"R2 = {example_results['r2_direct']:.4f}")
    print(f"  Adaptive: MAE = {example_results['mae_adaptive']:.4f} eV, "
          f"R2 = {example_results['r2_adaptive']:.4f}")
    print(f"  Delta:    MAE = {example_results['mae_delta']:.4f} eV, "
          f"R2 = {example_results['r2_delta']:.4f}")

    create_figure(lc_results, example_results, test_ext['r_grid'], test_ext['curves'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
For LJ, the physics equation V = 4*eps*[(sig/r)^12 - (sig/r)^6] is exact.
The only source of error in adaptive is parameter prediction error.
Delta learning can correct this slightly but the improvement is small
compared to Rose/UBER where the Rose equation is approximate.

This contrast demonstrates that delta learning is most valuable when
the physics equation is approximate (Rose/UBER, DFT diatomics) but
less needed when it is exact (LJ).
""")


if __name__ == "__main__":
    main()
