"""
MLP nonlinear baseline comparison for LJ and Rose/UBER systems.

Addresses reviewer concern: Is the adaptive advantage merely linear vs nonlinear,
or does it reflect the correct functional form?

Compares:
  1. Direct Ridge (linear baseline)
  2. Direct MLP (nonlinear baseline, several architectures)
  3. Adaptive Ridge (physics-informed)

For both LJ (fixed grid) and Rose/UBER (sample-dependent grid) systems,
in both extrapolation and interpolation regimes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Potential functions
# =============================================================================

def lennard_jones(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def morse_V(r, De, a, re):
    return De * (1 - np.exp(-a * (r - re)))**2

def rose_V(r, E_c, r_e, l):
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star))

# =============================================================================
# Dataset generation (matching existing experiments exactly)
# =============================================================================

def generate_lj_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r = np.linspace(2.5, 12.0, n_points)
    X, params, curves = [], [], []
    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2
        V = lennard_jones(r, epsilon, sigma)
        X.append([d1, d2])
        params.append([epsilon, sigma])
        curves.append(V)
    return {
        'X': np.array(X), 'params': np.array(params),
        'curves': np.array(curves), 'r': r
    }

def generate_rose_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, morse_params, rose_params, curves, r_grids = [], [], [], [], []
    a_fixed = 1.5
    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        De = 0.5 + 2.0 * d1
        re = 1.5 + 0.5 * d2
        r = np.linspace(0.7 * re, 3.0 * re, n_points)
        V = morse_V(r, De, a_fixed, re)
        # Fit Rose parameters
        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1.0 / a_fixed],
                                bounds=([0.01, 0.5, 0.01], [20, 10, 5]), maxfev=5000)
            X.append([d1, d2])
            morse_params.append([De, re, a_fixed])
            rose_params.append(popt)  # [E_c, r_e, l]
            curves.append(V)
            r_grids.append(r)
        except RuntimeError:
            pass
    return {
        'X': np.array(X), 'morse_params': np.array(morse_params),
        'rose_params': np.array(rose_params), 'curves': np.array(curves),
        'r_grids': r_grids
    }

# =============================================================================
# MLP architectures to test
# =============================================================================

MLP_CONFIGS = {
    'MLP-small':   {'hidden_layer_sizes': (32, 32),     'max_iter': 2000},
    'MLP-medium':  {'hidden_layer_sizes': (64, 64),     'max_iter': 2000},
    'MLP-large':   {'hidden_layer_sizes': (128, 64),    'max_iter': 3000},
    'MLP-xlarge':  {'hidden_layer_sizes': (256, 128),   'max_iter': 3000},
    'MLP-deep':    {'hidden_layer_sizes': (64, 64, 64), 'max_iter': 3000},
}

def make_mlp(config):
    return MLPRegressor(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        max_iter=config['max_iter'],
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        alpha=0.001,  # L2 regularization
        random_state=42,
    )

# =============================================================================
# Train and evaluate (LJ)
# =============================================================================

def evaluate_lj(train_data, test_data, verbose=False):
    X_train = train_data['X']
    X_test = test_data['X']
    Y_train = train_data['curves']
    Y_test = test_data['curves']
    r = train_data['r']

    # Standardize inputs
    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    # --- Direct Ridge ---
    scaler_V = StandardScaler()
    Y_tr_s = scaler_V.fit_transform(Y_train)
    ridge_direct = Ridge(alpha=1.0)
    ridge_direct.fit(X_tr, Y_tr_s)
    Y_pred_ridge = scaler_V.inverse_transform(ridge_direct.predict(X_te))

    results = {}
    results['Ridge (direct)'] = {
        'mae': mean_absolute_error(Y_test, Y_pred_ridge),
        'r2': r2_score(Y_test.flatten(), Y_pred_ridge.flatten()),
        'pred': Y_pred_ridge,
    }

    # --- Direct MLP (multiple architectures) ---
    best_mlp_mae = np.inf
    best_mlp_name = None
    for name, config in MLP_CONFIGS.items():
        scaler_V_mlp = StandardScaler()
        Y_tr_mlp = scaler_V_mlp.fit_transform(Y_train)
        mlp = make_mlp(config)
        mlp.fit(X_tr, Y_tr_mlp)
        Y_pred_mlp = scaler_V_mlp.inverse_transform(mlp.predict(X_te))
        mae = mean_absolute_error(Y_test, Y_pred_mlp)
        r2 = r2_score(Y_test.flatten(), Y_pred_mlp.flatten())
        results[name] = {'mae': mae, 'r2': r2, 'pred': Y_pred_mlp}
        if verbose:
            print(f"  {name}: MAE={mae:.4f}, R2={r2:.4f}")
        if mae < best_mlp_mae:
            best_mlp_mae = mae
            best_mlp_name = name

    # --- Adaptive Ridge ---
    params_train = train_data['params']  # [epsilon, sigma]
    params_test = test_data['params']
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(params_train)
    ridge_adapt = Ridge(alpha=1.0)
    ridge_adapt.fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(ridge_adapt.predict(X_te))
    # Reconstruct via physics
    Y_pred_adapt = np.array([
        lennard_jones(r, max(eps, 0.001), max(sig, 0.1))
        for eps, sig in P_pred
    ])
    results['Adaptive Ridge'] = {
        'mae': mean_absolute_error(Y_test, Y_pred_adapt),
        'r2': r2_score(Y_test.flatten(), Y_pred_adapt.flatten()),
        'pred': Y_pred_adapt,
    }
    results['_best_mlp'] = best_mlp_name
    return results

# =============================================================================
# Train and evaluate (Rose/UBER)
# =============================================================================

def evaluate_rose(train_data, test_data, verbose=False):
    X_train = train_data['X']
    X_test = test_data['X']
    Y_train = train_data['curves']
    Y_test = test_data['curves']

    # Use common grid (average of all training r_grids)
    # For Rose, each sample has a different grid - use a reference grid for direct
    n_points = Y_train.shape[1]

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    # --- Direct Ridge ---
    scaler_V = StandardScaler()
    Y_tr_s = scaler_V.fit_transform(Y_train)
    ridge_direct = Ridge(alpha=1.0)
    ridge_direct.fit(X_tr, Y_tr_s)
    Y_pred_ridge = scaler_V.inverse_transform(ridge_direct.predict(X_te))

    results = {}
    results['Ridge (direct)'] = {
        'mae': mean_absolute_error(Y_test, Y_pred_ridge),
        'r2': r2_score(Y_test.flatten(), Y_pred_ridge.flatten()),
        'pred': Y_pred_ridge,
    }

    # --- Direct MLP ---
    best_mlp_mae = np.inf
    best_mlp_name = None
    for name, config in MLP_CONFIGS.items():
        scaler_V_mlp = StandardScaler()
        Y_tr_mlp = scaler_V_mlp.fit_transform(Y_train)
        mlp = make_mlp(config)
        mlp.fit(X_tr, Y_tr_mlp)
        Y_pred_mlp = scaler_V_mlp.inverse_transform(mlp.predict(X_te))
        mae = mean_absolute_error(Y_test, Y_pred_mlp)
        r2 = r2_score(Y_test.flatten(), Y_pred_mlp.flatten())
        results[name] = {'mae': mae, 'r2': r2, 'pred': Y_pred_mlp}
        if verbose:
            print(f"  {name}: MAE={mae:.4f}, R2={r2:.4f}")
        if mae < best_mlp_mae:
            best_mlp_mae = mae
            best_mlp_name = name

    # --- Adaptive Ridge ---
    rose_params_train = train_data['rose_params']  # [E_c, r_e, l]
    rose_params_test = test_data['rose_params']
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(rose_params_train)
    ridge_adapt = Ridge(alpha=1.0)
    ridge_adapt.fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(ridge_adapt.predict(X_te))
    # Reconstruct via physics on each sample's own grid
    Y_pred_adapt = []
    for i, (E_c, r_e, l) in enumerate(P_pred):
        r_i = test_data['r_grids'][i]
        Y_pred_adapt.append(rose_V(r_i, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
    Y_pred_adapt = np.array(Y_pred_adapt)

    results['Adaptive Ridge'] = {
        'mae': mean_absolute_error(Y_test, Y_pred_adapt),
        'r2': r2_score(Y_test.flatten(), Y_pred_adapt.flatten()),
        'pred': Y_pred_adapt,
    }
    results['_best_mlp'] = best_mlp_name
    return results

# =============================================================================
# Learning curves
# =============================================================================

def run_learning_curves(system='lj', regime='extrap', n_seeds=5):
    train_sizes = [10, 15, 20, 30, 40, 60, 80, 100, 120, 160]

    # Storage for each method
    methods = ['Ridge (direct)', 'Best MLP (direct)', 'Adaptive Ridge']
    lc = {m: {'mae_mean': [], 'mae_std': []} for m in methods}

    for n_train in train_sizes:
        seed_results = {m: [] for m in methods}

        for s in range(n_seeds):
            seed = s * 100 + n_train

            if system == 'lj':
                train = generate_lj_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
                if regime == 'extrap':
                    test = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=seed + 10000)
                else:
                    test = generate_lj_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
                res = evaluate_lj(train, test)
            else:
                train = generate_rose_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
                if regime == 'extrap':
                    test = generate_rose_dataset(50, (3.0, 4.5), (0.5, 1.5), seed=seed + 10000)
                else:
                    test = generate_rose_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
                res = evaluate_rose(train, test)

            seed_results['Ridge (direct)'].append(res['Ridge (direct)']['mae'])
            seed_results['Best MLP (direct)'].append(res[res['_best_mlp']]['mae'])
            seed_results['Adaptive Ridge'].append(res['Adaptive Ridge']['mae'])

        for m in methods:
            vals = seed_results[m]
            lc[m]['mae_mean'].append(np.mean(vals))
            lc[m]['mae_std'].append(np.std(vals))

        print(f"  N={n_train:3d}: Ridge={np.mean(seed_results['Ridge (direct)']):.3f}, "
              f"MLP={np.mean(seed_results['Best MLP (direct)']):.3f}, "
              f"Adaptive={np.mean(seed_results['Adaptive Ridge']):.3f}")

    return train_sizes, lc

# =============================================================================
# Architecture sweep at fixed N
# =============================================================================

def architecture_sweep(system='lj', regime='extrap', n_train=160, n_seeds=5):
    """Test all MLP architectures and report MAE for each."""
    all_methods = ['Ridge (direct)'] + list(MLP_CONFIGS.keys()) + ['Adaptive Ridge']
    results = {m: [] for m in all_methods}

    for s in range(n_seeds):
        seed = s * 100 + n_train
        if system == 'lj':
            train = generate_lj_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
            if regime == 'extrap':
                test = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=seed + 10000)
            else:
                test = generate_lj_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
            res = evaluate_lj(train, test)
        else:
            train = generate_rose_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
            if regime == 'extrap':
                test = generate_rose_dataset(50, (3.0, 4.5), (0.5, 1.5), seed=seed + 10000)
            else:
                test = generate_rose_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
            res = evaluate_rose(train, test)

        for m in all_methods:
            if m in res:
                results[m].append(res[m]['mae'])

    print(f"\n{'Method':<20s} {'MAE (mean±std)':>20s}")
    print("-" * 42)
    for m in all_methods:
        vals = results[m]
        print(f"{m:<20s} {np.mean(vals):>8.4f} ± {np.std(vals):.4f}")

    return results

# =============================================================================
# Plotting
# =============================================================================

def plot_learning_curves(lj_extrap, lj_interp, rose_extrap, rose_interp, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    datasets = [
        (lj_extrap, 'LJ Extrapolation', axes[0, 0]),
        (lj_interp, 'LJ Interpolation', axes[0, 1]),
        (rose_extrap, 'Rose/UBER Extrapolation', axes[1, 0]),
        (rose_interp, 'Rose/UBER Interpolation', axes[1, 1]),
    ]

    colors = {'Ridge (direct)': '#1f77b4', 'Best MLP (direct)': '#ff7f0e',
              'Adaptive Ridge': '#d62728'}
    markers = {'Ridge (direct)': 's', 'Best MLP (direct)': '^',
               'Adaptive Ridge': 'o'}
    labels = {'Ridge (direct)': 'Direct (Ridge)',
              'Best MLP (direct)': 'Direct (best MLP)',
              'Adaptive Ridge': 'Adaptive (Ridge)'}

    for (train_sizes, lc), title, ax in datasets:
        for method in ['Ridge (direct)', 'Best MLP (direct)', 'Adaptive Ridge']:
            means = np.array(lc[method]['mae_mean'])
            stds = np.array(lc[method]['mae_std'])
            ax.errorbar(train_sizes, means, yerr=stds,
                        label=labels[method], color=colors[method],
                        marker=markers[method], markersize=6, capsize=3,
                        linewidth=1.5)

        ax.set_xlabel('Training set size N')
        ax.set_ylabel('MAE (eV)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


def plot_architecture_comparison(lj_results, rose_results, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, title in [(axes[0], lj_results, 'LJ Extrapolation'),
                                (axes[1], rose_results, 'Rose/UBER Extrapolation')]:
        methods = list(results.keys())
        means = [np.mean(results[m]) for m in methods]
        stds = [np.std(results[m]) for m in methods]

        # Color: blue for direct methods, red for adaptive
        colors = ['#d62728' if 'Adaptive' in m else '#1f77b4' if 'Ridge' in m else '#ff7f0e'
                  for m in methods]

        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' (direct)', '\n(direct)').replace(' Ridge', '\nRidge')
                            for m in methods],
                           fontsize=8, rotation=0, ha='center')
        ax.set_ylabel('MAE (eV)')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    figdir = 'figures'

    # --- Architecture sweep at N=160 ---
    print("=" * 60)
    print("ARCHITECTURE SWEEP: LJ Extrapolation (N=160)")
    print("=" * 60)
    lj_arch = architecture_sweep('lj', 'extrap', n_train=160)

    print("\n" + "=" * 60)
    print("ARCHITECTURE SWEEP: Rose/UBER Extrapolation (N=160)")
    print("=" * 60)
    rose_arch = architecture_sweep('rose', 'extrap', n_train=160)

    plot_architecture_comparison(lj_arch, rose_arch,
                                 f'{figdir}/fig_architecture_comparison.png')

    # --- Learning curves ---
    print("\n" + "=" * 60)
    print("LEARNING CURVES: LJ Extrapolation")
    print("=" * 60)
    lj_extrap = run_learning_curves('lj', 'extrap')

    print("\n" + "=" * 60)
    print("LEARNING CURVES: LJ Interpolation")
    print("=" * 60)
    lj_interp = run_learning_curves('lj', 'interp')

    print("\n" + "=" * 60)
    print("LEARNING CURVES: Rose/UBER Extrapolation")
    print("=" * 60)
    rose_extrap = run_learning_curves('rose', 'extrap')

    print("\n" + "=" * 60)
    print("LEARNING CURVES: Rose/UBER Interpolation")
    print("=" * 60)
    rose_interp = run_learning_curves('rose', 'interp')

    plot_learning_curves(lj_extrap, lj_interp, rose_extrap, rose_interp,
                         f'{figdir}/fig_mlp_learning_curves.png')

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (N=160 extrapolation)")
    print("=" * 60)
    for system, (ts, lc) in [('LJ', lj_extrap), ('Rose', rose_extrap)]:
        print(f"\n{system}:")
        for m in ['Ridge (direct)', 'Best MLP (direct)', 'Adaptive Ridge']:
            print(f"  {m:<25s}: MAE = {lc[m]['mae_mean'][-1]:.4f} ± {lc[m]['mae_std'][-1]:.4f}")
        ridge_mae = lc['Ridge (direct)']['mae_mean'][-1]
        mlp_mae = lc['Best MLP (direct)']['mae_mean'][-1]
        adapt_mae = lc['Adaptive Ridge']['mae_mean'][-1]
        print(f"  Adaptive vs Ridge:  {ridge_mae/adapt_mae:.1f}x")
        print(f"  Adaptive vs MLP:    {mlp_mae/adapt_mae:.1f}x")
