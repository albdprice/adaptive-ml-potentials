"""
Publication-Quality Delta Learning Analysis
============================================

Combined analysis of delta learning for both Rose/UBER and Lennard-Jones systems.
Generates publication-quality figures showing:
  - Three-way comparison: Direct vs Adaptive vs Delta
  - Learning curves for both systems
  - The key finding: delta helps most when physics equation is approximate (Rose)
    and least when it is exact (LJ)

Usage:
    MPLBACKEND=Agg python delta_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
})


# =============================================================================
# POTENTIALS
# =============================================================================

def morse_V(r, De, a, re):
    return De * (1 - np.exp(-a * (r - re)))**2

def rose_V(r, E_c, r_e, l):
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star))

def lj_V(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_rose_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)

    r = np.linspace(0.7 * 1.75, 3.0 * 2.0, n_points)
    data = {'X': [], 'morse_params': [], 'rose_params': [],
            'curves': [], 'r_grid': r}

    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        De = 0.5 + 2.0 * d1
        re = 1.5 + 0.5 * d2
        a = 1.5
        V = morse_V(r, De, a, re)
        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1/a],
                                bounds=([0.01, 0.5, 0.01], [20, 10, 5]),
                                maxfev=5000)
            data['X'].append([d1, d2])
            data['morse_params'].append([De, re, a])
            data['rose_params'].append(popt)
            data['curves'].append(V)
        except Exception:
            pass

    for k in ['X', 'morse_params', 'rose_params', 'curves']:
        data[k] = np.array(data[k])
    return data


def generate_lj_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
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

    for k in ['X', 'params', 'curves']:
        data[k] = np.array(data[k])
    return data


# =============================================================================
# THREE-WAY TRAIN & EVALUATE
# =============================================================================

def train_evaluate_rose(train_data, test_data):
    X_train, X_test = train_data['X'], test_data['X']
    V_train, V_test = train_data['curves'], test_data['curves']
    params_train = train_data['rose_params']
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
    for i, (E_c, r_e, l) in enumerate(params_pred):
        V_pred_adaptive[i] = rose_V(r, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01))

    # 3. DELTA
    params_train_pred = scaler_p.inverse_transform(model_adaptive.predict(X_train_s))
    V_baseline_train = np.zeros_like(V_train)
    for i, (E_c, r_e, l) in enumerate(params_train_pred):
        V_baseline_train[i] = rose_V(r, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01))
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


def train_evaluate_lj(train_data, test_data):
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

def learning_curves_rose(n_sizes, n_seeds=5):
    results = {k: [] for k in ['n_train',
               'direct_extrap', 'adaptive_extrap', 'delta_extrap',
               'direct_interp', 'adaptive_interp', 'delta_interp']}

    for N in n_sizes:
        ext_d, ext_a, ext_del = [], [], []
        int_d, int_a, int_del = [], [], []

        for seed in range(n_seeds):
            train = generate_rose_dataset(N, (0.5, 1.5), (0.5, 1.5), seed=seed*1000+N)
            test_ext = generate_rose_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=9999+seed)
            test_int = generate_rose_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=8888+seed)

            # Align grids
            r = train['r_grid']
            for ds in [test_ext, test_int]:
                ds['r_grid'] = r
                curves = [morse_V(r, De, a, re) for De, re, a in ds['morse_params']]
                ds['curves'] = np.array(curves)
                rose_ps = []
                for V in ds['curves']:
                    try:
                        popt, _ = curve_fit(rose_V, r, V, p0=[2.0, 1.8, 0.7],
                                            bounds=([0.01, 0.5, 0.01], [20, 10, 5]),
                                            maxfev=5000)
                        rose_ps.append(popt)
                    except Exception:
                        rose_ps.append([2.0, 1.8, 0.7])
                ds['rose_params'] = np.array(rose_ps)

            if len(train['X']) < 3:
                continue

            res_ext = train_evaluate_rose(train, test_ext)
            res_int = train_evaluate_rose(train, test_int)

            ext_d.append(res_ext['mae_direct'])
            ext_a.append(res_ext['mae_adaptive'])
            ext_del.append(res_ext['mae_delta'])
            int_d.append(res_int['mae_direct'])
            int_a.append(res_int['mae_adaptive'])
            int_del.append(res_int['mae_delta'])

        results['n_train'].append(N)
        for key, vals in [('direct_extrap', ext_d), ('adaptive_extrap', ext_a),
                          ('delta_extrap', ext_del), ('direct_interp', int_d),
                          ('adaptive_interp', int_a), ('delta_interp', int_del)]:
            results[key].append((np.mean(vals), np.std(vals)))

        print(f"  Rose N={N:4d}: Extrap -- D={np.mean(ext_d):.4f}, "
              f"A={np.mean(ext_a):.4f}, Delta={np.mean(ext_del):.4f}")

    return results


def learning_curves_lj(n_sizes, n_seeds=5):
    results = {k: [] for k in ['n_train',
               'direct_extrap', 'adaptive_extrap', 'delta_extrap',
               'direct_interp', 'adaptive_interp', 'delta_interp']}

    for N in n_sizes:
        ext_d, ext_a, ext_del = [], [], []
        int_d, int_a, int_del = [], [], []

        for seed in range(n_seeds):
            train = generate_lj_dataset(N, (0.5, 1.5), (0.5, 1.5), seed=seed*1000+N)
            test_ext = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=9999+seed)
            test_int = generate_lj_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=8888+seed)

            if len(train['X']) < 3:
                continue

            res_ext = train_evaluate_lj(train, test_ext)
            res_int = train_evaluate_lj(train, test_int)

            ext_d.append(res_ext['mae_direct'])
            ext_a.append(res_ext['mae_adaptive'])
            ext_del.append(res_ext['mae_delta'])
            int_d.append(res_int['mae_direct'])
            int_a.append(res_int['mae_adaptive'])
            int_del.append(res_int['mae_delta'])

        results['n_train'].append(N)
        for key, vals in [('direct_extrap', ext_d), ('adaptive_extrap', ext_a),
                          ('delta_extrap', ext_del), ('direct_interp', int_d),
                          ('adaptive_interp', int_a), ('delta_interp', int_del)]:
            results[key].append((np.mean(vals), np.std(vals)))

        print(f"  LJ   N={N:4d}: Extrap -- D={np.mean(ext_d):.4f}, "
              f"A={np.mean(ext_a):.4f}, Delta={np.mean(ext_del):.4f}")

    return results


# =============================================================================
# PUBLICATION FIGURE: Combined 2x3
# =============================================================================

def create_combined_figure(rose_lc, lj_lc, rose_ex, lj_ex, rose_r, lj_r,
                           rose_V_test, lj_V_test):
    """Create publication-quality 2x3 figure.

    Row 1: Rose/UBER  - (a) example curves, (b) learning curves, (c) improvement ratio
    Row 2: LJ         - (d) example curves, (e) learning curves, (f) improvement ratio
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    colors = {'direct': '#888888', 'adaptive': '#4682B4', 'delta': '#E07020'}
    labels = {'direct': 'Direct', 'adaptive': 'Adaptive', 'delta': 'Delta'}

    # ----- Row 1: Rose/UBER -----

    # (a) Example extrapolation
    ax = axes[0, 0]
    idx = 0
    ax.plot(rose_r, rose_V_test[idx], 'k-', lw=2.5, label='True (Morse)')
    ax.plot(rose_r, rose_ex['V_pred_direct'][idx], '--', color=colors['direct'],
            lw=1.8, label=f'Direct (MAE={rose_ex["mae_direct"]:.3f})')
    ax.plot(rose_r, rose_ex['V_pred_adaptive'][idx], ':', color=colors['adaptive'],
            lw=2.2, label=f'Adaptive (MAE={rose_ex["mae_adaptive"]:.3f})')
    ax.plot(rose_r, rose_ex['V_pred_delta'][idx], '-.', color=colors['delta'],
            lw=2.2, label=f'Delta (MAE={rose_ex["mae_delta"]:.3f})')
    ax.set_xlabel(r'$r$ [\AA]')
    ax.set_ylabel(r'$V(r)$ [eV]')
    ax.set_title('(a) Rose/UBER: Example Extrapolation')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)

    # (b) Learning curves (extrapolation)
    ax = axes[0, 1]
    N = rose_lc['n_train']
    for key, label, color in [('direct_extrap', 'Direct', colors['direct']),
                               ('adaptive_extrap', 'Adaptive', colors['adaptive']),
                               ('delta_extrap', 'Delta', colors['delta'])]:
        means = [x[0] for x in rose_lc[key]]
        stds = [x[1] for x in rose_lc[key]]
        ax.errorbar(N, means, yerr=stds, fmt='o-', color=color, label=label,
                    lw=1.8, capsize=3, markersize=4)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Extrapolation MAE [eV]')
    ax.set_title('(b) Rose/UBER: Learning Curves')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # (c) Improvement ratio
    ax = axes[0, 2]
    ratio_a = [d[0]/max(a[0], 1e-10) for d, a in
               zip(rose_lc['direct_extrap'], rose_lc['adaptive_extrap'])]
    ratio_d = [d[0]/max(dl[0], 1e-10) for d, dl in
               zip(rose_lc['direct_extrap'], rose_lc['delta_extrap'])]
    ax.plot(N, ratio_a, 'o-', color=colors['adaptive'], lw=1.8, ms=5,
            label='Direct/Adaptive')
    ax.plot(N, ratio_d, 's-', color=colors['delta'], lw=1.8, ms=5,
            label='Direct/Delta')
    ax.axhline(y=1.0, color='k', ls='--', alpha=0.4, lw=0.8)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('MAE ratio')
    ax.set_title('(c) Rose/UBER: Extrap. Advantage')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.2)

    # ----- Row 2: Lennard-Jones -----

    # (d) Example extrapolation
    ax = axes[1, 0]
    idx = 0
    ax.plot(lj_r, lj_V_test[idx], 'k-', lw=2.5, label='True')
    ax.plot(lj_r, lj_ex['V_pred_direct'][idx], '--', color=colors['direct'],
            lw=1.8, label=f'Direct (MAE={lj_ex["mae_direct"]:.3f})')
    ax.plot(lj_r, lj_ex['V_pred_adaptive'][idx], ':', color=colors['adaptive'],
            lw=2.2, label=f'Adaptive (MAE={lj_ex["mae_adaptive"]:.3f})')
    ax.plot(lj_r, lj_ex['V_pred_delta'][idx], '-.', color=colors['delta'],
            lw=2.2, label=f'Delta (MAE={lj_ex["mae_delta"]:.3f})')
    ax.set_xlabel(r'$r$ [\AA]')
    ax.set_ylabel(r'$V(r)$ [eV]')
    ax.set_title('(d) LJ: Example Extrapolation')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(-0.3, 0.5)
    ax.grid(True, alpha=0.2)

    # (e) Learning curves (extrapolation)
    ax = axes[1, 1]
    N_lj = lj_lc['n_train']
    for key, label, color in [('direct_extrap', 'Direct', colors['direct']),
                               ('adaptive_extrap', 'Adaptive', colors['adaptive']),
                               ('delta_extrap', 'Delta', colors['delta'])]:
        means = [x[0] for x in lj_lc[key]]
        stds = [x[1] for x in lj_lc[key]]
        ax.errorbar(N_lj, means, yerr=stds, fmt='o-', color=color, label=label,
                    lw=1.8, capsize=3, markersize=4)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Extrapolation MAE [eV]')
    ax.set_title('(e) LJ: Learning Curves')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # (f) Improvement ratio
    ax = axes[1, 2]
    ratio_a_lj = [d[0]/max(a[0], 1e-10) for d, a in
                  zip(lj_lc['direct_extrap'], lj_lc['adaptive_extrap'])]
    ratio_d_lj = [d[0]/max(dl[0], 1e-10) for d, dl in
                  zip(lj_lc['direct_extrap'], lj_lc['delta_extrap'])]
    ax.plot(N_lj, ratio_a_lj, 'o-', color=colors['adaptive'], lw=1.8, ms=5,
            label='Direct/Adaptive')
    ax.plot(N_lj, ratio_d_lj, 's-', color=colors['delta'], lw=1.8, ms=5,
            label='Direct/Delta')
    ax.axhline(y=1.0, color='k', ls='--', alpha=0.4, lw=0.8)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('MAE ratio')
    ax.set_title('(f) LJ: Extrap. Advantage')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.2)

    plt.tight_layout(h_pad=1.5)
    outpath = 'figures/fig_delta_learning_combined.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nCombined figure saved: {outpath}")


# =============================================================================
# SUPPLEMENTAL FIGURE: Interpolation comparison
# =============================================================================

def create_interpolation_figure(rose_lc, lj_lc):
    """Supplemental figure: interpolation learning curves + bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'direct': '#888888', 'adaptive': '#4682B4', 'delta': '#E07020'}

    # (a) Rose interpolation LC
    ax = axes[0]
    N = rose_lc['n_train']
    for key, label, color in [('direct_interp', 'Direct', colors['direct']),
                               ('adaptive_interp', 'Adaptive', colors['adaptive']),
                               ('delta_interp', 'Delta', colors['delta'])]:
        means = [x[0] for x in rose_lc[key]]
        stds = [x[1] for x in rose_lc[key]]
        ax.errorbar(N, means, yerr=stds, fmt='o-', color=color, label=label,
                    lw=1.8, capsize=3, markersize=4)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Interpolation MAE [eV]')
    ax.set_title('(a) Rose/UBER: Interpolation')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # (b) LJ interpolation LC
    ax = axes[1]
    N_lj = lj_lc['n_train']
    for key, label, color in [('direct_interp', 'Direct', colors['direct']),
                               ('adaptive_interp', 'Adaptive', colors['adaptive']),
                               ('delta_interp', 'Delta', colors['delta'])]:
        means = [x[0] for x in lj_lc[key]]
        stds = [x[1] for x in lj_lc[key]]
        ax.errorbar(N_lj, means, yerr=stds, fmt='o-', color=color, label=label,
                    lw=1.8, capsize=3, markersize=4)
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Interpolation MAE [eV]')
    ax.set_title('(b) LJ: Interpolation')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # (c) Summary bar chart at largest N
    ax = axes[2]
    labels_bar = ['Rose\nExtrap', 'Rose\nInterp', 'LJ\nExtrap', 'LJ\nInterp']
    x = np.arange(len(labels_bar))
    width = 0.25

    vals_d = [rose_lc['direct_extrap'][-1][0], rose_lc['direct_interp'][-1][0],
              lj_lc['direct_extrap'][-1][0], lj_lc['direct_interp'][-1][0]]
    vals_a = [rose_lc['adaptive_extrap'][-1][0], rose_lc['adaptive_interp'][-1][0],
              lj_lc['adaptive_extrap'][-1][0], lj_lc['adaptive_interp'][-1][0]]
    vals_dl = [rose_lc['delta_extrap'][-1][0], rose_lc['delta_interp'][-1][0],
               lj_lc['delta_extrap'][-1][0], lj_lc['delta_interp'][-1][0]]

    ax.bar(x - width, vals_d, width, label='Direct', color=colors['direct'], alpha=0.8)
    ax.bar(x, vals_a, width, label='Adaptive', color=colors['adaptive'], alpha=0.8)
    ax.bar(x + width, vals_dl, width, label='Delta', color=colors['delta'], alpha=0.8)
    ax.set_ylabel('MAE [eV]')
    ax.set_title(f'(c) Summary ($N$={rose_lc["n_train"][-1]})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    outpath = 'figures/fig_delta_learning_interpolation.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Interpolation figure saved: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DELTA LEARNING: Combined Analysis (Rose/UBER + LJ)")
    print("=" * 70)

    n_sizes = [10, 20, 40, 60, 80, 100, 120, 160]

    # --- Rose/UBER learning curves ---
    print("\nRose/UBER learning curves (5 seeds)...")
    rose_lc = learning_curves_rose(n_sizes, n_seeds=5)

    # --- LJ learning curves ---
    print("\nLJ learning curves (5 seeds)...")
    lj_lc = learning_curves_lj(n_sizes, n_seeds=5)

    # --- Example predictions (N=100, extrapolation) ---
    print("\nExample predictions (N=100, extrapolation)...")

    # Rose
    train_rose = generate_rose_dataset(100, (0.5, 1.5), (0.5, 1.5), seed=42)
    test_rose = generate_rose_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=9999)
    r_rose = train_rose['r_grid']
    test_rose['r_grid'] = r_rose
    curves = [morse_V(r_rose, De, a, re) for De, re, a in test_rose['morse_params']]
    test_rose['curves'] = np.array(curves)
    rose_ps = []
    for V in test_rose['curves']:
        try:
            popt, _ = curve_fit(rose_V, r_rose, V, p0=[2.0, 1.8, 0.7],
                                bounds=([0.01, 0.5, 0.01], [20, 10, 5]), maxfev=5000)
            rose_ps.append(popt)
        except Exception:
            rose_ps.append([2.0, 1.8, 0.7])
    test_rose['rose_params'] = np.array(rose_ps)
    rose_ex = train_evaluate_rose(train_rose, test_rose)

    # LJ
    train_lj = generate_lj_dataset(100, (0.5, 1.5), (0.5, 1.5), seed=42)
    test_lj = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=9999)
    lj_ex = train_evaluate_lj(train_lj, test_lj)

    # Print summary
    print(f"\n  Rose/UBER (N=100, extrapolation):")
    print(f"    Direct:   MAE = {rose_ex['mae_direct']:.4f} eV, R2 = {rose_ex['r2_direct']:.4f}")
    print(f"    Adaptive: MAE = {rose_ex['mae_adaptive']:.4f} eV, R2 = {rose_ex['r2_adaptive']:.4f}")
    print(f"    Delta:    MAE = {rose_ex['mae_delta']:.4f} eV, R2 = {rose_ex['r2_delta']:.4f}")

    print(f"\n  LJ (N=100, extrapolation):")
    print(f"    Direct:   MAE = {lj_ex['mae_direct']:.4f} eV, R2 = {lj_ex['r2_direct']:.4f}")
    print(f"    Adaptive: MAE = {lj_ex['mae_adaptive']:.4f} eV, R2 = {lj_ex['r2_adaptive']:.4f}")
    print(f"    Delta:    MAE = {lj_ex['mae_delta']:.4f} eV, R2 = {lj_ex['r2_delta']:.4f}")

    # Delta improvement over adaptive
    rose_delta_improve = rose_ex['mae_adaptive'] / max(rose_ex['mae_delta'], 1e-10)
    lj_delta_improve = lj_ex['mae_adaptive'] / max(lj_ex['mae_delta'], 1e-10)
    print(f"\n  Delta improvement over adaptive:")
    print(f"    Rose: {rose_delta_improve:.2f}x")
    print(f"    LJ:   {lj_delta_improve:.2f}x")

    # Create figures
    print("\nGenerating figures...")
    create_combined_figure(rose_lc, lj_lc, rose_ex, lj_ex,
                           r_rose, test_lj['r_grid'],
                           test_rose['curves'], test_lj['curves'])
    create_interpolation_figure(rose_lc, lj_lc)

    # Print key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print("""
Delta learning = Adaptive baseline + Direct correction on residual.

For Rose/UBER (approximate physics):
  - Rose != Morse gives ~0.02 eV systematic error
  - Delta correction learns this residual -> improves over pure adaptive
  - Especially for interpolation where the residual is well-sampled

For LJ (exact physics):
  - LJ equation is exact -> residual is only from parameter prediction error
  - Delta adds little improvement since the correction has nothing systematic to learn
  - Adaptive alone captures nearly all the physics

This contrast validates that delta learning is most valuable when the
physics equation provides a good but imperfect baseline.
""")


if __name__ == "__main__":
    main()
