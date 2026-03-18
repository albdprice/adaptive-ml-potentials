"""
Force prediction learning curves: Force MAE vs training set size.

Shows how force prediction accuracy scales with training data for
both direct (numerical differentiation) and adaptive (analytic) approaches.

Usage:
    MPLBACKEND=Agg python force_learning_curves.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# =============================================================================
# Potential functions and derivatives (dV/dr convention)
# =============================================================================

def lj_V(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def lj_F(r, epsilon, sigma):
    """dV/dr for LJ."""
    return 4 * epsilon * (-12 * sigma**12 / r**13 + 6 * sigma**6 / r**7)

def morse_V(r, De, a, re):
    return De * (1 - np.exp(-a * (r - re)))**2

def morse_F(r, De, a, re):
    """dV/dr for Morse."""
    return 2 * De * a * np.exp(-a * (r - re)) * (1 - np.exp(-a * (r - re)))

def rose_V(r, E_c, r_e, l):
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star))

def rose_F(r, E_c, r_e, l):
    """dV/dr for Rose/UBER."""
    a_star = (r - r_e) / l
    return E_c * a_star * np.exp(-a_star) / l

def numerical_force(r, V_pred):
    """dV/dr via central differences."""
    F = np.zeros_like(V_pred)
    dr = r[1] - r[0]
    F[1:-1] = (V_pred[2:] - V_pred[:-2]) / (2 * dr)
    F[0] = (V_pred[1] - V_pred[0]) / dr
    F[-1] = (V_pred[-1] - V_pred[-2]) / dr
    return F

# =============================================================================
# Dataset generation
# =============================================================================

def generate_lj_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r = np.linspace(2.5, 12.0, n_points)
    X, params, curves, forces = [], [], [], []
    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2
        X.append([d1, d2])
        params.append([epsilon, sigma])
        curves.append(lj_V(r, epsilon, sigma))
        forces.append(lj_F(r, epsilon, sigma))
    return {'X': np.array(X), 'params': np.array(params),
            'curves': np.array(curves), 'forces': np.array(forces), 'r': r}

def generate_rose_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, morse_params, rose_params, curves, forces, r_grids = [], [], [], [], [], []
    a_fixed = 1.5
    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        De = 0.5 + 2.0 * d1
        re = 1.5 + 0.5 * d2
        r = np.linspace(0.7 * re, 3.0 * re, n_points)
        V = morse_V(r, De, a_fixed, re)
        F = morse_F(r, De, a_fixed, re)
        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1.0 / a_fixed],
                                bounds=([0.01, 0.5, 0.01], [20, 10, 5]), maxfev=5000)
            X.append([d1, d2])
            morse_params.append([De, re, a_fixed])
            rose_params.append(popt)
            curves.append(V)
            forces.append(F)
            r_grids.append(r)
        except RuntimeError:
            pass
    return {'X': np.array(X), 'morse_params': np.array(morse_params),
            'rose_params': np.array(rose_params), 'curves': np.array(curves),
            'forces': np.array(forces), 'r_grids': r_grids}

# =============================================================================
# Train and evaluate forces
# =============================================================================

def evaluate_forces_lj(train_data, test_data):
    """Train direct/adaptive on LJ, return force and energy MAE."""
    X_train, X_test = train_data['X'], test_data['X']
    Y_train = train_data['curves']
    Y_test = test_data['curves']
    F_test_true = test_data['forces']
    r = test_data['r']

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    # Direct
    scaler_V = StandardScaler()
    Y_tr_s = scaler_V.fit_transform(Y_train)
    model_d = Ridge(alpha=1.0).fit(X_tr, Y_tr_s)
    V_pred_d = scaler_V.inverse_transform(model_d.predict(X_te))
    F_pred_d = np.array([numerical_force(r, V_pred_d[i]) for i in range(len(V_pred_d))])

    # Adaptive
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(train_data['params'])
    model_a = Ridge(alpha=1.0).fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(model_a.predict(X_te))
    V_pred_a = np.array([lj_V(r, max(e, 0.001), max(s, 0.1)) for e, s in P_pred])
    F_pred_a = np.array([lj_F(r, max(e, 0.001), max(s, 0.1)) for e, s in P_pred])

    # Metrics (exclude boundary points for force)
    sl = slice(2, -2)
    return {
        'mae_V_direct': mean_absolute_error(Y_test, V_pred_d),
        'mae_V_adaptive': mean_absolute_error(Y_test, V_pred_a),
        'mae_F_direct': mean_absolute_error(F_test_true[:, sl], F_pred_d[:, sl]),
        'mae_F_adaptive': mean_absolute_error(F_test_true[:, sl], F_pred_a[:, sl]),
        'r2_F_direct': r2_score(F_test_true[:, sl].flatten(), F_pred_d[:, sl].flatten()),
        'r2_F_adaptive': r2_score(F_test_true[:, sl].flatten(), F_pred_a[:, sl].flatten()),
    }

def evaluate_forces_rose(train_data, test_data):
    """Train direct/adaptive on Rose/UBER, return force and energy MAE."""
    X_train, X_test = train_data['X'], test_data['X']
    Y_train = train_data['curves']
    Y_test = test_data['curves']
    F_test_true = test_data['forces']

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    # Direct
    scaler_V = StandardScaler()
    Y_tr_s = scaler_V.fit_transform(Y_train)
    model_d = Ridge(alpha=1.0).fit(X_tr, Y_tr_s)
    V_pred_d = scaler_V.inverse_transform(model_d.predict(X_te))
    F_pred_d = np.array([numerical_force(test_data['r_grids'][i], V_pred_d[i])
                          for i in range(len(V_pred_d))])

    # Adaptive
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(train_data['rose_params'])
    model_a = Ridge(alpha=1.0).fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(model_a.predict(X_te))
    V_pred_a, F_pred_a = [], []
    for i, (E_c, r_e, l) in enumerate(P_pred):
        r_i = test_data['r_grids'][i]
        V_pred_a.append(rose_V(r_i, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
        F_pred_a.append(rose_F(r_i, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
    V_pred_a = np.array(V_pred_a)
    F_pred_a = np.array(F_pred_a)

    sl = slice(2, -2)
    return {
        'mae_V_direct': mean_absolute_error(Y_test, V_pred_d),
        'mae_V_adaptive': mean_absolute_error(Y_test, V_pred_a),
        'mae_F_direct': mean_absolute_error(F_test_true[:, sl], F_pred_d[:, sl]),
        'mae_F_adaptive': mean_absolute_error(F_test_true[:, sl], F_pred_a[:, sl]),
        'r2_F_direct': r2_score(F_test_true[:, sl].flatten(), F_pred_d[:, sl].flatten()),
        'r2_F_adaptive': r2_score(F_test_true[:, sl].flatten(), F_pred_a[:, sl].flatten()),
    }

# =============================================================================
# Learning curves
# =============================================================================

def run_force_learning_curves(system='lj', regime='extrap', n_seeds=5):
    train_sizes = [10, 15, 20, 30, 40, 60, 80, 100, 120, 160]
    metrics = ['mae_F_direct', 'mae_F_adaptive', 'mae_V_direct', 'mae_V_adaptive']
    lc = {m: {'mean': [], 'std': []} for m in metrics}

    for n_train in train_sizes:
        seed_results = {m: [] for m in metrics}
        for s in range(n_seeds):
            seed = s * 100 + n_train
            if system == 'lj':
                train = generate_lj_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
                if regime == 'extrap':
                    test = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=seed + 10000)
                else:
                    test = generate_lj_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
                res = evaluate_forces_lj(train, test)
            else:
                train = generate_rose_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
                if regime == 'extrap':
                    test = generate_rose_dataset(50, (3.0, 4.5), (0.5, 1.5), seed=seed + 10000)
                else:
                    test = generate_rose_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
                res = evaluate_forces_rose(train, test)
            for m in metrics:
                seed_results[m].append(res[m])

        for m in metrics:
            lc[m]['mean'].append(np.mean(seed_results[m]))
            lc[m]['std'].append(np.std(seed_results[m]))

        print(f"  N={n_train:3d}: F_direct={np.mean(seed_results['mae_F_direct']):.4f}, "
              f"F_adaptive={np.mean(seed_results['mae_F_adaptive']):.4f}, "
              f"ratio={np.mean(seed_results['mae_F_direct'])/np.mean(seed_results['mae_F_adaptive']):.1f}x")

    return train_sizes, lc

# =============================================================================
# Plotting
# =============================================================================

def plot_force_learning_curves(lj_extrap, lj_interp, rose_extrap, rose_interp, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    datasets = [
        (lj_extrap, 'LJ Extrapolation', axes[0, 0]),
        (lj_interp, 'LJ Interpolation', axes[0, 1]),
        (rose_extrap, 'Rose/UBER Extrapolation', axes[1, 0]),
        (rose_interp, 'Rose/UBER Interpolation', axes[1, 1]),
    ]

    for (train_sizes, lc), title, ax in datasets:
        # Force MAE
        for key, label, color, marker in [
            ('mae_F_direct', 'Direct (numerical dV/dr)', '#1f77b4', 's'),
            ('mae_F_adaptive', 'Adaptive (analytic dV/dr)', '#d62728', 'o'),
        ]:
            means = np.array(lc[key]['mean'])
            stds = np.array(lc[key]['std'])
            ax.errorbar(train_sizes, means, yerr=stds, label=label, color=color,
                        marker=marker, markersize=6, capsize=3, linewidth=1.5)

        ax.set_xlabel('Training set size N')
        ax.set_ylabel('Force MAE (eV/\u00c5)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


def plot_combined_energy_force(lj_extrap, rose_extrap, savepath):
    """2x2: energy and force learning curves side by side for both systems."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (data, sys_name) in enumerate([(lj_extrap, 'LJ'), (rose_extrap, 'Rose/UBER')]):
        train_sizes, lc = data

        # Left: Energy
        ax = axes[row, 0]
        for key, label, color, marker in [
            ('mae_V_direct', 'Direct', '#1f77b4', 's'),
            ('mae_V_adaptive', 'Adaptive', '#d62728', 'o'),
        ]:
            means = np.array(lc[key]['mean'])
            stds = np.array(lc[key]['std'])
            ax.errorbar(train_sizes, means, yerr=stds, label=label, color=color,
                        marker=marker, markersize=6, capsize=3, linewidth=1.5)
        ax.set_xlabel('Training set size N')
        ax.set_ylabel('Energy MAE (eV)')
        ax.set_title(f'{sys_name}: Energy (Extrapolation)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: Force
        ax = axes[row, 1]
        for key, label, color, marker in [
            ('mae_F_direct', 'Direct (numerical)', '#1f77b4', 's'),
            ('mae_F_adaptive', 'Adaptive (analytic)', '#d62728', 'o'),
        ]:
            means = np.array(lc[key]['mean'])
            stds = np.array(lc[key]['std'])
            ax.errorbar(train_sizes, means, yerr=stds, label=label, color=color,
                        marker=marker, markersize=6, capsize=3, linewidth=1.5)
        ax.set_xlabel('Training set size N')
        ax.set_ylabel('Force MAE (eV/\u00c5)')
        ax.set_title(f'{sys_name}: Force (Extrapolation)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    figdir = 'figures'

    print("=" * 60)
    print("FORCE LEARNING CURVES: LJ Extrapolation")
    print("=" * 60)
    lj_extrap = run_force_learning_curves('lj', 'extrap')

    print("\n" + "=" * 60)
    print("FORCE LEARNING CURVES: LJ Interpolation")
    print("=" * 60)
    lj_interp = run_force_learning_curves('lj', 'interp')

    print("\n" + "=" * 60)
    print("FORCE LEARNING CURVES: Rose/UBER Extrapolation")
    print("=" * 60)
    rose_extrap = run_force_learning_curves('rose', 'extrap')

    print("\n" + "=" * 60)
    print("FORCE LEARNING CURVES: Rose/UBER Interpolation")
    print("=" * 60)
    rose_interp = run_force_learning_curves('rose', 'interp')

    # 4-panel force-only learning curves
    plot_force_learning_curves(lj_extrap, lj_interp, rose_extrap, rose_interp,
                               f'{figdir}/fig_force_learning_curves.png')

    # 2x2 energy+force for extrapolation (publication figure)
    plot_combined_energy_force(lj_extrap, rose_extrap,
                               f'{figdir}/fig_force_energy_learning_curves.png')

    # Summary at N=160
    print("\n" + "=" * 60)
    print("SUMMARY: Force MAE at N=160 (extrapolation)")
    print("=" * 60)
    for name, (ts, lc) in [('LJ', lj_extrap), ('Rose/UBER', rose_extrap)]:
        fd = lc['mae_F_direct']['mean'][-1]
        fa = lc['mae_F_adaptive']['mean'][-1]
        print(f"  {name}: Direct={fd:.4f}, Adaptive={fa:.4f}, Ratio={fd/fa:.1f}x")
