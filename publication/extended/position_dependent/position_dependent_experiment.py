"""
Position-dependent adaptive learning experiment (Step 1).

Compares three approaches for predicting Morse binding curves:
1. Direct: descriptor -> V(r) on fixed grid
2. Global Rose: descriptor -> (E_c, r_e, l) -> V(r) via Rose equation
3. Local parabola: descriptor -> (a_j, b_j, c_j) at each grid point -> V(r_j)

The local parabola at grid point r_j is:
    P_j(r) = a_j*(r-r_j)^2 + b_j*(r-r_j) + c_j

where a_j = V''(r_j)/2 (curvature), b_j = V'(r_j) (slope), c_j = V(r_j) (value).

Since c_j = V(r_j), the local parabola prediction at grid points reduces to
predicting V directly. This experiment tests whether this is indeed the case
for Ridge regression (linear model), and contrasts with global Rose where
the nonlinear physics equation provides a genuine advantage.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Physics functions ---

def morse(r, D_e, alpha, r_e):
    """Morse potential (shifted convention): V(r_e) = -D_e, V(inf) = 0."""
    x = np.exp(-alpha * (r - r_e))
    return D_e * (1 - x)**2 - D_e


def morse_d1(r, D_e, alpha, r_e):
    """dV/dr of Morse potential."""
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha * (1 - x) * x


def morse_d2(r, D_e, alpha, r_e):
    """d²V/dr² of Morse potential."""
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha**2 * x * (2 * x - 1)


def rose_potential(r, E_c, r_e, l):
    """Rose/UBER potential (shifted): V(r_e) = -E_c, V(inf) = 0."""
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star)) - E_c


def fit_rose_to_morse(r, V_morse, r_e_guess):
    """Fit Rose parameters to a Morse curve."""
    try:
        popt, _ = curve_fit(
            rose_potential, r, V_morse,
            p0=[abs(V_morse.min()), r_e_guess, 0.5],
            bounds=([0.01, 0.5, 0.01], [20.0, 10.0, 5.0]),
            maxfev=5000
        )
        return popt  # (E_c, r_e, l)
    except RuntimeError:
        return np.array([abs(V_morse.min()), r_e_guess, 0.5])


# --- Dataset generation ---

def generate_dataset(n_samples, d1_range, d2_range, r_grid, seed=42):
    """Generate Morse curves parameterized by descriptors (d1, d2).

    Mapping:
        D_e   = 1.0 + 2.0 * d1
        alpha = 0.8 + 0.6 * d2
        r_e   = 1.5 + 0.5 * d1
    """
    rng = np.random.RandomState(seed)
    d1 = rng.uniform(*d1_range, size=n_samples)
    d2 = rng.uniform(*d2_range, size=n_samples)

    descriptors = np.column_stack([d1, d2])

    D_e = 1.0 + 2.0 * d1
    alpha = 0.8 + 0.6 * d2
    r_e = 1.5 + 0.5 * d1

    n_grid = len(r_grid)
    curves = np.zeros((n_samples, n_grid))
    param_a = np.zeros((n_samples, n_grid))  # V''/2
    param_b = np.zeros((n_samples, n_grid))  # V'
    param_c = np.zeros((n_samples, n_grid))  # V
    rose_params = np.zeros((n_samples, 3))

    for i in range(n_samples):
        V = morse(r_grid, D_e[i], alpha[i], r_e[i])
        curves[i] = V
        param_a[i] = morse_d2(r_grid, D_e[i], alpha[i], r_e[i]) / 2.0
        param_b[i] = morse_d1(r_grid, D_e[i], alpha[i], r_e[i])
        param_c[i] = V  # c_j = V(r_j) by definition
        rose_params[i] = fit_rose_to_morse(r_grid, V, r_e[i])

    return {
        'descriptors': descriptors,
        'morse_params': np.column_stack([D_e, alpha, r_e]),
        'curves': curves,
        'param_a': param_a,
        'param_b': param_b,
        'param_c': param_c,
        'rose_params': rose_params,
    }


# --- ML comparison ---

def run_comparison(train, test, r_grid):
    """Run Ridge ML comparison: Direct vs Rose vs Local Parabola."""
    X_train = train['descriptors']
    X_test = test['descriptors']

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct: predict V(r) ---
    scaler_V = StandardScaler()
    Y_tr = scaler_V.fit_transform(train['curves'])
    model_V = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    V_pred_direct = scaler_V.inverse_transform(model_V.predict(X_te))
    mae_direct = np.mean(np.abs(test['curves'] - V_pred_direct))
    results['direct'] = {
        'V_pred': V_pred_direct,
        'mae': mae_direct,
        'per_sample_mae': np.mean(np.abs(test['curves'] - V_pred_direct), axis=1),
    }

    # --- 2. Global Rose: predict (E_c, r_e, l) -> V via Rose ---
    scaler_P = StandardScaler()
    Y_tr = scaler_P.fit_transform(train['rose_params'])
    model_P = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    params_pred = scaler_P.inverse_transform(model_P.predict(X_te))

    V_pred_rose = np.zeros_like(test['curves'])
    for i in range(len(X_test)):
        E_c, r_e, l = params_pred[i]
        E_c = max(E_c, 0.01)
        l = max(l, 0.01)
        V_pred_rose[i] = rose_potential(r_grid, E_c, r_e, l)

    mae_rose = np.mean(np.abs(test['curves'] - V_pred_rose))
    results['rose'] = {
        'V_pred': V_pred_rose,
        'params_pred': params_pred,
        'mae': mae_rose,
        'per_sample_mae': np.mean(np.abs(test['curves'] - V_pred_rose), axis=1),
    }

    # --- 3. Local parabola: predict (a_j, b_j, c_j) -> V(r_j) = c_j ---
    # 3a. Predict c_j only (should be identical to direct)
    scaler_c = StandardScaler()
    Y_tr = scaler_c.fit_transform(train['param_c'])
    model_c = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    c_pred = scaler_c.inverse_transform(model_c.predict(X_te))
    mae_parabola_c = np.mean(np.abs(test['curves'] - c_pred))

    # 3b. Predict a_j (curvature)
    scaler_a = StandardScaler()
    Y_tr = scaler_a.fit_transform(train['param_a'])
    model_a = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    a_pred = scaler_a.inverse_transform(model_a.predict(X_te))
    mae_a = np.mean(np.abs(test['param_a'] - a_pred))

    # 3c. Predict b_j (slope = force)
    scaler_b = StandardScaler()
    Y_tr = scaler_b.fit_transform(train['param_b'])
    model_b = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    b_pred = scaler_b.inverse_transform(model_b.predict(X_te))
    mae_b = np.mean(np.abs(test['param_b'] - b_pred))

    # 3d. Overlap reconstruction: use predicted parabolas from neighbors
    # At grid point k, average over parabolas from neighboring points
    V_pred_overlap = reconstruct_from_parabolas(
        r_grid, a_pred, b_pred, c_pred, n_neighbors=3)
    mae_overlap = np.mean(np.abs(test['curves'] - V_pred_overlap))

    results['parabola'] = {
        'V_pred': c_pred,
        'mae_V': mae_parabola_c,
        'a_pred': a_pred, 'mae_a': mae_a,
        'b_pred': b_pred, 'mae_b': mae_b,
        'V_pred_overlap': V_pred_overlap,
        'mae_overlap': mae_overlap,
        'per_sample_mae': np.mean(np.abs(test['curves'] - c_pred), axis=1),
    }

    # --- 4. Rose-derived forces and curvature (for comparison) ---
    a_rose = np.zeros_like(test['param_a'])
    b_rose = np.zeros_like(test['param_b'])
    for i in range(len(X_test)):
        E_c, r_e, l = params_pred[i]
        E_c = max(E_c, 0.01)
        l = max(l, 0.01)
        a_star = (r_grid - r_e) / l
        # Rose V' and V'' (analytical)
        b_rose[i] = (E_c / l) * a_star * np.exp(-a_star)
        a_rose[i] = (E_c / (2 * l**2)) * (1 - a_star) * np.exp(-a_star)

    mae_a_rose = np.mean(np.abs(test['param_a'] - a_rose))
    mae_b_rose = np.mean(np.abs(test['param_b'] - b_rose))

    results['rose_derivs'] = {
        'a_pred': a_rose, 'mae_a': mae_a_rose,
        'b_pred': b_rose, 'mae_b': mae_b_rose,
    }

    return results


def reconstruct_from_parabolas(r_grid, a_pred, b_pred, c_pred, n_neighbors=3):
    """Reconstruct V at each grid point using neighboring parabola predictions.

    For grid point k, average V estimates from the n_neighbors nearest parabolas:
        V_est_j(r_k) = a_j*(r_k - r_j)^2 + b_j*(r_k - r_j) + c_j
    """
    n_samples, n_grid = c_pred.shape
    V_recon = np.zeros_like(c_pred)

    half = n_neighbors // 2
    for k in range(n_grid):
        # Indices of neighboring parabolas to use
        j_start = max(0, k - half)
        j_end = min(n_grid, k + half + 1)

        V_estimates = []
        for j in range(j_start, j_end):
            dr = r_grid[k] - r_grid[j]
            V_est = a_pred[:, j] * dr**2 + b_pred[:, j] * dr + c_pred[:, j]
            V_estimates.append(V_est)

        V_recon[:, k] = np.mean(V_estimates, axis=0)

    return V_recon


# --- Plotting ---

def plot_results(train, test, results, r_grid, regime, filename):
    """Main results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    mae_d = results['direct']['mae']
    mae_r = results['rose']['mae']
    mae_p = results['parabola']['mae_V']
    mae_o = results['parabola']['mae_overlap']

    # (0,0): Example prediction curves
    ax = axes[0, 0]
    idx = np.argmax(results['direct']['per_sample_mae'])  # worst direct case
    ax.plot(r_grid, test['curves'][idx], 'k-', lw=2, label='True Morse')
    ax.plot(r_grid, results['direct']['V_pred'][idx], 'b--', lw=1.5,
            label=f'Direct ({mae_d:.4f})')
    ax.plot(r_grid, results['rose']['V_pred'][idx], 'r-', lw=1.5,
            label=f'Rose ({mae_r:.4f})')
    ax.plot(r_grid, results['parabola']['V_pred_overlap'][idx], 'g:', lw=1.5,
            label=f'Parabola overlap ({mae_o:.4f})')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title(f'Example prediction ({regime})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,1): MAE bar chart
    ax = axes[0, 1]
    methods = ['Direct\nV(r)', 'Rose\n(E_c,r_e,l)', 'Parabola\nc_j', 'Parabola\noverlap']
    maes = [mae_d, mae_r, mae_p, mae_o]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    bars = ax.bar(methods, maes, color=colors, alpha=0.8)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('MAE [eV]')
    ax.set_title(f'Energy MAE ({regime})')
    ax.grid(True, alpha=0.3, axis='y')

    # (0,2): Per-sample MAE comparison
    ax = axes[0, 2]
    ax.scatter(results['direct']['per_sample_mae'],
               results['rose']['per_sample_mae'],
               c='red', alpha=0.5, s=20, label='Rose')
    ax.scatter(results['direct']['per_sample_mae'],
               results['parabola']['per_sample_mae'],
               c='green', alpha=0.5, s=20, label='Parabola c_j')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('Direct MAE per sample [eV]')
    ax.set_ylabel('Adaptive MAE per sample [eV]')
    ax.set_title('Per-sample comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0): Curvature (a) prediction comparison
    ax = axes[1, 0]
    mae_a_direct = results['parabola']['mae_a']
    mae_a_rose = results['rose_derivs']['mae_a']
    ax.bar(['Direct\na prediction', 'Rose\nanalytical a'],
           [mae_a_direct, mae_a_rose],
           color=['#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel("MAE of V''(r)/2 [eV/Å²]")
    ax.set_title(f'Curvature a(r) MAE ({regime})')
    ratio_a = mae_a_direct / max(mae_a_rose, 1e-10)
    ax.text(0.5, 0.9, f'Direct/Rose = {ratio_a:.2f}x',
            transform=ax.transAxes, ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,1): Slope (b = force) prediction comparison
    ax = axes[1, 1]
    mae_b_direct = results['parabola']['mae_b']
    mae_b_rose = results['rose_derivs']['mae_b']
    ax.bar(['Direct\nb prediction', 'Rose\nanalytical b'],
           [mae_b_direct, mae_b_rose],
           color=['#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel("MAE of V'(r) [eV/Å]")
    ax.set_title(f'Slope b(r) = force MAE ({regime})')
    ratio_b = mae_b_direct / max(mae_b_rose, 1e-10)
    ax.text(0.5, 0.9, f'Direct/Rose = {ratio_b:.2f}x',
            transform=ax.transAxes, ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,2): Direct vs Parabola c_j — should be identical
    ax = axes[1, 2]
    diff = np.abs(results['direct']['V_pred'] - results['parabola']['V_pred'])
    ax.hist(diff.ravel(), bins=50, color='gray', alpha=0.7)
    ax.set_xlabel('|V_direct - V_parabola_c| [eV]')
    ax.set_ylabel('Count')
    ax.set_title(f'Direct vs Parabola c_j difference\nmax={diff.max():.2e}')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Position-dependent parabola vs Rose vs Direct — {regime}',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')
    plt.close(fig)


def plot_parameter_functions(train, test, results, r_grid, regime, filename):
    """Show the learned parabola parameter functions a(r), b(r) vs Rose."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Pick a few test samples
    n_show = min(5, len(test['descriptors']))
    cmap = plt.cm.tab10

    for i in range(n_show):
        color = cmap(i)
        d1, d2 = test['descriptors'][i]
        label = f'd1={d1:.1f}, d2={d2:.1f}'

        # a(r) = V''/2
        axes[0].plot(r_grid, test['param_a'][i], '-', color=color, lw=1.5,
                     label=label if i == 0 else None)
        axes[0].plot(r_grid, results['parabola']['a_pred'][i], '--',
                     color=color, alpha=0.6)
        axes[0].plot(r_grid, results['rose_derivs']['a_pred'][i], ':',
                     color=color, alpha=0.6)

        # b(r) = V'
        axes[1].plot(r_grid, test['param_b'][i], '-', color=color, lw=1.5)
        axes[1].plot(r_grid, results['parabola']['b_pred'][i], '--',
                     color=color, alpha=0.6)
        axes[1].plot(r_grid, results['rose_derivs']['b_pred'][i], ':',
                     color=color, alpha=0.6)

        # c(r) = V
        axes[2].plot(r_grid, test['param_c'][i], '-', color=color, lw=1.5)
        axes[2].plot(r_grid, results['parabola']['V_pred'][i], '--',
                     color=color, alpha=0.6)
        axes[2].plot(r_grid, results['rose']['V_pred'][i], ':',
                     color=color, alpha=0.6)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='black', lw=1.5, ls='-', label='True'),
        Line2D([0], [0], color='black', lw=1.5, ls='--', label='Direct/Parabola pred'),
        Line2D([0], [0], color='black', lw=1.5, ls=':', label='Rose analytical'),
    ]

    axes[0].set_ylabel("a(r) = V''(r)/2 [eV/Å²]")
    axes[0].set_title('Curvature')
    axes[0].axhline(0, color='gray', lw=0.5)

    axes[1].set_ylabel("b(r) = V'(r) [eV/Å]")
    axes[1].set_title('Slope (= force)')
    axes[1].axhline(0, color='gray', lw=0.5)

    axes[2].set_ylabel('c(r) = V(r) [eV]')
    axes[2].set_title('Value (= energy)')

    for ax in axes:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)

    axes[0].legend(handles=legend_lines, fontsize=8)

    fig.suptitle(
        f'Predicted parabola parameters — {regime}\n'
        '(solid=true, dashed=direct ML, dotted=Rose analytical)',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')
    plt.close(fig)


# --- Main ---

def main():
    # Fixed grid covering all molecules
    r_grid = np.linspace(1.0, 8.0, 50)

    # Generate datasets
    n_train = 100
    n_test = 50

    train = generate_dataset(n_train, (0.5, 1.5), (0.5, 1.5), r_grid, seed=42)
    test_extrap = generate_dataset(n_test, (2.5, 4.0), (0.5, 1.5), r_grid,
                                   seed=123)
    test_interp = generate_dataset(n_test, (0.7, 1.3), (0.6, 1.4), r_grid,
                                   seed=456)

    print('=' * 70)
    print('Position-dependent parabola experiment')
    print('=' * 70)

    print(f'\nTraining: {n_train} samples, d1 in [0.5, 1.5], d2 in [0.5, 1.5]')
    print(f'  D_e range: [{train["morse_params"][:,0].min():.2f}, '
          f'{train["morse_params"][:,0].max():.2f}]')
    print(f'  r_e range: [{train["morse_params"][:,2].min():.2f}, '
          f'{train["morse_params"][:,2].max():.2f}]')

    print(f'\nTest (extrap): {n_test} samples, d1 in [2.5, 4.0]')
    print(f'  D_e range: [{test_extrap["morse_params"][:,0].min():.2f}, '
          f'{test_extrap["morse_params"][:,0].max():.2f}]')
    print(f'  r_e range: [{test_extrap["morse_params"][:,2].min():.2f}, '
          f'{test_extrap["morse_params"][:,2].max():.2f}]')

    print(f'\nTest (interp): {n_test} samples, d1 in [0.7, 1.3]')

    # Run comparisons
    for regime, test_data in [('Extrapolation', test_extrap),
                               ('Interpolation', test_interp)]:
        print(f'\n{"="*50}')
        print(f'{regime}')
        print(f'{"="*50}')

        res = run_comparison(train, test_data, r_grid)

        print(f'\n  Energy MAE:')
        print(f'    Direct V(r):          {res["direct"]["mae"]:.6f} eV')
        print(f'    Rose (E_c,r_e,l):     {res["rose"]["mae"]:.6f} eV')
        print(f'    Parabola c_j:         {res["parabola"]["mae_V"]:.6f} eV')
        print(f'    Parabola overlap:     {res["parabola"]["mae_overlap"]:.6f} eV')

        ratio_rose = res['direct']['mae'] / max(res['rose']['mae'], 1e-10)
        ratio_para = res['direct']['mae'] / max(res['parabola']['mae_V'], 1e-10)

        print(f'\n  Ratios (Direct/Adaptive):')
        print(f'    Rose:       {ratio_rose:.2f}x')
        print(f'    Parabola:   {ratio_para:.4f}x  '
              '(should be ~1.0 for Ridge)')

        diff_dp = np.max(np.abs(
            res['direct']['V_pred'] - res['parabola']['V_pred']))
        print(f'\n  Max |V_direct - V_parabola_c|: {diff_dp:.2e} eV')

        print(f'\n  Derivative MAE:')
        print(f'    a (curvature) - Direct ML:   {res["parabola"]["mae_a"]:.6f}')
        print(f'    a (curvature) - Rose analyt:  {res["rose_derivs"]["mae_a"]:.6f}')
        ratio_a = res['parabola']['mae_a'] / max(res['rose_derivs']['mae_a'], 1e-10)
        print(f'    Ratio Direct/Rose: {ratio_a:.2f}x')

        print(f'    b (slope/force) - Direct ML: {res["parabola"]["mae_b"]:.6f}')
        print(f'    b (slope/force) - Rose analyt: {res["rose_derivs"]["mae_b"]:.6f}')
        ratio_b = res['parabola']['mae_b'] / max(res['rose_derivs']['mae_b'], 1e-10)
        print(f'    Ratio Direct/Rose: {ratio_b:.2f}x')

        tag = regime.lower()[:6]
        plot_results(train, test_data, res, r_grid, regime,
                     f'fig_parabola_comparison_{tag}.png')
        plot_parameter_functions(train, test_data, res, r_grid, regime,
                                 f'fig_parabola_params_{tag}.png')


if __name__ == '__main__':
    main()
