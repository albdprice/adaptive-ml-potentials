"""
ML experiment: b=0 position-dependent parabola for Morse curves.

Direct:   descriptor -> V(r) on grid
Adaptive: descriptor -> r_e + a(r_j) at each grid point
          then reconstruct V(r_j) = a_j * (r_j - r_e)^2

Uses unshifted Morse convention: V(r_e) = 0, V(inf) = D_e.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


# --- Physics ---

def morse_unshifted(r, D_e, alpha, r_e):
    """Morse: V(r_e) = 0, V(inf) = D_e."""
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2


def compute_a_of_r(r, D_e, alpha, r_e):
    """a(r) = V(r) / (r - r_e)^2, with L'Hopital at r=r_e."""
    u = alpha * (r - r_e)
    f = np.where(np.abs(u) < 1e-8,
                 1.0 - u / 2.0 + u**2 / 6.0,
                 (1 - np.exp(-u)) / u)
    return D_e * alpha**2 * f**2


# --- Dataset ---

def generate_dataset(n_samples, d1_range, d2_range, r_grid, seed=42):
    """Generate Morse curves with descriptors.

    Mapping: D_e = 1 + 2*d1, alpha = 0.8 + 0.6*d2, r_e = 1.5 + 0.5*d1
    """
    rng = np.random.RandomState(seed)
    d1 = rng.uniform(*d1_range, size=n_samples)
    d2 = rng.uniform(*d2_range, size=n_samples)

    D_e = 1.0 + 2.0 * d1
    alpha = 0.8 + 0.6 * d2
    r_e = 1.5 + 0.5 * d1

    n_grid = len(r_grid)
    curves = np.zeros((n_samples, n_grid))
    a_values = np.zeros((n_samples, n_grid))

    for i in range(n_samples):
        curves[i] = morse_unshifted(r_grid, D_e[i], alpha[i], r_e[i])
        a_values[i] = compute_a_of_r(r_grid, D_e[i], alpha[i], r_e[i])

    return {
        'descriptors': np.column_stack([d1, d2]),
        'D_e': D_e, 'alpha': alpha, 'r_e': r_e,
        'curves': curves,
        'a_values': a_values,
        'sqrt_a_values': np.sqrt(a_values),
    }


# --- ML ---

def run_ml(train, test, r_grid):
    """Compare Direct vs Adaptive (b=0 parabola)."""
    X_train = train['descriptors']
    X_test = test['descriptors']
    n_test = len(X_test)

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct: predict V(r) ---
    scaler_V = StandardScaler()
    Y_tr = scaler_V.fit_transform(train['curves'])
    model_V = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    V_pred_direct = scaler_V.inverse_transform(model_V.predict(X_te))

    results['direct'] = {
        'V_pred': V_pred_direct,
        'mae': np.mean(np.abs(test['curves'] - V_pred_direct)),
        'per_sample': np.mean(np.abs(test['curves'] - V_pred_direct), axis=1),
    }

    # --- 2. Adaptive: predict r_e + a(r_j), reconstruct V = a*(r-r_e)^2 ---
    # Predict r_e
    scaler_re = StandardScaler()
    re_tr = scaler_re.fit_transform(train['r_e'].reshape(-1, 1))
    model_re = Ridge(alpha=1.0).fit(X_tr, re_tr)
    re_pred = scaler_re.inverse_transform(model_re.predict(X_te)).ravel()

    # Predict a(r_j)
    scaler_a = StandardScaler()
    Y_tr = scaler_a.fit_transform(train['a_values'])
    model_a = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    a_pred = scaler_a.inverse_transform(model_a.predict(X_te))

    # Reconstruct V = a * (r - r_e)^2
    V_pred_adaptive = np.zeros((n_test, len(r_grid)))
    for i in range(n_test):
        V_pred_adaptive[i] = a_pred[i] * (r_grid - re_pred[i])**2

    results['adaptive_a'] = {
        'V_pred': V_pred_adaptive,
        'mae': np.mean(np.abs(test['curves'] - V_pred_adaptive)),
        'per_sample': np.mean(np.abs(test['curves'] - V_pred_adaptive), axis=1),
        're_pred': re_pred,
        'a_pred': a_pred,
        're_mae': np.mean(np.abs(test['r_e'] - re_pred)),
        'a_mae': np.mean(np.abs(test['a_values'] - a_pred)),
    }

    # --- 3. Adaptive with sqrt(a): predict r_e + sqrt(a), reconstruct ---
    scaler_sa = StandardScaler()
    Y_tr = scaler_sa.fit_transform(train['sqrt_a_values'])
    model_sa = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    sqrt_a_pred = scaler_sa.inverse_transform(model_sa.predict(X_te))

    V_pred_sqrt = np.zeros((n_test, len(r_grid)))
    for i in range(n_test):
        V_pred_sqrt[i] = sqrt_a_pred[i]**2 * (r_grid - re_pred[i])**2

    results['adaptive_sqrt_a'] = {
        'V_pred': V_pred_sqrt,
        'mae': np.mean(np.abs(test['curves'] - V_pred_sqrt)),
        'per_sample': np.mean(np.abs(test['curves'] - V_pred_sqrt), axis=1),
        'sqrt_a_pred': sqrt_a_pred,
    }

    return results


# --- Plotting ---

def plot_results(train, test, results, r_grid, regime):
    """Results figure for one regime."""
    mae_d = results['direct']['mae']
    mae_a = results['adaptive_a']['mae']
    mae_s = results['adaptive_sqrt_a']['mae']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (0,0): Bar chart of MAEs
    ax = axes[0, 0]
    methods = ['Direct\nV(r)', 'Adaptive\na(r) + r_e', 'Adaptive\nsqrt(a) + r_e']
    maes = [mae_d, mae_a, mae_s]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    bars = ax.bar(methods, maes, color=colors, alpha=0.8)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('MAE [eV]')
    ax.set_title(f'Energy MAE — {regime}')
    ax.grid(True, alpha=0.3, axis='y')

    # (0,1): Example — worst direct prediction
    ax = axes[0, 1]
    idx = np.argmax(results['direct']['per_sample'])
    ax.plot(r_grid, test['curves'][idx], 'k-', lw=2, label='True')
    ax.plot(r_grid, results['direct']['V_pred'][idx], '--',
            color='#1f77b4', lw=1.5, label=f'Direct')
    ax.plot(r_grid, results['adaptive_a']['V_pred'][idx], '-',
            color='#2ca02c', lw=1.5, label=f'Adaptive a(r)')
    ax.plot(r_grid, results['adaptive_sqrt_a']['V_pred'][idx], ':',
            color='#d62728', lw=1.5, label=f'Adaptive sqrt(a)')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title(f'Hardest test case (worst direct error)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2): Example — median direct prediction
    ax = axes[0, 2]
    idx = np.argsort(results['direct']['per_sample'])[len(test['r_e'])//2]
    ax.plot(r_grid, test['curves'][idx], 'k-', lw=2, label='True')
    ax.plot(r_grid, results['direct']['V_pred'][idx], '--',
            color='#1f77b4', lw=1.5, label='Direct')
    ax.plot(r_grid, results['adaptive_a']['V_pred'][idx], '-',
            color='#2ca02c', lw=1.5, label='Adaptive a(r)')
    ax.plot(r_grid, results['adaptive_sqrt_a']['V_pred'][idx], ':',
            color='#d62728', lw=1.5, label='Adaptive sqrt(a)')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('Median test case')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0): Per-sample scatter
    ax = axes[1, 0]
    ax.scatter(results['direct']['per_sample'],
               results['adaptive_a']['per_sample'],
               c='#2ca02c', alpha=0.5, s=25, label='a(r) adaptive')
    ax.scatter(results['direct']['per_sample'],
               results['adaptive_sqrt_a']['per_sample'],
               c='#d62728', alpha=0.5, s=25, marker='^',
               label='sqrt(a) adaptive')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='equal')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Direct MAE [eV]')
    ax.set_ylabel('Adaptive MAE [eV]')
    ax.set_title('Per-sample: below diagonal = adaptive wins')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): r_e prediction quality
    ax = axes[1, 1]
    ax.scatter(test['r_e'], results['adaptive_a']['re_pred'],
               c='#2ca02c', alpha=0.5, s=25)
    lims = [min(test['r_e'].min(), results['adaptive_a']['re_pred'].min()),
            max(test['r_e'].max(), results['adaptive_a']['re_pred'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('True r_e [Å]')
    ax.set_ylabel('Predicted r_e [Å]')
    re_mae = results['adaptive_a']['re_mae']
    ax.set_title(f'r_e prediction (MAE = {re_mae:.4f} Å)')
    ax.grid(True, alpha=0.3)

    # (1,2): a(r) prediction at one grid point
    ax = axes[1, 2]
    j_mid = len(r_grid) // 4  # a grid point in the well region
    ax.scatter(test['a_values'][:, j_mid],
               results['adaptive_a']['a_pred'][:, j_mid],
               c='#2ca02c', alpha=0.5, s=25)
    lims = [min(test['a_values'][:, j_mid].min(),
                results['adaptive_a']['a_pred'][:, j_mid].min()),
            max(test['a_values'][:, j_mid].max(),
                results['adaptive_a']['a_pred'][:, j_mid].max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel(f'True a(r={r_grid[j_mid]:.1f} Å)')
    ax.set_ylabel(f'Predicted a(r={r_grid[j_mid]:.1f} Å)')
    ax.set_title(f'a(r) prediction at r={r_grid[j_mid]:.1f} Å')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'b=0 adaptive parabola ML — {regime}\n'
        f'Direct MAE = {mae_d:.4f},  '
        f'Adaptive a(r) MAE = {mae_a:.4f} ({mae_d/mae_a:.2f}x),  '
        f'Adaptive sqrt(a) MAE = {mae_s:.4f} ({mae_d/mae_s:.2f}x)',
        fontsize=11)
    fig.tight_layout()
    fname = f'fig_b0_ml_{regime.lower().replace(" ", "_")}.png'
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close(fig)


def main():
    r_grid = np.linspace(1.0, 8.0, 50)
    n_train, n_test = 100, 50

    train = generate_dataset(n_train, (0.5, 1.5), (0.5, 1.5), r_grid, seed=42)
    test_ext = generate_dataset(n_test, (2.5, 4.0), (0.5, 1.5), r_grid, seed=123)
    test_int = generate_dataset(n_test, (0.7, 1.3), (0.6, 1.4), r_grid, seed=456)

    print('=' * 60)
    print('b=0 adaptive parabola: V(r) = a(r) * (r - r_e)^2')
    print('=' * 60)
    print(f'\nTraining: {n_train} curves, d1 in [0.5,1.5], d2 in [0.5,1.5]')
    print(f'  D_e: [{train["D_e"].min():.2f}, {train["D_e"].max():.2f}]')
    print(f'  r_e: [{train["r_e"].min():.2f}, {train["r_e"].max():.2f}]')
    print(f'  alpha: [{train["alpha"].min():.2f}, {train["alpha"].max():.2f}]')

    for regime, test in [('Extrapolation', test_ext),
                         ('Interpolation', test_int)]:
        print(f'\n{"="*50}')
        print(f'  {regime}')
        print(f'{"="*50}')
        print(f'  Test: D_e [{test["D_e"].min():.2f}, {test["D_e"].max():.2f}], '
              f'r_e [{test["r_e"].min():.2f}, {test["r_e"].max():.2f}]')

        res = run_ml(train, test, r_grid)

        mae_d = res['direct']['mae']
        mae_a = res['adaptive_a']['mae']
        mae_s = res['adaptive_sqrt_a']['mae']

        print(f'\n  Energy MAE [eV]:')
        print(f'    Direct V(r):           {mae_d:.6f}')
        print(f'    Adaptive a(r) + r_e:   {mae_a:.6f}  '
              f'(ratio = {mae_d/mae_a:.2f}x)')
        print(f'    Adaptive sqrt(a) + r_e: {mae_s:.6f}  '
              f'(ratio = {mae_d/mae_s:.2f}x)')
        print(f'\n  r_e MAE: {res["adaptive_a"]["re_mae"]:.6f} Å')
        print(f'  a(r) MAE: {res["adaptive_a"]["a_mae"]:.4f} eV/Å²')

        plot_results(train, test, res, r_grid, regime)


if __name__ == '__main__':
    main()
