"""
Step 2: LJ as approximate physics for Morse curves.

Reference: Morse curves (ground truth)
Model: LJ potential (approximate — wrong functional form)

Direct:   descriptor -> V_Morse(r) on grid
Adaptive: descriptor -> (epsilon, sigma) -> V_LJ(r)

Since LJ != Morse, the adaptive approach uses approximate physics.
The question: does learning 2 parameters + approximate physics still
beat learning 50 grid values directly?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


# --- Physics ---

def morse_shifted(r, D_e, alpha, r_e):
    """Morse: V(r_e) = -D_e, V(inf) = 0."""
    x = np.exp(-alpha * (r - r_e))
    return D_e * (1 - x)**2 - D_e


def lennard_jones(r, epsilon, sigma):
    """LJ: V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]."""
    sr = sigma / r
    return 4 * epsilon * (sr**12 - sr**6)


def fit_lj_to_morse(r, V_morse):
    """Fit LJ (epsilon, sigma) to a Morse curve by least squares.

    Initial guess from Morse minimum: epsilon ~ |V_min|, sigma ~ r_min / 2^(1/6)
    """
    idx_min = np.argmin(V_morse)
    eps_guess = abs(V_morse[idx_min])
    sig_guess = r[idx_min] / 2**(1.0/6.0)

    try:
        popt, _ = curve_fit(
            lennard_jones, r, V_morse,
            p0=[eps_guess, sig_guess],
            bounds=([0.001, 0.1], [50.0, 10.0]),
            maxfev=10000
        )
        return popt  # (epsilon, sigma)
    except RuntimeError:
        return np.array([eps_guess, sig_guess])


# --- Dataset ---

def generate_dataset(n_samples, d1_range, d2_range, r_grid, seed=42):
    """Generate Morse curves and fit LJ parameters to each.

    Morse mapping (same as b0 experiment):
        D_e   = 1.0 + 2.0 * d1
        alpha = 0.8 + 0.6 * d2
        r_e   = 1.5 + 0.5 * d1
    """
    rng = np.random.RandomState(seed)
    d1 = rng.uniform(*d1_range, size=n_samples)
    d2 = rng.uniform(*d2_range, size=n_samples)

    D_e = 1.0 + 2.0 * d1
    alpha = 0.8 + 0.6 * d2
    r_e = 1.5 + 0.5 * d1

    n_grid = len(r_grid)
    curves = np.zeros((n_samples, n_grid))
    lj_params = np.zeros((n_samples, 2))  # (epsilon, sigma)
    lj_curves = np.zeros((n_samples, n_grid))
    fit_errors = np.zeros(n_samples)

    for i in range(n_samples):
        V = morse_shifted(r_grid, D_e[i], alpha[i], r_e[i])
        curves[i] = V
        eps_fit, sig_fit = fit_lj_to_morse(r_grid, V)
        lj_params[i] = [eps_fit, sig_fit]
        lj_curves[i] = lennard_jones(r_grid, eps_fit, sig_fit)
        fit_errors[i] = np.mean(np.abs(V - lj_curves[i]))

    return {
        'descriptors': np.column_stack([d1, d2]),
        'D_e': D_e, 'alpha': alpha, 'r_e': r_e,
        'curves': curves,          # Morse (ground truth)
        'lj_params': lj_params,    # best-fit LJ params per curve
        'lj_curves': lj_curves,    # best-fit LJ curves
        'fit_errors': fit_errors,   # LJ approximation error
    }


# --- ML ---

def run_ml(train, test, r_grid):
    """Compare Direct vs LJ-Adaptive."""
    X_train = train['descriptors']
    X_test = test['descriptors']
    n_test = len(X_test)

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct: predict V_Morse(r) ---
    scaler_V = StandardScaler()
    Y_tr = scaler_V.fit_transform(train['curves'])
    model_V = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    V_pred_direct = scaler_V.inverse_transform(model_V.predict(X_te))

    results['direct'] = {
        'V_pred': V_pred_direct,
        'mae': np.mean(np.abs(test['curves'] - V_pred_direct)),
        'per_sample': np.mean(np.abs(test['curves'] - V_pred_direct), axis=1),
    }

    # --- 2. LJ-Adaptive: predict (epsilon, sigma) -> V_LJ(r) ---
    scaler_P = StandardScaler()
    Y_tr = scaler_P.fit_transform(train['lj_params'])
    model_P = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    params_pred = scaler_P.inverse_transform(model_P.predict(X_te))

    V_pred_lj = np.zeros((n_test, len(r_grid)))
    for i in range(n_test):
        eps, sig = params_pred[i]
        eps = max(eps, 0.001)
        sig = max(sig, 0.1)
        V_pred_lj[i] = lennard_jones(r_grid, eps, sig)

    # MAE vs Morse (ground truth)
    mae_lj_vs_morse = np.mean(np.abs(test['curves'] - V_pred_lj))
    # MAE vs best-fit LJ (to separate ML error from physics error)
    mae_lj_vs_bestfit = np.mean(np.abs(test['lj_curves'] - V_pred_lj))

    results['lj_adaptive'] = {
        'V_pred': V_pred_lj,
        'params_pred': params_pred,
        'mae_vs_morse': mae_lj_vs_morse,
        'mae_vs_bestfit': mae_lj_vs_bestfit,
        'per_sample': np.mean(np.abs(test['curves'] - V_pred_lj), axis=1),
        'eps_mae': np.mean(np.abs(test['lj_params'][:, 0] - params_pred[:, 0])),
        'sig_mae': np.mean(np.abs(test['lj_params'][:, 1] - params_pred[:, 1])),
    }

    # --- Physics error: best-fit LJ vs Morse (irreducible) ---
    results['physics_error'] = {
        'mae': np.mean(test['fit_errors']),
        'per_sample': test['fit_errors'],
    }

    return results


# --- Plotting ---

def plot_results(train, test, results, r_grid, regime):
    """Results figure."""
    mae_d = results['direct']['mae']
    mae_lj = results['lj_adaptive']['mae_vs_morse']
    mae_phys = results['physics_error']['mae']
    mae_lj_bf = results['lj_adaptive']['mae_vs_bestfit']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (0,0): MAE bar chart
    ax = axes[0, 0]
    methods = ['Direct\nV(r)', 'LJ adaptive\nvs Morse', 'LJ best-fit\n(physics limit)']
    maes = [mae_d, mae_lj, mae_phys]
    colors = ['#1f77b4', '#2ca02c', '#aaaaaa']
    bars = ax.bar(methods, maes, color=colors, alpha=0.8)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('MAE vs Morse [eV]')
    ax.set_title(f'Energy MAE — {regime}')
    ax.grid(True, alpha=0.3, axis='y')

    # (0,1): Example — worst direct case
    ax = axes[0, 1]
    idx = np.argmax(results['direct']['per_sample'])
    ax.plot(r_grid, test['curves'][idx], 'k-', lw=2, label='Morse (true)')
    ax.plot(r_grid, results['direct']['V_pred'][idx], '--',
            color='#1f77b4', lw=1.5, label='Direct')
    ax.plot(r_grid, results['lj_adaptive']['V_pred'][idx], '-',
            color='#2ca02c', lw=1.5, label='LJ adaptive')
    ax.plot(r_grid, test['lj_curves'][idx], ':',
            color='gray', lw=1.5, label='LJ best-fit')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('Hardest test case')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2): Example — median case
    ax = axes[0, 2]
    idx = np.argsort(results['direct']['per_sample'])[len(test['r_e'])//2]
    ax.plot(r_grid, test['curves'][idx], 'k-', lw=2, label='Morse (true)')
    ax.plot(r_grid, results['direct']['V_pred'][idx], '--',
            color='#1f77b4', lw=1.5, label='Direct')
    ax.plot(r_grid, results['lj_adaptive']['V_pred'][idx], '-',
            color='#2ca02c', lw=1.5, label='LJ adaptive')
    ax.plot(r_grid, test['lj_curves'][idx], ':',
            color='gray', lw=1.5, label='LJ best-fit')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('Median test case')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0): Per-sample scatter
    ax = axes[1, 0]
    ax.scatter(results['direct']['per_sample'],
               results['lj_adaptive']['per_sample'],
               c='#2ca02c', alpha=0.5, s=25)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Direct MAE [eV]')
    ax.set_ylabel('LJ adaptive MAE [eV]')
    ax.set_title('Per-sample: below diagonal = LJ adaptive wins')
    ax.grid(True, alpha=0.3)

    # (1,1): LJ parameter prediction — epsilon
    ax = axes[1, 1]
    ax.scatter(test['lj_params'][:, 0],
               results['lj_adaptive']['params_pred'][:, 0],
               c='#2ca02c', alpha=0.5, s=25)
    lims = [min(test['lj_params'][:, 0].min(),
                results['lj_adaptive']['params_pred'][:, 0].min()),
            max(test['lj_params'][:, 0].max(),
                results['lj_adaptive']['params_pred'][:, 0].max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    eps_mae = results['lj_adaptive']['eps_mae']
    ax.set_xlabel(r'True $\epsilon$ [eV]')
    ax.set_ylabel(r'Predicted $\epsilon$ [eV]')
    ax.set_title(f'$\\epsilon$ prediction (MAE = {eps_mae:.4f} eV)')
    ax.grid(True, alpha=0.3)

    # (1,2): LJ parameter prediction — sigma
    ax = axes[1, 2]
    ax.scatter(test['lj_params'][:, 1],
               results['lj_adaptive']['params_pred'][:, 1],
               c='#2ca02c', alpha=0.5, s=25)
    lims = [min(test['lj_params'][:, 1].min(),
                results['lj_adaptive']['params_pred'][:, 1].min()),
            max(test['lj_params'][:, 1].max(),
                results['lj_adaptive']['params_pred'][:, 1].max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    sig_mae = results['lj_adaptive']['sig_mae']
    ax.set_xlabel(r'True $\sigma$ [Å]')
    ax.set_ylabel(r'Predicted $\sigma$ [Å]')
    ax.set_title(f'$\\sigma$ prediction (MAE = {sig_mae:.4f} Å)')
    ax.grid(True, alpha=0.3)

    ratio = mae_d / mae_lj if mae_lj > 0 else float('inf')
    fig.suptitle(
        f'Step 2: LJ adaptive for Morse — {regime}\n'
        f'Direct MAE = {mae_d:.4f} eV,  '
        f'LJ adaptive MAE = {mae_lj:.4f} eV ({ratio:.2f}x),  '
        f'LJ physics limit = {mae_phys:.4f} eV',
        fontsize=11)
    fig.tight_layout()
    fname = f'fig_lj_for_morse_{regime.lower().replace(" ", "_")}.png'
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close(fig)


def plot_fit_quality(train, r_grid):
    """Show how well LJ fits Morse (the physics approximation error)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Pick a few training curves to show the LJ fit
    show_idx = np.linspace(0, len(train['D_e']) - 1, 6).astype(int)
    cmap = plt.cm.tab10

    for ii, idx in enumerate(show_idx):
        color = cmap(ii)
        D_e, alpha, r_e = train['D_e'][idx], train['alpha'][idx], train['r_e'][idx]
        eps, sig = train['lj_params'][idx]
        label = f'$D_e$={D_e:.1f}, $r_e$={r_e:.2f}'

        axes[0].plot(r_grid, train['curves'][idx], '-', color=color, lw=1.5,
                     label=label)
        axes[0].plot(r_grid, train['lj_curves'][idx], '--', color=color,
                     alpha=0.6, lw=1)

        # Residual
        axes[1].plot(r_grid, train['curves'][idx] - train['lj_curves'][idx],
                     color=color, lw=1)

    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='black', lw=1.5, ls='-', label='Morse'),
        Line2D([0], [0], color='black', lw=1, ls='--', label='LJ best-fit'),
    ]
    axes[0].legend(handles=legend_lines, fontsize=9)
    axes[0].set_ylabel('V(r) [eV]')
    axes[0].set_title('Morse curves and their LJ fits')

    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].set_ylabel('V_Morse - V_LJ [eV]')
    axes[1].set_title('Residual (Morse - LJ)')

    # Histogram of fit errors
    axes[2].hist(train['fit_errors'], bins=20, color='#2ca02c', alpha=0.7)
    axes[2].set_xlabel('MAE of LJ fit [eV]')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'LJ fit quality (mean MAE = {train["fit_errors"].mean():.4f} eV)')

    for ax in axes[:2]:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('How well does LJ approximate Morse?', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_lj_morse_fit_quality.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_lj_morse_fit_quality.png')
    plt.close(fig)


def main():
    r_grid = np.linspace(1.0, 8.0, 50)
    n_train, n_test = 100, 50

    print('Generating datasets (including LJ fits)...')
    train = generate_dataset(n_train, (0.5, 1.5), (0.5, 1.5), r_grid, seed=42)
    test_ext = generate_dataset(n_test, (2.5, 4.0), (0.5, 1.5), r_grid, seed=123)
    test_int = generate_dataset(n_test, (0.7, 1.3), (0.6, 1.4), r_grid, seed=456)

    print('=' * 60)
    print('Step 2: LJ as approximate physics for Morse')
    print('=' * 60)
    print(f'\nTraining: {n_train} Morse curves')
    print(f'  D_e: [{train["D_e"].min():.2f}, {train["D_e"].max():.2f}] eV')
    print(f'  r_e: [{train["r_e"].min():.2f}, {train["r_e"].max():.2f}] Å')
    print(f'  LJ fit MAE (train): {train["fit_errors"].mean():.4f} eV '
          f'(physics approximation error)')

    # Show LJ fit quality
    plot_fit_quality(train, r_grid)

    for regime, test in [('Extrapolation', test_ext),
                         ('Interpolation', test_int)]:
        print(f'\n{"="*50}')
        print(f'  {regime}')
        print(f'{"="*50}')
        print(f'  Test D_e: [{test["D_e"].min():.2f}, {test["D_e"].max():.2f}]')
        print(f'  LJ fit MAE (test): {test["fit_errors"].mean():.4f} eV')

        res = run_ml(train, test, r_grid)

        mae_d = res['direct']['mae']
        mae_lj = res['lj_adaptive']['mae_vs_morse']
        mae_phys = res['physics_error']['mae']

        ratio = mae_d / mae_lj if mae_lj > 0 else float('inf')

        print(f'\n  MAE vs Morse (ground truth):')
        print(f'    Direct V(r):       {mae_d:.6f} eV')
        print(f'    LJ adaptive:       {mae_lj:.6f} eV  ({ratio:.2f}x)')
        print(f'    LJ best-fit limit: {mae_phys:.6f} eV  (irreducible physics error)')
        print(f'\n  LJ parameter prediction:')
        print(f'    epsilon MAE: {res["lj_adaptive"]["eps_mae"]:.4f} eV')
        print(f'    sigma MAE:   {res["lj_adaptive"]["sig_mae"]:.4f} Å')

        plot_results(train, test, res, r_grid, regime)


if __name__ == '__main__':
    main()
