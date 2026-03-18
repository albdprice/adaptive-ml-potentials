"""
Nuclear subtraction + b=0 parabola experiment on SYNTHETIC Morse curves.

Idea (from Anatole):
  The Morse curve IS the total binding curve (like DFT E_bind).
  Subtract the known nuclear repulsion to get the electronic part:
    V_elec(r) = V_Morse(r) - V_nn(r)
  Learn V_elec adaptively (via b=0 parabola), then reconstruct:
    V_pred(r) = V_elec_pred(r) + V_nn(r)

Compare:
  (A) Direct:             descriptor -> V_Morse(r) on grid
  (B) b=0 on V_Morse:    b=0 parabola directly on Morse (no subtraction)
  (C) b=0 on V_elec:     subtract V_nn, b=0 on remainder, add V_nn back
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

HART_TO_EV_ANGSTROM = 14.3996


# --- Physics ---

def morse_shifted(r, D_e, alpha, r_e):
    """Morse with V(r_e) = -D_e, V(inf) = 0.  (Physical convention.)"""
    return D_e * ((1 - np.exp(-alpha * (r - r_e)))**2 - 1)


def morse_unshifted(r, D_e, alpha, r_e):
    """Morse with V(r_e) = 0, V(inf) = D_e."""
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2


def nuclear_repulsion(r, Z_eff):
    """V_nn = Z_eff * 14.4 / r  [eV, r in Angstrom]."""
    return Z_eff * HART_TO_EV_ANGSTROM / r


def compute_b0_numerical(r, V, eps=1e-6):
    """Numerical b=0 decomposition for any curve with a minimum.

    Returns r_e, E_min, a(r), sqrt_a(r) where:
      V_shifted(r) = a(r) * (r - r_e)^2,  V_shifted = V - E_min
    """
    cs = CubicSpline(r, V)
    r_fine = np.linspace(r.min(), r.max(), 5000)
    V_fine = cs(r_fine)
    r_e = r_fine[np.argmin(V_fine)]
    E_min = V_fine.min()

    V_shifted = V - E_min
    dr = r - r_e
    a = np.full_like(r, np.nan, dtype=float)
    mask = np.abs(dr) > eps
    a[mask] = V_shifted[mask] / dr[mask]**2
    a[~mask] = cs(r_e, 2) / 2.0

    a_clipped = np.maximum(a, 0.0)
    return r_e, E_min, a, np.sqrt(a_clipped)


def compute_a_morse_analytical(r, D_e, alpha, r_e):
    """Analytical a(r) for unshifted Morse: V = a(r)*(r-r_e)^2."""
    u = alpha * (r - r_e)
    f = np.where(np.abs(u) < 1e-8,
                 1.0 - u / 2.0 + u**2 / 6.0,
                 (1 - np.exp(-u)) / u)
    return D_e * alpha**2 * f**2


# --- Dataset ---

def generate_dataset(n_samples, d1_range, d2_range, r_grid, seed=42):
    """Generate Morse curves (= total binding) with nuclear charges.

    Mapping:
      D_e = 1 + 2*d1, alpha = 0.8 + 0.6*d2, r_e = 1.5 + 0.5*d1
      Z_eff = 1 + 2*d1  (nuclear charge product, correlated with D_e)

    The Morse curve IS the total. V_elec = V_Morse - V_nn.
    """
    rng = np.random.RandomState(seed)
    d1 = rng.uniform(*d1_range, size=n_samples)
    d2 = rng.uniform(*d2_range, size=n_samples)

    D_e = 1.0 + 2.0 * d1
    alpha = 0.8 + 0.6 * d2
    r_e = 1.5 + 0.5 * d1
    Z_eff = 1.0 + 2.0 * d1

    n_grid = len(r_grid)

    # The "total" binding curve = Morse (unshifted: V(r_e)=0, V(inf)=D_e)
    V_morse = np.zeros((n_samples, n_grid))
    V_nn = np.zeros((n_samples, n_grid))
    V_elec = np.zeros((n_samples, n_grid))

    # b=0 on Morse directly (no subtraction baseline)
    a_morse = np.zeros((n_samples, n_grid))

    # b=0 on V_elec (after subtraction)
    re_elec = np.zeros(n_samples)
    emin_elec = np.zeros(n_samples)
    a_elec = np.zeros((n_samples, n_grid))
    sqrt_a_elec = np.zeros((n_samples, n_grid))

    elec_min_in_range = 0

    for i in range(n_samples):
        V_morse[i] = morse_unshifted(r_grid, D_e[i], alpha[i], r_e[i])
        V_nn[i] = nuclear_repulsion(r_grid, Z_eff[i])
        V_elec[i] = V_morse[i] - V_nn[i]

        # b=0 on Morse directly (analytical)
        a_morse[i] = compute_a_morse_analytical(r_grid, D_e[i], alpha[i], r_e[i])

        # b=0 on electronic remainder (numerical)
        re_e, emin_e, a_e, sa_e = compute_b0_numerical(r_grid, V_elec[i])
        re_elec[i] = re_e
        emin_elec[i] = emin_e
        a_elec[i] = a_e
        sqrt_a_elec[i] = sa_e

        # Check if electronic minimum is inside grid (not at edge)
        if r_grid[1] < re_e < r_grid[-2]:
            elec_min_in_range += 1

    print(f'  Electronic minimum inside grid: {elec_min_in_range}/{n_samples}')

    return {
        'descriptors': np.column_stack([d1, d2]),
        'D_e': D_e, 'alpha': alpha, 'r_e': r_e, 'Z_eff': Z_eff,
        'V_morse': V_morse, 'V_nn': V_nn, 'V_elec': V_elec,
        'a_morse': a_morse, 'sqrt_a_morse': np.sqrt(a_morse),
        're_elec': re_elec, 'emin_elec': emin_elec,
        'a_elec': a_elec, 'sqrt_a_elec': sqrt_a_elec,
    }


# --- ML ---

def run_ml(train, test, r_grid):
    """Compare methods."""
    X_train = train['descriptors']
    X_test = test['descriptors']
    n_test = len(X_test)
    n_grid = len(r_grid)

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct: predict V_Morse(r) ---
    scaler_V = StandardScaler()
    Y_tr = scaler_V.fit_transform(train['V_morse'])
    model_V = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    V_pred = scaler_V.inverse_transform(model_V.predict(X_te))

    results['direct'] = {
        'V_pred': V_pred,
        'mae': np.mean(np.abs(test['V_morse'] - V_pred)),
        'per_sample': np.mean(np.abs(test['V_morse'] - V_pred), axis=1),
    }

    # --- 2. b=0 on Morse directly (no subtraction) ---
    # Predict Morse r_e + a_morse(r), reconstruct V = a*(r-r_e)^2
    scaler_re_m = StandardScaler()
    re_m_tr = scaler_re_m.fit_transform(train['r_e'].reshape(-1, 1))
    model_re_m = Ridge(alpha=1.0).fit(X_tr, re_m_tr)
    re_m_pred = scaler_re_m.inverse_transform(model_re_m.predict(X_te)).ravel()

    scaler_am = StandardScaler()
    am_tr = scaler_am.fit_transform(train['a_morse'])
    model_am = Ridge(alpha=1.0).fit(X_tr, am_tr)
    am_pred = scaler_am.inverse_transform(model_am.predict(X_te))

    V_b0_morse = np.zeros((n_test, n_grid))
    for i in range(n_test):
        V_b0_morse[i] = am_pred[i] * (r_grid - re_m_pred[i])**2

    results['b0_morse'] = {
        'V_pred': V_b0_morse,
        'mae': np.mean(np.abs(test['V_morse'] - V_b0_morse)),
        'per_sample': np.mean(np.abs(test['V_morse'] - V_b0_morse), axis=1),
    }

    # sqrt(a) version
    scaler_sam = StandardScaler()
    sam_tr = scaler_sam.fit_transform(train['sqrt_a_morse'])
    model_sam = Ridge(alpha=1.0).fit(X_tr, sam_tr)
    sam_pred = scaler_sam.inverse_transform(model_sam.predict(X_te))

    V_b0s_morse = np.zeros((n_test, n_grid))
    for i in range(n_test):
        V_b0s_morse[i] = sam_pred[i]**2 * (r_grid - re_m_pred[i])**2

    results['b0_sqrt_morse'] = {
        'V_pred': V_b0s_morse,
        'mae': np.mean(np.abs(test['V_morse'] - V_b0s_morse)),
        'per_sample': np.mean(np.abs(test['V_morse'] - V_b0s_morse), axis=1),
    }

    # --- 3. b=0 on V_elec AFTER nuclear subtraction ---
    # Predict r_e_elec + E_min_elec + a_elec(r), then add V_nn back
    scaler_re_e = StandardScaler()
    re_e_tr = scaler_re_e.fit_transform(train['re_elec'].reshape(-1, 1))
    model_re_e = Ridge(alpha=1.0).fit(X_tr, re_e_tr)
    re_e_pred = scaler_re_e.inverse_transform(model_re_e.predict(X_te)).ravel()

    scaler_em = StandardScaler()
    em_tr = scaler_em.fit_transform(train['emin_elec'].reshape(-1, 1))
    model_em = Ridge(alpha=1.0).fit(X_tr, em_tr)
    em_pred = scaler_em.inverse_transform(model_em.predict(X_te)).ravel()

    scaler_ae = StandardScaler()
    ae_tr = scaler_ae.fit_transform(train['a_elec'])
    model_ae = Ridge(alpha=1.0).fit(X_tr, ae_tr)
    ae_pred = scaler_ae.inverse_transform(model_ae.predict(X_te))

    V_b0_elec = np.zeros((n_test, n_grid))
    for i in range(n_test):
        V_elec_pred = ae_pred[i] * (r_grid - re_e_pred[i])**2 + em_pred[i]
        V_b0_elec[i] = V_elec_pred + test['V_nn'][i]  # add known V_nn back

    results['b0_elec'] = {
        'V_pred': V_b0_elec,
        'mae': np.mean(np.abs(test['V_morse'] - V_b0_elec)),
        'per_sample': np.mean(np.abs(test['V_morse'] - V_b0_elec), axis=1),
        're_pred': re_e_pred,
        'emin_pred': em_pred,
    }

    # sqrt(a) version
    scaler_sae = StandardScaler()
    sae_tr = scaler_sae.fit_transform(train['sqrt_a_elec'])
    model_sae = Ridge(alpha=1.0).fit(X_tr, sae_tr)
    sae_pred = scaler_sae.inverse_transform(model_sae.predict(X_te))

    V_b0s_elec = np.zeros((n_test, n_grid))
    for i in range(n_test):
        V_elec_pred = sae_pred[i]**2 * (r_grid - re_e_pred[i])**2 + em_pred[i]
        V_b0s_elec[i] = V_elec_pred + test['V_nn'][i]

    results['b0_sqrt_elec'] = {
        'V_pred': V_b0s_elec,
        'mae': np.mean(np.abs(test['V_morse'] - V_b0s_elec)),
        'per_sample': np.mean(np.abs(test['V_morse'] - V_b0s_elec), axis=1),
    }

    return results


# --- Plotting ---

def plot_example_curves(test, r_grid):
    """Show what the decomposition looks like."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(min(5, len(test['Z_eff']))):
        color = plt.cm.tab10(i)
        label = (f'$D_e$={test["D_e"][i]:.1f}, '
                 f'$Z_{{eff}}$={test["Z_eff"][i]:.1f}')

        axes[0].plot(r_grid, test['V_morse'][i], color=color, lw=1.5, label=label)
        axes[1].plot(r_grid, test['V_nn'][i], color=color, lw=1.5, ls=':', label=label)
        axes[2].plot(r_grid, test['V_elec'][i], color=color, lw=1.5, label=label)

    axes[0].set_title(r'(A) $V_{Morse}(r)$ — the total binding curve')
    axes[0].set_ylabel('Energy [eV]')
    axes[1].set_title(r'(B) $V_{nn} = Z_{eff} \cdot 14.4 / r$ — known, subtract this')
    axes[2].set_title(r'(C) $V_{elec} = V_{Morse} - V_{nn}$ — learn this')

    for ax in axes:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.axhline(0, color='gray', lw=0.5)

    fig.suptitle(r'Nuclear subtraction: $V_{Morse} = V_{elec} + V_{nn}$, learn $V_{elec}$ adaptively',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_nuc_sub_example_curves.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_nuc_sub_example_curves.png')
    plt.close(fig)


def plot_a_comparison(test, r_grid):
    """Show a(r) for Morse (no sub) vs electronic (after sub)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(min(5, len(test['Z_eff']))):
        color = plt.cm.tab10(i)
        label = f'$D_e$={test["D_e"][i]:.1f}'

        # a(r) for Morse directly
        axes[0].plot(r_grid, test['a_morse'][i], color=color, lw=1.5, label=label)

        # a(r) for electronic (after V_nn subtraction)
        mask = np.isfinite(test['a_elec'][i]) & (np.abs(test['a_elec'][i]) < 500)
        axes[1].plot(r_grid[mask], test['a_elec'][i][mask], color=color,
                     lw=1.5, label=label)

        # sqrt(a) comparison
        axes[2].plot(r_grid, test['sqrt_a_morse'][i], color=color,
                     lw=1.5, label=f'Morse')
        axes[2].plot(r_grid[mask], test['sqrt_a_elec'][i][mask], color=color,
                     lw=1.5, ls='--', alpha=0.6)

    axes[0].set_title(r'(A) $a(r)$ of Morse directly')
    axes[0].set_ylabel(r'$a(r)$ [eV/Å²]')
    axes[1].set_title(r'(B) $a(r)$ of $V_{elec}$ (after subtraction)')
    axes[1].set_ylabel(r'$a(r)$ [eV/Å²]')
    axes[2].set_title(r'(C) $\sqrt{a}$: solid=Morse, dashed=electronic')
    axes[2].set_ylabel(r'$\sqrt{a}$')

    for ax in axes:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(r'Curvature $a(r)$: Morse (no subtraction) vs electronic (after subtraction)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_nuc_sub_a_comparison.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_nuc_sub_a_comparison.png')
    plt.close(fig)


def plot_ml_results(test, results, r_grid, regime):
    """Results figure."""
    methods = ['direct', 'b0_morse', 'b0_sqrt_morse', 'b0_elec', 'b0_sqrt_elec']
    short_labels = ['Direct', 'b0 Morse\n(no sub)', r'b0$\sqrt{a}$ Morse'
                    '\n(no sub)', 'b0 elec\n(with sub)',
                    r'b0$\sqrt{a}$ elec' '\n(with sub)']
    colors = ['#1f77b4', '#aec7e8', '#7f7f7f', '#2ca02c', '#d62728']

    maes = [results[m]['mae'] for m in methods]
    mae_d = maes[0]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (0,0): Bar chart
    ax = axes[0, 0]
    bars = ax.bar(range(len(methods)), maes, color=colors, alpha=0.8,
                  tick_label=[s.replace('\n', ' ') for s in short_labels])
    for bar, mae in zip(bars, maes):
        ratio_str = f'{mae:.2f}\n({mae_d/mae:.2f}x)' if mae > 0 else '0'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                ratio_str, ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('MAE [eV]')
    ax.set_title(f'Energy MAE — {regime}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=20, labelsize=6)

    # (0,1): Scatter — direct vs with-subtraction
    ax = axes[0, 1]
    ax.scatter(results['direct']['per_sample'],
               results['b0_elec']['per_sample'],
               c='#2ca02c', alpha=0.5, s=25, label='b0 elec (sub)')
    ax.scatter(results['direct']['per_sample'],
               results['b0_sqrt_elec']['per_sample'],
               c='#d62728', alpha=0.5, s=25, marker='^',
               label=r'b0$\sqrt{a}$ elec (sub)')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Direct MAE [eV]')
    ax.set_ylabel('With subtraction MAE [eV]')
    ax.set_title('With subtraction vs Direct')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2): Scatter — no sub vs with sub
    ax = axes[0, 2]
    ax.scatter(results['b0_morse']['per_sample'],
               results['b0_elec']['per_sample'],
               c='#2ca02c', alpha=0.5, s=25, label='a(r)')
    ax.scatter(results['b0_sqrt_morse']['per_sample'],
               results['b0_sqrt_elec']['per_sample'],
               c='#d62728', alpha=0.5, s=25, marker='^', label=r'$\sqrt{a}$')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('b=0 Morse (no sub) MAE [eV]')
    ax.set_ylabel('b=0 elec (with sub) MAE [eV]')
    ax.set_title('Effect of nuclear subtraction on b=0')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0-2): Example predictions
    sort_idx = np.argsort(results['direct']['per_sample'])
    examples = [sort_idx[-1], sort_idx[len(sort_idx)//2], sort_idx[0]]
    ex_labels = ['Worst direct', 'Median direct', 'Best direct']

    for ii, (idx, elabel) in enumerate(zip(examples, ex_labels)):
        ax = axes[1, ii]
        ax.plot(r_grid, test['V_morse'][idx], 'k-', lw=2.5, label='True')

        for method, slabel, color, ls in zip(
                methods,
                ['Direct', 'b0 Morse', r'b0$\sqrt{a}$ Morse',
                 'b0 elec', r'b0$\sqrt{a}$ elec'],
                colors,
                ['--', ':', '-.', '-', '-']):
            lw = 2.0 if 'elec' in method else 1.2
            ax.plot(r_grid, results[method]['V_pred'][idx],
                    ls=ls, color=color, lw=lw, label=slabel, alpha=0.8)

        ax.set_xlabel('r [Å]')
        ax.set_ylabel('V(r) [eV]')
        d1 = test['descriptors'][idx, 0]
        ax.set_title(f'{elabel} (d1={d1:.2f})')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    ratios = {m: mae_d / results[m]['mae'] if results[m]['mae'] > 0 else 0
              for m in methods}
    fig.suptitle(
        f'{regime} — V_total = V_Morse, V_elec = V_Morse - V_nn\n'
        f'Direct: {maes[0]:.2f} | '
        f'b0 Morse: {maes[1]:.2f} ({ratios["b0_morse"]:.2f}x) | '
        f'b0√ Morse: {maes[2]:.2f} ({ratios["b0_sqrt_morse"]:.2f}x) | '
        f'b0 elec: {maes[3]:.2f} ({ratios["b0_elec"]:.2f}x) | '
        f'b0√ elec: {maes[4]:.2f} ({ratios["b0_sqrt_elec"]:.2f}x)',
        fontsize=10)
    fig.tight_layout()
    fname = f'fig_nuc_sub_ml_{regime.lower().replace(" ", "_")}.png'
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close(fig)


def main():
    r_grid = np.linspace(1.0, 8.0, 50)
    n_train, n_test = 100, 50

    print('=' * 70)
    print('Nuclear subtraction experiment')
    print('V_Morse = total binding curve (like DFT E_bind)')
    print('V_elec = V_Morse - V_nn  (subtract known nuclear repulsion)')
    print('Learn V_elec adaptively, reconstruct V_Morse = V_elec + V_nn')
    print('=' * 70)

    print(f'\nGenerating training data (d1 in [0.5,1.5]):')
    train = generate_dataset(n_train, (0.5, 1.5), (0.5, 1.5), r_grid, seed=42)
    print(f'  D_e: [{train["D_e"].min():.2f}, {train["D_e"].max():.2f}]')
    print(f'  r_e: [{train["r_e"].min():.2f}, {train["r_e"].max():.2f}]')
    print(f'  Z_eff: [{train["Z_eff"].min():.2f}, {train["Z_eff"].max():.2f}]')
    print(f'  V_nn(r_e): [{(train["Z_eff"]*14.4/train["r_e"]).min():.1f}, '
          f'{(train["Z_eff"]*14.4/train["r_e"]).max():.1f}] eV')

    print(f'\nGenerating extrapolation test (d1 in [2.5,4.0]):')
    test_ext = generate_dataset(n_test, (2.5, 4.0), (0.5, 1.5), r_grid, seed=123)
    print(f'  D_e: [{test_ext["D_e"].min():.2f}, {test_ext["D_e"].max():.2f}]')
    print(f'  Z_eff: [{test_ext["Z_eff"].min():.2f}, {test_ext["Z_eff"].max():.2f}]')
    print(f'  V_nn(r_e): [{(test_ext["Z_eff"]*14.4/test_ext["r_e"]).min():.1f}, '
          f'{(test_ext["Z_eff"]*14.4/test_ext["r_e"]).max():.1f}] eV')

    print(f'\nGenerating interpolation test (d1 in [0.7,1.3]):')
    test_int = generate_dataset(n_test, (0.7, 1.3), (0.6, 1.4), r_grid, seed=456)
    print(f'  D_e: [{test_int["D_e"].min():.2f}, {test_int["D_e"].max():.2f}]')
    print(f'  Z_eff: [{test_int["Z_eff"].min():.2f}, {test_int["Z_eff"].max():.2f}]')

    # Visualizations
    plot_example_curves(test_ext, r_grid)
    plot_a_comparison(test_ext, r_grid)

    for regime, test in [('Extrapolation', test_ext), ('Interpolation', test_int)]:
        print(f'\n{"="*60}')
        print(f'  {regime}')
        print(f'{"="*60}')

        res = run_ml(train, test, r_grid)

        mae_d = res['direct']['mae']
        print(f'\n  Energy MAE [eV]:')
        print(f'    {"Method":<35s}  {"MAE":>10s}  {"Ratio":>8s}')
        print(f'    {"-"*55}')
        for method, label in [
                ('direct', 'Direct V_Morse(r)'),
                ('b0_morse', 'b=0 a(r) on Morse (no sub)'),
                ('b0_sqrt_morse', 'b=0 sqrt(a) on Morse (no sub)'),
                ('b0_elec', 'b=0 a(r) on V_elec (WITH sub)'),
                ('b0_sqrt_elec', 'b=0 sqrt(a) on V_elec (WITH sub)')]:
            mae = res[method]['mae']
            ratio = mae_d / mae if mae > 0 else float('inf')
            best = ' <-- best' if mae == min(res[m]['mae'] for m in res) else ''
            print(f'    {label:<35s}  {mae:10.4f}  {ratio:8.2f}x{best}')

        # Diagnostic: r_e prediction quality
        if 'b0_elec' in res and 're_pred' in res['b0_elec']:
            re_mae = np.mean(np.abs(test['re_elec'] - res['b0_elec']['re_pred']))
            em_mae = np.mean(np.abs(test['emin_elec'] - res['b0_elec']['emin_pred']))
            print(f'\n    r_e_elec prediction MAE: {re_mae:.4f} Å')
            print(f'    E_min_elec prediction MAE: {em_mae:.4f} eV')

        plot_ml_results(test, res, r_grid, regime)

    # Summary
    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    print(f'\n  The Morse curve IS the total. V_elec = V_Morse - Z_eff*14.4/r.')
    print(f'  "With subtraction" = learn V_elec via b=0, add V_nn back.')
    print(f'  "No subtraction" = learn V_Morse via b=0 directly.')


if __name__ == '__main__':
    main()
