"""
ML experiment: b=0 adaptive parabola on real DFT diatomics.

Two approaches:
  (A) b=0 on E_bind DIRECTLY: a(r)*(r - r_e)^2 + E_min = E_bind
  (B) b=0 after nuclear subtraction — FAILS because E_elec has no minimum in grid range

This script implements approach (A) and compares:
  1. Direct: descriptor -> E_bind(r) on grid
  2. Rose adaptive: descriptor -> (E_c, r_e, l) -> rose_V(r)
  3. b=0 adaptive a(r): descriptor -> r_e + E_min + a(r), reconstruct
  4. b=0 adaptive sqrt(a): descriptor -> r_e + E_min + sqrt(a), reconstruct

Cross-validation: LOO and LGO (train Z<=10, test Z>10).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
import sys
import os

sys.path.insert(0, '/Users/albd/research/adaptive_paper_anatole/publication/extended/dft_diatomics')
from diatomic_adaptive_vs_direct import (
    load_dft_data, assess_data_quality, filter_data,
    build_scaled_common_grid, rose_V
)

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def compute_b0_targets(r_scaled, E_bind, eps=1e-6):
    """Compute b=0 decomposition targets on E_bind directly.

    E_bind(r) = a(r) * (r_sc - r_e_sc)^2 + E_min

    Works in scaled coordinates (r_sc = r / rcov_sum).
    """
    # Find equilibrium via cubic spline
    cs = CubicSpline(r_scaled, E_bind)
    r_fine = np.linspace(r_scaled.min(), r_scaled.max(), 2000)
    E_fine = cs(r_fine)
    r_e_sc = r_fine[np.argmin(E_fine)]
    E_min = E_fine.min()

    # Shift so E_shifted(r_e) = 0, E_shifted >= 0
    E_shifted = E_bind - E_min

    # b=0 decomposition: a(r) = E_shifted / (r_sc - r_e_sc)^2
    dr = r_scaled - r_e_sc
    a_sc = np.full_like(r_scaled, np.nan, dtype=float)
    mask = np.abs(dr) > eps
    a_sc[mask] = E_shifted[mask] / dr[mask]**2

    # At r_e, use V''(r_e)/2 via spline
    a_sc[~mask] = cs(r_e_sc, 2) / 2.0

    # sqrt(a) — a should be >= 0 since E_shifted >= 0
    # Small negative values possible from spline interpolation noise
    a_sc_clipped = np.maximum(a_sc, 0.0)
    sqrt_a_sc = np.sqrt(a_sc_clipped)

    return {
        'r_e_sc': r_e_sc,
        'E_min': E_min,
        'a_sc': a_sc,
        'sqrt_a_sc': sqrt_a_sc,
        'E_shifted': E_shifted,
    }


def run_cv(data, r_scaled_common, curves_common, rcov_sums, train_idx, test_idx,
           b0_targets, rose_params):
    """Run one train/test split. Returns MAEs and predictions for each method."""
    descriptors = np.array(data['descriptors'])
    n_grid = len(r_scaled_common)

    X_train = descriptors[train_idx]
    X_test = descriptors[test_idx]
    n_test = len(test_idx)

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct: predict E_bind on common grid ---
    scaler_V = StandardScaler()
    Y_tr = scaler_V.fit_transform(curves_common[train_idx])
    model_V = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    V_pred = scaler_V.inverse_transform(model_V.predict(X_te))

    mae_direct = []
    for k, ti in enumerate(test_idx):
        mae_direct.append(np.mean(np.abs(curves_common[ti] - V_pred[k])))
    results['direct'] = {
        'V_pred': V_pred,
        'mae': np.array(mae_direct),
    }

    # --- 2. Rose adaptive: predict (E_c, r_e, l) ---
    rp_train = rose_params[train_idx]
    scaler_rp = StandardScaler()
    rp_tr = scaler_rp.fit_transform(rp_train)
    model_rp = Ridge(alpha=1.0).fit(X_tr, rp_tr)
    rp_pred = scaler_rp.inverse_transform(model_rp.predict(X_te))

    mae_rose = []
    V_rose_all = np.zeros((n_test, n_grid))
    for k, ti in enumerate(test_idx):
        r_real = r_scaled_common * rcov_sums[ti]
        V_rose = rose_V(r_real, rp_pred[k, 0], rp_pred[k, 1], rp_pred[k, 2])
        V_rose_all[k] = V_rose
        mae_rose.append(np.mean(np.abs(curves_common[ti] - V_rose)))
    results['rose'] = {
        'V_pred': V_rose_all,
        'mae': np.array(mae_rose),
        'params_pred': rp_pred,
    }

    # --- 3. b=0 adaptive: predict r_e + E_min + a(r) ---
    re_train = np.array([b0_targets[i]['r_e_sc'] for i in train_idx])
    emin_train = np.array([b0_targets[i]['E_min'] for i in train_idx])
    a_train = np.array([b0_targets[i]['a_sc'] for i in train_idx])
    sqrt_a_train = np.array([b0_targets[i]['sqrt_a_sc'] for i in train_idx])

    # Predict r_e (scaled)
    scaler_re = StandardScaler()
    re_tr = scaler_re.fit_transform(re_train.reshape(-1, 1))
    model_re = Ridge(alpha=1.0).fit(X_tr, re_tr)
    re_pred = scaler_re.inverse_transform(model_re.predict(X_te)).ravel()

    # Predict E_min
    scaler_em = StandardScaler()
    em_tr = scaler_em.fit_transform(emin_train.reshape(-1, 1))
    model_em = Ridge(alpha=1.0).fit(X_tr, em_tr)
    em_pred = scaler_em.inverse_transform(model_em.predict(X_te)).ravel()

    # Predict a(r)
    scaler_a = StandardScaler()
    a_tr = scaler_a.fit_transform(a_train)
    model_a = Ridge(alpha=1.0).fit(X_tr, a_tr)
    a_pred = scaler_a.inverse_transform(model_a.predict(X_te))

    # Reconstruct: E_bind = a*(r_sc - r_e)^2 + E_min
    mae_b0 = []
    V_b0_all = np.zeros((n_test, n_grid))
    for k, ti in enumerate(test_idx):
        dr = r_scaled_common - re_pred[k]
        V_b0 = a_pred[k] * dr**2 + em_pred[k]
        V_b0_all[k] = V_b0
        mae_b0.append(np.mean(np.abs(curves_common[ti] - V_b0)))
    results['b0_a'] = {
        'V_pred': V_b0_all,
        'mae': np.array(mae_b0),
        're_pred': re_pred,
        'emin_pred': em_pred,
        'a_pred': a_pred,
    }

    # --- 4. b=0 sqrt(a): predict sqrt(a), square to reconstruct ---
    scaler_sa = StandardScaler()
    sa_tr = scaler_sa.fit_transform(sqrt_a_train)
    model_sa = Ridge(alpha=1.0).fit(X_tr, sa_tr)
    sa_pred = scaler_sa.inverse_transform(model_sa.predict(X_te))

    mae_b0s = []
    V_b0s_all = np.zeros((n_test, n_grid))
    for k, ti in enumerate(test_idx):
        dr = r_scaled_common - re_pred[k]
        V_b0s = sa_pred[k]**2 * dr**2 + em_pred[k]
        V_b0s_all[k] = V_b0s
        mae_b0s.append(np.mean(np.abs(curves_common[ti] - V_b0s)))
    results['b0_sqrt'] = {
        'V_pred': V_b0s_all,
        'mae': np.array(mae_b0s),
        'sqrt_a_pred': sa_pred,
    }

    return results


def run_loo(data, r_scaled_common, curves_common, rcov_sums, b0_targets, rose_params):
    """Leave-one-out cross-validation."""
    n = len(data['names'])
    names = data['names']

    all_results = {method: {'mae': [], 'V_pred': [], 'V_true': []}
                   for method in ['direct', 'rose', 'b0_a', 'b0_sqrt']}

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        test_idx = [i]
        res = run_cv(data, r_scaled_common, curves_common, rcov_sums,
                     train_idx, test_idx, b0_targets, rose_params)

        for method in all_results:
            all_results[method]['mae'].append(res[method]['mae'][0])
            all_results[method]['V_pred'].append(res[method]['V_pred'][0])
            all_results[method]['V_true'].append(curves_common[i])

    for method in all_results:
        all_results[method]['mae'] = np.array(all_results[method]['mae'])
        all_results[method]['V_pred'] = np.array(all_results[method]['V_pred'])
        all_results[method]['V_true'] = np.array(all_results[method]['V_true'])

    return names, all_results


def run_lgo(data, r_scaled_common, curves_common, rcov_sums, b0_targets, rose_params):
    """Leave-group-out: train on Z<=10, test on Z>10."""
    descriptors = np.array(data['descriptors'])
    n = len(data['names'])
    names = data['names']

    train_idx = [i for i in range(n)
                 if descriptors[i, 0] <= 10 and descriptors[i, 1] <= 10]
    test_idx = [i for i in range(n) if i not in train_idx]

    train_names = [names[i] for i in train_idx]
    test_names = [names[i] for i in test_idx]

    res = run_cv(data, r_scaled_common, curves_common, rcov_sums,
                 train_idx, test_idx, b0_targets, rose_params)

    # Store V_true for plotting
    for method in res:
        res[method]['V_true'] = np.array([curves_common[ti] for ti in test_idx])

    return train_names, test_names, test_idx, res


def plot_comparison(names, results, regime, r_scaled_common):
    """Plot comparison across methods."""
    methods = ['direct', 'rose', 'b0_a', 'b0_sqrt']
    labels = ['Direct', 'Rose adaptive', r'b=0 $a(r)$', r'b=0 $\sqrt{a}$']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    mean_maes = [np.mean(results[m]['mae']) for m in methods]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0): Bar chart of mean MAEs
    ax = axes[0, 0]
    bars = ax.bar(labels, mean_maes, color=colors, alpha=0.8)
    for bar, mae in zip(bars, mean_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mae:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean MAE [eV]')
    ax.set_title(f'Energy MAE — {regime}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15, labelsize=8)

    # (0,1): Per-molecule MAE comparison
    ax = axes[0, 1]
    x = np.arange(len(names))
    w = 0.2
    for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
        ax.bar(x + j * w, results[method]['mae'], w, label=label,
               color=color, alpha=0.7)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE [eV]')
    ax.set_title(f'Per-molecule MAE — {regime}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # (0,2): Scatter direct vs adaptive
    ax = axes[0, 2]
    ax.scatter(results['direct']['mae'], results['b0_a']['mae'],
               c='#2ca02c', alpha=0.6, s=30, label=r'b=0 $a(r)$')
    ax.scatter(results['direct']['mae'], results['b0_sqrt']['mae'],
               c='#d62728', alpha=0.6, s=30, marker='^', label=r'b=0 $\sqrt{a}$')
    ax.scatter(results['direct']['mae'], results['rose']['mae'],
               c='#ff7f0e', alpha=0.6, s=30, marker='s', label='Rose')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Direct MAE [eV]')
    ax.set_ylabel('Adaptive MAE [eV]')
    ax.set_title('Below diagonal = adaptive wins')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0-2): Example predictions for 3 molecules
    mae_direct = results['direct']['mae']
    sort_idx = np.argsort(mae_direct)
    examples = [sort_idx[-1], sort_idx[len(sort_idx)//2], sort_idx[0]]
    example_labels = ['Worst direct', 'Median direct', 'Best direct']

    for ii, (idx, elabel) in enumerate(zip(examples, example_labels)):
        ax = axes[1, ii]
        ax.plot(r_scaled_common, results['direct']['V_true'][idx],
                'k-', lw=2, label='True')
        for method, label, color, ls in zip(
                methods, labels, colors,
                ['--', '-.', '-', ':']):
            ax.plot(r_scaled_common, results[method]['V_pred'][idx],
                    ls=ls, color=color, lw=1.5, label=label)
        ax.set_xlabel(r'$r / r_{cov,sum}$')
        ax.set_ylabel('E [eV]')
        ax.set_title(f'{names[idx]} ({elabel})')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    ratios = {m: np.mean(results['direct']['mae']) / np.mean(results[m]['mae'])
              if np.mean(results[m]['mae']) > 0 else float('inf')
              for m in methods}
    fig.suptitle(
        f'{regime}\n'
        f'Direct: {mean_maes[0]:.4f} eV | '
        f'Rose: {mean_maes[1]:.4f} ({ratios["rose"]:.2f}x) | '
        f'b=0 a: {mean_maes[2]:.4f} ({ratios["b0_a"]:.2f}x) | '
        f'b=0 sqrt: {mean_maes[3]:.4f} ({ratios["b0_sqrt"]:.2f}x)',
        fontsize=10)
    fig.tight_layout()
    fname = f'fig_dft_b0_ml_{regime.lower().replace(" ", "_").replace("/", "_").replace(",", "").replace("(", "").replace(")", "")}.png'
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close(fig)


def main():
    # ================================================================
    # Load and filter DFT data
    # ================================================================
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'dft_diatomics', 'data', 'diatomic_curves.npz')
    data = load_dft_data(data_path)
    keep_idx = assess_data_quality(data, max_residual=1.0, min_points=8)
    data = filter_data(data, keep_idx)

    names = data['names']
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])
    n_mol = len(names)
    print(f'\nUsing {n_mol} molecules')

    # ================================================================
    # Build common scaled grid
    # ================================================================
    r_scaled_common, curves_common, rcov_sums = build_scaled_common_grid(data, n_grid=50)

    # ================================================================
    # Compute b=0 targets on E_bind DIRECTLY (no nuclear subtraction)
    # ================================================================
    print('\n--- Computing b=0 targets on E_bind directly ---')
    b0_targets = []
    print(f'{"Mol":>6s}  {"r_e_sc":>7s}  {"E_min":>8s}  {"a(r_e)":>8s}  '
          f'{"min(a)":>8s}  {"max(a)":>8s}  {"neg_a?":>6s}')
    for i in range(n_mol):
        targets = compute_b0_targets(r_scaled_common, curves_common[i])
        b0_targets.append(targets)

        # a at r_e
        dr = r_scaled_common - targets['r_e_sc']
        j_re = np.argmin(np.abs(dr))
        a_re = targets['a_sc'][j_re]
        a_min = np.nanmin(targets['a_sc'])
        a_max = np.nanmax(targets['a_sc'])
        has_neg = 'YES' if np.any(targets['a_sc'] < -0.01) else 'no'

        print(f'{names[i]:>6s}  {targets["r_e_sc"]:7.3f}  {targets["E_min"]:8.4f}  '
              f'{a_re:8.2f}  {a_min:8.2f}  {a_max:8.2f}  {has_neg:>6s}')

    # ================================================================
    # LOO cross-validation
    # ================================================================
    print('\n' + '=' * 60)
    print('  Leave-One-Out Cross-Validation')
    print('=' * 60)

    loo_names, loo_results = run_loo(
        data, r_scaled_common, curves_common, rcov_sums, b0_targets, rose_params)

    print(f'\n{"Mol":>6s}  {"Direct":>8s}  {"Rose":>8s}  {"b0_a":>8s}  {"b0_sqrt":>8s}  '
          f'{"Rose/D":>7s}  {"b0a/D":>7s}  {"b0s/D":>7s}')
    for i, name in enumerate(loo_names):
        md = loo_results['direct']['mae'][i]
        mr = loo_results['rose']['mae'][i]
        ma = loo_results['b0_a']['mae'][i]
        ms = loo_results['b0_sqrt']['mae'][i]
        print(f'{name:>6s}  {md:8.4f}  {mr:8.4f}  {ma:8.4f}  {ms:8.4f}  '
              f'{md/mr if mr > 0 else 0:7.2f}x  '
              f'{md/ma if ma > 0 else 0:7.2f}x  '
              f'{md/ms if ms > 0 else 0:7.2f}x')

    print('\n  Mean MAEs:')
    for method, label in [('direct', 'Direct'), ('rose', 'Rose'),
                          ('b0_a', 'b=0 a(r)'), ('b0_sqrt', 'b=0 sqrt(a)')]:
        m = np.mean(loo_results[method]['mae'])
        print(f'    {label:20s} {m:.4f} eV')

    md = np.mean(loo_results['direct']['mae'])
    print('\n  Ratios (Direct / Adaptive):')
    for method, label in [('rose', 'Rose'), ('b0_a', 'b=0 a(r)'),
                          ('b0_sqrt', 'b=0 sqrt(a)')]:
        m = np.mean(loo_results[method]['mae'])
        print(f'    {label:20s} {md/m:.2f}x')

    plot_comparison(loo_names, loo_results, 'LOO', r_scaled_common)

    # ================================================================
    # LGO cross-validation
    # ================================================================
    print('\n' + '=' * 60)
    print('  Leave-Group-Out (train Z<=10, test Z>10)')
    print('=' * 60)

    lgo_train_names, lgo_test_names, lgo_test_idx, lgo_results = run_lgo(
        data, r_scaled_common, curves_common, rcov_sums, b0_targets, rose_params)

    print(f'\nTrain ({len(lgo_train_names)}): {", ".join(lgo_train_names)}')
    print(f'Test  ({len(lgo_test_names)}): {", ".join(lgo_test_names)}')

    print(f'\n{"Mol":>6s}  {"Direct":>8s}  {"Rose":>8s}  {"b0_a":>8s}  {"b0_sqrt":>8s}  '
          f'{"Rose/D":>7s}  {"b0a/D":>7s}  {"b0s/D":>7s}')
    for i, name in enumerate(lgo_test_names):
        md = lgo_results['direct']['mae'][i]
        mr = lgo_results['rose']['mae'][i]
        ma = lgo_results['b0_a']['mae'][i]
        ms = lgo_results['b0_sqrt']['mae'][i]
        print(f'{name:>6s}  {md:8.4f}  {mr:8.4f}  {ma:8.4f}  {ms:8.4f}  '
              f'{md/mr if mr > 0 else 0:7.2f}x  '
              f'{md/ma if ma > 0 else 0:7.2f}x  '
              f'{md/ms if ms > 0 else 0:7.2f}x')

    print('\n  Mean MAEs:')
    for method, label in [('direct', 'Direct'), ('rose', 'Rose'),
                          ('b0_a', 'b=0 a(r)'), ('b0_sqrt', 'b=0 sqrt(a)')]:
        m = np.mean(lgo_results[method]['mae'])
        print(f'    {label:20s} {m:.4f} eV')

    md_lgo = np.mean(lgo_results['direct']['mae'])
    print('\n  Ratios (Direct / Adaptive):')
    for method, label in [('rose', 'Rose'), ('b0_a', 'b=0 a(r)'),
                          ('b0_sqrt', 'b=0 sqrt(a)')]:
        m = np.mean(lgo_results[method]['mae'])
        print(f'    {label:20s} {md_lgo/m:.2f}x')

    plot_comparison(lgo_test_names, lgo_results, 'LGO', r_scaled_common)

    # ================================================================
    # Summary
    # ================================================================
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'\n  Mean MAE [eV]:')
    print(f'  {"Regime":<8s}  {"Direct":>8s}  {"Rose":>8s}  {"b0_a":>8s}  {"b0_sqrt":>8s}')
    for regime, res in [('LOO', loo_results), ('LGO', lgo_results)]:
        maes = {m: np.mean(res[m]['mae']) for m in ['direct', 'rose', 'b0_a', 'b0_sqrt']}
        print(f'  {regime:<8s}  {maes["direct"]:8.4f}  {maes["rose"]:8.4f}  '
              f'{maes["b0_a"]:8.4f}  {maes["b0_sqrt"]:8.4f}')

    print(f'\n  Ratios (Direct / Adaptive):')
    print(f'  {"Regime":<8s}  {"Rose":>8s}  {"b0_a":>8s}  {"b0_sqrt":>8s}')
    for regime, res in [('LOO', loo_results), ('LGO', lgo_results)]:
        md = np.mean(res['direct']['mae'])
        print(f'  {regime:<8s}  '
              f'{md/np.mean(res["rose"]["mae"]):8.2f}x  '
              f'{md/np.mean(res["b0_a"]["mae"]):8.2f}x  '
              f'{md/np.mean(res["b0_sqrt"]["mae"]):8.2f}x')


if __name__ == '__main__':
    main()
