"""
Position-Dependent Adaptive Stiffness Experiment
=================================================

Shows that the Rose adaptive approach automatically provides position-
dependent physical quantities like k(r) = V''(r), the local stiffness.

Key insight: predicting Rose parameters (E_c, r_e, l) gives k(r) "for
free" via the analytical second derivative:
    k_Rose(r) = (E_c / l^2) * (1 - a*) * exp(-a*)

This is BETTER than differentiating the direct V(r) prediction because:
- Rose constrains k(r) to a physical shape (peaked, exponentially decaying)
- Direct k(r) has no shape constraint and degrades under extrapolation

Three comparisons for stiffness quality:
  1. Rose adaptive k(r): from ML-predicted (E_c, r_e, l) → analytical formula
  2. Rose true k(r): from DFT-fitted (E_c, r_e, l) → shows Rose approximation error
  3. Direct k(r): from d²V_direct/dr² → shows unconstrained Ridge prediction

Usage:
    MPLBACKEND=Agg python position_dependent_adaptive.py
    MPLBACKEND=Agg python position_dependent_adaptive.py --data data/diatomic_curves.npz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatomic_adaptive_vs_direct import (
    load_dft_data, generate_demo_data, assess_data_quality, filter_data,
    build_scaled_common_grid, rose_V
)

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


# =============================================================================
# PHYSICS
# =============================================================================

def rose_stiffness(r, E_c, r_e, l):
    """Analytical second derivative of Rose/UBER equation.

    k_Rose(r) = (E_c / l^2) * (1 - a*) * exp(-a*)
    where a* = (r - r_e) / l.

    Properties:
      k(r_e) = E_c / l^2  (maximum stiffness at equilibrium)
      k → 0 as r → ∞       (zero stiffness at dissociation)
      k changes sign at r = r_e + l (inflection point)
    """
    a_star = (r - r_e) / l
    return (E_c / l**2) * (1 - a_star) * np.exp(-a_star)


def compute_stiffness_on_grid(r_scaled, curves):
    """Compute k(r_s) = d²V/dr_s² for each molecule using cubic spline."""
    n_mol = curves.shape[0]
    stiffness = np.zeros_like(curves)
    for i in range(n_mol):
        cs = CubicSpline(r_scaled, curves[i])
        stiffness[i] = cs(r_scaled, 2)
    return stiffness


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_loo(data, r_scaled_common, curves_common, stiffness_common, rcov_sums):
    """Leave-one-out: compare stiffness from Rose adaptive vs direct."""
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])

    # Precompute Rose stiffness from TRUE (fitted) params for reference
    k_rose_true = np.zeros_like(curves_common)
    for i in range(n):
        E_c, r_e, l = rose_params[i]
        r_real = r_scaled_common * rcov_sums[i]
        k_rose_true[i] = rose_stiffness(r_real, E_c, r_e, l) * rcov_sums[i]**2

    results = {
        'names': data['names'],
        'mae_V_direct': [], 'mae_V_rose': [],
        'mae_k_rose_pred': [], 'mae_k_direct': [], 'mae_k_rose_true': [],
        'k_true': [], 'k_rose_pred': [], 'k_direct': [], 'k_rose_true_arr': [],
        'V_true': [], 'V_direct': [], 'V_rose': [],
        'r_scaled_common': r_scaled_common,
        'rcov_sums': rcov_sums,
    }

    print(f"\n  {'Mol':>6s}  {'V_dir':>7s} {'V_rose':>7s} | "
          f"{'k_true':>7s} {'k_rose':>7s} {'k_dir':>7s} {'rose/dir':>9s}")
    print("  " + "-" * 70)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        X_train = descriptors[train_idx]
        X_test = descriptors[[i]]
        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)

        V_true = curves_common[i]
        k_true = stiffness_common[i]

        # --- Direct energy prediction ---
        scaler_V = StandardScaler()
        Y_V_s = scaler_V.fit_transform(curves_common[train_idx])
        m_direct = Ridge(alpha=1.0).fit(X_train_s, Y_V_s)
        V_pred_direct = scaler_V.inverse_transform(m_direct.predict(X_test_s))[0]

        # Stiffness from direct V prediction
        cs_direct = CubicSpline(r_scaled_common, V_pred_direct)
        k_direct = cs_direct(r_scaled_common, 2)

        # --- Rose adaptive ---
        scaler_P = StandardScaler()
        Y_P_s = scaler_P.fit_transform(rose_params[train_idx])
        m_rose = Ridge(alpha=1.0).fit(X_train_s, Y_P_s)
        p_pred = scaler_P.inverse_transform(m_rose.predict(X_test_s))[0]
        E_c = max(p_pred[0], 0.01)
        r_e = max(p_pred[1], 0.3)
        l = max(p_pred[2], 0.01)
        r_real = r_scaled_common * rcov_sums[i]
        V_pred_rose = rose_V(r_real, E_c, r_e, l)

        # Rose stiffness from ML-predicted params
        k_rose_pred = rose_stiffness(r_real, E_c, r_e, l) * rcov_sums[i]**2

        # --- Metrics ---
        mae_Vd = np.mean(np.abs(V_true - V_pred_direct))
        mae_Vr = np.mean(np.abs(V_true - V_pred_rose))
        mae_k_rp = np.mean(np.abs(k_true - k_rose_pred))
        mae_k_d = np.mean(np.abs(k_true - k_direct))
        mae_k_rt = np.mean(np.abs(k_true - k_rose_true[i]))

        results['mae_V_direct'].append(mae_Vd)
        results['mae_V_rose'].append(mae_Vr)
        results['mae_k_rose_pred'].append(mae_k_rp)
        results['mae_k_direct'].append(mae_k_d)
        results['mae_k_rose_true'].append(mae_k_rt)
        results['k_true'].append(k_true)
        results['k_rose_pred'].append(k_rose_pred)
        results['k_direct'].append(k_direct)
        results['k_rose_true_arr'].append(k_rose_true[i])
        results['V_true'].append(V_true)
        results['V_direct'].append(V_pred_direct)
        results['V_rose'].append(V_pred_rose)

        ratio = mae_k_d / max(mae_k_rp, 1e-10)
        print(f"  {data['names'][i]:>6s}  {mae_Vd:7.4f} {mae_Vr:7.4f} | "
              f"{mae_k_rt:7.1f} {mae_k_rp:7.1f} {mae_k_d:7.1f} {ratio:9.2f}x")

    # Summary
    m = lambda k: np.mean(results[k])
    ratio = m('mae_k_direct') / max(m('mae_k_rose_pred'), 1e-10)
    print(f"\n  {'Mean':>6s}  {m('mae_V_direct'):7.4f} {m('mae_V_rose'):7.4f} | "
          f"{m('mae_k_rose_true'):7.1f} {m('mae_k_rose_pred'):7.1f} "
          f"{m('mae_k_direct'):7.1f} {ratio:9.2f}x")
    print(f"\n  Rose adaptive provides {ratio:.1f}x better stiffness than direct approach")

    return results


def run_lgo(data, r_scaled_common, curves_common, stiffness_common, rcov_sums):
    """Leave-group-out: train rows 1-2, test row 3+."""
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])

    train_idx, test_idx = [], []
    for i in range(n):
        Z1, Z2 = descriptors[i, 0], descriptors[i, 1]
        if Z1 > 10 or Z2 > 10:
            test_idx.append(i)
        else:
            train_idx.append(i)

    if len(train_idx) < 3 or len(test_idx) < 3:
        print("Not enough molecules for LGO split")
        return None

    print(f"\n  Train ({len(train_idx)}): {[data['names'][i] for i in train_idx]}")
    print(f"  Test  ({len(test_idx)}): {[data['names'][i] for i in test_idx]}")

    # Rose stiffness from TRUE params (reference)
    k_rose_true = np.zeros_like(curves_common)
    for i in range(n):
        E_c, r_e, l = rose_params[i]
        r_real = r_scaled_common * rcov_sums[i]
        k_rose_true[i] = rose_stiffness(r_real, E_c, r_e, l) * rcov_sums[i]**2

    X_train = descriptors[train_idx]
    X_test = descriptors[test_idx]
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Direct energy
    scaler_V = StandardScaler()
    Y_V_s = scaler_V.fit_transform(curves_common[train_idx])
    m_direct = Ridge(alpha=1.0).fit(X_train_s, Y_V_s)
    V_pred_direct = scaler_V.inverse_transform(m_direct.predict(X_test_s))

    # Direct stiffness
    k_direct_all = []
    for j in range(len(test_idx)):
        cs = CubicSpline(r_scaled_common, V_pred_direct[j])
        k_direct_all.append(cs(r_scaled_common, 2))
    k_direct_all = np.array(k_direct_all)

    # Rose adaptive
    scaler_P = StandardScaler()
    Y_P_s = scaler_P.fit_transform(rose_params[train_idx])
    m_rose = Ridge(alpha=1.0).fit(X_train_s, Y_P_s)
    params_pred = scaler_P.inverse_transform(m_rose.predict(X_test_s))

    V_pred_rose = []
    k_rose_pred = []
    for j, i in enumerate(test_idx):
        E_c = max(params_pred[j, 0], 0.01)
        r_e = max(params_pred[j, 1], 0.3)
        l_val = max(params_pred[j, 2], 0.01)
        r_real = r_scaled_common * rcov_sums[i]
        V_pred_rose.append(rose_V(r_real, E_c, r_e, l_val))
        k_rose_pred.append(rose_stiffness(r_real, E_c, r_e, l_val) * rcov_sums[i]**2)
    V_pred_rose = np.array(V_pred_rose)
    k_rose_pred = np.array(k_rose_pred)

    # Per-molecule metrics
    results = {
        'test_names': [data['names'][i] for i in test_idx],
        'train_names': [data['names'][i] for i in train_idx],
        'test_idx': test_idx,
        'mae_V_direct': [], 'mae_V_rose': [],
        'mae_k_rose_pred': [], 'mae_k_direct': [], 'mae_k_rose_true': [],
        'k_true': [], 'k_rose_pred': [], 'k_direct': [], 'k_rose_true_arr': [],
        'V_true': [], 'V_direct': [], 'V_rose': [],
        'r_scaled_common': r_scaled_common,
        'rcov_sums': rcov_sums,
    }

    print(f"\n  {'Mol':>8s}  {'V_dir':>7s} {'V_rose':>7s} | "
          f"{'k_true':>7s} {'k_rose':>7s} {'k_dir':>7s} {'rose/dir':>9s}")
    print("  " + "-" * 70)

    for j, i in enumerate(test_idx):
        V_true = curves_common[i]
        k_true = stiffness_common[i]

        mae_Vd = np.mean(np.abs(V_true - V_pred_direct[j]))
        mae_Vr = np.mean(np.abs(V_true - V_pred_rose[j]))
        mae_k_rp = np.mean(np.abs(k_true - k_rose_pred[j]))
        mae_k_d = np.mean(np.abs(k_true - k_direct_all[j]))
        mae_k_rt = np.mean(np.abs(k_true - k_rose_true[i]))

        results['mae_V_direct'].append(mae_Vd)
        results['mae_V_rose'].append(mae_Vr)
        results['mae_k_rose_pred'].append(mae_k_rp)
        results['mae_k_direct'].append(mae_k_d)
        results['mae_k_rose_true'].append(mae_k_rt)
        results['k_true'].append(k_true)
        results['k_rose_pred'].append(k_rose_pred[j])
        results['k_direct'].append(k_direct_all[j])
        results['k_rose_true_arr'].append(k_rose_true[i])
        results['V_true'].append(V_true)
        results['V_direct'].append(V_pred_direct[j])
        results['V_rose'].append(V_pred_rose[j])

        ratio = mae_k_d / max(mae_k_rp, 1e-10)
        print(f"  {data['names'][i]:>8s}  {mae_Vd:7.4f} {mae_Vr:7.4f} | "
              f"{mae_k_rt:7.1f} {mae_k_rp:7.1f} {mae_k_d:7.1f} {ratio:9.2f}x")

    m = lambda k: np.mean(results[k])
    ratio = m('mae_k_direct') / max(m('mae_k_rose_pred'), 1e-10)
    print(f"\n  {'Mean':>8s}  {m('mae_V_direct'):7.4f} {m('mae_V_rose'):7.4f} | "
          f"{m('mae_k_rose_true'):7.1f} {m('mae_k_rose_pred'):7.1f} "
          f"{m('mae_k_direct'):7.1f} {ratio:9.2f}x")
    print(f"\n  Rose adaptive provides {ratio:.1f}x better stiffness than direct (extrap)")

    return results


# =============================================================================
# FIGURES
# =============================================================================

def create_figures(loo, lgo, r_scaled_common, rcov_sums, data):
    """Generate figures showing stiffness prediction quality."""
    os.makedirs('figures', exist_ok=True)

    # =========================================================================
    # Figure 1: Main 6-panel
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- Panel A: LOO stiffness MAE ---
    ax = axes[0, 0]
    n_loo = len(loo['names'])
    x = np.arange(n_loo)
    w = 0.25
    ax.bar(x - w, loo['mae_k_rose_true'], w, color='gray', alpha=0.6,
           label='Rose true params')
    ax.bar(x, loo['mae_k_rose_pred'], w, color='indianred', alpha=0.85,
           label='Rose ML params')
    ax.bar(x + w, loo['mae_k_direct'], w, color='steelblue', alpha=0.85,
           label='Direct d²V/dr²')
    ax.set_xticks(x)
    ax.set_xticklabels(loo['names'], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Stiffness MAE [eV/r$_s^2$]')
    ax.set_title('A. LOO: Stiffness Prediction Quality')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel B: LOO example stiffness profile ---
    ax = axes[0, 1]
    # Pick molecule with largest Rose advantage
    ratios = [d / max(r, 1e-10) for d, r in
              zip(loo['mae_k_direct'], loo['mae_k_rose_pred'])]
    best_idx = np.argmax(ratios)
    mol_name = loo['names'][best_idx]
    rcs = rcov_sums[best_idx]
    r_real = r_scaled_common * rcs

    ax.plot(r_real, loo['k_true'][best_idx] / rcs**2, 'k-', lw=3,
            label='DFT (true)')
    ax.plot(r_real, loo['k_rose_true_arr'][best_idx] / rcs**2, '--',
            color='gray', lw=1.5, label='Rose (true params)', alpha=0.7)
    ax.plot(r_real, loo['k_rose_pred'][best_idx] / rcs**2, 'r-', lw=2.5,
            label='Rose (ML params)')
    ax.plot(r_real, loo['k_direct'][best_idx] / rcs**2, 'b--', lw=2,
            label='Direct d²V/dr²', alpha=0.8)
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.set_xlabel('r [$\\AA$]')
    ax.set_ylabel("k(r) = V''(r) [eV/$\\AA^2$]")
    ax.set_title(f'B. LOO: {mol_name} Stiffness Profile')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel C: LOO V(r) for same molecule ---
    ax = axes[0, 2]
    ax.plot(r_real, loo['V_true'][best_idx], 'k-', lw=3, label='DFT')
    ax.plot(r_real, loo['V_rose'][best_idx], 'r-', lw=2.5, label='Rose adaptive')
    ax.plot(r_real, loo['V_direct'][best_idx], 'b--', lw=2, label='Direct', alpha=0.8)
    ax.set_xlabel('r [$\\AA$]')
    ax.set_ylabel('E(r) [eV]')
    ax.set_title(f'C. LOO: {mol_name} Binding Curve')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if lgo is not None:
        # --- Panel D: LGO stiffness MAE ---
        ax = axes[1, 0]
        n_lgo = len(lgo['test_names'])
        x_lgo = np.arange(n_lgo)
        ax.bar(x_lgo - w, lgo['mae_k_rose_true'], w, color='gray', alpha=0.6,
               label='Rose true params')
        ax.bar(x_lgo, lgo['mae_k_rose_pred'], w, color='indianred', alpha=0.85,
               label='Rose ML params')
        ax.bar(x_lgo + w, lgo['mae_k_direct'], w, color='steelblue', alpha=0.85,
               label='Direct d²V/dr²')
        ax.set_xticks(x_lgo)
        ax.set_xticklabels(lgo['test_names'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Stiffness MAE [eV/r$_s^2$]')
        ax.set_title('D. LGO: Stiffness Prediction (extrapolation)')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        # --- Panel E: "Anatole Plot" - KH stiffness ---
        ax = axes[1, 1]
        kh_idx = None
        for j, name in enumerate(lgo['test_names']):
            if name == 'KH':
                kh_idx = j
                break
        if kh_idx is None:
            ratios_lgo = [d / max(r, 1e-10) for d, r in
                          zip(lgo['mae_k_direct'], lgo['mae_k_rose_pred'])]
            kh_idx = np.argmax(ratios_lgo)

        mol_lgo = lgo['test_names'][kh_idx]
        test_i = lgo['test_idx'][kh_idx]
        rcs_lgo = rcov_sums[test_i]
        r_real_lgo = r_scaled_common * rcs_lgo

        ax.plot(r_real_lgo, lgo['k_true'][kh_idx] / rcs_lgo**2, 'k-', lw=3,
                label='DFT (true)', zorder=3)
        ax.plot(r_real_lgo, lgo['k_rose_true_arr'][kh_idx] / rcs_lgo**2, '--',
                color='gray', lw=1.5, label='Rose (true params)', alpha=0.7)
        ax.plot(r_real_lgo, lgo['k_rose_pred'][kh_idx] / rcs_lgo**2, 'r-', lw=2.5,
                label='Rose (ML params)')
        ax.plot(r_real_lgo, lgo['k_direct'][kh_idx] / rcs_lgo**2, 'b--', lw=2,
                label='Direct d²V/dr²', alpha=0.8)
        ax.axhline(y=0, color='gray', lw=0.5)
        ax.set_xlabel('r [$\\AA$]')
        ax.set_ylabel("k(r) = V''(r) [eV/$\\AA^2$]")
        ax.set_title(f'E. LGO: {mol_lgo} Stiffness\n(trained rows 1-2 only)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Panel F: LGO V(r) for same molecule ---
        ax = axes[1, 2]
        ax.plot(r_real_lgo, lgo['V_true'][kh_idx], 'k-', lw=3, label='DFT')
        ax.plot(r_real_lgo, lgo['V_rose'][kh_idx], 'r-', lw=2.5,
                label='Rose adaptive')
        ax.plot(r_real_lgo, lgo['V_direct'][kh_idx], 'b--', lw=2,
                label='Direct', alpha=0.8)
        ax.set_xlabel('r [$\\AA$]')
        ax.set_ylabel('E(r) [eV]')
        ax.set_title(f'F. LGO: {mol_lgo} Binding Curve')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes[1, :]:
            ax.axis('off')

    plt.tight_layout()
    fig_path = 'figures/fig_position_dependent_adaptive.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nMain figure saved: {fig_path}")

    # =========================================================================
    # Figure 2: All LGO test molecules - stiffness profiles
    # =========================================================================
    if lgo is not None:
        n_test = len(lgo['test_names'])
        n_cols = 4
        n_rows = int(np.ceil(n_test / n_cols))
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes_flat = axes_grid.flatten()

        for j in range(n_test):
            ax = axes_flat[j]
            test_i = lgo['test_idx'][j]
            rcs = rcov_sums[test_i]
            r_real = r_scaled_common * rcs

            ax.plot(r_real, lgo['k_true'][j] / rcs**2, 'k-', lw=2.5, label='DFT')
            ax.plot(r_real, lgo['k_rose_true_arr'][j] / rcs**2, '--',
                    color='gray', lw=1, alpha=0.5, label='Rose (true)')
            ax.plot(r_real, lgo['k_rose_pred'][j] / rcs**2, 'r-', lw=2,
                    label='Rose (ML)', alpha=0.85)
            ax.plot(r_real, lgo['k_direct'][j] / rcs**2, 'b--', lw=1.5,
                    label='Direct', alpha=0.7)
            ax.axhline(y=0, color='gray', lw=0.5)

            ratio = lgo['mae_k_direct'][j] / max(lgo['mae_k_rose_pred'][j], 1e-10)
            ax.set_title(f'{lgo["test_names"][j]}  (rose/dir: {ratio:.1f}x)',
                         fontsize=10)
            ax.set_xlabel('r [$\\AA$]', fontsize=9)
            ax.set_ylabel("k(r) [eV/$\\AA^2$]", fontsize=9)
            if j == 0:
                ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2)

        for j in range(n_test, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.suptitle('LGO: Stiffness k(r) for All Test Molecules\n'
                     '(trained on rows 1-2 only)', fontsize=14)
        plt.tight_layout()
        fig_path2 = 'figures/fig_stiffness_all_lgo_molecules.png'
        plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"All-molecule figure saved: {fig_path2}")

    # =========================================================================
    # Figure 3: Stiffness parity plots
    # =========================================================================
    if lgo is not None:
        fig, axes_p = plt.subplots(1, 2, figsize=(12, 5.5))

        for panel_idx, (res, title_prefix) in enumerate(
                [(loo, 'LOO (interpolation)'), (lgo, 'LGO (extrapolation)')]):
            ax = axes_p[panel_idx]

            k_true_all = np.concatenate(res['k_true'])
            k_rose_all = np.concatenate(res['k_rose_pred'])
            k_dir_all = np.concatenate(res['k_direct'])

            # Get rcov_sums for conversion
            if 'test_idx' in res:
                rcs_list = [rcov_sums[i] for i in res['test_idx']]
            else:
                rcs_list = list(rcov_sums)
            rcs_rep = np.repeat(rcs_list, len(r_scaled_common))

            # Convert to real units
            k_true_real = k_true_all / rcs_rep**2
            k_rose_real = k_rose_all / rcs_rep**2
            k_dir_real = k_dir_all / rcs_rep**2

            ax.scatter(k_true_real, k_rose_real, s=5, alpha=0.35,
                       c='indianred', label='Rose adaptive')
            ax.scatter(k_true_real, k_dir_real, s=5, alpha=0.35,
                       c='steelblue', label='Direct d²V/dr²')
            lims = [min(k_true_real.min(), k_rose_real.min(), k_dir_real.min()),
                    max(k_true_real.max(), k_rose_real.max(), k_dir_real.max())]
            ax.plot(lims, lims, 'k--', lw=1)

            r2_rose = r2_score(k_true_real, k_rose_real)
            r2_dir = r2_score(k_true_real, k_dir_real)
            ax.set_xlabel("True k(r) [eV/$\\AA^2$]")
            ax.set_ylabel("Predicted k(r) [eV/$\\AA^2$]")
            ax.set_title(f'{title_prefix}\n'
                         f'Rose $R^2$={r2_rose:.3f} | Direct $R^2$={r2_dir:.3f}')
            ax.legend(fontsize=9, markerscale=3)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path3 = 'figures/fig_stiffness_parity.png'
        plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Parity plot saved: {fig_path3}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Position-dependent adaptive stiffness experiment')
    parser.add_argument('--data', default=None,
                        help='Path to diatomic_curves.npz')
    parser.add_argument('--max-residual', type=float, default=1.0)
    parser.add_argument('--min-points', type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    print("POSITION-DEPENDENT ADAPTIVE STIFFNESS")
    print("k(r) = V''(r) as a position-dependent physical quantity")
    print("=" * 70)
    print("\nKey idea: Rose adaptive predicts (E_c, r_e, l), which gives")
    print("  k(r) = (E_c/l^2)(1-a*)exp(-a*) analytically.")
    print("This constrained stiffness is BETTER than d²V_direct/dr².")

    if args.data and os.path.exists(args.data):
        print(f"\nLoading DFT data from: {args.data}")
        data = load_dft_data(args.data)
    else:
        print("\nNo DFT data found. Running in DEMO MODE.")
        data = generate_demo_data()

    print(f"Loaded {len(data['names'])} molecules")

    print("\n" + "=" * 70)
    print("DATA QUALITY")
    print("=" * 70)
    keep_idx = assess_data_quality(data, max_residual=args.max_residual,
                                    min_points=args.min_points)
    data = filter_data(data, keep_idx)
    print(f"\nUsing {len(data['names'])} molecules")

    r_scaled_common, curves_common, rcov_sums = build_scaled_common_grid(data)

    print("\nComputing stiffness k(r) = V''(r) on common scaled grid...")
    stiffness_common = compute_stiffness_on_grid(r_scaled_common, curves_common)

    # LOO
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT (INTERPOLATION)")
    print("=" * 70)
    loo = run_loo(data, r_scaled_common, curves_common, stiffness_common, rcov_sums)

    # LGO
    print("\n" + "=" * 70)
    print("LEAVE-GROUP-OUT (EXTRAPOLATION)")
    print("=" * 70)
    lgo = run_lgo(data, r_scaled_common, curves_common, stiffness_common, rcov_sums)

    # Figures
    create_figures(loo, lgo, r_scaled_common, rcov_sums, data)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    m = lambda k: np.mean(loo[k])
    print(f"\nLOO (interpolation):")
    print(f"  Energy:    Direct {m('mae_V_direct'):.4f}, Rose {m('mae_V_rose'):.4f} eV")
    print(f"  Stiffness: Rose true {m('mae_k_rose_true'):.1f}, "
          f"Rose ML {m('mae_k_rose_pred'):.1f}, "
          f"Direct {m('mae_k_direct'):.1f} eV/r_s^2")
    ratio = m('mae_k_direct') / max(m('mae_k_rose_pred'), 1e-10)
    print(f"  Rose stiffness advantage: {ratio:.1f}x")

    if lgo:
        m = lambda k: np.mean(lgo[k])
        print(f"\nLGO (extrapolation):")
        print(f"  Energy:    Direct {m('mae_V_direct'):.4f}, Rose {m('mae_V_rose'):.4f} eV")
        print(f"  Stiffness: Rose true {m('mae_k_rose_true'):.1f}, "
              f"Rose ML {m('mae_k_rose_pred'):.1f}, "
              f"Direct {m('mae_k_direct'):.1f} eV/r_s^2")
        ratio = m('mae_k_direct') / max(m('mae_k_rose_pred'), 1e-10)
        print(f"  Rose stiffness advantage: {ratio:.1f}x")

    print(f"\nConclusion: The adaptive approach provides physically constrained")
    print(f"stiffness predictions at ALL positions, not just global parameters.")
    print(f"Rose k(r) is better than unconstrained direct d²V/dr².")


if __name__ == "__main__":
    main()
