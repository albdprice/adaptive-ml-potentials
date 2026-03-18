"""
Adaptive vs Direct Learning on DFT Diatomic Binding Curves
============================================================

Takes output from diatomic_scan.py and runs the adaptive vs direct comparison
on real DFT data, using leave-one-out or leave-group-out cross-validation.

Key methodological point: different molecules have very different equilibrium
distances (H2 ~0.74 A vs Na2 ~3.08 A), so a common r-grid in Angstrom would
have negligible overlap. We use scaled coordinates r_s = r / r_cov_sum (sum of
covalent radii, available from descriptors) to create a universal representation.

Usage:
    MPLBACKEND=Agg python diatomic_adaptive_vs_direct.py
    MPLBACKEND=Agg python diatomic_adaptive_vs_direct.py --data data/diatomic_curves.npz

If no DFT data is available, runs in "demo mode" with synthetic Morse curves
parameterized by realistic atomic descriptors.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


# =============================================================================
# PHYSICS
# =============================================================================

def rose_V(r, D_e, r_e, l):
    """Rose/UBER equation: E(r) = -D_e * (1 + a*) * exp(-a*).

    Convention: E(r_e) = -D_e (bound minimum), E(inf) = 0 (dissociation).
    D_e > 0 is the well depth.
    """
    a_star = (r - r_e) / l
    return -D_e * (1 + a_star) * np.exp(-a_star)

def morse_V(r, De, a, re):
    """Morse potential (bound state convention: V = -De at minimum)."""
    return De * ((1 - np.exp(-a * (r - re)))**2 - 1)


# =============================================================================
# SYNTHETIC DEMO DATA (when no DFT data available)
# =============================================================================

# Realistic parameters for 30 diatomics (from NIST/CRC handbook values)
DEMO_DIATOMICS = [
    # (name, De_eV, r_e_A, omega_cm-1, Z1, Z2, EN1, EN2, rcov1, rcov2, IE1, IE2)
    ('H2',   4.75, 0.74, 4401, 1,  1, 2.20, 2.20, 0.31, 0.31, 13.60, 13.60),
    ('LiH',  2.52, 1.60, 1405, 3,  1, 0.98, 2.20, 1.28, 0.31,  5.39, 13.60),
    ('BH',   3.54, 1.23, 2367, 5,  1, 2.04, 2.20, 0.84, 0.31,  8.30, 13.60),
    ('CH',   3.65, 1.12, 2861, 6,  1, 2.55, 2.20, 0.76, 0.31, 11.26, 13.60),
    ('NH',   3.47, 1.04, 3282, 7,  1, 3.04, 2.20, 0.71, 0.31, 14.53, 13.60),
    ('OH',   4.62, 0.97, 3738, 8,  1, 3.44, 2.20, 0.66, 0.31, 13.62, 13.60),
    ('HF',   5.87, 0.92, 4138, 9,  1, 3.98, 2.20, 0.57, 0.31, 17.42, 13.60),
    ('NaH',  1.97, 1.89,  1172, 11, 1, 0.93, 2.20, 1.66, 0.31,  5.14, 13.60),
    ('AlH',  3.06, 1.65, 1682, 13, 1, 1.61, 2.20, 1.21, 0.31,  5.99, 13.60),
    ('SiH',  3.23, 1.52, 2042, 14, 1, 1.90, 2.20, 1.11, 0.31,  8.15, 13.60),
    ('HCl',  4.43, 1.27, 2991, 1, 17, 2.20, 3.16, 0.31, 1.02, 13.60, 12.97),
    ('KH',   1.77, 2.24,  983, 19, 1, 0.82, 2.20, 2.03, 0.31,  4.34, 13.60),
    ('Li2',  1.05, 2.67,  351, 3,  3, 0.98, 0.98, 1.28, 1.28,  5.39,  5.39),
    ('N2',   9.79, 1.10, 2359, 7,  7, 3.04, 3.04, 0.71, 0.71, 14.53, 14.53),
    ('O2',   5.21, 1.21, 1580, 8,  8, 3.44, 3.44, 0.66, 0.66, 13.62, 13.62),
    ('F2',   1.66, 1.41,  917, 9,  9, 3.98, 3.98, 0.57, 0.57, 17.42, 17.42),
    ('Na2',  0.75, 3.08,  159, 11, 11, 0.93, 0.93, 1.66, 1.66,  5.14,  5.14),
    ('Cl2',  2.51, 1.99,  560, 17, 17, 3.16, 3.16, 1.02, 1.02, 12.97, 12.97),
    ('LiF',  5.91, 1.56, 910, 3,  9, 0.98, 3.98, 1.28, 0.57,  5.39, 17.42),
    ('NaF',  4.97, 1.93,  536, 11, 9, 0.93, 3.98, 1.66, 0.57,  5.14, 17.42),
    ('NaCl', 4.24, 2.36,  366, 11, 17, 0.93, 3.16, 1.66, 1.02,  5.14, 12.97),
    ('CO',   11.11, 1.13, 2170, 6,  8, 2.55, 3.44, 0.76, 0.66, 11.26, 13.62),
    ('NO',   6.50, 1.15, 1904, 7,  8, 3.04, 3.44, 0.71, 0.66, 14.53, 13.62),
    ('BF',   7.81, 1.27, 1402, 5,  9, 2.04, 3.98, 0.84, 0.57,  8.30, 17.42),
    ('AlF',  6.88, 1.65,  802, 13, 9, 1.61, 3.98, 1.21, 0.57,  5.99, 17.42),
    ('BeH',  2.27, 1.34, 2061, 4,  1, 1.57, 2.20, 0.96, 0.31,  9.32, 13.60),
    ('PH',   3.09, 1.42, 2364, 15, 1, 2.19, 2.20, 1.07, 0.31, 10.49, 13.60),
    ('SH',   3.73, 1.34, 2712, 16, 1, 2.58, 2.20, 1.05, 0.31, 10.36, 13.60),
    ('MgH',  1.34, 1.73, 1495, 12, 1, 1.31, 2.20, 1.41, 0.31,  7.65, 13.60),
    ('CN',   7.72, 1.17, 2069, 6,  7, 2.55, 3.04, 0.76, 0.71, 11.26, 14.53),
]


def generate_demo_data():
    """Generate synthetic Morse curves with realistic parameters and descriptors."""
    n_r = 50
    data = {
        'names': [],
        'descriptors': [],
        'rose_params': [],
        'rose_residuals': [],
        'curves': [],
        'r_grids': [],
    }

    for entry in DEMO_DIATOMICS:
        name = entry[0]
        De, r_e, omega = entry[1], entry[2], entry[3]
        Z1, Z2, EN1, EN2, rcov1, rcov2, IE1, IE2 = entry[4:]

        # Morse stiffness from harmonic frequency (approximate)
        a = omega / (1000.0 * np.sqrt(max(De, 0.1)))
        a = max(a, 0.5)
        a = min(a, 5.0)

        r = np.linspace(max(0.5, 0.6 * r_e), 3.5 * r_e, n_r)
        V = morse_V(r, De, a, r_e)

        # Fit Rose params
        try:
            popt, _ = curve_fit(
                rose_V, r, V,
                p0=[De, r_e, 1.0/a],
                bounds=([0.01, 0.3, 0.01], [20, 10, 5]),
                maxfev=10000,
            )
        except Exception:
            continue

        E_fit = rose_V(r, *popt)
        residual = np.sqrt(np.mean((V - E_fit)**2))

        descriptor = [
            Z1, Z2, EN1, EN2, rcov1, rcov2, IE1, IE2,
            EN1 + EN2, abs(EN1 - EN2),
            rcov1 + rcov2, abs(rcov1 - rcov2),
            IE1 + IE2, abs(IE1 - IE2),
        ]

        data['names'].append(name)
        data['descriptors'].append(descriptor)
        data['rose_params'].append(popt)
        data['rose_residuals'].append(residual)
        data['curves'].append(V)
        data['r_grids'].append(r)

    return data


# =============================================================================
# LOAD DFT DATA
# =============================================================================

def load_dft_data(path):
    """Load data from diatomic_scan.py output."""
    d = np.load(path, allow_pickle=True)
    n = int(d['n_molecules'])

    data = {
        'names': [str(x) for x in d['names']],
        'descriptors': d['descriptors'].copy(),
        'rose_params': d['rose_params'].copy(),
        'rose_residuals': [],
        'curves': [],
        'r_grids': [],
    }

    # Compute Rose residuals
    for i in range(n):
        r = d[f'r_{i}']
        E = d[f'E_bind_{i}']
        D_e, r_e, l = d['rose_params'][i]
        E_fit = rose_V(r, D_e, r_e, l)
        residual = np.sqrt(np.mean((E - E_fit)**2))
        data['rose_residuals'].append(residual)
        data['curves'].append(E)
        data['r_grids'].append(r)

    return data


# =============================================================================
# DATA QUALITY ASSESSMENT
# =============================================================================

def assess_data_quality(data, max_residual=1.0, min_points=8):
    """Assess data quality and return indices of usable molecules.

    Returns (keep_idx, quality_info) where keep_idx is a list of molecule
    indices to include, and quality_info is a dict with details.
    """
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    keep_idx = []
    excluded = []

    print(f"\n  {'Name':>6} {'N_pts':>5} {'Res[eV]':>8} {'E_c':>6} {'r_e':>6} "
          f"{'l':>5} {'Status':>12}")
    print("  " + "-" * 60)

    for i in range(n):
        name = data['names'][i]
        r = data['r_grids'][i]
        residual = data['rose_residuals'][i]
        D_e, r_e, l = data['rose_params'][i]
        n_pts = len(r)

        if residual > max_residual:
            status = "EXCLUDED(fit)"
            excluded.append((name, f"Rose residual {residual:.2f} > {max_residual}"))
        elif n_pts < min_points:
            status = "EXCLUDED(pts)"
            excluded.append((name, f"Only {n_pts} converged points"))
        else:
            if residual < 0.1:
                status = "clean"
            elif residual < 0.3:
                status = "good"
            else:
                status = "marginal"
            keep_idx.append(i)

        print(f"  {name:>6} {n_pts:>5} {residual:>8.3f} {D_e:>6.2f} {r_e:>6.3f} "
              f"{l:>5.3f} {status:>12}")

    if excluded:
        print(f"\n  Excluded {len(excluded)} molecules:")
        for name, reason in excluded:
            print(f"    {name}: {reason}")

    print(f"\n  Keeping {len(keep_idx)}/{n} molecules for ML comparison")
    return keep_idx


def filter_data(data, keep_idx):
    """Return a new data dict with only the kept molecules."""
    filtered = {
        'names': [data['names'][i] for i in keep_idx],
        'descriptors': np.array(data['descriptors'])[keep_idx],
        'rose_params': np.array(data['rose_params'])[keep_idx],
        'rose_residuals': [data['rose_residuals'][i] for i in keep_idx],
        'curves': [data['curves'][i] for i in keep_idx],
        'r_grids': [data['r_grids'][i] for i in keep_idx],
    }
    return filtered


# =============================================================================
# SCALED COMMON GRID
# =============================================================================

def build_scaled_common_grid(data, n_grid=50):
    """Build a common grid in scaled coordinates r_s = r / r_cov_sum.

    Using covalent radii (from descriptors) to normalize distances makes
    molecules of different sizes comparable. The overlap in scaled coordinates
    is much wider than in raw Angstrom.

    Returns (r_scaled_common, curves_common, rcov_sums).
    """
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])

    # r_cov_sum is descriptor index 10
    rcov_sums = descriptors[:, 10]

    # Compute scaled r-grids and find overlap
    scaled_mins = []
    scaled_maxs = []
    for i in range(n):
        r_scaled = data['r_grids'][i] / rcov_sums[i]
        scaled_mins.append(r_scaled.min())
        scaled_maxs.append(r_scaled.max())

    r_sc_min = max(scaled_mins)
    r_sc_max = min(scaled_maxs)

    print(f"\n  Scaled grid (r / r_cov_sum):")
    print(f"    Overlap: {r_sc_min:.3f} to {r_sc_max:.3f} "
          f"(width {r_sc_max - r_sc_min:.3f})")

    if r_sc_max <= r_sc_min:
        print("    WARNING: No overlap in scaled coordinates!")
        # Fall back to reasonable range
        r_sc_min = np.median(scaled_mins)
        r_sc_max = np.median(scaled_maxs)
        print(f"    Using median range: {r_sc_min:.3f} to {r_sc_max:.3f}")

    r_scaled_common = np.linspace(r_sc_min, r_sc_max, n_grid)

    # Show what this means in real units for a few molecules
    for idx in [0, n//2, n-1]:
        name = data['names'][idx]
        r_real_min = r_sc_min * rcov_sums[idx]
        r_real_max = r_sc_max * rcov_sums[idx]
        print(f"    {name}: r = {r_real_min:.2f} to {r_real_max:.2f} A "
              f"(r_cov_sum = {rcov_sums[idx]:.2f})")

    # Interpolate all curves onto the common scaled grid
    curves_common = []
    for i in range(n):
        r_scaled = data['r_grids'][i] / rcov_sums[i]
        curve_interp = np.interp(r_scaled_common, r_scaled, data['curves'][i])
        curves_common.append(curve_interp)
    curves_common = np.array(curves_common)

    return r_scaled_common, curves_common, rcov_sums


# =============================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
# =============================================================================

def leave_one_out_comparison(data):
    """Run LOO CV comparing direct vs adaptive approaches."""
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])

    r_scaled_common, curves_common, rcov_sums = build_scaled_common_grid(data)

    results = {
        'names': data['names'],
        'mae_direct': [], 'mae_adaptive': [],
        'V_true': [], 'V_direct': [], 'V_adaptive': [],
        'params_true': [], 'params_pred': [],
        'r_scaled_common': r_scaled_common,
        'rcov_sums': rcov_sums,
    }

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        X_train = descriptors[train_idx]
        X_test = descriptors[[i]]

        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)

        # --- Direct approach: predict curve on scaled grid ---
        Y_train_curves = curves_common[train_idx]
        Y_test_curves = curves_common[[i]]

        scaler_V = StandardScaler()
        Y_train_s = scaler_V.fit_transform(Y_train_curves)
        model_direct = Ridge(alpha=1.0)
        model_direct.fit(X_train_s, Y_train_s)
        V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

        # --- Adaptive approach: predict Rose params, reconstruct curve ---
        Y_train_params = rose_params[train_idx]

        scaler_P = StandardScaler()
        Y_train_p_s = scaler_P.fit_transform(Y_train_params)
        model_adaptive = Ridge(alpha=1.0)
        model_adaptive.fit(X_train_s, Y_train_p_s)
        params_pred = scaler_P.inverse_transform(model_adaptive.predict(X_test_s))

        E_c, r_e, l = params_pred[0]
        E_c = max(E_c, 0.01)
        r_e = max(r_e, 0.3)
        l = max(l, 0.01)
        # Convert scaled grid back to real r for this molecule
        r_real = r_scaled_common * rcov_sums[i]
        V_pred_adaptive = rose_V(r_real, E_c, r_e, l).reshape(1, -1)

        # Metrics
        mae_d = np.mean(np.abs(Y_test_curves[0] - V_pred_direct[0]))
        mae_a = np.mean(np.abs(Y_test_curves[0] - V_pred_adaptive[0]))

        results['mae_direct'].append(mae_d)
        results['mae_adaptive'].append(mae_a)
        results['V_true'].append(Y_test_curves[0])
        results['V_direct'].append(V_pred_direct[0])
        results['V_adaptive'].append(V_pred_adaptive[0])
        results['params_true'].append(rose_params[i])
        results['params_pred'].append(params_pred[0])

        ratio = mae_d / max(mae_a, 1e-10)
        print(f"  {data['names'][i]:>6s}: Direct MAE={mae_d:.4f} eV, "
              f"Adaptive MAE={mae_a:.4f} eV, Ratio={ratio:.2f}x")

    return results


# =============================================================================
# LEAVE-GROUP-OUT (EXTRAPOLATION TEST)
# =============================================================================

def leave_group_out_comparison(data):
    """Leave out all molecules with atoms from row 3+ of the periodic table.

    Train on rows 1-2 (H-Ne), test on row 3+ (Na-).
    This forces genuine extrapolation in descriptor space.
    """
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])

    r_scaled_common, curves_common, rcov_sums = build_scaled_common_grid(data)

    # Split: molecules with Z > 10 in either atom go to test
    train_idx = []
    test_idx = []
    for i in range(n):
        Z1, Z2 = descriptors[i, 0], descriptors[i, 1]
        if Z1 > 10 or Z2 > 10:
            test_idx.append(i)
        else:
            train_idx.append(i)

    if len(train_idx) < 3 or len(test_idx) < 3:
        print("Not enough molecules for leave-group-out split")
        return None

    print(f"\n  Train ({len(train_idx)}): {[data['names'][i] for i in train_idx]}")
    print(f"  Test  ({len(test_idx)}): {[data['names'][i] for i in test_idx]}")

    X_train = descriptors[train_idx]
    X_test = descriptors[test_idx]

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # --- Direct ---
    Y_train_curves = curves_common[train_idx]
    Y_test_curves = curves_common[test_idx]

    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_train_curves)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    # --- Adaptive ---
    Y_train_params = rose_params[train_idx]

    scaler_P = StandardScaler()
    Y_train_p_s = scaler_P.fit_transform(Y_train_params)
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_train_p_s)
    params_pred = scaler_P.inverse_transform(model_adaptive.predict(X_test_s))

    V_pred_adaptive = []
    for j, i in enumerate(test_idx):
        E_c, r_e, l = params_pred[j]
        r_real = r_scaled_common * rcov_sums[i]
        V_pred_adaptive.append(rose_V(r_real, max(E_c, 0.01), max(r_e, 0.3), max(l, 0.01)))
    V_pred_adaptive = np.array(V_pred_adaptive)

    # Per-molecule metrics
    print(f"\n  {'Molecule':>8} {'Direct MAE':>12} {'Adaptive MAE':>14} {'Ratio':>8}")
    print("  " + "-" * 50)

    mae_d_list, mae_a_list = [], []
    for j, i in enumerate(test_idx):
        mae_d = np.mean(np.abs(Y_test_curves[j] - V_pred_direct[j]))
        mae_a = np.mean(np.abs(Y_test_curves[j] - V_pred_adaptive[j]))
        ratio = mae_d / max(mae_a, 1e-10)
        mae_d_list.append(mae_d)
        mae_a_list.append(mae_a)
        print(f"  {data['names'][i]:>8} {mae_d:>12.4f} {mae_a:>14.4f} {ratio:>8.2f}x")

    mean_d = np.mean(mae_d_list)
    mean_a = np.mean(mae_a_list)
    print(f"\n  Mean:    Direct MAE={mean_d:.4f}, "
          f"Adaptive MAE={mean_a:.4f}, "
          f"Ratio={mean_d/max(mean_a, 1e-10):.2f}x")

    return {
        'train_names': [data['names'][i] for i in train_idx],
        'test_names': [data['names'][i] for i in test_idx],
        'test_idx': test_idx,
        'V_true': Y_test_curves,
        'V_direct': V_pred_direct,
        'V_adaptive': V_pred_adaptive,
        'params_true': rose_params[test_idx],
        'params_pred': params_pred,
        'r_scaled_common': r_scaled_common,
        'rcov_sums': rcov_sums,
        'mae_direct': mae_d_list,
        'mae_adaptive': mae_a_list,
    }


# =============================================================================
# FIGURE
# =============================================================================

def create_figure(loo_results, lgo_results, data):
    """Create figure showing DFT results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    r_sc = loo_results['r_scaled_common']
    rcov_sums = loo_results['rcov_sums']

    # --- Panel A: LOO MAE comparison ---
    ax = axes[0, 0]
    n = len(loo_results['mae_direct'])
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, loo_results['mae_direct'], w, color='steelblue', alpha=0.85, label='Direct')
    ax.bar(x + w/2, loo_results['mae_adaptive'], w, color='indianred', alpha=0.85, label='Adaptive')
    ax.set_xticks(x)
    ax.set_xticklabels(loo_results['names'], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('MAE [eV]')
    ax.set_title('A. Leave-One-Out: Per-Molecule MAE')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel B: Example LOO prediction (best adaptive win) ---
    ax = axes[0, 1]
    ratios = [d/max(a, 1e-10) for d, a in
              zip(loo_results['mae_direct'], loo_results['mae_adaptive'])]
    best_idx = np.argmax(ratios)
    # Plot in real r for this molecule
    r_real = r_sc * rcov_sums[best_idx]
    ax.plot(r_real, loo_results['V_true'][best_idx], 'k-', linewidth=3, label='DFT')
    ax.plot(r_real, loo_results['V_direct'][best_idx], 'b--', linewidth=2, label='Direct')
    ax.plot(r_real, loo_results['V_adaptive'][best_idx], 'r:', linewidth=3, label='Adaptive')
    name = loo_results['names'][best_idx]
    ax.set_xlabel('r [A]')
    ax.set_ylabel('E(r) [eV]')
    ax.set_title(f'B. LOO Example: {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel C: Rose parameter parity ---
    ax = axes[0, 2]
    pt = np.array(loo_results['params_true'])
    pp = np.array(loo_results['params_pred'])
    ax.scatter(pt[:, 0], pp[:, 0], s=60, alpha=0.7, edgecolors='black',
               c='steelblue', label='$D_e$')
    lims = [0, max(pt[:, 0].max(), pp[:, 0].max()) * 1.1]
    ax.plot(lims, lims, 'k--', linewidth=1)
    r2_Ec = r2_score(pt[:, 0], pp[:, 0])
    ax.set_xlabel('True $D_e$ [eV]')
    ax.set_ylabel('Predicted $D_e$ [eV]')
    ax.set_title(f'C. Parameter Prediction (LOO)\n$D_e$: $R^2$ = {r2_Ec:.3f}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- Panel D: Leave-group-out MAE ---
    if lgo_results is not None:
        ax = axes[1, 0]
        n_test = len(lgo_results['mae_direct'])
        x = np.arange(n_test)
        ax.bar(x - w/2, lgo_results['mae_direct'], w, color='steelblue', alpha=0.85, label='Direct')
        ax.bar(x + w/2, lgo_results['mae_adaptive'], w, color='indianred', alpha=0.85, label='Adaptive')
        ax.set_xticks(x)
        ax.set_xticklabels(lgo_results['test_names'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('MAE [eV]')
        ax.set_title('D. Leave-Group-Out (Row 3+ test)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # --- Panel E: LGO example prediction ---
        ax = axes[1, 1]
        ratios_lgo = [d/max(a, 1e-10) for d, a in
                      zip(lgo_results['mae_direct'], lgo_results['mae_adaptive'])]
        best_lgo = np.argmax(ratios_lgo)
        test_i = lgo_results['test_idx'][best_lgo]
        r_real_lgo = r_sc * rcov_sums[test_i]
        ax.plot(r_real_lgo, lgo_results['V_true'][best_lgo], 'k-', linewidth=3, label='DFT')
        ax.plot(r_real_lgo, lgo_results['V_direct'][best_lgo], 'b--', linewidth=2, label='Direct')
        ax.plot(r_real_lgo, lgo_results['V_adaptive'][best_lgo], 'r:', linewidth=3, label='Adaptive')
        name = lgo_results['test_names'][best_lgo]
        ax.set_xlabel('r [A]')
        ax.set_ylabel('E(r) [eV]')
        ax.set_title(f'E. LGO Example: {name}\n(trained on rows 1-2 only)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Panel F: Summary ---
        ax = axes[1, 2]
        ax.axis('off')
        mean_d_loo = np.mean(loo_results['mae_direct'])
        mean_a_loo = np.mean(loo_results['mae_adaptive'])
        mean_d_lgo = np.mean(lgo_results['mae_direct'])
        mean_a_lgo = np.mean(lgo_results['mae_adaptive'])

        n_loo_wins = sum(1 for d, a in zip(loo_results['mae_direct'],
                                             loo_results['mae_adaptive']) if a < d)
        n_lgo_wins = sum(1 for d, a in zip(lgo_results['mae_direct'],
                                             lgo_results['mae_adaptive']) if a < d)

        summary = (
            f"DIATOMIC BINDING CURVES SUMMARY\n"
            f"(r_scaled = r / r_cov_sum common grid)\n\n"
            f"Leave-One-Out (interpolation):\n"
            f"  Direct MAE:   {mean_d_loo:.3f} eV\n"
            f"  Adaptive MAE: {mean_a_loo:.3f} eV\n"
            f"  Ratio: {mean_d_loo/max(mean_a_loo, 1e-10):.2f}x\n"
            f"  Adaptive wins: {n_loo_wins}/{len(loo_results['mae_direct'])}\n\n"
            f"Leave-Group-Out (extrapolation):\n"
            f"  Train: rows 1-2 ({len(lgo_results['train_names'])} mol)\n"
            f"  Test:  row 3+ ({len(lgo_results['test_names'])} mol)\n"
            f"  Direct MAE:   {mean_d_lgo:.3f} eV\n"
            f"  Adaptive MAE: {mean_a_lgo:.3f} eV\n"
            f"  Ratio: {mean_d_lgo/max(mean_a_lgo, 1e-10):.2f}x\n"
            f"  Adaptive wins: {n_lgo_wins}/{len(lgo_results['mae_direct'])}\n"
        )
        ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=10,
                va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        for ax in axes[1, :]:
            ax.axis('off')

    plt.tight_layout()
    outpath = 'figures/fig_diatomic_comparison.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Adaptive vs Direct on diatomic data')
    parser.add_argument('--data', default=None,
                        help='Path to diatomic_curves.npz (default: use demo data)')
    parser.add_argument('--max-residual', type=float, default=1.0,
                        help='Exclude molecules with Rose RMSE > this (eV, default: 1.0)')
    parser.add_argument('--min-points', type=int, default=8,
                        help='Exclude molecules with fewer converged points (default: 8)')
    args = parser.parse_args()

    print("=" * 70)
    print("ADAPTIVE vs DIRECT LEARNING ON DIATOMIC BINDING CURVES")
    print("=" * 70)

    if args.data and os.path.exists(args.data):
        print(f"Loading DFT data from: {args.data}")
        data = load_dft_data(args.data)
    else:
        print("No DFT data found. Running in DEMO MODE with synthetic Morse curves")
        print("(parameterized by realistic atomic properties).")
        print("To use real DFT data, first run: python diatomic_scan.py")
        data = generate_demo_data()

    print(f"\nLoaded {len(data['names'])} molecules")

    # Data quality assessment
    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 70)
    keep_idx = assess_data_quality(data, max_residual=args.max_residual,
                                    min_points=args.min_points)
    data = filter_data(data, keep_idx)

    print(f"\nUsing {len(data['names'])} molecules: {data['names']}")
    print(f"Descriptors shape: {np.array(data['descriptors']).shape}")

    # LOO
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 70)
    loo_results = leave_one_out_comparison(data)

    # LGO
    print("\n" + "=" * 70)
    print("LEAVE-GROUP-OUT (EXTRAPOLATION)")
    print("=" * 70)
    lgo_results = leave_group_out_comparison(data)

    # Figure
    create_figure(loo_results, lgo_results, data)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    mean_d = np.mean(loo_results['mae_direct'])
    mean_a = np.mean(loo_results['mae_adaptive'])
    print(f"\nLOO: Direct MAE = {mean_d:.4f} eV, Adaptive MAE = {mean_a:.4f} eV, "
          f"Ratio = {mean_d/max(mean_a, 1e-10):.2f}x")

    if lgo_results:
        mean_d_lgo = np.mean(lgo_results['mae_direct'])
        mean_a_lgo = np.mean(lgo_results['mae_adaptive'])
        print(f"LGO: Direct MAE = {mean_d_lgo:.4f} eV, Adaptive MAE = {mean_a_lgo:.4f} eV, "
              f"Ratio = {mean_d_lgo/max(mean_a_lgo, 1e-10):.2f}x")

    n_loo_wins = sum(1 for d, a in zip(loo_results['mae_direct'],
                                         loo_results['mae_adaptive']) if a < d)
    n_total = len(loo_results['mae_direct'])
    print(f"\nAdaptive wins on {n_loo_wins}/{n_total} molecules (LOO)")

    if lgo_results:
        n_lgo_wins = sum(1 for d, a in zip(lgo_results['mae_direct'],
                                             lgo_results['mae_adaptive']) if a < d)
        n_lgo_total = len(lgo_results['mae_direct'])
        print(f"Adaptive wins on {n_lgo_wins}/{n_lgo_total} molecules (LGO)")


if __name__ == "__main__":
    main()
