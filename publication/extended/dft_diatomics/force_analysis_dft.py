"""
Force Prediction on DFT Diatomic Binding Curves
=================================================

Compares force prediction quality for direct vs adaptive approaches
on real DFT diatomic data, using the same LOO and LGO protocols.

"True" DFT forces: numerical central differences on DFT binding curves.
Adaptive forces: analytic Rose derivative with predicted (D_e, r_e, l).
Direct forces: numerical differentiation of the predicted V(r) curve.

Caveat: adaptive forces use Rose F = -dV_Rose/dr, which differs from
the true DFT force by the Rose approximation error. This is the same
situation as the synthetic Morse/Rose comparison.

Usage:
    MPLBACKEND=Agg python force_analysis_dft.py
    MPLBACKEND=Agg python force_analysis_dft.py --data data/diatomic_curves.npz
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
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
# PHYSICS
# =============================================================================

def rose_V(r, D_e, r_e, l):
    a_star = (r - r_e) / l
    return -D_e * (1 + a_star) * np.exp(-a_star)

def rose_F(r, D_e, r_e, l):
    """Rose gradient dV_Rose/dr (negative at short r = repulsion).

    For V = -D_e * (1 + a*) * exp(-a*):
    dV/dr = D_e * a* * exp(-a*) / l
    """
    a_star = (r - r_e) / l
    return D_e * a_star * np.exp(-a_star) / l

def morse_V(r, De, a, re):
    return De * ((1 - np.exp(-a * (r - re)))**2 - 1)

def numerical_force(r, E):
    """Compute gradient dE/dr via central differences."""
    F = np.zeros_like(E)
    dr = np.diff(r)
    # Central differences for interior (variable spacing)
    for i in range(1, len(r) - 1):
        dr_fwd = r[i+1] - r[i]
        dr_bwd = r[i] - r[i-1]
        F[i] = (E[i+1] - E[i-1]) / (dr_fwd + dr_bwd)
    # Forward/backward at boundaries
    F[0] = (E[1] - E[0]) / (r[1] - r[0])
    F[-1] = (E[-1] - E[-2]) / (r[-1] - r[-2])
    return F


# =============================================================================
# DEMO DATA (if no DFT data available)
# =============================================================================

DEMO_DIATOMICS = [
    ('H2',   4.75, 0.74, 4401, 1,  1, 2.20, 2.20, 0.31, 0.31, 13.60, 13.60),
    ('LiH',  2.52, 1.60, 1405, 3,  1, 0.98, 2.20, 1.28, 0.31,  5.39, 13.60),
    ('BH',   3.54, 1.23, 2367, 5,  1, 2.04, 2.20, 0.84, 0.31,  8.30, 13.60),
    ('CH',   3.65, 1.12, 2861, 6,  1, 2.55, 2.20, 0.76, 0.31, 11.26, 13.60),
    ('NH',   3.47, 1.04, 3282, 7,  1, 3.04, 2.20, 0.71, 0.31, 14.53, 13.60),
    ('OH',   4.62, 0.97, 3738, 8,  1, 3.44, 2.20, 0.66, 0.31, 13.62, 13.60),
    ('HF',   5.87, 0.92, 4138, 9,  1, 3.98, 2.20, 0.57, 0.31, 17.42, 13.60),
    ('NaH',  1.97, 1.89, 1172, 11, 1, 0.93, 2.20, 1.66, 0.31,  5.14, 13.60),
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
    ('LiF',  5.91, 1.56,  910, 3,  9, 0.98, 3.98, 1.28, 0.57,  5.39, 17.42),
    ('NaF',  4.97, 1.93,  536, 11, 9, 0.93, 3.98, 1.66, 0.57,  5.14, 17.42),
    ('NaCl', 4.24, 2.36,  366, 11, 17, 0.93, 3.16, 1.66, 1.02,  5.14, 12.97),
    ('CO',  11.11, 1.13, 2170, 6,  8, 2.55, 3.44, 0.76, 0.66, 11.26, 13.62),
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
    """Generate synthetic Morse curves with realistic parameters."""
    n_r = 25
    data = {
        'names': [], 'descriptors': [], 'rose_params': [],
        'rose_residuals': [], 'curves': [], 'r_grids': [],
    }

    for entry in DEMO_DIATOMICS:
        name = entry[0]
        De, r_e, omega = entry[1], entry[2], entry[3]
        Z1, Z2, EN1, EN2, rcov1, rcov2, IE1, IE2 = entry[4:]

        a = omega / (1000.0 * np.sqrt(max(De, 0.1)))
        a = np.clip(a, 0.5, 5.0)

        r = np.linspace(max(0.5, 0.7 * r_e), 5.0 * r_e, n_r)
        V = morse_V(r, De, a, r_e)

        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, r_e, 1.0/a],
                                bounds=([0.01, 0.3, 0.01], [20, 10, 5]),
                                maxfev=10000)
        except Exception:
            continue

        residual = np.sqrt(np.mean((V - rose_V(r, *popt))**2))

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


def filter_data(data, max_residual=1.0, min_points=8):
    """Exclude molecules with poor Rose fits or too few points."""
    keep_idx = []
    for i in range(len(data['names'])):
        if data['rose_residuals'][i] <= max_residual and len(data['r_grids'][i]) >= min_points:
            keep_idx.append(i)

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
# SCALED COMMON GRID + FORCES
# =============================================================================

def build_scaled_grid_with_forces(data, n_grid=50):
    """Build common scaled grid and compute forces via numerical diff."""
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rcov_sums = descriptors[:, 10]  # r_cov_sum is descriptor index 10

    # Find overlap in scaled coordinates
    scaled_mins, scaled_maxs = [], []
    for i in range(n):
        r_sc = data['r_grids'][i] / rcov_sums[i]
        scaled_mins.append(r_sc.min())
        scaled_maxs.append(r_sc.max())

    r_sc_min = max(scaled_mins)
    r_sc_max = min(scaled_maxs)

    if r_sc_max <= r_sc_min:
        r_sc_min = np.median(scaled_mins)
        r_sc_max = np.median(scaled_maxs)

    r_scaled = np.linspace(r_sc_min, r_sc_max, n_grid)

    # Interpolate curves and compute numerical forces on common grid
    curves_common = []
    forces_common = []
    for i in range(n):
        r_sc_i = data['r_grids'][i] / rcov_sums[i]
        curve_interp = np.interp(r_scaled, r_sc_i, data['curves'][i])
        curves_common.append(curve_interp)
        # Forces on the common scaled grid (in real units)
        r_real = r_scaled * rcov_sums[i]
        F = numerical_force(r_real, curve_interp)
        forces_common.append(F)

    return r_scaled, np.array(curves_common), np.array(forces_common), rcov_sums


# =============================================================================
# LOO WITH FORCES
# =============================================================================

def loo_with_forces(data):
    """Leave-one-out CV comparing force prediction quality."""
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])

    r_scaled, curves, forces_true, rcov_sums = build_scaled_grid_with_forces(data)

    results = {
        'names': data['names'], 'r_scaled': r_scaled, 'rcov_sums': rcov_sums,
        'mae_F_direct': [], 'mae_F_adaptive': [],
        'F_true': [], 'F_direct': [], 'F_adaptive': [],
        'mae_V_direct': [], 'mae_V_adaptive': [],
    }

    # Exclude boundary points for force metrics (numerical diff less accurate)
    sl = slice(2, -2)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        X_train = descriptors[train_idx]
        X_test = descriptors[[i]]
        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)

        # --- Direct ---
        scaler_V = StandardScaler()
        model_direct = Ridge(alpha=1.0)
        model_direct.fit(X_train_s, scaler_V.fit_transform(curves[train_idx]))
        V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))[0]
        r_real = r_scaled * rcov_sums[i]
        F_pred_direct = numerical_force(r_real, V_pred_direct)

        # --- Adaptive ---
        scaler_P = StandardScaler()
        model_adaptive = Ridge(alpha=1.0)
        model_adaptive.fit(X_train_s, scaler_P.fit_transform(rose_params[train_idx]))
        params_pred = scaler_P.inverse_transform(model_adaptive.predict(X_test_s))[0]
        D_e, r_e, l = max(params_pred[0], 0.01), max(params_pred[1], 0.3), max(params_pred[2], 0.01)
        V_pred_adaptive = rose_V(r_real, D_e, r_e, l)
        F_pred_adaptive = rose_F(r_real, D_e, r_e, l)

        # Force metrics (excluding boundaries)
        F_true_i = forces_true[i]
        mae_F_d = mean_absolute_error(F_true_i[sl], F_pred_direct[sl])
        mae_F_a = mean_absolute_error(F_true_i[sl], F_pred_adaptive[sl])

        # Energy metrics
        mae_V_d = mean_absolute_error(curves[i], V_pred_direct)
        mae_V_a = mean_absolute_error(curves[i], V_pred_adaptive)

        results['mae_F_direct'].append(mae_F_d)
        results['mae_F_adaptive'].append(mae_F_a)
        results['mae_V_direct'].append(mae_V_d)
        results['mae_V_adaptive'].append(mae_V_a)
        results['F_true'].append(F_true_i)
        results['F_direct'].append(F_pred_direct)
        results['F_adaptive'].append(F_pred_adaptive)

    return results


# =============================================================================
# LGO WITH FORCES
# =============================================================================

def lgo_with_forces(data):
    """Leave-group-out (row 3+ test) with force comparison."""
    n = len(data['names'])
    descriptors = np.array(data['descriptors'])
    rose_params = np.array(data['rose_params'])

    r_scaled, curves, forces_true, rcov_sums = build_scaled_grid_with_forces(data)

    # Split by periodic table row
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

    X_train = descriptors[train_idx]
    X_test = descriptors[test_idx]
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Direct
    scaler_V = StandardScaler()
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, scaler_V.fit_transform(curves[train_idx]))
    V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    # Adaptive
    scaler_P = StandardScaler()
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, scaler_P.fit_transform(rose_params[train_idx]))
    params_pred = scaler_P.inverse_transform(model_adaptive.predict(X_test_s))

    sl = slice(2, -2)
    results = {
        'test_names': [data['names'][i] for i in test_idx],
        'test_idx': test_idx, 'r_scaled': r_scaled, 'rcov_sums': rcov_sums,
        'mae_F_direct': [], 'mae_F_adaptive': [],
        'F_true': [], 'F_direct': [], 'F_adaptive': [],
    }

    for j, i in enumerate(test_idx):
        r_real = r_scaled * rcov_sums[i]

        # Direct forces
        F_pred_direct = numerical_force(r_real, V_pred_direct[j])

        # Adaptive forces
        D_e, r_e, l = params_pred[j]
        D_e, r_e, l = max(D_e, 0.01), max(r_e, 0.3), max(l, 0.01)
        F_pred_adaptive = rose_F(r_real, D_e, r_e, l)

        F_true_i = forces_true[i]
        mae_F_d = mean_absolute_error(F_true_i[sl], F_pred_direct[sl])
        mae_F_a = mean_absolute_error(F_true_i[sl], F_pred_adaptive[sl])

        results['mae_F_direct'].append(mae_F_d)
        results['mae_F_adaptive'].append(mae_F_a)
        results['F_true'].append(F_true_i)
        results['F_direct'].append(F_pred_direct)
        results['F_adaptive'].append(F_pred_adaptive)

    return results


# =============================================================================
# FIGURE
# =============================================================================

def create_figure(loo, lgo, data):
    """Create 2x3 figure: LOO forces (top), LGO forces (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    r_sc = loo['r_scaled']
    rcov = loo['rcov_sums']
    sl = slice(2, -2)

    # ===== Row 1: LOO =====

    # (a) Per-molecule force MAE
    ax = axes[0, 0]
    n = len(loo['mae_F_direct'])
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, loo['mae_F_direct'], w, color='#888888', alpha=0.85, label='Direct')
    ax.bar(x + w/2, loo['mae_F_adaptive'], w, color='#4682B4', alpha=0.85, label='Adaptive')
    ax.set_xticks(x)
    ax.set_xticklabels(loo['names'], rotation=60, ha='right', fontsize=6)
    ax.set_ylabel('Force MAE [eV/\u00c5]')
    ax.set_title('(a) LOO: Per-Molecule Force MAE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # (b) Example LOO force prediction (best adaptive win)
    ax = axes[0, 1]
    ratios = [d / max(a, 1e-10) for d, a in
              zip(loo['mae_F_direct'], loo['mae_F_adaptive'])]
    best = np.argmax(ratios)
    r_real = r_sc * rcov[best]
    ax.plot(r_real, loo['F_true'][best], 'k-', lw=2.5, label='DFT (num.)')
    ax.plot(r_real, loo['F_direct'][best], '--', color='#888888', lw=1.8, label='Direct (num.)')
    ax.plot(r_real, loo['F_adaptive'][best], ':', color='#4682B4', lw=2.2, label='Adaptive (analytic)')
    ax.set_xlabel(r'$r$ [\u00c5]')
    ax.set_ylabel(r'$F(r)$ [eV/\u00c5]')
    ax.set_title(f'(b) LOO Force: {loo["names"][best]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (c) LOO force parity plot
    ax = axes[0, 2]
    F_true_all = np.array(loo['F_true'])[:, sl].flatten()
    F_d_all = np.array(loo['F_direct'])[:, sl].flatten()
    F_a_all = np.array(loo['F_adaptive'])[:, sl].flatten()

    ax.scatter(F_true_all, F_d_all, alpha=0.15, s=8, c='#888888', label='Direct')
    ax.scatter(F_true_all, F_a_all, alpha=0.15, s=8, c='#4682B4', label='Adaptive')
    lims = [min(F_true_all.min(), F_d_all.min(), F_a_all.min()),
            max(F_true_all.max(), F_d_all.max(), F_a_all.max())]
    ax.plot(lims, lims, 'k--', lw=0.8)

    r2_d = r2_score(F_true_all, F_d_all)
    r2_a = r2_score(F_true_all, F_a_all)
    ax.set_xlabel(r'True Force [eV/\u00c5]')
    ax.set_ylabel(r'Predicted Force [eV/\u00c5]')
    ax.set_title(f'(c) LOO Force Parity\nDirect $R^2$={r2_d:.3f}, Adaptive $R^2$={r2_a:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ===== Row 2: LGO =====
    if lgo is not None:
        r_sc_lgo = lgo['r_scaled']
        rcov_lgo = lgo['rcov_sums']

        # (d) Per-molecule force MAE
        ax = axes[1, 0]
        n_test = len(lgo['mae_F_direct'])
        x = np.arange(n_test)
        ax.bar(x - w/2, lgo['mae_F_direct'], w, color='#888888', alpha=0.85, label='Direct')
        ax.bar(x + w/2, lgo['mae_F_adaptive'], w, color='#4682B4', alpha=0.85, label='Adaptive')
        ax.set_xticks(x)
        ax.set_xticklabels(lgo['test_names'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Force MAE [eV/\u00c5]')
        ax.set_title('(d) LGO: Per-Molecule Force MAE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

        # (e) Example LGO force prediction
        ax = axes[1, 1]
        ratios_lgo = [d / max(a, 1e-10) for d, a in
                      zip(lgo['mae_F_direct'], lgo['mae_F_adaptive'])]
        best_lgo = np.argmax(ratios_lgo)
        test_i = lgo['test_idx'][best_lgo]
        r_real_lgo = r_sc_lgo * rcov_lgo[test_i]
        ax.plot(r_real_lgo, lgo['F_true'][best_lgo], 'k-', lw=2.5, label='DFT (num.)')
        ax.plot(r_real_lgo, lgo['F_direct'][best_lgo], '--', color='#888888', lw=1.8, label='Direct (num.)')
        ax.plot(r_real_lgo, lgo['F_adaptive'][best_lgo], ':', color='#4682B4', lw=2.2, label='Adaptive (analytic)')
        ax.set_xlabel(r'$r$ [\u00c5]')
        ax.set_ylabel(r'$F(r)$ [eV/\u00c5]')
        ax.set_title(f'(e) LGO Force: {lgo["test_names"][best_lgo]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # (f) LGO force parity
        ax = axes[1, 2]
        F_true_lgo = np.array(lgo['F_true'])[:, sl].flatten()
        F_d_lgo = np.array(lgo['F_direct'])[:, sl].flatten()
        F_a_lgo = np.array(lgo['F_adaptive'])[:, sl].flatten()

        ax.scatter(F_true_lgo, F_d_lgo, alpha=0.2, s=10, c='#888888', label='Direct')
        ax.scatter(F_true_lgo, F_a_lgo, alpha=0.2, s=10, c='#4682B4', label='Adaptive')
        lims = [min(F_true_lgo.min(), F_d_lgo.min(), F_a_lgo.min()),
                max(F_true_lgo.max(), F_d_lgo.max(), F_a_lgo.max())]
        ax.plot(lims, lims, 'k--', lw=0.8)

        r2_d_lgo = r2_score(F_true_lgo, F_d_lgo)
        r2_a_lgo = r2_score(F_true_lgo, F_a_lgo)
        ax.set_xlabel(r'True Force [eV/\u00c5]')
        ax.set_ylabel(r'Predicted Force [eV/\u00c5]')
        ax.set_title(f'(f) LGO Force Parity\nDirect $R^2$={r2_d_lgo:.3f}, Adaptive $R^2$={r2_a_lgo:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    else:
        for ax in axes[1, :]:
            ax.axis('off')

    plt.tight_layout(h_pad=1.5)
    outpath = 'figures/fig_dft_force_analysis.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("FORCE PREDICTION ON DFT DIATOMIC BINDING CURVES")
    print("=" * 70)

    # Load data
    dft_path = args.data
    if dft_path is None:
        dft_path = 'data/diatomic_curves.npz'

    if os.path.exists(dft_path):
        print(f"Loading DFT data from: {dft_path}")
        data = load_dft_data(dft_path)
    else:
        print("No DFT data found. Using DEMO MODE with synthetic Morse curves.")
        data = generate_demo_data()

    print(f"Loaded {len(data['names'])} molecules")

    # Filter
    data = filter_data(data, max_residual=1.0, min_points=8)
    print(f"After filtering: {len(data['names'])} molecules")

    # LOO
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT: FORCE PREDICTION")
    print("=" * 70)
    loo = loo_with_forces(data)

    sl = slice(2, -2)
    mean_F_d = np.mean(loo['mae_F_direct'])
    mean_F_a = np.mean(loo['mae_F_adaptive'])
    n_F_wins = sum(1 for d, a in zip(loo['mae_F_direct'], loo['mae_F_adaptive']) if a < d)

    print(f"\n  LOO Force Summary:")
    print(f"    Direct  mean MAE: {mean_F_d:.4f} eV/A")
    print(f"    Adaptive mean MAE: {mean_F_a:.4f} eV/A")
    print(f"    Ratio: {mean_F_d / max(mean_F_a, 1e-10):.2f}x")
    print(f"    Adaptive wins: {n_F_wins}/{len(loo['mae_F_direct'])}")

    # Force parity R2
    F_true_all = np.array(loo['F_true'])[:, sl].flatten()
    F_d_all = np.array(loo['F_direct'])[:, sl].flatten()
    F_a_all = np.array(loo['F_adaptive'])[:, sl].flatten()
    print(f"    Direct  force R2: {r2_score(F_true_all, F_d_all):.4f}")
    print(f"    Adaptive force R2: {r2_score(F_true_all, F_a_all):.4f}")

    # LGO
    print("\n" + "=" * 70)
    print("LEAVE-GROUP-OUT: FORCE PREDICTION")
    print("=" * 70)
    lgo = lgo_with_forces(data)

    if lgo is not None:
        mean_F_d_lgo = np.mean(lgo['mae_F_direct'])
        mean_F_a_lgo = np.mean(lgo['mae_F_adaptive'])
        n_F_wins_lgo = sum(1 for d, a in zip(lgo['mae_F_direct'], lgo['mae_F_adaptive']) if a < d)

        print(f"\n  Per-molecule force MAE:")
        for j, name in enumerate(lgo['test_names']):
            ratio = lgo['mae_F_direct'][j] / max(lgo['mae_F_adaptive'][j], 1e-10)
            print(f"    {name:>6}: Direct={lgo['mae_F_direct'][j]:.4f}, "
                  f"Adaptive={lgo['mae_F_adaptive'][j]:.4f}, Ratio={ratio:.2f}x")

        print(f"\n  LGO Force Summary:")
        print(f"    Direct  mean MAE: {mean_F_d_lgo:.4f} eV/A")
        print(f"    Adaptive mean MAE: {mean_F_a_lgo:.4f} eV/A")
        print(f"    Ratio: {mean_F_d_lgo / max(mean_F_a_lgo, 1e-10):.2f}x")
        print(f"    Adaptive wins: {n_F_wins_lgo}/{len(lgo['mae_F_direct'])}")

        F_true_lgo = np.array(lgo['F_true'])[:, sl].flatten()
        F_d_lgo = np.array(lgo['F_direct'])[:, sl].flatten()
        F_a_lgo = np.array(lgo['F_adaptive'])[:, sl].flatten()
        print(f"    Direct  force R2: {r2_score(F_true_lgo, F_d_lgo):.4f}")
        print(f"    Adaptive force R2: {r2_score(F_true_lgo, F_a_lgo):.4f}")

    # Figure
    create_figure(loo, lgo, data)

    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
'True' forces: numerical differentiation of DFT binding curves.
Adaptive forces: analytic Rose derivative F = D_e * a* * exp(-a*) / l.
Direct forces: numerical differentiation of Ridge-predicted curves.

Caveat: Adaptive forces use the Rose equation, which is an approximation
to the true DFT potential. The Rose/DFT mismatch (same as Rose/Morse)
means adaptive forces have a systematic error floor. The comparison
measures whether the adaptive approach gives smoother, more physical
force predictions despite this approximation.
""")


if __name__ == "__main__":
    main()
