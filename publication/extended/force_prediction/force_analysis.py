"""
Extended Analysis: Force Prediction Comparison
===============================================

The adaptive approach provides analytic forces for free:
  F(r) = -dV/dr with predicted parameters plugged into the analytic derivative.

The direct approach must use numerical differentiation of the predicted V(r),
which introduces additional error and depends on grid spacing.

This script compares force prediction accuracy for both Rose/UBER and LJ systems.

Usage:
    MPLBACKEND=Agg python force_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


# =============================================================================
# POTENTIAL FUNCTIONS AND THEIR ANALYTIC DERIVATIVES
# =============================================================================

# --- Lennard-Jones ---

def lj_V(r, epsilon, sigma):
    """LJ potential."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def lj_F(r, epsilon, sigma):
    """LJ gradient dV/dr (negative at short r = repulsion)."""
    return 4 * epsilon * (-12 * sigma**12 / r**13 + 6 * sigma**6 / r**7)

# --- Morse ---

def morse_V(r, De, a, re):
    """Morse potential."""
    return De * (1 - np.exp(-a * (r - re)))**2

def morse_F(r, De, a, re):
    """Morse gradient dV/dr (negative at short r = repulsion)."""
    return 2 * De * a * np.exp(-a * (r - re)) * (1 - np.exp(-a * (r - re)))

# --- Rose/UBER ---

def rose_V(r, E_c, r_e, l):
    """Rose/UBER potential."""
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star))

def rose_F(r, E_c, r_e, l):
    """Rose/UBER gradient dV/dr (negative at short r = repulsion)."""
    a_star = (r - r_e) / l
    return E_c * a_star * np.exp(-a_star) / l


# =============================================================================
# NUMERICAL FORCE FROM PREDICTED CURVE
# =============================================================================

def numerical_force(r, V_pred):
    """Compute gradient dV/dr via central differences."""
    F = np.zeros_like(V_pred)
    # Central differences for interior points
    dr = r[1] - r[0]
    F[1:-1] = (V_pred[2:] - V_pred[:-2]) / (2 * dr)
    # Forward/backward at boundaries
    F[0] = (V_pred[1] - V_pred[0]) / dr
    F[-1] = (V_pred[-1] - V_pred[-2]) / dr
    return F


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_lj_dataset(n_samples, d1_range, d2_range, n_points=50):
    """Generate LJ potentials and forces on a fixed r-grid."""
    r = np.linspace(2.5, 12.0, n_points)
    data = {'X': [], 'params': [], 'curves': [], 'forces': [], 'r_grid': r}

    for i in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2
        V = lj_V(r, epsilon, sigma)
        F = lj_F(r, epsilon, sigma)
        data['X'].append([d1, d2])
        data['params'].append([epsilon, sigma])
        data['curves'].append(V)
        data['forces'].append(F)

    return data


def generate_rose_dataset(n_samples, d1_range, d2_range, n_points=50):
    """Generate Morse potentials with Rose fits, including forces."""
    data = {'X': [], 'morse_params': [], 'rose_params': [],
            'curves': [], 'forces': [], 'r_grids': []}

    for i in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        De = 0.5 + 2.0 * d1
        re = 1.5 + 0.5 * d2
        a = 1.5

        r = np.linspace(0.7 * re, 3.0 * re, n_points)
        V = morse_V(r, De, a, re)
        F = morse_F(r, De, a, re)

        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1/a],
                                bounds=([0.01, 0.5, 0.01], [20, 10, 5]), maxfev=5000)
            data['X'].append([d1, d2])
            data['morse_params'].append([De, re, a])
            data['rose_params'].append(popt)
            data['curves'].append(V)
            data['forces'].append(F)
            data['r_grids'].append(r)
        except Exception:
            pass

    return data


# =============================================================================
# TRAIN AND EVALUATE WITH FORCES
# =============================================================================

def train_lj_with_forces(train_data, test_data):
    """Train direct and adaptive for LJ, evaluate both energies and forces."""
    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])
    Y_params_train = np.array(train_data['params'])
    F_test_true = np.array(test_data['forces'])
    r = test_data['r_grid']

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # --- Direct ---
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    # Direct forces: numerical differentiation
    F_pred_direct = np.array([numerical_force(r, V_pred_direct[i])
                              for i in range(len(V_pred_direct))])

    # --- Adaptive ---
    scaler_params = StandardScaler()
    Y_params_s = scaler_params.fit_transform(Y_params_train)
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_params_s)
    Y_params_pred = scaler_params.inverse_transform(model_adaptive.predict(X_test_s))

    V_pred_adaptive = []
    F_pred_adaptive = []
    for eps, sig in Y_params_pred:
        eps_c = max(eps, 0.001)
        sig_c = max(sig, 0.1)
        V_pred_adaptive.append(lj_V(r, eps_c, sig_c))
        F_pred_adaptive.append(lj_F(r, eps_c, sig_c))
    V_pred_adaptive = np.array(V_pred_adaptive)
    F_pred_adaptive = np.array(F_pred_adaptive)

    # Metrics
    # Energy
    mae_V_direct = mean_absolute_error(Y_curves_test, V_pred_direct)
    mae_V_adaptive = mean_absolute_error(Y_curves_test, V_pred_adaptive)
    r2_V_direct = r2_score(Y_curves_test.flatten(), V_pred_direct.flatten())
    r2_V_adaptive = r2_score(Y_curves_test.flatten(), V_pred_adaptive.flatten())

    # Force (exclude boundary points where numerical diff is less accurate)
    sl = slice(2, -2)
    mae_F_direct = mean_absolute_error(F_test_true[:, sl], F_pred_direct[:, sl])
    mae_F_adaptive = mean_absolute_error(F_test_true[:, sl], F_pred_adaptive[:, sl])
    r2_F_direct = r2_score(F_test_true[:, sl].flatten(), F_pred_direct[:, sl].flatten())
    r2_F_adaptive = r2_score(F_test_true[:, sl].flatten(), F_pred_adaptive[:, sl].flatten())

    return {
        'mae_V_direct': mae_V_direct, 'mae_V_adaptive': mae_V_adaptive,
        'r2_V_direct': r2_V_direct, 'r2_V_adaptive': r2_V_adaptive,
        'mae_F_direct': mae_F_direct, 'mae_F_adaptive': mae_F_adaptive,
        'r2_F_direct': r2_F_direct, 'r2_F_adaptive': r2_F_adaptive,
        'V_pred_direct': V_pred_direct, 'V_pred_adaptive': V_pred_adaptive,
        'F_pred_direct': F_pred_direct, 'F_pred_adaptive': F_pred_adaptive,
        'F_test_true': F_test_true, 'Y_params_pred': Y_params_pred,
    }


def train_rose_with_forces(train_data, test_data):
    """Train direct and adaptive for Rose/UBER, evaluate both energies and forces."""
    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])
    Y_params_train = np.array(train_data['rose_params'])
    F_test_true = np.array(test_data['forces'])

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # --- Direct ---
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    V_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    # Direct forces: numerical differentiation (per-sample r-grid)
    F_pred_direct = []
    for i in range(len(V_pred_direct)):
        r = test_data['r_grids'][i]
        F_pred_direct.append(numerical_force(r, V_pred_direct[i]))
    F_pred_direct = np.array(F_pred_direct)

    # --- Adaptive ---
    scaler_params = StandardScaler()
    Y_params_s = scaler_params.fit_transform(Y_params_train)
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_params_s)
    Y_params_pred = scaler_params.inverse_transform(model_adaptive.predict(X_test_s))

    V_pred_adaptive = []
    F_pred_adaptive = []
    for i, (E_c, r_e, l) in enumerate(Y_params_pred):
        r = test_data['r_grids'][i]
        E_c_c = max(E_c, 0.01)
        r_e_c = max(r_e, 0.5)
        l_c = max(l, 0.01)
        V_pred_adaptive.append(rose_V(r, E_c_c, r_e_c, l_c))
        F_pred_adaptive.append(rose_F(r, E_c_c, r_e_c, l_c))
    V_pred_adaptive = np.array(V_pred_adaptive)
    F_pred_adaptive = np.array(F_pred_adaptive)

    # Metrics
    mae_V_direct = mean_absolute_error(Y_curves_test, V_pred_direct)
    mae_V_adaptive = mean_absolute_error(Y_curves_test, V_pred_adaptive)
    r2_V_direct = r2_score(Y_curves_test.flatten(), V_pred_direct.flatten())
    r2_V_adaptive = r2_score(Y_curves_test.flatten(), V_pred_adaptive.flatten())

    sl = slice(2, -2)
    mae_F_direct = mean_absolute_error(F_test_true[:, sl], F_pred_direct[:, sl])
    mae_F_adaptive = mean_absolute_error(F_test_true[:, sl], F_pred_adaptive[:, sl])
    r2_F_direct = r2_score(F_test_true[:, sl].flatten(), F_pred_direct[:, sl].flatten())
    r2_F_adaptive = r2_score(F_test_true[:, sl].flatten(), F_pred_adaptive[:, sl].flatten())

    return {
        'mae_V_direct': mae_V_direct, 'mae_V_adaptive': mae_V_adaptive,
        'r2_V_direct': r2_V_direct, 'r2_V_adaptive': r2_V_adaptive,
        'mae_F_direct': mae_F_direct, 'mae_F_adaptive': mae_F_adaptive,
        'r2_F_direct': r2_F_direct, 'r2_F_adaptive': r2_F_adaptive,
        'V_pred_direct': V_pred_direct, 'V_pred_adaptive': V_pred_adaptive,
        'F_pred_direct': F_pred_direct, 'F_pred_adaptive': F_pred_adaptive,
        'F_test_true': F_test_true, 'Y_params_pred': Y_params_pred,
    }


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiments():
    """Run force prediction experiments for both LJ and Rose/UBER."""
    print("=" * 70)
    print("FORCE PREDICTION ANALYSIS")
    print("Adaptive (analytic forces) vs Direct (numerical differentiation)")
    print("=" * 70)

    np.random.seed(42)

    # --- LJ ---
    print("\n" + "=" * 70)
    print("LENNARD-JONES")
    print("=" * 70)

    train_lj = generate_lj_dataset(100, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5))
    test_lj_extrap = generate_lj_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5))
    test_lj_interp = generate_lj_dataset(50, d1_range=(0.7, 1.3), d2_range=(0.6, 1.4))

    res_lj_extrap = train_lj_with_forces(train_lj, test_lj_extrap)
    res_lj_interp = train_lj_with_forces(train_lj, test_lj_interp)

    print("\n  EXTRAPOLATION:")
    print(f"    Energy -- Direct MAE: {res_lj_extrap['mae_V_direct']:.4f} eV, "
          f"Adaptive MAE: {res_lj_extrap['mae_V_adaptive']:.4f} eV")
    print(f"    Force  -- Direct MAE: {res_lj_extrap['mae_F_direct']:.4f} eV/A, "
          f"Adaptive MAE: {res_lj_extrap['mae_F_adaptive']:.4f} eV/A")
    print(f"    Force  -- Direct R2:  {res_lj_extrap['r2_F_direct']:.4f}, "
          f"Adaptive R2:  {res_lj_extrap['r2_F_adaptive']:.4f}")

    print("\n  INTERPOLATION:")
    print(f"    Energy -- Direct MAE: {res_lj_interp['mae_V_direct']:.6f} eV, "
          f"Adaptive MAE: {res_lj_interp['mae_V_adaptive']:.6f} eV")
    print(f"    Force  -- Direct MAE: {res_lj_interp['mae_F_direct']:.6f} eV/A, "
          f"Adaptive MAE: {res_lj_interp['mae_F_adaptive']:.6f} eV/A")
    print(f"    Force  -- Direct R2:  {res_lj_interp['r2_F_direct']:.4f}, "
          f"Adaptive R2:  {res_lj_interp['r2_F_adaptive']:.4f}")

    # --- Rose/UBER ---
    print("\n" + "=" * 70)
    print("ROSE/UBER (Morse potentials)")
    print("=" * 70)

    train_rose = generate_rose_dataset(100, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5))
    test_rose_extrap = generate_rose_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5))
    test_rose_interp = generate_rose_dataset(50, d1_range=(0.7, 1.3), d2_range=(0.6, 1.4))

    res_rose_extrap = train_rose_with_forces(train_rose, test_rose_extrap)
    res_rose_interp = train_rose_with_forces(train_rose, test_rose_interp)

    print("\n  EXTRAPOLATION:")
    print(f"    Energy -- Direct MAE: {res_rose_extrap['mae_V_direct']:.4f} eV, "
          f"Adaptive MAE: {res_rose_extrap['mae_V_adaptive']:.4f} eV")
    print(f"    Force  -- Direct MAE: {res_rose_extrap['mae_F_direct']:.4f} eV/A, "
          f"Adaptive MAE: {res_rose_extrap['mae_F_adaptive']:.4f} eV/A")
    print(f"    Force  -- Direct R2:  {res_rose_extrap['r2_F_direct']:.4f}, "
          f"Adaptive R2:  {res_rose_extrap['r2_F_adaptive']:.4f}")

    print("\n  INTERPOLATION:")
    print(f"    Energy -- Direct MAE: {res_rose_interp['mae_V_direct']:.4f} eV, "
          f"Adaptive MAE: {res_rose_interp['mae_V_adaptive']:.4f} eV")
    print(f"    Force  -- Direct MAE: {res_rose_interp['mae_F_direct']:.4f} eV/A, "
          f"Adaptive MAE: {res_rose_interp['mae_F_adaptive']:.4f} eV/A")
    print(f"    Force  -- Direct R2:  {res_rose_interp['r2_F_direct']:.4f}, "
          f"Adaptive R2:  {res_rose_interp['r2_F_adaptive']:.4f}")

    print("\n  NOTE: Rose/UBER force R2 may be poor because true forces are Morse")
    print("  derivatives while adaptive predicts Rose derivatives. The Rose/Morse")
    print("  approximation error dominates. The clean demonstration is LJ above.")

    return {
        'lj_extrap': res_lj_extrap, 'lj_interp': res_lj_interp,
        'rose_extrap': res_rose_extrap, 'rose_interp': res_rose_interp,
        'test_lj_extrap': test_lj_extrap, 'test_rose_extrap': test_rose_extrap,
    }


# =============================================================================
# FIGURE
# =============================================================================

def create_figure(results):
    """Create 6-panel figure: energy + force comparison for both systems."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- Row 1: Lennard-Jones ---
    r_lj = results['test_lj_extrap']['r_grid']
    idx = 0

    # Panel A: LJ energy predictions
    ax = axes[0, 0]
    V_true = results['test_lj_extrap']['curves'][idx]
    V_d = results['lj_extrap']['V_pred_direct'][idx]
    V_a = results['lj_extrap']['V_pred_adaptive'][idx]
    ax.plot(r_lj, V_true, 'k-', linewidth=3, label='True')
    ax.plot(r_lj, V_d, 'b--', linewidth=2, label='Direct')
    ax.plot(r_lj, V_a, 'r:', linewidth=3, label='Adaptive')
    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('A. LJ Energy (Extrapolation)')
    ax.legend()
    ax.set_ylim(-0.3, 0.5)
    ax.grid(True, alpha=0.3)

    # Panel B: LJ force predictions
    ax = axes[0, 1]
    F_true = results['test_lj_extrap']['forces'][idx]
    F_d = results['lj_extrap']['F_pred_direct'][idx]
    F_a = results['lj_extrap']['F_pred_adaptive'][idx]
    ax.plot(r_lj, F_true, 'k-', linewidth=3, label='True')
    ax.plot(r_lj, F_d, 'b--', linewidth=2, label='Direct (numerical)')
    ax.plot(r_lj, F_a, 'r:', linewidth=3, label='Adaptive (analytic)')
    ax.set_xlabel('r [A]')
    ax.set_ylabel('dV/dr [eV/A]')
    ax.set_title('B. LJ Force (Extrapolation)')
    ax.legend()
    ax.set_ylim(-0.1, 0.1)
    ax.grid(True, alpha=0.3)

    # Panel C: LJ force parity plot
    ax = axes[0, 2]
    sl = slice(2, -2)
    F_true_flat = np.array(results['test_lj_extrap']['forces'])[:, sl].flatten()
    F_d_flat = results['lj_extrap']['F_pred_direct'][:, sl].flatten()
    F_a_flat = results['lj_extrap']['F_pred_adaptive'][:, sl].flatten()

    ax.scatter(F_true_flat, F_d_flat, alpha=0.2, s=10, c='blue', label='Direct')
    ax.scatter(F_true_flat, F_a_flat, alpha=0.2, s=10, c='red', label='Adaptive')
    lims = [min(F_true_flat.min(), F_d_flat.min(), F_a_flat.min()),
            max(F_true_flat.max(), F_d_flat.max(), F_a_flat.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel('True Force [eV/A]')
    ax.set_ylabel('Predicted Force [eV/A]')
    ax.set_title(f'C. LJ Force Parity\n'
                 f'Direct R2={results["lj_extrap"]["r2_F_direct"]:.3f}, '
                 f'Adaptive R2={results["lj_extrap"]["r2_F_adaptive"]:.3f}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- Row 2: Rose/UBER ---
    idx = 0
    r_rose = results['test_rose_extrap']['r_grids'][idx]

    # Panel D: Rose energy
    ax = axes[1, 0]
    V_true = results['test_rose_extrap']['curves'][idx]
    V_d = results['rose_extrap']['V_pred_direct'][idx]
    V_a = results['rose_extrap']['V_pred_adaptive'][idx]
    ax.plot(r_rose, V_true, 'k-', linewidth=3, label='True')
    ax.plot(r_rose, V_d, 'b--', linewidth=2, label='Direct')
    ax.plot(r_rose, V_a, 'r:', linewidth=3, label='Adaptive')
    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('D. Rose/UBER Energy (Extrapolation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel E: Rose force
    ax = axes[1, 1]
    F_true = results['test_rose_extrap']['forces'][idx]
    F_d = results['rose_extrap']['F_pred_direct'][idx]
    F_a = results['rose_extrap']['F_pred_adaptive'][idx]
    ax.plot(r_rose, F_true, 'k-', linewidth=3, label='True')
    ax.plot(r_rose, F_d, 'b--', linewidth=2, label='Direct (numerical)')
    ax.plot(r_rose, F_a, 'r:', linewidth=3, label='Adaptive (analytic)')
    ax.set_xlabel('r [A]')
    ax.set_ylabel('dV/dr [eV/A]')
    ax.set_title('E. Rose/UBER Force (Extrapolation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel F: Rose force parity
    ax = axes[1, 2]
    F_true_flat = np.array(results['test_rose_extrap']['forces'])[:, sl].flatten()
    F_d_flat = results['rose_extrap']['F_pred_direct'][:, sl].flatten()
    F_a_flat = results['rose_extrap']['F_pred_adaptive'][:, sl].flatten()

    ax.scatter(F_true_flat, F_d_flat, alpha=0.2, s=10, c='blue', label='Direct')
    ax.scatter(F_true_flat, F_a_flat, alpha=0.2, s=10, c='red', label='Adaptive')
    lims = [min(F_true_flat.min(), F_d_flat.min(), F_a_flat.min()),
            max(F_true_flat.max(), F_d_flat.max(), F_a_flat.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel('True Force [eV/A]')
    ax.set_ylabel('Predicted Force [eV/A]')
    ax.set_title(f'F. Rose/UBER Force Parity\n'
                 f'Direct R2={results["rose_extrap"]["r2_F_direct"]:.3f}, '
                 f'Adaptive R2={results["rose_extrap"]["r2_F_adaptive"]:.3f}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'figures/fig_force_analysis.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


# =============================================================================
# GRID SPACING SENSITIVITY
# =============================================================================

def grid_spacing_sensitivity():
    """Show how numerical force accuracy depends on grid spacing."""
    print("\n" + "=" * 70)
    print("GRID SPACING SENSITIVITY (LJ)")
    print("=" * 70)

    np.random.seed(42)

    n_points_list = [10, 20, 30, 50, 100, 200]
    results = {'n_points': [], 'direct_F_mae': [], 'adaptive_F_mae': []}

    for n_pts in n_points_list:
        train_data = generate_lj_dataset(100, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5),
                                         n_points=n_pts)
        test_data = generate_lj_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5),
                                        n_points=n_pts)
        res = train_lj_with_forces(train_data, test_data)

        results['n_points'].append(n_pts)
        results['direct_F_mae'].append(res['mae_F_direct'])
        results['adaptive_F_mae'].append(res['mae_F_adaptive'])

        dr = (12.0 - 2.5) / (n_pts - 1)
        print(f"  N_points={n_pts:4d} (dr={dr:.3f} A): "
              f"Direct F MAE={res['mae_F_direct']:.6f}, "
              f"Adaptive F MAE={res['mae_F_adaptive']:.6f} eV/A")

    return results


def create_grid_figure(grid_results):
    """Plot force MAE vs grid spacing."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n_pts = grid_results['n_points']
    dr = [(12.0 - 2.5) / (n - 1) for n in n_pts]

    ax.loglog(dr, grid_results['direct_F_mae'], 'bo-', linewidth=2, markersize=8,
              label='Direct (numerical diff)')
    ax.loglog(dr, grid_results['adaptive_F_mae'], 'rs-', linewidth=2, markersize=8,
              label='Adaptive (analytic)')
    ax.set_xlabel('Grid Spacing dr [A]')
    ax.set_ylabel('Force MAE [eV/A]')
    ax.set_title('Force Accuracy vs Grid Spacing (LJ, Extrapolation)\n'
                 'Adaptive forces are analytic and grid-independent')
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = 'figures/fig_force_grid_sensitivity.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    results = run_experiments()
    create_figure(results)

    grid_results = grid_spacing_sensitivity()
    create_grid_figure(grid_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: FORCE PREDICTION")
    print("=" * 70)
    print(f"""
Adaptive approach provides ANALYTIC forces for free via:
  F(r) = -dV/dr with predicted parameters

Direct approach requires NUMERICAL differentiation:
  F(r) ~ -[V(r+dr) - V(r-dr)] / (2*dr)

Advantages of adaptive forces:
1. No discretization error (exact derivative of physics equation)
2. Grid-independent accuracy
3. Correct asymptotic force behavior (repulsive at short r, vanishing at long r)
4. Smooth forces even with noisy energy predictions

This is a "free" bonus of the adaptive approach -- by learning the parameters
of an analytic function, you get all derivatives for free.
""")


if __name__ == "__main__":
    main()
