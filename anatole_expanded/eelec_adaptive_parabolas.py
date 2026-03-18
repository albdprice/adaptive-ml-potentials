"""
ML experiment: can we learn local parabola parameters a(r) across molecules?

Setup:
  - Multiple diatomics with Morse-based E_elec(r) = V_Morse(r) - Z1*Z2*ke/r
  - Each molecule has different (D_e, alpha, r_e, Z1*Z2)
  - At each grid point r_j, compute local parabola: a_j, b_j, c_j
  - Train on some molecules, predict for test molecules

Compare:
  1. Direct:         Z → E_elec(r_j) at each grid point
  2. Local parabola: Z → a(r_j), b(r_j), c(r_j), then reconstruct E_elec
  3. Global adaptive: Z → (D_e, α, r_e) → E_elec(r) via analytical formula

Question: does learning a(r) help vs learning E_elec(r)?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

ke = 14.3996  # eV·Å

# ================================================================
# Define molecules: homonuclear-like diatomics with varying Z
# Parameters roughly follow periodic trends
# ================================================================
molecules = {}
Z_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Morse params scale with Z (rough physical trends)
for Z in Z_values:
    D_e = 1.0 + 0.5 * Z + 0.02 * Z**2    # deeper well for heavier atoms
    alpha = 1.2 + 0.08 * Z                 # width parameter
    r_e = 0.7 + 0.15 * Z - 0.003 * Z**2   # equilibrium distance
    Z1Z2 = Z**2                             # homonuclear: Z1=Z2=Z
    molecules[Z] = {'D_e': D_e, 'alpha': alpha, 'r_e': r_e, 'Z1Z2': Z1Z2}

# Train/test split: train on Z=1-8, test on Z=9-12 (extrapolation)
train_Z = np.array([1, 2, 3, 4, 5, 6, 7, 8])
test_Z = np.array([9, 10, 11, 12])

# Fixed r grid
r_grid = np.linspace(0.5, 5.0, 50)
n_grid = len(r_grid)

# ================================================================
# Compute E_elec and local parabola params for all molecules
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def V_morse_d1(r, D_e, alpha, r_e):
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha * (1 - x) * x

def V_morse_d2(r, D_e, alpha, r_e):
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha**2 * x * (2*x - 1)

def compute_eelec(r, p):
    return V_morse(r, p['D_e'], p['alpha'], p['r_e']) - p['Z1Z2'] * ke / r

def compute_a(r, p):
    """a(r) = E_elec''(r) / 2"""
    return V_morse_d2(r, p['D_e'], p['alpha'], p['r_e']) / 2.0 - p['Z1Z2'] * ke / r**3

def compute_b(r, p):
    """b(r) = E_elec'(r)"""
    return V_morse_d1(r, p['D_e'], p['alpha'], p['r_e']) + p['Z1Z2'] * ke / r**2

# Build data arrays
E_all = np.zeros((len(Z_values), n_grid))
a_all = np.zeros((len(Z_values), n_grid))
b_all = np.zeros((len(Z_values), n_grid))
params_all = np.zeros((len(Z_values), 3))  # D_e, alpha, r_e

for i, Z in enumerate(Z_values):
    p = molecules[Z]
    E_all[i] = compute_eelec(r_grid, p)
    a_all[i] = compute_a(r_grid, p)
    b_all[i] = compute_b(r_grid, p)
    params_all[i] = [p['D_e'], p['alpha'], p['r_e']]

train_mask = np.isin(Z_values, train_Z)
test_mask = np.isin(Z_values, test_Z)
train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

# ================================================================
# Method 1: Direct — Z → E_elec(r) multioutput
# ================================================================
X_tr = train_Z.reshape(-1, 1)
X_te = test_Z.reshape(-1, 1)

mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(X_tr, E_all[train_idx])
E_pred_direct = mdl_direct.predict(X_te)

mae_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_direct.ravel())

# ================================================================
# Method 2: Local parabola — Z → a(r), b(r), c(r), reconstruct
# Note: c(r) = E_elec(r), so we predict a, b, c separately
# then reconstruct... but c IS E_elec, so reconstruction is trivial.
#
# The REAL test: predict ONLY a(r) and b(r), then integrate.
# But that requires boundary conditions.
#
# More realistic: predict a(r) at each grid point as the target.
# Then E_elec is reconstructed from a(r) via the parabola.
# But each local parabola only covers a local region — you can't
# reconstruct the global curve from a(r) alone without c(r).
#
# So the fair comparison is: predict a(r) multioutput, and also c(r).
# ================================================================

# Method 2a: predict a(r) multioutput (same structure as direct, but different targets)
mdl_a = Ridge(alpha=1e-6)
mdl_a.fit(X_tr, a_all[train_idx])
a_pred = mdl_a.predict(X_te)

mdl_b = Ridge(alpha=1e-6)
mdl_b.fit(X_tr, b_all[train_idx])
b_pred = mdl_b.predict(X_te)

# Can we reconstruct E_elec from predicted a(r) and b(r)?
# Not directly — we'd need c(r) too, or integration.
# But predicting c(r) = E_elec(r) IS the direct method.
# So local parabola doesn't reduce what we need to predict.

# The ONLY way parabola helps: if a(r) is smoother across Z than E_elec(r)
# Let's check: MAE of a(r) prediction
mae_a = mean_absolute_error(a_all[test_idx].ravel(), a_pred.ravel())
mae_b = mean_absolute_error(b_all[test_idx].ravel(), b_pred.ravel())

# Method 2b: shared model (Z, r) → a(r)
# Stack all (Z, r) pairs into a single training set
X_shared_tr = np.column_stack([
    np.repeat(train_Z, n_grid),
    np.tile(r_grid, len(train_Z))
])
X_shared_te = np.column_stack([
    np.repeat(test_Z, n_grid),
    np.tile(r_grid, len(test_Z))
])

# Shared direct: (Z, r) → E_elec
mdl_shared_direct = Ridge(alpha=1e-6)
mdl_shared_direct.fit(X_shared_tr, E_all[train_idx].ravel())
E_pred_shared = mdl_shared_direct.predict(X_shared_te).reshape(len(test_Z), n_grid)
mae_shared_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_shared.ravel())

# Shared parabola: (Z, r) → a(r), then what?
mdl_shared_a = Ridge(alpha=1e-6)
mdl_shared_a.fit(X_shared_tr, a_all[train_idx].ravel())
a_pred_shared = mdl_shared_a.predict(X_shared_te).reshape(len(test_Z), n_grid)
mae_shared_a = mean_absolute_error(a_all[test_idx].ravel(), a_pred_shared.ravel())

# ================================================================
# Method 3: Global adaptive — Z → (D_e, α, r_e) → E_elec analytically
# ================================================================
pred_params = np.zeros((len(test_Z), 3))
for p in range(3):
    m = Ridge(alpha=1e-6)
    m.fit(X_tr, params_all[train_idx, p])
    pred_params[:, p] = m.predict(X_te)

# Reconstruct E_elec from predicted global params
E_pred_adaptive = np.zeros((len(test_Z), n_grid))
a_pred_adaptive = np.zeros((len(test_Z), n_grid))  # also get a(r) from predicted params
for j in range(len(test_Z)):
    Z = test_Z[j]
    Z1Z2 = Z**2
    D_e_p, alpha_p, r_e_p = pred_params[j]
    p_pred = {'D_e': D_e_p, 'alpha': alpha_p, 'r_e': r_e_p, 'Z1Z2': Z1Z2}
    E_pred_adaptive[j] = compute_eelec(r_grid, p_pred)
    a_pred_adaptive[j] = compute_a(r_grid, p_pred)

mae_adaptive = mean_absolute_error(E_all[test_idx].ravel(), E_pred_adaptive.ravel())
mae_adaptive_a = mean_absolute_error(a_all[test_idx].ravel(), a_pred_adaptive.ravel())

# ================================================================
# Results
# ================================================================
print(f'{"="*80}')
print(f'ML EXTRAPOLATION: train Z=1-8, test Z=9-12')
print(f'{"="*80}')
print(f'\n--- E_elec prediction ---')
print(f'{"Method":<50s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*70}')
print(f'{"Direct multioutput: Z → E(r)":<50s} | {mae_direct:8.2f} | {"base":>7s}')
print(f'{"Shared direct: (Z,r) → E":<50s} | {mae_shared_direct:8.2f} | {mae_shared_direct/mae_direct:7.3f}x')
print(f'{"Global adaptive: Z → (De,α,re) → E(r)":<50s} | {mae_adaptive:8.2f} | {mae_adaptive/mae_direct:7.3f}x')
print(f'{"-"*70}')

print(f'\n--- a(r) prediction ---')
print(f'{"Method":<50s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*70}')
print(f'{"Direct multioutput: Z → a(r)":<50s} | {mae_a:8.2f} | {"base":>7s}')
print(f'{"Shared direct: (Z,r) → a":<50s} | {mae_shared_a:8.2f} | {mae_shared_a/mae_a:7.3f}x')
print(f'{"Global adaptive: Z → (De,α,re) → a(r)":<50s} | {mae_adaptive_a:8.2f} | {mae_adaptive_a/mae_a:7.3f}x')
print(f'{"-"*70}')

# Per-molecule
print(f'\nPer-molecule E_elec MAE:')
print(f'{"Method":<30s} |', ' '.join(f'Z={Z:>3d}' for Z in test_Z))
for label, pred in [('Direct Z→E(r)', E_pred_direct),
                     ('Global adaptive', E_pred_adaptive)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<30s} |', ' '.join(f'{m:6.1f}' for m in maes))

# ================================================================
# FIGURE: The key comparison
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top row: E_elec curves and predictions for test molecules
for j in range(min(3, len(test_Z))):
    ax = axes[0, j]
    Z = test_Z[j]
    ax.plot(r_grid, E_all[test_idx[j]], 'k-', lw=2.5, label='True $E_{elec}$', zorder=5)
    ax.plot(r_grid, E_pred_direct[j], 'b--', lw=1.5,
            label=f'Direct ({mean_absolute_error(E_all[test_idx[j]], E_pred_direct[j]):.1f} eV)')
    ax.plot(r_grid, E_pred_adaptive[j], 'r-', lw=2,
            label=f'Adaptive ({mean_absolute_error(E_all[test_idx[j]], E_pred_adaptive[j]):.1f} eV)')
    ax.set_xlabel('$r$ [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=11)
    ax.set_title(f'Z = {Z}', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Bottom row: a(r) curves and predictions
for j in range(min(3, len(test_Z))):
    ax = axes[1, j]
    Z = test_Z[j]
    ax.plot(r_grid, a_all[test_idx[j]], 'k-', lw=2.5, label='True $a(r)$', zorder=5)

    # Direct a(r) prediction
    mae_a_j = mean_absolute_error(a_all[test_idx[j]], a_pred[j])
    ax.plot(r_grid, a_pred[j], 'b--', lw=1.5,
            label=f'Direct Z→a(r) ({mae_a_j:.1f})')

    # Adaptive a(r) from global params
    mae_adap_a_j = mean_absolute_error(a_all[test_idx[j]], a_pred_adaptive[j])
    ax.plot(r_grid, a_pred_adaptive[j], 'r-', lw=2,
            label=f'Adaptive (Dₑ,α,rₑ)→a(r) ({mae_adap_a_j:.1f})')

    ax.set_xlabel('$r$ [Å]', fontsize=11)
    ax.set_ylabel('$a(r) = E\'\'_{elec}/2$ [eV/Å²]', fontsize=11)
    ax.set_title(f'Z = {Z}: curvature $a(r)$', fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle('Extrapolation: direct $E_{elec}$ vs global adaptive vs local $a(r)$\n'
             'Top: $E_{elec}(r)$ predictions. Bottom: curvature $a(r)$ predictions.\n'
             'Global adaptive (red) wins for BOTH targets.',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_eelec_adaptive_parabolas.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_eelec_adaptive_parabolas.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Why local a(r) = direct for Ridge (mathematical identity)
# ================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: a(r) across all molecules — is it smoother than E_elec?
ax = axes2[0]
for i, Z in enumerate(Z_values):
    style = '-' if Z in train_Z else '--'
    color = '#1f77b4' if Z in train_Z else '#d62728'
    ax.plot(r_grid, a_all[i], style, color=color, alpha=0.7, lw=1.5,
            label=f'Z={Z}' if Z in [1, 4, 8, 9, 12] else '')
ax.set_xlabel('$r$ [Å]', fontsize=11)
ax.set_ylabel('$a(r)$ [eV/Å²]', fontsize=11)
ax.set_title('$a(r)$ across molecules\nBlue=train, Red=test', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 2: E_elec across all molecules
ax = axes2[1]
for i, Z in enumerate(Z_values):
    style = '-' if Z in train_Z else '--'
    color = '#1f77b4' if Z in train_Z else '#d62728'
    ax.plot(r_grid, E_all[i], style, color=color, alpha=0.7, lw=1.5,
            label=f'Z={Z}' if Z in [1, 4, 8, 9, 12] else '')
ax.set_xlabel('$r$ [Å]', fontsize=11)
ax.set_ylabel('$E_{elec}$ [eV]', fontsize=11)
ax.set_title('$E_{elec}(r)$ across molecules\nBlue=train, Red=test', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 3: Global parameters vs Z — smooth and learnable
ax = axes2[2]
param_names = ['$D_e$', r'$\alpha$', '$r_e$']
colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c']
for p in range(3):
    ax2_twin = ax if p == 0 else ax.twinx() if p == 1 else ax
    if p == 1:
        ax2_twin = ax.twinx()
        ax2_twin.plot(Z_values, params_all[:, p], 'o-', color=colors_p[p],
                      ms=6, label=param_names[p])
        ax2_twin.set_ylabel(param_names[p], color=colors_p[p], fontsize=11)
    else:
        ax.plot(Z_values, params_all[:, p], 'o-', color=colors_p[p],
                ms=6, label=param_names[p])

ax.axvline(8.5, color='gray', ls=':', alpha=0.5, label='train|test')
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('Parameter value', fontsize=11)
ax.set_title('Global Morse params vs Z\n3 smooth numbers per molecule', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

fig2.suptitle('Why global adaptive beats local parabola:\n'
              '3 smooth parameters vs 50 grid-point values',
              fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_eelec_why_global_wins.png'), dpi=150, bbox_inches='tight')
print('Saved fig_eelec_why_global_wins.png')
plt.close(fig2)

# ================================================================
# Key insight printout
# ================================================================
print(f'\n{"="*70}')
print(f'KEY INSIGHT')
print(f'{"="*70}')
print(f'For Ridge regression:')
print(f'  Predicting a(r) at 50 grid points = predicting E(r) at 50 grid points')
print(f'  Both are 50-dimensional multioutput problems.')
print(f'  Local parabola does NOT reduce dimensionality.')
print(f'')
print(f'  Direct E(r):     MAE = {mae_direct:.2f} eV  (50 targets)')
print(f'  Direct a(r):     MAE_a = {mae_a:.2f} eV/Å²  (50 targets, different units)')
print(f'  Global adaptive: MAE = {mae_adaptive:.2f} eV  (3 targets → 50 via physics)')
print(f'  Advantage ratio: {mae_direct/mae_adaptive:.1f}x')
print(f'')
print(f'The advantage comes from predicting 3 GLOBAL parameters')
print(f'and using the NONLINEAR physics equation to reconstruct 50 values.')
print(f'Local parabola has 3 params PER grid point (150 total), not 3 total.')
