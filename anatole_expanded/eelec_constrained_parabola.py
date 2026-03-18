"""
Anatole's constrained vertex parabola:

    P(r) = a · (r - r₀)²

where r₀ is NOT free — it's determined by a via the boundary condition
that at large r (r_ref = 10 Å), the slope matches the nuclear repulsion:

    P'(r_ref) = -V_nn'(r_ref) = Z₁Z₂·ke / r_ref²

    2a·(r_ref - r₀) = Z₁Z₂·ke / r_ref²

    r₀(a) = r_ref - Z₁Z₂·ke / (2a·r_ref²)

So we have ONE free parameter (a) per molecule.
This is true dimensionality reduction: 50 grid values → 1 parameter.

The ML question: can we learn a(Z) for new molecules?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

ke = 14.3996  # eV·Å
r_ref = 10.0  # Å — large-r reference for slope matching

# ================================================================
# Morse-based E_elec for multiple molecules
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def E_elec_func(r, D_e, alpha, r_e, Z1Z2):
    return V_morse(r, D_e, alpha, r_e) - Z1Z2 * ke / r

def E_elec_d1(r, D_e, alpha, r_e, Z1Z2):
    x = np.exp(-alpha * (r - r_e))
    dV = 2 * D_e * alpha * (1 - x) * x
    return dV + Z1Z2 * ke / r**2

# ================================================================
# Constrained parabola: P(r) = a · (r - r₀(a))²
# ================================================================
def r0_from_a(a, Z1Z2):
    """Vertex position from curvature a via slope boundary condition.

    At r_ref: P'(r_ref) = slope of -V_nn = Z1Z2*ke/r_ref²
    2a*(r_ref - r0) = Z1Z2*ke/r_ref²
    r0 = r_ref - Z1Z2*ke/(2*a*r_ref²)
    """
    slope_nn = Z1Z2 * ke / r_ref**2
    return r_ref - slope_nn / (2 * a)


def fit_a_to_eelec(r_grid, E_true, Z1Z2):
    """Find the best a that minimizes MAE of P(r) = a*(r-r0(a))² to E_true."""
    def objective(a):
        if a >= -1e-10:  # a must be negative
            return 1e10
        r0 = r0_from_a(a, Z1Z2)
        P = a * (r_grid - r0)**2
        return np.mean((E_true - P)**2)

    # Search over negative a values
    result = minimize_scalar(objective, bounds=(-50, -0.01), method='bounded')
    return result.x


# ================================================================
# Define molecules
# ================================================================
Z_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

molecules = {}
for Z in Z_values:
    D_e = 1.0 + 0.5 * Z + 0.02 * Z**2
    alpha = 1.2 + 0.08 * Z
    r_e = 0.7 + 0.15 * Z - 0.003 * Z**2
    Z1Z2 = Z**2
    molecules[Z] = {'D_e': D_e, 'alpha': alpha, 'r_e': r_e, 'Z1Z2': Z1Z2}

train_Z = np.array([1, 2, 3, 4, 5, 6, 7, 8])
test_Z = np.array([9, 10, 11, 12])

r_grid = np.linspace(0.5, 5.0, 50)
n_grid = len(r_grid)

# ================================================================
# Compute E_elec for all molecules
# ================================================================
E_all = np.zeros((len(Z_values), n_grid))
for i, Z in enumerate(Z_values):
    p = molecules[Z]
    E_all[i] = E_elec_func(r_grid, p['D_e'], p['alpha'], p['r_e'], p['Z1Z2'])

train_idx = np.where(np.isin(Z_values, train_Z))[0]
test_idx = np.where(np.isin(Z_values, test_Z))[0]

# ================================================================
# Fit constrained parabola to each molecule
# ================================================================
a_fitted = np.zeros(len(Z_values))
r0_fitted = np.zeros(len(Z_values))
fit_mae = np.zeros(len(Z_values))

print(f'{"="*70}')
print(f'Constrained parabola fit: P(r) = a·(r - r₀(a))²')
print(f'Slope at r={r_ref} Å matches nuclear repulsion')
print(f'{"="*70}')
print(f'{"Z":>3s} | {"a":>10s} | {"r₀":>8s} | {"fit MAE":>8s} | {"Z1Z2":>5s}')
print(f'{"-"*50}')

for i, Z in enumerate(Z_values):
    p = molecules[Z]
    a_fitted[i] = fit_a_to_eelec(r_grid, E_all[i], p['Z1Z2'])
    r0_fitted[i] = r0_from_a(a_fitted[i], p['Z1Z2'])
    P_fit = a_fitted[i] * (r_grid - r0_fitted[i])**2
    fit_mae[i] = mean_absolute_error(E_all[i], P_fit)
    marker = '  TEST' if Z in test_Z else ''
    print(f'{Z:3d} | {a_fitted[i]:10.4f} | {r0_fitted[i]:8.3f} | {fit_mae[i]:8.3f} | {p["Z1Z2"]:5d}{marker}')

# ================================================================
# ML experiment
# ================================================================
print(f'\n{"="*70}')
print(f'ML EXTRAPOLATION: train Z=1-8, test Z=9-12')
print(f'{"="*70}')

# Method 1: Direct multioutput Z → E_elec(r)
X_tr = train_Z.reshape(-1, 1)
X_te = test_Z.reshape(-1, 1)

mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(X_tr, E_all[train_idx])
E_pred_direct = mdl_direct.predict(X_te)
mae_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_direct.ravel())

# Method 2: Constrained parabola Z → a → r₀(a) → E_elec
mdl_a = Ridge(alpha=1e-6)
mdl_a.fit(X_tr, a_fitted[train_idx])
a_pred = mdl_a.predict(X_te)

E_pred_parab = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    p = molecules[Z]
    r0_pred = r0_from_a(a_pred[j], p['Z1Z2'])
    E_pred_parab[j] = a_pred[j] * (r_grid - r0_pred)**2

mae_parab = mean_absolute_error(E_all[test_idx].ravel(), E_pred_parab.ravel())

# Method 3: Global Morse adaptive Z → (D_e, α, r_e) → E_elec
params_all = np.array([[molecules[Z]['D_e'], molecules[Z]['alpha'], molecules[Z]['r_e']]
                        for Z in Z_values])
pred_params = np.zeros((len(test_Z), 3))
for p in range(3):
    m = Ridge(alpha=1e-6)
    m.fit(X_tr, params_all[train_idx, p])
    pred_params[:, p] = m.predict(X_te)

E_pred_morse = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    p_pred = pred_params[j]
    E_pred_morse[j] = E_elec_func(r_grid, p_pred[0], p_pred[1], p_pred[2], Z**2)

mae_morse = mean_absolute_error(E_all[test_idx].ravel(), E_pred_morse.ravel())

# Results
print(f'\n{"Method":<50s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*70}')
print(f'{"Direct multioutput: Z → E(r) [50 params]":<50s} | {mae_direct:8.2f} | {"base":>7s}')
print(f'{"Constrained parabola: Z → a → r₀(a) [1 param]":<50s} | {mae_parab:8.2f} | {mae_parab/mae_direct:7.3f}x')
print(f'{"Global Morse: Z → (De,α,re) → E(r) [3 params]":<50s} | {mae_morse:8.2f} | {mae_morse/mae_direct:7.3f}x')
print(f'{"-"*70}')

# Per-molecule
print(f'\nPer-molecule MAE:')
print(f'{"Method":<30s} |', ' '.join(f'Z={Z:>3d}' for Z in test_Z))
for label, pred in [('Direct', E_pred_direct),
                     ('Parabola (1 param)', E_pred_parab),
                     ('Morse (3 params)', E_pred_morse)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<30s} |', ' '.join(f'{m:6.1f}' for m in maes))

# Parameter prediction
print(f'\nParameter a: true vs predicted')
print(f'{"Z":>3s} | {"a_true":>10s} | {"a_pred":>10s} | {"r₀_true":>8s} | {"r₀_pred":>8s}')
for j, Z in enumerate(test_Z):
    i = np.where(Z_values == Z)[0][0]
    r0_p = r0_from_a(a_pred[j], Z**2)
    print(f'{Z:3d} | {a_fitted[i]:10.4f} | {a_pred[j]:10.4f} | {r0_fitted[i]:8.3f} | {r0_p:8.3f}')

# ================================================================
# FIGURES
# ================================================================

# Figure 1: Fit quality + ML predictions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top: constrained parabola fits for selected molecules
show_mols = [1, 4, 8, 9, 10, 12]
for idx_plot, Z in enumerate(show_mols):
    ax = axes.flat[idx_plot]
    i = np.where(Z_values == Z)[0][0]
    p = molecules[Z]

    ax.plot(r_grid, E_all[i], 'k-', lw=2.5, label='$E_{elec}$ (true)', zorder=5)

    # Fitted parabola
    P_fit = a_fitted[i] * (r_grid - r0_fitted[i])**2
    ax.plot(r_grid, P_fit, 'b--', lw=1.5,
            label=f'Fit: $a$={a_fitted[i]:.2f}, $r_0$={r0_fitted[i]:.1f} ({fit_mae[i]:.1f})')

    # If test molecule, show ML prediction
    if Z in test_Z:
        j = np.where(test_Z == Z)[0][0]
        ax.plot(r_grid, E_pred_direct[j], 'g:', lw=1.5, alpha=0.7,
                label=f'Direct ML ({mean_absolute_error(E_all[i], E_pred_direct[j]):.1f})')
        ax.plot(r_grid, E_pred_parab[j], 'r-', lw=2,
                label=f'Parabola ML ({mean_absolute_error(E_all[i], E_pred_parab[j]):.1f})')

        # Mark predicted vertex
        r0_p = r0_from_a(a_pred[j], Z**2)
        if r0_p < 15:
            ax.axvline(r0_p, color='r', ls=':', alpha=0.3)

    # Mark fitted vertex
    if r0_fitted[i] < 15:
        ax.axvline(r0_fitted[i], color='b', ls=':', alpha=0.3)

    is_test = Z in test_Z
    color = '#d62728' if is_test else '#1f77b4'
    ax.set_title(f'Z = {Z} (Z₁Z₂={p["Z1Z2"]}){"  [TEST]" if is_test else ""}',
                 fontsize=11, color=color, fontweight='bold' if is_test else 'normal')
    ax.set_xlabel('$r$ [Å]', fontsize=10)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=10)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)

fig.suptitle('Constrained parabola: $P(r) = a \\cdot (r - r_0(a))^2$\n'
             'Slope at $r$=10 Å matches nuclear repulsion. ONE parameter per molecule.',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_constrained_parabola_fits.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_constrained_parabola_fits.png')
plt.close(fig)

# Figure 2: Parameter a vs Z — is it learnable?
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: a vs Z
ax1.plot(train_Z, a_fitted[train_idx], 'bo', ms=10, label='Train (fitted)', zorder=5)
ax1.plot(test_Z, a_fitted[test_idx], 'rs', ms=10, label='Test (fitted)', zorder=5)
ax1.plot(test_Z, a_pred.ravel(), 'r^', ms=10, label='Test (predicted)', zorder=5)

# Show Ridge prediction line
Z_dense = np.linspace(0, 13, 100)
ax1.plot(Z_dense, mdl_a.predict(Z_dense.reshape(-1, 1)),
         'b-', alpha=0.4, lw=1.5, label='Ridge prediction')

for i, Z in enumerate(Z_values):
    ax1.annotate(f'Z={Z}', (Z, a_fitted[i]),
                 textcoords='offset points', xytext=(5, 5), fontsize=7)

ax1.set_xlabel('Z', fontsize=12)
ax1.set_ylabel('$a$ [eV/Å²]', fontsize=12)
ax1.set_title('Curvature parameter $a$ vs Z\nSmooth → learnable?', fontsize=11)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: r₀ vs Z
ax2.plot(Z_values, r0_fitted, 'ko-', ms=8)
ax2.axvline(8.5, color='gray', ls=':', alpha=0.5)
for i, Z in enumerate(Z_values):
    ax2.annotate(f'{r0_fitted[i]:.1f}', (Z, r0_fitted[i]),
                 textcoords='offset points', xytext=(5, 5), fontsize=8)
ax2.set_xlabel('Z', fontsize=12)
ax2.set_ylabel('$r_0$ [Å]', fontsize=12)
ax2.set_title('Vertex position $r_0$ vs Z\n(derived from $a$ via slope condition)', fontsize=11)
ax2.grid(True, alpha=0.3)

# Panel 3: fit quality
ax3.bar(Z_values, fit_mae, color=['#1f77b4']*8 + ['#d62728']*4)
ax3.set_xlabel('Z', fontsize=12)
ax3.set_ylabel('Fit MAE [eV]', fontsize=12)
ax3.set_title('Parabola fit quality\n(1 param per molecule)', fontsize=11)
ax3.grid(True, alpha=0.3)

fig2.suptitle('Constrained vertex parabola: $r_0(a) = r_{ref} - Z_1 Z_2 k_e / (2a \\cdot r_{ref}^2)$',
              fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_constrained_parabola_params.png'), dpi=150, bbox_inches='tight')
print('Saved fig_constrained_parabola_params.png')
plt.close(fig2)
