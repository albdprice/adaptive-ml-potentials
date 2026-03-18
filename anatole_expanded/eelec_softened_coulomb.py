"""
Softened Coulomb model for E_elec:

    E_elec(r) ≈ -Z_eff · ke / √(r² + a²)

Physical motivation:
  - At large r: behaves as -Z_eff·ke/r (cancels nuclear repulsion)
  - At r → 0: finite value -Z_eff·ke/a (united atom limit, no singularity)
  - 2 parameters: Z_eff (effective charge), a (screening length)

1-param version: fix Z_eff = Z₁Z₂, learn only a
2-param version: learn both Z_eff and a
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

ke = 14.3996  # eV·Å

# ================================================================
# Morse-based E_elec (ground truth)
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def E_elec_func(r, D_e, alpha, r_e, Z1Z2):
    return V_morse(r, D_e, alpha, r_e) - Z1Z2 * ke / r

# ================================================================
# Softened Coulomb model
# ================================================================
def softened_coulomb_1p(r, a, Z1Z2):
    """1-param: Z_eff = Z1Z2 fixed, only screening length a is free."""
    return -Z1Z2 * ke / np.sqrt(r**2 + a**2)

def softened_coulomb_2p(r, Z_eff, a):
    """2-param: both Z_eff and screening length a are free."""
    return -Z_eff * ke / np.sqrt(r**2 + a**2)

def fit_1param(r_grid, E_true, Z1Z2):
    """Fit screening length a with Z_eff = Z1Z2 fixed."""
    def obj(a):
        if a <= 0:
            return 1e10
        P = softened_coulomb_1p(r_grid, a, Z1Z2)
        return np.mean((E_true - P)**2)
    result = minimize_scalar(obj, bounds=(0.01, 20.0), method='bounded')
    return result.x

def fit_2param(r_grid, E_true):
    """Fit both Z_eff and screening length a."""
    def obj(params):
        Z_eff, a = params
        if a <= 0 or Z_eff <= 0:
            return 1e10
        P = softened_coulomb_2p(r_grid, Z_eff, a)
        return np.mean((E_true - P)**2)
    result = minimize(obj, x0=[1.0, 1.0], method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x

# ================================================================
# Define molecules (same as eelec_anatole_parabola.py)
# ================================================================
Z_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

molecules = {}
for Z in Z_values:
    D_e = 0.5 + 0.3 * Z
    alpha = 1.5
    r_e = 0.8 + 0.1 * Z
    Z1Z2 = Z**2
    molecules[Z] = {'D_e': D_e, 'alpha': alpha, 'r_e': r_e, 'Z1Z2': Z1Z2}

train_Z = np.array([1, 2, 3, 4, 5, 6, 7, 8])
test_Z = np.array([9, 10, 11, 12])

r_grid = np.linspace(0.5, 5.0, 50)
n_grid = len(r_grid)

# Compute E_elec
E_all = np.zeros((len(Z_values), n_grid))
for i, Z in enumerate(Z_values):
    p = molecules[Z]
    E_all[i] = E_elec_func(r_grid, p['D_e'], p['alpha'], p['r_e'], p['Z1Z2'])

train_idx = np.where(np.isin(Z_values, train_Z))[0]
test_idx = np.where(np.isin(Z_values, test_Z))[0]

# ================================================================
# Fit softened Coulomb to each molecule
# ================================================================
a_1p = np.zeros(len(Z_values))       # 1-param: screening length
Z_eff_2p = np.zeros(len(Z_values))   # 2-param: effective charge
a_2p = np.zeros(len(Z_values))       # 2-param: screening length
fit_mae_1p = np.zeros(len(Z_values))
fit_mae_2p = np.zeros(len(Z_values))

print(f'{"="*85}')
print(f'Softened Coulomb fits: E_elec ≈ -Z_eff·ke/√(r² + a²)')
print(f'{"="*85}')
print(f'{"Z":>3s} {"Z1Z2":>5s} | {"a(1p)":>8s} {"MAE 1p":>8s} | '
      f'{"Zeff(2p)":>8s} {"a(2p)":>8s} {"MAE 2p":>8s} | {"Zeff/Z1Z2":>9s}')
print(f'{"-"*85}')

for i, Z in enumerate(Z_values):
    p = molecules[Z]
    Z1Z2 = p['Z1Z2']

    # 1-param: fix Z_eff = Z1Z2
    a_1p[i] = fit_1param(r_grid, E_all[i], Z1Z2)
    P1 = softened_coulomb_1p(r_grid, a_1p[i], Z1Z2)
    fit_mae_1p[i] = mean_absolute_error(E_all[i], P1)

    # 2-param: free Z_eff and a
    Z_eff_2p[i], a_2p[i] = fit_2param(r_grid, E_all[i])
    P2 = softened_coulomb_2p(r_grid, Z_eff_2p[i], a_2p[i])
    fit_mae_2p[i] = mean_absolute_error(E_all[i], P2)

    marker = '  TEST' if Z in test_Z else ''
    print(f'{Z:3d} {Z1Z2:5d} | {a_1p[i]:8.3f} {fit_mae_1p[i]:8.2f} | '
          f'{Z_eff_2p[i]:8.3f} {a_2p[i]:8.3f} {fit_mae_2p[i]:8.2f} | '
          f'{Z_eff_2p[i]/Z1Z2:9.4f}{marker}')

# ================================================================
# ML predictions
# ================================================================
X_tr = train_Z.reshape(-1, 1)
X_te = test_Z.reshape(-1, 1)

# Direct baseline: Z → E(r)
mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(X_tr, E_all[train_idx])
E_pred_direct = mdl_direct.predict(X_te)
mae_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_direct.ravel())

# 1-param adaptive: Z → a (with Z_eff = Z1Z2 known)
mdl_a1 = Ridge(alpha=1e-6)
mdl_a1.fit(X_tr, a_1p[train_idx])
a1_pred = mdl_a1.predict(X_te)

E_pred_1p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_1p[j] = softened_coulomb_1p(r_grid, a1_pred[j], Z**2)
mae_1p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_1p.ravel())

# 2-param adaptive: Z → (Z_eff, a)
mdl_Zeff = Ridge(alpha=1e-6)
mdl_Zeff.fit(X_tr, Z_eff_2p[train_idx])
Zeff_pred = mdl_Zeff.predict(X_te)

mdl_a2 = Ridge(alpha=1e-6)
mdl_a2.fit(X_tr, a_2p[train_idx])
a2_pred = mdl_a2.predict(X_te)

E_pred_2p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_2p[j] = softened_coulomb_2p(r_grid, Zeff_pred[j], a2_pred[j])
mae_2p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_2p.ravel())

# Morse adaptive: Z → (D_e, α, r_e)
params_all = np.array([[molecules[Z]['D_e'], molecules[Z]['alpha'], molecules[Z]['r_e']]
                        for Z in Z_values])
pred_params = np.zeros((len(test_Z), 3))
for p in range(3):
    m = Ridge(alpha=1e-6)
    m.fit(X_tr, params_all[train_idx, p])
    pred_params[:, p] = m.predict(X_te)
E_pred_morse = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_morse[j] = E_elec_func(r_grid, pred_params[j, 0], pred_params[j, 1],
                                   pred_params[j, 2], Z**2)
mae_morse = mean_absolute_error(E_all[test_idx].ravel(), E_pred_morse.ravel())

# Results
print(f'\n{"="*80}')
print(f'ML EXTRAPOLATION: train Z=1-8, test Z=9-12')
print(f'{"="*80}')
print(f'{"Method":<55s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*75}')
print(f'{"Direct: Z → E(r) [50 targets]":<55s} | {mae_direct:8.2f} | {"base":>7s}')
print(f'{"Softened Coulomb 1p: Z → a [Z_eff=Z1Z2 fixed]":<55s} | {mae_1p:8.2f} | {mae_1p/mae_direct:7.3f}x')
print(f'{"Softened Coulomb 2p: Z → (Z_eff, a)":<55s} | {mae_2p:8.2f} | {mae_2p/mae_direct:7.3f}x')
print(f'{"Morse adaptive: Z → (De,α,re) [3 targets]":<55s} | {mae_morse:8.2f} | {mae_morse/mae_direct:7.3f}x')
print(f'{"-"*75}')

# Per-molecule
print(f'\nPer-molecule MAE:')
print(f'{"Method":<30s} |', ' '.join(f'Z={Z:>4d}' for Z in test_Z))
for label, pred in [('Direct (50 targets)', E_pred_direct),
                     ('Soft. Coulomb 1p (a)', E_pred_1p),
                     ('Soft. Coulomb 2p (Zeff,a)', E_pred_2p),
                     ('Morse (De,α,re)', E_pred_morse)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<30s} |', ' '.join(f'{m:7.1f}' for m in maes))

# ================================================================
# FIGURE: Parameter trends and ML results
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top-left: screening length a(1p) vs Z
ax = axes[0, 0]
ax.plot(train_Z, a_1p[train_idx], 'bo', ms=10, label='Train', zorder=5)
ax.plot(test_Z, a_1p[test_idx], 'rs', ms=10, label='Test (true)', zorder=5)
ax.plot(test_Z, a1_pred.ravel(), 'r^', ms=10, label='Test (pred)', zorder=5)
Z_dense = np.linspace(0, 13, 100)
ax.plot(Z_dense, mdl_a1.predict(Z_dense.reshape(-1, 1)), 'b-', alpha=0.4)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('Screening length $a$ [Å]', fontsize=11)
ax.set_title('1-param: $a$ vs Z ($Z_{eff} = Z_1 Z_2$ fixed)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Top-middle: Z_eff and a(2p) vs Z
ax = axes[0, 1]
ax.plot(Z_values, Z_eff_2p, 'bo-', ms=6, label='$Z_{eff}$ (fitted)')
ax.plot(test_Z, Zeff_pred.ravel(), 'b^', ms=8, label='$Z_{eff}$ (pred)')
ax2 = ax.twinx()
ax2.plot(Z_values, a_2p, 'rs-', ms=6, label='$a$ (fitted)')
ax2.plot(test_Z, a2_pred.ravel(), 'r^', ms=8, label='$a$ (pred)')
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$Z_{eff}$', fontsize=11, color='b')
ax2.set_ylabel('$a$ [Å]', fontsize=11, color='r')
ax.set_title('2-param: $Z_{eff}$ and $a$ vs Z', fontsize=11)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7)
ax.grid(True, alpha=0.3)

# Top-right: fit quality
ax = axes[0, 2]
x = np.arange(len(Z_values))
w = 0.35
ax.bar(x - w/2, fit_mae_1p, w, label='1-param ($a$ only)', color='steelblue')
ax.bar(x + w/2, fit_mae_2p, w, label='2-param ($Z_{eff}$, $a$)', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(Z_values)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('Fit MAE [eV]', fontsize=11)
ax.set_title('Fit quality (lower = better shape match)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Bottom row: predictions for test molecules
for j in range(min(3, len(test_Z))):
    ax = axes[1, j]
    Z = test_Z[j]
    i = test_idx[j]
    ax.plot(r_grid, E_all[i], 'k-', lw=2.5, label='True $E_{elec}$', zorder=5)

    mae_d = mean_absolute_error(E_all[i], E_pred_direct[j])
    ax.plot(r_grid, E_pred_direct[j], 'g:', lw=1.5,
            label=f'Direct ({mae_d:.0f})')

    mae_p1 = mean_absolute_error(E_all[i], E_pred_1p[j])
    ax.plot(r_grid, E_pred_1p[j], 'r-', lw=2,
            label=f'1-param ({mae_p1:.0f})')

    mae_p2 = mean_absolute_error(E_all[i], E_pred_2p[j])
    ax.plot(r_grid, E_pred_2p[j], 'b--', lw=1.5,
            label=f'2-param ({mae_p2:.0f})')

    mae_m = mean_absolute_error(E_all[i], E_pred_morse[j])
    ax.plot(r_grid, E_pred_morse[j], 'm-.', lw=1.5,
            label=f'Morse ({mae_m:.1f})')

    ax.set_xlabel('$r$ [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=11)
    ax.set_title(f'Z = {Z} ($Z_1 Z_2$={Z**2}) [TEST]', fontsize=11,
                 color='#d62728', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle('Softened Coulomb: $E_{elec} \\approx -Z_{eff} \\cdot k_e / \\sqrt{r^2 + a^2}$\n'
             '1-param: $Z_{eff} = Z_1 Z_2$ fixed, learn $a$.  '
             '2-param: learn both $Z_{eff}$ and $a$.',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_softened_coulomb_ml.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_softened_coulomb_ml.png')
plt.close(fig)
