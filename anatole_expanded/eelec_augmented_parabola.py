"""
Augmented parabola (Gemini suggestion): add k/r to capture singularity.

    E^m(r) = k/r + a·(r - r_c)² + b'·(r - r_c) + c'

Constraints at r_c (E_total → 0 at large r):
  Value:  k/r_c + c' = -Z²ke/r_c   →  c' = -(Z²ke + k)/r_c
  Slope: -k/r_c² + b' = Z²ke/r_c²  →  b' = (Z²ke + k)/r_c²

Three versions:
  1-param: k = -Z²ke (known nuclear charge), both constraints → only a free
  2-param: both constraints → k and a free
  3-param: value constraint only → k, a, b' free
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
r_c = 10.0    # Å

# ================================================================
# Morse-based E_elec (ground truth)
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def E_elec_func(r, D_e, alpha, r_e, Z1Z2):
    return V_morse(r, D_e, alpha, r_e) - Z1Z2 * ke / r

# ================================================================
# Augmented parabola: E^m = k/r + a(r-rc)² + b'(r-rc) + c'
# ================================================================
def augmented_1param(r, a, Z1Z2):
    """k = -Z²ke fixed. Both constraints → c'=0, b'=0.
    E^m = -Z²ke/r + a·(r-rc)²"""
    k = -Z1Z2 * ke
    return k / r + a * (r - r_c)**2

def augmented_2param(r, k, a, Z1Z2):
    """Both constraints. k and a free.
    c' = -(Z²ke + k)/rc,  b' = (Z²ke + k)/rc²"""
    Zke = Z1Z2 * ke
    c_prime = -(Zke + k) / r_c
    b_prime = (Zke + k) / r_c**2
    dr = r - r_c
    return k / r + a * dr**2 + b_prime * dr + c_prime

def augmented_3param(r, k, a, b_prime, Z1Z2):
    """Value constraint only. k, a, b' free.
    c' = -(Z²ke + k)/rc"""
    Zke = Z1Z2 * ke
    c_prime = -(Zke + k) / r_c
    dr = r - r_c
    return k / r + a * dr**2 + b_prime * dr + c_prime

# ================================================================
# Fitting functions
# ================================================================
def fit_1param(r_grid, E_true, Z1Z2):
    def obj(a):
        P = augmented_1param(r_grid, a, Z1Z2)
        return np.mean((E_true - P)**2)
    result = minimize_scalar(obj, bounds=(-50, 50), method='bounded')
    return result.x

def fit_2param(r_grid, E_true, Z1Z2):
    def obj(params):
        k, a = params
        P = augmented_2param(r_grid, k, a, Z1Z2)
        return np.mean((E_true - P)**2)
    k0 = -Z1Z2 * ke
    result = minimize(obj, x0=[k0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x

def fit_3param(r_grid, E_true, Z1Z2):
    def obj(params):
        k, a, b_prime = params
        P = augmented_3param(r_grid, k, a, b_prime, Z1Z2)
        return np.mean((E_true - P)**2)
    k0 = -Z1Z2 * ke
    result = minimize(obj, x0=[k0, 0.0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x

# ================================================================
# Define molecules (same physical params as other scripts)
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
# Fit all versions
# ================================================================
# 1-param: only a
a_1p = np.zeros(len(Z_values))
fit_mae_1p = np.zeros(len(Z_values))

# 2-param: k, a
k_2p = np.zeros(len(Z_values))
a_2p = np.zeros(len(Z_values))
fit_mae_2p = np.zeros(len(Z_values))

# 3-param: k, a, b'
k_3p = np.zeros(len(Z_values))
a_3p = np.zeros(len(Z_values))
b_3p = np.zeros(len(Z_values))
fit_mae_3p = np.zeros(len(Z_values))

print(f'{"="*95}')
print(f'Augmented parabola: E^m = k/r + a(r-rc)² + b\'(r-rc) + c\'')
print(f'{"="*95}')
print(f'{"Z":>3s} {"Z1Z2":>5s} | {"a(1p)":>8s} {"MAE":>6s} | '
      f'{"k(2p)":>8s} {"a(2p)":>8s} {"MAE":>6s} | '
      f'{"k(3p)":>8s} {"a(3p)":>8s} {"b(3p)":>8s} {"MAE":>6s}')
print(f'{"-"*95}')

for i, Z in enumerate(Z_values):
    p = molecules[Z]
    Z1Z2 = p['Z1Z2']

    # 1-param
    a_1p[i] = fit_1param(r_grid, E_all[i], Z1Z2)
    P1 = augmented_1param(r_grid, a_1p[i], Z1Z2)
    fit_mae_1p[i] = mean_absolute_error(E_all[i], P1)

    # 2-param
    k_2p[i], a_2p[i] = fit_2param(r_grid, E_all[i], Z1Z2)
    P2 = augmented_2param(r_grid, k_2p[i], a_2p[i], Z1Z2)
    fit_mae_2p[i] = mean_absolute_error(E_all[i], P2)

    # 3-param
    k_3p[i], a_3p[i], b_3p[i] = fit_3param(r_grid, E_all[i], Z1Z2)
    P3 = augmented_3param(r_grid, k_3p[i], a_3p[i], b_3p[i], Z1Z2)
    fit_mae_3p[i] = mean_absolute_error(E_all[i], P3)

    marker = '  TEST' if Z in test_Z else ''
    print(f'{Z:3d} {Z1Z2:5d} | {a_1p[i]:8.3f} {fit_mae_1p[i]:6.2f} | '
          f'{k_2p[i]:8.1f} {a_2p[i]:8.3f} {fit_mae_2p[i]:6.2f} | '
          f'{k_3p[i]:8.1f} {a_3p[i]:8.3f} {b_3p[i]:8.3f} {fit_mae_3p[i]:6.2f}{marker}')

# ================================================================
# ML predictions
# ================================================================
X_tr = train_Z.reshape(-1, 1)
X_te = test_Z.reshape(-1, 1)

# Direct baseline
mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(X_tr, E_all[train_idx])
E_pred_direct = mdl_direct.predict(X_te)
mae_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_direct.ravel())

# 1-param: Z → a (k = -Z²ke known)
mdl_a1 = Ridge(alpha=1e-6)
mdl_a1.fit(X_tr, a_1p[train_idx])
a1_pred = mdl_a1.predict(X_te)

E_pred_1p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_1p[j] = augmented_1param(r_grid, a1_pred[j], Z**2)
mae_1p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_1p.ravel())

# 2-param: Z → (k, a), both constraints
mdl_k2 = Ridge(alpha=1e-6)
mdl_k2.fit(X_tr, k_2p[train_idx])
k2_pred = mdl_k2.predict(X_te)

mdl_a2 = Ridge(alpha=1e-6)
mdl_a2.fit(X_tr, a_2p[train_idx])
a2_pred = mdl_a2.predict(X_te)

E_pred_2p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_2p[j] = augmented_2param(r_grid, k2_pred[j], a2_pred[j], test_Z[j]**2)
mae_2p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_2p.ravel())

# 3-param: Z → (k, a, b'), value constraint only
mdl_k3 = Ridge(alpha=1e-6)
mdl_k3.fit(X_tr, k_3p[train_idx])
k3_pred = mdl_k3.predict(X_te)

mdl_a3 = Ridge(alpha=1e-6)
mdl_a3.fit(X_tr, a_3p[train_idx])
a3_pred = mdl_a3.predict(X_te)

mdl_b3 = Ridge(alpha=1e-6)
mdl_b3.fit(X_tr, b_3p[train_idx])
b3_pred = mdl_b3.predict(X_te)

E_pred_3p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_3p[j] = augmented_3param(r_grid, k3_pred[j], a3_pred[j],
                                     b3_pred[j], test_Z[j]**2)
mae_3p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_3p.ravel())

# Morse adaptive
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
print(f'{"Aug. parabola 1p: Z → a [k=-Z²ke fixed]":<55s} | {mae_1p:8.2f} | {mae_1p/mae_direct:7.3f}x')
print(f'{"Aug. parabola 2p: Z → (k, a) [both constraints]":<55s} | {mae_2p:8.2f} | {mae_2p/mae_direct:7.3f}x')
print(f'{"Aug. parabola 3p: Z → (k, a, b) [value constr.]":<55s} | {mae_3p:8.2f} | {mae_3p/mae_direct:7.3f}x')
print(f'{"Morse adaptive: Z → (De,α,re) [3 targets]":<55s} | {mae_morse:8.2f} | {mae_morse/mae_direct:7.3f}x')
print(f'{"-"*75}')

# Per-molecule
print(f'\nPer-molecule MAE:')
print(f'{"Method":<30s} |', ' '.join(f'Z={Z:>4d}' for Z in test_Z))
for label, pred in [('Direct (50 targets)', E_pred_direct),
                     ('Aug. parab. 1p (a)', E_pred_1p),
                     ('Aug. parab. 2p (k,a)', E_pred_2p),
                     ('Aug. parab. 3p (k,a,b)', E_pred_3p),
                     ('Morse (De,α,re)', E_pred_morse)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<30s} |', ' '.join(f'{m:7.1f}' for m in maes))

# Parameter trends
print(f'\nParameter k values (should be close to -Z²ke):')
print(f'{"Z":>3s} {"Z1Z2":>5s} {"−Z²ke":>10s} {"k(2p)":>10s} {"k(3p)":>10s} {"k/(-Z²ke)":>10s}')
for i, Z in enumerate(Z_values):
    Z1Z2 = Z**2
    k_nuc = -Z1Z2 * ke
    print(f'{Z:3d} {Z1Z2:5d} {k_nuc:10.1f} {k_2p[i]:10.1f} {k_3p[i]:10.1f} {k_2p[i]/k_nuc:10.4f}')

# ================================================================
# FIGURE: same layout as fig_anatole_parabola_ml.png
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top-left: a(1p) vs Z
ax = axes[0, 0]
ax.plot(train_Z, a_1p[train_idx], 'bo', ms=10, label='Train', zorder=5)
ax.plot(test_Z, a_1p[test_idx], 'rs', ms=10, label='Test (true)', zorder=5)
ax.plot(test_Z, a1_pred.ravel(), 'r^', ms=10, label='Test (pred)', zorder=5)
Z_dense = np.linspace(0, 13, 100)
ax.plot(Z_dense, mdl_a1.predict(Z_dense.reshape(-1, 1)), 'b-', alpha=0.4)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$a$ [eV/Å²]', fontsize=11)
ax.set_title('1-param: $a$ vs Z\n($k = -Z^2 k_e$ fixed)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Top-middle: k(2p) and a(2p) vs Z
ax = axes[0, 1]
ax.plot(Z_values, k_2p, 'bo-', ms=6, label='$k$ (fitted)')
ax.plot(test_Z, k2_pred.ravel(), 'b^', ms=8, label='$k$ (pred)')
ax2_twin = ax.twinx()
ax2_twin.plot(Z_values, a_2p, 'rs-', ms=6, label='$a$ (fitted)')
ax2_twin.plot(test_Z, a2_pred.ravel(), 'r^', ms=8, label='$a$ (pred)')
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$k$ [eV·Å]', fontsize=11, color='b')
ax2_twin.set_ylabel('$a$ [eV/Å²]', fontsize=11, color='r')
ax.set_title('2-param: $k$ and $a$ vs Z', fontsize=11)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7)
ax.grid(True, alpha=0.3)

# Top-right: fit quality
ax = axes[0, 2]
x = np.arange(len(Z_values))
w = 0.25
ax.bar(x - w, fit_mae_1p, w, label='1p ($a$ only)', color='steelblue')
ax.bar(x, fit_mae_2p, w, label='2p ($k$, $a$)', color='coral')
ax.bar(x + w, fit_mae_3p, w, label='3p ($k$, $a$, $b$)', color='mediumpurple')
ax.set_xticks(x)
ax.set_xticklabels(Z_values)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('Fit MAE [eV]', fontsize=11)
ax.set_title('Fit quality (lower = better shape match)', fontsize=11)
ax.legend(fontsize=8)
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
            label=f'1p: $a$ ({mae_p1:.1f})')

    mae_p2 = mean_absolute_error(E_all[i], E_pred_2p[j])
    ax.plot(r_grid, E_pred_2p[j], 'b--', lw=1.5,
            label=f'2p: $k$,$a$ ({mae_p2:.1f})')

    mae_p3 = mean_absolute_error(E_all[i], E_pred_3p[j])
    ax.plot(r_grid, E_pred_3p[j], 'c-.', lw=1.5,
            label=f'3p: $k$,$a$,$b$ ({mae_p3:.1f})')

    mae_m = mean_absolute_error(E_all[i], E_pred_morse[j])
    ax.plot(r_grid, E_pred_morse[j], 'm-.', lw=1.5,
            label=f'Morse ({mae_m:.1f})')

    ax.set_xlabel('$r$ [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=11)
    ax.set_title(f'Z = {Z} ($Z_1 Z_2$={Z**2}) [TEST]', fontsize=11,
                 color='#d62728', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle('Augmented parabola: $E^m = k/r + a(r\\!-\\!r_c)^2 + b\'(r\\!-\\!r_c) + c\'$\n'
             'Constraints: $c\' = -(Z^2 k_e + k)/r_c$,  $b\' = (Z^2 k_e + k)/r_c^2$',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_augmented_parabola_ml.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_augmented_parabola_ml.png')
plt.close(fig)
