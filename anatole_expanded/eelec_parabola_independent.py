"""
Two-parameter constrained parabola with independent learning of a and b.

E^m(r) = a·(r - r_c)² + b·(r - r_c) + c,   c = -Z₁Z₂·ke/r_c  (fixed)

Three approaches:
  A) Joint fit then separate ML:  fit (a,b) jointly per mol → Ridge a(Z), Ridge b(Z)
  B) End-to-end shared Ridge:    features [Z·(r-rc)², (r-rc)², Z·(r-rc), (r-rc)] → E_elec-c
     This learns a = w1·Z + w2 and b = w3·Z + w4 optimized for E_elec reconstruction.
  C) Independent fitting: fit a from curvature, b from slope, then ML separately

Also: direct multioutput and Morse adaptive as baselines.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

ke = 14.3996  # eV·Å
r_c = 10.0

# ================================================================
# Morse-based E_elec
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def V_morse_d1(r, D_e, alpha, r_e):
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha * (1 - x) * x

def V_morse_d2(r, D_e, alpha, r_e):
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha**2 * x * (2*x - 1)

def E_elec_func(r, D_e, alpha, r_e, Z1Z2):
    return V_morse(r, D_e, alpha, r_e) - Z1Z2 * ke / r

def E_elec_d1(r, D_e, alpha, r_e, Z1Z2):
    return V_morse_d1(r, D_e, alpha, r_e) + Z1Z2 * ke / r**2

def E_elec_d2(r, D_e, alpha, r_e, Z1Z2):
    return V_morse_d2(r, D_e, alpha, r_e) - 2 * Z1Z2 * ke / r**3

# ================================================================
# Molecules
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
dr = r_grid - r_c     # (r - r_c) for each grid point
dr2 = dr**2            # (r - r_c)² for each grid point

# Compute E_elec
E_all = np.zeros((len(Z_values), n_grid))
for i, Z in enumerate(Z_values):
    p = molecules[Z]
    E_all[i] = E_elec_func(r_grid, p['D_e'], p['alpha'], p['r_e'], p['Z1Z2'])

# c values (fixed by nuclear repulsion)
c_all = np.array([-Z**2 * ke / r_c for Z in Z_values])

train_idx = np.where(np.isin(Z_values, train_Z))[0]
test_idx = np.where(np.isin(Z_values, test_Z))[0]

# ================================================================
# Method A: Joint fit then separate ML (what we had)
# ================================================================
def fit_joint(r_grid, E_true, Z1Z2):
    c = -Z1Z2 * ke / r_c
    def obj(params):
        a, b = params
        P = a * dr2 + b * dr + c
        return np.mean((E_true - P)**2)
    result = minimize(obj, x0=[-1.0, 1.0], method='Nelder-Mead')
    return result.x

a_joint = np.zeros(len(Z_values))
b_joint = np.zeros(len(Z_values))
for i, Z in enumerate(Z_values):
    a_joint[i], b_joint[i] = fit_joint(r_grid, E_all[i], Z**2)

# ML: separate Ridge for a(Z) and b(Z)
X_tr = train_Z.reshape(-1, 1)
X_te = test_Z.reshape(-1, 1)

mdl_a_joint = Ridge(alpha=1e-6)
mdl_a_joint.fit(X_tr, a_joint[train_idx])
a_pred_joint = mdl_a_joint.predict(X_te)

mdl_b_joint = Ridge(alpha=1e-6)
mdl_b_joint.fit(X_tr, b_joint[train_idx])
b_pred_joint = mdl_b_joint.predict(X_te)

E_pred_A = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_A[j] = a_pred_joint[j] * dr2 + b_pred_joint[j] * dr + c_all[test_idx[j]]
mae_A = mean_absolute_error(E_all[test_idx].ravel(), E_pred_A.ravel())

# ================================================================
# Method B: End-to-end shared Ridge
# Build features that encode the parabola structure:
#   E_elec - c = a*(r-rc)² + b*(r-rc)
#   If a = w1*Z + w2 and b = w3*Z + w4:
#   E_elec - c = w1*Z*(r-rc)² + w2*(r-rc)² + w3*Z*(r-rc) + w4*(r-rc)
#
# Features: [Z·dr², dr², Z·dr, dr]   (4 total)
# Target:   E_elec - c
# ================================================================
def make_shared_features(Z_arr, dr, dr2):
    """Build feature matrix for end-to-end parabola Ridge."""
    n_mol = len(Z_arr)
    n_d = len(dr)
    X = np.zeros((n_mol * n_d, 4))
    for i, Z in enumerate(Z_arr):
        for j in range(n_d):
            X[i * n_d + j] = [Z * dr2[j], dr2[j], Z * dr[j], dr[j]]
    return X

X_shared_tr = make_shared_features(train_Z, dr, dr2)
y_shared_tr = (E_all[train_idx] - c_all[train_idx, np.newaxis]).ravel()

X_shared_te = make_shared_features(test_Z, dr, dr2)

mdl_shared = Ridge(alpha=1e-6)
mdl_shared.fit(X_shared_tr, y_shared_tr)
pred_shared = mdl_shared.predict(X_shared_te).reshape(len(test_Z), n_grid)
E_pred_B = pred_shared + c_all[test_idx, np.newaxis]
mae_B = mean_absolute_error(E_all[test_idx].ravel(), E_pred_B.ravel())

# Extract learned a(Z), b(Z) from shared model
w = mdl_shared.coef_
b_intercept = mdl_shared.intercept_
# E - c = w[0]*Z*dr² + w[1]*dr² + w[2]*Z*dr + w[3]*dr + intercept
# = (w[0]*Z + w[1])*dr² + (w[2]*Z + w[3])*dr + intercept
# So: a(Z) = w[0]*Z + w[1],  b(Z) = w[2]*Z + w[3]
# intercept should be ~0 (absorbed into c)
print(f'Shared model weights:')
print(f'  a(Z) = {w[0]:.4f}·Z + {w[1]:.4f}')
print(f'  b(Z) = {w[2]:.4f}·Z + {w[3]:.4f}')
print(f'  intercept = {b_intercept:.4f}')

a_shared = w[0] * test_Z + w[1]
b_shared = w[2] * test_Z + w[3]

# ================================================================
# Method C: Independent parameter extraction
# a from second derivative, b from first derivative at a reference point
# ================================================================
r_mid = 2.0  # reference point for derivative matching

a_indep = np.zeros(len(Z_values))
b_indep = np.zeros(len(Z_values))
for i, Z in enumerate(Z_values):
    p = molecules[Z]
    # a = E_elec''(r_mid) / 2
    a_indep[i] = E_elec_d2(r_mid, p['D_e'], p['alpha'], p['r_e'], p['Z1Z2']) / 2.0
    # b from: at r_mid, slope of parabola = E_elec'(r_mid)
    # P'(r_mid) = 2a*(r_mid - r_c) + b = E_elec'(r_mid)
    # b = E_elec'(r_mid) - 2a*(r_mid - r_c)
    slope = E_elec_d1(r_mid, p['D_e'], p['alpha'], p['r_e'], p['Z1Z2'])
    b_indep[i] = slope - 2 * a_indep[i] * (r_mid - r_c)

mdl_a_ind = Ridge(alpha=1e-6)
mdl_a_ind.fit(X_tr, a_indep[train_idx])
a_pred_ind = mdl_a_ind.predict(X_te)

mdl_b_ind = Ridge(alpha=1e-6)
mdl_b_ind.fit(X_tr, b_indep[train_idx])
b_pred_ind = mdl_b_ind.predict(X_te)

E_pred_C = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_C[j] = a_pred_ind[j] * dr2 + b_pred_ind[j] * dr + c_all[test_idx[j]]
mae_C = mean_absolute_error(E_all[test_idx].ravel(), E_pred_C.ravel())

# ================================================================
# Baselines
# ================================================================
# Direct multioutput
mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(X_tr, E_all[train_idx])
E_pred_direct = mdl_direct.predict(X_te)
mae_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_direct.ravel())

# Morse adaptive
params_all = np.array([[molecules[Z]['D_e'], molecules[Z]['alpha'], molecules[Z]['r_e']]
                        for Z in Z_values])
pred_morse = np.zeros((len(test_Z), 3))
for p in range(3):
    m = Ridge(alpha=1e-6)
    m.fit(X_tr, params_all[train_idx, p])
    pred_morse[:, p] = m.predict(X_te)
E_pred_morse = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_morse[j] = E_elec_func(r_grid, pred_morse[j, 0], pred_morse[j, 1],
                                   pred_morse[j, 2], Z**2)
mae_morse = mean_absolute_error(E_all[test_idx].ravel(), E_pred_morse.ravel())

# ================================================================
# Results
# ================================================================
print(f'\n{"="*80}')
print(f'ML EXTRAPOLATION: train Z=1-8, test Z=9-12')
print(f'{"="*80}')
print(f'{"Method":<55s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*75}')
print(f'{"Direct multioutput: Z → E(r) [50 targets]":<55s} | {mae_direct:8.2f} | {"base":>7s}')
print(f'{"A) Joint fit → separate ML [2 params]":<55s} | {mae_A:8.2f} | {mae_A/mae_direct:7.3f}x')
print(f'{"B) End-to-end shared Ridge [4 weights]":<55s} | {mae_B:8.2f} | {mae_B/mae_direct:7.3f}x')
print(f'{"C) Independent fit (deriv matching) [2 params]":<55s} | {mae_C:8.2f} | {mae_C/mae_direct:7.3f}x')
print(f'{"Morse adaptive [3 params]":<55s} | {mae_morse:8.2f} | {mae_morse/mae_direct:7.3f}x')
print(f'{"-"*75}')

# Per-molecule
print(f'\nPer-molecule MAE:')
print(f'{"Method":<35s} |', ' '.join(f'Z={Z:>4d}' for Z in test_Z))
for label, pred in [('Direct (50)', E_pred_direct),
                     ('A) Joint→separate', E_pred_A),
                     ('B) End-to-end shared', E_pred_B),
                     ('C) Independent fit', E_pred_C),
                     ('Morse (3 params)', E_pred_morse)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<35s} |', ' '.join(f'{m:7.1f}' for m in maes))

# Parameter comparison
print(f'\nLearned parameters for test molecules:')
print(f'{"Z":>3s} | {"a_joint":>8s} {"b_joint":>8s} | {"a_shared":>8s} {"b_shared":>8s} | '
      f'{"a_indep":>8s} {"b_indep":>8s} | {"a_true":>8s} {"b_true":>8s}')
print(f'{"-"*90}')
for j, Z in enumerate(test_Z):
    i = test_idx[j]
    print(f'{Z:3d} | {a_pred_joint[j]:8.3f} {b_pred_joint[j]:8.3f} | '
          f'{a_shared[j]:8.3f} {b_shared[j]:8.3f} | '
          f'{a_pred_ind[j]:8.3f} {b_pred_ind[j]:8.3f} | '
          f'{a_joint[i]:8.3f} {b_joint[i]:8.3f}')

# ================================================================
# FIGURE
# ================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

# Top: parameter trends
ax = axes[0, 0]
ax.plot(Z_values, a_joint, 'ko-', ms=6, label='Joint fit $a$')
ax.plot(Z_values, a_indep, 'bs--', ms=5, label='Indep fit $a$')
ax.plot(test_Z, a_shared, 'r^', ms=8, label='Shared model $a$')
ax.axvline(8.5, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Z'); ax.set_ylabel('$a$')
ax.set_title('Curvature $a$ vs Z', fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(Z_values, b_joint, 'ko-', ms=6, label='Joint fit $b$')
ax.plot(Z_values, b_indep, 'bs--', ms=5, label='Indep fit $b$')
ax.plot(test_Z, b_shared, 'r^', ms=8, label='Shared model $b$')
ax.axvline(8.5, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Z'); ax.set_ylabel('$b$')
ax.set_title('Slope $b$ vs Z', fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Summary bar chart
ax = axes[0, 2]
methods = ['Direct\n(50)', 'A) Joint\n→sep (2)', 'B) E2E\nshared (4w)',
           'C) Indep\nfit (2)', 'Morse\n(3)']
maes_all = [mae_direct, mae_A, mae_B, mae_C, mae_morse]
colors = ['steelblue', 'coral', 'gold', 'mediumpurple', 'forestgreen']
ax.bar(methods, maes_all, color=colors)
for i, (m, v) in enumerate(zip(methods, maes_all)):
    ax.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('MAE [eV]')
ax.set_title('Extrapolation MAE comparison', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Fit quality for each method
ax = axes[0, 3]
for i, Z in enumerate(Z_values):
    p = molecules[Z]
    E_true = E_all[i]
    # Joint fit quality
    P_j = a_joint[i] * dr2 + b_joint[i] * dr + c_all[i]
    mae_j = mean_absolute_error(E_true, P_j)
    # Independent fit quality
    P_i = a_indep[i] * dr2 + b_indep[i] * dr + c_all[i]
    mae_i = mean_absolute_error(E_true, P_i)
    if i == 0:
        ax.plot(Z, mae_j, 'ko', ms=6, label='Joint fit')
        ax.plot(Z, mae_i, 'bs', ms=5, label='Indep fit')
    else:
        ax.plot(Z, mae_j, 'ko', ms=6)
        ax.plot(Z, mae_i, 'bs', ms=5)
ax.axvline(8.5, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Z'); ax.set_ylabel('Fit MAE [eV]')
ax.set_title('Fit quality per molecule', fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Bottom: predictions for test molecules
for j in range(4):
    ax = axes[1, j]
    Z = test_Z[j]
    i = test_idx[j]

    ax.plot(r_grid, E_all[i], 'k-', lw=2.5, label='True', zorder=5)

    for label, pred, style, lw in [
        ('Direct', E_pred_direct[j], 'g:', 1.5),
        ('A) Joint→sep', E_pred_A[j], 'r--', 1.5),
        ('B) E2E shared', E_pred_B[j], 'y-', 2),
        ('C) Indep', E_pred_C[j], 'b-.', 1.5),
        ('Morse', E_pred_morse[j], 'm:', 1.5),
    ]:
        mae_j = mean_absolute_error(E_all[i], pred)
        ax.plot(r_grid, pred, style, lw=lw,
                label=f'{label} ({mae_j:.0f})')

    ax.set_xlabel('$r$ [Å]', fontsize=10)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=10)
    ax.set_title(f'Z={Z} (Z₁Z₂={Z**2})', fontsize=11,
                 color='#d62728', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Parabola $E^m = a(r-r_c)^2 + b(r-r_c) + c$, $c=-Z_1Z_2 k_e/r_c$, $r_c={r_c}$ Å\n'
             f'A) Joint fit → separate ML    B) End-to-end shared    C) Independent derivative matching',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_parabola_independent_learning.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_parabola_independent_learning.png')
plt.close(fig)
