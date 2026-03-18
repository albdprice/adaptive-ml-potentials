"""
Augmented parabola with nonlinear ML models for parameter prediction.

E^m(r) = k/r + a·(r - r_c)² + b'·(r - r_c) + c'

Both constraints applied → 2 free params (k, a).

Compare ML approaches for predicting k(Z) and a(Z):
  - Ridge(Z)           — linear in Z, can't capture k~Z²
  - Ridge([Z, Z²])     — polynomial features, can capture quadratic
  - Ridge(Z₁Z₂)        — use charge product as feature
  - KRR poly kernel(Z)  — polynomial kernel, degree 2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

ke = 14.3996  # eV·Å
r_c = 10.0

# ================================================================
# Morse-based E_elec (ground truth)
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def E_elec_func(r, D_e, alpha, r_e, Z1Z2):
    return V_morse(r, D_e, alpha, r_e) - Z1Z2 * ke / r

# ================================================================
# Augmented parabola (both constraints → 2 params: k, a)
# ================================================================
def augmented_2param(r, k, a, Z1Z2):
    Zke = Z1Z2 * ke
    c_prime = -(Zke + k) / r_c
    b_prime = (Zke + k) / r_c**2
    dr = r - r_c
    return k / r + a * dr**2 + b_prime * dr + c_prime

def augmented_1param(r, a, Z1Z2):
    """k = -Z²ke fixed."""
    k = -Z1Z2 * ke
    return k / r + a * (r - r_c)**2

def fit_2param(r_grid, E_true, Z1Z2):
    def obj(params):
        k, a = params
        P = augmented_2param(r_grid, k, a, Z1Z2)
        return np.mean((E_true - P)**2)
    k0 = -Z1Z2 * ke
    result = minimize(obj, x0=[k0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x

def fit_1param(r_grid, E_true, Z1Z2):
    from scipy.optimize import minimize_scalar
    def obj(a):
        P = augmented_1param(r_grid, a, Z1Z2)
        return np.mean((E_true - P)**2)
    result = minimize_scalar(obj, bounds=(-50, 50), method='bounded')
    return result.x

# ================================================================
# Molecules
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
# Fit augmented parabola to each molecule
# ================================================================
k_fit = np.zeros(len(Z_values))
a_fit = np.zeros(len(Z_values))
a_1p_fit = np.zeros(len(Z_values))

for i, Z in enumerate(Z_values):
    p = molecules[Z]
    k_fit[i], a_fit[i] = fit_2param(r_grid, E_all[i], p['Z1Z2'])
    a_1p_fit[i] = fit_1param(r_grid, E_all[i], p['Z1Z2'])

# ================================================================
# ML feature matrices
# ================================================================
# Features for training
X_Z_tr = train_Z.reshape(-1, 1)                          # just Z
X_poly_tr = np.column_stack([train_Z, train_Z**2])        # [Z, Z²]
X_Z1Z2_tr = (train_Z**2).reshape(-1, 1)                  # Z₁Z₂ = Z²

X_Z_te = test_Z.reshape(-1, 1)
X_poly_te = np.column_stack([test_Z, test_Z**2])
X_Z1Z2_te = (test_Z**2).reshape(-1, 1)

# ================================================================
# Direct baseline
# ================================================================
mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(X_Z_tr, E_all[train_idx])
E_pred_direct = mdl_direct.predict(X_Z_te)
mae_direct = mean_absolute_error(E_all[test_idx].ravel(), E_pred_direct.ravel())

# ================================================================
# 1-param: k = -Z²ke fixed, learn a with Ridge(Z)
# ================================================================
mdl_a1 = Ridge(alpha=1e-6)
mdl_a1.fit(X_Z_tr, a_1p_fit[train_idx])
a1_pred = mdl_a1.predict(X_Z_te)

E_pred_1p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_1p[j] = augmented_1param(r_grid, a1_pred[j], Z**2)
mae_1p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_1p.ravel())

# ================================================================
# 2-param with different ML approaches
# ================================================================
methods = {}

# A) Ridge(Z) — baseline (can't capture Z²)
def predict_2param(mdl_k, mdl_a, X_te):
    k_pred = mdl_k.predict(X_te)
    a_pred = mdl_a.predict(X_te)
    E_pred = np.zeros((len(test_Z), n_grid))
    for j, Z in enumerate(test_Z):
        E_pred[j] = augmented_2param(r_grid, k_pred[j], a_pred[j], Z**2)
    return E_pred, k_pred, a_pred

# A) Ridge(Z)
mk_A = Ridge(alpha=1e-6); mk_A.fit(X_Z_tr, k_fit[train_idx])
ma_A = Ridge(alpha=1e-6); ma_A.fit(X_Z_tr, a_fit[train_idx])
E_A, k_A, a_A = predict_2param(mk_A, ma_A, X_Z_te)
mae_A = mean_absolute_error(E_all[test_idx].ravel(), E_A.ravel())
methods['Ridge(Z)'] = {'mae': mae_A, 'E': E_A, 'k': k_A, 'a': a_A}

# B) Ridge([Z, Z²])
mk_B = Ridge(alpha=1e-6); mk_B.fit(X_poly_tr, k_fit[train_idx])
ma_B = Ridge(alpha=1e-6); ma_B.fit(X_poly_tr, a_fit[train_idx])
E_B, k_B, a_B = predict_2param(mk_B, ma_B, X_poly_te)
mae_B = mean_absolute_error(E_all[test_idx].ravel(), E_B.ravel())
methods['Ridge([Z,Z²])'] = {'mae': mae_B, 'E': E_B, 'k': k_B, 'a': a_B}

# C) Ridge(Z₁Z₂)
mk_C = Ridge(alpha=1e-6); mk_C.fit(X_Z1Z2_tr, k_fit[train_idx])
ma_C = Ridge(alpha=1e-6); ma_C.fit(X_Z1Z2_tr, a_fit[train_idx])
E_C, k_C, a_C = predict_2param(mk_C, ma_C, X_Z1Z2_te)
mae_C = mean_absolute_error(E_all[test_idx].ravel(), E_C.ravel())
methods['Ridge(Z₁Z₂)'] = {'mae': mae_C, 'E': E_C, 'k': k_C, 'a': a_C}

# D) KRR polynomial kernel (degree 2)
mk_D = KernelRidge(alpha=1e-6, kernel='poly', degree=2)
mk_D.fit(X_Z_tr, k_fit[train_idx])
ma_D = KernelRidge(alpha=1e-6, kernel='poly', degree=2)
ma_D.fit(X_Z_tr, a_fit[train_idx])
k_D = mk_D.predict(X_Z_te)
a_D = ma_D.predict(X_Z_te)
E_D = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_D[j] = augmented_2param(r_grid, k_D[j], a_D[j], Z**2)
mae_D = mean_absolute_error(E_all[test_idx].ravel(), E_D.ravel())
methods['KRR poly(Z)'] = {'mae': mae_D, 'E': E_D, 'k': k_D, 'a': a_D}

# Morse adaptive
params_all = np.array([[molecules[Z]['D_e'], molecules[Z]['alpha'], molecules[Z]['r_e']]
                        for Z in Z_values])
pred_params = np.zeros((len(test_Z), 3))
for p in range(3):
    m = Ridge(alpha=1e-6)
    m.fit(X_Z_tr, params_all[train_idx, p])
    pred_params[:, p] = m.predict(X_Z_te)
E_pred_morse = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_morse[j] = E_elec_func(r_grid, pred_params[j, 0], pred_params[j, 1],
                                   pred_params[j, 2], Z**2)
mae_morse = mean_absolute_error(E_all[test_idx].ravel(), E_pred_morse.ravel())

# ================================================================
# Results
# ================================================================
print(f'{"="*80}')
print(f'ML EXTRAPOLATION: train Z=1-8, test Z=9-12')
print(f'Augmented parabola: E^m = k/r + a(r-rc)² + b\'(r-rc) + c\'')
print(f'{"="*80}')
print(f'{"Method":<55s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*75}')
print(f'{"Direct: Ridge(Z) → E(r) [50 targets]":<55s} | {mae_direct:8.2f} | {"base":>7s}')
print(f'{"1-param: k=-Z²ke, Ridge(Z) → a":<55s} | {mae_1p:8.2f} | {mae_1p/mae_direct:7.3f}x')
print(f'{"-"*75}')
print(f'  2-param (k, a) with different ML:')
for name, d in methods.items():
    print(f'{"  " + name + " → (k, a)":<55s} | {d["mae"]:8.2f} | {d["mae"]/mae_direct:7.3f}x')
print(f'{"-"*75}')
print(f'{"Morse: Ridge(Z) → (De,α,re) [3 targets]":<55s} | {mae_morse:8.2f} | {mae_morse/mae_direct:7.3f}x')
print(f'{"-"*75}')

# Per-molecule
print(f'\nPer-molecule MAE:')
print(f'{"Method":<30s} |', ' '.join(f'Z={Z:>4d}' for Z in test_Z))
for label, pred in [('Direct', E_pred_direct),
                     ('1p (k fixed)', E_pred_1p)] + \
                    [(name, d['E']) for name, d in methods.items()] + \
                    [('Morse', E_pred_morse)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<30s} |', ' '.join(f'{m:7.1f}' for m in maes))

# Parameter prediction quality
print(f'\nk prediction (true vs predicted):')
print(f'{"Z":>3s} {"k_true":>10s}', ' '.join(f'{"k_"+name[:8]:>12s}' for name in methods))
for j, Z in enumerate(test_Z):
    i = test_idx[j]
    vals = ' '.join(f'{d["k"][j]:12.1f}' for d in methods.values())
    print(f'{Z:3d} {k_fit[i]:10.1f} {vals}')

print(f'\na prediction (true vs predicted):')
print(f'{"Z":>3s} {"a_true":>10s}', ' '.join(f'{"a_"+name[:8]:>12s}' for name in methods))
for j, Z in enumerate(test_Z):
    i = test_idx[j]
    vals = ' '.join(f'{d["a"][j]:12.4f}' for d in methods.values())
    print(f'{Z:3d} {a_fit[i]:10.4f} {vals}')

# ================================================================
# FIGURE
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

colors = {'Ridge(Z)': 'gray', 'Ridge([Z,Z²])': 'blue',
          'Ridge(Z₁Z₂)': 'green', 'KRR poly(Z)': 'orange'}
styles = {'Ridge(Z)': ':', 'Ridge([Z,Z²])': '-',
          'Ridge(Z₁Z₂)': '--', 'KRR poly(Z)': '-.'}

# Top-left: k predictions
ax = axes[0, 0]
ax.plot(Z_values, k_fit, 'ko-', ms=7, lw=2, label='$k$ (fitted)', zorder=5)
for name, d in methods.items():
    ax.plot(test_Z, d['k'].ravel(), marker='^', ms=9, ls='none',
            color=colors[name], label=f'{name}', zorder=6)
ax.axvline(8.5, color='gray', ls=':', alpha=0.4)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$k$ [eV·Å]', fontsize=11)
ax.set_title('Parameter $k$ prediction', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Top-middle: a predictions
ax = axes[0, 1]
ax.plot(Z_values, a_fit, 'ko-', ms=7, lw=2, label='$a$ (fitted)', zorder=5)
for name, d in methods.items():
    ax.plot(test_Z, d['a'].ravel(), marker='^', ms=9, ls='none',
            color=colors[name], label=f'{name}', zorder=6)
ax.axvline(8.5, color='gray', ls=':', alpha=0.4)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$a$ [eV/Å²]', fontsize=11)
ax.set_title('Parameter $a$ prediction', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Top-right: MAE comparison bar chart
ax = axes[0, 2]
all_methods = ['Direct'] + ['1p (k fixed)'] + list(methods.keys()) + ['Morse']
all_maes = [mae_direct, mae_1p] + [d['mae'] for d in methods.values()] + [mae_morse]
all_colors = ['gray', 'red'] + [colors[n] for n in methods] + ['magenta']

bars = ax.bar(range(len(all_methods)), all_maes, color=all_colors, alpha=0.8)
ax.set_xticks(range(len(all_methods)))
ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('MAE [eV]', fontsize=11)
ax.set_title('Extrapolation MAE comparison', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add ratio labels on bars
for bar, mae in zip(bars, all_maes):
    if mae > 0.01:
        ratio = mae / mae_direct
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{ratio:.3f}x', ha='center', va='bottom', fontsize=7)

# Bottom row: test molecule predictions
for j in range(min(3, len(test_Z))):
    ax = axes[1, j]
    Z = test_Z[j]
    i = test_idx[j]
    ax.plot(r_grid, E_all[i], 'k-', lw=3, label='True', zorder=5)

    mae_d = mean_absolute_error(E_all[i], E_pred_direct[j])
    ax.plot(r_grid, E_pred_direct[j], color='gray', ls=':', lw=1.5,
            label=f'Direct ({mae_d:.0f})', alpha=0.7)

    mae_p1 = mean_absolute_error(E_all[i], E_pred_1p[j])
    ax.plot(r_grid, E_pred_1p[j], 'r-', lw=2,
            label=f'1p fixed ({mae_p1:.1f})')

    for name, d in methods.items():
        mae_m = mean_absolute_error(E_all[i], d['E'][j])
        ax.plot(r_grid, d['E'][j], color=colors[name], ls=styles[name], lw=1.5,
                label=f'{name} ({mae_m:.1f})')

    mae_m = mean_absolute_error(E_all[i], E_pred_morse[j])
    ax.plot(r_grid, E_pred_morse[j], 'm-.', lw=1.5,
            label=f'Morse ({mae_m:.1f})')

    ax.set_xlabel('$r$ [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=11)
    ax.set_title(f'Z = {Z} ($Z_1 Z_2$={Z**2}) [TEST]', fontsize=11,
                 color='#d62728', fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)

fig.suptitle('Augmented parabola: $k/r + a(r-r_c)^2 + b\'(r-r_c) + c\'$\n'
             'Effect of nonlinear ML for parameter prediction ($k$, $a$)',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_augmented_nonlinear_ml.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_augmented_nonlinear_ml.png')
plt.close(fig)
