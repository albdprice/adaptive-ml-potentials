"""
Anatole's constrained parabola model for E_elec.

Model:  E^m(r) = a·(r - r_c)² + b·(r - r_c) + c

At r_c (large distance, e.g. 10 Å), E_total = E_elec + E_nuc ≈ 0.
This gives two constraints:

  Constraint 1 (value):   c = -E_nuc(r_c) = -Z₁Z₂·ke/r_c
  Constraint 2 (slope):   b = -∂_r E_nuc|_{r_c} = Z₁Z₂·ke/r_c²

With both constraints: only a is free (1 param per molecule).
With only value constraint: a and b are free (2 params per molecule).

We test both on H₂ first (visualization), then across molecules (ML).
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
r_c = 10.0    # Å — large-r reference

# ================================================================
# Morse-based E_elec
# ================================================================
def V_morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def E_elec_func(r, D_e, alpha, r_e, Z1Z2):
    return V_morse(r, D_e, alpha, r_e) - Z1Z2 * ke / r

def E_nuc(r, Z1Z2):
    return Z1Z2 * ke / r

def E_nuc_d1(r, Z1Z2):
    return -Z1Z2 * ke / r**2

# ================================================================
# Constrained parabola
# ================================================================
def parabola_both_constraints(r, a, Z1Z2):
    """E^m = a·(r-r_c)² + b·(r-r_c) + c  with b,c fixed by nuclear."""
    c = -E_nuc(r_c, Z1Z2)          # = -Z1Z2·ke/r_c
    b = -E_nuc_d1(r_c, Z1Z2)       # = Z1Z2·ke/r_c²
    dr = r - r_c
    return a * dr**2 + b * dr + c

def parabola_value_constraint(r, a, b, Z1Z2):
    """E^m = a·(r-r_c)² + b·(r-r_c) + c  with only c fixed."""
    c = -E_nuc(r_c, Z1Z2)
    dr = r - r_c
    return a * dr**2 + b * dr + c

def fit_1param(r_grid, E_true, Z1Z2):
    """Fit with both constraints: only a is free."""
    def obj(a):
        P = parabola_both_constraints(r_grid, a, Z1Z2)
        return np.mean((E_true - P)**2)
    result = minimize_scalar(obj, bounds=(-50, 50), method='bounded')
    return result.x

def fit_2param(r_grid, E_true, Z1Z2):
    """Fit with value constraint only: a and b free."""
    def obj(params):
        a, b = params
        P = parabola_value_constraint(r_grid, a, b, Z1Z2)
        return np.mean((E_true - P)**2)
    result = minimize(obj, x0=[-1.0, 1.0], method='Nelder-Mead')
    return result.x

# ================================================================
# H₂ visualization first
# ================================================================
D_e_H2, alpha_H2, r_e_H2, Z1Z2_H2 = 4.75, 1.94, 0.74, 1
r_plot = np.linspace(0.35, 12.0, 1000)
E_H2 = E_elec_func(r_plot, D_e_H2, alpha_H2, r_e_H2, Z1Z2_H2)

# Constraints at r_c
c_H2 = -E_nuc(r_c, Z1Z2_H2)   # value at r_c
b_H2 = -E_nuc_d1(r_c, Z1Z2_H2) # slope at r_c

print(f'H2 constraints at r_c = {r_c} Å:')
print(f'  c = -E_nuc(r_c) = {c_H2:.4f} eV')
print(f'  b = Z1Z2·ke/r_c² = {b_H2:.4f} eV/Å')
print(f'  E_elec(r_c) = {E_elec_func(r_c, D_e_H2, alpha_H2, r_e_H2, Z1Z2_H2):.4f} eV')

# Fit H2 with both constraints
r_fit = np.linspace(0.5, 5.0, 50)
E_H2_fit = E_elec_func(r_fit, D_e_H2, alpha_H2, r_e_H2, Z1Z2_H2)
a_H2_1p = fit_1param(r_fit, E_H2_fit, Z1Z2_H2)
a_H2_2p, b_H2_2p = fit_2param(r_fit, E_H2_fit, Z1Z2_H2)

print(f'\nH2 fits (r ∈ [0.5, 5.0] Å):')
print(f'  1-param (a only):  a = {a_H2_1p:.4f}')
P_1p = parabola_both_constraints(r_fit, a_H2_1p, Z1Z2_H2)
print(f'    MAE = {mean_absolute_error(E_H2_fit, P_1p):.3f} eV')
print(f'  2-param (a, b):    a = {a_H2_2p:.4f}, b = {b_H2_2p:.4f} (vs constrained b = {b_H2:.4f})')
P_2p = parabola_value_constraint(r_fit, a_H2_2p, b_H2_2p, Z1Z2_H2)
print(f'    MAE = {mean_absolute_error(E_H2_fit, P_2p):.3f} eV')

# ================================================================
# FIGURE 1: H₂ — show the parabola on E_elec
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# Panel 1: E_elec + both constraint parabola
ax = axes[0]
ax.plot(r_plot, E_H2, 'k-', lw=2.5, label='$E_{elec}(r)$', zorder=5)

# Show parabola with both constraints
P_1p_plot = parabola_both_constraints(r_plot, a_H2_1p, Z1Z2_H2)
ax.plot(r_plot, P_1p_plot, 'r-', lw=2, alpha=0.8,
        label=f'1-param: $a$={a_H2_1p:.3f}')

# Show parabola with value constraint only
P_2p_plot = parabola_value_constraint(r_plot, a_H2_2p, b_H2_2p, Z1Z2_H2)
ax.plot(r_plot, P_2p_plot, 'b--', lw=1.5, alpha=0.8,
        label=f'2-param: $a$={a_H2_2p:.3f}, $b$={b_H2_2p:.3f}')

# Mark the constraint point at r_c
ax.plot(r_c, c_H2, 'go', ms=12, zorder=6, label=f'$(r_c, -E^{{nuc}})$ = ({r_c}, {c_H2:.2f})')

# Show -E_nuc for reference
ax.plot(r_plot, -E_nuc(r_plot, Z1Z2_H2), 'g:', lw=1.5, alpha=0.5, label='$-E_{nuc}(r)$')

ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$E_{elec}$ [eV]', fontsize=12)
ax.set_title('H$_2$: constrained parabola fit\n'
             '$E^m = a(r\\!-\\!r_c)^2 + b(r\\!-\\!r_c) + c$', fontsize=11)
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(0.3, 12)
ax.set_ylim(-35, 5)
ax.grid(True, alpha=0.3)

# Panel 2: Zoom to fitting region
ax = axes[1]
ax.plot(r_fit, E_H2_fit, 'k-', lw=2.5, label='$E_{elec}$', zorder=5)
ax.plot(r_fit, P_1p, 'r-', lw=2, label=f'1-param ({mean_absolute_error(E_H2_fit, P_1p):.2f} eV)')
ax.plot(r_fit, P_2p, 'b--', lw=1.5, label=f'2-param ({mean_absolute_error(E_H2_fit, P_2p):.2f} eV)')

ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$E_{elec}$ [eV]', fontsize=12)
ax.set_title('Zoom: fitting region [0.5, 5.0] Å', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Show a FAMILY of parabolas with different a
ax = axes[2]
ax.plot(r_plot, E_H2, 'k-', lw=2.5, label='$E_{elec}(r)$', zorder=5)

# Family of parabolas all passing through (r_c, c) with same slope b
a_values = np.linspace(-3.0, -0.05, 8)
cmap = plt.cm.coolwarm
for i, a_val in enumerate(a_values):
    P = parabola_both_constraints(r_plot, a_val, Z1Z2_H2)
    t = i / (len(a_values) - 1)
    ax.plot(r_plot, P, color=cmap(t), lw=1.5, alpha=0.6,
            label=f'$a$={a_val:.2f}' if i % 2 == 0 else '')

ax.plot(r_c, c_H2, 'go', ms=10, zorder=6)
ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('Energy [eV]', fontsize=12)
ax.set_title('Family of parabolas (varying $a$)\n'
             'All anchored at $r_c$ with same slope', fontsize=11)
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(0.3, 12)
ax.set_ylim(-40, 5)
ax.grid(True, alpha=0.3)

fig.suptitle(f'H$_2$: $E^m = a(r-r_c)^2 + b(r-r_c) + c$,  '
             f'$c = -E^{{nuc}}(r_c)$,  $b = Z_1 Z_2 k_e / r_c^2$,  '
             f'$r_c = {r_c}$ Å',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_anatole_parabola_H2.png'), dpi=150, bbox_inches='tight')
print('Saved fig_anatole_parabola_H2.png')
plt.close(fig)

# ================================================================
# Multi-molecule ML experiment
# ================================================================
Z_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Physical parameter mapping: ensure E_elec always dives to -∞ at small r.
# The Morse wall D_e*exp(2*alpha*r_e) must stay BELOW Z²*ke/r at small r.
# So D_e and alpha must grow gently relative to Z².
molecules = {}
for Z in Z_values:
    D_e = 0.5 + 0.3 * Z              # gentle growth (not Z²)
    alpha = 1.5                        # constant width
    r_e = 0.8 + 0.1 * Z              # linear growth
    Z1Z2 = Z**2                        # homonuclear
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

# Fit constrained parabolas
a_1p = np.zeros(len(Z_values))     # 1-param fit (both constraints)
a_2p = np.zeros(len(Z_values))     # 2-param fit (value constraint only)
b_2p = np.zeros(len(Z_values))
fit_mae_1p = np.zeros(len(Z_values))
fit_mae_2p = np.zeros(len(Z_values))

print(f'\n{"="*80}')
print(f'Constrained parabola fits: r_c = {r_c} Å')
print(f'{"="*80}')
print(f'{"Z":>3s} {"Z1Z2":>5s} | {"a(1p)":>8s} {"MAE 1p":>8s} | '
      f'{"a(2p)":>8s} {"b(2p)":>8s} {"MAE 2p":>8s} | {"b_constr":>8s}')
print(f'{"-"*80}')

for i, Z in enumerate(Z_values):
    p = molecules[Z]
    Z1Z2 = p['Z1Z2']

    # 1-param
    a_1p[i] = fit_1param(r_grid, E_all[i], Z1Z2)
    P = parabola_both_constraints(r_grid, a_1p[i], Z1Z2)
    fit_mae_1p[i] = mean_absolute_error(E_all[i], P)

    # 2-param
    a_2p[i], b_2p[i] = fit_2param(r_grid, E_all[i], Z1Z2)
    P2 = parabola_value_constraint(r_grid, a_2p[i], b_2p[i], Z1Z2)
    fit_mae_2p[i] = mean_absolute_error(E_all[i], P2)

    b_constr = Z1Z2 * ke / r_c**2
    marker = '  TEST' if Z in test_Z else ''
    print(f'{Z:3d} {Z1Z2:5d} | {a_1p[i]:8.3f} {fit_mae_1p[i]:8.2f} | '
          f'{a_2p[i]:8.3f} {b_2p[i]:8.3f} {fit_mae_2p[i]:8.2f} | {b_constr:8.4f}{marker}')

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

# 1-param adaptive: Z → a
mdl_a1 = Ridge(alpha=1e-6)
mdl_a1.fit(X_tr, a_1p[train_idx])
a1_pred = mdl_a1.predict(X_te)

E_pred_1p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_1p[j] = parabola_both_constraints(r_grid, a1_pred[j], Z**2)
mae_1p = mean_absolute_error(E_all[test_idx].ravel(), E_pred_1p.ravel())

# 2-param adaptive: Z → (a, b)
mdl_a2 = Ridge(alpha=1e-6)
mdl_a2.fit(X_tr, a_2p[train_idx])
a2_pred = mdl_a2.predict(X_te)

mdl_b2 = Ridge(alpha=1e-6)
mdl_b2.fit(X_tr, b_2p[train_idx])
b2_pred = mdl_b2.predict(X_te)

E_pred_2p = np.zeros((len(test_Z), n_grid))
for j, Z in enumerate(test_Z):
    E_pred_2p[j] = parabola_value_constraint(r_grid, a2_pred[j], b2_pred[j], Z**2)
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

print(f'\n{"="*80}')
print(f'ML EXTRAPOLATION: train Z=1-8, test Z=9-12')
print(f'{"="*80}')
print(f'{"Method":<55s} | {"MAE":>8s} | {"ratio":>7s}')
print(f'{"-"*75}')
print(f'{"Direct multioutput: Z → E(r) [50 targets]":<55s} | {mae_direct:8.2f} | {"base":>7s}')
print(f'{"Parabola 1-param: Z → a [c,b from E_nuc]":<55s} | {mae_1p:8.2f} | {mae_1p/mae_direct:7.3f}x')
print(f'{"Parabola 2-param: Z → (a,b) [c from E_nuc]":<55s} | {mae_2p:8.2f} | {mae_2p/mae_direct:7.3f}x')
print(f'{"Morse adaptive: Z → (De,α,re) [3 targets]":<55s} | {mae_morse:8.2f} | {mae_morse/mae_direct:7.3f}x')
print(f'{"-"*75}')

# Per-molecule
print(f'\nPer-molecule MAE:')
print(f'{"Method":<30s} |', ' '.join(f'Z={Z:>4d}' for Z in test_Z))
for label, pred in [('Direct (50 params)', E_pred_direct),
                     ('Parabola 1p (a)', E_pred_1p),
                     ('Parabola 2p (a,b)', E_pred_2p),
                     ('Morse (De,α,re)', E_pred_morse)]:
    maes = [mean_absolute_error(E_all[test_idx[j]], pred[j]) for j in range(len(test_Z))]
    print(f'{label:<30s} |', ' '.join(f'{m:7.1f}' for m in maes))

# ================================================================
# FIGURE 2: Parameter trends and ML results
# ================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))

# Top-left: a(1p) vs Z
ax = axes2[0, 0]
ax.plot(train_Z, a_1p[train_idx], 'bo', ms=10, label='Train', zorder=5)
ax.plot(test_Z, a_1p[test_idx], 'rs', ms=10, label='Test (true)', zorder=5)
ax.plot(test_Z, a1_pred.ravel(), 'r^', ms=10, label='Test (pred)', zorder=5)
Z_dense = np.linspace(0, 13, 100)
ax.plot(Z_dense, mdl_a1.predict(Z_dense.reshape(-1, 1)), 'b-', alpha=0.4)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$a$ [eV/Å²]', fontsize=11)
ax.set_title('1-param: curvature $a$ vs Z', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Top-middle: a(2p) and b(2p) vs Z
ax = axes2[0, 1]
ax.plot(Z_values, a_2p, 'bo-', ms=6, label='$a$ (fitted)')
ax.plot(test_Z, a2_pred.ravel(), 'b^', ms=8, label='$a$ (pred)')
ax2 = ax.twinx()
ax2.plot(Z_values, b_2p, 'rs-', ms=6, label='$b$ (fitted)')
ax2.plot(test_Z, b2_pred.ravel(), 'r^', ms=8, label='$b$ (pred)')
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('$a$', fontsize=11, color='b')
ax2.set_ylabel('$b$', fontsize=11, color='r')
ax.set_title('2-param: $a$ and $b$ vs Z', fontsize=11)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7)
ax.grid(True, alpha=0.3)

# Top-right: fit quality
ax = axes2[0, 2]
x = np.arange(len(Z_values))
w = 0.35
ax.bar(x - w/2, fit_mae_1p, w, label='1-param', color='steelblue')
ax.bar(x + w/2, fit_mae_2p, w, label='2-param', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(Z_values)
ax.set_xlabel('Z', fontsize=11)
ax.set_ylabel('Fit MAE [eV]', fontsize=11)
ax.set_title('Fit quality (lower = better shape match)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Bottom row: predictions for test molecules
for j in range(min(3, len(test_Z))):
    ax = axes2[1, j]
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
            label=f'Morse ({mae_m:.0f})')

    ax.set_xlabel('$r$ [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [eV]', fontsize=11)
    ax.set_title(f'Z = {Z} (Z₁Z₂={Z**2}) [TEST]', fontsize=11,
                 color='#d62728', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig2.suptitle(f'Anatole\'s constrained parabola: $E^m = a(r-r_c)^2 + b(r-r_c) + c$,  $r_c={r_c}$ Å\n'
              f'Constraints from $E_{{nuc}}$: $c = -Z_1Z_2 k_e/r_c$,  $b = Z_1Z_2 k_e/r_c^2$',
              fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_anatole_parabola_ml.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_anatole_parabola_ml.png')
plt.close(fig2)
