"""
Clean adaptive curvature test: inputs are Z and r, nothing else.

Anatole's formulation: a(r) = f(Z, r)
  - Z: atomic number (identifies the molecule)
  - r: internuclear distance

E_elec(r) = a(Z, r) * (r - r0)^2
V_total = E_elec + V_nn = E_elec + Z^2/r

Dataset: homonuclear diatomics with Morse binding curves.
  Training: Z = 5, 6, 7, ..., 13  (B-B to Al-Al)
  Test:     Z = 17, 18, ..., 27    (Cl-Cl to Co-Co)

Parameters follow periodic trends:
  D_e = 0.1 + 0.025*(Z - 3)     (deeper well for heavier atoms)
  R_e = 2.2                      (fixed for simplicity)
  alpha = 0.9                    (fixed for simplicity)

Methods compared:
  1. Direct multioutput Ridge:  Z -> V_total(r), 50 independent outputs
  2. Ridge a(Z, r):             shared model, predict curvature
  3. Ridge E_elec(Z, r):        shared model, predict E_elec directly (no parabola)
  4. MLP a(Z, r):               neural net, predict curvature
  5. MLP E_elec(Z, r):          neural net, predict E_elec directly
  6. Global Rose:               Z -> (D_e, R_e, alpha), Morse reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Dataset: one molecule per Z value
# ================================================================
def make_molecules(Z_values):
    """Generate Morse curves for homonuclear diatomics at given Z values."""
    r = np.linspace(1.0, 8.0, 50)
    n_mol = len(Z_values)
    Z = np.array(Z_values, dtype=float)
    Z_AB = Z**2

    # Simple periodic trends (only Z-dependent)
    D_e = 0.1 + 0.025 * (Z - 3)
    R_e = np.full(n_mol, 2.2)
    alpha = np.full(n_mol, 0.9)

    V_total = np.zeros((n_mol, len(r)))
    V_nn = np.zeros((n_mol, len(r)))
    E_elec = np.zeros((n_mol, len(r)))

    for i in range(n_mol):
        u = np.exp(-alpha[i] * (r - R_e[i]))
        V_total[i] = D_e[i] * (1 - u)**2 - D_e[i]
        V_nn[i] = Z_AB[i] / r
        E_elec[i] = V_total[i] - V_nn[i]

    return {
        'Z': Z, 'Z_AB': Z_AB,
        'D_e': D_e, 'R_e': R_e, 'alpha': alpha,
        'r': r, 'V_total': V_total, 'V_nn': V_nn, 'E_elec': E_elec,
    }


# ================================================================
# Generate data
# ================================================================
Z_train = list(range(5, 14))     # Z = 5,6,7,...,13 (9 molecules)
Z_test = list(range(17, 28))     # Z = 17,18,...,27 (11 molecules)

train = make_molecules(Z_train)
test = make_molecules(Z_test)
r = train['r']
n_train = len(Z_train)
n_test = len(Z_test)
n_grid = len(r)

r0 = 20.0
dr2 = (r - r0)**2

print(f'Training: {n_train} molecules, Z = {Z_train[0]}-{Z_train[-1]}')
print(f'Test:     {n_test} molecules, Z = {Z_test[0]}-{Z_test[-1]}')
print(f'r0 = {r0} Bohr')
print(f'(r-r0)^2 range: [{dr2.min():.0f}, {dr2.max():.0f}], ratio = {dr2.max()/dr2.min():.1f}x')

# ================================================================
# FIGURE 1: What does a(r) look like for these molecules?
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for i in range(n_train):
    Z_i = train['Z'][i]
    axes[0, 0].plot(r, train['V_total'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes[0, 1].plot(r, train['E_elec'][i], lw=1.5, alpha=0.7)
    a_i = train['E_elec'][i] / dr2
    axes[0, 2].plot(r, a_i, lw=1.5, alpha=0.7)

for i in range(n_test):
    Z_i = test['Z'][i]
    axes[1, 0].plot(r, test['V_total'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes[1, 1].plot(r, test['E_elec'][i], lw=1.5, alpha=0.7)
    a_i = test['E_elec'][i] / dr2
    axes[1, 2].plot(r, a_i, lw=1.5, alpha=0.7)

axes[0, 0].set_title('$V_{total}$ (Morse)')
axes[0, 1].set_title('$E_{elec} = V_{total} - Z^2/r$')
axes[0, 2].set_title(f'$a(r) = E_{{elec}} / (r - {r0:.0f})^2$')
axes[0, 0].set_ylabel(f'Training (Z={Z_train[0]}-{Z_train[-1]})\nEnergy [Ha]')
axes[1, 0].set_ylabel(f'Test (Z={Z_test[0]}-{Z_test[-1]})\nEnergy [Ha]')
axes[0, 0].legend(fontsize=6, ncol=2)
axes[1, 0].legend(fontsize=6, ncol=2)

for ax in axes[:, 0]:
    ax.set_ylim(-1, 3)
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('r [Bohr]')

fig.suptitle('Homonuclear diatomics: what the ML targets look like', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_clean_curvature_targets.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_clean_curvature_targets.png')
plt.close(fig)

# ================================================================
# Build features
# ================================================================
# Multioutput: one row per molecule, input = Z (scalar)
X_mol_tr = train['Z'].reshape(-1, 1)      # (n_train, 1)
X_mol_te = test['Z'].reshape(-1, 1)       # (n_test, 1)

# Shared model: one row per (molecule, grid point), input = [Z, r]
def stack_Zr(data, r):
    Z = data['Z']
    n_mol, n_grid = len(Z), len(r)
    X = np.zeros((n_mol * n_grid, 2))
    for i in range(n_mol):
        for j in range(n_grid):
            X[i * n_grid + j] = [Z[i], r[j]]
    return X

X_stk_tr = stack_Zr(train, r)   # (n_train * 50, 2)
X_stk_te = stack_Zr(test, r)    # (n_test * 50, 2)

scaler = StandardScaler()
X_stk_tr_sc = scaler.fit_transform(X_stk_tr)
X_stk_te_sc = scaler.transform(X_stk_te)

# Targets
a_train = (train['E_elec'] / dr2[np.newaxis, :]).ravel()
E_train = train['E_elec'].ravel()

print(f'\nTraining a(r) range: [{a_train.min():.3f}, {a_train.max():.5f}]')
print(f'Training E_elec range: [{E_train.min():.1f}, {E_train.max():.4f}]')

# ================================================================
# ML methods
# ================================================================
results = {}

# 1. Direct multioutput Ridge: Z -> V_total(r)
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
V_pred = mdl.predict(X_mol_te)
mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['Direct Ridge\nZ→V(r) multioutput'] = {'mae': mae, 'V': V_pred}
print(f'\n1. Direct Ridge multioutput:  {mae:.4f} Ha')

# 2. Ridge a(Z, r) shared
mdl = Ridge(alpha=1.0)
mdl.fit(X_stk_tr, a_train)
a_pred = mdl.predict(X_stk_te).reshape(n_test, n_grid)
V_pred = a_pred * dr2[np.newaxis, :] + test['V_nn']
mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['Ridge shared\na(Z,r) parabola'] = {'mae': mae, 'V': V_pred}
print(f'2. Ridge a(Z,r) parabola:    {mae:.4f} Ha')

# 3. Ridge E_elec(Z, r) shared (no parabola)
mdl = Ridge(alpha=1.0)
mdl.fit(X_stk_tr, E_train)
E_pred = mdl.predict(X_stk_te).reshape(n_test, n_grid)
V_pred = E_pred + test['V_nn']
mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['Ridge shared\nE_elec(Z,r) direct'] = {'mae': mae, 'V': V_pred}
print(f'3. Ridge E_elec(Z,r) direct: {mae:.4f} Ha')

# 4. MLP a(Z, r) shared
mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                   max_iter=3000, random_state=42, early_stopping=True,
                   validation_fraction=0.15)
mdl.fit(X_stk_tr_sc, a_train)
a_pred = mdl.predict(X_stk_te_sc).reshape(n_test, n_grid)
V_pred = a_pred * dr2[np.newaxis, :] + test['V_nn']
mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['MLP shared\na(Z,r) parabola'] = {'mae': mae, 'V': V_pred}
print(f'4. MLP a(Z,r) parabola:      {mae:.4f} Ha')

# 5. MLP E_elec(Z, r) shared (no parabola)
mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                   max_iter=3000, random_state=42, early_stopping=True,
                   validation_fraction=0.15)
mdl.fit(X_stk_tr_sc, E_train)
E_pred = mdl.predict(X_stk_te_sc).reshape(n_test, n_grid)
V_pred = E_pred + test['V_nn']
mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['MLP shared\nE_elec(Z,r) direct'] = {'mae': mae, 'V': V_pred}
print(f'5. MLP E_elec(Z,r) direct:   {mae:.4f} Ha')

# 6. Global Rose: Z -> (D_e, R_e, alpha)
mdl_De = Ridge(alpha=1.0)
mdl_De.fit(X_mol_tr, train['D_e'])
De_pred = mdl_De.predict(X_mol_te)

mdl_Re = Ridge(alpha=1.0)
mdl_Re.fit(X_mol_tr, train['R_e'])
Re_pred = mdl_Re.predict(X_mol_te)

mdl_al = Ridge(alpha=1.0)
mdl_al.fit(X_mol_tr, train['alpha'])
al_pred = mdl_al.predict(X_mol_te)

V_pred_global = np.zeros((n_test, n_grid))
for i in range(n_test):
    u = np.exp(-al_pred[i] * (r - Re_pred[i]))
    V_pred_global[i] = De_pred[i] * (1 - u)**2 - De_pred[i]
mae = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())
results['Global Rose\nZ→(D_e,R_e,α)'] = {'mae': mae, 'V': V_pred_global}
print(f'6. Global Rose:              {mae:.4f} Ha')

# ================================================================
# Summary
# ================================================================
mae_baseline = results['Direct Ridge\nZ→V(r) multioutput']['mae']

print(f'\n{"="*75}')
print(f'{"Method":<35s} | {"Input":<10s} | {"MAE [Ha]":>10s} | {"vs Direct":>12s}')
print(f'{"="*75}')

for name in [
    'Direct Ridge\nZ→V(r) multioutput',
    'Ridge shared\na(Z,r) parabola',
    'Ridge shared\nE_elec(Z,r) direct',
    'MLP shared\na(Z,r) parabola',
    'MLP shared\nE_elec(Z,r) direct',
    'Global Rose\nZ→(D_e,R_e,α)',
]:
    mae = results[name]['mae']
    ratio = mae / mae_baseline
    name_flat = name.replace('\n', ' ')
    inp = '[Z]→50' if 'multioutput' in name or 'Rose' in name else '[Z,r]→1'

    if 'Direct Ridge' in name:
        print(f'{name_flat:<35s} | {inp:<10s} | {mae:10.4f} | {"baseline":>12s}')
    else:
        marker = 'BETTER' if ratio < 1 else 'worse'
        print(f'{name_flat:<35s} | {inp:<10s} | {mae:10.4f} | {ratio:>10.1f}x  ({marker})')

print(f'{"="*75}')

# ================================================================
# FIGURE 2: Bar chart
# ================================================================
fig2, ax = plt.subplots(figsize=(12, 6))

plot_names = [
    'Direct Ridge\nZ→V(r) multioutput',
    'Ridge shared\na(Z,r) parabola',
    'MLP shared\na(Z,r) parabola',
    'Ridge shared\nE_elec(Z,r) direct',
    'MLP shared\nE_elec(Z,r) direct',
    'Global Rose\nZ→(D_e,R_e,α)',
]
color_map = {
    'Direct Ridge\nZ→V(r) multioutput': '#888888',
    'Ridge shared\na(Z,r) parabola': '#2ca02c',
    'MLP shared\na(Z,r) parabola': '#8fce8f',
    'Ridge shared\nE_elec(Z,r) direct': '#d62728',
    'MLP shared\nE_elec(Z,r) direct': '#ff9896',
    'Global Rose\nZ→(D_e,R_e,α)': '#1f77b4',
}

maes = [results[n]['mae'] for n in plot_names]
cols = [color_map[n] for n in plot_names]
x = np.arange(len(plot_names))

ax.bar(x, maes, color=cols, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(plot_names, fontsize=9)
ax.set_ylabel('MAE [Ha]', fontsize=12)
ax.set_title(f'Extrapolation: Z={Z_train[0]}-{Z_train[-1]} (train) → Z={Z_test[0]}-{Z_test[-1]} (test)\nInput: just Z and r', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for i in range(len(plot_names)):
    ratio = maes[i] / mae_baseline
    if ratio >= 1:
        label = f'{ratio:.1f}x'
    else:
        label = f'{1/ratio:.1f}x better'
    ax.annotate(label, xy=(x[i], maes[i]),
               xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=10, fontweight='bold')

max_show = mae_baseline * 10
if max(maes) > max_show:
    ax.set_ylim(0, max_show)
    for i, m in enumerate(maes):
        if m > max_show:
            ax.text(x[i], max_show * 0.85, f'{m:.1f} Ha', ha='center', fontsize=9,
                    color='red', fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#888888', label='Baseline: Z→V(r), 50 independent outputs'),
    Patch(facecolor='#2ca02c', label='Parabola: [Z,r]→a, reconstruct E=a·(r-r₀)²'),
    Patch(facecolor='#d62728', label='Direct E_elec: [Z,r]→E_elec, add V_nn'),
    Patch(facecolor='#1f77b4', label='Global Rose: Z→(D_e,R_e,α), Morse equation'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_clean_curvature_comparison.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_clean_curvature_comparison.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Example predictions
# ================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
ex_idx = [0, n_test//2, n_test-1]

for col, idx in enumerate(ex_idx):
    Z_i = test['Z'][idx]

    ax = axes3[0, col]
    ax.plot(r, test['V_total'][idx], 'b-', lw=2.5, label='True')
    for name, color, ls in [
        ('Direct Ridge\nZ→V(r) multioutput', 'gray', '--'),
        ('MLP shared\na(Z,r) parabola', 'green', '-'),
        ('MLP shared\nE_elec(Z,r) direct', 'red', ':'),
        ('Global Rose\nZ→(D_e,R_e,α)', 'black', '-.'),
    ]:
        name_short = name.replace('\n', ' ')
        ax.plot(r, results[name]['V'][idx], color=color, ls=ls, lw=1.5, label=name_short)
    ax.set_title(f'Z = {Z_i:.0f}', fontsize=11)
    ax.set_ylim(-1, 3)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$V_{total}$ [Ha]')
        ax.legend(fontsize=7)

    ax2 = axes3[1, col]
    for name, color, ls in [
        ('Direct Ridge\nZ→V(r) multioutput', 'gray', '--'),
        ('MLP shared\na(Z,r) parabola', 'green', '-'),
        ('MLP shared\nE_elec(Z,r) direct', 'red', ':'),
        ('Global Rose\nZ→(D_e,R_e,α)', 'black', '-.'),
    ]:
        err = results[name]['V'][idx] - test['V_total'][idx]
        ax2.plot(r, err, color=color, ls=ls, lw=1.5)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('r [Bohr]')
    ax2.grid(True, alpha=0.3)
    if col == 0:
        ax2.set_ylabel('Error [Ha]')

fig3.suptitle('Extrapolation predictions: input = [Z, r] only', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_clean_curvature_predictions.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_clean_curvature_predictions.png')
plt.close(fig3)
