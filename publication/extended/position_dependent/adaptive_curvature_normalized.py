"""
Normalized curvature: divide E_elec by Z_AB before learning.

Key idea: Z_AB is a KNOWN scalar per molecule. Use it to normalize.

    E_elec / Z_AB = V_Morse/Z_AB - 1/r

The -1/r part is UNIVERSAL (same for every molecule).
The V_Morse/Z_AB part is small and varies gently with Z.

Methods:
  1. Direct Ridge multioutput: Z → V_total(r)
  2. Shared model on E_elec (unnormalized): [Z, r] → E_elec
  3. Shared model on E_elec/Z_AB (normalized): [Z, r] → E_elec/Z_AB
  4. Parabola on E_elec: [Z, r] → a, E = a*(r-r0)^2
  5. Parabola on E_elec/Z_AB: [Z, r] → a_norm, E/Z_AB = a_norm*(r-r0)^2
  6. Global Rose: Z → (D_e, R_e, alpha) → Morse

For each: Ridge and MLP.
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
# Dataset
# ================================================================
def make_molecules(Z_values):
    r = np.linspace(1.0, 8.0, 50)
    n_mol = len(Z_values)
    Z = np.array(Z_values, dtype=float)
    Z_AB = Z**2

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


Z_train = list(range(5, 14))
Z_test = list(range(17, 28))

train = make_molecules(Z_train)
test = make_molecules(Z_test)
r = train['r']
n_train, n_test, n_grid = len(Z_train), len(Z_test), len(r)

r0 = 20.0
dr2 = (r - r0)**2

print(f'Training: {n_train} molecules, Z = {Z_train[0]}-{Z_train[-1]}')
print(f'Test:     {n_test} molecules, Z = {Z_test[0]}-{Z_test[-1]}')

# ================================================================
# FIGURE 1: What normalization does to the targets
# ================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for i in range(n_train):
    Z_i = train['Z'][i]
    Z_AB_i = train['Z_AB'][i]
    E_norm_i = train['E_elec'][i] / Z_AB_i
    a_i = train['E_elec'][i] / dr2
    a_norm_i = E_norm_i / dr2

    axes[0, 0].plot(r, train['E_elec'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes[0, 1].plot(r, E_norm_i, lw=1.5, alpha=0.7)
    axes[0, 2].plot(r, a_i, lw=1.5, alpha=0.7)
    axes[0, 3].plot(r, a_norm_i, lw=1.5, alpha=0.7)

for i in range(n_test):
    Z_i = test['Z'][i]
    Z_AB_i = test['Z_AB'][i]
    E_norm_i = test['E_elec'][i] / Z_AB_i
    a_i = test['E_elec'][i] / dr2
    a_norm_i = E_norm_i / dr2

    axes[1, 0].plot(r, test['E_elec'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes[1, 1].plot(r, E_norm_i, lw=1.5, alpha=0.7)
    axes[1, 2].plot(r, a_i, lw=1.5, alpha=0.7)
    axes[1, 3].plot(r, a_norm_i, lw=1.5, alpha=0.7)

axes[0, 0].set_title('$E_{elec}$\n(raw)')
axes[0, 1].set_title('$E_{elec} / Z^2$\n(normalized)')
axes[0, 2].set_title('$a(r) = E_{elec} / (r-r_0)^2$\n(raw curvature)')
axes[0, 3].set_title('$a(r) = (E_{elec}/Z^2) / (r-r_0)^2$\n(normalized curvature)')

axes[0, 0].set_ylabel(f'Training Z={Z_train[0]}-{Z_train[-1]}')
axes[1, 0].set_ylabel(f'Test Z={Z_test[0]}-{Z_test[-1]}')
axes[0, 0].legend(fontsize=6, ncol=2)

# Show ranges
E_train_range = [train['E_elec'].min(), train['E_elec'].max()]
E_test_range = [test['E_elec'].min(), test['E_elec'].max()]
E_norm_train = train['E_elec'] / train['Z_AB'][:, np.newaxis]
E_norm_test = test['E_elec'] / test['Z_AB'][:, np.newaxis]

axes[0, 0].text(0.05, 0.05, f'range: [{E_train_range[0]:.0f}, {E_train_range[1]:.1f}]',
                transform=axes[0, 0].transAxes, fontsize=8, va='bottom',
                bbox=dict(fc='white', alpha=0.8))
axes[1, 0].text(0.05, 0.05, f'range: [{E_test_range[0]:.0f}, {E_test_range[1]:.1f}]',
                transform=axes[1, 0].transAxes, fontsize=8, va='bottom',
                bbox=dict(fc='white', alpha=0.8))
axes[0, 1].text(0.05, 0.05, f'range: [{E_norm_train.min():.2f}, {E_norm_train.max():.4f}]',
                transform=axes[0, 1].transAxes, fontsize=8, va='bottom',
                bbox=dict(fc='white', alpha=0.8))
axes[1, 1].text(0.05, 0.05, f'range: [{E_norm_test.min():.2f}, {E_norm_test.max():.4f}]',
                transform=axes[1, 1].transAxes, fontsize=8, va='bottom',
                bbox=dict(fc='white', alpha=0.8))

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('r [Bohr]')

fig.suptitle('Effect of Z² normalization on targets and curvature', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_normalized_targets.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_normalized_targets.png')
plt.close(fig)

# Print ranges
print(f'\nTarget ranges:')
print(f'  E_elec      train: [{E_train_range[0]:.1f}, {E_train_range[1]:.2f}]')
print(f'  E_elec      test:  [{E_test_range[0]:.1f}, {E_test_range[1]:.2f}]')
print(f'  E_elec/Z²   train: [{E_norm_train.min():.3f}, {E_norm_train.max():.5f}]')
print(f'  E_elec/Z²   test:  [{E_norm_test.min():.3f}, {E_norm_test.max():.5f}]')
print(f'  Overlap? Train max = {E_norm_train.min():.3f}, Test min = {E_norm_test.min():.3f}')

# ================================================================
# Build stacked features [Z, r]
# ================================================================
def stack_Zr(data, r):
    Z = data['Z']
    n_mol, n_grid = len(Z), len(r)
    X = np.zeros((n_mol * n_grid, 2))
    for i in range(n_mol):
        for j in range(n_grid):
            X[i * n_grid + j] = [Z[i], r[j]]
    return X

X_stk_tr = stack_Zr(train, r)
X_stk_te = stack_Zr(test, r)

scaler = StandardScaler()
X_stk_tr_sc = scaler.fit_transform(X_stk_tr)
X_stk_te_sc = scaler.transform(X_stk_te)

# Molecular features for multioutput
X_mol_tr = train['Z'].reshape(-1, 1)
X_mol_te = test['Z'].reshape(-1, 1)

# ================================================================
# Targets
# ================================================================
# Raw
E_train_flat = train['E_elec'].ravel()
a_train_flat = (train['E_elec'] / dr2[np.newaxis, :]).ravel()

# Normalized by Z_AB
E_norm_train_flat = (train['E_elec'] / train['Z_AB'][:, np.newaxis]).ravel()
a_norm_train_flat = (train['E_elec'] / train['Z_AB'][:, np.newaxis] / dr2[np.newaxis, :]).ravel()

# ================================================================
# ML
# ================================================================
results = {}

# --- Baseline: Direct Ridge multioutput ---
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
V_pred = mdl.predict(X_mol_te)
mae_direct = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['Direct Ridge\nZ→V(r) multioutput'] = mae_direct

# --- Global Rose ---
V_pred_global = np.zeros((n_test, n_grid))
for j, param in enumerate(['D_e', 'R_e', 'alpha']):
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_mol_tr, train[param])
    pred = mdl.predict(X_mol_te)
    if j == 0: De_p = pred
    elif j == 1: Re_p = pred
    else: al_p = pred
for i in range(n_test):
    u = np.exp(-al_p[i] * (r - Re_p[i]))
    V_pred_global[i] = De_p[i] * (1 - u)**2 - De_p[i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())
results['Global Rose\nZ→(D_e,R_e,α)'] = mae_global

# --- Shared models ---
configs = [
    # (label, model, target_flat, reconstruction)
    # reconstruction: 'E_elec' means V = pred + V_nn
    #                 'E_norm' means V = pred * Z_AB + V_nn
    #                 'a_raw' means V = pred * dr2 + V_nn
    #                 'a_norm' means V = pred * dr2 * Z_AB + V_nn
]

for model_type in ['Ridge', 'MLP']:
    for target_name, target_data, recon_type in [
        ('E_elec', E_train_flat, 'E_elec'),
        ('E_elec/Z²', E_norm_train_flat, 'E_norm'),
        ('a(r) parabola', a_train_flat, 'a_raw'),
        ('a(r)/Z² parabola', a_norm_train_flat, 'a_norm'),
    ]:
        # Train
        if model_type == 'Ridge':
            mdl = Ridge(alpha=1.0)
            mdl.fit(X_stk_tr, target_data)
            pred = mdl.predict(X_stk_te).reshape(n_test, n_grid)
        else:
            mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                               max_iter=3000, random_state=42, early_stopping=True,
                               validation_fraction=0.15)
            mdl.fit(X_stk_tr_sc, target_data)
            pred = mdl.predict(X_stk_te_sc).reshape(n_test, n_grid)

        # Reconstruct V_total
        if recon_type == 'E_elec':
            V_pred = pred + test['V_nn']
        elif recon_type == 'E_norm':
            V_pred = pred * test['Z_AB'][:, np.newaxis] + test['V_nn']
        elif recon_type == 'a_raw':
            V_pred = pred * dr2[np.newaxis, :] + test['V_nn']
        elif recon_type == 'a_norm':
            V_pred = pred * dr2[np.newaxis, :] * test['Z_AB'][:, np.newaxis] + test['V_nn']

        mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
        key = f'{model_type}\n{target_name}'
        results[key] = mae

# ================================================================
# Print results
# ================================================================
print(f'\n{"="*70}')
print(f'{"Method":<30s} | {"MAE [Ha]":>10s} | {"vs Direct":>12s}')
print(f'{"="*70}')

print_order = [
    'Direct Ridge\nZ→V(r) multioutput',
    'Ridge\nE_elec',
    'Ridge\nE_elec/Z²',
    'Ridge\na(r) parabola',
    'Ridge\na(r)/Z² parabola',
    'MLP\nE_elec',
    'MLP\nE_elec/Z²',
    'MLP\na(r) parabola',
    'MLP\na(r)/Z² parabola',
    'Global Rose\nZ→(D_e,R_e,α)',
]

for name in print_order:
    mae = results[name]
    ratio = mae / mae_direct
    flat = name.replace('\n', ' ')
    if 'Direct' in name:
        print(f'{flat:<30s} | {mae:10.4f} | {"baseline":>12s}')
    else:
        marker = ' BETTER' if ratio < 1 else ''
        print(f'{flat:<30s} | {mae:10.4f} | {ratio:>10.1f}x{marker}')

print(f'{"="*70}')

# ================================================================
# FIGURE 2: Bar chart
# ================================================================
fig2, ax = plt.subplots(figsize=(14, 7))

# Group: baseline, Ridge variants, MLP variants, Global
plot_items = [
    ('Direct Ridge\nmultioutput', mae_direct, '#888888'),
    ('Ridge\nE_elec', results['Ridge\nE_elec'], '#d62728'),
    ('Ridge\nE_elec/Z²', results['Ridge\nE_elec/Z²'], '#ff9896'),
    ('Ridge\na(r) parabola', results['Ridge\na(r) parabola'], '#2ca02c'),
    ('Ridge\na(r)/Z²\nparabola', results['Ridge\na(r)/Z² parabola'], '#8fce8f'),
    ('MLP\nE_elec', results['MLP\nE_elec'], '#d62728'),
    ('MLP\nE_elec/Z²', results['MLP\nE_elec/Z²'], '#ff9896'),
    ('MLP\na(r)\nparabola', results['MLP\na(r) parabola'], '#2ca02c'),
    ('MLP\na(r)/Z²\nparabola', results['MLP\na(r)/Z² parabola'], '#8fce8f'),
    ('Global Rose\n(D_e,R_e,α)', mae_global, '#1f77b4'),
]

names = [p[0] for p in plot_items]
maes = [p[1] for p in plot_items]
cols = [p[2] for p in plot_items]
x = np.arange(len(names))

ax.bar(x, maes, color=cols, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('MAE [Ha]', fontsize=12)
ax.set_title('Does Z² normalization help? Input = [Z, r]', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

for i in range(len(names)):
    ratio = maes[i] / mae_direct
    if ratio < 1:
        label = f'{1/ratio:.1f}x\nbetter'
    elif ratio < 10:
        label = f'{ratio:.1f}x'
    else:
        label = f'{ratio:.0f}x'
    ax.annotate(label, xy=(x[i], min(maes[i], mae_direct * 12)),
               xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=8, fontweight='bold')

max_show = mae_direct * 15
if max(maes) > max_show:
    ax.set_ylim(0, max_show)
    for i, m in enumerate(maes):
        if m > max_show:
            ax.text(x[i], max_show * 0.85, f'{m:.1f}', ha='center', fontsize=8, color='red')

# Divider lines
ax.axvline(0.5, color='gray', ls=':', lw=0.5)
ax.axvline(4.5, color='gray', ls=':', lw=0.5)
ax.axvline(8.5, color='gray', ls=':', lw=0.5)
ax.text(0, max_show * 0.95, 'Baseline', ha='center', fontsize=8, color='gray')
ax.text(2.5, max_show * 0.95, 'Ridge shared [Z,r]→1', ha='center', fontsize=8, color='gray')
ax.text(6.5, max_show * 0.95, 'MLP shared [Z,r]→1', ha='center', fontsize=8, color='gray')
ax.text(9, max_show * 0.95, 'Global', ha='center', fontsize=8, color='gray')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#888888', label='Baseline'),
    Patch(facecolor='#d62728', label='E_elec (unnormalized)'),
    Patch(facecolor='#ff9896', label='E_elec/Z² (normalized)'),
    Patch(facecolor='#2ca02c', label='Parabola a(r) (unnormalized)'),
    Patch(facecolor='#8fce8f', label='Parabola a(r)/Z² (normalized)'),
    Patch(facecolor='#1f77b4', label='Global Rose'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_normalized_comparison.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_normalized_comparison.png')
plt.close(fig2)
