"""
Clean adaptive curvature v2: inputs are Z^2 and r.

User's correction: the descriptors are Z^2 (what atoms) and r (their distance).

Feature sets:
  [Z², r]          — what the user asked for
  [Z², r, Z²/r]    — adding V_nn explicitly (it's known physics)

Same dataset: homonuclear Morse diatomics.
  Training: Z = 5-13 (9 molecules)
  Test:     Z = 17-27 (11 molecules, extrapolation)

Methods:
  - Direct Ridge multioutput: Z → V_total(r)
  - Shared Ridge:  [Z², r] → scalar (parabola or direct E_elec)
  - Shared MLP:    [Z², r] → scalar (parabola or direct E_elec)
  - Global Rose:   Z → (D_e, R_e, α) → Morse
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
print(f'          Z² = {Z_train[0]**2}-{Z_train[-1]**2}')
print(f'Test:     {n_test} molecules, Z = {Z_test[0]}-{Z_test[-1]}')
print(f'          Z² = {Z_test[0]**2}-{Z_test[-1]**2}')
print(f'r0 = {r0} Bohr')

# ================================================================
# Build stacked features
# ================================================================
def stack(data, r, feature_set):
    Z_AB = data['Z_AB']
    n_mol, n_grid = len(Z_AB), len(r)
    rows = []
    for i in range(n_mol):
        for j in range(n_grid):
            if feature_set == 'Z2_r':
                rows.append([Z_AB[i], r[j]])
            elif feature_set == 'Z2_r_Z2r':
                rows.append([Z_AB[i], r[j], Z_AB[i]/r[j]])
    return np.array(rows)

# Baselines
X_mol_tr = train['Z'].reshape(-1, 1)
X_mol_te = test['Z'].reshape(-1, 1)

mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
V_pred_direct = mdl.predict(X_mol_te)
mae_direct = mean_absolute_error(test['V_total'].ravel(), V_pred_direct.ravel())

# Global Rose
V_pred_global = np.zeros((n_test, n_grid))
for j, param in enumerate(['D_e', 'R_e', 'alpha']):
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_mol_tr, train[param])
    pred = mdl.predict(X_mol_te)
    if j == 0: De_pred = pred
    elif j == 1: Re_pred = pred
    else: al_pred = pred
for i in range(n_test):
    u = np.exp(-al_pred[i] * (r - Re_pred[i]))
    V_pred_global[i] = De_pred[i] * (1 - u)**2 - De_pred[i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())

print(f'\n--- Baselines ---')
print(f'  Direct Ridge Z→V(r):  {mae_direct:.4f} Ha')
print(f'  Global Rose Z→params: {mae_global:.4f} Ha')

# ================================================================
# Test all combinations
# ================================================================
feature_configs = [
    ('Z2_r', '[Z², r]'),
    ('Z2_r_Z2r', '[Z², r, Z²/r]'),
]

print(f'\n{"="*90}')
print(f'{"Features":<18s} | {"Model":<7s} | {"Target":<15s} | {"MAE [Ha]":>10s} | {"vs Direct":>12s}')
print(f'{"="*90}')

best_results = {}

for fs_key, fs_label in feature_configs:
    X_tr = stack(train, r, fs_key)
    X_te = stack(test, r, fs_key)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    a_train = (train['E_elec'] / dr2[np.newaxis, :]).ravel()
    E_train = train['E_elec'].ravel()

    for model_name, target_name, targets in [
        ('Ridge', 'a(r) parabola', a_train),
        ('Ridge', 'E_elec direct', E_train),
        ('MLP', 'a(r) parabola', a_train),
        ('MLP', 'E_elec direct', E_train),
    ]:
        if model_name == 'Ridge':
            mdl = Ridge(alpha=1.0)
            mdl.fit(X_tr, targets)
            pred = mdl.predict(X_te).reshape(n_test, n_grid)
        else:
            mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                               max_iter=3000, random_state=42, early_stopping=True,
                               validation_fraction=0.15)
            mdl.fit(X_tr_sc, targets)
            pred = mdl.predict(X_te_sc).reshape(n_test, n_grid)

        if 'parabola' in target_name:
            V_pred = pred * dr2[np.newaxis, :] + test['V_nn']
        else:
            V_pred = pred + test['V_nn']

        mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
        ratio = mae / mae_direct
        marker = 'BETTER' if ratio < 1 else ''

        key = f'{fs_label} {model_name} {target_name}'
        best_results[key] = {'mae': mae, 'V': V_pred}

        print(f'{fs_label:<18s} | {model_name:<7s} | {target_name:<15s} | {mae:10.4f} | {ratio:>10.1f}x  {marker}')

    print(f'{"-"*90}')

print(f'\n--- Reference ---')
print(f'  Direct Ridge Z→V(r) multioutput:  {mae_direct:.4f} Ha (baseline)')
print(f'  Global Rose Z→(D_e,R_e,α)→Morse:  {mae_global:.4f} Ha ({mae_global/mae_direct:.1f}x)')

# ================================================================
# FIGURE: Bar chart
# ================================================================
fig, ax = plt.subplots(figsize=(14, 6))

plot_order = [
    ('Direct Ridge\nZ→V(r)\nmultioutput', mae_direct, '#888888'),
]

# Add best from each feature set
for fs_key, fs_label in feature_configs:
    for model_name in ['Ridge', 'MLP']:
        for target in ['a(r) parabola', 'E_elec direct']:
            key = f'{fs_label} {model_name} {target}'
            mae = best_results[key]['mae']
            if 'parabola' in target:
                color = '#2ca02c' if model_name == 'Ridge' else '#8fce8f'
            else:
                color = '#d62728' if model_name == 'Ridge' else '#ff9896'
            label = f'{model_name}\n{target}\n{fs_label}'
            plot_order.append((label, mae, color))

plot_order.append(('Global Rose\nZ→(D_e,R_e,α)\nMorse eq.', mae_global, '#1f77b4'))

names = [p[0] for p in plot_order]
maes = [p[1] for p in plot_order]
cols = [p[2] for p in plot_order]
x = np.arange(len(names))

ax.bar(x, maes, color=cols, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('MAE [Ha]', fontsize=12)
ax.set_title(f'Extrapolation: Z²={Z_train[0]**2}-{Z_train[-1]**2} (train) → Z²={Z_test[0]**2}-{Z_test[-1]**2} (test)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for i in range(len(names)):
    ratio = maes[i] / mae_direct
    if ratio >= 1:
        label = f'{ratio:.1f}x'
    else:
        label = f'{1/ratio:.1f}x'
    ax.annotate(label, xy=(x[i], maes[i]),
               xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=9, fontweight='bold')

max_show = mae_direct * 15
if max(maes) > max_show:
    ax.set_ylim(0, max_show)
    for i, m in enumerate(maes):
        if m > max_show:
            ax.text(x[i], max_show * 0.85, f'{m:.1f}', ha='center', fontsize=8, color='red')

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_clean_v2_comparison.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_clean_v2_comparison.png')
plt.close(fig)

# ================================================================
# FIGURE: Predictions
# ================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
ex_idx = [0, n_test//2, n_test-1]

# Find best parabola and best direct for plotting
best_para_key = min([k for k in best_results if 'parabola' in k],
                     key=lambda k: best_results[k]['mae'])
best_direct_key = min([k for k in best_results if 'direct' in k],
                      key=lambda k: best_results[k]['mae'])

for col, idx in enumerate(ex_idx):
    Z_i = test['Z'][idx]

    ax = axes2[0, col]
    ax.plot(r, test['V_total'][idx], 'b-', lw=2.5, label='True')
    ax.plot(r, V_pred_direct[idx], 'gray', ls='--', lw=1.5, label='Direct Ridge')
    ax.plot(r, best_results[best_para_key]['V'][idx], 'g-', lw=1.5,
            label=f'Best parabola')
    ax.plot(r, best_results[best_direct_key]['V'][idx], 'r:', lw=1.5,
            label=f'Best E_elec direct')
    ax.plot(r, V_pred_global[idx], 'k-.', lw=2, label='Global Rose')
    ax.set_title(f'Z = {Z_i:.0f}, Z² = {Z_i**2:.0f}', fontsize=11)
    ax.set_ylim(-1, 3)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$V_{total}$ [Ha]')
        ax.legend(fontsize=7)

    ax2 = axes2[1, col]
    for V_p, color, ls, lab in [
        (V_pred_direct, 'gray', '--', 'Direct Ridge'),
        (best_results[best_para_key]['V'], 'green', '-', 'Best parabola'),
        (best_results[best_direct_key]['V'], 'red', ':', 'Best E_elec direct'),
        (V_pred_global, 'black', '-.', 'Global Rose'),
    ]:
        ax2.plot(r, V_p[idx] - test['V_total'][idx], color=color, ls=ls, lw=1.5, label=lab)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('r [Bohr]')
    ax2.grid(True, alpha=0.3)
    if col == 0:
        ax2.set_ylabel('Error [Ha]')
        ax2.legend(fontsize=7)

fig2.suptitle(f'Extrapolation predictions — input: [Z², r]', fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_clean_v2_predictions.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_clean_v2_predictions.png')
plt.close(fig2)
