"""
Two matching figures for Anatole:
  fig_unnormalized_targets.png — E_elec and a(r) raw, train vs test
  fig_normalized_targets_v2.png — E_elec/Z² and a(r)/Z², train vs test

Plus ML errors measured as E_elec reconstruction MAE.
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
    Z = np.array(Z_values, dtype=float)
    Z_AB = Z**2
    D_e = 0.1 + 0.025 * (Z - 3)
    R_e = np.full(len(Z), 2.2)
    alpha = np.full(len(Z), 0.9)

    V_total = np.zeros((len(Z), len(r)))
    V_nn = np.zeros((len(Z), len(r)))
    E_elec = np.zeros((len(Z), len(r)))

    for i in range(len(Z)):
        u = np.exp(-alpha[i] * (r - R_e[i]))
        V_total[i] = D_e[i] * (1 - u)**2 - D_e[i]
        V_nn[i] = Z_AB[i] / r
        E_elec[i] = V_total[i] - V_nn[i]

    return {'Z': Z, 'Z_AB': Z_AB, 'D_e': D_e, 'R_e': R_e, 'alpha': alpha,
            'r': r, 'V_total': V_total, 'V_nn': V_nn, 'E_elec': E_elec}


Z_train = list(range(5, 14))
Z_test = list(range(17, 28))
train = make_molecules(Z_train)
test = make_molecules(Z_test)
r = train['r']
n_train, n_test, n_grid = len(Z_train), len(Z_test), len(r)

r0 = 20.0
dr2 = (r - r0)**2

# ================================================================
# FIGURE 1: UNNORMALIZED targets
# ================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))

for i in range(n_train):
    Z_i = train['Z'][i]
    axes1[0, 0].plot(r, train['E_elec'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes1[0, 1].plot(r, train['E_elec'][i] / dr2, lw=1.5, alpha=0.7)

for i in range(n_test):
    Z_i = test['Z'][i]
    axes1[1, 0].plot(r, test['E_elec'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes1[1, 1].plot(r, test['E_elec'][i] / dr2, lw=1.5, alpha=0.7)

axes1[0, 0].set_title('$E_{elec}(r)$', fontsize=13)
axes1[0, 1].set_title(f'$a(r) = E_{{elec}} / (r - {r0:.0f})^2$', fontsize=13)
axes1[0, 0].set_ylabel(f'Training (Z={Z_train[0]}-{Z_train[-1]})\n[Ha]', fontsize=11)
axes1[1, 0].set_ylabel(f'Test (Z={Z_test[0]}-{Z_test[-1]})\n[Ha]', fontsize=11)

# Range annotations
E_tr = train['E_elec']
E_te = test['E_elec']
a_tr = E_tr / dr2[np.newaxis, :]
a_te = E_te / dr2[np.newaxis, :]

axes1[0, 0].text(0.95, 0.05, f'range: [{E_tr.min():.0f}, {E_tr.max():.1f}]',
                 transform=axes1[0, 0].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))
axes1[1, 0].text(0.95, 0.05, f'range: [{E_te.min():.0f}, {E_te.max():.1f}]',
                 transform=axes1[1, 0].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))
axes1[0, 1].text(0.95, 0.05, f'range: [{a_tr.min():.3f}, {a_tr.max():.4f}]',
                 transform=axes1[0, 1].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))
axes1[1, 1].text(0.95, 0.05, f'range: [{a_te.min():.3f}, {a_te.max():.4f}]',
                 transform=axes1[1, 1].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))

axes1[0, 0].legend(fontsize=7, ncol=3, loc='lower left')
axes1[1, 0].legend(fontsize=7, ncol=3, loc='lower left')

for ax in axes1.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('r [Bohr]')

fig1.suptitle('Unnormalized targets — train vs test', fontsize=14)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'fig_unnormalized_targets.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_unnormalized_targets.png')
plt.close(fig1)

# ================================================================
# FIGURE 2: NORMALIZED targets (E_elec/Z² and a/Z²)
# ================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))

E_norm_tr = E_tr / train['Z_AB'][:, np.newaxis]
E_norm_te = E_te / test['Z_AB'][:, np.newaxis]
a_norm_tr = E_norm_tr / dr2[np.newaxis, :]
a_norm_te = E_norm_te / dr2[np.newaxis, :]

for i in range(n_train):
    Z_i = train['Z'][i]
    axes2[0, 0].plot(r, E_norm_tr[i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes2[0, 1].plot(r, a_norm_tr[i], lw=1.5, alpha=0.7)

for i in range(n_test):
    Z_i = test['Z'][i]
    axes2[1, 0].plot(r, E_norm_te[i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes2[1, 1].plot(r, a_norm_te[i], lw=1.5, alpha=0.7)

# Universal -1/r reference
axes2[0, 0].plot(r, -1.0/r, 'k--', lw=2, alpha=0.4, label='$-1/r$')
axes2[1, 0].plot(r, -1.0/r, 'k--', lw=2, alpha=0.4, label='$-1/r$')

axes2[0, 0].set_title('$E_{elec} / Z^2$', fontsize=13)
axes2[0, 1].set_title(f'$a(r) = (E_{{elec}} / Z^2) / (r - {r0:.0f})^2$', fontsize=13)
axes2[0, 0].set_ylabel(f'Training (Z={Z_train[0]}-{Z_train[-1]})\n[Ha]', fontsize=11)
axes2[1, 0].set_ylabel(f'Test (Z={Z_test[0]}-{Z_test[-1]})\n[Ha]', fontsize=11)

axes2[0, 0].text(0.95, 0.05, f'range: [{E_norm_tr.min():.3f}, {E_norm_tr.max():.3f}]',
                 transform=axes2[0, 0].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightgreen', ec='green', alpha=0.9))
axes2[1, 0].text(0.95, 0.05, f'range: [{E_norm_te.min():.3f}, {E_norm_te.max():.3f}]',
                 transform=axes2[1, 0].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightgreen', ec='green', alpha=0.9))
axes2[0, 1].text(0.95, 0.05, f'range: [{a_norm_tr.min():.5f}, {a_norm_tr.max():.5f}]',
                 transform=axes2[0, 1].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightgreen', ec='green', alpha=0.9))
axes2[1, 1].text(0.95, 0.05, f'range: [{a_norm_te.min():.5f}, {a_norm_te.max():.5f}]',
                 transform=axes2[1, 1].transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(fc='lightgreen', ec='green', alpha=0.9))

axes2[0, 0].legend(fontsize=7, ncol=3, loc='lower left')
axes2[1, 0].legend(fontsize=7, ncol=3, loc='lower left')

for ax in axes2.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('r [Bohr]')

fig2.suptitle('Normalized targets ($\\div Z^2$) — train vs test overlap', fontsize=14)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_normalized_targets_v2.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_normalized_targets_v2.png')
plt.close(fig2)

# ================================================================
# ML: E_elec reconstruction errors
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

X_mol_tr = train['Z'].reshape(-1, 1)
X_mol_te = test['Z'].reshape(-1, 1)

# Targets
E_train_flat = train['E_elec'].ravel()
E_norm_train_flat = (train['E_elec'] / train['Z_AB'][:, np.newaxis]).ravel()
a_train_flat = (train['E_elec'] / dr2[np.newaxis, :]).ravel()
a_norm_train_flat = (train['E_elec'] / train['Z_AB'][:, np.newaxis] / dr2[np.newaxis, :]).ravel()

print(f'\n{"="*75}')
print(f'E_elec RECONSTRUCTION ERRORS (MAE in Ha)')
print(f'{"="*75}')
print(f'{"Method":<40s} | {"E_elec MAE":>12s}')
print(f'{"-"*75}')

# Direct Ridge multioutput: Z → V_total, then E_elec = V_pred - V_nn
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
V_pred = mdl.predict(X_mol_te)
E_pred = V_pred - test['V_nn']
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
print(f'{"Direct Ridge Z→V(r) multioutput":<40s} | {mae:12.4f}')
mae_baseline = mae

# Global Rose
V_pred_g = np.zeros((n_test, n_grid))
for j, param in enumerate(['D_e', 'R_e', 'alpha']):
    m = Ridge(alpha=1.0); m.fit(X_mol_tr, train[param])
    p = m.predict(X_mol_te)
    if j == 0: De_p = p
    elif j == 1: Re_p = p
    else: al_p = p
for i in range(n_test):
    u = np.exp(-al_p[i] * (r - Re_p[i]))
    V_pred_g[i] = De_p[i] * (1 - u)**2 - De_p[i]
E_pred_g = V_pred_g - test['V_nn']
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred_g.ravel())
print(f'{"Global Rose Z→(D_e,R_e,α)":<40s} | {mae:12.4f}')

print(f'{"-"*75}')

for model_type in ['Ridge', 'MLP']:
    for target_name, target_data, recon in [
        ('E_elec', E_train_flat, 'raw'),
        ('E_elec/Z² (normalized)', E_norm_train_flat, 'norm'),
        ('a(r) parabola', a_train_flat, 'a_raw'),
        ('a(r)/Z² parabola (normalized)', a_norm_train_flat, 'a_norm'),
    ]:
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

        # Reconstruct E_elec
        if recon == 'raw':
            E_pred = pred
        elif recon == 'norm':
            E_pred = pred * test['Z_AB'][:, np.newaxis]
        elif recon == 'a_raw':
            E_pred = pred * dr2[np.newaxis, :]
        elif recon == 'a_norm':
            E_pred = pred * dr2[np.newaxis, :] * test['Z_AB'][:, np.newaxis]

        mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
        label = f'{model_type} [Z,r]→{target_name}'
        print(f'{label:<40s} | {mae:12.4f}')

    print(f'{"-"*75}')

print(f'{"="*75}')
print(f'\nAll errors are MAE on E_elec(r) in Ha.')
print(f'Lower is better. Baseline = Direct Ridge multioutput = {mae_baseline:.4f} Ha.')
