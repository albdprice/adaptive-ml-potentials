"""
Adaptive curvature v3: add Z_AB/r as explicit feature.

Key insight from v2: the shared model needs Z_AB/r interaction.
In multioutput, each grid point learns its own Z_AB coefficient.
In shared model, one Z_AB coefficient can't capture Z_AB/r.

Solution: add Z_AB/r as a feature. This IS known physics — V_nn = Z_AB/r.

Feature sets tested:
  A) [d1, d2, r]                    — original, no Z info
  B) [d1, d2, Z_AB, r]              — Z_AB but no interaction
  C) [d1, d2, Z_AB, r, Z_AB/r]      — with the key interaction
  D) [d1, d2, Z_AB/r, r]            — just give it V_nn essentially

For each: Ridge, MLP (with parabola on E_elec and without)

Also includes baselines: Direct Ridge multioutput, Global Rose.
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
def make_dataset(d1_range, d2_range, n_per_dim=20):
    d1_vals = np.linspace(d1_range[0], d1_range[1], n_per_dim)
    d2_vals = np.linspace(d2_range[0], d2_range[1], n_per_dim)
    d1_grid, d2_grid = np.meshgrid(d1_vals, d2_vals)
    d1_flat = d1_grid.ravel()
    d2_flat = d2_grid.ravel()
    n_samples = len(d1_flat)

    Z = 3 + 2 * d1_flat
    Z_AB = Z**2
    R_e = 2.0 + 0.2 * d2_flat
    D_e = 0.1 + 0.05 * d1_flat + 0.03 * d2_flat
    alpha = 0.8 + 0.1 * d2_flat

    r = np.linspace(1.0, 8.0, 50)

    V_total = np.zeros((n_samples, len(r)))
    V_nn = np.zeros((n_samples, len(r)))
    E_elec = np.zeros((n_samples, len(r)))

    for i in range(n_samples):
        u = np.exp(-alpha[i] * (r - R_e[i]))
        V_total[i] = D_e[i] * (1 - u)**2 - D_e[i]
        V_nn[i] = Z_AB[i] / r
        E_elec[i] = V_total[i] - V_nn[i]

    return {
        'd1': d1_flat, 'd2': d2_flat,
        'Z': Z, 'Z_AB': Z_AB,
        'params': np.column_stack([D_e, R_e, alpha]),
        'r': r, 'V_total': V_total, 'V_nn': V_nn, 'E_elec': E_elec,
    }


def stack_features(data, r, feature_set):
    """Build stacked features for all (molecule, grid point) pairs."""
    d1, d2, Z_AB = data['d1'], data['d2'], data['Z_AB']
    n_mol, n_grid = len(d1), len(r)
    rows = []
    for i in range(n_mol):
        for j in range(n_grid):
            rj = r[j]
            if feature_set == 'A':
                rows.append([d1[i], d2[i], rj])
            elif feature_set == 'B':
                rows.append([d1[i], d2[i], Z_AB[i], rj])
            elif feature_set == 'C':
                rows.append([d1[i], d2[i], Z_AB[i], rj, Z_AB[i]/rj])
            elif feature_set == 'D':
                rows.append([d1[i], d2[i], Z_AB[i]/rj, rj])
    return np.array(rows)


# ================================================================
print('Generating datasets...')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
test = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
r = train['r']
n_train, n_test, n_grid = len(train['d1']), len(test['d1']), len(r)

r0 = 20.0  # middle ground
dr2 = (r - r0)**2

print(f'Training: {n_train} mol, Z = {train["Z"].min():.0f}-{train["Z"].max():.0f}')
print(f'Test:     {n_test} mol, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}')
print(f'r0 = {r0} Bohr\n')

# Baselines
X_mol_tr = np.column_stack([train['d1'], train['d2']])
X_mol_te = np.column_stack([test['d1'], test['d2']])

mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
mae_direct = mean_absolute_error(test['V_total'].ravel(), mdl.predict(X_mol_te).ravel())

V_pred_global = np.zeros((n_test, n_grid))
ppreds = []
for j in range(3):
    m = Ridge(alpha=1.0); m.fit(X_mol_tr, train['params'][:, j])
    ppreds.append(m.predict(X_mol_te))
for i in range(n_test):
    u = np.exp(-ppreds[2][i] * (r - ppreds[1][i]))
    V_pred_global[i] = ppreds[0][i] * (1 - u)**2 - ppreds[0][i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())

# Also: multioutput Ridge with Z_AB feature (from z2_test, we know this works)
X_mol_tr_Z = np.column_stack([train['d1'], train['d2'], train['Z_AB']])
X_mol_te_Z = np.column_stack([test['d1'], test['d2'], test['Z_AB']])
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr_Z, train['E_elec'])
E_pred_mo = mdl.predict(X_mol_te_Z)
V_pred_mo_Z = E_pred_mo + test['V_nn']
mae_mo_Z = mean_absolute_error(test['V_total'].ravel(), V_pred_mo_Z.ravel())

print(f'--- Baselines ---')
print(f'  Direct Ridge [d1,d2]→V(r):              {mae_direct:.4f} Ha')
print(f'  Multioutput Ridge [d1,d2,Z²]→E_elec+Vnn: {mae_mo_Z:.4f} Ha')
print(f'  Global Rose:                              {mae_global:.4f} Ha')

# ================================================================
# Test each feature set with Ridge and MLP
# ================================================================
feature_sets = {
    'A': '[d1, d2, r]',
    'B': '[d1, d2, Z², r]',
    'C': '[d1, d2, Z², r, Z²/r]',
    'D': '[d1, d2, Z²/r, r]',
}

print(f'\n{"="*95}')
print(f'{"Features":<25s} | {"Model":<8s} | {"Target":<15s} | {"MAE [Ha]":>10s} | {"vs Direct":>12s}')
print(f'{"="*95}')

for fs_key, fs_label in feature_sets.items():
    X_tr = stack_features(train, r, fs_key)
    X_te = stack_features(test, r, fs_key)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # --- Ridge on curvature a(r) ---
    a_train = (train['E_elec'] / dr2[np.newaxis, :]).ravel()
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, a_train)
    a_pred = mdl.predict(X_te).reshape(n_test, n_grid)
    V_pred = a_pred * dr2[np.newaxis, :] + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else ''
    print(f'{fs_label:<25s} | {"Ridge":<8s} | {"a(r) parabola":<15s} | {mae:10.4f} | {ratio:>10.1f}x  {marker}')

    # --- MLP on curvature a(r) ---
    mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                       max_iter=2000, random_state=42, early_stopping=True,
                       validation_fraction=0.1)
    mdl.fit(X_tr_sc, a_train)
    a_pred = mdl.predict(X_te_sc).reshape(n_test, n_grid)
    V_pred = a_pred * dr2[np.newaxis, :] + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else ''
    print(f'{"":<25s} | {"MLP":<8s} | {"a(r) parabola":<15s} | {mae:10.4f} | {ratio:>10.1f}x  {marker}')

    # --- Ridge direct E_elec (no parabola) ---
    E_train = train['E_elec'].ravel()
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, E_train)
    E_pred = mdl.predict(X_te).reshape(n_test, n_grid)
    V_pred = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else ''
    print(f'{"":<25s} | {"Ridge":<8s} | {"E_elec direct":<15s} | {mae:10.4f} | {ratio:>10.1f}x  {marker}')

    # --- MLP direct E_elec (no parabola) ---
    mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                       max_iter=2000, random_state=42, early_stopping=True,
                       validation_fraction=0.1)
    mdl.fit(X_tr_sc, E_train)
    E_pred = mdl.predict(X_te_sc).reshape(n_test, n_grid)
    V_pred = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else ''
    print(f'{"":<25s} | {"MLP":<8s} | {"E_elec direct":<15s} | {mae:10.4f} | {ratio:>10.1f}x  {marker}')

    print(f'{"-"*95}')

print(f'\n--- Reference baselines ---')
print(f'  Direct Ridge multioutput [d1,d2]→V(r):   {mae_direct:.4f} Ha (1.0x)')
print(f'  Multioutput Ridge [d1,d2,Z²]→E_elec+Vnn:  {mae_mo_Z:.4f} Ha ({mae_mo_Z/mae_direct:.2f}x)')
print(f'  Global Rose [d1,d2]→(D_e,R_e,α):          {mae_global:.4f} Ha ({mae_global/mae_direct:.2f}x)')
