"""
Adaptive curvature v4: Anatole's full idea on E_elec.

a_E(d, r) = f(d1, d2, Z_AB, r) — single shared model
E_elec_pred = a_E * (r - r0)^2
V_pred = E_elec_pred + V_nn  (V_nn is known exactly)

Since Z_AB is KNOWN per molecule (not predicted), we include it as a
descriptor feature. This is what real DFT descriptors would do.

Compare:
  1. Direct multioutput:  [d1,d2]→V(r), 50 independent outputs
  2. Adaptive a_E(r):     [d1,d2,Z_AB] × φ(r) → scalar a, parabola reconstruction + V_nn
  3. Tensor V (no para):  [d1,d2,Z_AB] × φ(r) → V directly (no physics constraint)
  4. Global Rose:         [d1,d2]→(D_e,R_e,α), Morse reconstruction
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


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


def rbf_basis(r, centers, width):
    return np.exp(-((r[:, None] - centers[None, :]) ** 2) / (2 * width**2))


def build_tensor_features(desc_features, r, phi_r):
    """
    desc_features: (n_mol, n_desc)
    phi_r: (n_grid, n_basis)
    Returns: (n_mol * n_grid, n_desc * n_basis)
    """
    n_mol = desc_features.shape[0]
    n_desc = desc_features.shape[1]
    n_grid = len(r)
    n_basis = phi_r.shape[1]

    X = np.zeros((n_mol * n_grid, n_desc * n_basis))
    for i in range(n_mol):
        for j in range(n_grid):
            X[i * n_grid + j] = np.outer(desc_features[i], phi_r[j]).ravel()
    return X


print('Generating datasets...')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
test = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
r = train['r']
n_train, n_test, n_grid = len(train['d1']), len(test['d1']), len(r)

r0 = 10.0
dr2 = (r - r0)**2

print(f'Training: {n_train} mol, Z = {train["Z"].min():.0f}-{train["Z"].max():.0f}')
print(f'Test:     {n_test} mol, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}\n')

# Descriptor sets: with and without Z_AB
desc_train_basic = np.column_stack([np.ones(n_train), train['d1'], train['d2']])
desc_test_basic = np.column_stack([np.ones(n_test), test['d1'], test['d2']])
n_d_basic = 3

desc_train_Z = np.column_stack([np.ones(n_train), train['d1'], train['d2'], train['Z_AB']])
desc_test_Z = np.column_stack([np.ones(n_test), test['d1'], test['d2'], test['Z_AB']])
n_d_Z = 4

# ---- Baselines ----
X_mol_tr = np.column_stack([train['d1'], train['d2']])
X_mol_te = np.column_stack([test['d1'], test['d2']])

mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
mae_direct = mean_absolute_error(test['V_total'].ravel(), mdl.predict(X_mol_te).ravel())

# Direct with Z_AB descriptor
X_mol_tr_Z = np.column_stack([train['d1'], train['d2'], train['Z_AB']])
X_mol_te_Z = np.column_stack([test['d1'], test['d2'], test['Z_AB']])
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr_Z, train['V_total'])
mae_direct_Z = mean_absolute_error(test['V_total'].ravel(), mdl.predict(X_mol_te_Z).ravel())

# Elec direct with Z_AB
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr_Z, train['E_elec'])
E_pred = mdl.predict(X_mol_te_Z)
V_pred = E_pred + test['V_nn']
mae_elec_Z = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())

# Global Rose
V_pred_global = np.zeros((n_test, n_grid))
ppreds = []
for j in range(3):
    m = Ridge(alpha=1.0); m.fit(X_mol_tr, train['params'][:, j])
    ppreds.append(m.predict(X_mol_te))
for i in range(n_test):
    u = np.exp(-ppreds[2][i] * (r - ppreds[1][i]))
    V_pred_global[i] = ppreds[0][i] * (1 - u)**2 - ppreds[0][i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())

print(f'{"="*100}')
print(f'{"Method":<55s} | {"MAE [Ha]":>10s} | {"vs Direct":>10s} | {"#p":>5s}')
print(f'{"="*100}')
print(f'{"Direct [d1,d2]→V(r) multioutput":<55s} | {mae_direct:10.4f} | {"baseline":>10s} | {3*n_grid:>5d}')
print(f'{"Direct [d1,d2,Z²]→V(r) multioutput":<55s} | {mae_direct_Z:10.4f} | {mae_direct_Z/mae_direct:>9.2f}x | {4*n_grid:>5d}')
print(f'{"Elec [d1,d2,Z²]→E_elec multioutput + V_nn":<55s} | {mae_elec_Z:10.4f} | {mae_elec_Z/mae_direct:>9.2f}x | {4*n_grid:>5d}')
print(f'{"-"*100}')

# ---- Adaptive curvature on E_elec with Z_AB descriptor ----
for n_basis in [3, 5, 10, 15, 25, 50]:
    centers = np.linspace(r[0], r[-1], n_basis)
    width = (r[-1] - r[0]) / (n_basis - 1) * 1.0
    phi_r = rbf_basis(r, centers, width)

    # Targets: a_E = E_elec / (r - r0)^2
    a_E_train = (train['E_elec'] / dr2[np.newaxis, :]).ravel()

    # With Z_AB in descriptors
    X_tr = build_tensor_features(desc_train_Z, r, phi_r)
    X_te = build_tensor_features(desc_test_Z, r, phi_r)
    n_params = X_tr.shape[1]

    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, a_E_train)
    a_pred = mdl.predict(X_te).reshape(n_test, n_grid)
    E_pred = a_pred * dr2[np.newaxis, :]
    V_pred = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else 'worse'

    label = f'Adaptive a_E [1,d1,d2,Z²]×φ(r), {n_basis} RBF → a'
    print(f'{label:<55s} | {mae:10.4f} | {ratio:>9.2f}x | {n_params:>5d}  ({marker})')

print(f'{"-"*100}')

# ---- Same but on V_total (no nuclear subtraction) ----
for n_basis in [3, 10, 25, 50]:
    centers = np.linspace(r[0], r[-1], n_basis)
    width = (r[-1] - r[0]) / (n_basis - 1) * 1.0
    phi_r = rbf_basis(r, centers, width)

    a_V_train = (train['V_total'] / dr2[np.newaxis, :]).ravel()

    X_tr = build_tensor_features(desc_train_basic, r, phi_r)
    X_te = build_tensor_features(desc_test_basic, r, phi_r)
    n_params = X_tr.shape[1]

    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, a_V_train)
    a_pred = mdl.predict(X_te).reshape(n_test, n_grid)
    V_pred = a_pred * dr2[np.newaxis, :]
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else 'worse'

    label = f'Adaptive a_V [1,d1,d2]×φ(r), {n_basis} RBF → a'
    print(f'{label:<55s} | {mae:10.4f} | {ratio:>9.2f}x | {n_params:>5d}  ({marker})')

print(f'{"-"*100}')
print(f'{"Global Rose [d1,d2]→(D_e,R_e,α)":<55s} | {mae_global:10.4f} | {mae_global/mae_direct:>9.2f}x | {6:>5d}  (BETTER)')
print(f'{"="*100}')

print(f'\nSummary:')
print(f'  Global Rose: 6 params, {mae_global:.4f} Ha ({mae_global/mae_direct:.2f}x vs direct)')
print(f'  The parabola (r-r0)^2 is a weak physics equation compared to Morse.')
print(f'  Nuclear subtraction with Z_AB descriptor fixes Z^2 issue.')
print(f'  But shared-model parabola STILL cannot beat multioutput direct.')
