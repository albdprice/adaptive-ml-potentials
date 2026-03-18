"""
Adaptive curvature v3: a(d,r) with proper r-basis functions.

The issue with v2: Ridge on [d1, d2, r] gives a linear function of r,
so V = a*(r-r0)^2 is only a cubic polynomial in r. Too restrictive.

Fix: use basis functions of r, crossed with descriptors.

a(d, r) = sum_j sum_k  w_{jk} * d_j * phi_k(r)

where d_j are descriptor features and phi_k(r) are r-basis functions
(Gaussian RBFs, polynomials, etc).

This is LINEAR in the w_{jk} parameters -> Ridge still works.
Extrapolation in descriptor space is still linear.
r-dependence can be as rich as we want.

With N_d descriptor features and N_r basis functions:
  - Adaptive a(r): N_d * N_r shared parameters
  - Direct:        N_d * N_grid independent parameters

If N_r < N_grid, adaptive has fewer parameters -> constrained.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

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


def rbf_basis(r, centers, width):
    """Gaussian RBF basis functions."""
    # r: (n_grid,), centers: (n_centers,)
    # Returns: (n_grid, n_centers)
    return np.exp(-((r[:, None] - centers[None, :]) ** 2) / (2 * width**2))


def build_tensor_features(data, r, phi_r, include_bias=True):
    """
    Build tensor product features: d_j * phi_k(r).

    data: dataset dict
    r: grid array
    phi_r: (n_grid, n_basis) -- basis function values at grid points

    For each molecule i and grid point j:
      features = [1, d1, d2] outer [phi_1(r_j), phi_2(r_j), ..., phi_K(r_j)]
      = [phi_1, phi_2, ..., d1*phi_1, d1*phi_2, ..., d2*phi_1, d2*phi_2, ...]

    Returns X: (n_mol * n_grid, n_desc * n_basis)
    """
    d1, d2 = data['d1'], data['d2']
    n_mol = len(d1)
    n_grid = len(r)
    n_basis = phi_r.shape[1]

    # Descriptor features for each molecule
    if include_bias:
        desc_features = np.column_stack([np.ones(n_mol), d1, d2])  # (n_mol, 3)
    else:
        desc_features = np.column_stack([d1, d2])  # (n_mol, 2)
    n_desc = desc_features.shape[1]

    # Build tensor product: (n_mol, n_grid, n_desc * n_basis)
    X = np.zeros((n_mol * n_grid, n_desc * n_basis))
    for i in range(n_mol):
        for j in range(n_grid):
            # Outer product of descriptor features and r-basis
            row = np.outer(desc_features[i], phi_r[j]).ravel()
            X[i * n_grid + j] = row

    return X


# ================================================================
# Run
# ================================================================
print('Generating datasets...')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
test = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
r = train['r']
n_train = len(train['d1'])
n_test = len(test['d1'])
n_grid = len(r)

r0 = 10.0
dr2 = (r - r0)**2

print(f'Training: {n_train} molecules, Z = {train["Z"].min():.0f}-{train["Z"].max():.0f}')
print(f'Test:     {n_test} molecules, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}\n')

# ---- Baselines ----
X_mol_tr = np.column_stack([train['d1'], train['d2']])
X_mol_te = np.column_stack([test['d1'], test['d2']])

# Direct multioutput
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
mae_direct = mean_absolute_error(test['V_total'].ravel(), mdl.predict(X_mol_te).ravel())

# Global Rose
V_pred_global = np.zeros((n_test, n_grid))
ppreds = []
for j in range(3):
    m = Ridge(alpha=1.0)
    m.fit(X_mol_tr, train['params'][:, j])
    ppreds.append(m.predict(X_mol_te))
for i in range(n_test):
    u = np.exp(-ppreds[2][i] * (r - ppreds[1][i]))
    V_pred_global[i] = ppreds[0][i] * (1 - u)**2 - ppreds[0][i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())

print(f'{"="*95}')
print(f'{"Method":<50s} | {"MAE [Ha]":>10s} | {"vs Direct":>10s} | {"# params":>8s}')
print(f'{"="*95}')
print(f'{"Direct [d1,d2]→V(r) (multioutput, 50 outputs)":<50s} | {mae_direct:10.4f} | {"baseline":>10s} | {3*n_grid:>8d}')

# ---- Adaptive curvature on V_total with r-basis ----
# Try different numbers of RBF basis functions
for n_basis in [3, 5, 10, 15, 25, 50]:
    centers = np.linspace(r[0], r[-1], n_basis)
    width = (r[-1] - r[0]) / (n_basis - 1) * 1.0  # overlap
    phi_r = rbf_basis(r, centers, width)  # (n_grid, n_basis)

    # Targets: a_V = V_total / (r - r0)^2
    a_V_train = (train['V_total'] / dr2[np.newaxis, :]).ravel()  # (n_train * n_grid,)
    a_V_test_true = (test['V_total'] / dr2[np.newaxis, :])

    # Build tensor features
    X_tr = build_tensor_features(train, r, phi_r)  # (n_train*n_grid, 3*n_basis)
    X_te = build_tensor_features(test, r, phi_r)

    n_params = X_tr.shape[1]

    # Fit
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, a_V_train)
    a_pred = mdl.predict(X_te).reshape(n_test, n_grid)

    # Reconstruct V_total
    V_pred = a_pred * dr2[np.newaxis, :]
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else 'worse'

    label = f'Adaptive a_V, {n_basis} RBF, tensor [1,d1,d2]×φ(r)'
    print(f'{label:<50s} | {mae:10.4f} | {ratio:>9.2f}x | {n_params:>8d}  ({marker})')

# ---- Also try WITHOUT parabola (just tensor product for V directly) ----
print(f'{"-"*95}')
print(f'{"--- Without parabola (tensor product for V) ---":<50s}')
for n_basis in [3, 5, 10, 25, 50]:
    centers = np.linspace(r[0], r[-1], n_basis)
    width = (r[-1] - r[0]) / (n_basis - 1) * 1.0
    phi_r = rbf_basis(r, centers, width)

    # Targets: V_total directly (not curvature)
    V_train_flat = train['V_total'].ravel()

    X_tr = build_tensor_features(train, r, phi_r)
    X_te = build_tensor_features(test, r, phi_r)
    n_params = X_tr.shape[1]

    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, V_train_flat)
    V_pred = mdl.predict(X_te).reshape(n_test, n_grid)
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else 'worse'

    label = f'Tensor V, {n_basis} RBF (no parabola)'
    print(f'{label:<50s} | {mae:10.4f} | {ratio:>9.2f}x | {n_params:>8d}  ({marker})')

print(f'{"-"*95}')
print(f'{"Global Rose [d1,d2]→(D,R,α)":<50s} | {mae_global:10.4f} | {mae_global/mae_direct:>9.2f}x | {6:>8d}  (BETTER)')
print(f'{"="*95}')

print(f'\nDirect uses {3*n_grid} params (3 desc features × {n_grid} grid points)')
print(f'Global uses 6 params (2 descriptors × 3 physical params)')
print(f'Tensor with N RBFs uses 3*N params (3 desc features × N basis funcs)')
print(f'\nQuestion: does the parabola constraint help vs unconstrained tensor product?')
print(f'Question: at what N_basis does adaptive curvature beat direct?')
