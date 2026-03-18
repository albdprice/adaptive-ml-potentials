"""
Adaptive curvature learning: a(r) = f(descriptor, r).

Anatole's idea: learn a SINGLE scalar function a that depends on BOTH the
molecular descriptor (delta_Z) AND the internuclear distance r. Then reconstruct:

    E_elec(r) = a(descriptor, r) * (r - r_0)^2

This is ADAPTIVE because:
  - One model with shared parameters across all r values
  - Physics (parabola shape) constrains reconstruction
  - Not 50 independent regressions (that's point-wise/direct)

Methods compared:
  1. Direct:        X=[d1,d2], Y=V_total(r) — 50 independent outputs
  2. Elec direct:   X=[d1,d2], Y=E_elec(r) — 50 independent outputs, add V_nn
  3. Adaptive a(r): X=[d1,d2,r], Y=a — SINGLE output, shared model across r
                    Reconstruct E = a*(r-r0)^2, add V_nn
  4. Global:        X=[d1,d2], Y=(D_e,R_e,alpha) — 3 outputs, Morse reconstruction

Key: methods 1,2 have N_grid independent weight vectors.
     Method 3 has ONE weight vector shared across all r.
     Method 4 has 3 weight vectors + nonlinear physics.
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


# ================================================================
# Build stacked (descriptor, r) features for adaptive curvature
# ================================================================
def build_adaptive_features(data, r, r0, feature_set='basic'):
    """
    Stack (descriptor, r) pairs for all molecules and all grid points.

    Each molecule contributes N_grid rows. Features include both molecular
    descriptors AND position r, so the model shares weights across r.

    Returns:
        X: (N_mol * N_grid, n_features)
        a_targets: (N_mol * N_grid,) — curvature values
    """
    d1, d2 = data['d1'], data['d2']
    Z, Z_AB = data['Z'], data['Z_AB']
    n_mol = len(d1)
    n_grid = len(r)
    dr2 = (r - r0)**2

    # Compute curvature targets: a = E_elec / (r - r0)^2
    a_targets = data['E_elec'] / dr2[np.newaxis, :]  # (n_mol, n_grid)

    # Build stacked features
    rows = []
    a_flat = []
    for i in range(n_mol):
        for j in range(n_grid):
            if feature_set == 'basic':
                # [d1, d2, r]
                rows.append([d1[i], d2[i], r[j]])
            elif feature_set == 'with_Z':
                # [d1, d2, Z_AB, r]
                rows.append([d1[i], d2[i], Z_AB[i], r[j]])
            elif feature_set == 'interactions':
                # [d1, d2, r, d1*r, d2*r, r^2]
                rows.append([d1[i], d2[i], r[j], d1[i]*r[j], d2[i]*r[j], r[j]**2])
            elif feature_set == 'full':
                # [d1, d2, Z_AB, r, d1*r, d2*r, Z_AB*r, r^2, Z_AB/r]
                rows.append([d1[i], d2[i], Z_AB[i], r[j],
                           d1[i]*r[j], d2[i]*r[j], Z_AB[i]*r[j],
                           r[j]**2, Z_AB[i]/r[j]])
            a_flat.append(a_targets[i, j])

    return np.array(rows), np.array(a_flat)


# ================================================================
# Run experiment
# ================================================================
print('Generating datasets...')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
test = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
r = train['r']
n_test = len(test['d1'])
n_grid = len(r)

print(f'Training: {len(train["d1"])} molecules, Z = {train["Z"].min():.0f}-{train["Z"].max():.0f}')
print(f'Test:     {n_test} molecules, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}')

r0 = 10.0  # reference point (far out)
dr2 = (r - r0)**2

# ---- Method 1: Direct (multioutput) ----
X_train_mol = np.column_stack([train['d1'], train['d2']])
X_test_mol = np.column_stack([test['d1'], test['d2']])

mdl = Ridge(alpha=1.0)
mdl.fit(X_train_mol, train['V_total'])
V_pred_direct = mdl.predict(X_test_mol)
mae_direct = mean_absolute_error(test['V_total'].ravel(), V_pred_direct.ravel())

# ---- Method 2: Elec direct (multioutput) ----
mdl = Ridge(alpha=1.0)
mdl.fit(X_train_mol, train['E_elec'])
E_pred_elec = mdl.predict(X_test_mol)
V_pred_elec = E_pred_elec + test['V_nn']
mae_elec = mean_absolute_error(test['V_total'].ravel(), V_pred_elec.ravel())

# ---- Method 4: Global adaptive ----
V_pred_global = np.zeros((n_test, n_grid))
param_preds = []
for j in range(3):
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train_mol, train['params'][:, j])
    param_preds.append(mdl.predict(X_test_mol))
for i in range(n_test):
    u = np.exp(-param_preds[2][i] * (r - param_preds[1][i]))
    V_pred_global[i] = param_preds[0][i] * (1 - u)**2 - param_preds[0][i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())

# ---- Method 3: Adaptive curvature a(r) = f(desc, r) ----
print(f'\n{"="*90}')
print(f'{"Method":<30s} | {"MAE [Ha]":>12s} | {"vs Direct":>12s} | {"# weights":>10s}')
print(f'{"="*90}')
print(f'{"Direct [d1,d2]→V(r)":<30s} | {mae_direct:12.4f} | {"baseline":>12s} | {2*n_grid:>10d}')
print(f'{"Elec [d1,d2]→E_elec(r)":<30s} | {mae_elec:12.4f} | {mae_elec/mae_direct:>11.1f}x | {2*n_grid:>10d}')

feature_sets = ['basic', 'with_Z', 'interactions', 'full']
feature_labels = {
    'basic': 'Adaptive a [d1,d2,r]→a',
    'with_Z': 'Adaptive a [d1,d2,Z²,r]→a',
    'interactions': 'Adaptive a [d1,d2,r,d1r,d2r,r²]→a',
    'full': 'Adaptive a [d1,d2,Z²,r,...]→a',
}
feature_dims = {
    'basic': 3, 'with_Z': 4, 'interactions': 6, 'full': 9,
}

for fs in feature_sets:
    # Build stacked features
    X_train_stk, a_train = build_adaptive_features(train, r, r0, feature_set=fs)
    X_test_stk, a_test = build_adaptive_features(test, r, r0, feature_set=fs)

    # Fit single Ridge model on all (molecule, r) pairs
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train_stk, a_train)
    a_pred = mdl.predict(X_test_stk)

    # Reconstruct E_elec and V_total
    a_pred_2d = a_pred.reshape(n_test, n_grid)
    E_pred = a_pred_2d * dr2[np.newaxis, :]
    V_pred = E_pred + test['V_nn']

    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    n_weights = feature_dims[fs]  # single model, shared weights

    label = feature_labels[fs]
    print(f'{label:<30s} | {mae:12.4f} | {ratio:>11.2f}x | {n_weights:>10d}')

print(f'{"Global [d1,d2]→(D,R,α)":<30s} | {mae_global:12.4f} | {mae_global/mae_direct:>11.2f}x | {2*3:>10d}')
print(f'{"="*90}')

print(f'\n# weights column: Direct/Elec use {2*n_grid} weights (2 per grid point × {n_grid} points)')
print(f'#                 Adaptive uses shared weights across all r')
print(f'#                 Global uses 2×3 = 6 weights (2 descriptors × 3 params)')
print(f'\nNOTE: Direct uses {2*n_grid} independent parameters.')
print(f'      Adaptive a(r) uses only {feature_dims["basic"]}-{feature_dims["full"]} shared parameters.')
print(f'      This is the dimensionality reduction that matters.')
