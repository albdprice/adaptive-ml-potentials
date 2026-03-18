"""
Adaptive curvature v2: a(r) = f(descriptor, r), SINGLE shared model.

Key insight from v1: E_elec curvature has Z² scaling that Ridge can't learn.
Question: what if we do the parabola on V_total instead?

    V_total(r) = a_V(r) * (r - r_0)^2    (parabola on total energy)

V_total = Morse, which has only LINEAR d1 dependence. No Z² problem.

Also test: since V_nn is KNOWN, subtract it from the parabola reconstruction:
    E_elec_pred = a_V(r) * (r - r_0)^2 - V_nn    (nonsense, ignore this)

Actually the clean comparison is:
  A) a_V(r) on V_total: learn a_V = f(d1,d2,r), reconstruct V = a_V*(r-r0)²
  B) a_E(r) on E_elec:  learn a_E = f(d1,d2,r), reconstruct E = a_E*(r-r0)², V = E + V_nn
  C) Direct multioutput: [d1,d2] → V(r), 50 independent weight vectors
  D) Global Rose:        [d1,d2] → (D_e,R_e,α), reconstruct Morse
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
print(f'Test:     {n_test} molecules, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}')
print(f'r0 = {r0}, grid: {r[0]:.1f} to {r[-1]:.1f} Bohr, {n_grid} points')

# ---- Baseline: Direct multioutput ----
X_mol_train = np.column_stack([train['d1'], train['d2']])
X_mol_test = np.column_stack([test['d1'], test['d2']])

mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_train, train['V_total'])
V_pred_direct = mdl.predict(X_mol_test)
mae_direct = mean_absolute_error(test['V_total'].ravel(), V_pred_direct.ravel())

# ---- Global Rose ----
V_pred_global = np.zeros((n_test, n_grid))
ppreds = []
for j in range(3):
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_mol_train, train['params'][:, j])
    ppreds.append(mdl.predict(X_mol_test))
for i in range(n_test):
    u = np.exp(-ppreds[2][i] * (r - ppreds[1][i]))
    V_pred_global[i] = ppreds[0][i] * (1 - u)**2 - ppreds[0][i]
mae_global = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())


# ================================================================
# Adaptive curvature: build stacked (descriptor, r) features
# ================================================================
def build_stacked(data, r, feature_set):
    """Build (N_mol*N_grid, n_feat) feature matrix."""
    d1, d2 = data['d1'], data['d2']
    Z_AB = data['Z_AB']
    n_mol = len(d1)
    n_grid = len(r)

    rows = []
    for i in range(n_mol):
        for j in range(n_grid):
            rj = r[j]
            if feature_set == 'basic':
                rows.append([d1[i], d2[i], rj])
            elif feature_set == 'interact':
                rows.append([d1[i], d2[i], rj, d1[i]*rj, d2[i]*rj, rj**2])
            elif feature_set == 'interact_Z':
                rows.append([d1[i], d2[i], Z_AB[i], rj,
                           d1[i]*rj, d2[i]*rj, Z_AB[i]/rj, rj**2])
    return np.array(rows)


print(f'\n{"="*95}')
print(f'{"Method":<40s} | {"MAE [Ha]":>10s} | {"vs Direct":>10s} | {"# params":>8s}')
print(f'{"="*95}')
print(f'{"Direct [d1,d2]→V(r) (multioutput)":<40s} | {mae_direct:10.4f} | {"baseline":>10s} | {2*n_grid:>8d}')

# ---- A) Adaptive curvature on V_TOTAL ----
for fs, label, n_feat in [
    ('basic', 'a_V [d1,d2,r]→a', 3),
    ('interact', 'a_V [d1,d2,r,d1r,d2r,r²]→a', 6),
]:
    X_tr = build_stacked(train, r, fs)
    X_te = build_stacked(test, r, fs)

    # Target: a_V = V_total / (r - r0)^2
    a_tr = (train['V_total'] / dr2[np.newaxis, :]).ravel()

    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, a_tr)
    a_pred = mdl.predict(X_te).reshape(n_test, n_grid)

    V_pred = a_pred * dr2[np.newaxis, :]
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else 'worse'
    print(f'{"  V_total: " + label:<40s} | {mae:10.4f} | {ratio:>9.2f}x | {n_feat:>8d}  ({marker})')

# ---- B) Adaptive curvature on E_ELEC ----
for fs, label, n_feat in [
    ('basic', 'a_E [d1,d2,r]→a', 3),
    ('interact', 'a_E [d1,d2,r,d1r,d2r,r²]→a', 6),
    ('interact_Z', 'a_E [d1,d2,Z²,r,...]→a', 8),
]:
    X_tr = build_stacked(train, r, fs)
    X_te = build_stacked(test, r, fs)

    # Target: a_E = E_elec / (r - r0)^2
    a_tr = (train['E_elec'] / dr2[np.newaxis, :]).ravel()

    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr, a_tr)
    a_pred = mdl.predict(X_te).reshape(n_test, n_grid)

    E_pred = a_pred * dr2[np.newaxis, :]
    V_pred = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    ratio = mae / mae_direct
    marker = 'BETTER' if ratio < 1 else 'worse'
    print(f'{"  E_elec:  " + label:<40s} | {mae:10.4f} | {ratio:>9.2f}x | {n_feat:>8d}  ({marker})')

# ---- Elec direct multioutput (for comparison) ----
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_train, train['E_elec'])
E_pred = mdl.predict(X_mol_test)
V_pred = E_pred + test['V_nn']
mae_elec = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
print(f'{"Elec direct [d1,d2]→E_elec (multioutput)":<40s} | {mae_elec:10.4f} | {mae_elec/mae_direct:>9.1f}x | {2*n_grid:>8d}  (worse)')

print(f'{"Global Rose [d1,d2]→(D,R,α)":<40s} | {mae_global:10.4f} | {mae_global/mae_direct:>9.2f}x | {6:>8d}  (BETTER)')
print(f'{"="*95}')

print(f'\nKey comparison:')
print(f'  Adaptive a_V on V_total — does shared model + parabola help vs multioutput direct?')
print(f'  Adaptive a_E on E_elec  — does nuclear subtraction help or hurt?')
print(f'  Global Rose             — still the gold standard (3 params + nonlinear physics)')
