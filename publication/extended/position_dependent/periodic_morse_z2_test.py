"""
Test whether Z² problem is an artifact of descriptor choice.

In real systems, nuclear charges are KNOWN per molecule. V_nn = Z_A*Z_B/r is exact.
The question: does nuclear subtraction help when descriptors encode Z information?

Descriptor sets tested:
  1. [d1, d2]           — original (can't represent Z² = (3+2d1)²)
  2. [d1, d2, d1²]      — add quadratic feature (can represent Z²)
  3. [d1, d2, Z_AB]     — give Z_AB directly (as in real DFT descriptors)
  4. [Z, d2]            — use Z instead of d1

For each descriptor set, compare:
  - Direct: d -> V_total(r)
  - Elec + V_nn: d -> E_elec(r), add back known V_nn
  - Global: d -> (D_e, R_e, alpha), reconstruct V_Morse
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# ================================================================
# Dataset (same as periodic_morse_ml.py)
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
    n_grid = len(r)

    V_total = np.zeros((n_samples, n_grid))
    V_nn = np.zeros((n_samples, n_grid))
    E_elec = np.zeros((n_samples, n_grid))

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


def make_descriptor_sets(data):
    """Create different descriptor matrices from the same data."""
    d1, d2 = data['d1'], data['d2']
    Z, Z_AB = data['Z'], data['Z_AB']
    return {
        '[d1, d2]':           np.column_stack([d1, d2]),
        '[d1, d2, d1²]':     np.column_stack([d1, d2, d1**2]),
        '[d1, d2, Z_AB]':     np.column_stack([d1, d2, Z_AB]),
        '[Z, d2]':            np.column_stack([Z, d2]),
        '[Z, d2, Z²]':       np.column_stack([Z, d2, Z**2]),
    }


def run_ml(X_train, X_test, train, test, r):
    """Run Direct, Elec+V_nn, and Global methods."""
    n_test = len(X_test)
    results = {}

    # Direct: d -> V_total
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train, train['V_total'])
    V_pred = mdl.predict(X_test)
    results['Direct'] = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())

    # Elec direct: d -> E_elec, add V_nn
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train, train['E_elec'])
    E_pred = mdl.predict(X_test)
    V_pred = E_pred + test['V_nn']
    results['Elec+V_nn'] = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())

    # Global: d -> (D_e, R_e, alpha)
    V_pred_global = np.zeros((n_test, len(r)))
    preds = []
    for j in range(3):
        mdl = Ridge(alpha=1.0)
        mdl.fit(X_train, train['params'][:, j])
        preds.append(mdl.predict(X_test))
    for i in range(n_test):
        u = np.exp(-preds[2][i] * (r - preds[1][i]))
        V_pred_global[i] = preds[0][i] * (1 - u)**2 - preds[0][i]
    results['Global'] = mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel())

    return results


# ================================================================
# Run
# ================================================================
print('Generating datasets...')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
test = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
r = train['r']

train_descs = make_descriptor_sets(train)
test_descs = make_descriptor_sets(test)

print(f'\nTraining: {len(train["d1"])} samples, Z = {train["Z"].min():.0f}-{train["Z"].max():.0f}')
print(f'Test:     {len(test["d1"])} samples, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}')

print('\n' + '='*85)
print(f'{"Descriptors":20s} | {"Direct":>12s} | {"Elec+V_nn":>12s} | {"Global":>12s} | {"Elec/Direct":>12s}')
print('='*85)

for desc_name in train_descs:
    X_train = train_descs[desc_name]
    X_test = test_descs[desc_name]
    res = run_ml(X_train, X_test, train, test, r)

    ratio = res['Elec+V_nn'] / res['Direct']
    marker = 'BETTER' if ratio < 1 else 'WORSE'

    print(f'{desc_name:20s} | {res["Direct"]:10.4f} Ha | {res["Elec+V_nn"]:10.4f} Ha | '
          f'{res["Global"]:10.4f} Ha | {ratio:10.2f}x ({marker})')

print('='*85)

print('\n--- Analysis ---')
print('If Elec+V_nn < Direct, nuclear subtraction HELPS.')
print('If Elec+V_nn > Direct, nuclear subtraction HURTS (Z² problem).')
print('\nKey question: Does adding Z² to descriptors fix the nuclear subtraction problem?')
