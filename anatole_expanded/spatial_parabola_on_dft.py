"""
Can we learn E_elec(λ, d) as a shared spatial model f(λ, d)?
Same analysis we did on synthetic Morse — now on real DFT data.

Tests whether Anatole's spatial parabola E = a(d)·(d-d₀)² helps
when we enrich the feature set with λ², 1/d, λ²/d, etc.

Directly answers: does real DFT data have the same issues as synthetic?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Load data
# ================================================================
df = pd.read_csv('/Users/albd/research/alchemy_gradient/13844083/E_list_14.csv')

mol_info = {
    0: {'name': 'N₂',   'Z1': 7,  'Z2': 7,  'Z1Z2': 49},
    1: {'name': 'CO',   'Z1': 6,  'Z2': 8,  'Z1Z2': 48},
    2: {'name': 'BF',   'Z1': 5,  'Z2': 9,  'Z1Z2': 45},
    3: {'name': 'BeNe', 'Z1': 4,  'Z2': 10, 'Z1Z2': 40},
    4: {'name': 'LiNa', 'Z1': 3,  'Z2': 11, 'Z1Z2': 33},
    5: {'name': 'HeMg', 'Z1': 2,  'Z2': 12, 'Z1Z2': 24},
    6: {'name': 'HAl',  'Z1': 1,  'Z2': 13, 'Z1Z2': 13},
}

# Downsample
lambdas = sorted(df['Lambda'].unique())
d_full = df[df['Lambda'] == 0]['d'].values
step = max(1, len(d_full) // 100)
idx_keep = np.arange(0, len(d_full), step)
d_grid = d_full[idx_keep]
n_grid = len(d_grid)

E_elec = np.zeros((7, n_grid))
for i, lam in enumerate(lambdas):
    sub = df[df['Lambda'] == lam].sort_values('d')
    E_elec[i] = sub['E'].values[idx_keep]

train_idx = [0, 1, 2, 3]
test_idx = [4, 5, 6]
lam_train = np.array([lambdas[i] for i in train_idx])
lam_test = np.array([lambdas[i] for i in test_idx])

# Molecule-level scalars
Z1Z2_all = np.array([mol_info[int(l)]['Z1Z2'] for l in lambdas])
lam2_all = np.array(lambdas)**2

print(f"Grid: {n_grid} points, d ∈ [{d_grid[0]:.3f}, {d_grid[-1]:.3f}] Å")
print(f"Train: {[mol_info[int(l)]['name'] for l in lam_train]}")
print(f"Test:  {[mol_info[int(l)]['name'] for l in lam_test]}")

# Note the key molecular properties:
print(f"\nMolecular scalars:")
print(f"{'Mol':<6s} | {'λ':>4s} | {'λ²':>5s} | {'Z1Z2':>5s}")
for i in range(7):
    name = mol_info[int(lambdas[i])]['name']
    lam = lambdas[i]
    print(f"{name:<6s} | {lam:4.0f} | {lam**2:5.0f} | {Z1Z2_all[i]:5.0f}")

# ================================================================
# Parabola setup
# ================================================================
d0 = 5.0  # Å, outside data range
dr2 = (d_grid - d0)**2

# ================================================================
# Test many feature sets × {direct, parabola} × {Ridge, MLP}
# ================================================================
def make_stacked_features(lam_vals, d_grid, feature_set):
    """Build (n_mol*n_d, n_feat) feature matrix."""
    n_mol, n_d = len(lam_vals), len(d_grid)
    lam = np.array(lam_vals)

    # Molecule-level features
    lam2 = lam**2
    z1z2 = np.array([mol_info[int(l)]['Z1Z2'] for l in lam_vals])

    rows = []
    for i in range(n_mol):
        for j in range(n_d):
            d = d_grid[j]
            feat = {}
            feat['λ'] = lam[i]
            feat['d'] = d
            feat['λ²'] = lam2[i]
            feat['Z1Z2'] = z1z2[i]
            feat['1/d'] = 1.0 / d
            feat['λ²/d'] = lam2[i] / d
            feat['Z1Z2/d'] = z1z2[i] / d
            rows.append([feat[f] for f in feature_set])

    return np.array(rows)


feature_sets = {
    'A: [λ, d]':           ['λ', 'd'],
    'B: [λ, λ², d]':       ['λ', 'λ²', 'd'],
    'C: [λ, d, 1/d]':      ['λ', 'd', '1/d'],
    'D: [λ, λ², d, 1/d]':  ['λ', 'λ²', 'd', '1/d'],
    'E: [Z1Z2, d]':        ['Z1Z2', 'd'],
    'F: [Z1Z2, d, Z1Z2/d]': ['Z1Z2', 'd', 'Z1Z2/d'],
    'G: [λ, λ², d, λ²/d]': ['λ', 'λ²', 'd', 'λ²/d'],
}

# Also include multioutput baselines
print(f'\n{"="*90}')
print(f'E_elec RECONSTRUCTION ERRORS (MAE in Ha)')
print(f'{"="*90}')
print(f'{"Method":<55s} | {"MAE":>8s} | {"vs base":>8s} | per mol: {"LiNa":>6s} {"HeMg":>6s} {"HAl":>6s}')
print(f'{"-"*90}')

# Multioutput baselines
for feat_name, X_tr_feat, X_te_feat in [
    ('Multioutput [λ]',
     lam_train.reshape(-1, 1), lam_test.reshape(-1, 1)),
    ('Multioutput [λ, λ²]  (=AHA)',
     np.column_stack([lam_train, lam_train**2]),
     np.column_stack([lam_test, lam_test**2])),
]:
    mdl = Ridge(alpha=1e-6)
    mdl.fit(X_tr_feat, E_elec[train_idx])
    pred = mdl.predict(X_te_feat)
    mae = mean_absolute_error(E_elec[test_idx].ravel(), pred.ravel())
    per_mol = [mean_absolute_error(E_elec[test_idx[j]], pred[j]) for j in range(3)]
    if 'λ]' == feat_name[-2:]:
        mae_baseline = mae
    print(f'{"Ridge " + feat_name:<55s} | {mae:8.3f} | {"base" if mae == mae_baseline else f"{mae/mae_baseline:.3f}x":>8s} | {per_mol[0]:6.2f} {per_mol[1]:6.2f} {per_mol[2]:6.2f}')

print(f'{"-"*90}')

# Shared models with different feature sets
all_results = []

for feat_label, feat_list in feature_sets.items():
    X_tr = make_stacked_features(lam_train, d_grid, feat_list)
    X_te = make_stacked_features(lam_test, d_grid, feat_list)

    E_tr_flat = E_elec[train_idx].ravel()
    a_tr_flat = (E_elec[train_idx] / dr2[np.newaxis, :]).ravel()

    for target_name, target_data, recon in [
        ('E_elec direct', E_tr_flat, 'raw'),
        ('a(d) parabola', a_tr_flat, 'parabola'),
    ]:
        # Ridge
        mdl = Ridge(alpha=1e-6)
        mdl.fit(X_tr, target_data)
        pred = mdl.predict(X_te).reshape(3, n_grid)

        if recon == 'parabola':
            E_pred = pred * dr2[np.newaxis, :]
        else:
            E_pred = pred

        mae = mean_absolute_error(E_elec[test_idx].ravel(), E_pred.ravel())
        ratio = mae / mae_baseline
        per_mol = [mean_absolute_error(E_elec[test_idx[j]], E_pred[j]) for j in range(3)]

        label = f'Ridge {feat_label} → {target_name}'
        print(f'{label:<55s} | {mae:8.3f} | {ratio:8.3f}x | {per_mol[0]:6.2f} {per_mol[1]:6.2f} {per_mol[2]:6.2f}')

        all_results.append((label, mae, per_mol))

    print(f'{"-"*90}')

# MLP on the best feature sets
print(f'\nMLP results (best feature sets):')
print(f'{"-"*90}')

for feat_label, feat_list in [
    ('D: [λ, λ², d, 1/d]', ['λ', 'λ²', 'd', '1/d']),
    ('G: [λ, λ², d, λ²/d]', ['λ', 'λ²', 'd', 'λ²/d']),
]:
    X_tr = make_stacked_features(lam_train, d_grid, feat_list)
    X_te = make_stacked_features(lam_test, d_grid, feat_list)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    E_tr_flat = E_elec[train_idx].ravel()
    a_tr_flat = (E_elec[train_idx] / dr2[np.newaxis, :]).ravel()

    for target_name, target_data, recon in [
        ('E_elec direct', E_tr_flat, 'raw'),
        ('a(d) parabola', a_tr_flat, 'parabola'),
    ]:
        mdl = MLPRegressor(hidden_layer_sizes=(128, 128), activation='tanh',
                           max_iter=5000, random_state=42, early_stopping=True,
                           validation_fraction=0.15, learning_rate_init=0.001)
        mdl.fit(X_tr_sc, target_data)
        pred = mdl.predict(X_te_sc).reshape(3, n_grid)

        if recon == 'parabola':
            E_pred = pred * dr2[np.newaxis, :]
        else:
            E_pred = pred

        mae = mean_absolute_error(E_elec[test_idx].ravel(), E_pred.ravel())
        ratio = mae / mae_baseline
        per_mol = [mean_absolute_error(E_elec[test_idx[j]], E_pred[j]) for j in range(3)]

        label = f'MLP {feat_label} → {target_name}'
        print(f'{label:<55s} | {mae:8.3f} | {ratio:8.3f}x | {per_mol[0]:6.2f} {per_mol[1]:6.2f} {per_mol[2]:6.2f}')

print(f'{"-"*90}')

# ================================================================
# FIGURE: Compare all methods — sorted bar chart
# ================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Collect results for plotting
plot_data = [
    ('Multioutput [λ] (baseline)', mae_baseline),
]

# Add AHA
mdl = Ridge(alpha=1e-6)
mdl.fit(np.column_stack([lam_train, lam_train**2]), E_elec[train_idx])
pred = mdl.predict(np.column_stack([lam_test, lam_test**2]))
mae_aha = mean_absolute_error(E_elec[test_idx].ravel(), pred.ravel())
plot_data.append(('Multioutput [λ, λ²] (AHA)', mae_aha))

# Add shared model results
for label, mae, _ in all_results:
    plot_data.append((label.replace('Ridge ', ''), mae))

# Sort by MAE
plot_data.sort(key=lambda x: x[1])

labels = [p[0] for p in plot_data]
maes = [p[1] for p in plot_data]

colors = []
for l in labels:
    if 'AHA' in l:
        colors.append('forestgreen')
    elif 'parabola' in l:
        colors.append('coral')
    elif 'baseline' in l:
        colors.append('steelblue')
    else:
        colors.append('#aec7e8')

bars = ax.barh(range(len(labels)), maes, color=colors)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('MAE [Ha]')
ax.set_title('E_elec prediction error — shared spatial models vs multioutput\n(Red=parabola, Green=AHA, Blue=direct)')

for i, (mae, bar) in enumerate(zip(maes, bars)):
    ax.text(mae + 0.5, i, f'{mae:.2f}', va='center', fontsize=8)

ax.grid(True, alpha=0.3, axis='x')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_spatial_parabola_comparison.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_spatial_parabola_comparison.png')
plt.close(fig)

# ================================================================
# Key comparison: parabola vs direct at each feature set
# ================================================================
print(f'\n{"="*60}')
print(f'PARABOLA vs DIRECT at each feature set (Ridge)')
print(f'{"="*60}')
print(f'{"Feature set":<25s} | {"Direct":>8s} | {"Parabola":>8s} | {"Ratio":>8s}')
print(f'{"-"*60}')

for feat_label, feat_list in feature_sets.items():
    X_tr = make_stacked_features(lam_train, d_grid, feat_list)
    X_te = make_stacked_features(lam_test, d_grid, feat_list)

    E_tr_flat = E_elec[train_idx].ravel()
    a_tr_flat = (E_elec[train_idx] / dr2[np.newaxis, :]).ravel()

    # Direct
    mdl = Ridge(alpha=1e-6)
    mdl.fit(X_tr, E_tr_flat)
    pred = mdl.predict(X_te).reshape(3, n_grid)
    mae_dir = mean_absolute_error(E_elec[test_idx].ravel(), pred.ravel())

    # Parabola
    mdl = Ridge(alpha=1e-6)
    mdl.fit(X_tr, a_tr_flat)
    pred = mdl.predict(X_te).reshape(3, n_grid)
    E_pred = pred * dr2[np.newaxis, :]
    mae_par = mean_absolute_error(E_elec[test_idx].ravel(), E_pred.ravel())

    ratio = mae_par / mae_dir
    better = '← parabola better' if ratio < 0.95 else ('← direct better' if ratio > 1.05 else '≈ same')
    print(f'{feat_label:<25s} | {mae_dir:8.3f} | {mae_par:8.3f} | {ratio:8.3f}x {better}')

print(f'{"-"*60}')
print(f'\nConclusion: Does real DFT show the same issues as synthetic?')
