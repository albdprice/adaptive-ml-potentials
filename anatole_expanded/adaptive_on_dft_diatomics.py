"""
Test adaptive vs direct ML on REAL DFT data from Devika's thesis.

Dataset: 14-electron isoelectronic diatomic series (PBE0/cc-pVDZ)
  λ=0: N₂   (7,7)      λ=3: BeNe (4,10)     λ=6: HAl (1,13)
  λ=1: CO   (6,8)      λ=4: LiNa (3,11)
  λ=2: BF   (5,9)      λ=5: HeMg (2,12)

Extrapolation: train on λ=0-3 (N₂,CO,BF,BeNe), test on λ=4-6 (LiNa,HeMg,HAl)

Target: E_elec(λ, d) — DFT electronic energy (no nuclear repulsion)

Methods compared:
  1. Direct multioutput Ridge: λ → E_elec(d) at grid points (linear in λ)
  2. AHA-style multioutput Ridge: [λ, λ²] → E_elec(d) (quadratic in λ = Anatole's AHA)
  3. Anatole's spatial parabola (shared): (λ, d) → a, reconstruct E = a·(d-d₀)²
  4. Direct shared Ridge: (λ, d) → E_elec (no parabola)
  5-6. MLP versions of 3-4
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Load DFT data
# ================================================================
import pandas as pd
data_path = '/Users/albd/research/alchemy_gradient/13844083/E_list_14.csv'
df = pd.read_csv(data_path)

mol_info = {
    0: {'name': 'N₂',   'Z1': 7,  'Z2': 7},
    1: {'name': 'CO',   'Z1': 6,  'Z2': 8},
    2: {'name': 'BF',   'Z1': 5,  'Z2': 9},
    3: {'name': 'BeNe', 'Z1': 4,  'Z2': 10},
    4: {'name': 'LiNa', 'Z1': 3,  'Z2': 11},
    5: {'name': 'HeMg', 'Z1': 2,  'Z2': 12},
    6: {'name': 'HAl',  'Z1': 1,  'Z2': 13},
}

# Downsample 1024 → ~100 points
lambdas = sorted(df['Lambda'].unique())
d_full = df[df['Lambda'] == 0]['d'].values
step = max(1, len(d_full) // 100)
idx_keep = np.arange(0, len(d_full), step)
d_grid = d_full[idx_keep]
n_grid = len(d_grid)

# Build E_elec array [mol, d]
E_elec = np.zeros((len(lambdas), n_grid))
for i, lam in enumerate(lambdas):
    sub = df[df['Lambda'] == lam].sort_values('d')
    E_elec[i] = sub['E'].values[idx_keep]

print(f"Grid: {n_grid} points, d ∈ [{d_grid[0]:.3f}, {d_grid[-1]:.3f}] Å")
print(f"Molecules: {len(lambdas)}")
for i, lam in enumerate(lambdas):
    name = mol_info[lam]['name']
    print(f"  λ={lam:.0f} {name:5s}: E_elec ∈ [{E_elec[i].min():.2f}, {E_elec[i].max():.2f}] Ha")

# ================================================================
# Train/test split
# ================================================================
train_idx = [0, 1, 2, 3]  # N₂, CO, BF, BeNe
test_idx = [4, 5, 6]       # LiNa, HeMg, HAl

lam_train = np.array([lambdas[i] for i in train_idx])
lam_test = np.array([lambdas[i] for i in test_idx])

print(f"\nTrain: {[mol_info[int(l)]['name'] for l in lam_train]}")
print(f"Test:  {[mol_info[int(l)]['name'] for l in lam_test]}")

# ================================================================
# FIGURE 1: The raw data
# ================================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))

for i in train_idx:
    name = mol_info[lambdas[i]]['name']
    ax1.plot(d_grid, E_elec[i], lw=1.5, color='#1f77b4', alpha=0.7,
             label=f'{name} (λ={lambdas[i]:.0f})' if i in [0, 3] else f'{name}')

for i in test_idx:
    name = mol_info[lambdas[i]]['name']
    ax1.plot(d_grid, E_elec[i], lw=1.5, color='#d62728', alpha=0.7, ls='--',
             label=f'{name} (λ={lambdas[i]:.0f})' if i in [4, 6] else f'{name}')

ax1.set_xlabel('d [Å]', fontsize=12)
ax1.set_ylabel('$E_{elec}$ [Ha]', fontsize=12)
ax1.set_title('14e⁻ isoelectronic series — DFT electronic energy\nBlue=train (λ=0-3), Red=test (λ=4-6)', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'fig_Eelec_overview.png'), dpi=150, bbox_inches='tight')
print('\nSaved fig_Eelec_overview.png')
plt.close(fig1)

# ================================================================
# Method 1: Direct multioutput Ridge — λ → E_elec(d), linear in λ
# ================================================================
X_tr = lam_train.reshape(-1, 1)
X_te = lam_test.reshape(-1, 1)

ridge1 = Ridge(alpha=1e-6)
ridge1.fit(X_tr, E_elec[train_idx])
E_pred_1 = ridge1.predict(X_te)
mae_1 = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_1.ravel())

# ================================================================
# Method 2: AHA-style multioutput — [λ, λ²] → E_elec(d), quadratic in λ
# ================================================================
X_tr2 = np.column_stack([lam_train, lam_train**2])
X_te2 = np.column_stack([lam_test, lam_test**2])

ridge2 = Ridge(alpha=1e-6)
ridge2.fit(X_tr2, E_elec[train_idx])
E_pred_2 = ridge2.predict(X_te2)
mae_2 = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_2.ravel())

# ================================================================
# Methods 3-6: Shared models — (λ, d) → target
# ================================================================
def stack_features(lam_vals, d_grid):
    n_mol, n_d = len(lam_vals), len(d_grid)
    X = np.zeros((n_mol * n_d, 2))
    for i in range(n_mol):
        for j in range(n_d):
            X[i * n_d + j] = [lam_vals[i], d_grid[j]]
    return X

X_stk_tr = stack_features(lam_train, d_grid)
X_stk_te = stack_features(lam_test, d_grid)

scaler = StandardScaler()
X_stk_tr_sc = scaler.fit_transform(X_stk_tr)
X_stk_te_sc = scaler.transform(X_stk_te)

# Parabola: E_elec(d) = a(d) · (d - d₀)²
d0 = 5.0  # far from data range [0.7, 2.5]
dr2 = (d_grid - d0)**2

E_train_flat = E_elec[train_idx].ravel()
a_train_flat = (E_elec[train_idx] / dr2[np.newaxis, :]).ravel()

# ================================================================
# Results table
# ================================================================
print(f'\n{"="*80}')
print(f'E_elec RECONSTRUCTION ERRORS (MAE in Ha)')
print(f'{"="*80}')
print(f'{"Method":<50s} | {"MAE (Ha)":>10s} | {"vs Direct":>10s}')
print(f'{"-"*80}')

results = {}

# Method 1
label = 'Direct multioutput Ridge [λ]→E(d)'
results[label] = mae_1
print(f'{label:<50s} | {mae_1:10.4f} | {"baseline":>10s}')
mae_baseline = mae_1

# Method 2 — AHA quadratic
label = 'AHA-style multioutput Ridge [λ,λ²]→E(d)'
results[label] = mae_2
print(f'{label:<50s} | {mae_2:10.4f} | {mae_2/mae_baseline:10.2f}x')

print(f'{"-"*80}')

# Methods 3-6: shared models
shared_preds = {}
for model_type in ['Ridge', 'MLP']:
    for target_name, target_data, recon in [
        ('E_elec (direct)', E_train_flat, 'raw'),
        ('a(d) parabola', a_train_flat, 'parabola'),
    ]:
        if model_type == 'Ridge':
            mdl = Ridge(alpha=1e-6)
            mdl.fit(X_stk_tr, target_data)
            pred = mdl.predict(X_stk_te).reshape(len(test_idx), n_grid)
        else:
            mdl = MLPRegressor(hidden_layer_sizes=(128, 128), activation='tanh',
                               max_iter=5000, random_state=42, early_stopping=True,
                               validation_fraction=0.15, learning_rate_init=0.001)
            mdl.fit(X_stk_tr_sc, target_data)
            pred = mdl.predict(X_stk_te_sc).reshape(len(test_idx), n_grid)

        if recon == 'raw':
            E_pred = pred
        elif recon == 'parabola':
            E_pred = pred * dr2[np.newaxis, :]

        mae = mean_absolute_error(E_elec[test_idx].ravel(), E_pred.ravel())
        ratio = mae / mae_baseline
        label = f'{model_type} shared [{chr(955)},d]→{target_name}'
        results[label] = mae
        shared_preds[label] = E_pred
        print(f'{label:<50s} | {mae:10.4f} | {ratio:10.2f}x')

    print(f'{"-"*80}')

print(f'{"="*80}')

# ================================================================
# Per-molecule breakdown
# ================================================================
test_names = [mol_info[int(lam_test[j])]['name'] for j in range(len(test_idx))]

print(f'\nPer-molecule MAE (Ha):')
header = f'{"Method":<40s}' + ''.join(f' | {n:>8s}' for n in test_names)
print(header)
print(f'{"-"*75}')

def print_per_mol(label, E_pred):
    print(f'{label:<40s}', end='')
    for j in range(len(test_idx)):
        mae_j = mean_absolute_error(E_elec[test_idx[j]], E_pred[j])
        print(f' | {mae_j:8.3f}', end='')
    print()

print_per_mol('Direct multioutput [λ]', E_pred_1)
print_per_mol('AHA multioutput [λ,λ²]', E_pred_2)
for label, E_pred in shared_preds.items():
    short = label.replace(f'{chr(955)}', 'λ')
    print_per_mol(short[:40], E_pred)

# ================================================================
# FIGURE 2: Predictions for each test molecule
# ================================================================
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

for j, ax in enumerate(axes):
    idx = test_idx[j]
    name = mol_info[lambdas[idx]]['name']

    ax.plot(d_grid, E_elec[idx], 'k-', lw=2.5, label='DFT reference')

    mae_j = mean_absolute_error(E_elec[idx], E_pred_1[j])
    ax.plot(d_grid, E_pred_1[j], 'b--', lw=1.5,
            label=f'Direct Ridge (MAE={mae_j:.2f})')

    mae_j = mean_absolute_error(E_elec[idx], E_pred_2[j])
    ax.plot(d_grid, E_pred_2[j], 'g--', lw=1.5,
            label=f'AHA [λ,λ²] (MAE={mae_j:.2f})')

    # Best shared model
    best_shared = min(shared_preds.items(), key=lambda x: mean_absolute_error(
        E_elec[test_idx].ravel(), x[1].ravel()))
    mae_j = mean_absolute_error(E_elec[idx], best_shared[1][j])
    ax.plot(d_grid, best_shared[1][j], 'r--', lw=1.5,
            label=f'Best shared (MAE={mae_j:.2f})')

    ax.set_xlabel('d [Å]')
    ax.set_ylabel('$E_{elec}$ [Ha]')
    ax.set_title(f'{name} (λ={lambdas[idx]:.0f})')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

fig2.suptitle('Extrapolation: train on N₂,CO,BF,BeNe → predict LiNa,HeMg,HAl', fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_Eelec_predictions.png'), dpi=150, bbox_inches='tight')
print('\nSaved fig_Eelec_predictions.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: AHA-style — E_elec vs λ at several fixed distances
# ================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9))

d_samples = [0.7, 1.0, 1.3, 1.6, 2.0, 2.4]

for ax, d_target in zip(axes3.flat, d_samples):
    j_d = np.argmin(np.abs(d_grid - d_target))
    d_actual = d_grid[j_d]

    # E_elec at this d for all molecules
    E_at_d = E_elec[:, j_d]
    lam_all = np.array(lambdas)

    ax.plot(lam_all[train_idx], E_at_d[train_idx], 'bo', ms=8, label='Train')
    ax.plot(lam_all[test_idx], E_at_d[test_idx], 'rs', ms=8, label='Test (true)')

    # Linear fit (Ridge with λ)
    m1 = Ridge(alpha=1e-6)
    m1.fit(lam_train.reshape(-1, 1), E_at_d[train_idx])
    lam_dense = np.linspace(-0.5, 7, 100)
    ax.plot(lam_dense, m1.predict(lam_dense.reshape(-1, 1)), 'b-', alpha=0.5, label='Linear')

    # Quadratic fit (AHA-style with [λ, λ²])
    m2 = Ridge(alpha=1e-6)
    m2.fit(np.column_stack([lam_train, lam_train**2]), E_at_d[train_idx])
    ax.plot(lam_dense, m2.predict(np.column_stack([lam_dense, lam_dense**2])),
            'g-', alpha=0.5, label='Quadratic (AHA)')

    ax.set_xlabel('λ')
    ax.set_ylabel('$E_{elec}$ [Ha]')
    ax.set_title(f'd = {d_actual:.3f} Å')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig3.suptitle('E_elec vs λ at fixed distances — is it parabolic?', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_Eelec_vs_lambda.png'), dpi=150, bbox_inches='tight')
print('Saved fig_Eelec_vs_lambda.png')
plt.close(fig3)

# ================================================================
# FIGURE 4: Anatole's a(d) = E_elec/(d-d₀)² for each molecule
# ================================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

for i in train_idx:
    name = mol_info[lambdas[i]]['name']
    ax4a.plot(d_grid, E_elec[i] / dr2, lw=1.5, color='#1f77b4', alpha=0.7, label=name)

for i in test_idx:
    name = mol_info[lambdas[i]]['name']
    ax4a.plot(d_grid, E_elec[i] / dr2, lw=1.5, color='#d62728', alpha=0.7, ls='--', label=name)

ax4a.set_xlabel('d [Å]')
ax4a.set_ylabel(f'$a(d) = E_{{elec}} / (d - {d0:.0f})^2$')
ax4a.set_title(f'Curvature targets a(d) with d₀={d0:.0f} Å')
ax4a.legend(fontsize=8)
ax4a.grid(True, alpha=0.3)

# Also show E_elec vs λ² at d≈1.1 Å (near N₂ equilibrium)
j_eq = np.argmin(np.abs(d_grid - 1.1))
E_at_eq = E_elec[:, j_eq]
lam_all = np.array(lambdas)

ax4b.plot(lam_all**2, E_at_eq, 'ko-', ms=8)
for i in train_idx:
    ax4b.annotate(mol_info[lambdas[i]]['name'], (lam_all[i]**2, E_at_eq[i]),
                  textcoords='offset points', xytext=(5, 5), fontsize=9, color='blue')
for i in test_idx:
    ax4b.annotate(mol_info[lambdas[i]]['name'], (lam_all[i]**2, E_at_eq[i]),
                  textcoords='offset points', xytext=(5, 5), fontsize=9, color='red')

ax4b.set_xlabel('$λ^2$')
ax4b.set_ylabel('$E_{elec}$ [Ha]')
ax4b.set_title(f'E_elec vs λ² at d={d_grid[j_eq]:.3f} Å — linearity test')
ax4b.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(os.path.join(FIGDIR, 'fig_curvature_and_lambda2.png'), dpi=150, bbox_inches='tight')
print('Saved fig_curvature_and_lambda2.png')
plt.close(fig4)

# ================================================================
# Summary
# ================================================================
print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'{"="*60}')
print(f'')
print(f'E_elec range:')
print(f'  Train (λ=0-3): [{E_elec[train_idx].min():.1f}, {E_elec[train_idx].max():.1f}]')
print(f'  Test  (λ=4-6): [{E_elec[test_idx].min():.1f}, {E_elec[test_idx].max():.1f}]')
print(f'  Ratio: {(E_elec[test_idx].min() - E_elec[test_idx].max()) / (E_elec[train_idx].min() - E_elec[train_idx].max()):.1f}x')
print(f'')
print(f'Best method: {min(results, key=results.get)}')
print(f'  MAE = {min(results.values()):.4f} Ha')
print(f'')
print(f'Key insight: E_elec is dominated by atomic energies that')
print(f'scale roughly as ~Z², making it approximately parabolic')
print(f'in λ = (Z2-Z1)/2. The AHA quadratic model captures this.')
print(f'{"="*60}')
