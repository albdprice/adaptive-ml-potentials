"""
Adaptive curvature comparison: a(r) = f(descriptor, r).

Anatole's idea: decompose E_elec(r) = a(r) * (r - r0)^2, then learn
a as a function of (descriptor, r). Reconstruct E_elec, add V_nn.

Compare ML models for learning a(d1, d2, r) -> scalar:
  - Ridge regression (linear)
  - Kernel Ridge Regression (RBF kernel, nonlinear)
  - Neural network (MLP, nonlinear)

Each model takes [d1, d2, r] as input and outputs one number: a.

Also compare against:
  - Direct multioutput Ridge: [d1,d2] -> V_total(r) (50 independent outputs)
  - Global Rose: [d1,d2] -> (D_e, R_e, alpha) -> Morse reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
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


def stack_features(data, r):
    """
    Stack (d1, d2, r) for all molecules and all grid points.
    Each molecule contributes N_grid rows.
    Returns X: (N_mol * N_grid, 3), y_a: (N_mol * N_grid,)
    """
    d1, d2 = data['d1'], data['d2']
    n_mol = len(d1)
    n_grid = len(r)

    X = np.zeros((n_mol * n_grid, 3))
    for i in range(n_mol):
        for j in range(n_grid):
            X[i * n_grid + j] = [d1[i], d2[i], r[j]]

    return X


# ================================================================
# Generate data
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
print(f'r0 = {r0} Bohr (reference point for parabola)\n')

# ================================================================
# FIGURE 1: What does a(r) actually look like?
# ================================================================
print('--- What does a(r) = E_elec / (r - r0)^2 look like? ---')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Pick representative molecules
train_indices = [0, n_train//4, n_train//2, 3*n_train//4, n_train-1]
test_indices = [0, n_test//4, n_test//2, 3*n_test//4, n_test-1]

# Row 0: Training molecules
for idx in train_indices:
    Z_i = train['Z'][idx]
    label = f'Z={Z_i:.0f}'
    axes[0, 0].plot(r, train['V_total'][idx], lw=1.5, alpha=0.7, label=label)
    axes[0, 1].plot(r, train['E_elec'][idx], lw=1.5, alpha=0.7, label=label)
    a_i = train['E_elec'][idx] / dr2
    axes[0, 2].plot(r, a_i, lw=1.5, alpha=0.7, label=label)

# Row 1: Test molecules
for idx in test_indices:
    Z_i = test['Z'][idx]
    label = f'Z={Z_i:.0f}'
    axes[1, 0].plot(r, test['V_total'][idx], lw=1.5, alpha=0.7, label=label)
    axes[1, 1].plot(r, test['E_elec'][idx], lw=1.5, alpha=0.7, label=label)
    a_i = test['E_elec'][idx] / dr2
    axes[1, 2].plot(r, a_i, lw=1.5, alpha=0.7, label=label)

axes[0, 0].set_title('$V_{total}$ (Morse)')
axes[0, 1].set_title('$E_{elec} = V_{total} - V_{nn}$')
axes[0, 2].set_title('$a(r) = E_{elec} / (r - r_0)^2$')

for ax in axes[0, :]:
    ax.legend(fontsize=7, loc='upper right')
axes[0, 0].set_ylabel('Training (Z=5-13)\nEnergy [Ha]')
axes[1, 0].set_ylabel('Test extrap (Z=17-27)\nEnergy [Ha]')

for ax in axes[:, 0]:
    ax.set_ylim(-1, 3)
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('r [Bohr]')

# Print a(r) ranges
a_train_all = train['E_elec'] / dr2[np.newaxis, :]
a_test_all = test['E_elec'] / dr2[np.newaxis, :]
print(f'  Training a(r): range [{a_train_all.min():.2f}, {a_train_all.max():.4f}]')
print(f'  Test     a(r): range [{a_test_all.min():.2f}, {a_test_all.max():.4f}]')
print(f'  a(r) varies by {a_train_all.max() - a_train_all.min():.1f} across train')
print(f'  E_elec varies by {train["E_elec"].max() - train["E_elec"].min():.1f} across train')

fig.suptitle(f'What does a(r) look like? Parabola decomposition with $r_0 = {r0:.0f}$ Bohr',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_curvature_what_a_looks_like.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_curvature_what_a_looks_like.png\n')
plt.close(fig)

# ================================================================
# ML comparison
# ================================================================

# Stacked features for shared models: [d1, d2, r] per (molecule, grid point)
X_train_stk = stack_features(train, r)  # (n_train * n_grid, 3)
X_test_stk = stack_features(test, r)    # (n_test * n_grid, 3)

# Curvature targets
a_train_flat = (train['E_elec'] / dr2[np.newaxis, :]).ravel()
a_test_flat = (test['E_elec'] / dr2[np.newaxis, :]).ravel()

# Scale features for KRR and NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_stk)
X_test_scaled = scaler.transform(X_test_stk)

# Molecular descriptors for multioutput baselines
X_mol_tr = np.column_stack([train['d1'], train['d2']])
X_mol_te = np.column_stack([test['d1'], test['d2']])

results = {}

# ---- Baseline 1: Direct multioutput Ridge ----
print('Training Direct multioutput Ridge...')
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
V_pred = mdl.predict(X_mol_te)
mae_direct = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['Direct Ridge\n(multioutput)'] = {'mae': mae_direct, 'V_pred': V_pred}
print(f'  MAE = {mae_direct:.4f} Ha')

# ---- Baseline 2: Global Rose ----
print('Training Global Rose...')
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
results['Global Rose\n(3 params)'] = {'mae': mae_global, 'V_pred': V_pred_global}
print(f'  MAE = {mae_global:.4f} Ha')

# ---- Method 1: Ridge on a(r) = f(d1, d2, r) ----
print('Training Ridge a(r) = f(d1, d2, r)...')
mdl = Ridge(alpha=1.0)
mdl.fit(X_train_stk, a_train_flat)
a_pred = mdl.predict(X_test_stk).reshape(n_test, n_grid)
E_pred = a_pred * dr2[np.newaxis, :]
V_pred = E_pred + test['V_nn']
mae_ridge_a = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['Ridge\na(d1,d2,r)'] = {'mae': mae_ridge_a, 'V_pred': V_pred}
print(f'  MAE = {mae_ridge_a:.4f} Ha')

# ---- Method 2: KRR on a(r) = f(d1, d2, r) ----
print('Training KRR a(r) = f(d1, d2, r)...')
# Use subset for KRR (full dataset too large: 20000 rows)
# Subsample training: take every 5th molecule
sub_idx = np.arange(0, n_train, 5)
X_sub = []
a_sub = []
for i in sub_idx:
    for j in range(n_grid):
        X_sub.append([train['d1'][i], train['d2'][i], r[j]])
        a_sub.append(a_train_flat[i * n_grid + j])
X_sub = np.array(X_sub)
a_sub = np.array(a_sub)

scaler_sub = StandardScaler()
X_sub_scaled = scaler_sub.fit_transform(X_sub)
X_test_scaled_sub = scaler_sub.transform(X_test_stk)

mdl = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.5)
mdl.fit(X_sub_scaled, a_sub)
a_pred = mdl.predict(X_test_scaled_sub).reshape(n_test, n_grid)
E_pred = a_pred * dr2[np.newaxis, :]
V_pred = E_pred + test['V_nn']
mae_krr_a = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
results['KRR\na(d1,d2,r)'] = {'mae': mae_krr_a, 'V_pred': V_pred}
print(f'  MAE = {mae_krr_a:.4f} Ha (trained on {len(sub_idx)} of {n_train} molecules)')

# ---- Method 3: MLP on a(r) = f(d1, d2, r) ----
print('Training MLP a(r) = f(d1, d2, r)...')
mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                   max_iter=2000, random_state=42, early_stopping=True,
                   validation_fraction=0.1, learning_rate_init=0.001)
mdl.fit(X_train_scaled, a_train_flat)
a_pred = mdl.predict(X_test_scaled).reshape(n_test, n_grid)
E_pred = a_pred * dr2[np.newaxis, :]
V_pred_mlp = E_pred + test['V_nn']
mae_mlp_a = mean_absolute_error(test['V_total'].ravel(), V_pred_mlp.ravel())
results['MLP\na(d1,d2,r)'] = {'mae': mae_mlp_a, 'V_pred': V_pred_mlp}
print(f'  MAE = {mae_mlp_a:.4f} Ha')

# ---- Method 4: Direct MLP (no parabola, learn E_elec directly) ----
print('Training MLP E_elec(d1, d2, r) directly (no parabola)...')
E_train_flat = train['E_elec'].ravel()
mdl2 = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=2000, random_state=42, early_stopping=True,
                    validation_fraction=0.1, learning_rate_init=0.001)
mdl2.fit(X_train_scaled, E_train_flat)
E_pred_direct = mdl2.predict(X_test_scaled).reshape(n_test, n_grid)
V_pred_mlp_direct = E_pred_direct + test['V_nn']
mae_mlp_direct = mean_absolute_error(test['V_total'].ravel(), V_pred_mlp_direct.ravel())
results['MLP direct\nE_elec(d1,d2,r)'] = {'mae': mae_mlp_direct, 'V_pred': V_pred_mlp_direct}
print(f'  MAE = {mae_mlp_direct:.4f} Ha')

# ---- Method 5: Direct MLP on V_total (no nuclear subtraction) ----
print('Training MLP V_total(d1, d2, r) directly...')
V_train_flat = train['V_total'].ravel()
mdl3 = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=2000, random_state=42, early_stopping=True,
                    validation_fraction=0.1, learning_rate_init=0.001)
mdl3.fit(X_train_scaled, V_train_flat)
V_pred_mlp_vtotal = mdl3.predict(X_test_scaled).reshape(n_test, n_grid)
mae_mlp_vtotal = mean_absolute_error(test['V_total'].ravel(), V_pred_mlp_vtotal.ravel())
results['MLP direct\nV_total(d1,d2,r)'] = {'mae': mae_mlp_vtotal, 'V_pred': V_pred_mlp_vtotal}
print(f'  MAE = {mae_mlp_vtotal:.4f} Ha')

# ================================================================
# Print summary table
# ================================================================
print(f'\n{"="*75}')
print(f'{"Method":<30s} | {"MAE [Ha]":>10s} | {"vs Direct Ridge":>15s} | {"Category":>12s}')
print(f'{"="*75}')

method_order = [
    'Direct Ridge\n(multioutput)',
    'Ridge\na(d1,d2,r)',
    'KRR\na(d1,d2,r)',
    'MLP\na(d1,d2,r)',
    'MLP direct\nE_elec(d1,d2,r)',
    'MLP direct\nV_total(d1,d2,r)',
    'Global Rose\n(3 params)',
]

categories = {
    'Direct Ridge\n(multioutput)': 'baseline',
    'Ridge\na(d1,d2,r)': 'parabola',
    'KRR\na(d1,d2,r)': 'parabola',
    'MLP\na(d1,d2,r)': 'parabola',
    'MLP direct\nE_elec(d1,d2,r)': 'direct',
    'MLP direct\nV_total(d1,d2,r)': 'direct',
    'Global Rose\n(3 params)': 'adaptive',
}

for name in method_order:
    mae = results[name]['mae']
    ratio = mae / mae_direct
    cat = categories[name]
    name_flat = name.replace('\n', ' ')
    if name == 'Direct Ridge\n(multioutput)':
        print(f'{name_flat:<30s} | {mae:10.4f} | {"baseline":>15s} | {cat:>12s}')
    else:
        better = 'BETTER' if ratio < 1 else 'worse'
        print(f'{name_flat:<30s} | {mae:10.4f} | {ratio:>13.2f}x  | {cat:>12s}  ({better})')

print(f'{"="*75}')

# ================================================================
# FIGURE 2: Bar chart comparison
# ================================================================
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

names_plot = [
    'Direct Ridge\n(multioutput)',
    'Ridge\na(d1,d2,r)',
    'KRR\na(d1,d2,r)',
    'MLP\na(d1,d2,r)',
    'MLP direct\nE_elec(d1,d2,r)',
    'MLP direct\nV_total(d1,d2,r)',
    'Global Rose\n(3 params)',
]

colors = {
    'baseline': '#888888',
    'parabola': '#2ca02c',
    'direct': '#d62728',
    'adaptive': '#1f77b4',
}

maes = [results[n]['mae'] for n in names_plot]
cols = [colors[categories[n]] for n in names_plot]
x = np.arange(len(names_plot))

bars = ax.bar(x, maes, color=cols, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names_plot, fontsize=9)
ax.set_ylabel('MAE [Ha]', fontsize=12)
ax.set_title('Extrapolation: Z=5-13 (train) → Z=17-27 (test)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

# Annotate ratios
for i, name in enumerate(names_plot):
    if name != 'Direct Ridge\n(multioutput)':
        ratio = maes[i] / mae_direct
        label = f'{ratio:.1f}x' if ratio >= 1 else f'{1/ratio:.1f}x better'
        ax.annotate(label, xy=(x[i], maes[i]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['baseline'], label='Baseline (Direct Ridge)'),
    Patch(facecolor=colors['parabola'], label='Parabola: learn a(r), reconstruct E=a(r-r₀)²'),
    Patch(facecolor=colors['direct'], label='Direct: learn E_elec or V_total directly'),
    Patch(facecolor=colors['adaptive'], label='Global adaptive: learn (D_e,R_e,α), Morse eq.'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

# Cap y-axis if parabola methods are way off
max_display = max(mae_direct * 5, mae_global * 50)
if max(maes) > max_display:
    ax.set_ylim(0, max_display)
    for i, m in enumerate(maes):
        if m > max_display:
            ax.annotate(f'  {m:.1f}', xy=(x[i], max_display * 0.95),
                       ha='center', fontsize=8, color='red')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_curvature_ml_comparison.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_curvature_ml_comparison.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Example predictions for one test molecule
# ================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
example_indices = [0, n_test//2, n_test-1]

for col, idx in enumerate(example_indices):
    Z_i = test['Z'][idx]
    p = test['params'][idx]

    ax = axes3[0, col]
    ax.plot(r, test['V_total'][idx], 'b-', lw=2.5, label='True $V_{total}$')

    for name, color, ls, lw in [
        ('Direct Ridge\n(multioutput)', 'gray', '--', 1.5),
        ('MLP\na(d1,d2,r)', 'green', '-', 1.5),
        ('MLP direct\nE_elec(d1,d2,r)', 'red', ':', 1.5),
        ('Global Rose\n(3 params)', 'black', '-.', 2),
    ]:
        name_short = name.replace('\n', ' ')
        ax.plot(r, results[name]['V_pred'][idx], color=color, ls=ls, lw=lw,
                label=name_short)

    ax.set_title(f'Z={Z_i:.0f}, $D_e$={p[0]:.2f}, $R_e$={p[1]:.1f}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 3)
    if col == 0:
        ax.set_ylabel('$V_{total}$ [Ha]')
        ax.legend(fontsize=7, loc='upper right')

    # Error panel
    ax2 = axes3[1, col]
    for name, color, ls in [
        ('Direct Ridge\n(multioutput)', 'gray', '--'),
        ('MLP\na(d1,d2,r)', 'green', '-'),
        ('MLP direct\nE_elec(d1,d2,r)', 'red', ':'),
        ('Global Rose\n(3 params)', 'black', '-.'),
    ]:
        err = results[name]['V_pred'][idx] - test['V_total'][idx]
        name_short = name.replace('\n', ' ')
        ax2.plot(r, err, color=color, ls=ls, lw=1.5, label=name_short)

    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('r [Bohr]')
    ax2.grid(True, alpha=0.3)
    if col == 0:
        ax2.set_ylabel('Error [Ha]')
        ax2.legend(fontsize=7)

fig3.suptitle('Extrapolation predictions: 3 example test molecules', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_curvature_ml_predictions.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_curvature_ml_predictions.png')
plt.close(fig3)
