"""
Adaptive curvature comparison v2: fixes from discussion.

Fix 1: Include Z_AB as input feature [d1, d2, Z_AB, r].
        Z is KNOWN per molecule — it's an input, not a prediction target.

Fix 2: Test r0 = 10, 20, 50 Bohr. Larger r0 makes (r-r0)^2 more uniform
        across the grid, reducing distortion.

Methods:
  - Ridge:  a = f(d1, d2, Z_AB, r) — linear model, shared across r
  - KRR:    a = f(d1, d2, Z_AB, r) — kernel model (can it extrapolate with Z_AB?)
  - MLP:    a = f(d1, d2, Z_AB, r) — neural network, shared across r
  - MLP direct E_elec: no parabola, learn E_elec(d1, d2, Z_AB, r) directly
  - Direct Ridge multioutput: [d1,d2] -> V_total(r) (50 outputs, baseline)
  - Global Rose: [d1,d2] -> (D_e, R_e, alpha) -> Morse (gold standard)
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
    """Stack [d1, d2, Z_AB, r] for all (molecule, grid point) pairs."""
    d1, d2, Z_AB = data['d1'], data['d2'], data['Z_AB']
    n_mol, n_grid = len(d1), len(r)
    X = np.zeros((n_mol * n_grid, 4))
    for i in range(n_mol):
        for j in range(n_grid):
            X[i * n_grid + j] = [d1[i], d2[i], Z_AB[i], r[j]]
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

print(f'Training: {n_train} molecules, Z = {train["Z"].min():.0f}-{train["Z"].max():.0f}')
print(f'Test:     {n_test} molecules, Z = {test["Z"].min():.0f}-{test["Z"].max():.0f}')

# Stacked features [d1, d2, Z_AB, r]
X_train_stk = stack_features(train, r)
X_test_stk = stack_features(test, r)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_stk)
X_test_scaled = scaler.transform(X_test_stk)

# Molecular descriptors for multioutput baselines
X_mol_tr = np.column_stack([train['d1'], train['d2']])
X_mol_te = np.column_stack([test['d1'], test['d2']])

# ================================================================
# Baselines (don't depend on r0)
# ================================================================
# Direct Ridge multioutput
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['V_total'])
V_pred_direct = mdl.predict(X_mol_te)
mae_direct = mean_absolute_error(test['V_total'].ravel(), V_pred_direct.ravel())

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

# MLP direct E_elec (no parabola, with Z_AB input)
print('Training MLP E_elec direct (no parabola)...')
E_train_flat = train['E_elec'].ravel()
mdl_mlp_direct = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                               max_iter=2000, random_state=42, early_stopping=True,
                               validation_fraction=0.1)
mdl_mlp_direct.fit(X_train_scaled, E_train_flat)
E_pred_direct_mlp = mdl_mlp_direct.predict(X_test_scaled).reshape(n_test, n_grid)
V_pred_direct_mlp = E_pred_direct_mlp + test['V_nn']
mae_mlp_direct = mean_absolute_error(test['V_total'].ravel(), V_pred_direct_mlp.ravel())

print(f'\n--- Baselines (independent of r0) ---')
print(f'  Direct Ridge multioutput:  {mae_direct:.4f} Ha (baseline)')
print(f'  Global Rose:               {mae_global:.4f} Ha ({mae_global/mae_direct:.2f}x)')
print(f'  MLP direct E_elec:         {mae_mlp_direct:.4f} Ha ({mae_mlp_direct/mae_direct:.2f}x)')

# ================================================================
# Test different r0 values
# ================================================================
r0_values = [10.0, 20.0, 50.0]

# Store all results for plotting
all_results = {}

for r0 in r0_values:
    dr2 = (r - r0)**2
    dr2_ratio = dr2.max() / dr2.min()

    # Curvature targets
    a_train = (train['E_elec'] / dr2[np.newaxis, :]).ravel()
    a_test_true = test['E_elec'] / dr2[np.newaxis, :]

    a_range_train = [a_train.min(), a_train.max()]
    a_range_test = [a_test_true.min(), a_test_true.max()]

    print(f'\n{"="*80}')
    print(f'r0 = {r0:.0f} Bohr')
    print(f'  (r-r0)^2 range: [{dr2.min():.0f}, {dr2.max():.0f}], ratio = {dr2_ratio:.1f}x')
    print(f'  Training a(r):  [{a_range_train[0]:.3f}, {a_range_train[1]:.4f}]')
    print(f'  Test a(r):      [{a_range_test[0]:.3f}, {a_range_test[1]:.4f}]')
    print(f'{"="*80}')

    results_r0 = {}

    # --- Ridge ---
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train_stk, a_train)
    a_pred = mdl.predict(X_test_stk).reshape(n_test, n_grid)
    E_pred = a_pred * dr2[np.newaxis, :]
    V_pred = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    results_r0['Ridge'] = mae
    print(f'  Ridge a(d1,d2,Z²,r):   {mae:.4f} Ha  ({mae/mae_direct:.1f}x vs direct)')

    # --- KRR (subsample for speed) ---
    sub_idx = np.arange(0, n_train, 5)  # every 5th molecule
    X_sub, a_sub = [], []
    for i in sub_idx:
        for j in range(n_grid):
            X_sub.append([train['d1'][i], train['d2'][i], train['Z_AB'][i], r[j]])
            a_sub.append(train['E_elec'][i, j] / dr2[j])
    X_sub = np.array(X_sub)
    a_sub = np.array(a_sub)
    scaler_sub = StandardScaler()
    X_sub_sc = scaler_sub.fit_transform(X_sub)
    X_test_sc = scaler_sub.transform(X_test_stk)

    mdl = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.5)
    mdl.fit(X_sub_sc, a_sub)
    a_pred = mdl.predict(X_test_sc).reshape(n_test, n_grid)
    E_pred = a_pred * dr2[np.newaxis, :]
    V_pred = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred.ravel())
    results_r0['KRR'] = mae
    print(f'  KRR a(d1,d2,Z²,r):     {mae:.4f} Ha  ({mae/mae_direct:.1f}x vs direct)')

    # --- MLP ---
    mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                       max_iter=2000, random_state=42, early_stopping=True,
                       validation_fraction=0.1)
    mdl.fit(X_train_scaled, a_train)
    a_pred = mdl.predict(X_test_scaled).reshape(n_test, n_grid)
    E_pred = a_pred * dr2[np.newaxis, :]
    V_pred_mlp = E_pred + test['V_nn']
    mae = mean_absolute_error(test['V_total'].ravel(), V_pred_mlp.ravel())
    results_r0['MLP'] = mae
    print(f'  MLP a(d1,d2,Z²,r):     {mae:.4f} Ha  ({mae/mae_direct:.1f}x vs direct)')

    all_results[r0] = results_r0

# ================================================================
# Summary table
# ================================================================
print(f'\n\n{"="*90}')
print(f'SUMMARY TABLE')
print(f'{"="*90}')
print(f'{"Method":<35s} |', end='')
for r0 in r0_values:
    print(f'  r0={r0:.0f} Bohr  |', end='')
print()
print(f'{"-"*90}')

for method in ['Ridge', 'KRR', 'MLP']:
    label = f'Parabola {method} a(d1,d2,Z²,r)'
    print(f'{label:<35s} |', end='')
    for r0 in r0_values:
        mae = all_results[r0][method]
        ratio = mae / mae_direct
        print(f'  {mae:7.4f} ({ratio:6.1f}x) |', end='')
    print()

print(f'{"-"*90}')
print(f'{"MLP direct E_elec (no parabola)":<35s} |', end='')
for _ in r0_values:
    print(f'  {mae_mlp_direct:7.4f} ({mae_mlp_direct/mae_direct:6.1f}x) |', end='')
print()
print(f'{"Direct Ridge multioutput":<35s} |', end='')
for _ in r0_values:
    print(f'  {mae_direct:7.4f} (  1.0x) |', end='')
print()
print(f'{"Global Rose (D_e, R_e, alpha)":<35s} |', end='')
for _ in r0_values:
    print(f'  {mae_global:7.4f} ({mae_global/mae_direct:6.2f}x) |', end='')
print()
print(f'{"="*90}')

# ================================================================
# FIGURE 1: What a(r) looks like for different r0
# ================================================================
fig, axes = plt.subplots(2, len(r0_values), figsize=(6*len(r0_values), 8))

for col, r0 in enumerate(r0_values):
    dr2 = (r - r0)**2

    # Training
    for i in range(0, n_train, 50):
        Z_i = train['Z'][i]
        a_i = train['E_elec'][i] / dr2
        axes[0, col].plot(r, a_i, lw=1, alpha=0.5)
    axes[0, col].set_title(f'$r_0 = {r0:.0f}$ Bohr\n(train, Z={train["Z"].min():.0f}-{train["Z"].max():.0f})')
    axes[0, col].set_ylabel('$a(r) = E_{elec}/(r-r_0)^2$')
    axes[0, col].grid(True, alpha=0.3)

    # Test
    for i in range(0, n_test, 10):
        Z_i = test['Z'][i]
        a_i = test['E_elec'][i] / dr2
        axes[1, col].plot(r, a_i, lw=1, alpha=0.5)
    axes[1, col].set_title(f'(test extrap, Z={test["Z"].min():.0f}-{test["Z"].max():.0f})')
    axes[1, col].set_ylabel('$a(r)$')
    axes[1, col].set_xlabel('r [Bohr]')
    axes[1, col].grid(True, alpha=0.3)

    # Show (r-r0)^2 ratio
    axes[0, col].text(0.05, 0.95,
                      f'$(r-r_0)^2$: [{dr2.min():.0f}, {dr2.max():.0f}]\nratio: {dr2.max()/dr2.min():.1f}x',
                      transform=axes[0, col].transAxes, fontsize=8, va='top',
                      bbox=dict(boxstyle='round', fc='white', alpha=0.8))

fig.suptitle('Curvature $a(r) = E_{elec}/(r - r_0)^2$ for different reference points $r_0$', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_curvature_v2_a_vs_r0.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_curvature_v2_a_vs_r0.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Bar chart comparison (best r0 for each method)
# ================================================================
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

methods_plot = []
maes_plot = []
colors_plot = []
labels_plot = []

# Baselines
methods_plot.append('Direct Ridge\nmultioutput')
maes_plot.append(mae_direct)
colors_plot.append('#888888')

# Best parabola result per model
for method, color in [('Ridge', '#2ca02c'), ('KRR', '#ff7f0e'), ('MLP', '#2ca02c')]:
    best_r0 = min(r0_values, key=lambda r0: all_results[r0][method])
    best_mae = all_results[best_r0][method]
    methods_plot.append(f'Parabola {method}\na(d1,d2,Z²,r)\nr0={best_r0:.0f}')
    maes_plot.append(best_mae)
    colors_plot.append(color)

# MLP direct
methods_plot.append('MLP direct\nE_elec(d1,d2,Z²,r)')
maes_plot.append(mae_mlp_direct)
colors_plot.append('#d62728')

# Global
methods_plot.append('Global Rose\n(D_e,R_e,α)')
maes_plot.append(mae_global)
colors_plot.append('#1f77b4')

x = np.arange(len(methods_plot))
bars = ax.bar(x, maes_plot, color=colors_plot, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(methods_plot, fontsize=9)
ax.set_ylabel('MAE [Ha]', fontsize=12)
ax.set_title('Extrapolation: Z=5-13 (train) → Z=17-27 (test)\nFeatures include Z_AB, best r₀ per method', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for i in range(len(methods_plot)):
    ratio = maes_plot[i] / mae_direct
    if ratio >= 1:
        label = f'{ratio:.1f}x'
    else:
        label = f'{1/ratio:.1f}x better'
    ax.annotate(label, xy=(x[i], maes_plot[i]),
               xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=10, fontweight='bold')

# Cap y-axis
max_show = max(mae_direct * 8, 0.5)
if max(maes_plot) > max_show:
    ax.set_ylim(0, max_show)
    for i, m in enumerate(maes_plot):
        if m > max_show:
            ax.text(x[i], max_show * 0.9, f'{m:.2f} Ha', ha='center', fontsize=9, color='red')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_curvature_v2_comparison.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_curvature_v2_comparison.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Example predictions (best MLP parabola vs others)
# ================================================================
best_r0_mlp = min(r0_values, key=lambda r0: all_results[r0]['MLP'])
dr2_best = (r - best_r0_mlp)**2
a_train_best = (train['E_elec'] / dr2_best[np.newaxis, :]).ravel()

# Retrain best MLP for predictions
mdl_best = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                        max_iter=2000, random_state=42, early_stopping=True,
                        validation_fraction=0.1)
mdl_best.fit(X_train_scaled, a_train_best)
a_pred_best = mdl_best.predict(X_test_scaled).reshape(n_test, n_grid)
E_pred_best = a_pred_best * dr2_best[np.newaxis, :]
V_pred_best = E_pred_best + test['V_nn']

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
example_indices = [0, n_test//2, n_test-1]

for col, idx in enumerate(example_indices):
    Z_i = test['Z'][idx]
    p = test['params'][idx]

    ax = axes3[0, col]
    ax.plot(r, test['V_total'][idx], 'b-', lw=2.5, label='True')
    ax.plot(r, V_pred_direct[idx], 'gray', ls='--', lw=1.5, label='Direct Ridge')
    ax.plot(r, V_pred_best[idx], 'g-', lw=1.5, label=f'MLP parabola (r0={best_r0_mlp:.0f})')
    ax.plot(r, V_pred_direct_mlp[idx], 'r:', lw=1.5, label='MLP direct E_elec')
    ax.plot(r, V_pred_global[idx], 'k-.', lw=2, label='Global Rose')
    ax.set_title(f'Z={Z_i:.0f}, $D_e$={p[0]:.2f}, $R_e$={p[1]:.1f}')
    ax.set_ylim(-1, 3)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$V_{total}$ [Ha]')
        ax.legend(fontsize=7)

    ax2 = axes3[1, col]
    for V_p, color, ls, label in [
        (V_pred_direct, 'gray', '--', 'Direct Ridge'),
        (V_pred_best, 'green', '-', f'MLP parabola'),
        (V_pred_direct_mlp, 'red', ':', 'MLP direct E_elec'),
        (V_pred_global, 'black', '-.', 'Global Rose'),
    ]:
        err = V_p[idx] - test['V_total'][idx]
        ax2.plot(r, err, color=color, ls=ls, lw=1.5, label=label)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('r [Bohr]')
    ax2.grid(True, alpha=0.3)
    if col == 0:
        ax2.set_ylabel('Error [Ha]')
        ax2.legend(fontsize=7)

fig3.suptitle(f'Extrapolation predictions (Z_AB in features, r0={best_r0_mlp:.0f})', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_curvature_v2_predictions.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_curvature_v2_predictions.png')
plt.close(fig3)
