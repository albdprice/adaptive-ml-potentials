"""
Test parabola decomposition on a 1/r-type E_elec potential.

No Morse, no nuclear subtraction, no Z^2 issue.

E_elec(r) = -C / sqrt(r^2 + a^2)     (softened Coulomb)

  - Goes to -C/r at large r (Coulomb-like)
  - Finite at r=0: E_elec(0) = -C/a
  - Parameters C and a are LINEAR in Z (no Z^2 scaling)

V_nn = Z^2 / r  (added back later, known exactly, not part of the ML)
V_total = E_elec + V_nn

Parameter mapping:
  C = 2.0 + 0.5 * Z    (Coulomb strength, linear in Z)
  a = 0.5 + 0.05 * Z   (softening radius, linear in Z)

Training: Z = 5, 6, ..., 13
Test:     Z = 17, 18, ..., 27 (extrapolation)

Methods:
  1. Direct multioutput Ridge:  Z -> E_elec(r), 50 independent outputs
  2. Global adaptive Ridge:     Z -> (C, a), reconstruct E_elec from equation
  3. Shared Ridge [Z, r] -> E_elec directly (no parabola)
  4. Shared Ridge [Z, r] -> a_curv, parabola: E_elec = a_curv * (r - r0)^2
  5. Shared MLP  [Z, r] -> E_elec directly
  6. Shared MLP  [Z, r] -> a_curv, parabola
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
def make_molecules(Z_values):
    r = np.linspace(0.5, 10.0, 50)
    Z = np.array(Z_values, dtype=float)

    # Parameters: LINEAR in Z, no Z^2 anywhere
    C = 2.0 + 0.5 * Z       # Coulomb strength
    a = 0.5 + 0.05 * Z      # softening radius

    E_elec = np.zeros((len(Z), len(r)))
    for i in range(len(Z)):
        E_elec[i] = -C[i] / np.sqrt(r**2 + a[i]**2)

    # V_nn added later (known, not learned)
    Z_AB = Z**2
    V_nn = np.zeros((len(Z), len(r)))
    for i in range(len(Z)):
        V_nn[i] = Z_AB[i] / r

    V_total = E_elec + V_nn

    return {'Z': Z, 'Z_AB': Z_AB, 'C': C, 'a': a,
            'r': r, 'E_elec': E_elec, 'V_nn': V_nn, 'V_total': V_total}


Z_train = list(range(5, 14))
Z_test = list(range(17, 28))
train = make_molecules(Z_train)
test = make_molecules(Z_test)
r = train['r']
n_train, n_test, n_grid = len(Z_train), len(Z_test), len(r)

r0 = 20.0
dr2 = (r - r0)**2

print(f'Training: {n_train} molecules, Z = {Z_train[0]}-{Z_train[-1]}')
print(f'Test:     {n_test} molecules, Z = {Z_test[0]}-{Z_test[-1]}')
print(f'C(Z): train [{train["C"].min():.1f}, {train["C"].max():.1f}], '
      f'test [{test["C"].min():.1f}, {test["C"].max():.1f}]')
print(f'a(Z): train [{train["a"].min():.2f}, {train["a"].max():.2f}], '
      f'test [{test["a"].min():.2f}, {test["a"].max():.2f}]')
print(f'E_elec range: train [{train["E_elec"].min():.2f}, {train["E_elec"].max():.2f}], '
      f'test [{test["E_elec"].min():.2f}, {test["E_elec"].max():.2f}]')

# ================================================================
# FIGURE 1: What E_elec and a(r) look like
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for i in range(n_train):
    Z_i = train['Z'][i]
    axes[0, 0].plot(r, train['E_elec'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes[0, 1].plot(r, train['E_elec'][i] / dr2, lw=1.5, alpha=0.7)
    axes[0, 2].plot(r, train['V_total'][i], lw=1.5, alpha=0.7)

for i in range(n_test):
    Z_i = test['Z'][i]
    axes[1, 0].plot(r, test['E_elec'][i], lw=1.5, alpha=0.7, label=f'Z={Z_i:.0f}')
    axes[1, 1].plot(r, test['E_elec'][i] / dr2, lw=1.5, alpha=0.7)
    axes[1, 2].plot(r, test['V_total'][i], lw=1.5, alpha=0.7)

axes[0, 0].set_title('$E_{elec}(r) = -C / \\sqrt{r^2 + a^2}$', fontsize=12)
axes[0, 1].set_title(f'$a_{{curv}}(r) = E_{{elec}} / (r - {r0:.0f})^2$', fontsize=12)
axes[0, 2].set_title('$V_{total} = E_{elec} + Z^2/r$', fontsize=12)

axes[0, 0].set_ylabel(f'Training (Z={Z_train[0]}-{Z_train[-1]})\n[Ha]')
axes[1, 0].set_ylabel(f'Test (Z={Z_test[0]}-{Z_test[-1]})\n[Ha]')
axes[0, 0].legend(fontsize=6, ncol=3)
axes[1, 0].legend(fontsize=6, ncol=3)

# Annotate ranges
E_tr = train['E_elec']
E_te = test['E_elec']
a_tr = E_tr / dr2[np.newaxis, :]
a_te = E_te / dr2[np.newaxis, :]

axes[0, 0].text(0.95, 0.05, f'range: [{E_tr.min():.1f}, {E_tr.max():.2f}]',
                transform=axes[0, 0].transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))
axes[1, 0].text(0.95, 0.05, f'range: [{E_te.min():.1f}, {E_te.max():.2f}]',
                transform=axes[1, 0].transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))
axes[0, 1].text(0.95, 0.05, f'range: [{a_tr.min():.4f}, {a_tr.max():.5f}]',
                transform=axes[0, 1].transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))
axes[1, 1].text(0.95, 0.05, f'range: [{a_te.min():.4f}, {a_te.max():.5f}]',
                transform=axes[1, 1].transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(fc='lightyellow', ec='orange', alpha=0.9))

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('r [Bohr]')

fig.suptitle('Softened Coulomb: $E_{elec} = -C/\\sqrt{r^2 + a^2}$, C and a linear in Z\nNo $Z^2$ scaling in E_elec', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_1r_potential_targets.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_1r_potential_targets.png')
plt.close(fig)

# ================================================================
# ML
# ================================================================
def stack_Zr(data, r):
    Z = data['Z']
    n_mol, n_grid = len(Z), len(r)
    X = np.zeros((n_mol * n_grid, 2))
    for i in range(n_mol):
        for j in range(n_grid):
            X[i * n_grid + j] = [Z[i], r[j]]
    return X

X_stk_tr = stack_Zr(train, r)
X_stk_te = stack_Zr(test, r)
scaler = StandardScaler()
X_stk_tr_sc = scaler.fit_transform(X_stk_tr)
X_stk_te_sc = scaler.transform(X_stk_te)

X_mol_tr = train['Z'].reshape(-1, 1)
X_mol_te = test['Z'].reshape(-1, 1)

E_train_flat = train['E_elec'].ravel()
a_train_flat = (train['E_elec'] / dr2[np.newaxis, :]).ravel()

results = {}

# 1. Direct multioutput Ridge: Z -> E_elec(r)
mdl = Ridge(alpha=1.0)
mdl.fit(X_mol_tr, train['E_elec'])
E_pred = mdl.predict(X_mol_te)
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
results['Direct Ridge\nZ→E_elec(r)\nmultioutput'] = {'mae': mae, 'E': E_pred}
print(f'\n1. Direct Ridge multioutput:    {mae:.4f} Ha')
mae_baseline = mae

# 2. Global adaptive: Z -> (C, a), reconstruct
mdl_C = Ridge(alpha=1.0)
mdl_C.fit(X_mol_tr, train['C'])
C_pred = mdl_C.predict(X_mol_te)

mdl_a = Ridge(alpha=1.0)
mdl_a.fit(X_mol_tr, train['a'])
a_pred_global = mdl_a.predict(X_mol_te)

E_pred_global = np.zeros((n_test, n_grid))
for i in range(n_test):
    E_pred_global[i] = -C_pred[i] / np.sqrt(r**2 + a_pred_global[i]**2)
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred_global.ravel())
results['Global adaptive\nZ→(C,a)\nreconstruct'] = {'mae': mae, 'E': E_pred_global}
print(f'2. Global adaptive (C, a):      {mae:.4f} Ha')

# 3. Ridge shared [Z, r] -> E_elec
mdl = Ridge(alpha=1.0)
mdl.fit(X_stk_tr, E_train_flat)
E_pred = mdl.predict(X_stk_te).reshape(n_test, n_grid)
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
results['Ridge shared\n[Z,r]→E_elec\n(no parabola)'] = {'mae': mae, 'E': E_pred}
print(f'3. Ridge [Z,r]→E_elec direct:   {mae:.4f} Ha')

# 4. Ridge shared [Z, r] -> a_curv, parabola
mdl = Ridge(alpha=1.0)
mdl.fit(X_stk_tr, a_train_flat)
a_pred = mdl.predict(X_stk_te).reshape(n_test, n_grid)
E_pred = a_pred * dr2[np.newaxis, :]
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
results['Ridge shared\n[Z,r]→a(r)\nparabola'] = {'mae': mae, 'E': E_pred}
print(f'4. Ridge [Z,r]→a(r) parabola:   {mae:.4f} Ha')

# 5. MLP shared [Z, r] -> E_elec
mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                   max_iter=3000, random_state=42, early_stopping=True,
                   validation_fraction=0.15)
mdl.fit(X_stk_tr_sc, E_train_flat)
E_pred = mdl.predict(X_stk_te_sc).reshape(n_test, n_grid)
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
results['MLP shared\n[Z,r]→E_elec\n(no parabola)'] = {'mae': mae, 'E': E_pred}
print(f'5. MLP [Z,r]→E_elec direct:     {mae:.4f} Ha')

# 6. MLP shared [Z, r] -> a_curv, parabola
mdl = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                   max_iter=3000, random_state=42, early_stopping=True,
                   validation_fraction=0.15)
mdl.fit(X_stk_tr_sc, a_train_flat)
a_pred = mdl.predict(X_stk_te_sc).reshape(n_test, n_grid)
E_pred = a_pred * dr2[np.newaxis, :]
mae = mean_absolute_error(test['E_elec'].ravel(), E_pred.ravel())
results['MLP shared\n[Z,r]→a(r)\nparabola'] = {'mae': mae, 'E': E_pred}
print(f'6. MLP [Z,r]→a(r) parabola:     {mae:.4f} Ha')

# ================================================================
# Summary
# ================================================================
print(f'\n{"="*70}')
print(f'{"Method":<35s} | {"E_elec MAE":>12s} | {"vs Direct":>12s}')
print(f'{"="*70}')

for name in [
    'Direct Ridge\nZ→E_elec(r)\nmultioutput',
    'Global adaptive\nZ→(C,a)\nreconstruct',
    'Ridge shared\n[Z,r]→E_elec\n(no parabola)',
    'Ridge shared\n[Z,r]→a(r)\nparabola',
    'MLP shared\n[Z,r]→E_elec\n(no parabola)',
    'MLP shared\n[Z,r]→a(r)\nparabola',
]:
    mae = results[name]['mae']
    flat = name.replace('\n', ' ')
    if 'Direct' in name:
        print(f'{flat:<35s} | {mae:12.4f} | {"baseline":>12s}')
    else:
        ratio = mae / mae_baseline
        marker = ' BETTER' if ratio < 1 else ''
        print(f'{flat:<35s} | {mae:12.4f} | {ratio:>10.2f}x{marker}')

print(f'{"="*70}')

# ================================================================
# FIGURE 2: Bar chart
# ================================================================
fig2, ax = plt.subplots(figsize=(12, 6))

plot_items = [
    ('Direct Ridge\nmultioutput', results['Direct Ridge\nZ→E_elec(r)\nmultioutput']['mae'], '#888888'),
    ('Global\n(C, a)', results['Global adaptive\nZ→(C,a)\nreconstruct']['mae'], '#1f77b4'),
    ('Ridge\nE_elec direct', results['Ridge shared\n[Z,r]→E_elec\n(no parabola)']['mae'], '#d62728'),
    ('Ridge\na(r) parabola', results['Ridge shared\n[Z,r]→a(r)\nparabola']['mae'], '#2ca02c'),
    ('MLP\nE_elec direct', results['MLP shared\n[Z,r]→E_elec\n(no parabola)']['mae'], '#ff9896'),
    ('MLP\na(r) parabola', results['MLP shared\n[Z,r]→a(r)\nparabola']['mae'], '#8fce8f'),
]

names = [p[0] for p in plot_items]
maes = [p[1] for p in plot_items]
cols = [p[2] for p in plot_items]
x = np.arange(len(names))

ax.bar(x, maes, color=cols, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('E_elec MAE [Ha]', fontsize=12)
ax.set_title('Softened Coulomb $E_{elec} = -C/\\sqrt{r^2+a^2}$: no $Z^2$ scaling\nCan the parabola learn a 1/r-type curve?', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for i in range(len(names)):
    ratio = maes[i] / mae_baseline
    if ratio < 1:
        label = f'{1/ratio:.1f}x better'
    else:
        label = f'{ratio:.1f}x'
    ax.annotate(label, xy=(x[i], maes[i]),
               xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=10, fontweight='bold')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_1r_potential_comparison.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_1r_potential_comparison.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Example predictions
# ================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
ex_idx = [0, n_test//2, n_test-1]

for col, idx in enumerate(ex_idx):
    Z_i = test['Z'][idx]

    ax = axes3[0, col]
    ax.plot(r, test['E_elec'][idx], 'b-', lw=2.5, label='True')
    for name, color, ls in [
        ('Direct Ridge\nZ→E_elec(r)\nmultioutput', 'gray', '--'),
        ('Global adaptive\nZ→(C,a)\nreconstruct', 'black', '-.'),
        ('MLP shared\n[Z,r]→a(r)\nparabola', 'green', '-'),
        ('MLP shared\n[Z,r]→E_elec\n(no parabola)', 'red', ':'),
    ]:
        flat = name.replace('\n', ' ')
        ax.plot(r, results[name]['E'][idx], color=color, ls=ls, lw=1.5, label=flat)
    ax.set_title(f'Z = {Z_i:.0f}, C = {test["C"][idx]:.1f}, a = {test["a"][idx]:.2f}')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$E_{elec}$ [Ha]')
        ax.legend(fontsize=7)

    ax2 = axes3[1, col]
    for name, color, ls in [
        ('Direct Ridge\nZ→E_elec(r)\nmultioutput', 'gray', '--'),
        ('Global adaptive\nZ→(C,a)\nreconstruct', 'black', '-.'),
        ('MLP shared\n[Z,r]→a(r)\nparabola', 'green', '-'),
        ('MLP shared\n[Z,r]→E_elec\n(no parabola)', 'red', ':'),
    ]:
        ax2.plot(r, results[name]['E'][idx] - test['E_elec'][idx], color=color, ls=ls, lw=1.5)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('r [Bohr]')
    ax2.grid(True, alpha=0.3)
    if col == 0:
        ax2.set_ylabel('Error [Ha]')

fig3.suptitle('Extrapolation predictions on softened Coulomb', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_1r_potential_predictions.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_1r_potential_predictions.png')
plt.close(fig3)
