"""
Clean figures and results for Anatole.

Dataset: 14e⁻ isoelectronic diatomic series (PBE0/cc-pVDZ, from Devika's thesis)
  Train: N₂(λ=0), CO(λ=1), BF(λ=2), BeNe(λ=3)
  Test:  LiNa(λ=4), HeMg(λ=5), HAl(λ=6)    ← extrapolation

λ = (Z₂ - Z₁)/2  is the alchemical coordinate.
d = internuclear distance [Å].
E_elec = DFT electronic energy [Ha].

Methods:
  1. Direct multioutput Ridge: λ → E_elec(d) at each grid point independently
     Each d_j has its own weight: E(d_j) = w_j · λ + b_j  (linear in λ)

  2. AHA multioutput Ridge: [λ, λ²] → E_elec(d)
     Each d_j: E(d_j) = w1_j · λ + w2_j · λ² + b_j  (quadratic in λ)
     This is Anatole's Alchemical Harmonic Approximation.

  3. Anatole's spatial parabola (shared): one model (λ, d) → a(d)
     Targets: a(d) = E_elec(d) / (d - d₀)²
     Ridge: a = w₁·λ + w₂·d + b  (3 weights shared across ALL molecules and ALL distances)
     Reconstruction: E_elec = a · (d - d₀)²

  4. Direct shared: one model (λ, d) → E_elec
     Ridge: E_elec = w₁·λ + w₂·d + b  (3 weights shared)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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

mol_names = {0:'N₂', 1:'CO', 2:'BF', 3:'BeNe', 4:'LiNa', 5:'HeMg', 6:'HAl'}

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
lam_train = np.array([0., 1., 2., 3.])
lam_test = np.array([4., 5., 6.])

# ================================================================
# Fit all methods
# ================================================================

# Method 1: Direct multioutput [λ] → E(d)
mdl1 = Ridge(alpha=1e-6)
mdl1.fit(lam_train.reshape(-1, 1), E_elec[train_idx])
E_pred_1 = mdl1.predict(lam_test.reshape(-1, 1))

# Method 2: AHA multioutput [λ, λ²] → E(d)
mdl2 = Ridge(alpha=1e-6)
mdl2.fit(np.column_stack([lam_train, lam_train**2]), E_elec[train_idx])
E_pred_2 = mdl2.predict(np.column_stack([lam_test, lam_test**2]))

# Shared model features
def stack(lam_vals, d_grid):
    X = []
    for l in lam_vals:
        for d in d_grid:
            X.append([l, d])
    return np.array(X)

X_stk_tr = stack(lam_train, d_grid)
X_stk_te = stack(lam_test, d_grid)

d0 = 5.0
dr2 = (d_grid - d0)**2

# Method 3: Shared parabola (λ, d) → a(d)
a_train = (E_elec[train_idx] / dr2[np.newaxis, :]).ravel()
mdl3 = Ridge(alpha=1e-6)
mdl3.fit(X_stk_tr, a_train)
a_pred = mdl3.predict(X_stk_te).reshape(3, n_grid)
E_pred_3 = a_pred * dr2[np.newaxis, :]

# Method 4: Shared direct (λ, d) → E_elec
mdl4 = Ridge(alpha=1e-6)
mdl4.fit(X_stk_tr, E_elec[train_idx].ravel())
E_pred_4 = mdl4.predict(X_stk_te).reshape(3, n_grid)

# ================================================================
# MAE summary
# ================================================================
methods = {
    'Direct multioutput [λ]→E(d)':          E_pred_1,
    'AHA multioutput [λ,λ²]→E(d)':         E_pred_2,
    'Shared parabola [λ,d]→a(d)':           E_pred_3,
    'Shared direct [λ,d]→E(d)':             E_pred_4,
}

mae_baseline = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_1.ravel())

print(f'{"="*80}')
print(f'E_elec EXTRAPOLATION ERRORS (MAE in Ha)')
print(f'{"="*80}')
print(f'{"Method":<40s} | {"MAE":>7s} | {"ratio":>7s} | {"LiNa":>7s} {"HeMg":>7s} {"HAl":>7s}')
print(f'{"-"*80}')

for label, pred in methods.items():
    mae = mean_absolute_error(E_elec[test_idx].ravel(), pred.ravel())
    per = [mean_absolute_error(E_elec[test_idx[j]], pred[j]) for j in range(3)]
    rat = f'{mae/mae_baseline:.3f}x' if label != list(methods.keys())[0] else 'base'
    print(f'{label:<40s} | {mae:7.2f} | {rat:>7s} | {per[0]:7.2f} {per[1]:7.2f} {per[2]:7.2f}')
print(f'{"="*80}')

# ================================================================
# FIGURE 1: E_elec(d) — the raw data
# ================================================================
fig1, ax = plt.subplots(figsize=(9, 6))

for i in train_idx:
    ax.plot(d_grid, E_elec[i], lw=2, color='#1f77b4', alpha=0.7,
            label=f'{mol_names[i]} (λ={i})')
for i in test_idx:
    ax.plot(d_grid, E_elec[i], lw=2, color='#d62728', alpha=0.7, ls='--',
            label=f'{mol_names[i]} (λ={i})')

ax.set_xlabel('d [Å]', fontsize=13)
ax.set_ylabel('$E_{elec}$ [Ha]', fontsize=13)
ax.set_title('14e⁻ isoelectronic series — DFT electronic energy\n'
             'Blue = training molecules,  Red = test molecules (extrapolation)', fontsize=12)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'fig1_data.png'), dpi=150, bbox_inches='tight')
print('Saved fig1_data.png')
plt.close(fig1)

# ================================================================
# FIGURE 2: E_elec vs λ at fixed d — shows parabolic character
# ================================================================
fig2, axes = plt.subplots(2, 3, figsize=(15, 9))
d_samples = [0.7, 1.0, 1.3, 1.6, 2.0, 2.4]

for ax, d_target in zip(axes.flat, d_samples):
    j_d = np.argmin(np.abs(d_grid - d_target))
    d_actual = d_grid[j_d]
    E_at_d = E_elec[:, j_d]

    ax.plot([0,1,2,3], E_at_d[train_idx], 'bo', ms=10, zorder=5, label='Train')
    ax.plot([4,5,6], E_at_d[test_idx], 'rs', ms=10, zorder=5, label='Test (true)')

    lam_dense = np.linspace(-0.5, 7, 200)

    # Linear fit
    m1 = Ridge(alpha=1e-6)
    m1.fit(lam_train.reshape(-1, 1), E_at_d[train_idx])
    ax.plot(lam_dense, m1.predict(lam_dense.reshape(-1, 1)),
            'b-', lw=1.5, alpha=0.6, label='Linear in λ')

    # Quadratic fit (AHA)
    m2 = Ridge(alpha=1e-6)
    m2.fit(np.column_stack([lam_train, lam_train**2]), E_at_d[train_idx])
    ax.plot(lam_dense, m2.predict(np.column_stack([lam_dense, lam_dense**2])),
            'g-', lw=2, alpha=0.8, label='Quadratic in λ (AHA)')

    ax.set_xlabel('λ', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [Ha]', fontsize=11)
    ax.set_title(f'd = {d_actual:.2f} Å', fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig2.suptitle('$E_{elec}$ vs λ at fixed distance — parabolic in λ (chemical space)\n'
              'Blue circles = train,  Red squares = test,  Green = quadratic fit from train only',
              fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig2_Eelec_vs_lambda.png'), dpi=150, bbox_inches='tight')
print('Saved fig2_Eelec_vs_lambda.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Curvature targets a(d) = E_elec / (d - d₀)²
# ================================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# Left: a(d) targets
for i in train_idx:
    ax3a.plot(d_grid, E_elec[i] / dr2, lw=2, color='#1f77b4', alpha=0.7,
              label=mol_names[i])
for i in test_idx:
    ax3a.plot(d_grid, E_elec[i] / dr2, lw=2, color='#d62728', alpha=0.7, ls='--',
              label=mol_names[i])

ax3a.set_xlabel('d [Å]', fontsize=12)
ax3a.set_ylabel(f'$a(d) = E_{{elec}} / (d - {d0:.0f})^2$  [Ha/Å²]', fontsize=12)
ax3a.set_title(f'Spatial parabola curvature targets\n'
               f'a(d) = E_{{elec}}(d) / (d - {d0:.0f})²', fontsize=12)
ax3a.legend(fontsize=9)
ax3a.grid(True, alpha=0.3)

# Right: E_elec(d) — NOT parabolic in d
ax3b.plot(d_grid, d_grid * 0, 'k--', alpha=0.3)
# Show E_elec for N₂ and annotate why it's not parabolic
ax3b.plot(d_grid, E_elec[0] - E_elec[0, -1], lw=2, color='#1f77b4', label='N₂ (shifted)')
ax3b.plot(d_grid, E_elec[6] - E_elec[6, -1], lw=2, color='#d62728', ls='--', label='HAl (shifted)')
# Overlay parabola shape for reference
d_eq = 1.1
scale = (E_elec[0, 0] - E_elec[0, -1]) / (d_grid[0] - d0)**2
ax3b.plot(d_grid, scale * (d_grid - d0)**2, 'k:', lw=1.5, alpha=0.4,
          label=f'Parabola (d-{d0:.0f})²')

ax3b.set_xlabel('d [Å]', fontsize=12)
ax3b.set_ylabel('$E_{elec}(d) - E_{elec}(d_{max})$  [Ha]', fontsize=12)
ax3b.set_title('E_elec(d) shape vs parabola\n(d-d₀)² cannot match the d-dependence', fontsize=12)
ax3b.legend(fontsize=9)
ax3b.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig3_curvature_targets.png'), dpi=150, bbox_inches='tight')
print('Saved fig3_curvature_targets.png')
plt.close(fig3)

# ================================================================
# FIGURE 4: Predictions for test molecules (NO Morse)
# ================================================================
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))

plot_methods = [
    ('Direct [λ]→E(d)', E_pred_1, 'b--', 1.5),
    ('AHA [λ,λ²]→E(d)', E_pred_2, 'g-', 2.0),
    ('Parabola [λ,d]→a(d)', E_pred_3, 'r:', 1.5),
]

for j, ax in enumerate(axes4):
    idx = test_idx[j]
    name = mol_names[idx]

    ax.plot(d_grid, E_elec[idx], 'k-', lw=2.5, label='DFT reference', zorder=5)

    for mlabel, pred, style, lw in plot_methods:
        mae_j = mean_absolute_error(E_elec[idx], pred[j])
        ax.plot(d_grid, pred[j], style, lw=lw,
                label=f'{mlabel} ({mae_j:.2f} Ha)')

    ax.set_xlabel('d [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [Ha]', fontsize=11)
    ax.set_title(f'{name} (λ={idx})', fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

fig4.suptitle('Extrapolation predictions — train on N₂,CO,BF,BeNe → predict test molecules\n'
              'Green (AHA) captures parabolic λ-dependence; Blue/Red miss it',
              fontsize=12)
fig4.tight_layout()
fig4.savefig(os.path.join(FIGDIR, 'fig4_predictions.png'), dpi=150, bbox_inches='tight')
print('Saved fig4_predictions.png')
plt.close(fig4)

# ================================================================
# FIGURE 5: Summary bar chart
# ================================================================
fig5, ax5 = plt.subplots(figsize=(10, 5))

labels = list(methods.keys())
maes = [mean_absolute_error(E_elec[test_idx].ravel(), methods[l].ravel()) for l in labels]
colors = ['steelblue', 'forestgreen', 'coral', '#aec7e8']

bars = ax5.barh(range(len(labels)), maes, color=colors)
ax5.set_yticks(range(len(labels)))
ax5.set_yticklabels(labels, fontsize=11)
ax5.set_xlabel('MAE [Ha]', fontsize=12)
ax5.set_title('E_elec extrapolation error on test molecules', fontsize=13)

for i, (m, bar) in enumerate(zip(maes, bars)):
    ax5.text(m + 0.5, i, f'{m:.2f} Ha', va='center', fontsize=11, fontweight='bold')

ax5.grid(True, alpha=0.3, axis='x')
fig5.tight_layout()
fig5.savefig(os.path.join(FIGDIR, 'fig5_summary.png'), dpi=150, bbox_inches='tight')
print('Saved fig5_summary.png')
plt.close(fig5)

print('\nAll figures saved to anatole_expanded/figures/')
