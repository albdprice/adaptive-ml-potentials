"""
Periodic-table-inspired Morse ML experiment.

Physics:
  V_total(r) = V_Morse(r) = D_e*(1-exp(-alpha*(r-R_e)))^2 - D_e
  V_nn(r)    = Z_A * Z_B / r   (known, homonuclear: Z_AB = Z^2)
  E_elec(r)  = V_total(r) - V_nn(r)

Descriptor -> parameter mapping (periodic trends):
  d1 -> atomic number Z (heavier elements)
  d2 -> bonding variation (bond order, spin state)

  Z     = 3 + 2*d1           (atomic number)
  Z_AB  = Z^2                (homonuclear nuclear repulsion)
  R_e   = 2.0 + 0.2*d2       (equilibrium distance, modest variation)
  D_e   = 0.1 + 0.05*d1 + 0.03*d2  (well depth)
  alpha = 0.8 + 0.1*d2       (width, kept gentle)

Training: d1 in [1, 5], d2 in [0.5, 1.5]  -> Z = 5-13 (B to Al)
Testing:  d1 in [7, 12], d2 in [0.5, 1.5] -> Z = 17-27 (Cl to Co)

ML methods:
  1. Direct:           d -> V_total(r)
  2. Elec direct:      d -> E_elec(r), add back known V_nn
  3. Global adaptive:  d -> (D_e, R_e, alpha), reconstruct V_Morse
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Dataset generation
# ================================================================
def params_from_descriptors(d1, d2):
    """Map descriptors to physical parameters following periodic trends."""
    Z = 3 + 2 * d1
    Z_AB = Z**2
    R_e = 2.0 + 0.2 * d2
    D_e = 0.1 + 0.05 * d1 + 0.03 * d2
    alpha = 0.8 + 0.1 * d2
    return Z, Z_AB, R_e, D_e, alpha


def make_dataset(d1_range, d2_range, n_per_dim=20):
    """Generate periodic-table-inspired Morse curves."""
    d1_vals = np.linspace(d1_range[0], d1_range[1], n_per_dim)
    d2_vals = np.linspace(d2_range[0], d2_range[1], n_per_dim)
    d1_grid, d2_grid = np.meshgrid(d1_vals, d2_vals)
    d1_flat = d1_grid.ravel()
    d2_flat = d2_grid.ravel()
    n_samples = len(d1_flat)

    Z, Z_AB, R_e, D_e, alpha = params_from_descriptors(d1_flat, d2_flat)

    # Grid (Bohr)
    r = np.linspace(1.0, 8.0, 50)
    n_grid = len(r)

    V_total = np.zeros((n_samples, n_grid))
    V_nn_all = np.zeros((n_samples, n_grid))
    E_elec = np.zeros((n_samples, n_grid))

    n_not_monotonic = 0
    for i in range(n_samples):
        u = np.exp(-alpha[i] * (r - R_e[i]))
        V_total[i] = D_e[i] * (1 - u)**2 - D_e[i]
        V_nn_all[i] = Z_AB[i] / r
        E_elec[i] = V_total[i] - V_nn_all[i]

        # Check monotonicity: dE_elec/dr should be > 0 everywhere
        dE = np.diff(E_elec[i])
        if np.any(dE < 0):
            n_not_monotonic += 1

    if n_not_monotonic > 0:
        print(f'  WARNING: {n_not_monotonic}/{n_samples} curves not monotonically increasing')
    else:
        print(f'  All {n_samples} curves monotonically increasing (E_elec)')

    descriptors = np.column_stack([d1_flat, d2_flat])
    params = np.column_stack([D_e, R_e, alpha])

    return {
        'descriptors': descriptors,
        'params': params,
        'Z': Z,
        'Z_AB': Z_AB,
        'r': r,
        'V_total': V_total,
        'V_nn': V_nn_all,
        'E_elec': E_elec,
    }


# ================================================================
# ML experiment
# ================================================================
def run_experiment(train, test):
    r = train['r']
    X_train = train['descriptors']
    X_test = test['descriptors']
    n_test = len(X_test)

    results = {}

    # --- Method 1: Direct (d -> V_total) ---
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train, train['V_total'])
    V_pred = mdl.predict(X_test)
    results['Direct'] = {
        'V_pred': V_pred,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred.ravel()),
    }

    # --- Method 2: Elec direct (d -> E_elec, add V_nn back) ---
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train, train['E_elec'])
    E_pred = mdl.predict(X_test)
    V_pred_elec = E_pred + test['V_nn']
    results['Elec direct'] = {
        'V_pred': V_pred_elec,
        'E_elec_pred': E_pred,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_elec.ravel()),
    }

    # --- Method 3: Global adaptive (d -> D_e, R_e, alpha) ---
    mdl_De = Ridge(alpha=1.0)
    mdl_De.fit(X_train, train['params'][:, 0])
    De_pred = mdl_De.predict(X_test)

    mdl_Re = Ridge(alpha=1.0)
    mdl_Re.fit(X_train, train['params'][:, 1])
    Re_pred = mdl_Re.predict(X_test)

    mdl_al = Ridge(alpha=1.0)
    mdl_al.fit(X_train, train['params'][:, 2])
    al_pred = mdl_al.predict(X_test)

    V_pred_global = np.zeros((n_test, len(r)))
    for i in range(n_test):
        u = np.exp(-al_pred[i] * (r - Re_pred[i]))
        V_pred_global[i] = De_pred[i] * (1 - u)**2 - De_pred[i]

    results['Global (Rose)'] = {
        'V_pred': V_pred_global,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel()),
    }

    return results


# ================================================================
# Run
# ================================================================
print('=== Generating datasets ===')
print('Training (Z = 5-13, B to Al):')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
print(f'  {len(train["descriptors"])} samples, Z range: [{train["Z"].min():.0f}, {train["Z"].max():.0f}]')
print(f'  Z_AB range: [{train["Z_AB"].min():.0f}, {train["Z_AB"].max():.0f}]')
print(f'  D_e: [{train["params"][:,0].min():.3f}, {train["params"][:,0].max():.3f}] Ha')
print(f'  R_e: [{train["params"][:,1].min():.2f}, {train["params"][:,1].max():.2f}] Bohr')

print('\nTest extrapolation (Z = 17-27, Cl to Co):')
test_ext = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
print(f'  {len(test_ext["descriptors"])} samples, Z range: [{test_ext["Z"].min():.0f}, {test_ext["Z"].max():.0f}]')
print(f'  Z_AB range: [{test_ext["Z_AB"].min():.0f}, {test_ext["Z_AB"].max():.0f}]')

print('\nTest interpolation:')
test_int = make_dataset([1.5, 4.5], [0.6, 1.4], n_per_dim=10)
print(f'  {len(test_int["descriptors"])} samples, Z range: [{test_int["Z"].min():.0f}, {test_int["Z"].max():.0f}]')

# Run ML
print('\n=== EXTRAPOLATION ===')
res_ext = run_experiment(train, test_ext)
direct_mae = res_ext['Direct']['mae']
for name, res in res_ext.items():
    ratio = direct_mae / res['mae'] if res['mae'] > 0 else float('inf')
    marker = ' <-- baseline' if name == 'Direct' else f' ({ratio:.2f}x vs direct)'
    print(f'  {name:20s}: MAE = {res["mae"]:.4f} Ha{marker}')

print('\n=== INTERPOLATION ===')
res_int = run_experiment(train, test_int)
direct_mae_int = res_int['Direct']['mae']
for name, res in res_int.items():
    ratio = direct_mae_int / res['mae'] if res['mae'] > 0 else float('inf')
    marker = ' <-- baseline' if name == 'Direct' else f' ({ratio:.2f}x vs direct)'
    print(f'  {name:20s}: MAE = {res["mae"]:.6f} Ha{marker}')

# ================================================================
# FIGURE 1: Dataset overview — train vs test curves
# ================================================================
r = train['r']
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

# Training curves (sample every 40th)
for i in range(0, len(train['V_total']), 40):
    Z_i = train['Z'][i]
    axes[0, 0].plot(r, train['V_total'][i], alpha=0.5, lw=1)
    axes[0, 1].plot(r, train['V_nn'][i], alpha=0.5, lw=1)
    axes[0, 2].plot(r, train['E_elec'][i], alpha=0.5, lw=1)

axes[0, 0].set_title('$V_{total} = V_{Morse}$ (train)')
axes[0, 1].set_title('$V_{nn} = Z^2/r$ (train, known)')
axes[0, 2].set_title('$E_{elec} = V_{total} - V_{nn}$ (train)')

# Test curves (sample every 10th)
for i in range(0, len(test_ext['V_total']), 10):
    axes[1, 0].plot(r, test_ext['V_total'][i], alpha=0.5, lw=1)
    axes[1, 1].plot(r, test_ext['V_nn'][i], alpha=0.5, lw=1)
    axes[1, 2].plot(r, test_ext['E_elec'][i], alpha=0.5, lw=1)

axes[1, 0].set_title('$V_{total}$ (extrap test)')
axes[1, 1].set_title('$V_{nn}$ (extrap test)')
axes[1, 2].set_title('$E_{elec}$ (extrap test)')

for ax in axes[:, 0]:
    ax.set_ylim(-0.8, 3)
    ax.set_ylabel('Energy [Ha]')
for ax in axes[:, 1]:
    ax.set_ylim(0, None)
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)
for ax in axes[1, :]:
    ax.set_xlabel('r [Bohr]')

fig.suptitle('Periodic-table Morse: training (Z=5-13) vs extrapolation (Z=17-27)', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_periodic_morse_curves.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_periodic_morse_curves.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Example extrapolation predictions
# ================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
examples = [0, len(test_ext['descriptors'])//2, -1]

for col, idx in enumerate(examples):
    d = test_ext['descriptors'][idx]
    Z_i = test_ext['Z'][idx]
    p = test_ext['params'][idx]

    ax = axes2[0, col]
    ax.plot(r, test_ext['V_total'][idx], 'b-', lw=2.5, label='True')
    ax.plot(r, res_ext['Direct']['V_pred'][idx], 'r:', lw=2, label='Direct')
    ax.plot(r, res_ext['Elec direct']['V_pred'][idx], 'm--', lw=1.5, label='Elec direct')
    ax.plot(r, res_ext['Global (Rose)']['V_pred'][idx], 'k-.', lw=2, label='Global')
    ax.set_title(f'Z={Z_i:.0f}, $D_e$={p[0]:.2f}, $R_e$={p[1]:.1f}, $\\alpha$={p[2]:.2f}')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$V_{total}$ [Ha]')
        ax.legend(fontsize=8)

    ax = axes2[1, col]
    for name, color, ls in [('Direct', 'r', ':'), ('Elec direct', 'm', '--'),
                             ('Global (Rose)', 'k', '-.')]:
        err = res_ext[name]['V_pred'][idx] - test_ext['V_total'][idx]
        ax.plot(r, err, color=color, ls=ls, lw=1.5, label=name)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('r [Bohr]')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('Error [Ha]')
        ax.legend(fontsize=8)

fig2.suptitle('Extrapolation: 3 methods compared', fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_periodic_morse_predictions.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_periodic_morse_predictions.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Summary bar chart
# ================================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
methods = list(res_ext.keys())
colors = ['#d62728', '#9467bd', '#1f77b4']

mae_ext = [res_ext[m]['mae'] for m in methods]
mae_int = [res_int[m]['mae'] for m in methods]
x = np.arange(len(methods))

ax1.bar(x, mae_ext, color=colors)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
ax1.set_ylabel('MAE [Ha]')
ax1.set_title('Extrapolation')
ax1.grid(True, alpha=0.3, axis='y')
for i, m in enumerate(methods):
    if m != 'Direct':
        ratio = direct_mae / res_ext[m]['mae']
        ax1.annotate(f'{ratio:.1f}x', xy=(x[i], mae_ext[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

ax2.bar(x, mae_int, color=colors, alpha=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
ax2.set_ylabel('MAE [Ha]')
ax2.set_title('Interpolation')
ax2.grid(True, alpha=0.3, axis='y')
for i, m in enumerate(methods):
    if m != 'Direct':
        ratio = direct_mae_int / res_int[m]['mae']
        ax2.annotate(f'{ratio:.1f}x', xy=(x[i], mae_int[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_periodic_morse_summary.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_periodic_morse_summary.png')
plt.close(fig3)
