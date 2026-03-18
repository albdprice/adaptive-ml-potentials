"""
Periodic Morse ML: learn parabola curvature a(r) of E_elec.

E_elec(r) goes to ~0 at r->inf. Decompose as:
  E_elec(r) = a(r) * (r - r_0)^2

with r_0 placed far out (where E_elec ~ 0). a(r) is the position-dependent
curvature. ML learns a(r) instead of E_elec(r).

Reconstruction: E_elec_pred = a_pred(r) * (r - r_0)^2, V_total = E_elec + V_nn

Methods compared:
  1. Direct:           d -> V_total(r)
  2. Elec direct:      d -> E_elec(r), add V_nn back
  3. Curvature a(r):   d -> a(r), reconstruct E_elec = a*(r-r0)^2, add V_nn
  4. Global:           d -> (D_e, R_e, alpha), reconstruct V_Morse

Test several r_0 values to find optimal.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

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
        'descriptors': np.column_stack([d1_flat, d2_flat]),
        'params': np.column_stack([D_e, R_e, alpha]),
        'Z': Z, 'Z_AB': Z_AB,
        'r': r, 'V_total': V_total, 'V_nn': V_nn, 'E_elec': E_elec,
    }


# ================================================================
# ML experiment
# ================================================================
def run_experiment(train, test, r0_values):
    r = train['r']
    X_train = train['descriptors']
    X_test = test['descriptors']
    n_test = len(X_test)

    results = {}

    # Method 1: Direct
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train, train['V_total'])
    V_pred = mdl.predict(X_test)
    results['Direct'] = {
        'V_pred': V_pred,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred.ravel()),
    }

    # Method 2: Elec direct
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_train, train['E_elec'])
    E_pred = mdl.predict(X_test)
    V_pred_elec = E_pred + test['V_nn']
    results['Elec direct'] = {
        'V_pred': V_pred_elec,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_elec.ravel()),
    }

    # Method 3: Curvature a(r) for each r_0
    for r0 in r0_values:
        dr2 = (r - r0)**2  # shape (n_grid,)

        # Compute a(r) = E_elec / (r - r0)^2
        a_train = train['E_elec'] / dr2[np.newaxis, :]
        a_test_true = test['E_elec'] / dr2[np.newaxis, :]

        mdl = Ridge(alpha=1.0)
        mdl.fit(X_train, a_train)
        a_pred = mdl.predict(X_test)

        # Reconstruct
        E_pred_a = a_pred * dr2[np.newaxis, :]
        V_pred_a = E_pred_a + test['V_nn']

        results[f'a(r), r0={r0}'] = {
            'V_pred': V_pred_a,
            'a_pred': a_pred,
            'a_true': a_test_true,
            'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_a.ravel()),
        }

    # Method 4: Global
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
print('Generating datasets...')
train = make_dataset([1, 5], [0.5, 1.5], n_per_dim=20)
test_ext = make_dataset([7, 12], [0.5, 1.5], n_per_dim=10)
test_int = make_dataset([1.5, 4.5], [0.6, 1.4], n_per_dim=10)

r0_values = [10.0, 15.0, 20.0, 50.0]

print('\n=== EXTRAPOLATION ===')
res_ext = run_experiment(train, test_ext, r0_values)
direct_mae = res_ext['Direct']['mae']
for name, res in sorted(res_ext.items(), key=lambda x: x[1]['mae']):
    ratio = direct_mae / res['mae'] if res['mae'] > 0 else float('inf')
    print(f'  {name:22s}: MAE = {res["mae"]:.4f} Ha  ({ratio:.2f}x vs direct)')

print('\n=== INTERPOLATION ===')
res_int = run_experiment(train, test_int, r0_values)
direct_mae_int = res_int['Direct']['mae']
for name, res in sorted(res_int.items(), key=lambda x: x[1]['mae']):
    ratio = direct_mae_int / res['mae'] if res['mae'] > 0 else float('inf')
    print(f'  {name:22s}: MAE = {res["mae"]:.6f} Ha  ({ratio:.2f}x vs direct)')

# ================================================================
# FIGURE 1: What a(r) looks like (train vs test)
# ================================================================
r = train['r']
r0_plot = 10.0
dr2 = (r - r0_plot)**2

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

# Training
for i in range(0, len(train['E_elec']), 40):
    axes[0, 0].plot(r, train['V_total'][i], alpha=0.4, lw=1)
    axes[0, 1].plot(r, train['E_elec'][i], alpha=0.4, lw=1)
    a_i = train['E_elec'][i] / dr2
    axes[0, 2].plot(r, a_i, alpha=0.4, lw=1)

# Test
for i in range(0, len(test_ext['E_elec']), 10):
    axes[1, 0].plot(r, test_ext['V_total'][i], alpha=0.4, lw=1)
    axes[1, 1].plot(r, test_ext['E_elec'][i], alpha=0.4, lw=1)
    a_i = test_ext['E_elec'][i] / dr2
    axes[1, 2].plot(r, a_i, alpha=0.4, lw=1)

axes[0, 0].set_title('$V_{total}$ (train)')
axes[0, 1].set_title('$E_{elec}$ (train)')
axes[0, 2].set_title(f'$a(r) = E_{{elec}} / (r - {r0_plot:.0f})^2$ (train)')
axes[1, 0].set_title('$V_{total}$ (extrap test)')
axes[1, 1].set_title('$E_{elec}$ (extrap test)')
axes[1, 2].set_title(f'$a(r)$ (extrap test)')

for ax in axes[:, 0]:
    ax.set_ylabel('Energy [Ha]')
    ax.set_ylim(-1, 3)
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
for ax in axes[1, :]:
    ax.set_xlabel('r [Bohr]')

fig.suptitle(f'Parabola curvature decomposition: $E_{{elec}}(r) = a(r) \\cdot (r - r_0)^2$, $r_0 = {r0_plot:.0f}$ Bohr',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_periodic_parabola_curves.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_periodic_parabola_curves.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Summary bar chart
# ================================================================
methods_to_plot = ['Direct', 'Elec direct', f'a(r), r0=10.0', 'Global (Rose)']
labels = ['Direct', 'Elec\ndirect', 'Curvature\na(r)', 'Global\n(Rose)']
colors = ['#d62728', '#9467bd', '#2ca02c', '#1f77b4']

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

mae_ext = [res_ext[m]['mae'] for m in methods_to_plot]
mae_int = [res_int[m]['mae'] for m in methods_to_plot]
x = np.arange(len(methods_to_plot))

ax1.bar(x, mae_ext, color=colors)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('MAE [Ha]')
ax1.set_title('Extrapolation')
ax1.grid(True, alpha=0.3, axis='y')
for i, m in enumerate(methods_to_plot):
    if m != 'Direct':
        ratio = direct_mae / res_ext[m]['mae']
        ax1.annotate(f'{ratio:.1f}x', xy=(x[i], mae_ext[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

ax2.bar(x, mae_int, color=colors, alpha=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('MAE [Ha]')
ax2.set_title('Interpolation')
ax2.grid(True, alpha=0.3, axis='y')
for i, m in enumerate(methods_to_plot):
    if m != 'Direct':
        ratio = direct_mae_int / res_int[m]['mae']
        ax2.annotate(f'{ratio:.1f}x', xy=(x[i], mae_int[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_periodic_parabola_summary.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_periodic_parabola_summary.png')
plt.close(fig2)
