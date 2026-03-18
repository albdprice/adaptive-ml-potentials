"""
Anatole's proposal: Morse as total energy, subtract V_nn, learn E_elec.

Setup (H-H like, atomic units):
  V_total(r) = V_Morse(r)            (synthetic total energy)
  V_nn(r)    = Z_A * Z_B / r = 1/r   (known, Z_A = Z_B = 1)
  E_elec(r)  = V_Morse(r) - 1/r      (electronic remainder)

Descriptor-based parameter mapping:
  D_e   = 0.1 + 0.1 * d1
  R_e   = 1.8 + 0.2 * d2     (small variation so grid covers all equilibria)
  alpha = 0.8 + 0.4 * d1

Train: d1, d2 in [0.5, 1.5]
Test:  d1, d2 in [2.5, 4.0]  (extrapolation)

ML methods:
  1. Direct:       d -> V_total(r)
  2. Elec direct:  d -> E_elec(r), reconstruct V_total = E_elec + V_nn
  3. b=0 on Elec:  d -> (r_e_elec, a(r_i)), reconstruct E_elec = a*(r-r_e_elec)^2
  4. Global:       d -> (D_e, R_e, alpha), reconstruct V_Morse analytically
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
Z_AB = 1.0  # Z_A * Z_B for H-H like system

def make_dataset(d1_range, d2_range, n_per_dim=20):
    """Generate Morse-based curves with nuclear subtraction."""
    d1_vals = np.linspace(d1_range[0], d1_range[1], n_per_dim)
    d2_vals = np.linspace(d2_range[0], d2_range[1], n_per_dim)
    d1_grid, d2_grid = np.meshgrid(d1_vals, d2_vals)
    d1_flat = d1_grid.ravel()
    d2_flat = d2_grid.ravel()
    n_samples = len(d1_flat)

    # Parameters from descriptors
    D_e = 0.1 + 0.1 * d1_flat
    R_e = 1.8 + 0.2 * d2_flat
    alpha = 0.8 + 0.4 * d1_flat

    # Grid (Bohr), truncated at 0.7 to avoid worst singularity
    r = np.linspace(0.7, 8.0, 50)
    n_grid = len(r)

    # Compute curves
    V_total = np.zeros((n_samples, n_grid))
    V_nn = Z_AB / r  # same for all (fixed Z)
    E_elec = np.zeros((n_samples, n_grid))
    r_e_elec = np.zeros(n_samples)
    a_r = np.zeros((n_samples, n_grid))  # b=0 coefficient

    for i in range(n_samples):
        u = np.exp(-alpha[i] * (r - R_e[i]))
        V_total[i] = D_e[i] * (1 - u)**2 - D_e[i]
        E_elec[i] = V_total[i] - V_nn

        # Find equilibrium of E_elec
        idx_min = np.argmin(E_elec[i])
        r_e_elec[i] = r[idx_min]

        # b=0 decomposition: a(r) = E_elec / (r - r_e_elec)^2
        dr = r - r_e_elec[i]
        safe = np.abs(dr) > 0.05
        a_r[i, safe] = E_elec[i, safe] / dr[safe]**2
        # At equilibrium, use L'Hopital: a = E_elec''(r_e)/2
        V_morse_d2 = 2 * D_e[i] * alpha[i]**2 * u * (2*u - 1)
        V_nn_d2 = 2 * Z_AB / r**3
        a_eq = (V_morse_d2[idx_min] - V_nn_d2[idx_min]) / 2.0
        a_r[i, ~safe] = a_eq

    descriptors = np.column_stack([d1_flat, d2_flat])
    params = np.column_stack([D_e, R_e, alpha])

    return {
        'descriptors': descriptors,
        'params': params,
        'r': r,
        'V_total': V_total,
        'V_nn': V_nn,
        'E_elec': E_elec,
        'r_e_elec': r_e_elec,
        'a_r': a_r,
    }


# ================================================================
# ML experiment
# ================================================================
def run_experiment(train, test, label=''):
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
    V_pred_elec = E_pred + test['V_nn'][np.newaxis, :] if test['V_nn'].ndim == 1 else E_pred + test['V_nn']
    # V_nn is the same for all test samples (same Z)
    V_pred_elec = E_pred + test['V_nn']
    results['Elec direct'] = {
        'V_pred': V_pred_elec,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_elec.ravel()),
    }

    # --- Method 3: b=0 on E_elec (d -> r_e_elec + a(r), reconstruct) ---
    # Predict r_e_elec
    mdl_re = Ridge(alpha=1.0)
    mdl_re.fit(X_train, train['r_e_elec'])
    r_e_elec_pred = mdl_re.predict(X_test)

    # Predict a(r) at each grid point
    mdl_a = Ridge(alpha=1.0)
    mdl_a.fit(X_train, train['a_r'])
    a_pred = mdl_a.predict(X_test)

    # Reconstruct: E_elec = a(r) * (r - r_e_elec)^2, V_total = E_elec + V_nn
    V_pred_b0 = np.zeros((n_test, len(r)))
    for i in range(n_test):
        dr = r - r_e_elec_pred[i]
        E_elec_recon = a_pred[i] * dr**2
        V_pred_b0[i] = E_elec_recon + test['V_nn']

    results['b=0 on Elec'] = {
        'V_pred': V_pred_b0,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_b0.ravel()),
    }

    # --- Method 4: Global adaptive (d -> D_e, R_e, alpha) ---
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

    results['Global (D_e,R_e,α)'] = {
        'V_pred': V_pred_global,
        'mae': mean_absolute_error(test['V_total'].ravel(), V_pred_global.ravel()),
    }

    return results


# ================================================================
# Run
# ================================================================
print('Generating datasets...')
train = make_dataset([0.5, 1.5], [0.5, 1.5], n_per_dim=20)
test_ext = make_dataset([2.5, 4.0], [2.5, 4.0], n_per_dim=10)
test_int = make_dataset([0.6, 1.4], [0.6, 1.4], n_per_dim=10)

print(f'Training: {len(train["descriptors"])} samples')
print(f'  D_e:   [{train["params"][:,0].min():.3f}, {train["params"][:,0].max():.3f}] Ha')
print(f'  R_e:   [{train["params"][:,1].min():.2f}, {train["params"][:,1].max():.2f}] Bohr')
print(f'  alpha: [{train["params"][:,2].min():.2f}, {train["params"][:,2].max():.2f}]')
print(f'Test extrap: {len(test_ext["descriptors"])} samples')
print(f'  D_e:   [{test_ext["params"][:,0].min():.3f}, {test_ext["params"][:,0].max():.3f}]')
print(f'  R_e:   [{test_ext["params"][:,1].min():.2f}, {test_ext["params"][:,1].max():.2f}]')
print(f'  alpha: [{test_ext["params"][:,2].min():.2f}, {test_ext["params"][:,2].max():.2f}]')

print('\n=== EXTRAPOLATION ===')
res_ext = run_experiment(train, test_ext)
direct_mae = res_ext['Direct']['mae']
for name, res in res_ext.items():
    ratio = direct_mae / res['mae'] if res['mae'] > 0 else float('inf')
    marker = ' <-- baseline' if name == 'Direct' else f' ({ratio:.2f}x vs direct)'
    print(f'  {name:22s}: MAE = {res["mae"]:.4f} Ha{marker}')

print('\n=== INTERPOLATION ===')
res_int = run_experiment(train, test_int)
direct_mae_int = res_int['Direct']['mae']
for name, res in res_int.items():
    ratio = direct_mae_int / res['mae'] if res['mae'] > 0 else float('inf')
    marker = ' <-- baseline' if name == 'Direct' else f' ({ratio:.2f}x vs direct)'
    print(f'  {name:22s}: MAE = {res["mae"]:.6f} Ha{marker}')

# ================================================================
# FIGURE 1: Visualize the curves (train + test)
# ================================================================
r = train['r']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Row 1: Training examples
for i in range(0, len(train['V_total']), 80):
    axes[0, 0].plot(r, train['V_total'][i], alpha=0.5, lw=1)
    axes[0, 1].plot(r, train['E_elec'][i], alpha=0.5, lw=1)
    axes[0, 2].plot(r, train['a_r'][i], alpha=0.5, lw=1)
axes[0, 0].set_title('$V_{total} = V_{Morse}$ (train)')
axes[0, 1].set_title('$E_{elec} = V_{Morse} - 1/r$ (train)')
axes[0, 2].set_title('$a(r) = E_{elec}/(r - r_{e,elec})^2$ (train)')

# Row 2: Test (extrapolation) examples
for i in range(0, len(test_ext['V_total']), 20):
    axes[1, 0].plot(r, test_ext['V_total'][i], alpha=0.5, lw=1)
    axes[1, 1].plot(r, test_ext['E_elec'][i], alpha=0.5, lw=1)
    axes[1, 2].plot(r, test_ext['a_r'][i], alpha=0.5, lw=1)
axes[1, 0].set_title('$V_{total}$ (extrap test)')
axes[1, 1].set_title('$E_{elec}$ (extrap test)')
axes[1, 2].set_title('$a(r)$ (extrap test)')

for ax in axes[:, 0]:
    ax.set_ylim(-1.5, 3)
    ax.set_ylabel('Energy [Ha]')
for ax in axes[:, 1]:
    ax.set_ylim(-4, 1)
for ax in axes[:, 2]:
    ax.set_ylim(-1, 5)
for ax in axes[1, :]:
    ax.set_xlabel('r [Bohr]')
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)

fig.suptitle('Training vs Extrapolation: curves and decomposition', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_morse_elec_curves.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_morse_elec_curves.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Example extrapolation predictions
# ================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
examples = [0, len(test_ext['descriptors'])//2, -1]

for col, idx in enumerate(examples):
    d = test_ext['descriptors'][idx]
    p = test_ext['params'][idx]

    # Top: V_total predictions
    ax = axes2[0, col]
    ax.plot(r, test_ext['V_total'][idx], 'b-', lw=2.5, label='True')
    ax.plot(r, res_ext['Direct']['V_pred'][idx], 'r:', lw=2, label='Direct')
    ax.plot(r, res_ext['Elec direct']['V_pred'][idx], 'm--', lw=1.5, label='Elec direct')
    ax.plot(r, res_ext['b=0 on Elec']['V_pred'][idx], 'g--', lw=2, label='b=0 on Elec')
    ax.plot(r, res_ext['Global (D_e,R_e,α)']['V_pred'][idx], 'k-.', lw=2, label='Global')
    ax.set_title(f'$d_1$={d[0]:.1f}, $d_2$={d[1]:.1f}\n'
                 f'$D_e$={p[0]:.2f}, $R_e$={p[1]:.1f}, $\\alpha$={p[2]:.1f}')
    ax.set_ylim(-2, 5)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$V_{total}$ [Ha]')
        ax.legend(fontsize=7)

    # Bottom: Error (V_pred - V_true)
    ax = axes2[1, col]
    for name, color, ls in [('Direct', 'r', ':'), ('Elec direct', 'm', '--'),
                             ('b=0 on Elec', 'g', '--'), ('Global (D_e,R_e,α)', 'k', '-.')]:
        err = res_ext[name]['V_pred'][idx] - test_ext['V_total'][idx]
        ax.plot(r, err, color=color, ls=ls, lw=1.5, label=name)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('r [Bohr]')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('Error [Ha]')
        ax.legend(fontsize=7)

fig2.suptitle('Extrapolation predictions: 4 methods compared', fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_morse_elec_ml_predictions.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_morse_elec_ml_predictions.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Summary bar chart
# ================================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

methods = list(res_ext.keys())
mae_ext = [res_ext[m]['mae'] for m in methods]
mae_int = [res_int[m]['mae'] for m in methods]

x = np.arange(len(methods))
colors = ['#d62728', '#9467bd', '#2ca02c', '#1f77b4']

bars = ax1.bar(x, mae_ext, color=colors)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
ax1.set_ylabel('MAE [Ha]')
ax1.set_title('Extrapolation MAE')
ax1.grid(True, alpha=0.3, axis='y')
for i, (m, mae) in enumerate(zip(methods, mae_ext)):
    if m != 'Direct':
        ratio = direct_mae / mae
        ax1.annotate(f'{ratio:.1f}x', xy=(x[i], mae), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

bars = ax2.bar(x, mae_int, color=colors, alpha=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
ax2.set_ylabel('MAE [Ha]')
ax2.set_title('Interpolation MAE')
ax2.grid(True, alpha=0.3, axis='y')
for i, (m, mae) in enumerate(zip(methods, mae_int)):
    if m != 'Direct':
        ratio = direct_mae_int / mae
        ax2.annotate(f'{ratio:.1f}x', xy=(x[i], mae), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_morse_elec_ml_summary.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_morse_elec_ml_summary.png')
plt.close(fig3)
