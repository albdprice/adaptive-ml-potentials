"""
Multi-system ML experiment on softened Coulomb model.

Physics:
  E_elec(r; Z, a) = -Z / sqrt(r^2 + a^2)     (electronic energy)
  V_nn(r; A)       = A / r                     (nuclear repulsion, KNOWN)
  E_total(r)       = E_elec(r) + V_nn(r)

Parameters from descriptors:
  Z(d1) = 10 + 5*d1      (electronic attraction strength)
  a(d2) = 0.5 + 0.3*d2   (softening / atomic size)
  A(d1) = 5 + 3*d1       (nuclear charge product, KNOWN classically)

ML methods compared:
  1. Direct:     d -> E_total(r_i)           [learn full curve]
  2. Screening:  d -> S(r_i) = E_elec*r      [learn smooth screening, reconstruct]
  3. Global:     d -> (Z, a)                 [learn 2 parameters, reconstruct]

Training:  d1, d2 in [0.5, 1.5]
Testing:   d1, d2 in [2.5, 4.0]  (extrapolation)
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
def make_dataset(d1_range, d2_range, n_per_dim=20):
    """Generate softened Coulomb curves for a grid of descriptors."""
    d1_vals = np.linspace(d1_range[0], d1_range[1], n_per_dim)
    d2_vals = np.linspace(d2_range[0], d2_range[1], n_per_dim)
    d1_grid, d2_grid = np.meshgrid(d1_vals, d2_vals)
    d1_flat = d1_grid.ravel()
    d2_flat = d2_grid.ravel()

    n_samples = len(d1_flat)

    # Parameters from descriptors
    Z = 10 + 5 * d1_flat       # electronic attraction
    a_param = 0.5 + 0.3 * d2_flat   # softening
    A = 5 + 3 * d1_flat        # nuclear repulsion (KNOWN)

    # Grid
    r = np.linspace(0.3, 6.0, 50)  # avoid r=0 for V_nn
    n_grid = len(r)

    # Compute curves
    E_total = np.zeros((n_samples, n_grid))
    E_elec = np.zeros((n_samples, n_grid))
    V_nn = np.zeros((n_samples, n_grid))
    S_screen = np.zeros((n_samples, n_grid))  # screening function

    for i in range(n_samples):
        rho = np.sqrt(r**2 + a_param[i]**2)
        E_elec[i] = -Z[i] / rho
        V_nn[i] = A[i] / r
        E_total[i] = E_elec[i] + V_nn[i]
        S_screen[i] = E_elec[i] * r  # C_{-1}(r) = E_elec * r

    descriptors = np.column_stack([d1_flat, d2_flat])
    params = np.column_stack([Z, a_param, A])

    return {
        'descriptors': descriptors,
        'params': params,  # Z, a, A
        'r': r,
        'E_total': E_total,
        'E_elec': E_elec,
        'V_nn': V_nn,
        'S_screen': S_screen,
    }


# ================================================================
# ML experiment
# ================================================================
def run_experiment(train, test):
    """Compare direct, screening, and global adaptive."""
    r = train['r']
    n_grid = len(r)

    X_train = train['descriptors']
    X_test = test['descriptors']

    results = {}

    # --- Method 1: Direct (d -> E_total) ---
    ridge_direct = Ridge(alpha=1.0)
    ridge_direct.fit(X_train, train['E_total'])
    E_pred_direct = ridge_direct.predict(X_test)

    results['Direct'] = {
        'E_pred': E_pred_direct,
        'mae': mean_absolute_error(test['E_total'].ravel(), E_pred_direct.ravel()),
    }

    # --- Method 2: Screening (d -> S(r), reconstruct E_total = S/r + A/r) ---
    ridge_screen = Ridge(alpha=1.0)
    ridge_screen.fit(X_train, train['S_screen'])
    S_pred = ridge_screen.predict(X_test)

    # Reconstruct: E_elec_pred = S_pred / r, E_total_pred = E_elec_pred + V_nn (known)
    E_elec_pred_screen = S_pred / r[np.newaxis, :]
    E_pred_screen = E_elec_pred_screen + test['V_nn']  # V_nn is KNOWN

    results['Screening'] = {
        'E_pred': E_pred_screen,
        'S_pred': S_pred,
        'mae': mean_absolute_error(test['E_total'].ravel(), E_pred_screen.ravel()),
    }

    # --- Method 3: Global adaptive (d -> Z, a; reconstruct analytically) ---
    # Learn Z and a separately
    Z_train = train['params'][:, 0]
    a_train = train['params'][:, 1]

    ridge_Z = Ridge(alpha=1.0)
    ridge_Z.fit(X_train, Z_train)
    Z_pred = ridge_Z.predict(X_test)

    ridge_a = Ridge(alpha=1.0)
    ridge_a.fit(X_train, a_train)
    a_pred = ridge_a.predict(X_test)

    # Reconstruct
    E_pred_global = np.zeros_like(test['E_total'])
    for i in range(len(X_test)):
        rho = np.sqrt(r**2 + a_pred[i]**2)
        E_elec_i = -Z_pred[i] / rho
        E_pred_global[i] = E_elec_i + test['V_nn'][i]

    results['Global (Z, a)'] = {
        'E_pred': E_pred_global,
        'Z_pred': Z_pred,
        'a_pred': a_pred,
        'mae': mean_absolute_error(test['E_total'].ravel(), E_pred_global.ravel()),
    }

    return results


# ================================================================
# Run
# ================================================================
print('Generating datasets...')
train = make_dataset([0.5, 1.5], [0.5, 1.5], n_per_dim=20)
test_extrap = make_dataset([2.5, 4.0], [2.5, 4.0], n_per_dim=10)
test_interp = make_dataset([0.6, 1.4], [0.6, 1.4], n_per_dim=10)

print(f'Training: {len(train["descriptors"])} samples')
print(f'  Z range: [{train["params"][:,0].min():.1f}, {train["params"][:,0].max():.1f}]')
print(f'  a range: [{train["params"][:,1].min():.2f}, {train["params"][:,1].max():.2f}]')
print(f'  A range: [{train["params"][:,2].min():.1f}, {train["params"][:,2].max():.1f}]')
print(f'Test (extrap): {len(test_extrap["descriptors"])} samples')
print(f'  Z range: [{test_extrap["params"][:,0].min():.1f}, {test_extrap["params"][:,0].max():.1f}]')
print(f'  a range: [{test_extrap["params"][:,1].min():.2f}, {test_extrap["params"][:,1].max():.2f}]')

print('\n--- Extrapolation ---')
res_ext = run_experiment(train, test_extrap)
for name, res in res_ext.items():
    print(f'  {name:20s}: MAE = {res["mae"]:.4f} eV')
direct_mae_ext = res_ext['Direct']['mae']
for name, res in res_ext.items():
    if name != 'Direct':
        print(f'    -> {name} ratio (Direct/method): {direct_mae_ext / res["mae"]:.2f}x')

print('\n--- Interpolation ---')
res_int = run_experiment(train, test_interp)
for name, res in res_int.items():
    print(f'  {name:20s}: MAE = {res["mae"]:.6f} eV')
direct_mae_int = res_int['Direct']['mae']
for name, res in res_int.items():
    if name != 'Direct':
        print(f'    -> {name} ratio (Direct/method): {direct_mae_int / res["mae"]:.2f}x')

# ================================================================
# Diagnostic: what do the screening functions look like?
# ================================================================
print('\n--- Screening function S(r) = E_elec * r diagnostics ---')
print(f'Train S range: [{train["S_screen"].min():.2f}, {train["S_screen"].max():.2f}]')
print(f'Test  S range: [{test_extrap["S_screen"].min():.2f}, {test_extrap["S_screen"].max():.2f}]')
print(f'Train E_total range: [{train["E_total"].min():.2f}, {train["E_total"].max():.2f}]')
print(f'Test  E_total range: [{test_extrap["E_total"].min():.2f}, {test_extrap["E_total"].max():.2f}]')

# ================================================================
# FIGURE 1: Example predictions (extrapolation)
# ================================================================
r = train['r']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Pick 3 example test molecules
examples = [0, len(test_extrap['descriptors'])//2, -1]
titles = ['Low Z, small a', 'Mid Z, mid a', 'High Z, large a']

for col, (idx, title) in enumerate(zip(examples, titles)):
    d = test_extrap['descriptors'][idx]
    Z_true = test_extrap['params'][idx, 0]
    a_true = test_extrap['params'][idx, 1]

    # Top: E_total predictions
    ax = axes[0, col]
    ax.plot(r, test_extrap['E_total'][idx], 'b-', lw=2.5, label='True')
    ax.plot(r, res_ext['Direct']['E_pred'][idx], 'r:', lw=2, label='Direct')
    ax.plot(r, res_ext['Screening']['E_pred'][idx], 'g--', lw=2, label='Screening')
    ax.plot(r, res_ext['Global (Z, a)']['E_pred'][idx], 'k-.', lw=2, label='Global')
    ax.set_title(f'{title}\n$d_1$={d[0]:.1f}, $d_2$={d[1]:.1f} | Z={Z_true:.0f}, a={a_true:.2f}')
    ax.set_ylabel('$E_{total}$ [eV]')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(fontsize=8)

    # Bottom: Screening function S(r)
    ax = axes[1, col]
    ax.plot(r, test_extrap['S_screen'][idx], 'g-', lw=2.5, label='$S_{true}$')
    ax.plot(r, res_ext['Screening']['S_pred'][idx], 'k--', lw=2, label='$S_{pred}$')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('$S(r)$ [eV·Å]')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(fontsize=8)

axes[0, 1].set_title('Extrapolation: $E_{total}$ predictions\n' + axes[0, 1].get_title())
axes[1, 1].set_title('Screening function $S(r) = E_{elec} \\cdot r$')

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_softened_coulomb_ml_examples.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_softened_coulomb_ml_examples.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Summary bar chart + screening smoothness
# ================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
methods = list(res_ext.keys())
mae_ext = [res_ext[m]['mae'] for m in methods]
mae_int = [res_int[m]['mae'] for m in methods]

x = np.arange(len(methods))
width = 0.35
bars1 = ax1.bar(x - width/2, mae_ext, width, label='Extrapolation', color=['#d62728', '#2ca02c', '#1f77b4'])
bars2 = ax1.bar(x + width/2, mae_int, width, label='Interpolation', color=['#d62728', '#2ca02c', '#1f77b4'], alpha=0.4)

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylabel('MAE [eV]')
ax1.set_title('Extrapolation vs Interpolation MAE')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add ratio annotations
for i, m in enumerate(methods):
    if m != 'Direct':
        ratio_ext = direct_mae_ext / res_ext[m]['mae']
        ax1.annotate(f'{ratio_ext:.1f}x', xy=(x[i] - width/2, mae_ext[i]),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

# Screening function: train vs test
ax2.set_title('Screening function $S(r)$: train vs extrapolation')
# Plot a few training curves
for i in range(0, len(train['S_screen']), 40):
    ax2.plot(r, train['S_screen'][i], 'b-', alpha=0.15, lw=1)
ax2.plot([], [], 'b-', alpha=0.4, label=f'Training ({len(train["S_screen"])} curves)')

# Plot a few test curves
for i in range(0, len(test_extrap['S_screen']), 10):
    ax2.plot(r, test_extrap['S_screen'][i], 'r-', alpha=0.3, lw=1)
ax2.plot([], [], 'r-', alpha=0.5, label=f'Extrap test ({len(test_extrap["S_screen"])} curves)')

ax2.set_xlabel('r [Å]')
ax2.set_ylabel('$S(r)$ [eV·Å]')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_softened_coulomb_ml_summary.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_softened_coulomb_ml_summary.png')
plt.close(fig2)
