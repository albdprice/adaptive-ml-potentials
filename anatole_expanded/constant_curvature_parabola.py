"""
Test Anatole's parabola idea CORRECTLY: one constant curvature per molecule.

For each molecule i, fit a SINGLE scalar curvature a_i (plus offset c_i):
  E_elec(d) ≈ a_i · (d - d₀)² + c_i

Then learn a_i and c_i from λ using Ridge, and reconstruct for test molecules.

This is a true adaptive parameterization:
  λ → (a, c) → parabola equation → E_elec(d)
analogous to:
  λ → (D_e, r_e, α) → Morse equation → V(d)
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
# Fit constant-curvature parabola to each molecule
# ================================================================
# Model: E_elec(d) = a · (d - d₀)² + c
# Two parameters per molecule: a_i and c_i
# d₀ is fixed (shared across molecules)
# For the fit, this is just linear regression:
#   E_elec = a · (d-d₀)² + c
#   features: [(d-d₀)², 1], targets: E_elec
# Closed-form least squares for each molecule.

# Also try: E_elec(d) = a·(d-d₀)² + b·(d-d₀) + c  (general quadratic, 3 params)

print(f'{"="*80}')
print(f'Fitting constant-curvature parabolas to E_elec(d)')
print(f'{"="*80}')

# Try different d₀ values
for d0_label, d0 in [('d₀=0 Å', 0.0), ('d₀=1.5 Å', 1.5), ('d₀=5 Å', 5.0)]:

    dr = d_grid - d0
    dr2 = dr**2

    # ---- 2-parameter fit: E = a·(d-d₀)² + c ----
    params_2p = np.zeros((7, 2))  # [a, c]
    fit_mae_2p = np.zeros(7)

    # ---- 3-parameter fit: E = a·(d-d₀)² + b·(d-d₀) + c ----
    params_3p = np.zeros((7, 3))  # [a, b, c]
    fit_mae_3p = np.zeros(7)

    for i in range(7):
        # 2-param: E = a·dr² + c
        A2 = np.column_stack([dr2, np.ones(n_grid)])
        p2 = np.linalg.lstsq(A2, E_elec[i], rcond=None)[0]
        params_2p[i] = p2
        fit_mae_2p[i] = mean_absolute_error(E_elec[i], A2 @ p2)

        # 3-param: E = a·dr² + b·dr + c
        A3 = np.column_stack([dr2, dr, np.ones(n_grid)])
        p3 = np.linalg.lstsq(A3, E_elec[i], rcond=None)[0]
        params_3p[i] = p3
        fit_mae_3p[i] = mean_absolute_error(E_elec[i], A3 @ p3)

    print(f'\n--- {d0_label} ---')
    print(f'{"Mol":<6s} | {"a (2p)":>10s} {"c (2p)":>10s} {"MAE 2p":>8s} | '
          f'{"a (3p)":>10s} {"b (3p)":>10s} {"c (3p)":>10s} {"MAE 3p":>8s}')
    print(f'{"-"*90}')
    for i in range(7):
        marker = ' ←train' if i in train_idx else ' ←TEST'
        print(f'{mol_names[i]:<6s} | {params_2p[i,0]:10.4f} {params_2p[i,1]:10.2f} '
              f'{fit_mae_2p[i]:8.4f} | '
              f'{params_3p[i,0]:10.4f} {params_3p[i,1]:10.4f} {params_3p[i,2]:10.2f} '
              f'{fit_mae_3p[i]:8.4f}{marker}')


# ================================================================
# Use the best d₀ and do the ML experiment
# ================================================================
# Use d₀=1.5 (near equilibrium, so parabola is centered in the data)
# and also d₀=0 (vertex at d=0, so parabola is monotonic over data range)

for d0_label, d0 in [('d₀=0.0', 0.0), ('d₀=1.5', 1.5)]:
    dr = d_grid - d0
    dr2 = dr**2

    print(f'\n{"="*80}')
    print(f'ML EXPERIMENT with {d0_label}')
    print(f'{"="*80}')

    # Fit parameters for each molecule
    params_2p = np.zeros((7, 2))
    params_3p = np.zeros((7, 3))

    for i in range(7):
        A2 = np.column_stack([dr2, np.ones(n_grid)])
        params_2p[i] = np.linalg.lstsq(A2, E_elec[i], rcond=None)[0]

        A3 = np.column_stack([dr2, dr, np.ones(n_grid)])
        params_3p[i] = np.linalg.lstsq(A3, E_elec[i], rcond=None)[0]

    # ML: learn parameters from λ using Ridge
    X_tr = lam_train.reshape(-1, 1)
    X_te = lam_test.reshape(-1, 1)

    # 2-param adaptive: λ → (a, c)
    pred_2p = np.zeros((3, 2))
    for p in range(2):
        m = Ridge(alpha=1e-6)
        m.fit(X_tr, params_2p[train_idx, p])
        pred_2p[:, p] = m.predict(X_te)

    E_pred_2p = np.zeros((3, n_grid))
    for j in range(3):
        E_pred_2p[j] = pred_2p[j, 0] * dr2 + pred_2p[j, 1]

    # 3-param adaptive: λ → (a, b, c)
    pred_3p = np.zeros((3, 3))
    for p in range(3):
        m = Ridge(alpha=1e-6)
        m.fit(X_tr, params_3p[train_idx, p])
        pred_3p[:, p] = m.predict(X_te)

    E_pred_3p = np.zeros((3, n_grid))
    for j in range(3):
        E_pred_3p[j] = pred_3p[j, 0] * dr2 + pred_3p[j, 1] * dr + pred_3p[j, 2]

    # Baselines
    mdl_direct = Ridge(alpha=1e-6)
    mdl_direct.fit(X_tr, E_elec[train_idx])
    E_pred_direct = mdl_direct.predict(X_te)

    mdl_aha = Ridge(alpha=1e-6)
    mdl_aha.fit(np.column_stack([lam_train, lam_train**2]), E_elec[train_idx])
    E_pred_aha = mdl_aha.predict(np.column_stack([lam_test, lam_test**2]))

    # Results
    mae_direct = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_direct.ravel())
    mae_aha = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_aha.ravel())
    mae_2p = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_2p.ravel())
    mae_3p = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_3p.ravel())

    print(f'\n{"Method":<45s} | {"MAE":>7s} | {"ratio":>7s}')
    print(f'{"-"*65}')
    print(f'{"Direct multioutput [λ]→E(d)":<45s} | {mae_direct:7.2f} | {"base":>7s}')
    print(f'{"AHA multioutput [λ,λ²]→E(d)":<45s} | {mae_aha:7.2f} | {mae_aha/mae_direct:7.3f}x')
    print(f'{"Adaptive parabola λ→(a,c) 2-param":<45s} | {mae_2p:7.2f} | {mae_2p/mae_direct:7.3f}x')
    print(f'{"Adaptive quadratic λ→(a,b,c) 3-param":<45s} | {mae_3p:7.2f} | {mae_3p/mae_direct:7.3f}x')
    print(f'{"-"*65}')

    # Per-molecule
    print(f'\nPer-molecule MAE:')
    print(f'{"Method":<35s} | {"LiNa":>7s} {"HeMg":>7s} {"HAl":>7s}')
    for label, pred in [('Direct [λ]', E_pred_direct),
                         ('AHA [λ,λ²]', E_pred_aha),
                         ('Parabola (a,c)', E_pred_2p),
                         ('Quadratic (a,b,c)', E_pred_3p)]:
        per = [mean_absolute_error(E_elec[test_idx[j]], pred[j]) for j in range(3)]
        print(f'{label:<35s} | {per[0]:7.2f} {per[1]:7.2f} {per[2]:7.2f}')

    # Check: how do parameters vary with λ?
    print(f'\nParameter trends (2-param):')
    print(f'{"Mol":<6s} | {"λ":>4s} | {"a":>10s} | {"c":>10s}')
    for i in range(7):
        marker = '  ←pred' if i in test_idx else ''
        if i in test_idx:
            j = test_idx.index(i)
            print(f'{mol_names[i]:<6s} | {lambdas[i]:4.0f} | {pred_2p[j,0]:10.4f} | {pred_2p[j,1]:10.2f}{marker}')
        else:
            print(f'{mol_names[i]:<6s} | {lambdas[i]:4.0f} | {params_2p[i,0]:10.4f} | {params_2p[i,1]:10.2f}')

    # ================================================================
    # Figure: fits + predictions
    # ================================================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Top row: parabola fit quality for each molecule
    for i in range(7):
        ax = axes.flat[i]
        ax.plot(d_grid, E_elec[i], 'k-', lw=2, label='DFT')
        fit_2p = params_2p[i, 0] * dr2 + params_2p[i, 1]
        fit_3p = params_3p[i, 0] * dr2 + params_3p[i, 1] * dr + params_3p[i, 2]
        mae2 = mean_absolute_error(E_elec[i], fit_2p)
        mae3 = mean_absolute_error(E_elec[i], fit_3p)
        ax.plot(d_grid, fit_2p, 'r--', lw=1.5, label=f'2p fit ({mae2:.2f})')
        ax.plot(d_grid, fit_3p, 'g:', lw=1.5, label=f'3p fit ({mae3:.2f})')

        color = '#d62728' if i in test_idx else '#1f77b4'
        ax.set_title(f'{mol_names[i]} (λ={i})', fontsize=11, color=color,
                     fontweight='bold' if i in test_idx else 'normal')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('d [Å]')
        ax.set_ylabel('$E_{elec}$ [Ha]')

    # Bottom-right: predictions for test molecules (overlaid)
    ax_pred = axes[1, 3]
    ax_pred.set_visible(True)

    # Show parameter a vs λ
    ax_pred.plot(range(7), params_2p[:, 0], 'ko-', ms=8, label='True a (2-param)')
    ax_pred.plot([4, 5, 6], pred_2p[:, 0], 'rs', ms=10, label='Predicted a')
    for i in range(7):
        ax_pred.annotate(mol_names[i], (i, params_2p[i, 0]),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax_pred.set_xlabel('λ', fontsize=11)
    ax_pred.set_ylabel('Curvature a', fontsize=11)
    ax_pred.set_title('Curvature a vs λ — is it learnable?', fontsize=11)
    ax_pred.legend(fontsize=8)
    ax_pred.grid(True, alpha=0.3)

    fig.suptitle(f'Constant-curvature parabola fits ({d0_label})\n'
                 f'Top: fit quality per molecule. Bottom-right: parameter a vs λ.',
                 fontsize=13)
    fig.tight_layout()
    fname = f'fig_constant_parabola_{d0_label.replace("=","").replace(".","p").replace(" ","")}.png'
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close(fig)

    # ================================================================
    # Figure: predictions vs DFT for test molecules
    # ================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    for j, ax in enumerate(axes2):
        idx = test_idx[j]
        name = mol_names[idx]

        ax.plot(d_grid, E_elec[idx], 'k-', lw=2.5, label='DFT reference')

        mae_d = mean_absolute_error(E_elec[idx], E_pred_direct[j])
        ax.plot(d_grid, E_pred_direct[j], 'b--', lw=1.5,
                label=f'Direct [λ] ({mae_d:.2f} Ha)')

        mae_a = mean_absolute_error(E_elec[idx], E_pred_aha[j])
        ax.plot(d_grid, E_pred_aha[j], 'g-', lw=2,
                label=f'AHA [λ,λ²] ({mae_a:.2f} Ha)')

        mae_p = mean_absolute_error(E_elec[idx], E_pred_3p[j])
        ax.plot(d_grid, E_pred_3p[j], 'r--', lw=1.5,
                label=f'Adaptive quad (a,b,c) ({mae_p:.2f} Ha)')

        ax.set_xlabel('d [Å]', fontsize=11)
        ax.set_ylabel('$E_{elec}$ [Ha]', fontsize=11)
        ax.set_title(f'{name} (λ={idx})', fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f'Extrapolation predictions ({d0_label}) — adaptive parabola vs AHA vs direct', fontsize=12)
    fig2.tight_layout()
    fname2 = f'fig_predictions_{d0_label.replace("=","").replace(".","p").replace(" ","")}.png'
    fig2.savefig(os.path.join(FIGDIR, fname2), dpi=150, bbox_inches='tight')
    print(f'Saved {fname2}')
    plt.close(fig2)
