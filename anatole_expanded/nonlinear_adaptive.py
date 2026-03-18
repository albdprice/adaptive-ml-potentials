"""
Find a nonlinear equation for E_elec(d) that enables adaptive advantage.

The parabola E = a·(d-d₀)² + c is LINEAR in parameters → no advantage.
We need equations NONLINEAR in parameters, like Rose/Morse.

Candidates:
  1. Exponential:  E(d) = A·exp(-α·d) + C          (nonlinear in α)
  2. KKL:          E(d) = A·exp(-c·√d) + C          (nonlinear in c)
  3. Power law:    E(d) = A·d^(-n) + C               (nonlinear in n)

For each: fit to each molecule, learn params from λ, reconstruct, compare.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
# Define candidate equations
# ================================================================
def func_exp(d, A, alpha, C):
    """E = A·exp(-α·d) + C — nonlinear in α"""
    return A * np.exp(-alpha * d) + C

def func_kkl(d, A, c, C):
    """E = A·exp(-c·√d) + C — KKL form, nonlinear in c"""
    return A * np.exp(-c * np.sqrt(d)) + C

def func_power(d, A, n, C):
    """E = A·d^(-n) + C — nonlinear in n"""
    return A * d**(-n) + C

equations = {
    'Exponential: A·exp(-αd)+C': {
        'func': func_exp,
        'p0': lambda E: [-30, 2.0, E[-1]],
        'bounds': ([-500, 0.1, -300], [0, 20, 0]),
        'param_names': ['A', 'α', 'C'],
    },
    'KKL: A·exp(-c√d)+C': {
        'func': func_kkl,
        'p0': lambda E: [-30, 2.0, E[-1]],
        'bounds': ([-500, 0.1, -300], [0, 20, 0]),
        'param_names': ['A', 'c', 'C'],
    },
    'Power: A·d⁻ⁿ+C': {
        'func': func_power,
        'p0': lambda E: [-20, 1.0, E[-1]],
        'bounds': ([-500, 0.01, -300], [0, 10, 0]),
        'param_names': ['A', 'n', 'C'],
    },
}

# ================================================================
# Fit each equation to each molecule
# ================================================================
all_params = {}
all_fit_mae = {}

for eq_name, eq_info in equations.items():
    func = eq_info['func']
    params = np.zeros((7, 3))
    fit_maes = np.zeros(7)

    print(f'\n{"="*70}')
    print(f'{eq_name}')
    print(f'{"="*70}')
    print(f'{"Mol":<6s} | {eq_info["param_names"][0]:>8s} {eq_info["param_names"][1]:>8s} '
          f'{eq_info["param_names"][2]:>8s} | {"fit MAE":>8s}')
    print(f'{"-"*50}')

    for i in range(7):
        try:
            p0 = eq_info['p0'](E_elec[i])
            popt, _ = curve_fit(func, d_grid, E_elec[i], p0=p0,
                               bounds=eq_info['bounds'], maxfev=10000)
            params[i] = popt
            fit_pred = func(d_grid, *popt)
            fit_maes[i] = mean_absolute_error(E_elec[i], fit_pred)
        except Exception as e:
            print(f'  FIT FAILED for {mol_names[i]}: {e}')
            params[i] = [np.nan]*3
            fit_maes[i] = np.nan

        marker = '  TEST' if i in test_idx else ''
        print(f'{mol_names[i]:<6s} | {params[i,0]:8.3f} {params[i,1]:8.4f} '
              f'{params[i,2]:8.2f} | {fit_maes[i]:8.4f}{marker}')

    all_params[eq_name] = params
    all_fit_mae[eq_name] = fit_maes

# ================================================================
# ML experiment: learn params from λ, reconstruct
# ================================================================
print(f'\n\n{"="*80}')
print(f'ML EXTRAPOLATION RESULTS')
print(f'{"="*80}')

# Baseline: direct multioutput
mdl_direct = Ridge(alpha=1e-6)
mdl_direct.fit(lam_train.reshape(-1, 1), E_elec[train_idx])
E_pred_direct = mdl_direct.predict(lam_test.reshape(-1, 1))
mae_direct = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_direct.ravel())

# AHA baseline
mdl_aha = Ridge(alpha=1e-6)
mdl_aha.fit(np.column_stack([lam_train, lam_train**2]), E_elec[train_idx])
E_pred_aha = mdl_aha.predict(np.column_stack([lam_test, lam_test**2]))
mae_aha = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_aha.ravel())

print(f'\n{"Method":<45s} | {"MAE":>7s} | {"ratio":>7s} | {"LiNa":>6s} {"HeMg":>6s} {"HAl":>6s}')
print(f'{"-"*85}')
print(f'{"Direct multioutput [λ]→E(d)":<45s} | {mae_direct:7.2f} | {"base":>7s} | '
      f'{mean_absolute_error(E_elec[4], E_pred_direct[0]):6.2f} '
      f'{mean_absolute_error(E_elec[5], E_pred_direct[1]):6.2f} '
      f'{mean_absolute_error(E_elec[6], E_pred_direct[2]):6.2f}')
print(f'{"AHA multioutput [λ,λ²]→E(d)":<45s} | {mae_aha:7.2f} | {mae_aha/mae_direct:7.3f}x | '
      f'{mean_absolute_error(E_elec[4], E_pred_aha[0]):6.2f} '
      f'{mean_absolute_error(E_elec[5], E_pred_aha[1]):6.2f} '
      f'{mean_absolute_error(E_elec[6], E_pred_aha[2]):6.2f}')
print(f'{"-"*85}')

all_preds = {}
all_preds['Direct [λ]'] = E_pred_direct
all_preds['AHA [λ,λ²]'] = E_pred_aha

for eq_name, eq_info in equations.items():
    func = eq_info['func']
    params = all_params[eq_name]

    if np.any(np.isnan(params)):
        print(f'{eq_name:<45s} | SKIPPED (fit failures)')
        continue

    # Learn each parameter from λ
    X_tr = lam_train.reshape(-1, 1)
    X_te = lam_test.reshape(-1, 1)

    pred_params = np.zeros((3, 3))  # 3 test mols × 3 params
    true_params_test = params[test_idx]

    print(f'\n  Parameter prediction for {eq_name}:')
    for p in range(3):
        m = Ridge(alpha=1e-6)
        m.fit(X_tr, params[train_idx, p])
        pred_params[:, p] = m.predict(X_te)
        pname = eq_info['param_names'][p]
        print(f'    {pname}: true={true_params_test[:, p]}, pred={pred_params[:, p]}')

    # Reconstruct E_elec
    E_pred = np.zeros((3, n_grid))
    for j in range(3):
        E_pred[j] = func(d_grid, *pred_params[j])

    mae = mean_absolute_error(E_elec[test_idx].ravel(), E_pred.ravel())
    ratio = mae / mae_direct
    per_mol = [mean_absolute_error(E_elec[test_idx[j]], E_pred[j]) for j in range(3)]

    short_name = eq_name.split(':')[0]
    label = f'Adaptive {short_name} λ→params→E(d)'
    print(f'\n{label:<45s} | {mae:7.2f} | {ratio:7.3f}x | '
          f'{per_mol[0]:6.2f} {per_mol[1]:6.2f} {per_mol[2]:6.2f}')

    all_preds[short_name] = E_pred

print(f'{"-"*85}')

# ================================================================
# FIGURE 1: Fit quality for best equation
# ================================================================
fig1, axes = plt.subplots(2, 4, figsize=(18, 9))

for i in range(7):
    ax = axes.flat[i]
    ax.plot(d_grid, E_elec[i], 'k-', lw=2.5, label='DFT')

    for eq_name, eq_info in equations.items():
        func = eq_info['func']
        params = all_params[eq_name]
        if np.any(np.isnan(params[i])):
            continue
        fit = func(d_grid, *params[i])
        mae = all_fit_mae[eq_name][i]
        short = eq_name.split(':')[0]
        ax.plot(d_grid, fit, lw=1.5, alpha=0.8, label=f'{short} ({mae:.3f})')

    color = '#d62728' if i in test_idx else '#1f77b4'
    ax.set_title(f'{mol_names[i]} (λ={i})', fontsize=11, color=color,
                 fontweight='bold' if i in test_idx else 'normal')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('d [Å]')
    ax.set_ylabel('$E_{elec}$ [Ha]')

# Parameter trends in bottom-right
ax_p = axes[1, 3]
for eq_name, eq_info in equations.items():
    params = all_params[eq_name]
    if np.any(np.isnan(params)):
        continue
    # Plot the nonlinear parameter (index 1: α, c, or n)
    short = eq_name.split(':')[0]
    ax_p.plot(range(7), params[:, 1], 'o-', ms=6, label=f'{short}: {eq_info["param_names"][1]}')

ax_p.set_xlabel('λ', fontsize=11)
ax_p.set_ylabel('Nonlinear parameter value', fontsize=11)
ax_p.set_title('Nonlinear params vs λ', fontsize=11)
ax_p.legend(fontsize=8)
ax_p.grid(True, alpha=0.3)

fig1.suptitle('Nonlinear equation fits to E_elec(d) — each has 3 parameters', fontsize=13)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'fig_nonlinear_fits.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved fig_nonlinear_fits.png')
plt.close(fig1)

# ================================================================
# FIGURE 2: Predictions for test molecules
# ================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

colors = {'Direct [λ]': 'b', 'AHA [λ,λ²]': 'g',
          'Exponential': 'r', 'KKL': 'm', 'Power': 'orange'}
styles = {'Direct [λ]': '--', 'AHA [λ,λ²]': '-',
          'Exponential': '--', 'KKL': ':', 'Power': '-.'}

for j, ax in enumerate(axes2):
    idx = test_idx[j]
    name = mol_names[idx]

    ax.plot(d_grid, E_elec[idx], 'k-', lw=2.5, label='DFT reference', zorder=5)

    for pred_label, pred in all_preds.items():
        mae_j = mean_absolute_error(E_elec[idx], pred[j])
        c = colors.get(pred_label, 'gray')
        s = styles.get(pred_label, '--')
        ax.plot(d_grid, pred[j], color=c, ls=s, lw=1.5,
                label=f'{pred_label} ({mae_j:.2f})')

    ax.set_xlabel('d [Å]', fontsize=11)
    ax.set_ylabel('$E_{elec}$ [Ha]', fontsize=11)
    ax.set_title(f'{name} (λ={idx})', fontsize=13)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

fig2.suptitle('Extrapolation: nonlinear adaptive equations vs AHA vs direct', fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_nonlinear_predictions.png'), dpi=150, bbox_inches='tight')
print('Saved fig_nonlinear_predictions.png')
plt.close(fig2)

# ================================================================
# FIGURE 3: Parameter smoothness — all 3 params for best equation
# ================================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Use KKL as the featured equation (it's from the thesis!)
eq_name = 'KKL: A·exp(-c√d)+C'
eq_info = equations[eq_name]
params = all_params[eq_name]

for p, ax in enumerate(axes3):
    pname = eq_info['param_names'][p]
    true_vals = params[:, p]

    ax.plot(range(4), true_vals[train_idx], 'bo', ms=10, zorder=5, label='Train (true)')
    ax.plot([4,5,6], true_vals[test_idx], 'rs', ms=10, zorder=5, label='Test (true)')

    # Ridge prediction
    m = Ridge(alpha=1e-6)
    m.fit(lam_train.reshape(-1, 1), true_vals[train_idx])
    lam_dense = np.linspace(-0.5, 7, 100)
    ax.plot(lam_dense, m.predict(lam_dense.reshape(-1, 1)), 'b-', alpha=0.5, label='Linear pred')

    # Label molecules
    for i in range(7):
        ax.annotate(mol_names[i], (i, true_vals[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel(pname, fontsize=12)
    ax.set_title(f'Parameter {pname}', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig3.suptitle('KKL: A·exp(-c·√d) + C — parameter trends across the series\n'
              'Parameters vary smoothly but nonlinearly with λ', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGDIR, 'fig_kkl_parameter_trends.png'), dpi=150, bbox_inches='tight')
print('Saved fig_kkl_parameter_trends.png')
plt.close(fig3)

# ================================================================
# KEY TEST: Use [λ, λ²] to predict nonlinear equation parameters
# This combines AHA's quadratic features with physics equation.
# Question: does the nonlinear equation add anything on top of λ²?
# ================================================================
print(f'\n\n{"="*85}')
print(f'KEY TEST: [λ, λ²] → nonlinear params → E(d)  vs  AHA [λ,λ²] → E(d)')
print(f'{"="*85}')
print(f'\n{"Method":<50s} | {"MAE":>7s} | {"vs AHA":>7s} | {"LiNa":>6s} {"HeMg":>6s} {"HAl":>6s}')
print(f'{"-"*85}')

# AHA reference
print(f'{"AHA multioutput [λ,λ²]→E(d)":<50s} | {mae_aha:7.2f} | {"base":>7s} | '
      f'{mean_absolute_error(E_elec[4], E_pred_aha[0]):6.2f} '
      f'{mean_absolute_error(E_elec[5], E_pred_aha[1]):6.2f} '
      f'{mean_absolute_error(E_elec[6], E_pred_aha[2]):6.2f}')

X_tr_q = np.column_stack([lam_train, lam_train**2])
X_te_q = np.column_stack([lam_test, lam_test**2])

for eq_name, eq_info in equations.items():
    func = eq_info['func']
    params = all_params[eq_name]

    if np.any(np.isnan(params)):
        continue

    # Predict params with [λ, λ²]
    pred_params_q = np.zeros((3, 3))
    for p in range(3):
        m = Ridge(alpha=1e-6)
        m.fit(X_tr_q, params[train_idx, p])
        pred_params_q[:, p] = m.predict(X_te_q)

    # Reconstruct
    E_pred_q = np.zeros((3, n_grid))
    for j in range(3):
        E_pred_q[j] = func(d_grid, *pred_params_q[j])

    mae_q = mean_absolute_error(E_elec[test_idx].ravel(), E_pred_q.ravel())
    ratio_q = mae_q / mae_aha
    per_mol_q = [mean_absolute_error(E_elec[test_idx[j]], E_pred_q[j]) for j in range(3)]

    short = eq_name.split(':')[0]
    label = f'Adaptive {short} [λ,λ²]→params→E(d)'
    print(f'{label:<50s} | {mae_q:7.2f} | {ratio_q:7.3f}x | '
          f'{per_mol_q[0]:6.2f} {per_mol_q[1]:6.2f} {per_mol_q[2]:6.2f}')

    # Also show param prediction quality with [λ,λ²]
    print(f'  Param prediction with [λ,λ²]:')
    for p in range(3):
        pname = eq_info['param_names'][p]
        true_test = params[test_idx, p]
        pred_test = pred_params_q[:, p]
        print(f'    {pname}: true={np.round(true_test, 3)}, pred={np.round(pred_test, 3)}')

print(f'{"-"*85}')
print(f'\nDirect [λ] baseline for reference: {mae_direct:.2f} Ha')
print(f'AHA [λ,λ²] baseline:               {mae_aha:.2f} Ha')

# ================================================================
# FIGURE 4: Parameter trends with [λ,λ²] prediction overlay
# ================================================================
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

eq_name = 'KKL: A·exp(-c√d)+C'
eq_info = equations[eq_name]
params = all_params[eq_name]

for p, ax in enumerate(axes4):
    pname = eq_info['param_names'][p]
    true_vals = params[:, p]

    ax.plot(range(4), true_vals[train_idx], 'bo', ms=10, zorder=5, label='Train (true)')
    ax.plot([4,5,6], true_vals[test_idx], 'rs', ms=10, zorder=5, label='Test (true)')

    lam_dense = np.linspace(-0.5, 7, 100)

    # Linear prediction [λ]
    m1 = Ridge(alpha=1e-6)
    m1.fit(lam_train.reshape(-1, 1), true_vals[train_idx])
    ax.plot(lam_dense, m1.predict(lam_dense.reshape(-1, 1)),
            'b-', alpha=0.5, lw=1.5, label='Linear [λ]')

    # Quadratic prediction [λ,λ²]
    m2 = Ridge(alpha=1e-6)
    m2.fit(X_tr_q, true_vals[train_idx])
    ax.plot(lam_dense, m2.predict(np.column_stack([lam_dense, lam_dense**2])),
            'g-', alpha=0.8, lw=2, label='Quadratic [λ,λ²]')

    for i in range(7):
        ax.annotate(mol_names[i], (i, true_vals[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel(pname, fontsize=12)
    ax.set_title(f'Parameter {pname}', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig4.suptitle('KKL parameter prediction: [λ] (blue) vs [λ,λ²] (green)\n'
              'Quadratic features capture the curvature in parameter space', fontsize=13)
fig4.tight_layout()
fig4.savefig(os.path.join(FIGDIR, 'fig_kkl_param_linear_vs_quadratic.png'), dpi=150, bbox_inches='tight')
print('Saved fig_kkl_param_linear_vs_quadratic.png')
plt.close(fig4)
