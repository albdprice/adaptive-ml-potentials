"""
Physics-Informed Ridge Regression on Softened Coulomb Data.

Strategy: E(r) = S(r) / r
  - Physics (1/r) handles the singularity
  - ML learns the smooth screening function S(r) = E(r) * r

Model 1 (Physics-Informed): Ridge on polynomial features of r -> S(r), then E = S/r
Model 2 (Naive Baseline):   Ridge on polynomial features of r -> E(r) directly

This tests single-curve fitting, NOT cross-system extrapolation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Parameters ---
Z = 14.4    # eV * Angstrom
a = 0.8     # Angstrom (softening)

# --- Data generation ---
N = 200
r = np.linspace(0.05, 6.0, N)
rho = np.sqrt(r**2 + a**2)

E_true = -Z / rho                 # ground truth energy
S_target = E_true * r             # screening function target

# Save training data
df = pd.DataFrame({'r': r, 'E_true': E_true, 'S_target': S_target})
data_path = os.path.join(os.path.dirname(__file__), 'training_data.csv')
df.to_csv(data_path, index=False)
print('Training data:')
print(df.head().to_string(index=False))
print(f'\nSaved to {data_path}')

# --- Feature matrix: polynomial features of r ---
degree = 5
X = r.reshape(-1, 1)
poly = PolynomialFeatures(degree=degree, include_bias=True)
X_poly = poly.fit_transform(X)
print(f'\nPolynomial features: degree={degree}, n_features={X_poly.shape[1]}')

# --- Model 1: Physics-Informed (learn S, reconstruct E = S/r) ---
ridge_S = Ridge(alpha=1e-6)
ridge_S.fit(X_poly, S_target)
S_pred = ridge_S.predict(X_poly)
E_phys = S_pred / r  # reconstruct energy using physics

# --- Model 2: Naive Baseline (learn E directly) ---
ridge_E = Ridge(alpha=1e-6)
ridge_E.fit(X_poly, E_true)
E_naive = ridge_E.predict(X_poly)

# --- Metrics ---
mae_S = mean_absolute_error(S_target, S_pred)
mae_E_phys = mean_absolute_error(E_true, E_phys)
mae_E_naive = mean_absolute_error(E_true, E_naive)

print(f'\n--- Results ---')
print(f'MAE on S(r):      {mae_S:.6f} eV*Å')
print(f'MAE E (physics):   {mae_E_phys:.6f} eV')
print(f'MAE E (naive):     {mae_E_naive:.6f} eV')
print(f'Ratio naive/phys:  {mae_E_naive / mae_E_phys:.1f}x')

# Near-origin performance (r < 0.5 Å)
mask_near = r < 0.5
mae_near_phys = mean_absolute_error(E_true[mask_near], E_phys[mask_near])
mae_near_naive = mean_absolute_error(E_true[mask_near], E_naive[mask_near])
print(f'\nNear-origin (r < 0.5 Å):')
print(f'MAE E (physics):   {mae_near_phys:.6f} eV')
print(f'MAE E (naive):     {mae_near_naive:.6f} eV')
print(f'Ratio naive/phys:  {mae_near_naive / mae_near_phys:.1f}x')

# ================================================================
# PLOTTING
# ================================================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1]})

# --- Top: Screening function S(r) ---
ax_top.plot(r, S_target, 'g-', lw=2.5, label=r'$S_{target}(r) = E_{true} \cdot r$')
ax_top.plot(r, S_pred, 'k--', lw=2, label=f'$S_{{pred}}$ (Ridge, deg={degree})')
ax_top.set_ylabel(r'$S(r)$ [eV$\cdot$Å]')
ax_top.set_title(f'Screening function: smooth, easy to learn  (MAE = {mae_S:.2e} eV·Å)')
ax_top.legend(fontsize=10)
ax_top.grid(True, alpha=0.3)

# --- Bottom: Energy reconstruction ---
ax_bot.plot(r, E_true, 'b-', lw=2.5, label=r'$E_{true}(r) = -Z/\sqrt{r^2+a^2}$')
ax_bot.plot(r, E_phys, 'g--', lw=2,
            label=f'Physics-informed $S_{{pred}}/r$ (MAE={mae_E_phys:.2e})')
ax_bot.plot(r, E_naive, 'r:', lw=2,
            label=f'Naive Ridge on $E$ (MAE={mae_E_naive:.2e})')

ax_bot.set_xlabel('r [Å]')
ax_bot.set_ylabel('Energy [eV]')
ax_bot.set_title('Energy reconstruction: physics-informed captures the dive at $r \\to 0$')
ax_bot.legend(fontsize=9)
ax_bot.grid(True, alpha=0.3)
ax_bot.set_ylim(-22, 2)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_screening_ridge_test.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_screening_ridge_test.png')
plt.close(fig)
