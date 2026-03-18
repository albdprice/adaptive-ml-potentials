"""
Softened Coulomb adaptive parabola test.

Physics model:
  E_elec(r) = -Z / sqrt(r^2 + a^2)     (softened Coulomb, finite at r=0)
  V_NN(r)   = A / r                     (nuclear repulsion)
  E_total(r) = E_elec(r) + V_NN(r)

Adaptive parabola decomposition of E_elec:
  E_model(r) = k(r) * (r - r_0(r))^2 + b(r)

where:
  k(r)   = E_elec''(r) / 2
  r_0(r) = r - E_elec'(r) / E_elec''(r)
  b(r)   = E_elec(r) - k(r) * (r - r_0(r))^2

Goal: demonstrate that r_0(r) diverges at the inflection point where E_elec'' = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Parameters ---
Z = 14.4    # eV * Angstrom
a = 0.8     # Angstrom (softening parameter)
A = 14.4    # eV * Angstrom (nuclear repulsion strength)

r = np.linspace(0.01, 6.0, 2000)

# --- Electronic energy: softened Coulomb ---
rho2 = r**2 + a**2          # r^2 + a^2
rho = np.sqrt(rho2)         # sqrt(r^2 + a^2)

E_elec = -Z / rho

# Analytical first derivative
# d/dr [-Z / sqrt(r^2+a^2)] = Z*r / (r^2+a^2)^(3/2)
E_elec_d1 = Z * r / rho2**(1.5)

# Analytical second derivative
# d^2/dr^2 = Z * (a^2 - 2r^2) / (r^2+a^2)^(5/2)
E_elec_d2 = Z * (a**2 - 2 * r**2) / rho2**(2.5)

# --- Nuclear repulsion ---
V_nn = A / r

# --- Total energy ---
E_total = E_elec + V_nn

# --- Inflection point: E_elec'' = 0 when a^2 - 2r^2 = 0, i.e. r = a/sqrt(2) ---
r_inflection = a / np.sqrt(2)
print(f'Inflection point (analytical): r = a/sqrt(2) = {r_inflection:.4f} Å')
print(f'E_elec at inflection: {-Z / np.sqrt(r_inflection**2 + a**2):.4f} eV')
print(f'E_elec at r=0: {-Z / a:.4f} eV')

# --- Adaptive parabola parameters ---
k = E_elec_d2 / 2.0
r_0 = r - E_elec_d1 / E_elec_d2    # diverges at inflection
b_param = E_elec - k * (r - r_0)**2

# ================================================================
# PLOTTING
# ================================================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1]})

# --- Top plot: Energy curves ---
ax_top.plot(r, E_total, 'k-', lw=2, label=r'$E_{total} = E_{elec} + V_{NN}$')
ax_top.plot(r, V_nn, 'r--', lw=1.5, label=r'$V_{NN} = A/r$')
ax_top.plot(r, E_elec, 'b-', lw=2, label=r'$E_{elec} = -Z/\sqrt{r^2 + a^2}$')
ax_top.axhline(0, color='gray', lw=0.5)
ax_top.axvline(r_inflection, color='orange', ls=':', lw=1.5, alpha=0.7,
               label=f'Inflection $r = a/\\sqrt{{2}}$ = {r_inflection:.3f} Å')

ax_top.set_ylabel('Energy [eV]')
ax_top.set_title(f'Softened Coulomb model  ($Z = {Z}$, $a = {a}$, $A = {A}$)')
ax_top.legend(fontsize=9, loc='right')
ax_top.grid(True, alpha=0.3)
ax_top.set_ylim(-20, 30)

# --- Bottom plot: r_0(r) showing the divergence ---
# Mask extreme values for clean plotting
clip = 20
r_0_clipped = np.copy(r_0)
r_0_clipped[r_0_clipped > clip] = np.nan
r_0_clipped[r_0_clipped < -clip] = np.nan

ax_bot.plot(r, r_0_clipped, 'b-', lw=2, label=r'$r_0(r) = r - E^\prime / E^{\prime\prime}$')
ax_bot.axvline(r_inflection, color='orange', ls=':', lw=1.5, alpha=0.7,
               label=f'Inflection ($E^{{\prime\prime}} = 0$)')
ax_bot.axhline(0, color='gray', lw=0.5)

# Also plot r itself for reference
ax_bot.plot(r, r, 'k--', lw=1, alpha=0.3, label='$r_0 = r$ (identity)')

ax_bot.set_ylim(-clip, clip)
ax_bot.set_xlabel('r [Å]')
ax_bot.set_ylabel(r'$r_0(r)$ [Å]')
ax_bot.set_title(r'Parabola center $r_0(r)$: diverges at inflection point where $E_{elec}^{\prime\prime} = 0$')
ax_bot.legend(fontsize=9)
ax_bot.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_softened_coulomb_parabola.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_softened_coulomb_parabola.png')
plt.close(fig)

# --- Print diagnostics ---
print(f'\n--- Diagnostics ---')
print(f'E_elec(r=0.01) = {E_elec[0]:.4f} eV  (united atom limit → {-Z/a:.2f} eV)')
print(f'E_elec(r=6.0)  = {E_elec[-1]:.4f} eV  (long range → 0)')
print(f'E_elec is monotonically increasing: {np.all(np.diff(E_elec) > 0)}')
print(f'\nE_total minimum: {E_total.min():.4f} eV at r = {r[np.argmin(E_total)]:.3f} Å')
