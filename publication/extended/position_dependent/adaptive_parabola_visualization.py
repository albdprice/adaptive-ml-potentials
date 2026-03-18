"""
Visualize the "learnability" of an adaptive parabolic model for a binding curve.

Ground truth:
  V_Morse(r) = D_e * (1 - exp(-a*(r - r_e)))^2 - D_e
  V_NN(r) = A / r
  E_elec(r) = V_Morse(r) - V_NN(r)

Adaptive parabola decomposition via local Taylor expansion:
  E_model(r) = k(r) * (r - r_0(r))^2 + b(r)

where at each point r:
  k(r)   = E_elec''(r) / 2           (half-curvature)
  r_0(r) = r - E_elec'(r) / E_elec''(r)  (vertex position)
  b(r)   = E_elec(r) - k(r) * (r - r_0(r))^2  (vertex value)

Goal: inspect whether k(r), r_0(r), b(r) are smooth functions a model could learn.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Parameters ---
D_e = 0.2
r_e = 2.0
a = 1.5
A = 0.1
r_min, r_max = 0.5, 6.0

# --- Dense grid ---
r = np.linspace(r_min, r_max, 1000)

# --- Morse potential (shifted: V(r_e) = -D_e, V(inf) = 0) ---
exp_term = np.exp(-a * (r - r_e))
V_morse = D_e * (1 - exp_term)**2 - D_e

# Analytical derivatives of V_Morse
V_morse_d1 = 2 * D_e * a * (1 - exp_term) * exp_term
V_morse_d2 = 2 * D_e * a**2 * exp_term * (2 * exp_term - 1)

# --- Nuclear repulsion ---
V_nn = A / r

# Analytical derivatives of V_NN
V_nn_d1 = -A / r**2
V_nn_d2 = 2 * A / r**3

# --- Electronic energy = V_Morse - V_NN ---
E_elec = V_morse - V_nn
E_elec_d1 = V_morse_d1 - V_nn_d1  # = V_morse' + A/r^2
E_elec_d2 = V_morse_d2 - V_nn_d2  # = V_morse'' - 2A/r^3

# --- Adaptive parabola parameters ---
k = E_elec_d2 / 2.0
r_0 = r - E_elec_d1 / E_elec_d2  # diverges where E_elec'' = 0
b = E_elec - k * (r - r_0)**2

# --- Find inflection point(s) where E_elec'' = 0 ---
sign_changes = np.where(np.diff(np.sign(E_elec_d2)))[0]
r_inflection = []
for idx in sign_changes:
    # Linear interpolation for zero crossing
    r_infl = r[idx] - E_elec_d2[idx] * (r[idx+1] - r[idx]) / (E_elec_d2[idx+1] - E_elec_d2[idx])
    r_inflection.append(r_infl)
    print(f'Inflection point at r = {r_infl:.4f} Å')

# --- Find equilibrium of E_elec ---
idx_min = np.argmin(E_elec)
r_eq = r[idx_min]
E_eq = E_elec[idx_min]
print(f'E_elec equilibrium at r = {r_eq:.4f} Å, E = {E_eq:.4f} eV')

# ================================================================
# PLOTTING
# ================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1.2]})

# --- Top: Energy curves ---
ax1.plot(r, V_morse, 'k--', lw=1.5, label=r'$V_{Morse}(r)$')
ax1.plot(r, V_nn, 'r--', lw=1.5, label=r'$V_{NN}(r) = A/r$')
ax1.plot(r, E_elec, 'b-', lw=2, label=r'$E_{elec} = V_{Morse} - V_{NN}$')
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(r_eq, color='blue', ls=':', alpha=0.4)

# Mark inflection
for ri in r_inflection:
    ax1.axvline(ri, color='orange', ls='--', alpha=0.7,
                label=f'Inflection r={ri:.2f}')

ax1.set_ylabel('Energy [eV]')
ax1.set_title(f'Energy curves ($D_e$={D_e}, $r_e$={r_e}, $a$={a}, $A$={A})')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.4, 0.5)

# --- Bottom: Adaptive parameters k(r), r_0(r), b(r) ---
# These diverge near inflection points, so mask the divergent region
safe = np.abs(E_elec_d2) > 0.005  # mask where E'' is near zero

# k(r) on primary axis
color_k = '#2ca02c'
color_r0 = '#d62728'
color_b = '#1f77b4'

ax2.plot(r[safe], k[safe], color=color_k, lw=2, label=r'$k(r) = E_{elec}^{\prime\prime}/2$')
ax2.set_ylabel(r'$k(r)$ [eV/Å$^2$] and $b(r)$ [eV]')
ax2.set_xlabel('r [Å]')

# b(r) on primary axis (similar scale to k)
ax2.plot(r[safe], b[safe], color=color_b, lw=2, ls='--',
         label=r'$b(r)$ [eV]')

# r_0(r) on secondary axis (different scale — it's in Angstrom)
ax2b = ax2.twinx()
ax2b.plot(r[safe], r_0[safe], color=color_r0, lw=2, ls='-.',
          label=r'$r_0(r)$ [Å]')
ax2b.set_ylabel(r'$r_0(r)$ [Å]', color=color_r0)
ax2b.tick_params(axis='y', labelcolor=color_r0)

# Mark inflection
for ri in r_inflection:
    ax2.axvline(ri, color='orange', ls='--', alpha=0.7,
                label=f'Inflection: params diverge here')

ax2.axhline(0, color='gray', lw=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_title(r'Adaptive parabola parameters: $E_{model}(r) = k(r) \cdot (r - r_0(r))^2 + b(r)$')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_adaptive_parabola_visualization.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_adaptive_parabola_visualization.png')
plt.close(fig)

# ================================================================
# Print diagnostics
# ================================================================
print(f'\n--- Parameter ranges (excluding inflection region) ---')
print(f'  k(r):   [{np.nanmin(k[safe]):.4f}, {np.nanmax(k[safe]):.4f}] eV/Å²')
print(f'  r_0(r): [{np.nanmin(r_0[safe]):.4f}, {np.nanmax(r_0[safe]):.4f}] Å')
print(f'  b(r):   [{np.nanmin(b[safe]):.4f}, {np.nanmax(b[safe]):.4f}] eV')
print(f'\nAt equilibrium r={r_eq:.3f}:')
print(f'  k = {k[idx_min]:.4f}, r_0 = {r_0[idx_min]:.4f}, b = {b[idx_min]:.4f}')
print(f'  E_elec = {E_elec[idx_min]:.4f}, E_model = {(k[idx_min]*(r_eq - r_0[idx_min])**2 + b[idx_min]):.4f}')
