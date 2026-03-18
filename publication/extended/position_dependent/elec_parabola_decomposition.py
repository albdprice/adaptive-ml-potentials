"""
Parabola decomposition of E_elec with two reference point choices.

E_elec(r) = a(r) * (r - r_0)^2 + b

where b = E_elec(r_0), so:
  Delta_E(r) = E_elec(r) - E_elec(r_0) = a(r) * (r - r_0)^2
  a(r) = [E_elec(r) - E_elec(r_0)] / (r - r_0)^2

Two choices for r_0:
  A) r_0 = R_e (equilibrium of V_total, where dE_elec/dr = -dV_nn/dr)
  B) r_0 = 10 Bohr (far out, avoids singularity entirely in grid)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

r = np.linspace(1.0, 8.0, 500)

molecules = [
    {'d1': 1, 'd2': 1.0, 'label': 'Z=5 (B-B, train)'},
    {'d1': 3, 'd2': 1.0, 'label': 'Z=9 (F-F, train)'},
    {'d1': 5, 'd2': 1.0, 'label': 'Z=13 (Al-Al, train)'},
    {'d1': 9, 'd2': 1.0, 'label': 'Z=21 (Sc-Sc, test)'},
]

r0_far = 10.0  # far reference point (option B)

fig, axes = plt.subplots(len(molecules), 4, figsize=(18, 3 * len(molecules)),
                          sharex=True)

for row, mol in enumerate(molecules):
    d1, d2 = mol['d1'], mol['d2']
    Z = 3 + 2 * d1
    Z_AB = Z**2
    R_e = 2.0 + 0.2 * d2
    D_e = 0.1 + 0.05 * d1 + 0.03 * d2
    alpha = 0.8 + 0.1 * d2

    u = np.exp(-alpha * (r - R_e))
    V_total = D_e * (1 - u)**2 - D_e
    V_nn = Z_AB / r
    E_elec = V_total - V_nn

    # E_elec at reference points
    u_Re = np.exp(-alpha * (R_e - R_e))  # = 1
    E_elec_Re = (D_e * (1 - u_Re)**2 - D_e) - Z_AB / R_e  # = -D_e - Z^2/R_e

    u_far = np.exp(-alpha * (r0_far - R_e))
    E_elec_far = (D_e * (1 - u_far)**2 - D_e) - Z_AB / r0_far

    # Option A: r_0 = R_e
    dr_A = r - R_e
    safe_A = np.abs(dr_A) > 0.05
    Delta_E_A = E_elec - E_elec_Re
    a_A = np.full_like(r, np.nan)
    a_A[safe_A] = Delta_E_A[safe_A] / dr_A[safe_A]**2
    # At R_e: a = E_elec''(R_e)/2 by L'Hopital
    V_d2_Re = 2 * D_e * alpha**2  # V_Morse''(R_e) = 2*D_e*alpha^2
    Vnn_d2_Re = 2 * Z_AB / R_e**3
    a_at_Re = (V_d2_Re - Vnn_d2_Re) / 2.0
    a_A[~safe_A] = a_at_Re

    # Option B: r_0 = 10 (far out)
    dr_B = r - r0_far
    Delta_E_B = E_elec - E_elec_far
    a_B = Delta_E_B / dr_B**2  # no singularity since r0_far outside grid

    print(f'{mol["label"]}: E_elec(R_e)={E_elec_Re:.2f}, '
          f'a(R_e)={a_at_Re:.4f}, '
          f'a_B range=[{a_B.min():.4f}, {a_B.max():.4f}]')

    # --- Panel 1: E_elec ---
    ax = axes[row, 0]
    ax.plot(r, E_elec, 'b-', lw=2)
    ax.axhline(E_elec_Re, color='green', ls=':', lw=1, alpha=0.5)
    ax.axvline(R_e, color='green', ls=':', lw=1, alpha=0.5)
    ax.plot(R_e, E_elec_Re, 'go', ms=6)
    ax.set_ylabel(f'{mol["label"]}\n[Ha]', fontsize=8)
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title('$E_{elec}(r)$')

    # --- Panel 2: Delta_E (shifted) ---
    ax = axes[row, 1]
    ax.plot(r, Delta_E_A, 'g-', lw=2, label=f'$r_0 = R_e = {R_e:.1f}$')
    ax.plot(r, Delta_E_B, 'r--', lw=2, label=f'$r_0 = {r0_far:.0f}$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title('$\\Delta E = E_{elec} - E_{elec}(r_0)$')
        ax.legend(fontsize=7)

    # --- Panel 3: a(r) with r_0 = R_e ---
    ax = axes[row, 2]
    ax.plot(r[safe_A], a_A[safe_A], 'g-', lw=2)
    ax.axhline(a_at_Re, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axvline(R_e, color='green', ls=':', lw=1, alpha=0.3)
    ax.set_ylim(min(a_A[safe_A].min() * 1.2, -5), max(a_A[safe_A].max() * 1.2, 5))
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title('$a(r)$, $r_0 = R_e$')

    # --- Panel 4: a(r) with r_0 = 10 ---
    ax = axes[row, 3]
    ax.plot(r, a_B, 'r-', lw=2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title(f'$a(r)$, $r_0 = {r0_far:.0f}$ Bohr')

for ax in axes[-1, :]:
    ax.set_xlabel('r [Bohr]')

fig.suptitle('E$_{elec}$ parabola decomposition: $\\Delta E = a(r) \\cdot (r - r_0)^2$',
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_elec_parabola_decomposition.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_elec_parabola_decomposition.png')
plt.close(fig)
