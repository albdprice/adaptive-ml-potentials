"""
Morse decomposition: V_total = E_elec + V_nn.

Z_AB is scaled per molecule so that E_elec = V_Morse - Z_AB/r < 0 everywhere.
Condition: Z_AB > max_r [r * V_Morse(r)] ensures V_nn > V_Morse at all r.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Grid
r = np.linspace(0.5, 8.0, 500)

# Example molecules with different Morse parameters
molecules = [
    {'label': 'Molecule A', 'D_e': 0.15, 'R_e': 1.8, 'alpha': 1.0},
    {'label': 'Molecule B', 'D_e': 0.25, 'R_e': 2.0, 'alpha': 1.2},
    {'label': 'Molecule C', 'D_e': 0.35, 'R_e': 2.2, 'alpha': 1.5},
    {'label': 'Molecule D', 'D_e': 0.50, 'R_e': 2.5, 'alpha': 2.0},
]

fig, axes = plt.subplots(len(molecules), 3, figsize=(14, 3.2 * len(molecules)),
                          sharex=True)

for row, mol in enumerate(molecules):
    D_e = mol['D_e']
    R_e = mol['R_e']
    alpha = mol['alpha']

    u = np.exp(-alpha * (r - R_e))
    V_total = D_e * (1 - u)**2 - D_e

    # Find Z_AB so that E_elec is monotonically increasing in [r_min, r_max]
    # Condition 1: E_elec < 0 everywhere -> Z_AB > max(r * V_Morse)
    # Condition 2: dE_elec/dr > 0 everywhere -> Z_AB > r^2 * |dV_Morse/dr|
    # Morse derivative
    V_d1 = 2 * D_e * alpha * (1 - u) * u  # dV/dr (negative for r < R_e)
    # Need Z_AB/r^2 > |dV/dr| for all r, i.e. Z_AB > max(r^2 * |dV/dr|)
    cond1 = np.max(r * np.maximum(V_total, 0))
    cond2 = np.max(r**2 * np.abs(V_d1))
    Z_AB = 1.5 * max(cond1, cond2)
    Z_AB = max(Z_AB, 2.0)
    mol['Z_AB'] = Z_AB

    V_nn = Z_AB / r
    E_elec = V_total - V_nn

    print(f'{mol["label"]}: D_e={D_e}, R_e={R_e}, alpha={alpha}, '
          f'Z_AB={Z_AB:.1f}, E_elec range=[{E_elec.min():.2f}, {E_elec.max():.2f}]')

    # Panel 1: V_total (Morse)
    ax = axes[row, 0]
    ax.plot(r, V_total, 'k-', lw=2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(-D_e, color='gray', ls=':', lw=0.5)
    ax.axvline(R_e, color='gray', ls=':', lw=0.5, alpha=0.3)
    ax.set_ylim(-0.8, 1.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(f'{mol["label"]}\nEnergy [Ha]', fontsize=9)
    if row == 0:
        ax.set_title('$V_{total} = V_{Morse}$', fontsize=11)

    # Panel 2: V_nn
    ax = axes[row, 1]
    ax.plot(r, V_nn, 'r-', lw=2)
    ax.plot(r, V_total, 'k--', lw=1, alpha=0.3, label='$V_{total}$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylim(-0.5, max(V_nn.max() * 0.3, 5))
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.95, f'$Z_{{AB}}$ = {Z_AB:.1f}',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    if row == 0:
        ax.set_title('$V_{nn} = Z_{AB} / r$', fontsize=11)

    # Panel 3: E_elec
    ax = axes[row, 2]
    ax.plot(r, E_elec, 'b-', lw=2)
    ax.axhline(0, color='gray', lw=0.5)
    idx_min = np.argmin(E_elec)
    ax.plot(r[idx_min], E_elec[idx_min], 'bo', ms=5)
    ax.set_ylim(E_elec.min() * 1.15, 0.3)
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title('$E_{elec} = V_{total} - V_{nn}$', fontsize=11)

    # Annotate parameters on left panel
    axes[row, 0].text(0.97, 0.95,
                      f'$D_e$={D_e}, $R_e$={R_e}, $\\alpha$={alpha}',
                      transform=axes[row, 0].transAxes, fontsize=8,
                      ha='right', va='top',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

for ax in axes[-1, :]:
    ax.set_xlabel('r [Bohr]')

fig.suptitle('Morse decomposition: $V_{total} = E_{elec} + V_{nn}$', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_morse_decomposition_simple.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_morse_decomposition_simple.png')
plt.close(fig)
