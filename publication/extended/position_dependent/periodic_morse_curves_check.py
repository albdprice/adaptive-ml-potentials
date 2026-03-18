"""
Verify periodic Morse decomposition for individual molecules.
Show V_total, V_nn, E_elec, and the reconstruction V_total = E_elec + V_nn.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

r = np.linspace(1.0, 8.0, 500)

# Select molecules spanning the training and test ranges
molecules = [
    # Training range
    {'d1': 1, 'd2': 1.0, 'label': 'Training: Z=5 (B-B like)'},
    {'d1': 3, 'd2': 1.0, 'label': 'Training: Z=9 (F-F like)'},
    {'d1': 5, 'd2': 1.0, 'label': 'Training: Z=13 (Al-Al like)'},
    # Test range (extrapolation)
    {'d1': 9, 'd2': 1.0, 'label': 'Test: Z=21 (Sc-Sc like)'},
    {'d1': 12, 'd2': 1.0, 'label': 'Test: Z=27 (Co-Co like)'},
]

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
    V_recon = E_elec + V_nn  # should match V_total exactly

    # Panel 1: V_total
    ax = axes[row, 0]
    ax.plot(r, V_total, 'k-', lw=2)
    ax.axhline(-D_e, color='gray', ls=':', lw=0.5)
    ax.axvline(R_e, color='gray', ls=':', alpha=0.3)
    ax.set_ylim(-1, 3)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(f'{mol["label"]}\n[Ha]', fontsize=8)
    if row == 0:
        ax.set_title('$V_{total}$ (Morse)')

    # Panel 2: V_nn
    ax = axes[row, 1]
    ax.plot(r, V_nn, 'r-', lw=2)
    ax.set_ylim(0, min(V_nn.max() * 0.4, 200))
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title(f'$V_{{nn}} = Z^2/r$')
    ax.text(0.95, 0.95, f'Z={Z}, $Z^2$={Z_AB}',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Panel 3: E_elec
    ax = axes[row, 2]
    ax.plot(r, E_elec, 'b-', lw=2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylim(E_elec.min() * 1.1, max(E_elec.max() * 1.1, 1))
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title('$E_{elec} = V_{total} - V_{nn}$')

    # Panel 4: Overlay — verify reconstruction
    ax = axes[row, 3]
    ax.plot(r, V_total, 'k-', lw=2.5, label='$V_{total}$')
    ax.plot(r, V_nn, 'r--', lw=1.5, alpha=0.6, label='$V_{nn}$')
    ax.plot(r, E_elec, 'b--', lw=1.5, alpha=0.6, label='$E_{elec}$')
    ax.axhline(0, color='gray', lw=0.5)
    y_lo = min(E_elec.min(), -1)
    y_hi = max(V_nn[len(r)//4], 5)  # show some of V_nn
    ax.set_ylim(max(y_lo, -50), min(y_hi, 50))
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.set_title('All three overlaid')
        ax.legend(fontsize=7, loc='upper right')

    # Print info
    print(f'{mol["label"]}: D_e={D_e:.3f}, R_e={R_e:.1f}, alpha={alpha:.2f}, '
          f'Z_AB={Z_AB}, E_elec range=[{E_elec.min():.1f}, {E_elec.max():.3f}], '
          f'monotonic={np.all(np.diff(E_elec) > 0)}')

for ax in axes[-1, :]:
    ax.set_xlabel('r [Bohr]')

fig.suptitle('Periodic Morse decomposition: individual molecules', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_periodic_morse_check.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_periodic_morse_check.png')
plt.close(fig)
