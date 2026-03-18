"""
Anatole's proposal: Morse as total energy, subtract V_nn, fit remainder with parabola.

V_total(r)  = V_Morse(r)                    (synthetic "total energy")
V_nn(r)     = Z_eff / r                     (known nuclear repulsion)
V_elec(r)   = V_Morse(r) - Z_eff / r        (electronic remainder)

Then decompose V_elec using b=0 parabola:
  V_elec(r) = a(r) * (r - r_e_elec)^2

where r_e_elec is the equilibrium of V_elec, and a(r) = V_elec(r) / (r - r_e_elec)^2.

Question: is a(r) smooth and learnable?

Test with several Z_eff values to see sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Morse parameters ---
D_e = 1.0
alpha = 1.2
r_e = 2.5

r = np.linspace(0.5, 10.0, 2000)

# --- Morse potential ---
u = np.exp(-alpha * (r - r_e))
V_morse = D_e * (1 - u)**2 - D_e

# --- Test several Z_eff values ---
Z_eff_values = [0.5, 2.0, 5.0, 10.0]

fig, axes = plt.subplots(3, len(Z_eff_values), figsize=(16, 11),
                          sharex=True)

for col, Z_eff in enumerate(Z_eff_values):
    V_nn = Z_eff / r
    V_elec = V_morse - V_nn

    # Find equilibrium of V_elec
    idx_min = np.argmin(V_elec)
    r_e_elec = r[idx_min]
    V_e_min = V_elec[idx_min]

    # b=0 decomposition: a(r) = V_elec / (r - r_e_elec)^2
    dr = r - r_e_elec
    # Mask near equilibrium to avoid 0/0
    safe = np.abs(dr) > 0.05
    a_r = np.full_like(r, np.nan)
    a_r[safe] = V_elec[safe] / dr[safe]**2

    # Also compute with b: V_elec = a(r)(r-r_e)^2 + b
    # Use L'Hopital at r_e: a(r_e) = V_elec''(r_e)/2
    # V_elec'' = V_morse'' - V_nn'' = V_morse'' - 2*Z_eff/r^3
    V_morse_d2 = 2 * D_e * alpha**2 * u * (2*u - 1)
    V_nn_d2 = 2 * Z_eff / r**3
    V_elec_d2 = V_morse_d2 - V_nn_d2
    a_at_eq = V_elec_d2[idx_min] / 2.0

    # sqrt(a) for the smooth version
    sqrt_a = np.full_like(r, np.nan)
    sqrt_a[safe & (a_r > 0)] = np.sqrt(a_r[safe & (a_r > 0)])
    sqrt_a[safe & (a_r < 0)] = -np.sqrt(np.abs(a_r[safe & (a_r < 0)]))

    print(f'\nZ_eff = {Z_eff}:')
    print(f'  V_elec equilibrium: r = {r_e_elec:.3f} Å, V = {V_e_min:.4f} eV')
    print(f'  a(r_e) = V_elec"(r_e)/2 = {a_at_eq:.4f} eV/Å²')
    print(f'  V_elec(r=0.5) = {V_elec[0]:.2f} eV')
    print(f'  V_elec(r=10)  = {V_elec[-1]:.4f} eV')

    # Check if V_elec has inflection point
    sign_changes = np.where(np.diff(np.sign(V_elec_d2)))[0]
    if len(sign_changes) > 0:
        r_infl = r[sign_changes[0]]
        print(f'  Inflection point at r = {r_infl:.3f} Å')
    else:
        r_infl = None
        print(f'  No inflection point in range')

    # --- Row 1: Energy curves ---
    ax = axes[0, col]
    ax.plot(r, V_morse, 'k-', lw=1.5, label='$V_{Morse}$')
    ax.plot(r, V_nn, 'r--', lw=1, alpha=0.6, label=f'$V_{{nn}} = {Z_eff}/r$')
    ax.plot(r, V_elec, 'b-', lw=2, label='$V_{elec}$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(r_e_elec, color='blue', ls=':', alpha=0.3)
    if r_infl is not None:
        ax.axvline(r_infl, color='orange', ls='--', alpha=0.5, label=f'Inflection')
    ax.set_ylim(-5, 10)
    ax.set_title(f'$Z_{{eff}} = {Z_eff}$')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('Energy [eV]')
        ax.legend(fontsize=7, loc='upper right')

    # --- Row 2: a(r) ---
    ax = axes[1, col]
    ax.plot(r[safe], a_r[safe], 'b-', lw=2, label='$a(r)$')
    ax.axhline(a_at_eq, color='gray', ls=':', lw=1, alpha=0.5,
               label=f'$a(r_e) = {a_at_eq:.3f}$')
    ax.axhline(0, color='gray', lw=0.5)
    if r_infl is not None:
        ax.axvline(r_infl, color='orange', ls='--', alpha=0.5)
    ax.set_ylim(-2, 3)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('$a(r)$ [eV/Å²]')
        ax.legend(fontsize=7)

    # --- Row 3: sqrt(|a(r)|) * sign ---
    ax = axes[2, col]
    ax.plot(r[safe], sqrt_a[safe], 'g-', lw=2, label=r'$\mathrm{sign}(a)\sqrt{|a|}$')
    ax.axhline(0, color='gray', lw=0.5)
    if r_infl is not None:
        ax.axvline(r_infl, color='orange', ls='--', alpha=0.5)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('r [Å]')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel(r'$\sqrt{|a(r)|}$ [eV$^{1/2}$/Å]')
        ax.legend(fontsize=7)

fig.suptitle(r'$V_{elec} = V_{Morse} - Z_{eff}/r$,  then $V_{elec} = a(r)(r - r_{e,elec})^2$',
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_morse_nuc_sub_parabola.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_morse_nuc_sub_parabola.png')
plt.close(fig)
