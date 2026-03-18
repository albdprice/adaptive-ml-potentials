"""
Adaptive Monomial Basis Selection Test.

Ground truth: E_elec(r) = -Z / sqrt(r^2 + a^2)   (softened Coulomb)

We approximate E_elec(r) = C(r) * r^n and ask: for which n is C(r)
smoothest and most learnable?

Case A (n=2):  C_2(r)   = E_elec(r) / r^2      -- diverges at r->0
Case B (n=0):  C_0(r)   = E_elec(r)             -- trivial (just the energy)
Case C (n=-1): C_-1(r)  = E_elec(r) * r         -- "effective charge", smooth

Also includes E_total = E_elec + V_NN to show binding curve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Parameters ---
Z = 14.4    # eV * Angstrom (electronic attraction strength)
a = 0.8     # Angstrom (softening parameter)
A = 5.0     # eV * Angstrom (nuclear repulsion, A < Z to get binding)

r = np.linspace(0.05, 6.0, 2000)

# --- Ground truth: softened Coulomb ---
rho = np.sqrt(r**2 + a**2)
E_elec = -Z / rho

# --- Nuclear repulsion ---
V_nn = A / r

# --- Total energy (for reference) ---
E_total = E_elec + V_nn

# Find equilibrium
idx_eq = np.argmin(E_total)
r_eq = r[idx_eq]
E_eq = E_total[idx_eq]

# --- Monomial coefficients ---
C_2 = E_elec / r**2          # n=2: diverges at r->0
C_0 = E_elec                  # n=0: trivial
C_minus1 = E_elec * r         # n=-1: effective charge Z_eff(r)

# --- DataFrame ---
df = pd.DataFrame({
    'r': r,
    'E_elec': E_elec,
    'C_2': C_2,
    'C_0': C_0,
    'C_minus1': C_minus1,
})

print('Training data:')
print(df.head(10).to_string(index=False))
print(f'\nE_total equilibrium: r = {r_eq:.3f} Å, E = {E_eq:.4f} eV')
print(f'E_elec(r=0.05) = {E_elec[0]:.4f} eV')
print(f'E_elec(r=6.0)  = {E_elec[-1]:.4f} eV')
print(f'\nC_2(r=0.05)    = {C_2[0]:.1f} eV/Å²  (diverges!)')
print(f'C_-1(r=0.05)   = {C_minus1[0]:.4f} eV·Å  (bounded)')
print(f'C_-1(r=6.0)    = {C_minus1[-1]:.4f} eV·Å')
print(f'\nC_-1 range: [{C_minus1.min():.4f}, {C_minus1.max():.4f}] eV·Å')
print(f'C_2 range:  [{C_2.min():.1f}, {C_2.max():.4f}] eV/Å²')

# ================================================================
# PLOTTING
# ================================================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1]})

# --- Top: E_elec(r) and E_total(r) ---
ax_top.plot(r, E_elec, 'b-', lw=2.5, label=r'$E_{elec}(r) = -Z/\sqrt{r^2 + a^2}$')
ax_top.plot(r, E_total, 'k-', lw=1.5, alpha=0.7,
            label=r'$E_{total} = E_{elec} + A/r$')
ax_top.plot(r, V_nn, 'r--', lw=1, alpha=0.5, label=r'$V_{NN} = A/r$')
ax_top.axhline(0, color='gray', lw=0.5)
ax_top.plot(r_eq, E_eq, 'ko', ms=6)
ax_top.annotate(f'  $r_{{eq}}$ = {r_eq:.2f} Å\n  $E_{{eq}}$ = {E_eq:.2f} eV',
                xy=(r_eq, E_eq), fontsize=9)

ax_top.set_ylabel('Energy [eV]')
ax_top.set_title(f'Softened Coulomb  ($Z = {Z}$, $a = {a}$, $A = {A}$)')
ax_top.legend(fontsize=9, loc='upper right')
ax_top.grid(True, alpha=0.3)
ax_top.set_ylim(-20, 15)

# --- Bottom: Coefficient functions C_n(r) ---
ax_bot.plot(r, C_minus1, 'g-', lw=2.5,
            label=r'$n = -1$: $C_{-1}(r) = E_{elec} \cdot r$ (effective charge)')
ax_bot.plot(r, C_0, 'b--', lw=1.5, alpha=0.7,
            label=r'$n = 0$: $C_0(r) = E_{elec}$ (trivial)')
ax_bot.plot(r, C_2, 'r--', lw=2,
            label=r'$n = 2$: $C_2(r) = E_{elec}/r^2$ (diverges)')

ax_bot.axhline(0, color='gray', lw=0.5)
ax_bot.set_ylim(-20, 5)
ax_bot.set_xlabel('r [Å]')
ax_bot.set_ylabel(r'$C_n(r)$')
ax_bot.set_title(r'Adaptive monomial: $E_{elec}(r) = C_n(r) \cdot r^n$  —  which $C_n$ is smoothest?')
ax_bot.legend(fontsize=9, loc='lower right')
ax_bot.grid(True, alpha=0.3)

# Add annotation about smoothness
ax_bot.annotate(r'$C_{-1}$ bounded: smooth "effective charge"',
                xy=(3.0, C_minus1[np.argmin(np.abs(r - 3.0))]),
                xytext=(3.5, -2), fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax_bot.annotate(r'$C_2 \to -\infty$ at $r \to 0$',
                xy=(0.3, -18), xytext=(1.5, -17), fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_adaptive_monomial_test.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_adaptive_monomial_test.png')
plt.close(fig)

# ================================================================
# Smoothness metrics: derivative magnitude as proxy for learnability
# ================================================================
print('\n--- Smoothness metrics (|dC/dr| averaged) ---')
for name, C in [('C_2', C_2), ('C_0', C_0), ('C_-1', C_minus1)]:
    dC = np.abs(np.gradient(C, r))
    # Exclude first few points where C_2 diverges
    mask = r > 0.5
    print(f'  {name:5s}: mean |dC/dr| (r>0.5) = {np.mean(dC[mask]):.4f}, '
          f'max |dC/dr| = {np.max(dC[mask]):.4f}, '
          f'range = [{np.min(C[mask]):.4f}, {np.max(C[mask]):.4f}]')
