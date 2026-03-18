"""
Compare three local decompositions of a Morse curve.

Given V_Morse(r) = D_e * (1 - exp(-alpha*(r-r_e)))^2 - D_e, we decompose
it at each point r into:

1. Standard parabola:  V(r) = a(r)*r^2 + b(r)*r + c(r)
   Using local Taylor matching (value + 1st + 2nd derivative):
     a(r) = V''(r)/2
     b(r) = V'(r) - V''(r)*r
     c(r) = V(r) - V'(r)*r + V''(r)*r^2/2

2. Vertex parabola:  V(r) = k(r)*(r - r_0(r))^2 + b_v(r)
   Using local Taylor matching:
     k(r)   = V''(r)/2
     r_0(r) = r - V'(r)/V''(r)     [diverges at inflection!]
     b_v(r) = V(r) - k(r)*(r - r_0(r))^2

3. Monomial:  V(r) = C_n(r) * r^n
   Just value matching:
     C_n(r) = V(r) / r^n

Goal: visualize which parameter functions are smoothest across r.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Morse parameters ---
D_e = 1.0       # eV
alpha = 1.2     # 1/Angstrom
r_e = 2.5       # Angstrom

r = np.linspace(0.5, 8.0, 2000)

# --- Morse potential and derivatives ---
u = np.exp(-alpha * (r - r_e))
V = D_e * (1 - u)**2 - D_e                         # V(r_e) = -D_e, V(inf) = 0
V_d1 = 2 * D_e * alpha * (1 - u) * u               # dV/dr
V_d2 = 2 * D_e * alpha**2 * u * (2*u - 1)          # d2V/dr2

# --- Inflection point: V'' = 0 when 2u - 1 = 0, i.e. u = 1/2 ---
# exp(-alpha*(r-r_e)) = 1/2 => r_infl = r_e + ln(2)/alpha
r_infl = r_e + np.log(2) / alpha
print(f'Morse parameters: D_e={D_e}, alpha={alpha}, r_e={r_e}')
print(f'Inflection point: r = {r_infl:.4f} Å')
print(f'V(r_infl) = {D_e * (1 - 0.5)**2 - D_e:.4f} eV')

# ================================================================
# Decomposition 1: Standard parabola  V = a*r^2 + b*r + c
# ================================================================
a_std = V_d2 / 2.0
b_std = V_d1 - V_d2 * r
c_std = V - V_d1 * r + V_d2 * r**2 / 2.0

# Verify reconstruction
V_recon_std = a_std * r**2 + b_std * r + c_std
err_std = np.max(np.abs(V_recon_std - V))
print(f'\nStandard parabola reconstruction error: {err_std:.2e}')

# ================================================================
# Decomposition 2: Vertex parabola  V = k*(r - r_0)^2 + b_v
# ================================================================
k_vtx = V_d2 / 2.0
# r_0 diverges where V_d2 = 0 (inflection point)
with np.errstate(divide='ignore', invalid='ignore'):
    r_0_vtx = r - V_d1 / V_d2
b_vtx = V - k_vtx * (r - r_0_vtx)**2

# ================================================================
# Decomposition 3: Monomials V = C_n * r^n
# ================================================================
C_2 = V / r**2
C_1 = V / r
C_0 = V          # trivial
C_m1 = V * r     # n = -1

# ================================================================
# FIGURE 1: The three decompositions side by side
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Panel A: Morse curve with inflection ---
ax = axes[0, 0]
ax.plot(r, V, 'b-', lw=2.5, label=r'$V_{Morse}(r)$')
ax.axhline(0, color='gray', lw=0.5)
ax.axhline(-D_e, color='gray', ls=':', lw=0.5)
ax.axvline(r_infl, color='orange', ls='--', lw=1.5, alpha=0.7,
           label=f'Inflection $r = {r_infl:.2f}$ Å')
ax.axvline(r_e, color='green', ls=':', lw=1, alpha=0.5,
           label=f'Equilibrium $r_e = {r_e}$ Å')
ax.set_ylabel('Energy [eV]')
ax.set_title(f'(A)  Morse: $D_e={D_e}$, $\\alpha={alpha}$, $r_e={r_e}$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.2, 2.0)

# --- Panel B: Standard parabola parameters a(r), b(r), c(r) ---
ax = axes[0, 1]
ax.plot(r, a_std, 'r-', lw=2, label=r'$a(r) = V^{\prime\prime}/2$')
ax.plot(r, b_std, 'g-', lw=2, label=r'$b(r) = V^\prime - V^{\prime\prime} r$')
ax.plot(r, c_std, 'b-', lw=2, label=r'$c(r) = V - V^\prime r + V^{\prime\prime} r^2/2$')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(r_infl, color='orange', ls='--', lw=1.5, alpha=0.4)
ax.set_ylabel('Parameter value')
ax.set_title(r'(B)  Standard: $V = a(r) r^2 + b(r) r + c(r)$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-10, 5)

# --- Panel C: Vertex parabola parameters ---
ax = axes[1, 0]
# Mask near inflection for r_0
clip = 15
r_0_plot = np.copy(r_0_vtx)
r_0_plot[np.abs(r_0_plot) > clip] = np.nan

ax.plot(r, k_vtx, 'r-', lw=2, label=r'$k(r) = V^{\prime\prime}/2$')
ax.plot(r, r_0_plot, 'g-', lw=2, label=r'$r_0(r) = r - V^\prime/V^{\prime\prime}$')
ax.plot(r, b_vtx, 'b-', lw=2, label=r'$b_v(r)$')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(r_infl, color='orange', ls='--', lw=1.5, alpha=0.4,
           label='Inflection')
ax.set_xlabel('r [Å]')
ax.set_ylabel('Parameter value')
ax.set_title(r'(C)  Vertex: $V = k(r)(r - r_0(r))^2 + b_v(r)$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-clip, clip)

# --- Panel D: Monomial coefficients ---
ax = axes[1, 1]
ax.plot(r, C_m1, 'g-', lw=2.5, label=r'$n=-1$: $C_{-1} = V \cdot r$')
ax.plot(r, C_0, 'b--', lw=1.5, alpha=0.7, label=r'$n=0$: $C_0 = V$')
ax.plot(r, C_1, 'm:', lw=2, label=r'$n=1$: $C_1 = V/r$')
ax.plot(r, C_2, 'r--', lw=2, label=r'$n=2$: $C_2 = V/r^2$')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(r_infl, color='orange', ls='--', lw=1.5, alpha=0.4)
ax.set_xlabel('r [Å]')
ax.set_ylabel(r'$C_n(r)$')
ax.set_title(r'(D)  Monomial: $V = C_n(r) \cdot r^n$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-3, 1)

fig.suptitle('Morse curve: three local decompositions compared', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_morse_decomposition_comparison.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved fig_morse_decomposition_comparison.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Smoothness comparison — |dC/dr| for each parameter
# ================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: derivatives of standard parabola params
for name, param, color in [('a(r)', a_std, 'r'), ('b(r)', b_std, 'g'), ('c(r)', c_std, 'b')]:
    dpdr = np.abs(np.gradient(param, r))
    ax1.plot(r, dpdr, color=color, lw=2, label=f'|d{name}/dr|')
ax1.axvline(r_infl, color='orange', ls='--', lw=1.5, alpha=0.4, label='Inflection')
ax1.set_xlabel('r [Å]')
ax1.set_ylabel('|dP/dr|')
ax1.set_title('Standard parabola: derivative magnitudes')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 10)

# Right: derivatives of monomial coefficients
for name, C, color, ls in [('C_{-1}', C_m1, 'g', '-'), ('C_0', C_0, 'b', '--'),
                             ('C_1', C_1, 'm', ':'), ('C_2', C_2, 'r', '--')]:
    dCdr = np.abs(np.gradient(C, r))
    ax2.plot(r, dCdr, color=color, ls=ls, lw=2, label=f'|d{name}/dr|')
ax2.axvline(r_infl, color='orange', ls='--', lw=1.5, alpha=0.4, label='Inflection')
ax2.set_xlabel('r [Å]')
ax2.set_ylabel('|dC/dr|')
ax2.set_title('Monomials: derivative magnitudes')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 3)

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_morse_decomposition_smoothness.png'),
             dpi=150, bbox_inches='tight')
print('Saved fig_morse_decomposition_smoothness.png')
plt.close(fig2)

# ================================================================
# Print diagnostics
# ================================================================
print('\n--- Parameter ranges ---')
print(f'Standard parabola:')
print(f'  a(r): [{a_std.min():.4f}, {a_std.max():.4f}]')
print(f'  b(r): [{b_std.min():.4f}, {b_std.max():.4f}]')
print(f'  c(r): [{c_std.min():.4f}, {c_std.max():.4f}]')
print(f'\nVertex parabola:')
safe = np.abs(V_d2) > 0.01
print(f'  k(r):   [{k_vtx[safe].min():.4f}, {k_vtx[safe].max():.4f}]')
print(f'  r_0(r): [{r_0_vtx[safe].min():.4f}, {r_0_vtx[safe].max():.4f}]  (excl. inflection)')
print(f'  b_v(r): [{b_vtx[safe].min():.4f}, {b_vtx[safe].max():.4f}]')
print(f'\nMonomials:')
for name, C in [('C_2', C_2), ('C_1', C_1), ('C_0', C_0), ('C_{-1}', C_m1)]:
    dC = np.abs(np.gradient(C, r))
    print(f'  {name:6s}: range=[{C.min():.4f}, {C.max():.4f}], '
          f'mean |dC/dr|={dC.mean():.4f}')

print(f'\n--- Homogeneity check ---')
print(f'Monomial C*r^n: homogeneous of degree n at each r (virial = n*V)')
print(f'Parabola ar^2+br+c: NOT homogeneous (mixes degrees 2,1,0)')
print(f'  -> virial r*V\' = 2ar^2 + br ≠ n*(ar^2+br+c) for any fixed n')
print(f'Vertex k(r-r0)^2+b: NOT homogeneous either')
