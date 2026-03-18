"""
For a single Morse potential (fixed Z), decompose it into position-dependent
parabolas: at each r, find a(r) such that

    V(r) ≈ a(r) · (r - r_e)² + V_min

So:  a(r) = [V(r) - V(r_e)] / (r - r_e)²

At r = r_e:  a → V''(r_e)/2 = D_e · α²   (harmonic limit, by L'Hôpital)
Away from r_e: a(r) varies because Morse is anharmonic.

This shows:
  1. The Morse curve is NOT a constant-curvature parabola
  2. a(r) varies smoothly with r
  3. The adaptive idea: learn a(r) instead of V(r) directly
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Morse potential
# ================================================================
def morse(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def morse_d2(r, D_e, alpha, r_e):
    """Exact second derivative of Morse: V''(r)"""
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha**2 * x * (2*x - 1)

# ================================================================
# Compute a(r) for several Morse curves
# ================================================================
# Use parameters similar to our synthetic dataset
molecules = {
    'Molecule A': {'D_e': 3.0, 'alpha': 1.5, 'r_e': 2.8},
    'Molecule B': {'D_e': 2.0, 'alpha': 1.2, 'r_e': 3.2},
    'Molecule C': {'D_e': 4.5, 'alpha': 1.8, 'r_e': 2.5},
}

r_grid = np.linspace(1.5, 8.0, 500)

# ================================================================
# FIGURE 1: Single Morse curve + parabola decomposition
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Pick one molecule for the detailed view
p = molecules['Molecule A']
D_e, alpha, r_e = p['D_e'], p['alpha'], p['r_e']
V = morse(r_grid, D_e, alpha, r_e)
V_min = -D_e

# a(r) = (V(r) - V_min) / (r - r_e)²
dr = r_grid - r_e
dr2 = dr**2
# Avoid division by zero at r = r_e
mask = np.abs(dr) > 1e-8
a_r = np.full_like(r_grid, np.nan)
a_r[mask] = (V[mask] - V_min) / dr2[mask]
# At r_e: a = V''(r_e)/2 = D_e * alpha^2
a_at_re = D_e * alpha**2
idx_re = np.argmin(np.abs(r_grid - r_e))
a_r[~mask] = a_at_re

# Panel 1: The Morse curve + parabolas at several r values
ax = axes[0]
ax.plot(r_grid, V, 'k-', lw=2.5, label='Morse $V(r)$')

# Show parabolas at selected r points
r_samples = [1.8, 2.2, 2.8, 3.5, 4.5, 6.0]
colors_samp = plt.cm.viridis(np.linspace(0.1, 0.9, len(r_samples)))

for r_s, col in zip(r_samples, colors_samp):
    idx = np.argmin(np.abs(r_grid - r_s))
    a_s = a_r[idx]
    if np.isnan(a_s):
        a_s = a_at_re
    # Draw the parabola: V_parab(r) = a_s * (r - r_e)^2 + V_min
    V_parab = a_s * (r_grid - r_e)**2 + V_min
    ax.plot(r_grid, V_parab, color=col, ls='--', lw=1.2, alpha=0.7,
            label=f'$a({r_s:.1f})={a_s:.2f}$')
    # Mark the point where parabola touches Morse
    ax.plot(r_s, V[idx], 'o', color=col, ms=8, zorder=5)

ax.set_xlim(1.5, 7.0)
ax.set_ylim(-D_e - 0.5, D_e + 1)
ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$V(r)$ [eV]', fontsize=12)
ax.set_title('Morse + local parabolas\n$V \\approx a(r) \\cdot (r - r_e)^2 + V_{min}$', fontsize=11)
ax.legend(fontsize=7, loc='upper right')
ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.grid(True, alpha=0.3)

# Panel 2: a(r) vs r — the position-dependent curvature
ax = axes[1]
ax.plot(r_grid, a_r, 'k-', lw=2.5, label='$a(r) = [V(r) - V_{min}] / (r - r_e)^2$')
ax.axhline(a_at_re, color='r', ls='--', lw=1.5, alpha=0.7,
           label=f'Harmonic: $a = D_e \\alpha^2 = {a_at_re:.2f}$')
ax.axvline(r_e, color='gray', ls=':', alpha=0.5)

# Mark sample points
for r_s, col in zip(r_samples, colors_samp):
    idx = np.argmin(np.abs(r_grid - r_s))
    a_s = a_r[idx]
    if np.isnan(a_s):
        a_s = a_at_re
    ax.plot(r_s, a_s, 'o', color=col, ms=8, zorder=5)

ax.set_xlim(1.5, 7.0)
ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$a(r)$ [eV/Å²]', fontsize=12)
ax.set_title('Position-dependent curvature $a(r)$\nSmooth but NOT constant', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: a(r) for multiple molecules — all smooth
ax = axes[2]
for name, p in molecules.items():
    D_e, alpha, r_e = p['D_e'], p['alpha'], p['r_e']
    V = morse(r_grid, D_e, alpha, r_e)
    V_min = -D_e
    dr = r_grid - r_e
    dr2 = dr**2
    mask = np.abs(dr) > 1e-8
    a = np.full_like(r_grid, D_e * alpha**2)
    a[mask] = (V[mask] - V_min) / dr2[mask]
    ax.plot(r_grid, a, lw=2, label=f'{name} ($D_e$={p["D_e"]}, $\\alpha$={p["alpha"]}, $r_e$={p["r_e"]})')

ax.set_xlim(1.5, 7.0)
ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$a(r)$ [eV/Å²]', fontsize=12)
ax.set_title('$a(r)$ across molecules\nSame shape, different scale', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig.suptitle('Morse curve decomposed into position-dependent parabolas', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_morse_local_curvature.png'), dpi=150, bbox_inches='tight')
print('Saved fig_morse_local_curvature.png')
plt.close(fig)

# ================================================================
# FIGURE 2: The key insight — analytical form of a(r)
# ================================================================
# For Morse: V(r) = D_e*(1 - exp(-α(r-r_e)))² - D_e
# V(r) - V_min = D_e * (1 - exp(-α(r-r_e)))²
# a(r) = D_e * [(1 - exp(-α(r-r_e))) / (r - r_e)]²
#       = D_e * α² * [sinc_exp(α(r-r_e))]²
# where sinc_exp(x) = (1 - e^(-x)) / x
#
# At x=0: sinc_exp → 1, so a → D_e·α²
# At x→∞: sinc_exp → 1/x, so a → D_e·α²/x² → 0
# At x→-∞: sinc_exp → e^|x|/|x| → ∞ (repulsive wall)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

p = molecules['Molecule A']
D_e, alpha, r_e = p['D_e'], p['alpha'], p['r_e']

# Show the decomposition:  a(r) = D_e * α² * [sinc_exp(α·Δr)]²
x = alpha * (r_grid - r_e)  # dimensionless
sinc_exp = np.ones_like(x)
mask = np.abs(x) > 1e-10
sinc_exp[mask] = (1 - np.exp(-x[mask])) / x[mask]

a_analytical = D_e * alpha**2 * sinc_exp**2

# Verify against direct computation
V = morse(r_grid, D_e, alpha, r_e)
a_direct = np.full_like(r_grid, D_e * alpha**2)
mask2 = np.abs(r_grid - r_e) > 1e-8
a_direct[mask2] = (V[mask2] + D_e) / (r_grid[mask2] - r_e)**2

ax1.plot(r_grid, a_direct, 'k-', lw=2.5, label='$a(r)$ from $V(r)$')
ax1.plot(r_grid, a_analytical, 'r--', lw=1.5, label='$D_e \\alpha^2 \\cdot [\\mathrm{sinc}_{\\exp}(\\alpha \\Delta r)]^2$')
ax1.axhline(D_e * alpha**2, color='b', ls=':', alpha=0.5, label=f'Harmonic: $D_e\\alpha^2$ = {D_e*alpha**2:.2f}')
ax1.axvline(r_e, color='gray', ls=':', alpha=0.3)

ax1.set_xlabel('$r$ [Å]', fontsize=12)
ax1.set_ylabel('$a(r)$ [eV/Å²]', fontsize=12)
ax1.set_title('Analytical form of $a(r)$ for Morse', fontsize=12)
ax1.set_xlim(1.5, 7.0)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Show sinc_exp function
x_dense = np.linspace(-3, 8, 500)
se = np.ones_like(x_dense)
m = np.abs(x_dense) > 1e-10
se[m] = (1 - np.exp(-x_dense[m])) / x_dense[m]

ax2.plot(x_dense, se**2, 'k-', lw=2.5, label='$[\\mathrm{sinc}_{\\exp}(x)]^2$')
ax2.axhline(1, color='b', ls=':', alpha=0.5, label='$x=0$ limit: 1')
ax2.axvline(0, color='gray', ls=':', alpha=0.3)

ax2.set_xlabel('$x = \\alpha (r - r_e)$', fontsize=12)
ax2.set_ylabel('$[\\mathrm{sinc}_{\\exp}(x)]^2$', fontsize=12)
ax2.set_title('Universal shape factor\n$a(r) = D_e \\alpha^2 \\cdot f(x)$', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig2.suptitle('$a(r) = D_e \\alpha^2 \\cdot [(1-e^{-\\alpha\\Delta r})/(\\alpha\\Delta r)]^2$ — analytical decomposition',
              fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_morse_curvature_analytical.png'), dpi=150, bbox_inches='tight')
print('Saved fig_morse_curvature_analytical.png')
plt.close(fig2)

# ================================================================
# Print key values
# ================================================================
print(f'\nFor Molecule A (D_e={molecules["Molecule A"]["D_e"]}, '
      f'α={molecules["Molecule A"]["alpha"]}, r_e={molecules["Molecule A"]["r_e"]}):')
print(f'  Harmonic curvature: a(r_e) = D_e·α² = {molecules["Molecule A"]["D_e"] * molecules["Molecule A"]["alpha"]**2:.3f} eV/Å²')
print(f'  a(r) = D_e·α² · [(1-exp(-α·Δr))/(α·Δr)]²')
print(f'  → At r_e: a = {molecules["Molecule A"]["D_e"] * molecules["Molecule A"]["alpha"]**2:.3f}')

r_samples_print = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
D_e, alpha, r_e = molecules['Molecule A']['D_e'], molecules['Molecule A']['alpha'], molecules['Molecule A']['r_e']
print(f'\n  {"r":>5s} | {"a(r)":>8s} | {"V(r)":>8s} | {"parabola":>9s} | {"residual":>9s}')
print(f'  {"-"*50}')
for r in r_samples_print:
    v = morse(r, D_e, alpha, r_e)
    x = alpha * (r - r_e)
    if abs(x) < 1e-10:
        se = 1.0
    else:
        se = (1 - np.exp(-x)) / x
    a = D_e * alpha**2 * se**2
    v_parab = a * (r - r_e)**2 - D_e
    print(f'  {r:5.1f} | {a:8.4f} | {v:8.4f} | {v_parab:9.4f} | {abs(v - v_parab):9.6f}')
