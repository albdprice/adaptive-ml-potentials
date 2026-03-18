"""
Local parabolas on E_elec(r) = V_Morse(r) - Z1*Z2/r  for a fixed molecule (H2).

Vertex-form parabola: P(r) = a · (r - r₀)²
  - r₀ is the vertex (maximum, since a < 0)
  - At r₀: P = 0 (maximum)
  - a < 0: parabola opens downward

At each point r_j on the curve, match value and slope:
  E_elec(r_j) = a · (r_j - r₀)²         ...(1)
  E_elec'(r_j) = 2a · (r_j - r₀)        ...(2)

Solving:
  r₀ = r_j - 2 · E_elec(r_j) / E_elec'(r_j)
  a  = E_elec'(r_j)² / (4 · E_elec(r_j))
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Physical constants and H2 parameters
# ================================================================
ke = 14.3996  # eV·Å

# H2 Morse parameters
D_e = 4.75    # eV
alpha = 1.94  # Å⁻¹
r_e = 0.74    # Å
Z1, Z2 = 1, 1

# ================================================================
# E_elec = V_Morse - V_nn  and derivatives
# ================================================================
def V_morse(r):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def V_nn(r):
    return Z1 * Z2 * ke / r

def E_elec(r):
    return V_morse(r) - V_nn(r)

def E_elec_d1(r):
    x = np.exp(-alpha * (r - r_e))
    dV_morse = 2 * D_e * alpha * (1 - x) * x
    dV_nn = -Z1 * Z2 * ke / r**2
    return dV_morse - dV_nn

def E_elec_d2(r):
    x = np.exp(-alpha * (r - r_e))
    d2V_morse = 2 * D_e * alpha**2 * x * (2*x - 1)
    d2V_nn = 2 * Z1 * Z2 * ke / r**3
    return d2V_morse - d2V_nn

# ================================================================
# Vertex-form parabola: P(r) = a · (r - r₀)²
# ================================================================
def vertex_parabola_params(rj):
    """At point r_j, find (a, r₀) for P(r) = a·(r-r₀)².

    Match value and slope:
      E(r_j) = a·(r_j - r₀)²
      E'(r_j) = 2a·(r_j - r₀)

    Returns a, r0.
    """
    E = E_elec(rj)
    dE = E_elec_d1(rj)

    if abs(dE) < 1e-12 or abs(E) < 1e-12:
        return np.nan, np.nan

    r0 = rj - 2 * E / dE
    a = dE**2 / (4 * E)
    return a, r0

# ================================================================
# Grid
# ================================================================
r_plot = np.linspace(0.35, 5.0, 1000)
E_plot = E_elec(r_plot)

# ================================================================
# FIGURE 1: E_elec with vertex-form parabolas
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# --- Panel 1: Decomposition ---
ax = axes[0]
ax.plot(r_plot, V_morse(r_plot), 'b-', lw=2, label='$V_{Morse}(r)$')
ax.plot(r_plot, V_nn(r_plot), 'r--', lw=1.5, label='$V_{nn} = Z_1 Z_2 / r$')
ax.plot(r_plot, E_plot, 'k-', lw=2.5, label='$E_{elec} = V_{Morse} - V_{nn}$')
ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('Energy [eV]', fontsize=12)
ax.set_title('H$_2$: energy decomposition', fontsize=11)
ax.set_ylim(-35, 15)
ax.set_xlim(0.35, 5.0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 2: E_elec with vertex-form parabolas ---
ax = axes[1]
ax.plot(r_plot, E_plot, 'k-', lw=2.5, label='$E_{elec}(r)$', zorder=5)

r_samples = np.linspace(0.5, 3.5, 12)
cmap = plt.cm.viridis

for i, rj in enumerate(r_samples):
    a, r0 = vertex_parabola_params(rj)
    if np.isnan(a):
        continue

    # Draw parabola over a range around the contact point
    dr = 0.6
    r_local = np.linspace(max(0.3, rj - dr), min(5.0, rj + dr), 100)
    P_local = a * (r_local - r0)**2

    # Clip to visible range
    P_local = np.clip(P_local, -45, 5)

    t = i / (len(r_samples) - 1)
    color = cmap(t)
    ax.plot(r_local, P_local, color=color, alpha=0.7, lw=1.5)
    ax.plot(rj, E_elec(rj), 'o', color=color, ms=6, zorder=6)

    # Mark vertex (maximum) at r₀
    if 0.3 < r0 < 5.5:
        ax.plot(r0, 0, 'v', color=color, ms=5, alpha=0.6, zorder=6)

ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$E_{elec}$ [eV]', fontsize=12)
ax.set_title('$E_{elec}(r)$ with vertex-form parabolas\n$P(r) = a \\cdot (r - r_0)^2$, max at $r_0$',
             fontsize=11)
ax.set_ylim(-40, 5)
ax.set_xlim(0.35, 5.0)
ax.grid(True, alpha=0.3)

# --- Panel 3: Parameters a(r) and r₀(r) ---
ax = axes[2]
r_param = np.linspace(0.42, 4.0, 300)
a_vals = np.zeros_like(r_param)
r0_vals = np.zeros_like(r_param)
for k, rj in enumerate(r_param):
    a_vals[k], r0_vals[k] = vertex_parabola_params(rj)

ax.plot(r_param, r0_vals, 'b-', lw=2.5, label='$r_0(r)$ — vertex position [Å]')
ax.plot(r_param, r_param, 'k:', alpha=0.3, lw=1, label='$r_0 = r$ line')

# Mark sample points
for i, rj in enumerate(r_samples):
    a, r0 = vertex_parabola_params(rj)
    if np.isnan(a):
        continue
    t = i / (len(r_samples) - 1)
    ax.plot(rj, r0, 'o', color=cmap(t), ms=6, zorder=6)

ax.set_xlabel('$r$ [Å]', fontsize=12)
ax.set_ylabel('$r_0$ [Å]', fontsize=12)
ax.set_title('Vertex position $r_0(r)$\n$r_0 = r - 2 E_{elec} / E\'_{elec}$', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.4, 4.0)

fig.suptitle('H$_2$: $E_{elec}$ decomposed into vertex-form parabolas $a \\cdot (r - r_0)^2$',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig_eelec_vertex_parabolas.png'), dpi=150, bbox_inches='tight')
print('Saved fig_eelec_vertex_parabolas.png')
plt.close(fig)

# ================================================================
# FIGURE 2: Cleaner view — just the curve + parabolas, annotated
# ================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: E_elec with parabolas, labeled
ax = ax1
ax.plot(r_plot, E_plot, 'k-', lw=3, label='$E_{elec}(r)$', zorder=5)

r_samples2 = np.array([0.5, 0.8, 1.2, 1.7, 2.3, 3.0])
for i, rj in enumerate(r_samples2):
    a, r0 = vertex_parabola_params(rj)
    if np.isnan(a):
        continue

    dr = 0.7
    r_local = np.linspace(max(0.3, rj - dr), min(5.5, rj + dr), 150)
    P_local = a * (r_local - r0)**2
    P_local = np.clip(P_local, -45, 5)

    t = i / (len(r_samples2) - 1)
    color = cmap(t)
    ax.plot(r_local, P_local, color=color, alpha=0.8, lw=2)
    ax.plot(rj, E_elec(rj), 'o', color=color, ms=8, zorder=6)

    # Mark vertex with triangle and label
    if 0 < r0 < 6:
        ax.plot(r0, 0, '^', color=color, ms=8, zorder=6)
        ax.annotate(f'$r_0$={r0:.2f}', (r0, 0),
                    textcoords='offset points', xytext=(5, 8),
                    fontsize=7, color=color, fontweight='bold')

ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('$r$ [Å]', fontsize=13)
ax.set_ylabel('$E_{elec}$ [eV]', fontsize=13)
ax.set_title('$P(r) = a \\cdot (r - r_0)^2$\nTriangles mark vertex (max) at $r_0$', fontsize=12)
ax.set_ylim(-40, 5)
ax.set_xlim(0.3, 5.0)
ax.grid(True, alpha=0.3)

# Right: both parameters a(r) and r₀(r)
ax = ax2
ax_twin = ax.twinx()

ax.plot(r_param, a_vals, 'b-', lw=2.5, label='$a(r)$ [eV/Å²]')
ax_twin.plot(r_param, r0_vals, 'r-', lw=2.5, label='$r_0(r)$ [Å]')

# Mark sample points on both
for i, rj in enumerate(r_samples2):
    a, r0 = vertex_parabola_params(rj)
    if np.isnan(a):
        continue
    t = i / (len(r_samples2) - 1)
    ax.plot(rj, a, 'o', color=cmap(t), ms=7, zorder=6)
    ax_twin.plot(rj, r0, 's', color=cmap(t), ms=7, zorder=6, alpha=0.6)

ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.set_xlabel('$r$ [Å]', fontsize=13)
ax.set_ylabel('$a(r)$ [eV/Å²]', fontsize=13, color='b')
ax_twin.set_ylabel('$r_0(r)$ [Å]', fontsize=13, color='r')
ax.set_title('Vertex parabola parameters\n$a(r)$ and $r_0(r)$ — both smooth', fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax_twin.tick_params(axis='y', labelcolor='r')
ax.set_xlim(0.4, 4.0)
ax.grid(True, alpha=0.3)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower right')

fig2.suptitle('H$_2$: vertex-form parabola $a \\cdot (r - r_0)^2$ on $E_{elec}(r)$',
              fontsize=14)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_eelec_vertex_parabolas_clean.png'), dpi=150, bbox_inches='tight')
print('Saved fig_eelec_vertex_parabolas_clean.png')
plt.close(fig2)

# ================================================================
# Print diagnostics
# ================================================================
print(f'\nH2: E_elec = V_Morse - Z1Z2/r')
print(f'  D_e={D_e} eV, α={alpha} Å⁻¹, r_e={r_e} Å, Z1Z2={Z1*Z2}')
print(f'\nVertex parabola P(r) = a·(r-r₀)² at selected r:')
print(f'  {"r":>5s} | {"a":>10s} | {"r₀":>8s} | {"E_elec":>10s} | {"P(r)":>10s} | {"residual":>10s}')
print(f'  {"-"*65}')
for r in [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    a, r0 = vertex_parabola_params(r)
    E = E_elec(r)
    P = a * (r - r0)**2
    print(f'  {r:5.2f} | {a:10.4f} | {r0:8.3f} | {E:10.3f} | {P:10.3f} | {abs(E-P):10.6f}')
