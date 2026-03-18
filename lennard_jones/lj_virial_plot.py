"""
Virial Plot Demo for Lennard-Jones
====================================

Shows that LJ is NON-HOMOGENEOUS: the virial plot (r*F vs V) forms loops,
not straight lines. Compare with Coulomb (degree -1) and Dispersion (degree -6)
which are homogeneous and give linear virial plots.

Euler's theorem: for homogeneous V of degree d, r*V'(r) = d*V(r)
- If d is constant: virial plot is a straight line through origin
- If d varies (non-homogeneous): virial plot forms loops/curves
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14


# =============================================================================
# POTENTIAL AND FORCE FUNCTIONS
# =============================================================================

def coulomb_V(r, q):
    return q / r

def coulomb_F(r, q):
    return -q / r**2  # dV/dr

def dispersion_V(r, C6):
    return -C6 / r**6

def dispersion_F(r, C6):
    return 6 * C6 / r**7  # dV/dr (note: F = -dV/dr, but virial uses dV/dr)

def lj_V(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def lj_F(r, epsilon, sigma):
    """dV/dr for LJ."""
    return 4 * epsilon * (-12 * sigma**12 / r**13 + 6 * sigma**6 / r**7)


def main():
    print("=" * 70)
    print("VIRIAL PLOT DEMONSTRATION (Lennard-Jones)")
    print("=" * 70)
    print()
    print("Euler's theorem: r*dV/dr = d*V for homogeneous functions")
    print("  Homogeneous: virial plot is LINEAR")
    print("  Non-homogeneous: virial plot forms LOOPS")

    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    r = np.linspace(0.8, 4.0, 300)

    # =========================================================================
    # Column 1: Coulomb (homogeneous, degree -1)
    # =========================================================================
    q_vals = np.random.uniform(0.5, 3.0, 10)

    ax = axes[0, 0]
    for q in q_vals:
        V = coulomb_V(r, q)
        mask = V < 5
        ax.plot(r[mask], V[mask], alpha=0.6, linewidth=1.5)
    ax.set_xlabel('r')
    ax.set_ylabel('V(r)')
    ax.set_title('Coulomb: V = q/r\n(homogeneous, d = -1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    ax = axes[1, 0]
    for q in q_vals:
        V = coulomb_V(r, q)
        dVdr = coulomb_F(r, q)
        virial = r * dVdr
        mask = (V > 0) & (V < 5)
        ax.plot(V[mask], virial[mask], alpha=0.6, linewidth=1.5)
    # Theory line: r*dV/dr = -1 * V
    V_theory = np.linspace(0, 5, 100)
    ax.plot(V_theory, -1 * V_theory, 'k--', linewidth=2, label='d = -1 (theory)')
    ax.set_xlabel('V(r)')
    ax.set_ylabel('r * dV/dr')
    ax.set_title('Virial plot: LINEAR\n(slope = -1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Column 2: Dispersion (homogeneous, degree -6)
    # =========================================================================
    C6_vals = np.random.uniform(1.0, 10.0, 10)

    ax = axes[0, 1]
    for C6 in C6_vals:
        V = dispersion_V(r, C6)
        ax.plot(r, V, alpha=0.6, linewidth=1.5)
    ax.set_xlabel('r')
    ax.set_ylabel('V(r)')
    ax.set_title('Dispersion: V = -C6/r^6\n(homogeneous, d = -6)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for C6 in C6_vals:
        V = dispersion_V(r, C6)
        dVdr = dispersion_F(r, C6)
        virial = r * dVdr
        ax.plot(V, virial, alpha=0.6, linewidth=1.5)
    V_theory = np.linspace(-3, 0, 100)
    ax.plot(V_theory, -6 * V_theory, 'k--', linewidth=2, label='d = -6 (theory)')
    ax.set_xlabel('V(r)')
    ax.set_ylabel('r * dV/dr')
    ax.set_title('Virial plot: LINEAR\n(slope = -6)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Column 3: Lennard-Jones (NON-homogeneous)
    # =========================================================================
    epsilons = np.random.uniform(0.05, 0.3, 10)
    sigmas = np.random.uniform(2.0, 3.5, 10)

    ax = axes[0, 2]
    for eps, sig in zip(epsilons, sigmas):
        r_lj = np.linspace(0.9 * sig, 3.0 * sig, 300)
        V = lj_V(r_lj, eps, sig)
        ax.plot(r_lj, V, alpha=0.6, linewidth=1.5)
    ax.set_xlabel('r')
    ax.set_ylabel('V(r)')
    ax.set_title('LJ: 4eps[(sig/r)^12 - (sig/r)^6]\n(NON-homogeneous)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax = axes[1, 2]
    for eps, sig in zip(epsilons, sigmas):
        r_lj = np.linspace(0.9 * sig, 3.0 * sig, 300)
        V = lj_V(r_lj, eps, sig)
        dVdr = lj_F(r_lj, eps, sig)
        virial = r_lj * dVdr
        ax.plot(V, virial, alpha=0.6, linewidth=1.5)
    # Reference lines for pure r^-12 and r^-6 behavior
    V_ref = np.linspace(-0.3, 0.5, 100)
    ax.plot(V_ref, -12 * V_ref, ':', color='blue', linewidth=1.5, alpha=0.5, label='d = -12 (repulsive)')
    ax.plot(V_ref, -6 * V_ref, ':', color='green', linewidth=1.5, alpha=0.5, label='d = -6 (attractive)')
    ax.set_xlabel('V(r)')
    ax.set_ylabel('r * dV/dr')
    ax.set_title('Virial plot: LOOPS!\n(non-homogeneous)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_lj_virial.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nFigure saved: fig_lj_virial.png")
    print()
    print("RESULTS:")
    print("  Coulomb:    virial plot is LINEAR (slope = -1)  -> HOMOGENEOUS")
    print("  Dispersion: virial plot is LINEAR (slope = -6)  -> HOMOGENEOUS")
    print("  LJ:         virial plot forms LOOPS              -> NON-HOMOGENEOUS")
    print()
    print("IMPLICATION:")
    print("  Homogeneous potentials: just learn coefficient (q, C6)")
    print("  Non-homogeneous (LJ): learn ADAPTIVE parameters (epsilon, sigma)")
    print("  Physics equation provides correct shape!")


if __name__ == "__main__":
    main()
