"""
Virial Plot Demo: Linear Scaling for Homogeneous Functions
==========================================================

This demonstrates Euler's theorem for homogeneous functions:
    r·∇V = d·V  (for homogeneous function of degree d)

Or equivalently:
    r·F = -d·V  (where F = -∇V)

For HOMOGENEOUS functions: Virial plot (r·F vs V) is a STRAIGHT LINE
For NON-HOMOGENEOUS functions: Virial plot forms LOOPS

This is the CORE of the adaptive homogeneity concept!
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11


# =============================================================================
# POTENTIAL FUNCTIONS AND FORCES
# =============================================================================

def dispersion_V(r, C6):
    """V(r) = -C6/r^6 (homogeneous degree -6)"""
    return -C6 / r**6

def dispersion_F(r, C6):
    """F(r) = -dV/dr = -6*C6/r^7"""
    return -6 * C6 / r**7

def coulomb_V(r, q):
    """V(r) = q/r (homogeneous degree -1)"""
    return q / r

def coulomb_F(r, q):
    """F(r) = -dV/dr = q/r^2"""
    return q / r**2

def lj_V(r, eps, sig):
    """LJ: V = 4ε[(σ/r)^12 - (σ/r)^6]"""
    return 4 * eps * ((sig/r)**12 - (sig/r)**6)

def lj_F(r, eps, sig):
    """F = -dV/dr = 4ε[12σ^12/r^13 - 6σ^6/r^7]"""
    return 4 * eps * (12 * sig**12 / r**13 - 6 * sig**6 / r**7)

def morse_V(r, De, a, re):
    """Morse: V = De(1 - e^(-a(r-re)))^2"""
    return De * (1 - np.exp(-a * (r - re)))**2

def morse_F(r, De, a, re):
    """F = -dV/dr = 2*De*a*(1 - e^(-a(r-re)))*e^(-a(r-re))"""
    x = np.exp(-a * (r - re))
    return 2 * De * a * (1 - x) * x


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    print("=" * 70)
    print("VIRIAL PLOT DEMO: Euler's Theorem for Homogeneous Functions")
    print("=" * 70)
    print()
    print("Euler's theorem: r·∇V = d·V for homogeneous functions of degree d")
    print("Virial form: r·F = -d·V (where F = -dV/dr)")
    print()
    print("→ HOMOGENEOUS: Virial plot is a STRAIGHT LINE (slope = -d)")
    print("→ NON-HOMOGENEOUS: Virial plot forms LOOPS")
    print()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # =========================================================================
    # Row 1: Potentials V(r)
    # =========================================================================

    # --- Coulomb ---
    r = np.linspace(0.8, 4.0, 200)
    q = 1.0
    V = coulomb_V(r, q)
    F = coulomb_F(r, q)

    axes[0, 0].plot(r, V, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('V(r)')
    axes[0, 0].set_title('Coulomb: V = q/r\n(degree d = -1)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.5)

    # Virial plot
    rF = r * F
    axes[1, 0].plot(V, rF, 'b-', linewidth=2)
    # Theory line: r·F = -d·V = -(-1)·V = V
    V_theory = np.linspace(0.01, 1.5, 100)
    axes[1, 0].plot(V_theory, 1 * V_theory, 'r--', linewidth=2, label='Theory: r·F = V')
    axes[1, 0].set_xlabel('V')
    axes[1, 0].set_ylabel('r·F')
    axes[1, 0].set_title('Virial: STRAIGHT LINE\n(slope = -d = 1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # --- Dispersion (C6) ---
    r = np.linspace(0.8, 4.0, 200)
    C6 = 1.0
    V = dispersion_V(r, C6)
    F = dispersion_F(r, C6)

    axes[0, 1].plot(r, V, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('V(r)')
    axes[0, 1].set_title('Dispersion: V = -C₆/r⁶\n(degree d = -6)')
    axes[0, 1].grid(True, alpha=0.3)

    # Virial plot
    rF = r * F
    axes[1, 1].plot(V, rF, 'g-', linewidth=2)
    # Theory: r·F = -d·V = -(-6)·V = 6V
    V_theory = np.linspace(-5, -0.01, 100)
    axes[1, 1].plot(V_theory, 6 * V_theory, 'r--', linewidth=2, label='Theory: r·F = 6V')
    axes[1, 1].set_xlabel('V')
    axes[1, 1].set_ylabel('r·F')
    axes[1, 1].set_title('Virial: STRAIGHT LINE\n(slope = -d = 6)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # --- Lennard-Jones ---
    eps, sig = 1.0, 1.0
    r = np.linspace(0.95 * sig, 3.0 * sig, 500)
    V = lj_V(r, eps, sig)
    F = lj_F(r, eps, sig)

    axes[0, 2].plot(r, V, 'purple', linewidth=2)
    axes[0, 2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('V(r)')
    axes[0, 2].set_title('Lennard-Jones\n(NOT homogeneous)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-1.5, 2)

    # Virial plot - LOOPS!
    rF = r * F
    axes[1, 2].plot(V, rF, 'purple', linewidth=2)
    axes[1, 2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 2].axvline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 2].set_xlabel('V')
    axes[1, 2].set_ylabel('r·F')
    axes[1, 2].set_title('Virial: LOOPS!\n(non-homogeneous)')
    axes[1, 2].grid(True, alpha=0.3)

    # Add arrows to show direction
    mid = len(r) // 4
    axes[1, 2].annotate('', xy=(V[mid+5], rF[mid+5]), xytext=(V[mid], rF[mid]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # --- Morse ---
    De, a, re = 1.0, 1.5, 1.0
    r = np.linspace(0.5 * re, 4.0 * re, 500)
    V = morse_V(r, De, a, re)
    F = morse_F(r, De, a, re)

    axes[0, 3].plot(r, V, 'r-', linewidth=2)
    axes[0, 3].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0, 3].set_xlabel('r')
    axes[0, 3].set_ylabel('V(r)')
    axes[0, 3].set_title('Morse\n(NOT homogeneous)')
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].set_ylim(-0.2, 1.5)

    # Virial plot - LOOPS!
    rF = r * F
    axes[1, 3].plot(V, rF, 'r-', linewidth=2)
    axes[1, 3].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 3].axvline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 3].set_xlabel('V')
    axes[1, 3].set_ylabel('r·F')
    axes[1, 3].set_title('Virial: LOOPS!\n(non-homogeneous)')
    axes[1, 3].grid(True, alpha=0.3)

    # Add arrow
    mid = len(r) // 3
    axes[1, 3].annotate('', xy=(V[mid+10], rF[mid+10]), xytext=(V[mid], rF[mid]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    plt.tight_layout()
    plt.savefig('fig_virial_plots.png', dpi=150)
    plt.close(fig)

    print("RESULTS:")
    print("-" * 60)
    print("Coulomb (d=-1):   Virial plot is LINE with slope 1  ✓")
    print("Dispersion (d=-6): Virial plot is LINE with slope 6  ✓")
    print("Lennard-Jones:    Virial plot is LOOP  ✗")
    print("Morse:            Virial plot is LOOP  ✗")
    print("-" * 60)
    print()
    print("IMPLICATION:")
    print("  For HOMOGENEOUS potentials (Coulomb, C6):")
    print("    → r·F = d·V gives LINEAR relationship")
    print("    → Can learn d directly from forces!")
    print("    → No 'adaptive' parameter needed (d is constant)")
    print()
    print("  For NON-HOMOGENEOUS potentials (LJ, Morse):")
    print("    → Virial relationship is NOT linear")
    print("    → Need ADAPTIVE parameters to capture the variation")
    print("    → This is where Rose/UBER approach helps!")


    # =========================================================================
    # Second figure: Multiple C6 curves showing they all have same slope
    # =========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)
    n_curves = 10
    C6_values = np.random.uniform(0.5, 3.0, n_curves)
    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))

    r = np.linspace(1.0, 4.0, 200)

    # Panel 1: Raw V(r) curves
    ax = axes[0]
    for C6, color in zip(C6_values, colors):
        V = dispersion_V(r, C6)
        ax.plot(r, V, color=color, linewidth=2, alpha=0.7)
    ax.set_xlabel('r')
    ax.set_ylabel('V(r)')
    ax.set_title(f'{n_curves} Different C₆ Values\n(different curves)')
    ax.grid(True, alpha=0.3)

    # Panel 2: Virial plots - ALL SAME SLOPE!
    ax = axes[1]
    for C6, color in zip(C6_values, colors):
        V = dispersion_V(r, C6)
        F = dispersion_F(r, C6)
        rF = r * F
        ax.plot(V, rF, color=color, linewidth=2, alpha=0.7)

    # Theory line
    V_range = np.linspace(-10, -0.1, 100)
    ax.plot(V_range, 6 * V_range, 'k--', linewidth=3, label='Slope = 6 (theory)')
    ax.set_xlabel('V')
    ax.set_ylabel('r·F')
    ax.set_title('Virial Plots: ALL SAME SLOPE!\n(d = -6 for all)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: What we can learn
    ax = axes[2]
    ax.text(0.5, 0.85, 'For Homogeneous Functions:', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.70, 'r·F = d·V', fontsize=16, ha='center', transform=ax.transAxes,
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.text(0.5, 0.55, '↓', fontsize=20, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.40, 'Linear regression on (V, r·F)\ngives slope = d directly!',
            fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.20, 'For C₆/r⁶: slope = 6\nFor q/r: slope = 1\nFor A/r¹²: slope = 12',
            fontsize=11, ha='center', transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig_c6_virial.png', dpi=150)
    plt.close(fig)

    print()
    print("=" * 60)
    print("C₆ DISPERSION: Learning from Forces")
    print("=" * 60)
    print()
    print("For V = -C₆/r⁶:")
    print("  F = -dV/dr = -6C₆/r⁷")
    print("  r·F = -6C₆/r⁶ = 6V")
    print()
    print("So: Linear regression of r·F vs V gives slope = 6")
    print("This confirms homogeneity with degree d = -6")
    print()
    print("KEY POINT: The slope is ALWAYS 6, regardless of C₆ value!")
    print("This is what makes it 'homogeneous' - the degree is constant.")


if __name__ == "__main__":
    main()
