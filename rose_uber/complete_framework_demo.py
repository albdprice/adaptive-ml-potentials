"""
Complete Framework Demo: Connecting Euler's Theorem to aPBE0
============================================================

This demonstrates the FULL story:

1. HOMOGENEOUS functions (C6, Coulomb):
   - Euler: r·F = d·V (linear!)
   - Can learn d directly from linear regression on forces
   - d is CONSTANT → no adaptive parameter needed

2. NON-HOMOGENEOUS functions (Morse, LJ, real DFT):
   - Virial plot forms LOOPS, not lines
   - k_eff(r) = r·V'/V varies with position
   - Need ADAPTIVE parameters (like aPBE0's α)

3. The aPBE0 analogy:
   - aPBE0: Learn α → get E from physics
   - Here: Learn (E_c, r_e, l) → get V from Rose equation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def dispersion_V(r, C6):
    return -C6 / r**6

def dispersion_F(r, C6):
    return -6 * C6 / r**7

def morse_V(r, De, a, re):
    return De * (1 - np.exp(-a * (r - re)))**2

def morse_F(r, De, a, re):
    x = np.exp(-a * (r - re))
    return 2 * De * a * (1 - x) * x


def main():
    print("=" * 70)
    print("COMPLETE FRAMEWORK: From Euler's Theorem to aPBE0")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 12))

    # =========================================================================
    # TOP ROW: The Linear Case (C6 - Homogeneous)
    # =========================================================================

    # Panel 1: Multiple C6 potentials
    ax1 = fig.add_subplot(2, 3, 1)
    np.random.seed(42)
    C6_values = [0.5, 1.0, 2.0, 3.0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(C6_values)))
    r = np.linspace(1.0, 4.0, 100)

    for C6, color in zip(C6_values, colors):
        V = dispersion_V(r, C6)
        ax1.plot(r, V, color=color, linewidth=2, label=f'C₆={C6}')

    ax1.set_xlabel('r')
    ax1.set_ylabel('V(r)')
    ax1.set_title('C₆ Dispersion Potentials\n(HOMOGENEOUS, d=-6)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Virial plot - LINEAR!
    ax2 = fig.add_subplot(2, 3, 2)
    all_V, all_rF = [], []

    for C6, color in zip(C6_values, colors):
        V = dispersion_V(r, C6)
        F = dispersion_F(r, C6)
        rF = r * F
        ax2.plot(V, rF, color=color, linewidth=2, label=f'C₆={C6}')
        all_V.extend(V)
        all_rF.extend(rF)

    # Linear fit to get slope
    all_V = np.array(all_V).reshape(-1, 1)
    all_rF = np.array(all_rF)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(all_V, all_rF)
    slope = reg.coef_[0]

    V_line = np.linspace(min(all_V), max(all_V), 100)
    ax2.plot(V_line, slope * V_line, 'k--', linewidth=3,
             label=f'Fit: slope = {slope:.2f}')

    ax2.set_xlabel('V')
    ax2.set_ylabel('r·F')
    ax2.set_title(f'Virial Plot: LINEAR!\nSlope = {slope:.1f} = -d (theory: 6)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: What this means
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.text(0.5, 0.85, 'For HOMOGENEOUS Functions:', fontsize=14, fontweight='bold',
             ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.70, 'Euler: r·F = d·V', fontsize=14, ha='center',
             transform=ax3.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax3.text(0.5, 0.55, '↓', fontsize=20, ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.42, 'Linear regression on (V, r·F)\nlearns d DIRECTLY from forces!',
             fontsize=12, ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.25, 'd is CONSTANT\n→ No adaptive parameter needed',
             fontsize=12, ha='center', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax3.text(0.5, 0.08, '(This is the "easy" case)', fontsize=11, ha='center',
             transform=ax3.transAxes, style='italic')
    ax3.axis('off')

    # =========================================================================
    # BOTTOM ROW: The Non-Linear Case (Morse - Non-Homogeneous)
    # =========================================================================

    # Panel 4: Multiple Morse potentials
    ax4 = fig.add_subplot(2, 3, 4)
    morse_params = [(1.0, 1.5, 1.0), (2.0, 1.2, 1.5), (1.5, 2.0, 1.2), (3.0, 1.0, 2.0)]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(morse_params)))

    for (De, a, re), color in zip(morse_params, colors):
        r = np.linspace(0.5 * re, 3.0 * re, 100)
        V = morse_V(r, De, a, re)
        ax4.plot(r, V, color=color, linewidth=2, label=f'De={De}')

    ax4.set_xlabel('r')
    ax4.set_ylabel('V(r)')
    ax4.set_title('Morse Potentials\n(NOT homogeneous)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.2, 4)

    # Panel 5: Virial plot - LOOPS!
    ax5 = fig.add_subplot(2, 3, 5)

    for (De, a, re), color in zip(morse_params, colors):
        r = np.linspace(0.5 * re, 3.0 * re, 200)
        V = morse_V(r, De, a, re)
        F = morse_F(r, De, a, re)
        rF = r * F
        ax5.plot(V, rF, color=color, linewidth=2, label=f'De={De}')

    ax5.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax5.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax5.set_xlabel('V')
    ax5.set_ylabel('r·F')
    ax5.set_title('Virial Plot: LOOPS!\n(NOT linear - no single d)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: The aPBE0 solution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.92, 'For NON-HOMOGENEOUS Functions:', fontsize=14, fontweight='bold',
             ha='center', transform=ax6.transAxes)
    ax6.text(0.5, 0.78, 'Virial plot = LOOPS\nNo single degree d', fontsize=11, ha='center',
             transform=ax6.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    ax6.text(0.5, 0.65, '↓', fontsize=20, ha='center', transform=ax6.transAxes)
    ax6.text(0.5, 0.52, 'Solution: ADAPTIVE PARAMETERS', fontsize=13, ha='center',
             transform=ax6.transAxes, fontweight='bold', color='darkgreen')

    ax6.text(0.25, 0.38, 'aPBE0:', fontsize=11, ha='center', transform=ax6.transAxes,
             fontweight='bold')
    ax6.text(0.25, 0.28, 'Learn α\n↓\nE = (1-α)E_PBE + αE_HF', fontsize=10, ha='center',
             transform=ax6.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax6.text(0.75, 0.38, 'Rose/UBER:', fontsize=11, ha='center', transform=ax6.transAxes,
             fontweight='bold')
    ax6.text(0.75, 0.28, 'Learn (E_c, r_e, l)\n↓\nV = E_c·f(r*)', fontsize=10, ha='center',
             transform=ax6.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax6.text(0.5, 0.08, 'Same principle: Learn bounded, smooth parameters\n'
             'Physics equation provides the rest!', fontsize=11, ha='center',
             transform=ax6.transAxes, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig('fig_complete_framework.png', dpi=150)
    plt.close(fig)

    print("\nSUMMARY:")
    print("-" * 60)
    print("HOMOGENEOUS (C6, Coulomb):")
    print("  - Euler's theorem: r·F = d·V (linear relationship)")
    print(f"  - Linear regression gives slope = {slope:.2f} ≈ 6 (theory)")
    print("  - d is CONSTANT → learn it directly from forces")
    print()
    print("NON-HOMOGENEOUS (Morse, LJ, real DFT):")
    print("  - Virial plot forms LOOPS, not lines")
    print("  - No single d works → need ADAPTIVE parameters")
    print("  - This is where aPBE0 philosophy applies!")
    print("-" * 60)


if __name__ == "__main__":
    main()
