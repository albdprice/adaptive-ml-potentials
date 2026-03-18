"""
Explore b=0 parabola decomposition of Morse curves.

Key idea: any Morse curve V(r) can be EXACTLY written as
    V(r) = a(r) * (r - r_e)^2
where a(r) is a position-dependent curvature that we compute.

This is the "b=0" form: no linear term, no constant. Just curvature
times displacement squared.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def morse_unshifted(r, D_e, alpha, r_e):
    """Morse potential with V(r_e) = 0, V(inf) = D_e."""
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2


def compute_a_of_r(r, D_e, alpha, r_e):
    """Compute a(r) = V(r) / (r - r_e)^2.

    Analytically: a(r) = D_e * alpha^2 * [(1-exp(-u))/u]^2
    where u = alpha*(r - r_e).
    At r = r_e (u=0), L'Hopital gives a = D_e * alpha^2.
    """
    u = alpha * (r - r_e)
    # Safe computation of (1-exp(-u))/u near u=0
    f = np.where(np.abs(u) < 1e-8,
                 1.0 - u / 2.0 + u**2 / 6.0,
                 (1 - np.exp(-u)) / u)
    return D_e * alpha**2 * f**2


def main():
    curves = [
        (3.0, 1.5, 1.5, r'$D_e$=3, $\alpha$=1.5, $r_e$=1.5'),
        (2.0, 1.0, 2.0, r'$D_e$=2, $\alpha$=1.0, $r_e$=2.0'),
        (5.0, 2.0, 1.0, r'$D_e$=5, $\alpha$=2.0, $r_e$=1.0'),
        (1.5, 0.8, 2.5, r'$D_e$=1.5, $\alpha$=0.8, $r_e$=2.5'),
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # =================================================================
    # FIGURE A: What is the decomposition? (single Morse curve example)
    # =================================================================
    D_e, alpha, r_e = 3.0, 1.5, 1.5
    r = np.linspace(r_e * 0.5, r_e * 4.0, 500)
    V = morse_unshifted(r, D_e, alpha, r_e)
    a = compute_a_of_r(r, D_e, alpha, r_e)
    sqrt_a = np.sqrt(a)
    displacement_sq = (r - r_e)**2

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Panel A: The Morse curve
    ax = axes[0, 0]
    ax.plot(r, V, 'k-', lw=2)
    ax.axvline(r_e, color='gray', ls='--', alpha=0.5, label=f'$r_e$ = {r_e}')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('(A)  Morse potential V(r)')
    ax.legend(fontsize=9)
    ax.annotate('V = 0 at equilibrium', xy=(r_e, 0), xytext=(r_e + 1.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # Panel B: The displacement squared
    ax = axes[0, 1]
    ax.plot(r, displacement_sq, 'k-', lw=2)
    ax.axvline(r_e, color='gray', ls='--', alpha=0.5)
    ax.set_ylabel(r'$(r - r_e)^2$ [Å²]')
    ax.set_title(r'(B)  Displacement squared $(r - r_e)^2$')

    # Panel C: The position-dependent curvature a(r)
    ax = axes[1, 0]
    ax.plot(r, a, 'b-', lw=2, label='$a(r)$')
    ax.plot(r, sqrt_a, 'r--', lw=2, label=r'$\sqrt{a(r)}$')
    ax.axvline(r_e, color='gray', ls='--', alpha=0.5)
    ax.axhline(D_e * alpha**2, color='blue', ls=':', alpha=0.4)
    ax.annotate(f'$a(r_e) = D_e \\alpha^2 = {D_e * alpha**2:.1f}$',
                xy=(r_e, D_e * alpha**2),
                xytext=(r_e + 1.5, D_e * alpha**2 + 2),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')
    ax.set_ylabel(r'[eV/Å²] or [eV$^{1/2}$/Å]')
    ax.set_title(r'(C)  Position-dependent curvature $a(r)$ and $\sqrt{a(r)}$')
    ax.legend(fontsize=10)

    # Panel D: Verify the decomposition works
    ax = axes[1, 1]
    V_recon = a * displacement_sq
    ax.plot(r, V, 'k-', lw=2, label='Original V(r)')
    ax.plot(r, V_recon, 'r--', lw=2, alpha=0.7,
            label=r'$a(r) \cdot (r - r_e)^2$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('V(r) [eV]')
    ax.set_title(r'(D)  Check: $V(r) = a(r) \cdot (r - r_e)^2$ is exact')
    ax.legend(fontsize=10)

    for ax in axes.flat:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r'Decomposition of Morse into $V(r) = a(r) \cdot (r - r_e)^2$'
        f'\n(Example: $D_e$={D_e}, $\\alpha$={alpha}, $r_e$={r_e})',
        fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_b0_decomposition.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_b0_decomposition.png')
    plt.close(fig)

    # =================================================================
    # FIGURE B: a(r) and sqrt(a) for multiple curves — how do they vary?
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for (D_e, alpha, r_e, label), color in zip(curves, colors):
        r = np.linspace(r_e * 0.5, r_e * 4.0, 500)
        a = compute_a_of_r(r, D_e, alpha, r_e)

        axes[0].plot(r, a, color=color, lw=1.5, label=label)
        axes[1].plot(r, np.sqrt(a), color=color, lw=1.5, label=label)

        # Mark equilibrium
        axes[0].plot(r_e, D_e * alpha**2, 'o', color=color, ms=6)
        axes[1].plot(r_e, np.sqrt(D_e) * alpha, 'o', color=color, ms=6)

    axes[0].set_ylabel(r'$a(r)$ [eV/Å²]')
    axes[0].set_title(r'(A)  Curvature $a(r)$ — varies a lot across curves')

    axes[1].set_ylabel(r'$\sqrt{a(r)}$ [eV$^{1/2}$/Å]')
    axes[1].set_title(r'(B)  $\sqrt{a(r)}$ — varies less')

    # Panel C: check how different the value at equilibrium is
    ax = axes[2]
    for (D_e, alpha, r_e, label), color in zip(curves, colors):
        ax.bar(label.replace('$', '').replace('\\', ''),
               D_e * alpha**2, color=color, alpha=0.7)
    ax.set_ylabel(r'$a(r_e) = D_e \alpha^2$ [eV/Å²]')
    ax.set_title(r'(C)  Curvature at equilibrium')
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    for ax in axes[:2]:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        r'How $a(r)$ and $\sqrt{a(r)}$ vary across different Morse curves'
        '\n(dots = equilibrium values)',
        fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_b0_across_curves.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_b0_across_curves.png')
    plt.close(fig)

    # =================================================================
    # FIGURE C: Universal collapse in scaled coordinates
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for (D_e, alpha, r_e, label), color in zip(curves, colors):
        u = np.linspace(-3, 8, 500)  # u = alpha*(r - r_e)
        r = r_e + u / alpha

        a = compute_a_of_r(r, D_e, alpha, r_e)

        # Left: raw a(r) vs u — does NOT collapse (different D_e, alpha)
        axes[0].plot(u, a, color=color, lw=1.5, label=label)

        # Middle: a / (D_e*alpha^2) vs u — should collapse
        axes[1].plot(u, a / (D_e * alpha**2), color=color, lw=1.5,
                     label=label)

        # Right: sqrt(a) / (sqrt(D_e)*alpha) vs u — should also collapse
        axes[2].plot(u, np.sqrt(a) / (np.sqrt(D_e) * alpha), color=color,
                     lw=1.5, label=label)

    axes[0].set_ylabel(r'$a$ [eV/Å²]')
    axes[0].set_title(r'(A)  Raw $a$ vs $u$ — different curves')

    axes[1].set_ylabel(r'$a / (D_e \alpha^2)$')
    axes[1].set_title(r'(B)  Scaled $a$ vs $u$ — all curves collapse')

    axes[2].set_ylabel(r'$\sqrt{a} / (\sqrt{D_e}\,\alpha)$')
    axes[2].set_title(r'(C)  Scaled $\sqrt{a}$ vs $u$ — also collapses')

    for ax in axes:
        ax.set_xlabel(r'$u = \alpha \, (r - r_e)$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        'Universal collapse: when plotted against '
        r'$u = \alpha(r-r_e)$ and scaled by $D_e, \alpha$,'
        '\nall Morse curves give the same shape',
        fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_b0_universal_collapse.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_b0_universal_collapse.png')
    plt.close(fig)

    # --- Print diagnostics ---
    print('\n--- Equilibrium values ---')
    print(f'{"Curve":<35s}  {"a(r_e)":>8s}  {"sqrt(a(r_e))":>12s}')
    for D_e, alpha, r_e, label in curves:
        a_eq = D_e * alpha**2
        sa_eq = np.sqrt(D_e) * alpha
        print(f'{label:<35s}  {a_eq:8.2f}  {sa_eq:12.4f}')


if __name__ == '__main__':
    main()
