"""
Step 0: Visualize local parabola parameters for Morse curves.

At each grid point r_j along a Morse curve, fit a local parabola:
    P_j(r) = a_j * (r - r_j)^2 + b_j * (r - r_j) + c_j

where (Taylor expansion):
    a_j = V''(r_j) / 2    (curvature)
    b_j = V'(r_j)         (slope)
    c_j = V(r_j)          (value)

All three are smooth everywhere, including at the inflection point.
Also shows the vertex form a(r-r0)^2 + b0 for comparison.

Plot a(r), b(r), c(r) as functions of r for several Morse curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def morse(r, D_e, alpha, r_e):
    """Morse potential: V(r) = D_e * [1 - exp(-alpha*(r-r_e))]^2 - D_e"""
    x = np.exp(-alpha * (r - r_e))
    return D_e * (1 - x)**2 - D_e


def morse_deriv1(r, D_e, alpha, r_e):
    """First derivative of Morse potential."""
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha * (1 - x) * x


def morse_deriv2(r, D_e, alpha, r_e):
    """Second derivative of Morse potential."""
    x = np.exp(-alpha * (r - r_e))
    return 2 * D_e * alpha**2 * x * (2 * x - 1)


def local_parabola_params_abc(r, D_e, alpha, r_e):
    """Compute local parabola parameters in ax^2+bx+c form (Taylor expansion).

    P_j(r) = a_j*(r-r_j)^2 + b_j*(r-r_j) + c_j

    Returns:
        a: V''(r)/2  — curvature (smooth, crosses zero at inflection)
        b: V'(r)     — slope (smooth)
        c: V(r)      — value (smooth)
    """
    c = morse(r, D_e, alpha, r_e)
    b = morse_deriv1(r, D_e, alpha, r_e)
    a = morse_deriv2(r, D_e, alpha, r_e) / 2.0
    return a, b, c


def plot_abc_params(curves, colors, filename):
    """Plot a(r), b(r), c(r) for multiple Morse curves."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for (D_e, alpha, r_e, label), color in zip(curves, colors):
        r_infl = r_e + np.log(2) / alpha
        r = np.linspace(r_e * 0.6, r_e * 4.0, 500)
        a, b, c = local_parabola_params_abc(r, D_e, alpha, r_e)

        # Top-left: Morse curve
        axes[0, 0].plot(r, c, color=color, label=label)
        axes[0, 0].axvline(r_infl, color=color, ls=':', alpha=0.4)

        # Top-right: a(r) = V''(r)/2 — curvature
        axes[0, 1].plot(r, a, color=color, label=label)
        axes[0, 1].axvline(r_infl, color=color, ls=':', alpha=0.4)
        axes[0, 1].axhline(0, color='gray', ls='-', lw=0.5)

        # Bottom-left: b(r) = V'(r) — slope
        axes[1, 0].plot(r, b, color=color, label=label)
        axes[1, 0].axvline(r_infl, color=color, ls=':', alpha=0.4)
        axes[1, 0].axhline(0, color='gray', ls='-', lw=0.5)

        # Bottom-right: c(r) = V(r) — value
        axes[1, 1].plot(r, c, color=color, label=label)
        axes[1, 1].axvline(r_infl, color=color, ls=':', alpha=0.4)

    axes[0, 0].set_ylabel('V(r) [eV]')
    axes[0, 0].set_title('Morse potential V(r)')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_ylabel(r"$a(r) = V''(r)/2$ [eV/Å²]")
    axes[0, 1].set_title('Curvature  a(r)')

    axes[1, 0].set_ylabel(r"$b(r) = V'(r)$ [eV/Å]")
    axes[1, 0].set_title('Slope  b(r)')

    axes[1, 1].set_ylabel('c(r) = V(r) [eV]')
    axes[1, 1].set_title('Value  c(r)')

    for ax in axes.flat:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r'Local parabola $P(r) = a(r\!-\!r_j)^2 + b(r\!-\!r_j) + c$ '
        'parameters along Morse curves'
        '\n(dotted lines = inflection points)',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')
    plt.close(fig)


def plot_abc_params_scaled(curves, colors, filename):
    """Plot a, b, c in scaled coordinates to check universality."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    r_scaled_pts = np.linspace(0.7, 3.5, 300)

    for (D_e, alpha, r_e, label), color in zip(curves, colors):
        r = r_scaled_pts * r_e
        a, b, c = local_parabola_params_abc(r, D_e, alpha, r_e)

        # Scale: a has units eV/Å², b has eV/Å, c has eV
        # Natural scales: D_e*alpha^2 for a, D_e*alpha for b, D_e for c
        axes[0].plot(r_scaled_pts, a / (D_e * alpha**2), color=color,
                     label=label)
        axes[1].plot(r_scaled_pts, b / (D_e * alpha), color=color,
                     label=label)
        axes[2].plot(r_scaled_pts, c / D_e, color=color, label=label)

    axes[0].set_ylabel(r'$a / (D_e \alpha^2)$')
    axes[0].set_title('Scaled curvature')
    axes[0].axhline(0, color='gray', ls='-', lw=0.5)

    axes[1].set_ylabel(r"$b / (D_e \alpha)$")
    axes[1].set_title('Scaled slope')
    axes[1].axhline(0, color='gray', ls='-', lw=0.5)

    axes[2].set_ylabel(r'$c / D_e$')
    axes[2].set_title('Scaled value')

    for ax in axes:
        ax.set_xlabel(r'$r / r_e$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # Mark inflection points
    for (D_e, alpha, r_e, _), color in zip(curves, colors):
        r_infl_scaled = 1 + np.log(2) / (alpha * r_e)
        for ax in axes:
            ax.axvline(r_infl_scaled, color=color, ls=':', alpha=0.3)

    fig.suptitle(
        r'Scaled local parabola parameters — universality check'
        '\n(dotted lines = inflection points)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')
    plt.close(fig)


def plot_local_parabolas_on_curve(D_e, alpha, r_e, filename):
    """Show a Morse curve with local parabolas (ax^2+bx+c form) drawn."""
    r = np.linspace(r_e * 0.6, r_e * 4.0, 500)
    V = morse(r, D_e, alpha, r_e)
    r_infl = r_e + np.log(2) / alpha

    # Grid points for local parabolas
    r_grid = np.linspace(r_e * 0.75, r_e * 3.5, 40)
    a_grid, b_grid, c_grid = local_parabola_params_abc(r_grid, D_e, alpha, r_e)

    show_idx = np.linspace(0, len(r_grid) - 1, 12).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.coolwarm

    # Left: full view
    ax = axes[0]
    ax.plot(r, V, 'k-', lw=2, label='Morse', zorder=5)
    ax.axvline(r_infl, color='gray', ls=':', alpha=0.5, label='inflection')

    for idx in show_idx:
        rj = r_grid[idx]
        aj, bj, cj = a_grid[idx], b_grid[idx], c_grid[idx]
        dr = min(0.8, abs(r_e * 0.3))
        r_local = np.linspace(rj - dr, rj + dr, 50)
        P_local = aj * (r_local - rj)**2 + bj * (r_local - rj) + cj
        P_local = np.clip(P_local, V.min() - 1, V.max() + 1)
        t = idx / (len(r_grid) - 1)
        ax.plot(r_local, P_local, color=cmap(t), alpha=0.6, lw=1.5)
        ax.plot(rj, cj, 'o', color=cmap(t), ms=4, zorder=6)

    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_ylim(V.min() - 0.5, min(2.0, V.max()))
    ax.set_title(f'Morse (D_e={D_e}, α={alpha}, r_e={r_e}) with local parabolas')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: zoom near equilibrium
    ax = axes[1]
    r_zoom = np.linspace(r_e * 0.7, r_e * 2.5, 300)
    V_zoom = morse(r_zoom, D_e, alpha, r_e)
    ax.plot(r_zoom, V_zoom, 'k-', lw=2, label='Morse', zorder=5)
    ax.axvline(r_infl, color='gray', ls=':', alpha=0.5)

    for idx in show_idx:
        rj = r_grid[idx]
        if rj < r_e * 0.7 or rj > r_e * 2.5:
            continue
        aj, bj, cj = a_grid[idx], b_grid[idx], c_grid[idx]
        dr = 0.4
        r_local = np.linspace(rj - dr, rj + dr, 50)
        P_local = aj * (r_local - rj)**2 + bj * (r_local - rj) + cj
        P_local = np.clip(P_local, V_zoom.min() - 0.5, 1.0)
        t = idx / (len(r_grid) - 1)
        ax.plot(r_local, P_local, color=cmap(t), alpha=0.7, lw=1.5)
        ax.plot(rj, cj, 'o', color=cmap(t), ms=5, zorder=6)

    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_ylim(-D_e - 0.3, 0.5)
    ax.set_title('Zoom: includes inflection region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')
    plt.close(fig)


def main():
    curves = [
        (3.0, 1.5, 1.5, r'$D_e$=3.0, $\alpha$=1.5, $r_e$=1.5'),
        (2.0, 1.0, 2.0, r'$D_e$=2.0, $\alpha$=1.0, $r_e$=2.0'),
        (5.0, 2.0, 1.0, r'$D_e$=5.0, $\alpha$=2.0, $r_e$=1.0'),
        (1.5, 0.8, 2.5, r'$D_e$=1.5, $\alpha$=0.8, $r_e$=2.5'),
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # --- Figure 1: a(r), b(r), c(r) in absolute coordinates ---
    plot_abc_params(curves, colors, 'fig_local_parabola_params_abc.png')

    # --- Figure 2: Morse curve with local parabolas drawn ---
    D_e, alpha, r_e = 3.0, 1.5, 1.5
    plot_local_parabolas_on_curve(D_e, alpha, r_e,
                                  'fig_local_parabolas_on_curve_abc.png')

    # --- Figure 3: Scaled parameters — universality check ---
    plot_abc_params_scaled(curves, colors,
                           'fig_local_parabola_params_abc_scaled.png')

    # --- Diagnostics ---
    print('\n--- Diagnostics ---')
    for D_e, alpha, r_e, label in curves:
        r_infl = r_e + np.log(2) / alpha
        a_eq = morse_deriv2(r_e, D_e, alpha, r_e) / 2
        b_eq = morse_deriv1(r_e, D_e, alpha, r_e)
        c_eq = morse(r_e, D_e, alpha, r_e)
        print(f'{label}:')
        print(f'  Inflection: r/r_e = {r_infl/r_e:.3f}')
        print(f'  At equilibrium: a={a_eq:.4f}, b={b_eq:.6f}, c={c_eq:.4f}')
        print(f'  At inflection:  a={morse_deriv2(r_infl, D_e, alpha, r_e)/2:.2e}, '
              f'b={morse_deriv1(r_infl, D_e, alpha, r_e):.4f}, '
              f'c={morse(r_infl, D_e, alpha, r_e):.4f}')


if __name__ == '__main__':
    main()
