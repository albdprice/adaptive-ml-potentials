"""
Detailed visualization of nuclear subtraction experiment.

Addresses:
1. Why are extrapolation MAEs so large? (repulsive wall dominates)
2. How does b=0 work on Morse without subtraction? (no inflection point problem)
3. What does V_elec = V_Morse - V_nn look like?
4. Where do prediction errors concentrate?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

HART_TO_EV_ANGSTROM = 14.3996


def morse_unshifted(r, D_e, alpha, r_e):
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2


def nuclear_repulsion(r, Z_eff):
    return Z_eff * HART_TO_EV_ANGSTROM / r


def compute_a_morse(r, D_e, alpha, r_e):
    u = alpha * (r - r_e)
    f = np.where(np.abs(u) < 1e-8,
                 1.0 - u / 2.0 + u**2 / 6.0,
                 (1 - np.exp(-u)) / u)
    return D_e * alpha**2 * f**2


def compute_b0_numerical(r, V, eps=1e-6):
    cs = CubicSpline(r, V)
    r_fine = np.linspace(r.min(), r.max(), 5000)
    V_fine = cs(r_fine)
    r_e = r_fine[np.argmin(V_fine)]
    E_min = V_fine.min()
    V_shifted = V - E_min
    dr = r - r_e
    a = np.full_like(r, np.nan, dtype=float)
    mask = np.abs(dr) > eps
    a[mask] = V_shifted[mask] / dr[mask]**2
    a[~mask] = cs(r_e, 2) / 2.0
    return r_e, E_min, a


def main():
    r_grid = np.linspace(1.0, 8.0, 50)
    r_fine = np.linspace(1.0, 8.0, 500)

    # ================================================================
    # FIGURE 1: The curves — what does subtraction actually do?
    # ================================================================
    # Use a training-range example and an extrapolation example
    examples = [
        # (D_e, alpha, r_e, Z_eff, label)
        (3.0, 1.1, 2.0, 3.0, 'Training (d1=1.0)'),
        (7.0, 1.1, 3.0, 7.0, 'Extrapolation (d1=3.0)'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for row, (D_e, alpha, r_e, Z_eff, label) in enumerate(examples):
        V_morse = morse_unshifted(r_fine, D_e, alpha, r_e)
        V_nn = nuclear_repulsion(r_fine, Z_eff)
        V_elec = V_morse - V_nn

        # Panel A: The Morse curve (total)
        ax = axes[row, 0]
        ax.plot(r_fine, V_morse, 'k-', lw=2)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axhline(D_e, color='gray', ls=':', alpha=0.5)
        ax.axvline(r_e, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('Energy [eV]')
        ax.set_title(f'(A) Morse = "total"\n{label}')
        ax.set_ylim(-2, min(50, V_morse.max()))
        ax.annotate(f'$D_e$={D_e:.0f} eV', xy=(6, D_e), fontsize=9,
                    color='gray')

        # Panel B: V_nn
        ax = axes[row, 1]
        ax.plot(r_fine, V_nn, 'r-', lw=2)
        ax.set_title(f'(B) $V_{{nn}} = {Z_eff:.0f} \\times 14.4/r$')
        ax.set_ylim(-2, min(50, V_nn[len(r_fine)//5]))

        # Panel C: V_elec = Morse - V_nn
        ax = axes[row, 2]
        ax.plot(r_fine, V_elec, 'b-', lw=2)
        ax.axhline(0, color='gray', lw=0.5)

        # Find and mark the electronic minimum
        idx_min = np.argmin(V_elec)
        ax.plot(r_fine[idx_min], V_elec[idx_min], 'ro', ms=8,
                label=f'min at r={r_fine[idx_min]:.2f} Å')
        ax.axvline(r_e, color='gray', ls='--', alpha=0.5,
                   label=f'Morse $r_e$={r_e:.1f}')
        ax.set_title('(C) $V_{elec} = V_{Morse} - V_{nn}$')
        ax.legend(fontsize=8)

        # Find inflection points
        V_elec_cs = CubicSpline(r_fine, V_elec)
        V_elec_d2 = V_elec_cs(r_fine, 2)
        # Panel D: second derivative (curvature) of Morse vs V_elec
        ax = axes[row, 3]
        V_morse_cs = CubicSpline(r_fine, V_morse)
        V_morse_d2 = V_morse_cs(r_fine, 2)
        ax.plot(r_fine, V_morse_d2, 'k-', lw=1.5, label='Morse $V\'\'(r)$')
        ax.plot(r_fine, V_elec_d2, 'b--', lw=1.5, label='$V_{elec}\'\'(r)$')
        ax.axhline(0, color='gray', lw=0.5)
        # Mark inflection points (V'' = 0)
        for sign_change_label, d2, color in [
                ('Morse', V_morse_d2, 'black'),
                ('V_elec', V_elec_d2, 'blue')]:
            for j in range(len(r_fine)-1):
                if d2[j] * d2[j+1] < 0:
                    r_infl = r_fine[j] + (r_fine[j+1]-r_fine[j]) * abs(d2[j]) / (abs(d2[j])+abs(d2[j+1]))
                    ax.axvline(r_infl, color=color, ls=':', alpha=0.5)
                    ax.annotate(f'{sign_change_label}\ninfl={r_infl:.2f}',
                                xy=(r_infl, 0), fontsize=7, color=color,
                                ha='center', va='bottom')
        ax.set_title("(D) Second derivative $V''(r)$\n(inflection where $V''=0$)")
        ax.legend(fontsize=8)
        ax.set_ylim(-20, 40)

    for ax in axes.flat:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)

    fig.suptitle('What does nuclear subtraction do to the curves?', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_nuc_sub_detailed_curves.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_nuc_sub_detailed_curves.png')
    plt.close(fig)

    # ================================================================
    # FIGURE 2: a(r) comparison — Morse vs V_elec
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, (D_e, alpha, r_e, Z_eff, label) in enumerate(examples):
        V_morse_g = morse_unshifted(r_grid, D_e, alpha, r_e)
        V_nn_g = nuclear_repulsion(r_grid, Z_eff)
        V_elec_g = V_morse_g - V_nn_g

        # a(r) for Morse (analytical)
        a_morse = compute_a_morse(r_grid, D_e, alpha, r_e)

        # a(r) for V_elec (numerical)
        re_e, emin_e, a_elec = compute_b0_numerical(r_grid, V_elec_g)

        # Panel A: a(r) for Morse
        ax = axes[row, 0]
        ax.plot(r_grid, a_morse, 'k-o', lw=1.5, ms=3)
        ax.set_title(f'(A) $a(r)$ for Morse\n{label}')
        ax.set_ylabel(r'$a(r)$ [eV/Å²]')
        ax.annotate(f'$a(r_e) = D_e\\alpha^2 = {D_e*alpha**2:.1f}$',
                    xy=(r_e, D_e*alpha**2), fontsize=8,
                    xytext=(r_e+1, D_e*alpha**2+1),
                    arrowprops=dict(arrowstyle='->', color='blue'))

        # Panel B: a(r) for V_elec
        ax = axes[row, 1]
        mask = np.isfinite(a_elec) & (np.abs(a_elec) < 300)
        ax.plot(r_grid[mask], a_elec[mask], 'b-o', lw=1.5, ms=3)
        ax.set_title(f'(B) $a(r)$ for $V_{{elec}}$\n$r_{{e,elec}}$={re_e:.2f}, $E_{{min}}$={emin_e:.1f}')
        ax.set_ylabel(r'$a(r)$ [eV/Å²]')
        # Mark where a goes negative
        neg_mask = a_elec < 0
        if np.any(neg_mask):
            ax.plot(r_grid[neg_mask], a_elec[neg_mask], 'rx', ms=8,
                    label='negative a(r)')
            ax.legend(fontsize=8)

        # Panel C: overlay both on same plot
        ax = axes[row, 2]
        ax.plot(r_grid, a_morse, 'k-', lw=2, label='Morse (smooth)')
        ax.plot(r_grid[mask], a_elec[mask], 'b--', lw=2,
                label='$V_{elec}$ (after sub)')
        ax.set_title('(C) Both overlaid')
        ax.set_ylabel(r'$a(r)$ [eV/Å²]')
        ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)

    fig.suptitle(r'Position-dependent curvature $a(r)$: Morse is already smooth!'
                 '\nSubtracting $V_{nn}$ shifts the minimum and changes $a(r)$ shape',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_nuc_sub_detailed_a.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_nuc_sub_detailed_a.png')
    plt.close(fig)

    # ================================================================
    # FIGURE 3: Why are extrapolation MAEs so large?
    # ================================================================
    # Show the SCALE of extrapolation curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Training curves
    rng = np.random.RandomState(42)
    d1_train = rng.uniform(0.5, 1.5, 10)
    d1_extrap = np.array([2.5, 3.0, 3.5, 4.0])

    ax = axes[0]
    for d1 in d1_train[:5]:
        D_e = 1 + 2*d1
        alpha = 1.1  # fix alpha for clarity
        r_e_val = 1.5 + 0.5*d1
        V = morse_unshifted(r_fine, D_e, alpha, r_e_val)
        ax.plot(r_fine, V, 'b-', alpha=0.3, lw=1)
    for d1 in d1_extrap:
        D_e = 1 + 2*d1
        r_e_val = 1.5 + 0.5*d1
        V = morse_unshifted(r_fine, D_e, alpha, r_e_val)
        ax.plot(r_fine, V, 'r-', lw=1.5,
                label=f'd1={d1:.1f}, $D_e$={D_e:.0f}')
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('(A) Full scale: training (blue) vs extrap (red)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Zoom into the well region
    ax = axes[1]
    for d1 in d1_train[:5]:
        D_e = 1 + 2*d1
        r_e_val = 1.5 + 0.5*d1
        V = morse_unshifted(r_fine, D_e, alpha, r_e_val)
        ax.plot(r_fine, V, 'b-', alpha=0.3, lw=1)
    for d1 in d1_extrap:
        D_e = 1 + 2*d1
        r_e_val = 1.5 + 0.5*d1
        V = morse_unshifted(r_fine, D_e, alpha, r_e_val)
        ax.plot(r_fine, V, 'r-', lw=1.5, label=f'd1={d1:.1f}')
    ax.set_ylim(-1, 20)
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('(B) Zoomed: the interesting physics is < 20 eV')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Show V at r=1.0 (first grid point)
    ax = axes[2]
    d1_all = np.linspace(0.5, 4.0, 100)
    V_at_r1 = []
    for d1 in d1_all:
        D_e = 1 + 2*d1
        r_e_val = 1.5 + 0.5*d1
        V = morse_unshifted(np.array([1.0]), D_e, alpha, r_e_val)[0]
        V_at_r1.append(V)
    ax.plot(d1_all, V_at_r1, 'k-', lw=2)
    ax.axvspan(0.5, 1.5, alpha=0.2, color='blue', label='Training')
    ax.axvspan(2.5, 4.0, alpha=0.2, color='red', label='Extrapolation')
    ax.set_xlabel('d1')
    ax.set_ylabel('V(r=1.0 Å) [eV]')
    ax.set_title('(C) Value at r=1 Å (first grid point)\nThis drives the huge MAEs')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Why are extrapolation MAEs so large?\n'
                 'The repulsive wall at r=1 Å can be thousands of eV for extrapolation curves',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_nuc_sub_detailed_scale.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_nuc_sub_detailed_scale.png')
    plt.close(fig)

    # ================================================================
    # FIGURE 4: The inflection point — b=0 handles it fine
    # ================================================================
    D_e, alpha, r_e = 3.0, 1.1, 2.0
    V = morse_unshifted(r_fine, D_e, alpha, r_e)
    a = compute_a_morse(r_fine, D_e, alpha, r_e)

    # Find inflection point
    V_cs = CubicSpline(r_fine, V)
    V_d2 = V_cs(r_fine, 2)
    for j in range(len(r_fine)-1):
        if V_d2[j] * V_d2[j+1] < 0:
            r_infl = r_fine[j]
            break

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Morse curve with inflection point marked
    ax = axes[0]
    ax.plot(r_fine, V, 'k-', lw=2)
    ax.axvline(r_infl, color='red', ls='--', lw=1.5,
               label=f'Inflection r={r_infl:.2f}')
    ax.axvline(r_e, color='gray', ls=':', alpha=0.5, label=f'$r_e$={r_e:.1f}')
    ax.set_ylim(-0.5, 8)
    ax.set_title("(A) Morse curve\ninflection point marked")
    ax.legend(fontsize=9)

    # Second derivative
    ax = axes[1]
    ax.plot(r_fine, V_d2, 'k-', lw=2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(r_infl, color='red', ls='--', lw=1.5)
    ax.set_title("(B) $V''(r)$\ncrosses zero at inflection")
    ax.set_ylim(-5, 15)

    # a(r) — SMOOTH through inflection
    ax = axes[2]
    ax.plot(r_fine, a, 'b-', lw=2)
    ax.axvline(r_infl, color='red', ls='--', lw=1.5,
               label='Inflection point')
    ax.set_title("(C) $a(r) = V/(r-r_e)^2$\nSMOOTH through inflection!")
    ax.set_ylabel(r'$a(r)$ [eV/Å²]')
    ax.legend(fontsize=9)

    # sqrt(a) — also smooth
    ax = axes[3]
    ax.plot(r_fine, np.sqrt(a), 'r-', lw=2)
    ax.axvline(r_infl, color='red', ls='--', lw=1.5,
               label='Inflection point')
    ax.set_title(r"(D) $\sqrt{a(r)}$" "\nalso smooth through inflection")
    ax.set_ylabel(r'$\sqrt{a}$')
    ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)

    fig.suptitle("The inflection point is NOT a problem for b=0\n"
                 r"$a(r) = D_e \alpha^2 [(1-e^{-u})/u]^2$ is smooth everywhere, "
                 "even where $V''(r) = 0$",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_nuc_sub_detailed_inflection.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_nuc_sub_detailed_inflection.png')
    plt.close(fig)

    # Print key numbers
    print('\n--- Key diagnostic numbers ---')
    for d1 in [0.5, 1.0, 1.5, 2.5, 3.0, 4.0]:
        D_e = 1 + 2*d1
        r_e_val = 1.5 + 0.5*d1
        alpha_val = 1.1
        V_r1 = morse_unshifted(np.array([1.0]), D_e, alpha_val, r_e_val)[0]
        V_r2 = morse_unshifted(np.array([2.0]), D_e, alpha_val, r_e_val)[0]
        regime = 'TRAIN' if 0.5 <= d1 <= 1.5 else 'EXTRAP'
        print(f'  d1={d1:.1f} ({regime:6s}): D_e={D_e:.1f}, r_e={r_e_val:.2f}, '
              f'V(r=1)={V_r1:.0f} eV, V(r=2)={V_r2:.1f} eV')


if __name__ == '__main__':
    main()
