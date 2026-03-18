"""
Apply nuclear repulsion subtraction + b=0 decomposition to real DFT diatomics.

Pipeline:
1. Load DFT binding curves: E_bind(r) = E_total(r) - E_atoms  [includes V_nn]
2. Subtract nuclear repulsion: E_elec(r) = E_bind(r) - Z_A*Z_B*14.3996/r
3. Shift: E_shifted(r) = E_elec(r) - E_elec(r_e)  [so E_shifted(r_e) = 0]
4. Decompose: E_shifted(r) = a(r) * (r - r_e)^2
5. Visualize a(r) — is it smooth? Universal?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import sys
import os

sys.path.insert(0, '/Users/albd/research/adaptive_paper_anatole/publication/extended/dft_diatomics')
from diatomic_adaptive_vs_direct import load_dft_data, assess_data_quality, filter_data

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

HART_TO_EV_ANGSTROM = 14.3996  # Z_A*Z_B * this / r_angstrom = V_nn in eV


def nuclear_repulsion(r, Z1, Z2):
    """V_nn(r) = Z1 * Z2 * 14.3996 / r  [eV, with r in Angstrom]."""
    return Z1 * Z2 * HART_TO_EV_ANGSTROM / r


def compute_b0_curvature(r, V, r_e, eps=1e-8):
    """Compute a(r) = V(r) / (r - r_e)^2 for the b=0 decomposition.

    V should satisfy V(r_e) = 0. Uses cubic spline for V''(r_e)/2 at r=r_e.
    """
    dr = r - r_e
    a = np.full_like(r, np.nan, dtype=float)
    mask = np.abs(dr) > eps
    a[mask] = V[mask] / dr[mask]**2

    # At r_e, use V''(r_e)/2 via spline
    cs = CubicSpline(r, V)
    a[~mask] = cs(r_e, 2) / 2.0

    return a


def main():
    # Load and filter DFT data
    data = load_dft_data(
        '/Users/albd/research/adaptive_paper_anatole/publication/extended/'
        'dft_diatomics/data/diatomic_curves.npz')
    keep_idx = assess_data_quality(data, max_residual=1.0, min_points=8)
    data = filter_data(data, keep_idx)

    names = data['names']
    descriptors = np.array(data['descriptors'])
    n_mol = len(names)

    print(f'Loaded {n_mol} molecules')

    # ================================================================
    # For each molecule: subtract V_nn, shift, decompose
    # ================================================================
    results = []
    for i in range(n_mol):
        name = names[i]
        Z1, Z2 = int(descriptors[i, 0]), int(descriptors[i, 1])
        r = np.array(data['r_grids'][i])
        E_bind = np.array(data['curves'][i])
        r_e = data['rose_params'][i][1]  # equilibrium from Rose fit

        # Nuclear repulsion
        V_nn = nuclear_repulsion(r, Z1, Z2)

        # Electronic binding (subtract nuclear repulsion)
        E_elec = E_bind - V_nn

        # Find equilibrium of electronic curve
        cs_elec = CubicSpline(r, E_elec)
        # Search for minimum near r_e
        r_fine = np.linspace(r.min(), r.max(), 1000)
        E_fine = cs_elec(r_fine)
        r_e_elec = r_fine[np.argmin(E_fine)]
        E_e_min = E_fine.min()

        # Shift so E_shifted(r_e_elec) = 0
        E_shifted = E_elec - E_e_min

        # b=0 decomposition
        a_values = compute_b0_curvature(r, E_shifted, r_e_elec)

        # Also do b=0 on the ORIGINAL curve (without subtracting V_nn)
        # Shift original: E_bind_shifted = E_bind - E_bind_min
        E_bind_min = E_bind.min()
        r_e_orig = r[np.argmin(E_bind)]
        E_bind_shifted = E_bind - E_bind_min
        a_orig = compute_b0_curvature(r, E_bind_shifted, r_e_orig)

        results.append({
            'name': name, 'Z1': Z1, 'Z2': Z2,
            'r': r, 'E_bind': E_bind, 'V_nn': V_nn,
            'E_elec': E_elec, 'E_shifted': E_shifted,
            'r_e_elec': r_e_elec, 'E_e_min': E_e_min,
            'a_elec': a_values,
            'r_e_orig': r_e_orig, 'E_bind_min': E_bind_min,
            'a_orig': a_orig,
        })

    # ================================================================
    # FIGURE 1: Show a few example curves before/after V_nn subtraction
    # ================================================================
    show_mols = ['HF', 'LiH', 'NaCl', 'CO', 'Cl2', 'KH']
    show_idx = [i for i, n in enumerate(names) if n in show_mols]
    if len(show_idx) < len(show_mols):
        show_idx = list(range(min(6, n_mol)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    cmap = plt.cm.tab10

    for ii, idx in enumerate(show_idx):
        res = results[idx]
        ax = axes[ii // 3, ii % 3]

        ax.plot(res['r'], res['E_bind'], 'k-', lw=2,
                label='$E_{bind}(r)$ (total)')
        ax.plot(res['r'], res['V_nn'], 'r:', lw=1.5,
                label=f'$V_{{nn}} = {res["Z1"]}\\times{res["Z2"]}\\times14.4/r$')
        ax.plot(res['r'], res['E_elec'], 'b--', lw=1.5,
                label='$E_{elec}(r)$ (after subtraction)')
        ax.axhline(0, color='gray', lw=0.5)

        ax.set_xlabel('r [Å]')
        ax.set_ylabel('Energy [eV]')
        ax.set_title(f'{res["name"]} (Z={res["Z1"]}×{res["Z2"]})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'DFT binding curves: before and after nuclear repulsion subtraction',
        fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_dft_nuclear_subtraction.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_dft_nuclear_subtraction.png')
    plt.close(fig)

    # ================================================================
    # FIGURE 2: b=0 curvature a(r) — with and without V_nn subtraction
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ii, idx in enumerate(show_idx):
        res = results[idx]
        color = cmap(ii)

        # a(r) of electronic curve (V_nn subtracted)
        mask = np.isfinite(res['a_elec']) & (res['a_elec'] > -100) & (res['a_elec'] < 500)
        axes[0].plot(res['r'][mask], res['a_elec'][mask], color=color,
                     lw=1.5, label=res['name'])

        # sqrt(a) of electronic curve
        a_pos = np.maximum(res['a_elec'], 0)
        axes[1].plot(res['r'][mask], np.sqrt(a_pos[mask]), color=color,
                     lw=1.5, label=res['name'])

        # a(r) of original curve (no V_nn subtraction)
        mask2 = np.isfinite(res['a_orig']) & (res['a_orig'] > -100) & (res['a_orig'] < 500)
        axes[2].plot(res['r'][mask2], res['a_orig'][mask2], color=color,
                     lw=1.5, label=res['name'])

    axes[0].set_ylabel(r'$a_{elec}(r)$ [eV/Å²]')
    axes[0].set_title(r'(A)  $a(r)$ after $V_{nn}$ subtraction')

    axes[1].set_ylabel(r'$\sqrt{a_{elec}(r)}$')
    axes[1].set_title(r'(B)  $\sqrt{a(r)}$ after subtraction')

    axes[2].set_ylabel(r'$a_{orig}(r)$ [eV/Å²]')
    axes[2].set_title(r'(C)  $a(r)$ WITHOUT subtraction')

    for ax in axes:
        ax.set_xlabel('r [Å]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        r'b=0 curvature $a(r)$ for DFT diatomics — effect of $V_{nn}$ subtraction',
        fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_dft_b0_curvature.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_dft_b0_curvature.png')
    plt.close(fig)

    # ================================================================
    # FIGURE 3: Reconstruction check — does a(r)*(r-r_e)^2 + shift work?
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ii, idx in enumerate(show_idx):
        res = results[idx]
        ax = axes[ii // 3, ii % 3]

        # Reconstruct
        V_recon = (res['a_elec'] * (res['r'] - res['r_e_elec'])**2
                   + res['E_e_min'] + res['V_nn'])

        ax.plot(res['r'], res['E_bind'], 'k-', lw=2, label='Original $E_{bind}$')
        ax.plot(res['r'], V_recon, 'r--', lw=1.5,
                label=r'$a(r)(r-r_e)^2 + E_{min} + V_{nn}$')
        ax.axhline(0, color='gray', lw=0.5)

        resid = np.abs(res['E_bind'] - V_recon)
        ax.set_xlabel('r [Å]')
        ax.set_ylabel('Energy [eV]')
        ax.set_title(f'{res["name"]} — max|error| = {np.nanmax(resid):.2e} eV')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Reconstruction check: does the decomposition reproduce the curves?',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_dft_b0_reconstruction.png'),
                dpi=150, bbox_inches='tight')
    print('Saved fig_dft_b0_reconstruction.png')
    plt.close(fig)

    # --- Print summary ---
    print('\n--- Summary ---')
    print(f'{"Mol":>6s}  {"Z1×Z2":>6s}  {"V_nn(r_e)":>10s}  {"E_bind_min":>10s}  '
          f'{"E_elec_min":>10s}  {"r_e_orig":>8s}  {"r_e_elec":>8s}')
    for res in results:
        vnn_re = res['Z1'] * res['Z2'] * HART_TO_EV_ANGSTROM / res['r_e_orig']
        print(f'{res["name"]:>6s}  {res["Z1"]*res["Z2"]:6d}  {vnn_re:10.2f}  '
              f'{res["E_bind_min"]:10.4f}  {res["E_e_min"]:10.2f}  '
              f'{res["r_e_orig"]:8.3f}  {res["r_e_elec"]:8.3f}')


if __name__ == '__main__':
    main()
