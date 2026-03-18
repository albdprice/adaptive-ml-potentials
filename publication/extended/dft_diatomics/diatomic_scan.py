"""
DFT Binding Curve Scanner for Diatomic Molecules
==================================================

Computes binding energy curves E(r) for a set of diatomic molecules using PySCF.
Each molecule gets ~20 r-points around the equilibrium, producing clean 1D
binding curves that can be fit with Rose/UBER.

Requirements:
    pip install pyscf

Usage:
    python diatomic_scan.py                      # Compute all diatomics
    python diatomic_scan.py --molecules H2 LiH   # Specific molecules
    python diatomic_scan.py --level pbe           # Use PBE functional
    python diatomic_scan.py --basis aug-cc-pvtz   # Different basis set

Output:
    data/diatomic_curves.npz  -- all binding curves and metadata
"""

import argparse
import json
import os
import time
import numpy as np

try:
    from pyscf import gto, scf, dft
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    print("WARNING: PySCF not installed. Install with: pip install pyscf")
    print("This script requires PySCF for DFT calculations.")


# =============================================================================
# DIATOMIC MOLECULE DEFINITIONS
# =============================================================================

# Each entry: (name, atom1, atom2, r_eq_approx_angstrom, spin, charge)
# r_eq is approximate -- used to define scan range
DIATOMICS = [
    # Simple hydrides
    ('H2',   'H', 'H',  0.74, 0, 0),
    ('LiH',  'Li', 'H', 1.60, 0, 0),
    ('BeH',  'Be', 'H', 1.34, 1, 0),  # doublet
    ('BH',   'B', 'H',  1.23, 0, 0),
    ('CH',   'C', 'H',  1.12, 1, 0),  # doublet
    ('NH',   'N', 'H',  1.04, 2, 0),  # triplet
    ('OH',   'O', 'H',  0.97, 1, 0),  # doublet
    ('HF',   'H', 'F',  0.92, 0, 0),
    ('NaH',  'Na', 'H', 1.89, 0, 0),
    ('MgH',  'Mg', 'H', 1.73, 1, 0),  # doublet
    ('AlH',  'Al', 'H', 1.65, 0, 0),
    ('SiH',  'Si', 'H', 1.52, 1, 0),  # doublet
    ('PH',   'P', 'H',  1.42, 2, 0),  # triplet
    ('SH',   'S', 'H',  1.34, 1, 0),  # doublet
    ('HCl',  'H', 'Cl', 1.27, 0, 0),
    ('KH',   'K', 'H',  2.24, 0, 0),

    # Homonuclear diatomics
    ('Li2',  'Li', 'Li', 2.67, 0, 0),
    ('N2',   'N', 'N',  1.10, 0, 0),
    ('O2',   'O', 'O',  1.21, 2, 0),  # triplet
    ('F2',   'F', 'F',  1.41, 0, 0),
    ('Na2',  'Na', 'Na', 3.08, 0, 0),
    ('Cl2',  'Cl', 'Cl', 1.99, 0, 0),

    # Heteronuclear
    ('LiF',  'Li', 'F', 1.56, 0, 0),
    ('NaF',  'Na', 'F', 1.93, 0, 0),
    ('NaCl', 'Na', 'Cl', 2.36, 0, 0),
    ('CO',   'C', 'O',  1.13, 0, 0),
    ('NO',   'N', 'O',  1.15, 1, 0),  # doublet
    ('CN',   'C', 'N',  1.17, 1, 0),  # doublet
    ('BF',   'B', 'F',  1.27, 0, 0),
    ('AlF',  'Al', 'F', 1.65, 0, 0),
]

# Atomic descriptors: (Z, electronegativity_Pauling, covalent_radius_A, ionization_eV)
ATOM_PROPERTIES = {
    'H':  (1,  2.20, 0.31, 13.60),
    'He': (2,  0.00, 0.28, 24.59),
    'Li': (3,  0.98, 1.28,  5.39),
    'Be': (4,  1.57, 0.96,  9.32),
    'B':  (5,  2.04, 0.84,  8.30),
    'C':  (6,  2.55, 0.76, 11.26),
    'N':  (7,  3.04, 0.71, 14.53),
    'O':  (8,  3.44, 0.66, 13.62),
    'F':  (9,  3.98, 0.57, 17.42),
    'Na': (11, 0.93, 1.66,  5.14),
    'Mg': (12, 1.31, 1.41,  7.65),
    'Al': (13, 1.61, 1.21,  5.99),
    'Si': (14, 1.90, 1.11,  8.15),
    'P':  (15, 2.19, 1.07, 10.49),
    'S':  (16, 2.58, 1.05, 10.36),
    'Cl': (17, 3.16, 1.02, 12.97),
    'K':  (19, 0.82, 2.03,  4.34),
}


# =============================================================================
# DFT CALCULATION
# =============================================================================

def compute_energy(atom1, atom2, r, basis='def2-svp', xc='pbe', spin=0, charge=0,
                   verbose=0, dm0=None):
    """Compute DFT energy for a diatomic at distance r (Angstrom).

    Returns (energy_hartree, dm) where dm is the density matrix for use as
    initial guess at the next geometry.

    Always uses UKS (even for singlets) so the wavefunction can break spin
    symmetry at stretched geometries where RKS fails. Three-level fallback:
    1. UKS with damping
    2. UKS without damping
    3. UKS with Fermi smearing (helps near-degeneracy at dissociation)
    """
    mol = gto.M(
        atom=f'{atom1} 0 0 0; {atom2} 0 0 {r}',
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=verbose,
        unit='Angstrom',
    )

    # Always use UKS -- RKS breaks down at stretched geometries for singlets
    # (e.g., F2, Cl2, HCl, N2 all fail with RKS at dissociation)
    if xc.lower() == 'hf':
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc

    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    mf.damp = 0.5

    # Attempt 1: UKS with damping
    energy = mf.kernel(dm0=dm0)
    dm = mf.make_rdm1()

    if not mf.converged:
        # Attempt 2: retry without damping
        mf.damp = 0
        energy = mf.kernel(dm0=dm)
        dm = mf.make_rdm1()

    if not mf.converged:
        # Attempt 3: Fermi smearing helps with near-degeneracy at dissociation
        if xc.lower() != 'hf':
            mf2 = dft.UKS(mol)
            mf2.xc = xc
        else:
            mf2 = scf.UHF(mol)
        mf2.conv_tol = 1e-9
        mf2.max_cycle = 500
        mf2 = scf.addons.smearing_(mf2, sigma=0.01, method='fermi')
        energy = mf2.kernel(dm0=dm)
        dm = mf2.make_rdm1()
        if not mf2.converged:
            return None, dm

    return energy, dm  # energy in Hartree


def compute_atom_energy(atom, basis='def2-svp', xc='pbe', verbose=0):
    """Compute energy of an isolated atom."""
    # Determine spin for isolated atom (number of unpaired electrons)
    # Simple rules based on Hund's rule for ground-state atoms
    atom_spins = {
        'H': 1, 'He': 0, 'Li': 1, 'Be': 0, 'B': 1, 'C': 2, 'N': 3,
        'O': 2, 'F': 1, 'Na': 1, 'Mg': 0, 'Al': 1, 'Si': 2, 'P': 3,
        'S': 2, 'Cl': 1, 'K': 1,
    }
    spin = atom_spins.get(atom, 0)

    mol = gto.M(
        atom=f'{atom} 0 0 0',
        basis=basis,
        charge=0,
        spin=spin,
        verbose=verbose,
        unit='Angstrom',
    )

    if xc.lower() == 'hf':
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc

    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    energy = mf.kernel()

    return energy if mf.converged else None


def scan_diatomic(name, atom1, atom2, r_eq, spin=0, charge=0,
                  basis='def2-svp', xc='pbe', n_points=20, verbose=0):
    """Compute binding energy curve for a diatomic molecule.

    Returns r_grid (Angstrom), E_binding (eV), E_total (Hartree).
    Binding energy = E(r) - E(atom1) - E(atom2), converted to eV.

    Scans from equilibrium outward in both directions, propagating the
    density matrix for better SCF convergence at stretched geometries.
    """
    HA_TO_EV = 27.2114

    # Scan range: 0.7*r_eq to 5.0*r_eq (extended to capture dissociation tail)
    r_min = max(0.5, 0.7 * r_eq)
    r_max = 5.0 * r_eq
    r_grid = np.linspace(r_min, r_max, n_points)

    print(f"  Scanning {name} ({atom1}-{atom2}): r = {r_min:.2f} to {r_max:.2f} A, "
          f"spin={spin}, {n_points} points...")

    # Compute atom energies
    E_atom1 = compute_atom_energy(atom1, basis=basis, xc=xc, verbose=verbose)
    E_atom2 = compute_atom_energy(atom2, basis=basis, xc=xc, verbose=verbose)

    if E_atom1 is None or E_atom2 is None:
        print(f"    WARNING: Atom energy failed for {name}")
        return None

    E_atoms = E_atom1 + E_atom2

    # Scan from equilibrium outward to propagate density matrix
    # First: equilibrium outward (increasing r)
    idx_eq = np.argmin(np.abs(r_grid - r_eq))
    r_forward = r_grid[idx_eq:]   # r_eq to r_max
    r_backward = r_grid[:idx_eq][::-1]  # r_eq-1 down to r_min

    results = {}  # r -> (E_total, E_binding)

    # Forward scan (equilibrium -> dissociation)
    dm = None
    for r in r_forward:
        E, dm = compute_energy(atom1, atom2, r, basis=basis, xc=xc,
                               spin=spin, charge=charge, verbose=verbose, dm0=dm)
        if E is not None:
            E_bind = (E - E_atoms) * HA_TO_EV
            results[r] = (E, E_bind)
        else:
            print(f"    WARNING: SCF not converged at r={r:.3f} A")

    # Backward scan (equilibrium -> repulsive wall)
    dm = None
    for r in r_backward:
        E, dm = compute_energy(atom1, atom2, r, basis=basis, xc=xc,
                               spin=spin, charge=charge, verbose=verbose, dm0=dm)
        if E is not None:
            E_bind = (E - E_atoms) * HA_TO_EV
            results[r] = (E, E_bind)
        else:
            print(f"    WARNING: SCF not converged at r={r:.3f} A")

    if len(results) < n_points * 0.5:
        print(f"    WARNING: Too many unconverged points for {name}")
        return None

    # Sort by r
    sorted_r = sorted(results.keys())
    converged_r = np.array(sorted_r)
    E_total = np.array([results[r][0] for r in sorted_r])
    E_binding = np.array([results[r][1] for r in sorted_r])

    return {
        'name': name,
        'atom1': atom1, 'atom2': atom2,
        'r_grid': converged_r,
        'E_binding': E_binding,
        'E_total': E_total,
        'E_atoms': E_atoms,
        'r_eq_approx': r_eq,
        'spin': spin,
        'charge': charge,
    }


# =============================================================================
# DESCRIPTORS
# =============================================================================

def get_pair_descriptor(atom1, atom2):
    """Build a descriptor vector for an atom pair.

    Uses atomic properties: Z, electronegativity, covalent radius, ionization energy.
    For a pair, we use both individual values and combinations (sum, diff, product).
    """
    p1 = ATOM_PROPERTIES[atom1]
    p2 = ATOM_PROPERTIES[atom2]

    # Individual properties: (Z, EN, r_cov, IE) for each atom
    # Pair properties: sum, abs_diff, product for EN, r_cov, IE
    descriptor = [
        p1[0], p2[0],                          # Z1, Z2
        p1[1], p2[1],                          # EN1, EN2
        p1[2], p2[2],                          # r_cov1, r_cov2
        p1[3], p2[3],                          # IE1, IE2
        p1[1] + p2[1],                          # EN_sum
        abs(p1[1] - p2[1]),                     # EN_diff
        p1[2] + p2[2],                          # r_cov_sum
        abs(p1[2] - p2[2]),                     # r_cov_diff
        p1[3] + p2[3],                          # IE_sum
        abs(p1[3] - p2[3]),                     # IE_diff
    ]

    return np.array(descriptor)


DESCRIPTOR_NAMES = [
    'Z1', 'Z2', 'EN1', 'EN2', 'r_cov1', 'r_cov2', 'IE1', 'IE2',
    'EN_sum', 'EN_diff', 'r_cov_sum', 'r_cov_diff', 'IE_sum', 'IE_diff',
]


# =============================================================================
# ROSE/UBER FITTING
# =============================================================================

def rose_binding(r, D_e, r_e, l):
    """Rose/UBER equation for binding energy curves.

    E(r) = -D_e * (1 + a*) * exp(-a*)  where a* = (r - r_e) / l

    Convention: E(r_e) = -D_e (bound minimum), E(inf) = 0 (dissociation).
    D_e > 0 is the well depth.
    """
    a_star = (r - r_e) / l
    return -D_e * (1 + a_star) * np.exp(-a_star)


def fit_rose(r, E_binding):
    """Fit Rose/UBER parameters to a DFT binding curve.

    E_binding should be negative at the minimum and approach 0 at dissociation.
    Returns (D_e, r_e, l) or None if fitting fails.
    D_e is the well depth (positive for bound systems).
    """
    # Initial guess from the data
    idx_min = np.argmin(E_binding)
    E_min = E_binding[idx_min]
    r_min = r[idx_min]

    if E_min >= 0:
        # No bound state found
        return None, None

    D_e_guess = -E_min  # positive
    r_e_guess = r_min
    # Estimate l from the curvature: half-width of the well
    # Find where E_binding crosses E_min/2
    half_depth = E_min / 2
    above_half = np.where(E_binding > half_depth)[0]
    if len(above_half) > 0 and above_half[0] > idx_min:
        l_guess = r[above_half[0]] - r_min
    else:
        l_guess = 0.5

    try:
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            rose_binding, r, E_binding,
            p0=[D_e_guess, r_e_guess, max(l_guess, 0.1)],
            bounds=([0.01, r.min() * 0.8, 0.05], [50, r.max(), 5]),
            maxfev=10000,
        )

        # Compute fit quality
        E_fit = rose_binding(r, *popt)
        residual = np.sqrt(np.mean((E_binding - E_fit)**2))

        return popt, residual

    except Exception as e:
        print(f"    Rose fit failed: {e}")
        return None, None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute diatomic binding curves')
    parser.add_argument('--molecules', nargs='+', default=None,
                        help='Specific molecules to compute (default: all)')
    parser.add_argument('--basis', default='def2-svp',
                        help='Basis set (default: def2-svp, covers H-Rn)')
    parser.add_argument('--level', default='pbe',
                        help='DFT functional or "hf" (default: pbe)')
    parser.add_argument('--n-points', type=int, default=25,
                        help='Number of r-points per molecule (default: 25)')
    parser.add_argument('--output', default='data/diatomic_curves.npz',
                        help='Output file (default: data/diatomic_curves.npz)')
    parser.add_argument('--verbose', type=int, default=0,
                        help='PySCF verbosity (default: 0)')
    args = parser.parse_args()

    if not HAS_PYSCF:
        print("ERROR: PySCF is required. Install with: pip install pyscf")
        return

    print("=" * 70)
    print("DIATOMIC BINDING CURVE SCANNER")
    print("=" * 70)
    print(f"Level: {args.level}/{args.basis}")
    print(f"Points per curve: {args.n_points}")

    # Select molecules
    if args.molecules:
        selected = [d for d in DIATOMICS if d[0] in args.molecules]
        if not selected:
            print(f"No matching molecules found. Available: {[d[0] for d in DIATOMICS]}")
            return
    else:
        selected = DIATOMICS

    print(f"Molecules: {len(selected)}")
    print()

    # Compute binding curves
    all_results = []
    all_descriptors = []
    all_rose_params = []
    failed = []

    t0 = time.time()

    for name, a1, a2, r_eq, spin, charge in selected:
        result = scan_diatomic(name, a1, a2, r_eq, spin=spin, charge=charge,
                               basis=args.basis, xc=args.level,
                               n_points=args.n_points, verbose=args.verbose)

        if result is None:
            failed.append(name)
            continue

        # Fit Rose/UBER
        rose_params, residual = fit_rose(result['r_grid'], result['E_binding'])
        if rose_params is None:
            print(f"    WARNING: Rose fit failed for {name}")
            failed.append(name)
            continue

        result['rose_params'] = rose_params
        result['rose_residual'] = residual

        # Compute descriptor
        descriptor = get_pair_descriptor(a1, a2)
        result['descriptor'] = descriptor

        all_results.append(result)
        all_descriptors.append(descriptor)
        all_rose_params.append(rose_params)

        print(f"    {name}: E_c={rose_params[0]:.3f} eV, r_e={rose_params[1]:.3f} A, "
              f"l={rose_params[2]:.3f} A, residual={residual:.4f} eV")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Successful: {len(all_results)}/{len(selected)}")
    if failed:
        print(f"Failed: {failed}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Pack into arrays where possible
    save_dict = {
        'names': np.array([r['name'] for r in all_results]),
        'descriptors': np.array(all_descriptors),
        'descriptor_names': np.array(DESCRIPTOR_NAMES),
        'rose_params': np.array(all_rose_params),
        'rose_param_names': np.array(['E_c', 'r_e', 'l']),
        'basis': np.array(args.basis),
        'xc': np.array(args.level),
        'n_molecules': np.array(len(all_results)),
    }

    # Save individual curves (variable length due to unconverged points)
    for i, result in enumerate(all_results):
        save_dict[f'r_{i}'] = result['r_grid']
        save_dict[f'E_bind_{i}'] = result['E_binding']
        save_dict[f'E_total_{i}'] = result['E_total']

    np.savez(args.output, **save_dict)
    print(f"Results saved to: {args.output}")

    # Also save a human-readable summary
    summary_path = args.output.replace('.npz', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Diatomic Binding Curves Summary\n")
        f.write(f"Level: {args.level}/{args.basis}\n")
        f.write(f"{'Molecule':>8} {'E_c [eV]':>10} {'r_e [A]':>10} {'l [A]':>10} "
                f"{'Residual':>10}\n")
        f.write("-" * 55 + "\n")
        for r in all_results:
            p = r['rose_params']
            f.write(f"{r['name']:>8} {p[0]:>10.4f} {p[1]:>10.4f} {p[2]:>10.4f} "
                    f"{r['rose_residual']:>10.4f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
