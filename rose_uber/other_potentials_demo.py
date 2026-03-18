"""
Other Potentials Demo: Coulomb, Dispersion, and Lennard-Jones
=============================================================

This script demonstrates:
1. Coulomb (1/r) and Dispersion (1/r^6) are HOMOGENEOUS - constant k_eff
2. Lennard-Jones is NON-HOMOGENEOUS - varying k_eff, benefits from adaptive params
3. ML comparison showing adaptive approach wins for LJ

Connection to main story:
- Homogeneous functions don't need adaptive parameters (k_eff = constant)
- Non-homogeneous functions (Morse, LJ) benefit from learning adaptive params
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11


# =============================================================================
# POTENTIAL FUNCTIONS
# =============================================================================

def coulomb(r, q1q2):
    """Coulomb potential: V(r) = q1*q2/r (homogeneous degree -1)"""
    return q1q2 / r

def dispersion(r, C6):
    """Dispersion potential: V(r) = -C6/r^6 (homogeneous degree -6)"""
    return -C6 / r**6

def lennard_jones(r, epsilon, sigma):
    """LJ potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6] (NOT homogeneous)"""
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def lj_reduced(r_star):
    """Reduced LJ: V*(r*) = 4[(1/r*)^12 - (1/r*)^6] where r* = r/σ, V* = V/ε"""
    return 4 * (r_star**(-12) - r_star**(-6))


# =============================================================================
# EFFECTIVE HOMOGENEITY DEGREE
# =============================================================================

def k_eff(r, V, dVdr):
    """Calculate effective homogeneity degree: k_eff = r*V'(r)/V(r)"""
    return r * dVdr / V


# =============================================================================
# EXPERIMENT 1: Homogeneous vs Non-Homogeneous
# =============================================================================

def experiment1_homogeneity_comparison():
    """
    Show that Coulomb and Dispersion have constant k_eff,
    while Lennard-Jones has varying k_eff.
    """
    print("EXPERIMENT 1: Homogeneous vs Non-Homogeneous Potentials")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    r = np.linspace(0.8, 3.0, 200)

    # --- Coulomb ---
    q1q2 = 1.0  # arbitrary units
    V_coul = coulomb(r, q1q2)
    dV_coul = -q1q2 / r**2  # analytical derivative
    k_coul = k_eff(r, V_coul, dV_coul)

    axes[0, 0].plot(r, V_coul, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('V(r)')
    axes[0, 0].set_title('Coulomb: V(r) = q/r')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 2)

    axes[1, 0].plot(r, k_coul, 'b-', linewidth=2)
    axes[1, 0].axhline(-1, color='red', linestyle='--', label='k = -1 (theory)')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('k_eff(r)')
    axes[1, 0].set_title('k_eff = -1 (CONSTANT)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-1.5, 0)

    # --- Dispersion ---
    C6 = 1.0
    V_disp = dispersion(r, C6)
    dV_disp = 6 * C6 / r**7  # analytical derivative
    k_disp = k_eff(r, V_disp, dV_disp)

    axes[0, 1].plot(r, V_disp, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('V(r)')
    axes[0, 1].set_title('Dispersion: V(r) = -C₆/r⁶')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(r, k_disp, 'g-', linewidth=2)
    axes[1, 1].axhline(-6, color='red', linestyle='--', label='k = -6 (theory)')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('k_eff(r)')
    axes[1, 1].set_title('k_eff = -6 (CONSTANT)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-8, -4)

    # --- Lennard-Jones ---
    epsilon, sigma = 1.0, 1.0
    r_lj = np.linspace(0.95, 3.0, 200)  # avoid singularity
    V_lj = lennard_jones(r_lj, epsilon, sigma)
    dV_lj = 4 * epsilon * (-12 * sigma**12 / r_lj**13 + 6 * sigma**6 / r_lj**7)

    # k_eff only defined where V != 0
    mask = np.abs(V_lj) > 0.01
    k_lj = np.full_like(V_lj, np.nan)
    k_lj[mask] = k_eff(r_lj[mask], V_lj[mask], dV_lj[mask])

    axes[0, 2].plot(r_lj, V_lj, 'r-', linewidth=2)
    axes[0, 2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0, 2].set_xlabel('r')
    axes[0, 2].set_ylabel('V(r)')
    axes[0, 2].set_title('Lennard-Jones: 4ε[(σ/r)¹² - (σ/r)⁶]')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-1.5, 2)

    axes[1, 2].plot(r_lj, k_lj, 'r-', linewidth=2)
    axes[1, 2].axhline(-12, color='blue', linestyle=':', alpha=0.7, label='k = -12 (repulsive)')
    axes[1, 2].axhline(-6, color='green', linestyle=':', alpha=0.7, label='k = -6 (attractive)')
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('k_eff(r)')
    axes[1, 2].set_title('k_eff VARIES (non-homogeneous!)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(-15, 5)

    plt.tight_layout()
    plt.savefig('fig6_homogeneity_comparison.png', dpi=150)
    plt.close(fig)

    print("Coulomb:    k_eff = -1 (constant) → HOMOGENEOUS")
    print("Dispersion: k_eff = -6 (constant) → HOMOGENEOUS")
    print("Lennard-Jones: k_eff varies from -12 to +∞ → NON-HOMOGENEOUS")
    print()
    print("IMPLICATION: Non-homogeneous potentials benefit from adaptive parameters!")


# =============================================================================
# EXPERIMENT 2: Universal Scaling for LJ
# =============================================================================

def experiment2_lj_universal():
    """
    Show that LJ potentials collapse to universal curve when scaled.
    """
    print("\nEXPERIMENT 2: LJ Universal Scaling")
    print("=" * 60)

    np.random.seed(42)
    n_potentials = 50

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate random LJ parameters
    epsilons = np.random.uniform(0.01, 0.5, n_potentials)  # eV
    sigmas = np.random.uniform(2.0, 4.0, n_potentials)     # Angstrom

    colors = plt.cm.viridis(np.linspace(0, 1, n_potentials))

    # Panel 1: Raw LJ curves
    ax = axes[0]
    for i in range(n_potentials):
        r = np.linspace(0.9 * sigmas[i], 3.0 * sigmas[i], 100)
        V = lennard_jones(r, epsilons[i], sigmas[i])
        ax.plot(r, V, color=colors[i], alpha=0.5, linewidth=1)
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title(f'Raw LJ curves\n({n_potentials} different potentials)')
    ax.set_ylim(-0.6, 1.0)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Panel 2: Scaled by epsilon only
    ax = axes[1]
    for i in range(n_potentials):
        r = np.linspace(0.9 * sigmas[i], 3.0 * sigmas[i], 100)
        V = lennard_jones(r, epsilons[i], sigmas[i])
        ax.plot(r, V / epsilons[i], color=colors[i], alpha=0.5, linewidth=1)
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) / ε')
    ax.set_title('Scaled by ε only\n(still scattered)')
    ax.set_ylim(-1.5, 2.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Panel 3: Fully scaled - COLLAPSE!
    ax = axes[2]
    for i in range(n_potentials):
        r = np.linspace(0.9 * sigmas[i], 3.0 * sigmas[i], 100)
        V = lennard_jones(r, epsilons[i], sigmas[i])
        r_star = r / sigmas[i]
        V_star = V / epsilons[i]
        ax.plot(r_star, V_star, color=colors[i], alpha=0.5, linewidth=1)

    # Universal curve
    r_theory = np.linspace(0.9, 3.0, 200)
    V_theory = lj_reduced(r_theory)
    ax.plot(r_theory, V_theory, 'k--', linewidth=3, label='Universal: 4[(1/r*)¹² - (1/r*)⁶]')

    ax.set_xlabel('r* = r/σ')
    ax.set_ylabel('V* = V/ε')
    ax.set_title('UNIVERSAL COLLAPSE!\n(all follow ONE curve)')
    ax.set_ylim(-1.5, 2.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(2**(1/6), color='red', linestyle=':', alpha=0.5, label=f'r*_min = 2^(1/6)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig7_lj_universal.png', dpi=150)
    plt.close(fig)

    print("All LJ potentials collapse to ONE universal curve when scaled by (ε, σ)")
    print("This is the SAME principle as Rose/UBER for Morse!")


# =============================================================================
# EXPERIMENT 3: ML Comparison for LJ
# =============================================================================

def experiment3_lj_ml():
    """
    ML comparison for Lennard-Jones:
    - Direct: descriptor → V(r)
    - Adaptive: descriptor → (ε, σ) → V(r) via LJ equation

    Setup matches Morse/Rose experiment:
    - Linear descriptor-to-parameter relationships
    - Extrapolation to larger σ values
    """
    print("\nEXPERIMENT 3: ML Comparison for Lennard-Jones")
    print("=" * 60)

    np.random.seed(42)
    n_r_points = 50

    # Generate TRAINING data: small atoms (low σ)
    train_data = {'X': [], 'params': [], 'curves': [], 'r_grids': []}

    for i in range(80):
        # Descriptors with linear relationships to parameters
        d1 = np.random.uniform(0.5, 1.5)  # correlates with σ
        d2 = np.random.uniform(0.5, 1.5)  # correlates with ε

        # Linear relationships (like Morse experiment)
        sigma = 2.5 + 0.5 * d1   # Training: σ ~ 2.75-3.25 Å
        epsilon = 0.05 + 0.1 * d2  # ε ~ 0.1-0.2 eV

        r = np.linspace(0.9 * sigma, 2.5 * sigma, n_r_points)
        V = lennard_jones(r, epsilon, sigma)

        train_data['X'].append([d1, d2])
        train_data['params'].append([epsilon, sigma])
        train_data['curves'].append(V)
        train_data['r_grids'].append(r)

    # Generate TEST data: LARGE atoms (high σ) - EXTRAPOLATION!
    test_data = {'X': [], 'params': [], 'curves': [], 'r_grids': []}

    for i in range(40):
        # Extrapolate to larger d1 (larger atoms)
        d1 = np.random.uniform(2.5, 4.0)  # EXTRAPOLATION in d1
        d2 = np.random.uniform(0.5, 1.5)  # same range for d2

        sigma = 2.5 + 0.5 * d1   # Test: σ ~ 3.75-4.5 Å (larger!)
        epsilon = 0.05 + 0.1 * d2

        r = np.linspace(0.9 * sigma, 2.5 * sigma, n_r_points)
        V = lennard_jones(r, epsilon, sigma)

        test_data['X'].append([d1, d2])
        test_data['params'].append([epsilon, sigma])
        test_data['curves'].append(V)
        test_data['r_grids'].append(r)

    X_train = np.array(train_data['X'])
    Y_params_train = np.array(train_data['params'])
    Y_curves_train = np.array(train_data['curves'])

    X_test = np.array(test_data['X'])
    Y_params_test = np.array(test_data['params'])
    Y_curves_test = np.array(test_data['curves'])

    print(f"Training: {len(X_train)} small atoms (σ ~ 2.4-3.1 Å)")
    print(f"Testing:  {len(X_test)} LARGE atoms (σ ~ 3.4-4.5 Å) - EXTRAPOLATION!")

    # Scale data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # --- Direct approach ---
    scaler_Y = StandardScaler()
    Y_curves_train_scaled = scaler_Y.fit_transform(Y_curves_train)

    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_scaled, Y_curves_train_scaled)

    Y_pred_scaled = model_direct.predict(X_test_scaled)
    Y_pred_direct = scaler_Y.inverse_transform(Y_pred_scaled)

    mse_direct = mean_squared_error(Y_curves_test, Y_pred_direct)
    r2_direct = r2_score(Y_curves_test.flatten(), Y_pred_direct.flatten())

    print(f"\nDirect approach: MSE = {mse_direct:.4f}, R² = {r2_direct:.4f}")

    # --- Adaptive approach ---
    scaler_params = StandardScaler()
    Y_params_train_scaled = scaler_params.fit_transform(Y_params_train)

    model_params = Ridge(alpha=1.0)
    model_params.fit(X_train_scaled, Y_params_train_scaled)

    Y_params_pred_scaled = model_params.predict(X_test_scaled)
    Y_params_pred = scaler_params.inverse_transform(Y_params_pred_scaled)

    # Reconstruct curves via physics
    Y_pred_adaptive = []
    for i, (eps, sig) in enumerate(Y_params_pred):
        r = test_data['r_grids'][i]
        V_pred = lennard_jones(r, max(eps, 0.001), max(sig, 0.1))
        Y_pred_adaptive.append(V_pred)
    Y_pred_adaptive = np.array(Y_pred_adaptive)

    mse_adaptive = mean_squared_error(Y_curves_test, Y_pred_adaptive)
    r2_adaptive = r2_score(Y_curves_test.flatten(), Y_pred_adaptive.flatten())
    r2_params = r2_score(Y_params_test, Y_params_pred)

    print(f"Adaptive approach: MSE = {mse_adaptive:.4f}, R² = {r2_adaptive:.4f}")
    print(f"Parameter R²: {r2_params:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Example predictions
    for col, idx in enumerate([0, len(X_test)//2, -1]):
        ax = axes[0, col]
        r = test_data['r_grids'][idx]
        eps, sig = test_data['params'][idx]

        ax.plot(r, Y_curves_test[idx], 'k-', linewidth=3, label='True')
        ax.plot(r, Y_pred_direct[idx], 'b--', linewidth=2, label='Direct ML')
        ax.plot(r, Y_pred_adaptive[idx], 'r:', linewidth=2, label='Adaptive ML')

        ax.set_xlabel('r [Å]')
        ax.set_ylabel('V(r) [eV]')
        ax.set_title(f'Large atom σ={sig:.2f} Å\n(extrapolation from training)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # Bottom left: MSE comparison
    ax = axes[1, 0]
    methods = ['Direct\n(learn V(r))', 'Adaptive\n(learn ε,σ)']
    mses = [mse_direct, mse_adaptive]
    colors_bar = ['blue', 'red']
    ax.bar(methods, mses, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MSE [eV²]')
    ax.set_title('Extrapolation Error')

    if mse_adaptive < mse_direct:
        improvement = (mse_direct - mse_adaptive) / mse_direct * 100
        ax.text(0.5, 0.7, f'Adaptive is\n{improvement:.0f}% better!',
                ha='center', fontsize=14, fontweight='bold', color='green',
                transform=ax.transAxes)

    # Bottom middle: R² comparison
    ax = axes[1, 1]
    r2s = [r2_direct, r2_adaptive]
    ax.bar(methods, r2s, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R² score')
    ax.set_title('Extrapolation R²')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    # Bottom right: σ extrapolation
    ax = axes[1, 2]
    ax.scatter(Y_params_test[:, 1], Y_params_pred[:, 1], alpha=0.7, s=60, c='tab:orange')
    lims = [2, max(Y_params_test[:, 1].max(), Y_params_pred[:, 1].max()) * 1.1]
    ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect')

    train_sigma_max = Y_params_train[:, 1].max()
    ax.axvline(train_sigma_max, color='red', linestyle=':', linewidth=2,
               label=f'Training max σ={train_sigma_max:.1f}')
    ax.axhline(train_sigma_max, color='red', linestyle=':', linewidth=2)

    ax.set_xlabel('True σ [Å]')
    ax.set_ylabel('Predicted σ [Å]')
    ax.set_title('σ Extrapolation\n(predicting beyond training)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('fig8_lj_ml_comparison.png', dpi=150)
    plt.close(fig)

    print("\n" + "-" * 60)
    if mse_adaptive < mse_direct:
        print(f"RESULT: Adaptive is {(mse_direct-mse_adaptive)/mse_direct*100:.0f}% better at extrapolation!")
    print("Same principle works for LJ as for Morse/Rose!")
    print("-" * 60)

    return {
        'mse_direct': mse_direct,
        'mse_adaptive': mse_adaptive,
        'r2_direct': r2_direct,
        'r2_adaptive': r2_adaptive
    }


# =============================================================================
# EXPERIMENT 4: Summary - When to Use Adaptive Parameters
# =============================================================================

def experiment4_summary():
    """
    Summary figure showing when adaptive parameters help.
    """
    print("\nEXPERIMENT 4: Summary")
    print("=" * 60)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title
    ax.text(0.5, 0.95, 'When Do Adaptive Parameters Help?',
            fontsize=18, fontweight='bold', ha='center', va='top')

    # Left column: Homogeneous
    ax.text(0.25, 0.82, 'HOMOGENEOUS\nPOTENTIALS', fontsize=14, fontweight='bold',
            ha='center', va='top', color='green')

    ax.text(0.25, 0.70, 'Coulomb: V = q/r\nDispersion: V = -C₆/r⁶\nPauli: V = A/r¹²',
            fontsize=11, ha='center', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.text(0.25, 0.50, 'k_eff = constant\n(degree is fixed)',
            fontsize=12, ha='center', va='top')

    ax.text(0.25, 0.38, 'No adaptive parameter needed!\nJust learn the coefficient\n(q, C₆, A, etc.)',
            fontsize=11, ha='center', va='top', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Right column: Non-homogeneous
    ax.text(0.75, 0.82, 'NON-HOMOGENEOUS\nPOTENTIALS', fontsize=14, fontweight='bold',
            ha='center', va='top', color='red')

    ax.text(0.75, 0.70, 'Morse: V = Dₑ(1-e^(-a(r-rₑ)))²\nLennard-Jones: V = 4ε[(σ/r)¹²-(σ/r)⁶]\nReal DFT binding curves',
            fontsize=11, ha='center', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.text(0.75, 0.50, 'k_eff varies with r\n(no single degree)',
            fontsize=12, ha='center', va='top')

    ax.text(0.75, 0.38, 'ADAPTIVE PARAMETERS HELP!\nLearn (ε,σ) or (Eₒ,rₑ,l)\nPhysics provides shape',
            fontsize=11, ha='center', va='top', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Dividing line
    ax.axvline(0.5, ymin=0.25, ymax=0.85, color='gray', linestyle='-', linewidth=2)

    # Bottom: Key message
    ax.text(0.5, 0.15, 'KEY INSIGHT: Adaptive parameters are most valuable for\n'
            'non-homogeneous potentials where the "effective degree" varies.\n'
            'This is exactly the situation in real materials!',
            fontsize=12, ha='center', va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig9_summary.png', dpi=150)
    plt.close(fig)

    print("Homogeneous potentials: constant k_eff → no adaptive params needed")
    print("Non-homogeneous potentials: varying k_eff → ADAPTIVE PARAMS HELP!")
    print()
    print("Real materials have non-homogeneous binding → this approach is valuable!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("OTHER POTENTIALS DEMO: Coulomb, Dispersion, Lennard-Jones")
    print("=" * 70)
    print()

    experiment1_homogeneity_comparison()
    experiment2_lj_universal()
    lj_results = experiment3_lj_ml()
    experiment4_summary()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
FINDINGS:

1. HOMOGENEOUS POTENTIALS (Coulomb, Dispersion):
   - k_eff is constant (equals the degree)
   - No adaptive parameter needed
   - Just learn the coefficient directly

2. NON-HOMOGENEOUS POTENTIALS (Morse, Lennard-Jones):
   - k_eff varies with r
   - Universal scaling exists when properly parameterized
   - Adaptive parameters improve ML extrapolation

3. ML COMPARISON FOR LJ:
   - Train on small atoms, test on LARGE atoms
   - Direct: learns V(r) directly
   - Adaptive: learns (ε, σ) → V(r) via physics
   - Adaptive extrapolates better!

4. CONNECTION TO aPBE0:
   - Same principle: learn bounded, smooth parameters
   - Physics equation provides correct shape
   - Works for molecules (Morse) AND materials (LJ, DFT)
""")


if __name__ == "__main__":
    main()
